"""
api_v2.py
----------
API FastAPI v2 - usa engine_tri_v2 (LightGBM canonico) e
gerar_vetor_estrategico_v2 (mapeamento canonico v6).

Mantem rotas e formato de resposta do api.py antigo para compatibilidade
com o frontend atual. A cor passa a ser opcional (default AMARELA) e
e usada apenas para escolher o PDF de referencia. Modelos sao
cor-agnosticos internamente.

Rodar:
  uvicorn api_v2:app --reload --port 8000
"""

import os, sys, time, json
from typing import Optional
from functools import lru_cache
from collections import OrderedDict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import engine_tri_v2 as engine
import gerar_vetor_estrategico_v2 as gve

engine.configurar(dir_modelos="modelos_v2",
                   mapeamento_path="mapeamento_canonico_v6.json")
gve.configurar(mapeamento_path="mapeamento_canonico_v6.json")
print(f"[ENGINE_V2] inicializado: {engine.status()}")


# ─── Cache LRU de estimativas ────────────────────────────────────────
_CACHE_ESTIMATIVAS = OrderedDict()
_CACHE_MAX = 4000


def _cache_put(key, valor):
    _CACHE_ESTIMATIVAS[key] = valor
    if len(_CACHE_ESTIMATIVAS) > _CACHE_MAX:
        _CACHE_ESTIMATIVAS.popitem(last=False)


# ─── Cache de itens da prova (b_por_pos, mascara, questoes) ──────────
@lru_cache(maxsize=512)
def _carregar_itens_cache(area, ano, tipo, cor, lingua):
    itens, mascara, erro = gve.carregar_itens_prova(area, ano, tipo, cor, lingua)
    if erro:
        return None, None, None, erro
    b_por_pos = tuple(b for b, _, _ in itens)
    questoes = tuple({
        "pos":       i + 1,
        "b":         round(b, 4) if b is not None else None,
        "tp_ling":   tp_ling or "comum",
        "co_item":   ci,
        "aplicavel": mascara[i] == "1",
    } for i, (b, tp_ling, ci) in enumerate(itens))
    return b_por_pos, mascara, questoes, None


# ─── FastAPI ─────────────────────────────────────────────────────────
app = FastAPI(title="TRI Vision API v2", version="3.0")

_origens_env = os.environ.get("ALLOWED_ORIGIN", "").strip()
if _origens_env:
    _origens = [o.strip() for o in _origens_env.split(",") if o.strip()]
    for local in ("http://localhost:8000", "http://localhost:3000",
                  "http://127.0.0.1:8000", "http://127.0.0.1:3000"):
        if local not in _origens: _origens.append(local)
else:
    _origens = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origens,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    t0 = time.time()
    info = engine.status()
    print(f"[API_V2] engine: {info}")
    print(f"[API_V2] ready ({(time.time()-t0)*1000:.0f}ms)")


# ─── Modelos pydantic ────────────────────────────────────────────────
class EstimarRequest(BaseModel):
    area: str
    ano: int
    tipo: str
    cor: Optional[str] = "AMARELA"   # compatibilidade; agora opcional
    vetor: str
    mascara: Optional[str] = None
    lingua: Optional[str] = "ing"


class QuestoesRequest(BaseModel):
    area: str
    ano: int
    tipo: str
    cor: Optional[str] = "AMARELA"
    lingua: str = "ing"


class WarmupRequest(BaseModel):
    area: str
    ano: int
    tipo: str
    cor: Optional[str] = "AMARELA"
    lingua: str = "ing"


class VetorEstrategicoRequest(BaseModel):
    area: str
    ano: int
    tipo: str
    cor: Optional[str] = "AMARELA"
    modo: str
    n: Optional[int] = None
    b_min: Optional[float] = None
    b_max: Optional[float] = None
    lingua: str = "ing"
    seed: Optional[int] = None


# ─── Endpoints ───────────────────────────────────────────────────────
@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "engine": engine.status(),
        "cache_estimativas": len(_CACHE_ESTIMATIVAS),
        "cache_itens":       _carregar_itens_cache.cache_info()._asdict(),
    }


@app.post("/questoes")
def post_questoes(req: QuestoesRequest):
    b_por_pos, mascara, questoes, erro = _carregar_itens_cache(
        req.area, req.ano, req.tipo, req.cor, req.lingua
    )
    if erro:
        raise HTTPException(status_code=404, detail=erro)
    return {
        "area": req.area, "ano": req.ano, "tipo": req.tipo, "cor": req.cor,
        "lingua":   req.lingua,
        "tamanho":  len(questoes),
        "mascara":  mascara,
        "questoes": list(questoes),
    }


@app.post("/warmup")
def post_warmup(req: WarmupRequest):
    t0 = time.time()
    b_por_pos, mascara_auto, _, erro = _carregar_itens_cache(
        req.area, req.ano, req.tipo, req.cor, req.lingua
    )
    if erro:
        raise HTTPException(status_code=404, detail=erro)

    # Constroi vetor todo "0" (zerado) com 9 nos itens da outra lingua (mask=0)
    vetor_zero = "".join(
        "9" if mascara_auto[i] == "0" else "0"
        for i in range(len(mascara_auto))
    )
    mascara = mascara_auto if req.area == "LC" else None
    cache_key = (req.area, req.ano, req.tipo, (req.cor or "AMARELA").upper(),
                 vetor_zero, mascara or "", req.lingua)
    nota_zero = _CACHE_ESTIMATIVAS.get(cache_key)
    if nota_zero is None:
        try:
            nota_zero = engine.estimar_nota(
                vetor=vetor_zero,
                b_por_posicao=list(b_por_pos),
                area=req.area, ano=req.ano, tipo=req.tipo, cor=req.cor,
                mascara=mascara, lingua=req.lingua,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        if "erro" in nota_zero:
            raise HTTPException(status_code=500, detail=nota_zero["erro"])
        _cache_put(cache_key, nota_zero)

    load_ms = (time.time() - t0) * 1000
    print(f"[WARMUP_V2] {req.area} {req.ano} {req.tipo} {req.cor} "
          f"lang={req.lingua} -> {load_ms:.0f}ms")
    return {
        "status":       "ready",
        "area":         req.area,
        "ano":          req.ano,
        "tipo":         req.tipo,
        "cor":          req.cor,
        "lingua":       req.lingua,
        "load_time_ms": round(load_ms, 1),
        "nota_zero":    nota_zero,
    }


@app.post("/estimar")
def post_estimar(req: EstimarRequest):
    t0 = time.time()
    cor_norm = (req.cor or "AMARELA").upper()
    cache_key = (req.area, req.ano, req.tipo, cor_norm,
                 req.vetor, req.mascara or "", req.lingua or "ing")
    cached = _CACHE_ESTIMATIVAS.get(cache_key)
    if cached is not None:
        _CACHE_ESTIMATIVAS.move_to_end(cache_key)
        dt = (time.time() - t0) * 1000
        print(f"[ESTIMAR_V2] {req.area} {req.ano} {req.tipo} -> {dt:.0f}ms [HIT]")
        return cached

    b_por_pos, mascara_auto, _, erro = _carregar_itens_cache(
        req.area, req.ano, req.tipo, req.cor, req.lingua or "ing"
    )
    if erro:
        raise HTTPException(status_code=404, detail=erro)
    if len(req.vetor) != len(b_por_pos):
        raise HTTPException(
            status_code=400,
            detail=f"Vetor tem {len(req.vetor)} chars; prova requer {len(b_por_pos)}"
        )
    mascara = req.mascara or (mascara_auto if req.area == "LC" else None)

    try:
        resultado = engine.estimar_nota(
            vetor=req.vetor,
            b_por_posicao=list(b_por_pos),
            area=req.area, ano=req.ano, tipo=req.tipo, cor=req.cor,
            mascara=mascara, lingua=req.lingua or "ing",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if "erro" in resultado:
        raise HTTPException(status_code=500, detail=resultado["erro"])

    _cache_put(cache_key, resultado)
    dt = (time.time() - t0) * 1000
    print(f"[ESTIMAR_V2] {req.area} {req.ano} {req.tipo} {len(req.vetor)}q "
          f"-> {dt:.0f}ms nota={resultado.get('nota_estimada')}")
    return resultado


@app.post("/vetor-estrategico")
def post_vetor_estrategico(req: VetorEstrategicoRequest):
    if req.modo == "intervalo-b" and (req.b_min is None or req.b_max is None):
        raise HTTPException(status_code=400, detail="b_min e b_max obrigatorios")
    if req.modo != "intervalo-b" and req.n is None:
        raise HTTPException(status_code=400, detail=f"n obrigatorio em {req.modo}")

    itens, mascara, erro = gve.carregar_itens_prova(
        req.area, req.ano, req.tipo, req.cor, req.lingua)
    if erro:
        raise HTTPException(status_code=404, detail=erro)

    fn = {
        "faceis":      lambda: gve.gerar_faceis(itens, mascara, req.n),
        "dificeis":    lambda: gve.gerar_dificeis(itens, mascara, req.n),
        "aleatorio":   lambda: gve.gerar_aleatorio(itens, mascara, req.n, req.seed),
        "intervalo-b": lambda: gve.gerar_intervalo_b(itens, mascara, req.b_min, req.b_max),
        "coerente":    lambda: gve.gerar_coerente(itens, mascara, req.n),
        "incoerente":  lambda: gve.gerar_incoerente(itens, mascara, req.n, req.seed),
    }
    if req.modo not in fn:
        raise HTTPException(status_code=400, detail=f"modo desconhecido: {req.modo}")
    marcados = fn[req.modo]()

    vetor = gve.montar_vetor(itens, marcados)
    info = gve.resumo(itens, mascara, marcados)
    return {
        "area": req.area, "ano": req.ano, "tipo": req.tipo, "cor": req.cor,
        "modo": req.modo, "tamanho_vetor": len(vetor),
        "vetor":             vetor,
        "mascara":           mascara,
        "n_marcadas":        info["n_marcadas"],
        "posicoes_marcadas": info["posicoes"],
        "questoes_marcadas": info["questoes"],
        "media_b_acertos":   info["media_b_acertos"],
    }
