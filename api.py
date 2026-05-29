"""
api.py — Camada FastAPI conectando frontend ao motor TRI Vision.

Wraps (sem modificar):
  - estimador_nota.py
  - gerar_vetor_estrategico.py

Otimizações vs versão original:
  • Pré-carrega Parquets em memória no startup
  • Substitui _iter_candidatos por iterador in-memory via monkey-patch
  • Cache LRU de estimativas por (area, ano, tipo, cor, vetor)
  • Logs de timing por request

Rodar:
  uvicorn api:app --reload --port 8000
"""

import os, sys, time, csv, json
from typing import Optional
from functools import lru_cache
from collections import OrderedDict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Imports diretos dos módulos existentes (não modificados)
import estimador_nota as en
from estimador_nota import estimar_nota_tri
import gerar_vetor_estrategico as gve

# Camada unificada de motores (supervisionado por default; fallback automático)
import engine_tri
engine_tri.configurar(
    dir_modelos="modelos_supervisionados",
    path_modelo_compacto="modelo_compacto_tri.json",
)
print(f"[ENGINE] inicializado: {engine_tri.status()}")


# ═══════════════════════════════════════════════════════════════════════
#  PRE-LOAD: Parquets em memória
# ═══════════════════════════════════════════════════════════════════════

try:
    import pyarrow.parquet as papq
    PARQUET_OK = True
except ImportError:
    PARQUET_OK = False

PASTA_FEATURES = "features"

# Cache de candidatos por (area, ano, tipo) → lista pré-decodificada
# Cada candidato é dict com {vetor, mascara, nota, acertos, coerencia, media_b_a, media_b_e, cor}
_BASE_HISTORICA = {}      # {(area, ano, tipo): [candidato_dict, ...]}
_BASE_POR_COR   = {}      # {(area, ano, tipo, cor.upper()): [candidato_dict, ...]}

# Cache de respostas
_CACHE_ESTIMATIVAS = OrderedDict()
_CACHE_MAX = 4000


def _carregar_parquet(path):
    """Lê parquet e devolve lista de dicts (decodificado uma só vez)."""
    if not PARQUET_OK or not os.path.exists(path):
        return None
    tabela = papq.read_table(
        path,
        columns=["cor","vetor_binario_acertos","mascara_aplicavel",
                 "nota_real","acertos","coerencia_inversoes",
                 "media_b_acertos","media_b_erros"]
    )
    cor_arr  = tabela.column("cor").to_pylist()
    vet_arr  = tabela.column("vetor_binario_acertos").to_pylist()
    msk_arr  = (tabela.column("mascara_aplicavel").to_pylist()
                if "mascara_aplicavel" in tabela.column_names else [None]*len(cor_arr))
    nota_arr = tabela.column("nota_real").to_pylist()
    ac_arr   = tabela.column("acertos").to_pylist()
    coer_arr = tabela.column("coerencia_inversoes").to_pylist()
    mba_arr  = tabela.column("media_b_acertos").to_pylist()
    mbe_arr  = tabela.column("media_b_erros").to_pylist()
    lista = []
    for i in range(len(cor_arr)):
        lista.append({
            "vetor":     vet_arr[i],
            "mascara":   msk_arr[i],
            "nota":      nota_arr[i],
            "acertos":   ac_arr[i],
            "coerencia": coer_arr[i],
            "media_b_a": mba_arr[i],
            "media_b_e": mbe_arr[i],
            "cor":       str(cor_arr[i]).upper(),
        })
    return lista


def _carregar_base_lazy(area, ano, tipo):
    """
    Carrega Parquet sob demanda. Cacheia em _BASE_HISTORICA e _BASE_POR_COR.
    Retorna a lista de candidatos (vazia se não encontrar).
    """
    chave = (area, ano, tipo)
    if chave in _BASE_HISTORICA:
        print(f"[CACHE HIT] {area}_{ano}_{tipo} already loaded "
              f"({len(_BASE_HISTORICA[chave]):,} reg)")
        return _BASE_HISTORICA[chave]

    print(f"[CACHE MISS] loading {area}_{ano}_{tipo} ...")
    path = os.path.join(PASTA_FEATURES, f"features_{area}_{ano}_{tipo}.parquet")
    if not os.path.exists(path):
        print(f"[CACHE MISS] arquivo não encontrado: {path}")
        _BASE_HISTORICA[chave] = []  # cache negativo
        return []

    t0 = time.time()
    lista = _carregar_parquet(path)
    if not lista:
        _BASE_HISTORICA[chave] = []
        return []

    _BASE_HISTORICA[chave] = lista

    # Indexa por cor
    por_cor = {}
    for c in lista:
        por_cor.setdefault(c["cor"], []).append(c)
    for cor, sub in por_cor.items():
        _BASE_POR_COR[(area, ano, tipo, cor)] = sub

    # Registra path → chave para o monkey-patch reconhecer
    _PATH_PARA_CHAVE[path] = chave

    dt = (time.time()-t0)*1000
    print(f"[CACHE MISS] {area}_{ano}_{tipo}  {len(lista):,} reg  {dt:.0f}ms")
    return lista


# ═══════════════════════════════════════════════════════════════════════
#  Monkey-patch do _iter_candidatos do estimador
#  → usa cache em memória; mantém mesmo contrato.
# ═══════════════════════════════════════════════════════════════════════

# Para alcançar a chave (area, ano, tipo) a partir do path, mantemos um índice reverso
_PATH_PARA_CHAVE = {}


def _iter_candidatos_cached(path, fmt, cor=None):
    chave_base = _PATH_PARA_CHAVE.get(path)
    if chave_base:
        area, ano, tipo = chave_base
        if cor:
            lista = _BASE_POR_COR.get((area, ano, tipo, cor.upper()))
            if lista is not None:
                for c in lista:
                    yield c
                return
        lista = _BASE_HISTORICA.get((area, ano, tipo))
        if lista is not None:
            if cor:
                cor_u = cor.upper()
                for c in lista:
                    if c["cor"] == cor_u: yield c
            else:
                for c in lista: yield c
            return
    # Fallback: chama implementação original (raro)
    yield from _iter_candidatos_original(path, fmt, cor)


def _arquivo_features_cached(area, ano, tipo):
    """Wrap original — só registra o path no índice reverso."""
    path, fmt = _arquivo_features_original(area, ano, tipo)
    if path:
        _PATH_PARA_CHAVE[path] = (area, ano, tipo)
    return path, fmt


# Guarda referências originais e substitui
_iter_candidatos_original   = en._iter_candidatos
_arquivo_features_original  = en._arquivo_features
en._iter_candidatos  = _iter_candidatos_cached
en._arquivo_features = _arquivo_features_cached


# ═══════════════════════════════════════════════════════════════════════
#  Cache de b_por_posicao + questoes (parser de ITENS_PROVA)
# ═══════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=512)
def _carregar_itens_cache(area, ano, tipo, cor, lingua):
    """Cache permanente — ITENS_PROVA não muda em runtime."""
    itens, mascara, erro = gve.carregar_itens_prova(area, ano, tipo, cor, lingua)
    if erro:
        return None, None, None, erro
    b_por_posicao = tuple(b for b, _, _ in itens)
    questoes = tuple({
        "pos":       i + 1,
        "b":         round(b, 4) if b is not None else None,
        "tp_ling":   tp_ling or "comum",
        "co_item":   ci,
        "aplicavel": mascara[i] == "1",
    } for i, (b, tp_ling, ci) in enumerate(itens))
    return b_por_posicao, mascara, questoes, None


# ═══════════════════════════════════════════════════════════════════════
#  FastAPI
# ═══════════════════════════════════════════════════════════════════════

app = FastAPI(title="TRI Vision API", version="2.0")

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
    # Não pré-carrega Parquets — carregamento é lazy (sob demanda na 1ª /estimar)
    if not os.path.exists(PASTA_FEATURES):
        print(f"[API] AVISO: pasta '{PASTA_FEATURES}' não encontrada.")
    else:
        n_arq = len([f for f in os.listdir(PASTA_FEATURES)
                     if f.startswith("features_") and f.endswith(".parquet")])
        print(f"[API] {n_arq} Parquets disponíveis para carregamento lazy.")
    print(f"[API] estimator ready ({(time.time()-t0)*1000:.0f}ms)")


# ─── Modelos ───────────────────────────────────────────────────────────

class EstimarRequest(BaseModel):
    area: str
    ano: int
    tipo: str
    cor: str
    vetor: str
    mascara: Optional[str] = None


class QuestoesRequest(BaseModel):
    area: str
    ano: int
    tipo: str
    cor: str
    lingua: str = "ing"


class WarmupRequest(BaseModel):
    area: str
    ano: int
    tipo: str
    cor: str
    lingua: str = "ing"


class VetorEstrategicoRequest(BaseModel):
    area: str
    ano: int
    tipo: str
    cor: str
    modo: str
    n: Optional[int] = None
    b_min: Optional[float] = None
    b_max: Optional[float] = None
    lingua: str = "ing"
    seed: Optional[int] = None


# ─── Endpoints ─────────────────────────────────────────────────────────

@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "bases_carregadas": len(_BASE_HISTORICA),
        "cache_estimativas": len(_CACHE_ESTIMATIVAS),
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
        "tamanho":  len(questoes),
        "mascara":  mascara,
        "questoes": list(questoes),
    }


def _cache_put(key, valor):
    _CACHE_ESTIMATIVAS[key] = valor
    if len(_CACHE_ESTIMATIVAS) > _CACHE_MAX:
        _CACHE_ESTIMATIVAS.popitem(last=False)


@app.post("/warmup")
def post_warmup(req: WarmupRequest):
    """
    Pré-carrega base + itens da prova e roda uma estimativa com vetor zerado.
    Retorna a nota mínima histórica (acc=0) para o frontend exibir como ponto inicial.
    """
    t0 = time.time()

    # 1. Lazy load da base (Parquet + índice por cor + path → chave)
    lista = _carregar_base_lazy(req.area, req.ano, req.tipo)
    base_size = len(lista)

    # 2. Carrega cache de itens (LRU já trata repetição)
    b_por_pos, mascara_auto, _, erro = _carregar_itens_cache(
        req.area, req.ano, req.tipo, req.cor, req.lingua
    )
    if erro:
        raise HTTPException(status_code=404, detail=erro)

    # 3. Estimativa com vetor todo-zeros → cai na rota minimo_historico
    vetor_zero = "0" * len(b_por_pos)
    mascara = mascara_auto if req.area == "LC" else None
    cache_key = (req.area, req.ano, req.tipo, req.cor.upper(), vetor_zero, mascara or "")

    nota_zero = _CACHE_ESTIMATIVAS.get(cache_key)
    if nota_zero is None:
        try:
            nota_zero = engine_tri.estimar_nota(
                vetor=vetor_zero,
                b_por_posicao=list(b_por_pos),
                area=req.area, ano=req.ano, tipo=req.tipo, cor=req.cor,
                mascara=mascara,
                engine="supervisionado",
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        _cache_put(cache_key, nota_zero)

    load_time_ms = (time.time() - t0) * 1000
    print(f"[WARMUP] {req.area} {req.ano} {req.tipo} {req.cor} "
          f"-> {load_time_ms:.0f}ms (base={base_size:,})")

    return {
        "status":       "ready",
        "area":         req.area,
        "ano":          req.ano,
        "tipo":         req.tipo,
        "cor":          req.cor,
        "load_time_ms": round(load_time_ms, 1),
        "base_size":    base_size,
        "nota_zero":    nota_zero,
    }


@app.post("/estimar")
def post_estimar(req: EstimarRequest):
    t0 = time.time()

    # Cache hit
    cache_key = (req.area, req.ano, req.tipo, req.cor.upper(), req.vetor, req.mascara or "")
    cached = _CACHE_ESTIMATIVAS.get(cache_key)
    if cached is not None:
        # Move para o final (LRU)
        _CACHE_ESTIMATIVAS.move_to_end(cache_key)
        dt = (time.time()-t0)*1000
        print(f"[ESTIMAR] {req.area} {req.ano} {req.tipo} {len(req.vetor)}q "
              f"-> {dt:.0f}ms [HIT]")
        return cached

    # Carrega b das posições (do cache LRU)
    b_por_pos, mascara_auto, _, erro = _carregar_itens_cache(
        req.area, req.ano, req.tipo, req.cor, "ing"
    )
    if erro:
        raise HTTPException(status_code=404, detail=erro)

    if len(req.vetor) != len(b_por_pos):
        # Caso comum em LC: frontend envia vetor com apenas as posições
        # aplicáveis (40 comuns + 5 da língua escolhida = 45), mas a estrutura
        # interna tem 50 (5 ing + 5 esp + 40 comuns). Expande inserindo '9'
        # (não aplicável) nas posições da outra língua.
        if (req.area == "LC"
                and mascara_auto
                and len(req.vetor) == mascara_auto.count("1")
                and len(mascara_auto) == len(b_por_pos)):
            # Reconstrói vetor de 50 chars usando a máscara: '9' onde a posição
            # não é aplicável, e os chars do vetor recebido nas demais.
            expandido = []
            it = iter(req.vetor)
            for m in mascara_auto:
                expandido.append(next(it) if m == "1" else "9")
            req.vetor = "".join(expandido)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Vetor tem {len(req.vetor)} chars; prova requer {len(b_por_pos)}"
            )

    mascara = req.mascara or (mascara_auto if req.area == "LC" else None)

    # Lazy load — carrega base se ainda não estiver em memória
    _carregar_base_lazy(req.area, req.ano, req.tipo)

    try:
        resultado = engine_tri.estimar_nota(
            vetor=req.vetor,
            b_por_posicao=list(b_por_pos),
            area=req.area, ano=req.ano, tipo=req.tipo, cor=req.cor,
            mascara=mascara,
            engine="supervisionado",   # fallback automático para compacto/historico
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    _cache_put(cache_key, resultado)
    dt = (time.time()-t0)*1000
    print(f"[ESTIMAR] {req.area} {req.ano} {req.tipo} {len(req.vetor)}q -> {dt:.0f}ms")
    return resultado


@app.post("/vetor-estrategico")
def post_vetor_estrategico(req: VetorEstrategicoRequest):
    if req.modo == "intervalo-b" and (req.b_min is None or req.b_max is None):
        raise HTTPException(status_code=400, detail="b_min e b_max obrigatórios em intervalo-b")
    if req.modo != "intervalo-b" and req.n is None:
        raise HTTPException(status_code=400, detail=f"n obrigatório no modo {req.modo}")

    itens, mascara, erro = gve.carregar_itens_prova(req.area, req.ano, req.tipo, req.cor, req.lingua)
    if erro:
        raise HTTPException(status_code=404, detail=erro)

    if   req.modo == "faceis":      marcados = gve.gerar_faceis(itens, mascara, req.n)
    elif req.modo == "dificeis":    marcados = gve.gerar_dificeis(itens, mascara, req.n)
    elif req.modo == "aleatorio":   marcados = gve.gerar_aleatorio(itens, mascara, req.n, req.seed)
    elif req.modo == "intervalo-b": marcados = gve.gerar_intervalo_b(itens, mascara, req.b_min, req.b_max)
    elif req.modo == "coerente":    marcados = gve.gerar_coerente(itens, mascara, req.n)
    elif req.modo == "incoerente":  marcados = gve.gerar_incoerente(itens, mascara, req.n, req.seed)
    else:
        raise HTTPException(status_code=400, detail=f"modo desconhecido: {req.modo}")

    vetor = gve.montar_vetor(itens, marcados)
    info  = gve.resumo(itens, mascara, marcados)
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
