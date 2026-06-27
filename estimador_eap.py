"""
estimador_eap.py
================
Motor TRI primario: EAP do modelo 3PL (o mesmo do INEP), embarcado.

Le UM artefato (parametros_eap.json, gerado pela fase2_calibrar_eap.py):
parametros a/b/c por prova + calibracao theta->nota isotonica + faixa de
incerteza real por nivel de acertos.

NAO depende de microdados, ITENS_PROVA nem API. Tudo offline, em numpy.

Interface compativel com estimador_lgb.estimar_nota_supervisionado:
recebe o vetor na ordem da cor e devolve o mesmo dict (nota_estimada,
intervalo_min/max, etc.), para a API e o frontend nao mudarem.
"""

import json, os
from functools import lru_cache

import numpy as np

_PARAMS_PATH = "parametros_eap.json"
_MAPEAMENTO_PATH = "mapeamento_canonico_v6.json"

_PARAMS = None
_MAPEAMENTO = None
_THETA_GRID = None
_PRIOR = (0.0, 1.0)
_LOGP_CACHE = {}   # chave -> (logP, log1mP, idx, D)


def configurar_dir(params_path="parametros_eap.json",
                   mapeamento_path="mapeamento_canonico_v6.json"):
    global _PARAMS, _MAPEAMENTO, _PARAMS_PATH, _MAPEAMENTO_PATH
    global _THETA_GRID, _PRIOR, _LOGP_CACHE
    _PARAMS_PATH = params_path
    _MAPEAMENTO_PATH = mapeamento_path
    _PARAMS = None; _MAPEAMENTO = None; _LOGP_CACHE = {}


def _carregar():
    global _PARAMS, _MAPEAMENTO, _THETA_GRID, _PRIOR
    if _PARAMS is None:
        if not os.path.exists(_PARAMS_PATH):
            raise FileNotFoundError(f"{_PARAMS_PATH} ausente")
        with open(_PARAMS_PATH, encoding="utf-8") as f:
            _PARAMS = json.load(f)
        meta = _PARAMS.get("_meta", {})
        g = meta.get("theta_grid", [-4.5, 4.5, 81])
        _THETA_GRID = np.linspace(g[0], g[1], int(g[2]))
        _PRIOR = tuple(meta.get("prior", [0.0, 1.0]))
    if _MAPEAMENTO is None:
        with open(_MAPEAMENTO_PATH, encoding="utf-8") as f:
            _MAPEAMENTO = json.load(f)
    return _PARAMS, _MAPEAMENTO


def _resolver_chave(area, ano, tipo, lingua=None):
    t = tipo.lower()
    t = "ppl" if t == "ppl" else "regular"
    if area == "LC":
        ling = (lingua or "ing").lower()
        if ling not in ("ing", "esp"): ling = "ing"
        return f"{area}_{ano}_{t}_{ling}"
    return f"{area}_{ano}_{t}"


def _resolver_info_cor(mapa, cor):
    cores = mapa.get("cores", {})
    if not cores: return None
    cu = (cor or "").upper()
    if cu in cores: return cores[cu]
    for k, v in cores.items():
        if k.upper() == cu: return v
    return next(iter(cores.values()))


def _logp_da_chave(chave, par):
    """Pre-computa (e cacheia) logP/log1mP do EAP para a chave."""
    cached = _LOGP_CACHE.get(chave)
    if cached is not None:
        return cached
    itens = par["itens"]
    D = par.get("D", 1.0)
    a, b, c, idx = [], [], [], []
    for i, it in enumerate(itens):
        if it.get("in_aban") == "1": continue
        if it.get("a") is None or it.get("b") is None or it.get("c") is None:
            continue
        a.append(it["a"]); b.append(it["b"]); c.append(it["c"]); idx.append(i)
    a = np.asarray(a); b = np.asarray(b); c = np.asarray(c)
    idx = np.asarray(idx, dtype=int)
    Z = D * a[None, :] * (_THETA_GRID[:, None] - b[None, :])
    P = c[None, :] + (1 - c[None, :]) / (1 + np.exp(-Z))
    P = np.clip(P, 1e-6, 1 - 1e-6)
    out = (np.log(P), np.log(1 - P), idx)
    _LOGP_CACHE[chave] = out
    return out


def _canonicalizar(vetor_cor, info_cor, mapa):
    """
    vetor_cor: string '0'/'1'/'9' na ordem da cor.
    Devolve vetor canonico binario + n_itens canonicos.
    Itens anulados = acerto. Itens '9' (outra lingua) = nao contam (0 e
    fora da verossimilhanca porque nao estao no idx do EAP).
    """
    perm = info_cor.get("perm_resp") or info_cor.get("perm")
    if perm is None: return None
    itens = mapa["itens_canonicos"]
    n = len(perm)
    out = ["0"] * n
    for ic, p in enumerate(perm):
        if itens[ic].get("in_aban") == "1":
            out[ic] = "1"; continue
        if p is None or p >= len(vetor_cor):
            out[ic] = "0"; continue
        ch = vetor_cor[p]
        out[ic] = "1" if ch == "1" else "0"
    return "".join(out)


def _theta_eap(vetor_canon, logP, log1mP, idx):
    x = np.array([1.0 if vetor_canon[i] == "1" else 0.0 for i in idx])
    ll = x @ logP.T + (1 - x) @ log1mP.T
    mu, sg = _PRIOR
    ll = ll + (-0.5 * ((_THETA_GRID - mu) / sg) ** 2)
    ll -= ll.max()
    w = np.exp(ll)
    return float((w @ _THETA_GRID) / w.sum())


def _theta_para_nota(theta, par):
    """Isotonica se disponivel; senao reta linear."""
    xs = par.get("iso_theta"); ys = par.get("iso_nota")
    if xs and ys:
        nota = float(np.interp(theta, xs, ys))
    else:
        lin = par["linear"]
        nota = lin["alpha"] + lin["beta"] * theta
    return max(0.0, min(1000.0, nota))


def modelo_disponivel(area, ano, tipo, cor=None, lingua=None):
    par, _ = _carregar()
    return _resolver_chave(area, ano, tipo, lingua) in par


def estimar_nota_supervisionado(vetor, b_por_posicao, area, ano, tipo, cor,
                                mascara=None, lingua=None):
    """Mesma assinatura do estimador_lgb. b_por_posicao/mascara ignorados."""
    par_all, mape = _carregar()
    chave = _resolver_chave(area, ano, tipo, lingua)
    par = par_all.get(chave)
    if par is None:
        return {"erro": f"sem parametros EAP para {chave}", "chave": chave}
    mapa = mape.get(chave)
    if mapa is None:
        return {"erro": f"mapeamento sem {chave}", "chave": chave}
    info_cor = _resolver_info_cor(mapa, cor)
    if info_cor is None:
        return {"erro": f"sem cor para {chave}", "chave": chave}

    vetor_canon = _canonicalizar(vetor, info_cor, mapa)
    if vetor_canon is None:
        return {"erro": "falha canonicalizando", "chave": chave}

    logP, log1mP, idx = _logp_da_chave(chave, par)
    theta = _theta_eap(vetor_canon, logP, log1mP, idx)
    nota = _theta_para_nota(theta, par)

    # acertos canonicos validos (mesma definicao do auditor)
    n_validos = len(idx)
    acertos = sum(1 for i in idx if vetor_canon[i] == "1")

    # coerencia 0..1: fracao de pares (acerto, erro) com b_acerto < b_erro
    # (mesma metrica que o frontend usava do LightGBM, recalculada do 3PL)
    b_por_idx = {it_i: par["itens"][it_i]["b"]
                 for it_i in idx if par["itens"][it_i].get("b") is not None}
    ac_b = [b_por_idx[i] for i in idx if vetor_canon[i] == "1" and i in b_por_idx]
    er_b = [b_por_idx[i] for i in idx if vetor_canon[i] == "0" and i in b_por_idx]
    if ac_b and er_b:
        ok = sum(1 for ba in ac_b for be in er_b if ba < be)
        coerencia = round(ok / (len(ac_b) * len(er_b)), 4)
    elif ac_b and not er_b:
        coerencia = 1.0
    else:
        coerencia = None

    # GANHO ESTRATEGICO (Fase 5): para os mesmos N acertos, quanto a
    # estrategia coerente rende vs a nota tipica (mediana real do nivel).
    # Usa a faixa real por n_acertos como referencia da "nota tipica".
    ganho = None
    faixa_nac = par.get("faixa_por_nacertos", {})
    # itens anulados contam como acerto na nota mas reporta-se a parte
    anulados = sum(1 for it in par["itens"] if it.get("in_aban") == "1")
    acertos_reportados = acertos + anulados

    # faixa de incerteza REAL por nivel de acertos
    faixa = par.get("faixa_por_nacertos", {})
    fb = faixa.get(str(acertos_reportados)) or faixa.get(str(acertos))
    if fb:
        std = fb["std"]
        intervalo_min = max(0.0, nota - 1.5 * std)
        intervalo_max = min(1000.0, nota + 1.5 * std)
        if std <= 8: conf = "alta"
        elif std <= 18: conf = "media"
        else: conf = "baixa"
        # ganho estrategico: nota EAP - nota tipica (mediana) do nivel.
        # a mediana do nivel ~ ponto medio entre p10 e p90 quando nao ha
        # mediana explicita; usa p10/p90 reais para estimar o tipico.
        tipica = (fb.get("p10", nota) + fb.get("p90", nota)) / 2.0
        if tipica > 0:
            ganho = round(nota - tipica, 1)
    else:
        std = None
        intervalo_min = max(0.0, nota - 12.0)
        intervalo_max = min(1000.0, nota + 12.0)
        conf = "media"

    return {
        "nota_estimada":    round(nota, 2),
        "theta":            round(theta, 4),
        "coerencia":        coerencia,
        "ganho_estrategico": ganho,
        "acertos":          acertos_reportados,
        "erros":            n_validos - acertos,
        "aplicaveis":       n_validos,
        "intervalo_min":    round(intervalo_min, 2),
        "intervalo_max":    round(intervalo_max, 2),
        "confianca":        conf,
        "std_faixa":        std,
        "metodo":           "eap_3pl",
        "metodo_ancoragem": "isotonica" if par.get("iso_nota") else "linear",
        "modelo_nome":      f"eap_{chave}",
        "chave":            chave,
        "r2_calibracao":    par.get("r2"),
        "lingua_usada":     lingua,
        "vetor_canonico":   vetor_canon,
    }


def status():
    try:
        par, _ = _carregar()
    except Exception as e:
        return {"motor": "eap", "erro": str(e), "n_chaves": 0}
    n = len([k for k in par if not k.startswith("_")])
    return {"motor": "eap_3pl", "n_chaves": n,
            "meta": par.get("_meta", {})}


if __name__ == "__main__":
    print(json.dumps(status(), indent=2, ensure_ascii=False))
