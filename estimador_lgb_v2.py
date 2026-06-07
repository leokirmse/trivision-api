"""
estimador_lgb_v2.py
--------------------
Wrapper para os 75 modelos LightGBM treinados em modelos_v2/.

Resolve uma chave a partir de (area, ano, tipo, cor, lingua):
  - MT/CN/CH: <area>_<ano>_<tipo>           (cor e ignorada)
  - LC:       <area>_<ano>_<tipo>_<lingua>  (cor ignorada, lingua importa)

Recebe vetor "no formato antigo" (mascara/cor) e CANONICALIZA antes de
calcular features e chamar o LightGBM.

Interface compatível com a antiga (`estimar_nota_supervisionado`):
  retorna dict com nota_estimada + métricas + metadados.
"""

import json, os, pickle, math
from functools import lru_cache
from collections import OrderedDict


_DIR_MODELOS = "modelos_v2"
_MAPEAMENTO_PATH = "mapeamento_canonico_v6.json"

# Cache de modelos pickled (carregamento lazy)
_MODELOS = {}   # chave -> dict (payload pickle)
_MAPEAMENTO = None

# Anos com TX_RESPOSTAS_LC de 50 chars (inclui 5 itens da outra lingua)
ANOS_LC_RESP_50 = {2016, 2017, 2018, 2019, 2020, 2021}


def configurar_dir(dir_modelos="modelos_v2",
                    mapeamento_path="mapeamento_canonico_v6.json"):
    global _DIR_MODELOS, _MAPEAMENTO_PATH, _MAPEAMENTO
    _DIR_MODELOS = dir_modelos
    _MAPEAMENTO_PATH = mapeamento_path
    _MAPEAMENTO = None  # forca reload


def _carregar_mapeamento():
    global _MAPEAMENTO
    if _MAPEAMENTO is None:
        if not os.path.exists(_MAPEAMENTO_PATH):
            raise FileNotFoundError(f"{_MAPEAMENTO_PATH} ausente")
        with open(_MAPEAMENTO_PATH, encoding="utf-8") as f:
            _MAPEAMENTO = json.load(f)
    return _MAPEAMENTO


def _resolver_chave(area, ano, tipo, cor=None, lingua=None):
    """Resolve a chave do mapeamento/modelo a partir dos parametros."""
    tipo_norm = tipo.lower()
    if "reaplic" in tipo_norm: tipo_norm = "regular"  # fallback
    elif tipo_norm == "ppl":   tipo_norm = "ppl"
    else:                       tipo_norm = "regular"

    if area == "LC":
        ling = (lingua or "ing").lower()
        if ling not in ("ing", "esp"): ling = "ing"
        return f"{area}_{ano}_{tipo_norm}_{ling}"
    return f"{area}_{ano}_{tipo_norm}"


def _resolver_chave_e_info_cor(area, ano, tipo, cor, lingua=None):
    """
    Retorna (chave, info_cor) onde info_cor tem perm_resp / perm_gab / etc.
    Se cor for None ou nao mapeada, usa a primeira cor disponivel.
    """
    chave = _resolver_chave(area, ano, tipo, cor, lingua)
    mapeamento = _carregar_mapeamento()
    if chave not in mapeamento:
        return chave, None
    mapa = mapeamento[chave]
    cores = mapa.get("cores", {})
    if not cores: return chave, None
    cor_upper = (cor or "").upper()
    if cor_upper in cores:
        return chave, cores[cor_upper]
    # Procura case-insensitive
    for c_k, c_v in cores.items():
        if c_k.upper() == cor_upper:
            return chave, c_v
    # Fallback: primeira cor disponivel
    primeira = next(iter(cores.values()))
    return chave, primeira


def _carregar_modelo(chave):
    if chave in _MODELOS:
        return _MODELOS[chave]
    path = os.path.join(_DIR_MODELOS, f"{chave}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        payload = pickle.load(f)
    _MODELOS[chave] = payload
    return payload


def modelo_disponivel(area, ano, tipo, cor=None, lingua=None):
    chave = _resolver_chave(area, ano, tipo, cor, lingua)
    path = os.path.join(_DIR_MODELOS, f"{chave}.pkl")
    return os.path.exists(path)


# ── Canonicalizacao do vetor de respostas ─────────────────────────────

def _canonicalizar_vetor(vetor_original, info_cor, mapa, area, ano):
    """
    Reordena o vetor recebido (na ordem da cor) para a ordem canonica
    do mapeamento. Retorna (vetor_canonico, mascara_canonica).

    O vetor_original eh assumido como uma string '0'/'1'/'9' onde:
      - '1' = acerto
      - '0' = erro
      - '9' = nao aplicavel (LC, item da outra lingua)

    Em MT/CN/CH, usa info_cor['perm'].
    Em LC,     usa info_cor['perm_resp'].
    """
    perm = info_cor.get("perm_resp") or info_cor.get("perm")
    if perm is None:
        return None, None

    itens_canon = mapa["itens_canonicos"]
    n_canon = len(perm)

    vetor_canon = ["0"] * n_canon
    mascara_canon = ["1"] * n_canon

    for i_canon, p in enumerate(perm):
        item_info = itens_canon[i_canon]
        # Item anulado: acerto automatico
        if item_info.get("in_aban") == "1":
            vetor_canon[i_canon] = "1"
            continue
        # Posicao nula = item da outra lingua
        if p is None:
            mascara_canon[i_canon] = "0"
            vetor_canon[i_canon] = "0"
            continue
        if p >= len(vetor_original):
            vetor_canon[i_canon] = "0"
            continue
        ch = vetor_original[p]
        if ch == "9":
            mascara_canon[i_canon] = "0"
            vetor_canon[i_canon] = "0"
        elif ch == "1":
            vetor_canon[i_canon] = "1"
        else:
            vetor_canon[i_canon] = "0"

    return "".join(vetor_canon), "".join(mascara_canon)


def _calcular_inversoes(acertos_b, erros_b):
    if not acertos_b or not erros_b:
        return 0, 0, 1.0
    inv = sum(1 for a in acertos_b for e in erros_b if a > e)
    pares = len(acertos_b) * len(erros_b)
    coer = 1.0 - (inv / pares) if pares > 0 else 1.0
    return inv, pares, round(coer, 4)


def _calcular_features(vetor_canon, mascara_canon, b_aplic_prova):
    """
    Replica calcular_features_canonico do extrair_features_v2.py +
    quartis (acertos_q1..q4) do treinar_modelos_v2.py.
    """
    import numpy as np

    acertos_b = []; erros_b = []
    acertos = 0; aplicaveis = 0

    for i, (v, m) in enumerate(zip(vetor_canon, mascara_canon)):
        if m != "1": continue
        aplicaveis += 1
        b = b_aplic_prova[i] if i < len(b_aplic_prova) else None
        if v == "1":
            acertos += 1
            if b is not None: acertos_b.append(b)
        else:
            if b is not None: erros_b.append(b)

    inv, pares, coer = _calcular_inversoes(acertos_b, erros_b)
    mb_ac = sum(acertos_b) / len(acertos_b) if acertos_b else 0.0
    mb_er = sum(erros_b)   / len(erros_b)   if erros_b   else 0.0

    ac_f = sum(1 for b in acertos_b if b < 1.0)
    ac_m = sum(1 for b in acertos_b if 1.0 <= b < 2.0)
    ac_d = sum(1 for b in acertos_b if b >= 2.0)
    er_f = sum(1 for b in erros_b   if b < 1.0)
    er_m = sum(1 for b in erros_b   if 1.0 <= b < 2.0)
    er_d = sum(1 for b in erros_b   if b >= 2.0)

    # Quartis: bs FIXOS da prova
    bs_validos = [b for b in b_aplic_prova if b is not None]
    if bs_validos:
        bs_arr = np.array(bs_validos)
        q25 = np.quantile(bs_arr, 0.25)
        q50 = np.quantile(bs_arr, 0.50)
        q75 = np.quantile(bs_arr, 0.75)
        q1 = q2 = q3 = q4 = 0
        for i, (v, m) in enumerate(zip(vetor_canon, mascara_canon)):
            if m != "1" or v != "1": continue
            if i >= len(b_aplic_prova): continue
            b = b_aplic_prova[i]
            if b is None: continue
            if b < q25: q1 += 1
            elif b < q50: q2 += 1
            elif b < q75: q3 += 1
            else: q4 += 1
    else:
        q1 = q2 = q3 = q4 = 0

    # Vetor de features na ordem exata do treino
    features = [
        float(acertos), float(mb_ac), float(mb_er),
        float(coer), float(inv),
        float(ac_f), float(ac_m), float(ac_d),
        float(er_f), float(er_m), float(er_d),
        float(q1), float(q2), float(q3), float(q4),
    ]

    # Hardest hit / easiest miss para metadata
    hardest_hit = max(acertos_b) if acertos_b else None
    easiest_miss = min(erros_b) if erros_b else None

    return {
        "features": features,
        "acertos": acertos,
        "erros":   aplicaveis - acertos,
        "aplicaveis": aplicaveis,
        "coerencia": coer,
        "inversoes": inv,
        "media_b_acertos": round(mb_ac, 4),
        "media_b_erros":   round(mb_er, 4),
        "hardest_hit": round(hardest_hit, 4) if hardest_hit is not None else None,
        "easiest_miss": round(easiest_miss, 4) if easiest_miss is not None else None,
    }


# ── Interface principal ────────────────────────────────────────────────

def estimar_nota_supervisionado(vetor, b_por_posicao, area, ano, tipo, cor,
                                  mascara=None, lingua=None):
    """
    API compativel com estimador_supervisionado antigo.

    A diferenca: agora canonicalizamos antes de calcular features.

    Parametros:
      vetor:          string '0'/'1'/'9' na ordem da cor (45 ou 50 chars)
      b_por_posicao:  IGNORADO pelos novos modelos (usamos b_aplic_prova do mapa)
      area, ano, tipo, cor: identificacao
      mascara:        IGNORADA (a mascara canonica eh derivada da perm)
      lingua:         "ing"/"esp" para LC (None para outros)
    """
    chave, info_cor = _resolver_chave_e_info_cor(area, ano, tipo, cor, lingua)
    if info_cor is None:
        return {"erro": f"mapeamento sem cor para {chave}", "chave": chave}

    payload = _carregar_modelo(chave)
    if payload is None:
        return {"erro": f"modelo nao encontrado: {chave}.pkl", "chave": chave}

    mapa = _carregar_mapeamento()[chave]
    b_aplic = payload["b_aplic_prova"]
    modelo = payload["modelo"]

    # Canonicaliza
    vetor_canon, mascara_canon = _canonicalizar_vetor(vetor, info_cor, mapa, area, ano)
    if vetor_canon is None:
        return {"erro": "falha canonicalizando vetor", "chave": chave}

    # Features
    feat = _calcular_features(vetor_canon, mascara_canon, b_aplic)

    # Predicao
    try:
        import numpy as np
        X = np.array([feat["features"]], dtype=float)
        y_pred = float(modelo.predict(X)[0])
    except Exception as ex:
        return {"erro": f"predict falhou: {ex}", "chave": chave}

    # Clip basico
    nota_estimada = max(0.0, min(1000.0, y_pred))

    # Metricas do modelo (do treino) para intervalo de confianca
    m_hold = payload.get("metricas_holdout", {})
    rmse_local = m_hold.get("rmse")
    mae_local = m_hold.get("mae")
    if rmse_local is not None:
        intervalo_min = max(0.0, nota_estimada - 1.5 * rmse_local)
        intervalo_max = min(1000.0, nota_estimada + 1.5 * rmse_local)
        if rmse_local <= 10: conf = "alta"
        elif rmse_local <= 20: conf = "media"
        else: conf = "baixa"
    else:
        intervalo_min = intervalo_max = nota_estimada
        conf = "indefinida"

    status = payload.get("status", "?")

    return {
        "nota_estimada":     round(nota_estimada, 2),
        "acertos":           feat["acertos"],
        "erros":             feat["erros"],
        "aplicaveis":        feat["aplicaveis"],
        "coerencia":         feat["coerencia"],
        "inversoes":         feat["inversoes"],
        "media_b_acertos":   feat["media_b_acertos"],
        "media_b_erros":     feat["media_b_erros"],
        "hardest_hit":       feat["hardest_hit"],
        "easiest_miss":      feat["easiest_miss"],
        "intervalo_min":     round(intervalo_min, 2),
        "intervalo_max":     round(intervalo_max, 2),
        "confianca":         conf,
        "metodo":            "lightgbm",
        "metodo_ancoragem":  "canonical_v6",
        "modelo_nome":       f"lgb_{chave}",
        "chave":             chave,
        "rmse_local":        rmse_local,
        "mae_local":         mae_local,
        "status_modelo":     status,
        "lingua_usada":      lingua,
        "vetor_canonico":    vetor_canon,
        "mascara_canonica":  mascara_canon,
    }


def status():
    """Diagnostico: quantos modelos disponíveis e em quarentena."""
    info = {"dir": _DIR_MODELOS, "n_pkl": 0, "aprovados": 0, "quarentena": 0}
    if not os.path.isdir(_DIR_MODELOS): return info
    for arq in os.listdir(_DIR_MODELOS):
        if not arq.endswith(".pkl"): continue
        info["n_pkl"] += 1
        chave = arq.replace(".pkl", "")
        try:
            payload = _carregar_modelo(chave)
            if payload and payload.get("status") == "aprovado":
                info["aprovados"] += 1
            else:
                info["quarentena"] += 1
        except Exception:
            pass
    return info


if __name__ == "__main__":
    import json
    print(json.dumps(status(), indent=2, ensure_ascii=False))
