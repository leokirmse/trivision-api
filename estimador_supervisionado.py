"""
estimador_supervisionado.py
───────────────────────────
Estima nota TRI usando modelos supervisionados locais (.pkl por prova).

Interface compatível com estimar_nota_compacto:
  estimar_nota_supervisionado(vetor, b_por_posicao, area, ano, tipo, cor,
                              mascara=None)

Retorna dict com:
  nota_estimada, modelo_usado, confianca, rmse_local, r2_local, ...
"""

import os, pickle
import numpy as np

import gerar_vetor_estrategico as gve   # apenas para reaproveitar FEATURES_ALL


# ═══════════════════════════════════════════════════════════════════════
#  CACHE DE MODELOS (lazy load)
# ═══════════════════════════════════════════════════════════════════════

_CACHE = {}     # chave -> pkg dict
_DIR_MODELOS = "modelos_supervisionados"


def configurar_dir(path):
    global _DIR_MODELOS
    _DIR_MODELOS = path


def carregar_modelo(area, ano, tipo, cor):
    chave = f"{area}_{ano}_{tipo}_{cor}"
    if chave in _CACHE: return _CACHE[chave]
    path = os.path.join(_DIR_MODELOS, f"{chave}.pkl")
    if not os.path.exists(path): return None
    with open(path, "rb") as f:
        pkg = pickle.load(f)
    _CACHE[chave] = pkg
    return pkg


# ═══════════════════════════════════════════════════════════════════════
#  RECONSTRUÇÃO DE FEATURES (espelha o gerador)
# ═══════════════════════════════════════════════════════════════════════

# Lista canônica (deve bater com gerar_modelos_supervisionados_tri.py)
FEATURES_BASICAS = [
    "ac", "coer", "inversoes",
    "mba", "mbe", "hardest_hit", "easiest_miss",
]
FEATURES_AVANCADAS = [
    "soma_b_acertos", "soma_b_erros",
    "skew_b_acertos", "kurt_b_acertos",
    "inversoes_fortes",
    "acertos_faceis", "acertos_medios", "acertos_dificeis",
    "erros_faceis",   "erros_dificeis",
    "acertos_q1", "acertos_q2", "acertos_q3", "acertos_q4",
    "easiest_hit", "hardest_miss",
    "longest_coherent_streak", "longest_incoherent_streak",
]
FEATURES_ALL = FEATURES_BASICAS + FEATURES_AVANCADAS


def calcular_features(vetor, b_por_pos, mascara=None):
    """Calcula as 24 features estruturais a partir do vetor + b."""
    if not vetor: return None
    bs_a, bs_e = [], []
    for i, ch in enumerate(vetor):
        if mascara and mascara[i] != "1": continue
        if ch == "9": continue
        b = b_por_pos[i]
        if b is None: continue
        (bs_a if ch == "1" else bs_e).append(b)

    if not (bs_a or bs_e): return None
    n_a, n_e = len(bs_a), len(bs_e)

    # Coerência e inversões (calcula aqui — não vem do parquet)
    inv = sum(1 for a in bs_a for e in bs_e if a > e)
    pares = n_a * n_e
    coer = (1 - inv/pares) if pares else 1.0

    todos_b = bs_a + bs_e
    f = {}
    f["ac"]            = n_a
    f["coer"]          = coer
    f["inversoes"]     = inv
    f["mba"]           = sum(bs_a)/n_a if n_a else 0.0
    f["mbe"]           = sum(bs_e)/n_e if n_e else 0.0
    f["hardest_hit"]   = max(bs_a) if bs_a else 0.0
    f["easiest_miss"]  = min(bs_e) if bs_e else 0.0
    f["soma_b_acertos"] = sum(bs_a)
    f["soma_b_erros"]   = sum(bs_e)

    if n_a >= 4:
        arr = np.array(bs_a)
        mean = arr.mean(); std = arr.std()
        if std > 0.01:
            f["skew_b_acertos"] = float(np.mean(((arr - mean)/std) ** 3))
            f["kurt_b_acertos"] = float(np.mean(((arr - mean)/std) ** 4) - 3)
        else:
            f["skew_b_acertos"] = 0.0; f["kurt_b_acertos"] = 0.0
    else:
        f["skew_b_acertos"] = 0.0; f["kurt_b_acertos"] = 0.0

    inv_fortes = 0
    if bs_a and bs_e:
        for ba in bs_a:
            for be in bs_e:
                if ba > be + 1.0: inv_fortes += 1
    f["inversoes_fortes"] = inv_fortes

    f["acertos_faceis"]   = sum(1 for b in bs_a if b < 0)
    f["acertos_medios"]   = sum(1 for b in bs_a if 0 <= b <= 1)
    f["acertos_dificeis"] = sum(1 for b in bs_a if b > 1)
    f["erros_faceis"]     = sum(1 for b in bs_e if b < 0)
    f["erros_dificeis"]   = sum(1 for b in bs_e if b > 1)

    q25, q50, q75 = np.percentile(todos_b, [25, 50, 75])
    f["acertos_q1"] = sum(1 for b in bs_a if b <= q25)
    f["acertos_q2"] = sum(1 for b in bs_a if q25 < b <= q50)
    f["acertos_q3"] = sum(1 for b in bs_a if q50 < b <= q75)
    f["acertos_q4"] = sum(1 for b in bs_a if b > q75)

    f["easiest_hit"]  = min(bs_a) if bs_a else 0.0
    f["hardest_miss"] = max(bs_e) if bs_e else 0.0

    marcas = sorted([(b, 1) for b in bs_a] + [(b, 0) for b in bs_e])
    coh = inc = 0
    if marcas:
        for _, m in marcas:
            if m == 1: coh += 1
            else: break
        for _, m in marcas:
            if m == 0: inc += 1
            else: break
    f["longest_coherent_streak"]   = coh
    f["longest_incoherent_streak"] = inc

    # Anota também o que precisamos no retorno
    f["_bs_a"] = bs_a; f["_bs_e"] = bs_e
    return f


def _predict(modelo, X):
    """Wrapper que lida com poly2 (tupla)."""
    if isinstance(modelo, tuple) and len(modelo) == 2:
        poly, m = modelo
        return m.predict(poly.transform(X))
    return modelo.predict(X)


# ═══════════════════════════════════════════════════════════════════════
#  ESTIMATIVA — INTERFACE PÚBLICA
# ═══════════════════════════════════════════════════════════════════════

def estimar_nota_supervisionado(vetor, b_por_posicao, area, ano, tipo, cor,
                                 mascara=None):
    pkg = carregar_modelo(area, ano, tipo, cor)
    if pkg is None:
        return {
            "erro": f"modelo não encontrado: {area}_{ano}_{tipo}_{cor}",
            "metodo": "indisponivel",
        }

    f = calcular_features(vetor, b_por_posicao, mascara)
    if f is None:
        return {"erro": "features inválidas", "metodo": "indisponivel"}

    feature_names = pkg.get("feature_names", FEATURES_ALL)
    X = np.array([[f[k] for k in feature_names]], dtype=np.float64)

    try:
        nota = float(_predict(pkg["modelo"], X)[0])
    except Exception as e:
        return {"erro": f"falha predição: {e}", "metodo": "erro_predicao"}

    mh = pkg.get("metricas_holdout", {})

    # Confiança baseada em RMSE local + amostra
    rmse = mh.get("rmse", 999)
    r2   = mh.get("r2", 0)
    if   r2 >= 0.95 and rmse < 20: confianca = "alta"
    elif r2 >= 0.85 and rmse < 35: confianca = "media"
    elif r2 >= 0.70:               confianca = "baixa"
    else:                          confianca = "muito_baixa"

    return {
        "nota_estimada":   round(nota, 1),
        "metodo":          f"supervisionado_{pkg['modelo_nome']}",
        "modelo_nome":     pkg["modelo_nome"],
        "metodo_ancoragem":(f"area={area}|ano={ano}|tipo={tipo}|cor={cor}"
                            f"|features={len(feature_names)}"
                            f"|n_treino={pkg['n_treino']}"),
        "confianca":       confianca,
        "rmse_local":      mh.get("rmse"),
        "mae_local":       mh.get("mae"),
        "r2_local":        mh.get("r2"),
        "pct_lt_30_local": mh.get("pct_lt_30"),
        "delta_max_local": mh.get("delta_max"),
        "intervalo_min":   round(nota - mh.get("mae", 30), 1),
        "intervalo_max":   round(nota + mh.get("mae", 30), 1),
        # Features (compat com estimador antigo)
        "acertos":         f["ac"],
        "erros":           len(f["_bs_e"]),
        "total_aplicaveis":f["ac"] + len(f["_bs_e"]),
        "coerencia":       round(f["coer"], 4),
        "inversoes":       f["inversoes"],
        "media_b_acertos": round(f["mba"], 4),
        "media_b_erros":   round(f["mbe"], 4),
        "hardest_hit":     round(f["hardest_hit"], 4),
        "easiest_miss":    round(f["easiest_miss"], 4),
        # Metadados locais
        "modelo_localidade":      pkg["label"],
        "densidade_amostral":     pkg["n_candidatos"],
        "confiabilidade_local":   pkg.get("confiabilidade"),
        "area": area, "ano": ano, "tipo": tipo, "cor": cor,
    }


# CLI rápido
if __name__ == "__main__":
    import argparse, sys
    p = argparse.ArgumentParser()
    p.add_argument("--area", required=True)
    p.add_argument("--ano",  type=int, required=True)
    p.add_argument("--tipo", default="regular")
    p.add_argument("--cor",  required=True)
    p.add_argument("--dir-modelos", default="modelos_supervisionados")
    args = p.parse_args()
    configurar_dir(args.dir_modelos)
    pkg = carregar_modelo(args.area, args.ano, args.tipo, args.cor)
    if pkg is None:
        print(f"[ERRO] modelo não encontrado"); sys.exit(1)
    print(f"Modelo: {pkg['label']}")
    print(f"  algoritmo:       {pkg['modelo_nome']}")
    print(f"  candidatos:      {pkg['n_candidatos']:,}")
    print(f"  features:        {len(pkg['feature_names'])}")
    print(f"  RMSE holdout:    {pkg['metricas_holdout']['rmse']:.2f}")
    print(f"  R²   holdout:    {pkg['metricas_holdout']['r2']:.4f}")
    print(f"  MAE  holdout:    {pkg['metricas_holdout']['mae']:.2f}")
    if pkg.get("importancia_features"):
        imp = pkg["importancia_features"]; tot = sum(imp.values()) or 1
        print(f"\n  Top 10 features:")
        for k in sorted(imp.keys(), key=lambda x: -imp[x])[:10]:
            print(f"    {k:<28} {100*imp[k]/tot:>5.1f}%")
