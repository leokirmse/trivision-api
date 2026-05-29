"""
estimador_nota.py
─────────────────
Estima a nota TRI de um candidato hipotético, a partir do vetor binário
de acertos, usando a base histórica real.

Fluxo:
  1) Calcula features do vetor (coerência, inversões, médias de b, hardest_hit, easiest_miss).
  2) Carrega o arquivo da prova (Parquet preferido, CSV fallback).
  3) Busca candidatos similares por distância de Hamming.
  4) Estima nota a partir da distribuição dos similares.
  5) Se houver poucos similares (< MIN_SIMILARES), faz fallback por regressão
     linear local entre acertos × média_b → nota usando todos os candidatos.
  6) Retorna dict com nota_estimada, intervalo, confiança e metadados.

USO COMO MÓDULO:
    from estimador_nota import estimar_nota_tri
    resultado = estimar_nota_tri(
        vetor='110101010...',
        b_por_posicao=[0.5, 1.2, 2.1, ...],  # mesmo tamanho do vetor
        area='MT', ano=2023, tipo='regular', cor='AMARELA',
        mascara='111...1' or None  # opcional, só para LC
    )

USO CLI:
    python estimador_nota.py --area MT --ano 2023 --tipo regular --cor AMARELA \
        --vetor 110101010101010...

Retorno:
    {
      "nota_estimada":    678.4,
      "intervalo_min":    642.0,
      "intervalo_max":    715.0,
      "confianca":        "alta" | "media" | "baixa",
      "candidatos_similares": 1281,
      "metodo":           "knn" | "regressao",
      "coerencia":        0.78,
      "acertos":          21,
      "inversoes":        87,
      "media_b_acertos":  0.62,
      "media_b_erros":    1.81,
      "hardest_hit":      1.92,
      "easiest_miss":     0.43,
    }
"""

import argparse, csv, json, os, sys, statistics, time
from collections import defaultdict
import heapq

PASTA = "features"

# Parâmetros do estimador
MAX_DIST_INICIAL  = 3       # tentativa inicial de Hamming
MAX_DIST_LIMITE   = 8       # até onde expandir se não houver candidatos
MIN_SIMILARES     = 50      # abaixo disso, fallback para regressão
LIMITE_SIMILARES  = 5000    # cap superior na quantidade de similares

try:
    import pyarrow.parquet as papq
    PARQUET = True
except ImportError:
    PARQUET = False


# ───────────────────────────────────────────────────────────────────────
#  FEATURES DO VETOR
# ───────────────────────────────────────────────────────────────────────

def calcular_features_vetor(vetor, b_por_posicao, mascara=None):
    """
    Calcula features do vetor binário.

    vetor:         string de '0'/'1' (45 ou 50 chars)
    b_por_posicao: lista de floats com b de cada posição (mesmo tamanho)
    mascara:       string opcional '0'/'1' indicando aplicabilidade (LC)

    Retorna dict com:
      acertos, erros, total_aplicaveis,
      coerencia, inversoes, pares_possiveis,
      media_b_acertos, media_b_erros,
      hardest_hit  (maior b acertado),
      easiest_miss (menor b errado).
    """
    if mascara and len(mascara) != len(vetor):
        raise ValueError("Máscara e vetor devem ter mesmo tamanho")
    if len(b_por_posicao) != len(vetor):
        raise ValueError("b_por_posicao deve ter mesmo tamanho do vetor")

    bs_acertos = []
    bs_erros   = []

    for i, c in enumerate(vetor):
        if mascara and mascara[i] != '1':
            continue
        b = b_por_posicao[i]
        if b is None: continue
        if c == '1': bs_acertos.append(b)
        elif c == '0': bs_erros.append(b)

    acertos = len(bs_acertos)
    erros   = len(bs_erros)
    aplicaveis = acertos + erros

    # Inversões (acertou difícil E errou fácil)
    inversoes = sum(1 for a in bs_acertos for e in bs_erros if a > e)
    pares = acertos * erros
    coerencia = (1 - inversoes/pares) if pares > 0 else 1.0

    return {
        "acertos":          acertos,
        "erros":            erros,
        "total_aplicaveis": aplicaveis,
        "coerencia":        round(coerencia, 4),
        "inversoes":        inversoes,
        "pares_possiveis":  pares,
        "media_b_acertos":  round(sum(bs_acertos)/acertos, 4) if acertos else 0.0,
        "media_b_erros":    round(sum(bs_erros)/erros, 4)   if erros   else 0.0,
        "hardest_hit":      round(max(bs_acertos), 4) if bs_acertos else None,
        "easiest_miss":     round(min(bs_erros), 4)   if bs_erros   else None,
    }


# ───────────────────────────────────────────────────────────────────────
#  CARREGAMENTO DA BASE
# ───────────────────────────────────────────────────────────────────────

def _arquivo_features(area, ano, tipo):
    base = os.path.join(PASTA, f"features_{area}_{ano}_{tipo}")
    if PARQUET and os.path.exists(base + ".parquet"):
        return base + ".parquet", "parquet"
    if os.path.exists(base + ".csv"):
        return base + ".csv", "csv"
    return None, None


def _iter_candidatos(path, fmt, cor=None):
    """Itera candidatos da base (lazy)."""
    if fmt == "parquet":
        tabela = papq.read_table(
            path,
            columns=["cor","vetor_binario_acertos","mascara_aplicavel",
                     "nota_real","acertos","coerencia_inversoes",
                     "media_b_acertos","media_b_erros"]
        )
        cols = tabela.column_names
        # Itera batch a batch para economizar memória
        for batch in tabela.to_batches(max_chunksize=8192):
            cor_arr = batch.column(cols.index("cor")).to_pylist()
            vet_arr = batch.column(cols.index("vetor_binario_acertos")).to_pylist()
            msk_arr = batch.column(cols.index("mascara_aplicavel")).to_pylist() \
                      if "mascara_aplicavel" in cols else [None]*len(cor_arr)
            nota_arr  = batch.column(cols.index("nota_real")).to_pylist()
            ac_arr    = batch.column(cols.index("acertos")).to_pylist()
            coer_arr  = batch.column(cols.index("coerencia_inversoes")).to_pylist()
            mba_arr   = batch.column(cols.index("media_b_acertos")).to_pylist()
            mbe_arr   = batch.column(cols.index("media_b_erros")).to_pylist()

            for i in range(len(cor_arr)):
                if cor and str(cor_arr[i]).upper() != cor.upper(): continue
                yield {
                    "vetor":     vet_arr[i],
                    "mascara":   msk_arr[i] if msk_arr[i] is not None else None,
                    "nota":      nota_arr[i],
                    "acertos":   ac_arr[i],
                    "coerencia": coer_arr[i],
                    "media_b_a": mba_arr[i],
                    "media_b_e": mbe_arr[i],
                }
    else:
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if cor and str(row.get("cor","")).upper() != cor.upper(): continue
                try:
                    yield {
                        "vetor":     row["vetor_binario_acertos"],
                        "mascara":   row.get("mascara_aplicavel") or None,
                        "nota":      float(row["nota_real"]),
                        "acertos":   int(row["acertos"]),
                        "coerencia": float(row["coerencia_inversoes"]),
                        "media_b_a": float(row.get("media_b_acertos") or 0),
                        "media_b_e": float(row.get("media_b_erros")   or 0),
                    }
                except (KeyError, ValueError, TypeError):
                    continue


def _hamming_mascarado(v1, v2, mascara=None):
    """Distância de Hamming considerando apenas posições aplicáveis."""
    n = min(len(v1), len(v2))
    if mascara:
        return sum(1 for i in range(n) if mascara[i] == '1' and v1[i] != v2[i])
    return sum(1 for i in range(n) if v1[i] != v2[i])


# ───────────────────────────────────────────────────────────────────────
#  BITMASK — Otimização realtime
# ───────────────────────────────────────────────────────────────────────

def _str_para_int(s):
    """Converte vetor binário string → int (bitmask)."""
    return int(s, 2) if s else 0


def _hamming_bitmask(int1, int2, mask_int=None):
    """
    Distância de Hamming via XOR + popcount nativo.
    int1.bit_count() (Python 3.10+) — milhares de vezes mais rápido que loop em string.
    """
    xor = int1 ^ int2
    if mask_int is not None:
        xor &= mask_int
    return xor.bit_count()


# Cache de bitmasks por base — preenchido sob demanda na primeira chamada
# Chave: id(lista_candidatos) → lista de ints alinhada
_BITMASK_CACHE = {}


def _obter_bitmasks(candidatos):
    """
    Retorna lista de ints (bitmask de cada vetor) alinhada com candidatos.
    Computa 1x por lista — chave é o id() do objeto cacheado em RAM.
    """
    key = id(candidatos)
    cache = _BITMASK_CACHE.get(key)
    if cache is not None and len(cache) == len(candidatos):
        return cache
    bms = [int(c["vetor"], 2) if c.get("vetor") else 0 for c in candidatos]
    _BITMASK_CACHE[key] = bms
    return bms


# ───────────────────────────────────────────────────────────────────────
#  ESTIMATIVA
# ───────────────────────────────────────────────────────────────────────

def _estimar_por_knn(vetor, similares):
    """Estatísticas dos N candidatos mais parecidos."""
    notas = [c["nota"] for c in similares]
    n = len(notas)
    if n == 0: return None
    notas_ord = sorted(notas)
    p10 = notas_ord[max(0, int(n*0.10))]
    p25 = notas_ord[max(0, int(n*0.25))]
    p75 = notas_ord[min(n-1, int(n*0.75))]
    p90 = notas_ord[min(n-1, int(n*0.90))]
    return {
        "nota_estimada":    round(statistics.mean(notas), 1),
        "nota_mediana":     round(statistics.median(notas), 1),
        "intervalo_min":    round(p10, 1),
        "intervalo_25":     round(p25, 1),
        "intervalo_75":     round(p75, 1),
        "intervalo_max":    round(p90, 1),
        "desvio_padrao":    round(statistics.stdev(notas), 1) if n >= 2 else 0,
    }


def _ancoragem_historica(base_path, base_fmt, cor, ac_alvo):
    """
    Estatísticas históricas globais e por número de acertos.
    Retorna nota_min/media/max globais e nota_min/media/max para candidatos
    com acertos == ac_alvo. Usado como âncora para evitar saltos artificiais.
    """
    notas_todas = []
    notas_ac    = []
    for c in _iter_candidatos(base_path, base_fmt, cor):
        notas_todas.append(c["nota"])
        if c["acertos"] == ac_alvo:
            notas_ac.append(c["nota"])

    if not notas_todas: return None

    nota_min   = min(notas_todas)
    nota_max   = max(notas_todas)
    nota_media = sum(notas_todas) / len(notas_todas)

    if notas_ac:
        ac_min = min(notas_ac)
        ac_max = max(notas_ac)
        ac_med = sum(notas_ac) / len(notas_ac)
        ac_n   = len(notas_ac)
        notas_ac_ord = sorted(notas_ac)
        p10 = notas_ac_ord[max(0, int(ac_n*0.10))]
        p15 = notas_ac_ord[max(0, int(ac_n*0.15))]
        p25 = notas_ac_ord[max(0, int(ac_n*0.25))]
    else:
        ac_min = ac_max = ac_med = None
        p10 = p15 = p25 = None
        ac_n = 0

    return {
        "nota_min_global":         round(nota_min, 1),
        "nota_max_global":         round(nota_max, 1),
        "nota_media_global":       round(nota_media, 1),
        "nota_media_por_acertos":  round(ac_med, 1) if ac_med is not None else None,
        "nota_min_por_acertos":    round(ac_min, 1) if ac_min is not None else None,
        "nota_max_por_acertos":    round(ac_max, 1) if ac_max is not None else None,
        "p10_por_acertos":         round(p10, 1) if p10 is not None else None,
        "p15_por_acertos":         round(p15, 1) if p15 is not None else None,
        "p25_por_acertos":         round(p25, 1) if p25 is not None else None,
        "n_por_acertos":           ac_n,
    }


def _estimar_por_regressao(features, base_path, base_fmt, cor, ancoragem, n_similares=0):
    """
    Fallback: regressão local com pesos baseados na similaridade de FEATURES.

    Mudanças vs versão anterior:
      - coerência tem peso negativo forte (baixa coerência → penaliza)
      - inversões e easiest_miss penalizam diretamente a nota
      - media_b_erros entra como sinal negativo
      - resultado é CLAMPADO ao intervalo histórico real
      - ancoragem usa média da nota para o nº de acertos alvo
    """
    ac_alvo   = features["acertos"]
    mb_a_alvo = features["media_b_acertos"]
    mb_e_alvo = features["media_b_erros"]
    coer_alvo = features["coerencia"]

    notas_proximas = []
    for c in _iter_candidatos(base_path, base_fmt, cor):
        if abs(c["acertos"] - ac_alvo) <= 2:
            # Diferenças por feature
            diff_b_a = abs((c["media_b_a"] or 0) - mb_a_alvo)
            diff_b_e = abs((c["media_b_e"] or 0) - mb_e_alvo)
            diff_coer = abs((c["coerencia"] or 0) - coer_alvo)

            # Peso composto — coerência domina (peso 5x), depois b dos erros, depois b dos acertos
            peso = 1.0 / (1.0
                          + 5.0 * diff_coer
                          + 2.0 * diff_b_e
                          + 1.0 * diff_b_a)
            notas_proximas.append((c["nota"], peso))
        if len(notas_proximas) > 50_000: break

    if not notas_proximas: return None

    soma_pesos = sum(p for _, p in notas_proximas)
    media_pond = sum(n*p for n, p in notas_proximas) / soma_pesos
    notas = [n for n, _ in notas_proximas]
    notas_ord = sorted(notas)
    nn = len(notas)
    p15 = notas_ord[max(0, int(nn*0.15))]
    p85 = notas_ord[min(nn-1, int(nn*0.85))]

    # ─── Ancoragem histórica ────────────────────────────────────────────────
    nota_estimada = media_pond
    metodo_ancoragem = "regressao_simples"
    easiest_miss = features.get("easiest_miss")

    if ancoragem:
        # ─── Caso especial: acc=0 ─────────────────────────────────────────
        if ac_alvo == 0 and ancoragem.get("nota_min_global") is not None:
            nota_estimada = ancoragem["nota_min_global"]
            metodo_ancoragem = "minimo_historico"

        # ─── Ancoragem por coerência ──────────────────────────────────────
        elif ancoragem.get("n_por_acertos", 0) >= 30:
            media_emp = ancoragem.get("nota_media_por_acertos")
            p10 = ancoragem.get("p10_por_acertos")
            p15 = ancoragem.get("p15_por_acertos")
            p25 = ancoragem.get("p25_por_acertos")

            # Escolhe a âncora conforme nível de coerência
            if coer_alvo < 0.3:
                # Padrão extremamente incoerente: âncora muito baixa
                base_anchor = p10 if p10 is not None else (p15 if p15 is not None else media_emp)
                metodo_ancoragem = "ancoragem_p10_incoerente_extremo"
            elif coer_alvo < 0.5:
                base_anchor = p25 if p25 is not None else media_emp
                metodo_ancoragem = "ancoragem_p25_baixa_coerencia"
            else:
                base_anchor = media_emp
                metodo_ancoragem = "ancoragem_empirica"

            if base_anchor is None:
                base_anchor = media_pond

            # Mistura: 70% âncora + 30% regressão por features
            nota_estimada = 0.70 * base_anchor + 0.30 * media_pond

            # ─── Penalidades cumulativas ─────────────────────────────────
            # 1. Penalidade por coerência baixa (aumentada de 80 → 120)
            if coer_alvo < 0.5:
                penalidade_coer = (0.5 - coer_alvo) * 120
                nota_estimada -= penalidade_coer

            # 2. Penalidade por easiest_miss baixo (errou questão muito fácil)
            #    Quanto menor o b da questão mais fácil errada, maior a penalidade.
            if easiest_miss is not None and easiest_miss < 0.5:
                penalidade_em = (0.5 - easiest_miss) * 30  # até ~24 pontos
                nota_estimada -= penalidade_em

            # 3. Penalidade adicional quando NÃO houve nenhum vizinho real
            if n_similares == 0:
                nota_estimada -= 25  # vetor "impossível" na base histórica

        # ─── Clamp dentro do intervalo histórico ──────────────────────────
        nmin = ancoragem.get("nota_min_global")
        nmax = ancoragem.get("nota_max_global")
        if nmin is not None: nota_estimada = max(nota_estimada, nmin)
        if nmax is not None: nota_estimada = min(nota_estimada, nmax)

    return {
        "nota_estimada":     round(nota_estimada, 1),
        "nota_mediana":      round(statistics.median(notas), 1),
        "intervalo_min":     round(p15, 1),
        "intervalo_max":     round(p85, 1),
        "desvio_padrao":     round(statistics.stdev(notas), 1) if nn >= 2 else 0,
        "n_base":            nn,
        "metodo_ancoragem":  metodo_ancoragem,
    }


def _calcular_confianca(n_similares, dist_media):
    """Retorna 'alta', 'media' ou 'baixa' baseado em quantidade e proximidade."""
    if n_similares >= 200 and dist_media <= 2: return "alta"
    if n_similares >= 50  and dist_media <= 4: return "media"
    return "baixa"


# ───────────────────────────────────────────────────────────────────────
#  API PRINCIPAL
# ───────────────────────────────────────────────────────────────────────

def estimar_nota_tri(vetor, b_por_posicao, area, ano, tipo, cor,
                     mascara=None, max_dist_inicial=MAX_DIST_INICIAL):
    """
    Estima a nota TRI de um candidato hipotético.

    Retorna dict completo (ver docstring do módulo).
    """
    # 1. Features do vetor
    t_stage = time.time()
    feats = calcular_features_vetor(vetor, b_por_posicao, mascara)
    t_features = (time.time()-t_stage)*1000

    # 2. Carrega arquivo da prova
    t_stage = time.time()
    path, fmt = _arquivo_features(area, ano, tipo)
    if not path:
        return {
            "erro": f"Base não encontrada: features_{area}_{ano}_{tipo}",
            **feats,
        }

    # Materializa lista de candidatos (do cache em RAM se houver — vide api.py)
    candidatos = list(_iter_candidatos(path, fmt, cor))
    t_load = (time.time()-t_stage)*1000

    if not candidatos:
        return {"erro": "Sem candidatos na base", **feats}

    # 3. PASSADA ÚNICA: calcula Hamming via bitmask + coleta para ancoragem
    t_stage = time.time()
    vetor_int = int(vetor, 2)
    mask_int  = int(mascara, 2) if mascara else None

    bms = _obter_bitmasks(candidatos)

    ac_alvo = feats["acertos"]
    # Heap min de (dist, nota, candidato_idx) limitado a LIMITE_SIMILARES
    # Mas para ser ainda mais rápido, mantemos buckets por distância
    buckets = defaultdict(list)  # dist → [(nota, idx)]

    # Ancoragem: notas globais e por acertos
    notas_todas = []
    notas_ac    = []

    for i, c in enumerate(candidatos):
        # Hamming via XOR mascarado (nativo Python — bit_count é C otimizado)
        xor = vetor_int ^ bms[i]
        if mask_int is not None:
            xor &= mask_int
        d = xor.bit_count()

        if d <= MAX_DIST_LIMITE:
            buckets[d].append((c["nota"], i))

        # Ancoragem na mesma passagem
        notas_todas.append(c["nota"])
        if c["acertos"] == ac_alvo:
            notas_ac.append(c["nota"])

    t_search = (time.time()-t_stage)*1000

    # 4. Expansão progressiva: começa com dist=MAX_DIST_INICIAL e cresce
    t_stage = time.time()
    similares = []
    dist_max  = MAX_DIST_INICIAL
    while dist_max <= MAX_DIST_LIMITE:
        similares = []
        for d in range(dist_max + 1):
            for nota, idx in buckets.get(d, []):
                c = candidatos[idx]
                similares.append({
                    "vetor":     c["vetor"],
                    "mascara":   c.get("mascara"),
                    "nota":      c["nota"],
                    "acertos":   c["acertos"],
                    "coerencia": c["coerencia"],
                    "media_b_a": c["media_b_a"],
                    "media_b_e": c["media_b_e"],
                    "cor":       c.get("cor"),
                    "distancia": d,
                })
                if len(similares) >= LIMITE_SIMILARES: break
            if len(similares) >= LIMITE_SIMILARES: break
        if len(similares) >= MIN_SIMILARES: break
        dist_max += 1
    t_filter = (time.time()-t_stage)*1000

    n_sim = len(similares)
    dist_media = (sum(s["distancia"] for s in similares) / n_sim) if n_sim else None

    # 4b. Ancoragem in-line (sem reler base)
    t_stage = time.time()
    if notas_todas:
        nota_min_g = min(notas_todas)
        nota_max_g = max(notas_todas)
        nota_med_g = sum(notas_todas) / len(notas_todas)
        if notas_ac:
            ac_n = len(notas_ac)
            notas_ac_ord = sorted(notas_ac)
            p10 = notas_ac_ord[max(0, int(ac_n*0.10))]
            p15 = notas_ac_ord[max(0, int(ac_n*0.15))]
            p25 = notas_ac_ord[max(0, int(ac_n*0.25))]
            ancoragem = {
                "nota_min_global":         round(nota_min_g, 1),
                "nota_max_global":         round(nota_max_g, 1),
                "nota_media_global":       round(nota_med_g, 1),
                "nota_media_por_acertos":  round(sum(notas_ac)/ac_n, 1),
                "nota_min_por_acertos":    round(min(notas_ac), 1),
                "nota_max_por_acertos":    round(max(notas_ac), 1),
                "p10_por_acertos":         round(p10, 1),
                "p15_por_acertos":         round(p15, 1),
                "p25_por_acertos":         round(p25, 1),
                "n_por_acertos":           ac_n,
            }
        else:
            ancoragem = {
                "nota_min_global":         round(nota_min_g, 1),
                "nota_max_global":         round(nota_max_g, 1),
                "nota_media_global":       round(nota_med_g, 1),
                "nota_media_por_acertos":  None, "nota_min_por_acertos": None,
                "nota_max_por_acertos":    None, "p10_por_acertos":     None,
                "p15_por_acertos":         None, "p25_por_acertos":     None,
                "n_por_acertos":           0,
            }
    else:
        ancoragem = None
    t_anchor = (time.time()-t_stage)*1000

    # 4b. Caso especial: acc=0 → retorna mínimo histórico direto
    metodo_ancoragem = None
    if feats["acertos"] == 0 and ancoragem and ancoragem.get("nota_min_global") is not None:
        nota_min = ancoragem["nota_min_global"]
        resultado = {
            "nota_estimada":        nota_min,
            "nota_mediana":         nota_min,
            "intervalo_min":        nota_min,
            "intervalo_max":        ancoragem.get("nota_min_por_acertos") or nota_min,
            "desvio_padrao":        0,
            "confianca":            "alta",
            "candidatos_similares": ancoragem.get("n_por_acertos", 0),
            "distancia_max_usada":  None,
            "distancia_media":      None,
            "metodo":               "minimo_historico",
            "metodo_ancoragem":     "minimo_historico",
            "acertos":              feats["acertos"],
            "erros":                feats["erros"],
            "coerencia":            feats["coerencia"],
            "inversoes":            feats["inversoes"],
            "pares_possiveis":      feats["pares_possiveis"],
            "media_b_acertos":      feats["media_b_acertos"],
            "media_b_erros":        feats["media_b_erros"],
            "hardest_hit":          feats["hardest_hit"],
            "easiest_miss":         feats["easiest_miss"],
            "nota_minima_historica":  ancoragem["nota_min_global"],
            "nota_maxima_historica":  ancoragem["nota_max_global"],
            "nota_media_por_acertos": ancoragem["nota_media_por_acertos"],
            "area": area, "ano": ano, "tipo": tipo, "cor": cor,
        }
        return resultado

    # 5. kNN ou regressão de fallback
    if n_sim >= MIN_SIMILARES:
        estat = _estimar_por_knn(vetor, similares)
        metodo = "knn"
        confianca = _calcular_confianca(n_sim, dist_media)
        metodo_ancoragem = None

        # ─── Penalidade kNN progressiva por incoerência ────────────────
        # A penalidade cresce com:
        #   • quão baixa é a coerência (gap até 0.5)
        #   • quantos acertos foram feitos (mais acertos → nota base maior → penalidade maior)
        #   • quão baixo é easiest_miss (errou questão muito fácil)
        #   • quão alto é media_b_acertos (acertou difícil = padrão suspeito se incoerente)
        coer = feats["coerencia"]
        ac   = feats["acertos"]
        em   = feats["easiest_miss"]
        mba  = feats["media_b_acertos"]

        if ancoragem and coer < 0.5:
            # Fator base: cresce com o gap até 0.5 (atinge 1.0 quando coer=0)
            gap_coer = (0.5 - coer) / 0.5
            # Escala por nº de acertos: penalidade leve em n baixos, forte em n altos
            # 0 acertos → fator 0; 20 → ~0.5; 40 → ~1.0
            escala_n = min(1.0, ac / 40.0)
            # Componente base da penalidade
            penalidade = gap_coer * 60 + (gap_coer * escala_n * 120)

            # Bônus: errou questão muito fácil
            if em is not None and em < 0.5 and coer < 0.5:
                penalidade += (0.5 - em) * 25

            # Bônus: media_b_acertos alta + coerência baixa = padrão suspeito
            if mba is not None and mba > 1.5 and coer < 0.3:
                penalidade += (mba - 1.5) * 30

            # ─── Ancoragem dura: nota nunca pode ultrapassar mediana   ─
            #     dos candidatos REAIS com o mesmo nº de acertos.
            #     Para coerência muito baixa, usa p25 ou p10.
            teto = None
            if ancoragem.get("n_por_acertos", 0) >= 30:
                if coer < 0.2 and ancoragem.get("p10_por_acertos") is not None:
                    teto = ancoragem["p10_por_acertos"]
                elif coer < 0.35 and ancoragem.get("p25_por_acertos") is not None:
                    teto = ancoragem["p25_por_acertos"]
                elif coer < 0.5 and ancoragem.get("nota_media_por_acertos") is not None:
                    teto = ancoragem["nota_media_por_acertos"]

            nota_apos_penalidade = estat["nota_estimada"] - penalidade
            if teto is not None:
                nota_apos_penalidade = min(nota_apos_penalidade, teto)

            # Floor: mínimo histórico global
            estat["nota_estimada"] = max(
                ancoragem["nota_min_global"],
                nota_apos_penalidade
            )
            metodo_ancoragem = "knn_com_penalidade_coerencia"
            # Confiança cai quando precisamos penalizar fortemente
            if coer < 0.3:
                confianca = "muito_baixa" if confianca in ("alta","media") else confianca
            elif coer < 0.5 and confianca == "alta":
                confianca = "media"
    else:
        estat = _estimar_por_regressao(feats, path, fmt, cor, ancoragem, n_similares=n_sim)
        metodo = "regressao"
        # Confiança baixa em geral; piora se coerência também é baixa
        confianca = "muito_baixa" if feats["coerencia"] < 0.3 else "baixa"
        if estat is None:
            return {
                "erro": "Sem dados suficientes para estimativa",
                **feats,
            }
        metodo_ancoragem = estat.get("metodo_ancoragem", "regressao_simples")

    # 5. Monta retorno final
    resultado = {
        "nota_estimada":        estat["nota_estimada"],
        "nota_mediana":         estat.get("nota_mediana"),
        "intervalo_min":        estat["intervalo_min"],
        "intervalo_max":        estat["intervalo_max"],
        "desvio_padrao":        estat.get("desvio_padrao"),
        "confianca":            confianca,
        "candidatos_similares": n_sim,
        "distancia_max_usada":  dist_max if metodo == "knn" else None,
        "distancia_media":      round(dist_media, 2) if dist_media is not None else None,
        "metodo":               metodo,
        "metodo_ancoragem":     metodo_ancoragem,

        # Features do vetor
        "acertos":              feats["acertos"],
        "erros":                feats["erros"],
        "coerencia":            feats["coerencia"],
        "inversoes":            feats["inversoes"],
        "pares_possiveis":      feats["pares_possiveis"],
        "media_b_acertos":      feats["media_b_acertos"],
        "media_b_erros":        feats["media_b_erros"],
        "hardest_hit":          feats["hardest_hit"],
        "easiest_miss":         feats["easiest_miss"],

        # Ancoragem histórica
        "nota_minima_historica":  ancoragem["nota_min_global"]        if ancoragem else None,
        "nota_maxima_historica":  ancoragem["nota_max_global"]        if ancoragem else None,
        "nota_media_por_acertos": ancoragem["nota_media_por_acertos"] if ancoragem else None,

        # Contexto
        "area": area, "ano": ano, "tipo": tipo, "cor": cor,
    }

    # Log de timing por estágio (útil para profile via API logs)
    total = t_features + t_load + t_search + t_filter + t_anchor
    if os.environ.get("TRI_TIMING"):
        print(f"[TIMING] feat={t_features:.1f} load={t_load:.1f} "
              f"search={t_search:.1f} filt={t_filter:.1f} anchor={t_anchor:.1f} "
              f"total={total:.1f}ms cand={len(candidatos)} sim={n_sim}")

    return resultado


# ───────────────────────────────────────────────────────────────────────
#  CLI
# ───────────────────────────────────────────────────────────────────────

def _carregar_b_por_posicao(area, ano, tipo, cor):
    """
    Carrega b de cada posição da prova consultando provas_todas.json
    + ITENS_PROVA_XXXX.csv. Retorna (b_por_posicao, mascara_aplicavel).
    Útil para o CLI sem precisar passar manualmente os b.
    """
    # Tenta encontrar co_prova no provas_todas.json
    if not os.path.exists("provas_todas.json"):
        return None, None
    with open("provas_todas.json", encoding="utf-8") as f:
        provas = json.load(f)

    def norm_tipo(t):
        t = t.lower()
        if t in ("reaplic","reaplicacao","reaplicação"): return "reaplicacao"
        if t == "ppl": return "ppl"
        return "regular"

    cores_ano = provas.get(str(ano), {}).get(area, {})
    co_prova = None
    for c, tipos in cores_ano.items():
        if c.upper() != cor.upper(): continue
        for t, info in tipos.items():
            if norm_tipo(t) == norm_tipo(tipo):
                co_prova = str(info.get("co_prova","")).strip()
                break

    if not co_prova:
        return None, None

    # Lê ITENS_PROVA_{ano}.csv
    path_itens = os.path.join("dados_inep", f"ITENS_PROVA_{ano}.csv")
    if not os.path.exists(path_itens):
        return None, None

    with open(path_itens, encoding="latin-1") as f:
        sep = ";" if ";" in f.readline() else ","

    # Coleta itens daquela prova
    itens = []  # (CO_POSICAO, b, tp_lingua)
    with open(path_itens, encoding="latin-1", newline="") as f:
        for row in csv.DictReader(f, delimiter=sep):
            if str(row.get("CO_PROVA","")).strip() != co_prova: continue
            try: pos = int(row.get("CO_POSICAO",0) or 0)
            except: pos = 0
            try:
                b = float(str(row.get("NU_PARAM_B","")).replace(",","."))
                if not (-5 <= b <= 6): b = None
            except: b = None
            tp_ling = row.get("TP_LINGUA","").strip()
            itens.append((pos, b, tp_ling))

    if not itens: return None, None

    # Para LC, monta 50 itens: 5 ing + 5 esp + 40 comuns
    if area == "LC":
        ingles   = sorted([i for i in itens if i[2] == "0"], key=lambda x: x[0])[:5]
        espanhol = sorted([i for i in itens if i[2] == "1"], key=lambda x: x[0])[:5]
        comuns   = sorted([i for i in itens if i[2] not in ("0","1")], key=lambda x: x[0])[:40]
        b_pos = [b for _, b, _ in ingles + espanhol + comuns]
        return b_pos, None  # máscara depende do candidato, fora do escopo do CLI
    else:
        itens.sort(key=lambda x: x[0])
        b_pos = [b for _, b, _ in itens]
        return b_pos, None


def main():
    p = argparse.ArgumentParser(description="Estima nota TRI a partir de vetor binário")
    p.add_argument("--area", required=True, choices=["MT","CN","CH","LC"])
    p.add_argument("--ano",  required=True, type=int)
    p.add_argument("--tipo", required=True, choices=["regular","reaplicacao","ppl"])
    p.add_argument("--cor",  required=True)
    p.add_argument("--vetor",   required=True)
    p.add_argument("--mascara", default=None, help="Máscara de aplicabilidade (LC)")
    p.add_argument("--json", action="store_true", help="Saída como JSON puro")
    args = p.parse_args()

    b_pos, _ = _carregar_b_por_posicao(args.area, args.ano, args.tipo, args.cor)
    if b_pos is None:
        print(f"[ERRO] Não foi possível carregar b das posições. "
              f"Verifique provas_todas.json e dados_inep/ITENS_PROVA_{args.ano}.csv")
        sys.exit(1)

    if len(b_pos) != len(args.vetor):
        print(f"[ERRO] Vetor tem {len(args.vetor)} chars, prova tem {len(b_pos)} posições")
        sys.exit(1)

    resultado = estimar_nota_tri(
        vetor=args.vetor, b_por_posicao=b_pos,
        area=args.area, ano=args.ano, tipo=args.tipo, cor=args.cor,
        mascara=args.mascara,
    )

    if args.json:
        print(json.dumps(resultado, ensure_ascii=False, indent=2))
        return

    if "erro" in resultado:
        print(f"\n[ERRO] {resultado['erro']}\n")
        return

    print(f"\n{'='*60}")
    print(f"  ESTIMATIVA TRI — {args.area} {args.ano} {args.tipo} {args.cor}")
    print(f"{'='*60}\n")
    print(f"  Acertos:          {resultado['acertos']} / {resultado['acertos']+resultado['erros']}")
    print(f"  Coerência:        {resultado['coerencia']:.3f}  (inversões: {resultado['inversoes']})")
    print(f"  Média b acertos:  {resultado['media_b_acertos']:.2f}")
    print(f"  Média b erros:    {resultado['media_b_erros']:.2f}")
    if resultado["hardest_hit"]   is not None:
        print(f"  Hardest hit:      b={resultado['hardest_hit']:.2f}")
    if resultado["easiest_miss"]  is not None:
        print(f"  Easiest miss:     b={resultado['easiest_miss']:.2f}")
    print()
    print(f"  Nota estimada:    {resultado['nota_estimada']}")
    print(f"  Mediana:          {resultado.get('nota_mediana','—')}")
    print(f"  Intervalo:        [{resultado['intervalo_min']} — {resultado['intervalo_max']}]")
    if resultado.get('desvio_padrao'):
        print(f"  Desvio padrão:    {resultado['desvio_padrao']}")
    print(f"  Confiança:        {resultado['confianca'].upper()}")
    print(f"  Método:           {resultado['metodo']}")
    if resultado.get('metodo_ancoragem'):
        print(f"  Ancoragem:        {resultado['metodo_ancoragem']}")
    print(f"  Similares:        {resultado['candidatos_similares']:,}")
    if resultado.get('distancia_media') is not None:
        print(f"  Dist. média:      {resultado['distancia_media']}")
    if resultado.get('nota_minima_historica') is not None:
        print(f"\n  Referência histórica:")
        print(f"    Min global:           {resultado['nota_minima_historica']}")
        print(f"    Max global:           {resultado['nota_maxima_historica']}")
        if resultado.get('nota_media_por_acertos') is not None:
            print(f"    Média p/ {resultado['acertos']} acertos: {resultado['nota_media_por_acertos']}")


if __name__ == "__main__":
    main()
