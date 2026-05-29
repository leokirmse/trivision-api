"""
estimador_compacto.py
─────────────────────
Estima notas TRI usando apenas o artefato compacto `modelo_compacto_tri.json`,
sem depender de `features/*.parquet`. Tamanho típico: 10–30 MB total.

API compatível com `estimador_nota.estimar_nota_tri`:
  estimar_nota_compacto(vetor, b_por_posicao, area, ano, tipo, cor, mascara=None)
    → dict idêntico em estrutura ao do estimador histórico

Como funciona:
  1. Calcula features do vetor (acertos, coerência, b's médios, etc.)
  2. Lookup no modelo compacto:
     - distribuição estatística para esse nº de acertos
     - se coerência alta → usa média histórica de (acertos, faixa coer alta)
     - se coerência baixa → aplica penalidades calibradas + teto p25/p10
  3. Refina pela posição de media_b_acertos / media_b_erros (bins)
  4. Aplica clamp ao intervalo histórico global

USO:
  from estimador_compacto import carregar_modelo, estimar_nota_compacto
  modelo = carregar_modelo("modelo_compacto_tri.json")
  res = estimar_nota_compacto(vetor, b_por_pos, "MT", 2024, "regular", "AMARELA",
                              modelo=modelo)
"""

import json, os, statistics

# ───────────────────────────────────────────────────────────────────────
#  CARREGAMENTO
# ───────────────────────────────────────────────────────────────────────

_MODELO_GLOBAL = None


def carregar_modelo(path="modelo_compacto_tri.json"):
    """Lê o modelo compacto do disco e cacheia globalmente."""
    global _MODELO_GLOBAL
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as f:
        _MODELO_GLOBAL = json.load(f)
    return _MODELO_GLOBAL


def _modelo():
    global _MODELO_GLOBAL
    if _MODELO_GLOBAL is None:
        carregar_modelo()
    return _MODELO_GLOBAL


# ───────────────────────────────────────────────────────────────────────
#  FEATURES DO VETOR (mesma lógica do estimador_nota)
# ───────────────────────────────────────────────────────────────────────

def calcular_features(vetor, b_por_posicao, mascara=None):
    if mascara and len(mascara) != len(vetor):
        raise ValueError("máscara e vetor com tamanhos diferentes")
    if len(b_por_posicao) != len(vetor):
        raise ValueError("b_por_posicao com tamanho diferente do vetor")

    bs_a, bs_e = [], []
    for i, c in enumerate(vetor):
        if mascara and mascara[i] != "1": continue
        b = b_por_posicao[i]
        if b is None: continue
        (bs_a if c == "1" else bs_e).append(b)

    ac, er = len(bs_a), len(bs_e)
    inv = sum(1 for a in bs_a for e in bs_e if a > e)
    pares = ac * er
    coer = (1 - inv/pares) if pares else 1.0
    return {
        "acertos":         ac,
        "erros":           er,
        "total_aplicaveis": ac + er,
        "coerencia":       round(coer, 4),
        "inversoes":       inv,
        "pares_possiveis": pares,
        "media_b_acertos": round(sum(bs_a)/ac, 4) if ac else 0.0,
        "media_b_erros":   round(sum(bs_e)/er, 4) if er else 0.0,
        "hardest_hit":     round(max(bs_a), 4) if bs_a else None,
        "easiest_miss":    round(min(bs_e), 4) if bs_e else None,
    }


# ───────────────────────────────────────────────────────────────────────
#  LOOKUP NO MODELO COMPACTO
# ───────────────────────────────────────────────────────────────────────

def _faixa_coerencia(coer):
    if   coer < 0.15: return "0.00-0.15"
    elif coer < 0.30: return "0.15-0.30"
    elif coer < 0.45: return "0.30-0.45"
    elif coer < 0.60: return "0.45-0.60"
    elif coer < 0.75: return "0.60-0.75"
    elif coer < 0.90: return "0.75-0.90"
    else:             return "0.90-1.00"


def _bin_b(valor):
    if valor is None: return None
    bins = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    for i in range(len(bins)-1):
        if bins[i] <= valor < bins[i+1]:
            return f"{bins[i]}-{bins[i+1]}"
    return None


def _faixas_coer_adjacentes(coer):
    """Retorna as 2 faixas mais próximas + pesos para interpolação."""
    pontos = [
        (0.075, "0.00-0.15"),
        (0.225, "0.15-0.30"),
        (0.375, "0.30-0.45"),
        (0.525, "0.45-0.60"),
        (0.675, "0.60-0.75"),
        (0.825, "0.75-0.90"),
        (0.95,  "0.90-1.00"),
    ]
    # Encontra a faixa atual e a adjacente mais próxima
    for i, (centro, lab) in enumerate(pontos):
        if coer <= centro:
            if i == 0: return [(lab, 1.0)]
            # interpola entre i-1 e i
            c_prev, lab_prev = pontos[i-1]
            peso_prev = (centro - coer) / (centro - c_prev)
            peso_curr = 1 - peso_prev
            return [(lab_prev, peso_prev), (lab, peso_curr)]
    return [(pontos[-1][1], 1.0)]


def _refinar_por_bins_b(prova, ac, mba, mbe, base):
    """
    Ajusta a nota base pela diferença entre o valor médio do bin de
    media_b_acertos / media_b_erros e a média geral daquele nº de acertos.
    """
    media_ac = prova["por_acertos"].get(str(ac), {}).get("media")
    if media_ac is None: return base, "sem_ref"
    ajuste = 0.0
    info = []
    bins_a = prova.get("bins_media_b_acertos", {}).get(str(ac), {})
    if bins_a and mba is not None:
        lab = _bin_b(mba)
        if lab and lab in bins_a:
            diff = bins_a[lab]["media_nota"] - media_ac
            ajuste += diff * 0.35   # peso 0.35 do desvio do bin de acertos
            info.append(f"mba_bin={lab}({diff:+.0f})")
    bins_e = prova.get("bins_media_b_erros", {}).get(str(ac), {})
    if bins_e and mbe is not None:
        lab = _bin_b(mbe)
        if lab and lab in bins_e:
            diff = bins_e[lab]["media_nota"] - media_ac
            ajuste += diff * 0.20   # peso 0.20 do desvio do bin de erros
            info.append(f"mbe_bin={lab}({diff:+.0f})")
    return base + ajuste, "+".join(info) if info else "sem_bins"


# ───────────────────────────────────────────────────────────────────────
#  ESTIMATIVA
# ───────────────────────────────────────────────────────────────────────

def _motor_incoerencia_controlada(prova, feats, nota_base, n_amostra):
    """
    Submotor de incoerência que consome `prova["incoerencia_model"]`
    aprendido a partir dos próprios candidatos desta prova específica.

    NÃO usa curva universal. NÃO tem limiares de coer fixos.
    A "regra" vem 100% dos dados reais armazenados em:
      prova["incoerencia_model"][str(ac)]["por_coerencia"][faixa]["nota"]

    Quando o ac do candidato não tem dados, faz interpolação LOCAL entre
    acertos vizinhos da MESMA prova.

    Retorna: (nota_final, teto_local, piso_local, info)
    """
    ac    = feats["acertos"]
    coer  = feats["coerencia"]
    em    = feats["easiest_miss"]
    mba   = feats["media_b_acertos"]
    mbe   = feats["media_b_erros"]
    g     = prova["global"]

    inco_model = prova.get("incoerencia_model", {})
    info_ac    = inco_model.get(str(ac))

    # Se não há dados locais para este ac, busca vizinhos no próprio modelo
    fonte = "direta"
    suavizado_externo = False
    if info_ac is None:
        # Procura ac mais próximo com dados, dentro da MESMA prova
        candidatos_ac = []
        for delta in range(1, 6):
            for ac_viz in (ac - delta, ac + delta):
                viz = inco_model.get(str(ac_viz))
                if viz: candidatos_ac.append((delta, ac_viz, viz))
            if candidatos_ac: break
        if candidatos_ac:
            _, ac_viz, info_ac = candidatos_ac[0]
            fonte = f"vizinho_ac={ac_viz}"
            suavizado_externo = True
        else:
            # Sem nada — devolve nota_base (motor não pode trabalhar)
            return nota_base, None, None, {
                "razao": "sem_dados_locais", "fonte": "indisponivel",
                "ancora": None, "ajustes": [],
            }

    por_coer = info_ac["por_coerencia"]

    # ── Identifica a faixa do candidato e a faixa adjacente ───────────
    FAIXAS_CENTROS = [
        ("0.00-0.15", 0.075),
        ("0.15-0.30", 0.225),
        ("0.30-0.45", 0.375),
        ("0.45-0.60", 0.525),
        ("0.60-0.75", 0.675),
        ("0.75-0.90", 0.825),
        ("0.90-1.00", 0.95),
    ]

    # Faixa exata do candidato
    faixa_atual = next((f for f, lo, hi in [
        ("0.00-0.15", 0.0, 0.15),
        ("0.15-0.30", 0.15, 0.30),
        ("0.30-0.45", 0.30, 0.45),
        ("0.45-0.60", 0.45, 0.60),
        ("0.60-0.75", 0.60, 0.75),
        ("0.75-0.90", 0.75, 0.90),
        ("0.90-1.00", 0.90, 1.01),
    ] if lo <= coer < hi), "0.00-0.15")

    # ── Âncora: interpolação entre 2 faixas LOCAIS adjacentes ──────────
    # Encontra a faixa do candidato e a vizinha mais próxima
    ancora = None
    faixa_usada = None
    suavizado_faixa = False
    teto_local = piso_local = None

    if faixa_atual in por_coer:
        ancora = por_coer[faixa_atual]["nota"]
        faixa_usada = faixa_atual
        suavizado_faixa = por_coer[faixa_atual].get("suavizado", False)
    else:
        # Busca faixa mais próxima entre as disponíveis
        idx_atual = next((i for i, (f, _) in enumerate(FAIXAS_CENTROS) if f == faixa_atual), 0)
        for delta in range(1, len(FAIXAS_CENTROS)):
            for direcao in (-1, 1):
                idx_viz = idx_atual + direcao*delta
                if 0 <= idx_viz < len(FAIXAS_CENTROS):
                    f_viz = FAIXAS_CENTROS[idx_viz][0]
                    if f_viz in por_coer:
                        ancora = por_coer[f_viz]["nota"]
                        faixa_usada = f_viz
                        suavizado_faixa = True
                        break
            if ancora is not None: break

    if ancora is None:
        # Sem nenhuma faixa disponível — usa ancora_coerente como fallback
        ancora = info_ac.get("ancora_coerente") or nota_base

    # ── Interpolação suave entre faixa atual e adjacente ──────────────
    # Se ambas as faixas adjacentes existem, usa peso linear entre os centros
    centro_atual = next((c for f, c in FAIXAS_CENTROS if f == faixa_atual), 0.5)
    if faixa_atual in por_coer:
        nota_atual = por_coer[faixa_atual]["nota"]
        # Vizinha mais próxima
        idx = next((i for i,(f,_) in enumerate(FAIXAS_CENTROS) if f == faixa_atual), 0)
        viz_idx = None
        if coer < centro_atual and idx > 0: viz_idx = idx - 1
        elif coer >= centro_atual and idx < len(FAIXAS_CENTROS)-1: viz_idx = idx + 1
        if viz_idx is not None:
            f_viz, c_viz = FAIXAS_CENTROS[viz_idx]
            if f_viz in por_coer:
                nota_viz = por_coer[f_viz]["nota"]
                # Interpolação linear
                t = abs(coer - centro_atual) / abs(c_viz - centro_atual)
                t = max(0, min(1, t))
                ancora = nota_atual * (1-t) + nota_viz * t
                faixa_usada = f"{faixa_atual}~{f_viz}"

    # ── Teto e piso vêm dos próprios dados desta prova ────────────────
    # Teto: nota máxima da MESMA prova nessa faixa de coerência (p75)
    # Piso: nota mínima (p25)
    if faixa_atual in por_coer:
        p25_local = por_coer[faixa_atual].get("p25")
        p75_local = por_coer[faixa_atual].get("p75")
        if p75_local is not None: teto_local = p75_local
        if p25_local is not None: piso_local = p25_local

    # Se faltam percentis locais, usa por_acertos[ac]
    por_ac_stats = prova["por_acertos"].get(str(ac), {})
    if teto_local is None:
        teto_local = por_ac_stats.get("mediana", g.get("nota_mediana"))
    if piso_local is None:
        piso_local = por_ac_stats.get("p10", g.get("nota_min"))

    # ── Decide se mistura âncora com nota_base ────────────────────────
    # Confia mais na âncora local quando há dados densos
    densidade_faixa = 0
    if faixa_atual in por_coer:
        densidade_faixa = por_coer[faixa_atual].get("n", 0)

    if densidade_faixa >= 50:
        peso_ancora = 0.9
    elif densidade_faixa >= 20:
        peso_ancora = 0.7
    elif suavizado_faixa or suavizado_externo:
        peso_ancora = 0.55     # menos confiança quando dados suavizados
    else:
        peso_ancora = 0.5

    nota_blend = peso_ancora * ancora + (1 - peso_ancora) * nota_base

    # ── Aplica teto/piso locais ───────────────────────────────────────
    nota_final = max(piso_local, min(teto_local, nota_blend))

    razao = []
    if coer < 0.40: razao.append(f"coer={coer:.2f}")
    if em is not None and em < 0.0 and ac >= 10: razao.append(f"em={em:.2f}_ac={ac}")
    if mba is not None and mbe is not None and mba > mbe and coer < 0.60:
        razao.append(f"mba>mbe")

    info = {
        "razao":           "|".join(razao) if razao else "limiar",
        "fonte":           fonte,
        "faixa_usada":     faixa_usada,
        "densidade_faixa": densidade_faixa,
        "ancora":          round(ancora, 1) if ancora is not None else None,
        "nota_base":       round(nota_base, 1),
        "nota_blend":      round(nota_blend, 1),
        "peso_ancora":     round(peso_ancora, 2),
        "suavizado":       suavizado_externo or suavizado_faixa,
        "baixa_densidade_inco": info_ac.get("baixa_densidade", False),
        "ajustes":         [],
    }
    return nota_final, \
           round(teto_local, 1) if teto_local is not None else None, \
           round(piso_local, 1) if piso_local is not None else None, \
           info


def estimar_nota_compacto(vetor, b_por_posicao, area, ano, tipo, cor,
                          mascara=None, modelo=None):
    if modelo is None: modelo = _modelo()

    chave = f"{area}_{ano}_{tipo}_{cor.upper()}"
    prova = modelo.get("provas", {}).get(chave)
    if not prova:
        return {
            "erro":     f"prova não encontrada no modelo compacto: {chave}",
            "metodo":   "indisponivel",
        }

    feats = calcular_features(vetor, b_por_posicao, mascara)
    ac      = feats["acertos"]
    coer    = feats["coerencia"]
    em      = feats["easiest_miss"]
    mba     = feats["media_b_acertos"]
    mbe     = feats["media_b_erros"]
    g       = prova["global"]

    # ─── 1. Caso especial: acc = 0 → mínimo histórico
    if ac == 0:
        return _resultado(prova, feats, g["nota_min"], "minimo_historico", "minimo_historico",
                          confianca="alta", area=area, ano=ano, tipo=tipo, cor=cor)

    # ─── 1b. Caso especial: acertou TUDO → âncora conservadora ────────
    # Quando ac == total_aplicaveis, a mediana dos perfeitos infla a nota
    # versus o kNN do motor histórico (que mistura vizinhos com ac-1 / ac-2
    # a distância Hamming ≤ 3). Para imitar o comportamento conservador
    # do kNN sem reconstruir vizinhança:
    #   • começa pelo p25 dos perfeitos (não a mediana)
    #   • puxa adicionalmente para baixo usando a média de quem acertou ac-1
    #     (vizinhos a Hamming=1 que pesam no kNN real)
    total_aplic = feats["total_aplicaveis"]
    if ac >= total_aplic and ac > 0:
        por_ac_max = prova["por_acertos"].get(str(ac), {})
        if por_ac_max:
            n_perfeitos = por_ac_max.get("n", 0)

            # Âncora 1: p25 dos perfeitos (conservadora vs mediana)
            ancora_perfeitos = por_ac_max.get("p25") or por_ac_max.get("mediana")

            # Âncora 2: média de quem acertou ac-1 (vizinhos Hamming=1)
            por_ac_prev = prova["por_acertos"].get(str(ac - 1), {})
            ancora_vizinhos = por_ac_prev.get("media") if por_ac_prev else None

            # Combinação ponderada — imita o kNN do motor histórico
            if ancora_vizinhos is not None and n_perfeitos >= 30:
                # 75% p25 perfeitos + 25% média ac-1
                # (o kNN mistura ~maioria perfeitos com poucos vizinhos a Hamming=1)
                nota_base = 0.75 * ancora_perfeitos + 0.25 * ancora_vizinhos
                anc_label = f"acc={ac}_blend(p25_perf+media_ac-1, n={n_perfeitos})"
            elif n_perfeitos >= 30:
                # Sem dados de ac-1: usa só p25 dos perfeitos
                nota_base = ancora_perfeitos
                anc_label = f"acc={ac}_p25_perfeitos(n={n_perfeitos})"
            else:
                # Amostra muito pequena: usa p10 para ser conservador
                nota_base = por_ac_max.get("p10", ancora_perfeitos)
                anc_label = f"acc={ac}_p10_perfeitos(n={n_perfeitos})"

            nota_base = max(g["nota_min"], min(g["nota_max"], nota_base))
            return _resultado(
                prova, feats, nota_base,
                metodo="compacto_perfeito",
                ancoragem=anc_label,
                confianca="alta",
                n_similares=n_perfeitos,
                intervalo_min=por_ac_max.get("p10"),
                intervalo_max=por_ac_max.get("p75"),
                area=area, ano=ano, tipo=tipo, cor=cor,
            )

    # ─── 2. Lookup hierárquico (v3) ───────────────────────────────────
    # Ordem de preferência:
    #   2a. cubo 3D direto                       (acertos, coer_faixa, mba_bin)
    #   2b. cubo 3D interpolado em mba_bin       (mistura bins adjacentes)
    #   2c. cubo 3D interpolado em coer + mba    (vizinhança 2x2 no plano)
    #   2d. interpolação 2D entre faixas de coer adjacentes
    #   2e. cubo 2D direto
    #   2f. fallback ponderado: ac com vizinhos ac-1, ac+1 misturados
    #   2g. global da prova
    faixa = _faixa_coerencia(coer)
    mba_bin = _bin_b(mba)
    cubo_3d = prova.get("cubo_3d", {}).get(str(ac), {})
    cubo_2d = prova.get("por_acertos_coerencia", {}).get(str(ac), {})

    nota_base = None
    metodo, anc = None, None
    intervalo_min = intervalo_max = None
    n_amostra = 0

    # Helper: pega faixas de mba adjacentes c/ pesos
    def _bins_b_adjacentes(valor):
        if valor is None: return []
        bins = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        # encontra o intervalo
        for i in range(len(bins)-1):
            if bins[i] <= valor < bins[i+1]:
                centro = (bins[i] + bins[i+1]) / 2
                lab = f"{bins[i]}-{bins[i+1]}"
                # se valor < centro → interpola com bin anterior
                # se valor >= centro → interpola com bin seguinte
                if valor < centro and i > 0:
                    lab_prev = f"{bins[i-1]}-{bins[i]}"
                    peso = (centro - valor) / 0.5
                    return [(lab, 1 - peso), (lab_prev, peso)]
                elif valor >= centro and i < len(bins)-2:
                    lab_next = f"{bins[i+1]}-{bins[i+2]}"
                    peso = (valor - centro) / 0.5
                    return [(lab, 1 - peso), (lab_next, peso)]
                return [(lab, 1.0)]
        return []

    # 2a. Lookup 3D direto (threshold relaxado: n>=5)
    if cubo_3d and mba_bin:
        info_3d = cubo_3d.get(faixa, {}).get(mba_bin)
        if info_3d and info_3d["n"] >= 5:
            nota_base = info_3d["media"]
            metodo, anc = "compacto_3d", f"acc={ac}+coer={faixa}+mba={mba_bin}"
            intervalo_min = info_3d["p25"]; intervalo_max = info_3d["p75"]
            n_amostra = info_3d["n"]

    # 2b. Interpolação 3D em mba (mistura bins adjacentes de mba na mesma faixa de coer)
    if nota_base is None and cubo_3d and mba is not None:
        mba_pares = _bins_b_adjacentes(mba)
        partes = []
        n_total = 0
        for lab_b, peso_b in mba_pares:
            info = cubo_3d.get(faixa, {}).get(lab_b)
            if info and info["n"] >= 5:
                partes.append((info["media"], peso_b, info["n"]))
                n_total += info["n"]
        if partes:
            sp = sum(p for _, p, _ in partes)
            nota_base = sum(m*p for m, p, _ in partes) / sp
            metodo, anc = "compacto_3d_interp_b", f"acc={ac}+coer={faixa}+mba~{mba:.2f}"
            n_amostra = n_total

    # 2c. Interpolação 3D completa (vizinhança 2x2 em coer × mba)
    if nota_base is None and cubo_3d and mba is not None:
        coer_pares = _faixas_coer_adjacentes(coer)
        mba_pares  = _bins_b_adjacentes(mba)
        partes = []
        n_total = 0
        for lab_c, peso_c in coer_pares:
            for lab_b, peso_b in mba_pares:
                info = cubo_3d.get(lab_c, {}).get(lab_b)
                if info and info["n"] >= 5:
                    peso = peso_c * peso_b
                    partes.append((info["media"], peso, info["n"]))
                    n_total += info["n"]
        if partes:
            sp = sum(p for _, p, _ in partes)
            nota_base = sum(m*p for m, p, _ in partes) / sp
            metodo, anc = "compacto_3d_interp_2d", f"acc={ac}+coer~{coer:.2f}+mba~{mba:.2f}"
            n_amostra = n_total

    # 2d. Interpolação 2D entre faixas de coer adjacentes (sem mba)
    if nota_base is None and cubo_2d:
        faixas_pesos = _faixas_coer_adjacentes(coer)
        partes = []
        n_total = 0
        for lab, peso in faixas_pesos:
            info_2d = cubo_2d.get(lab)
            if info_2d and info_2d["n"] >= 10:   # threshold reduzido
                partes.append((info_2d["media"], peso, info_2d["n"],
                                info_2d.get("p25"), info_2d.get("p75")))
                n_total += info_2d["n"]
        if partes:
            sp = sum(p for _, p, _, _, _ in partes)
            nota_base = sum(m*p for m, p, _, _, _ in partes) / sp
            metodo, anc = "compacto_2d_interp", f"acc={ac}+coer~{coer:.2f}"
            p25s = [p25 for _, _, _, p25, _ in partes if p25 is not None]
            p75s = [p75 for _, _, _, _, p75 in partes if p75 is not None]
            intervalo_min = min(p25s) if p25s else None
            intervalo_max = max(p75s) if p75s else None
            n_amostra = n_total

    # 2e. Cubo 2D direto (sem interpolação)
    if nota_base is None and cubo_2d:
        info_2d = cubo_2d.get(faixa)
        if info_2d and info_2d["n"] >= 10:
            nota_base = info_2d["media"]
            metodo, anc = "compacto_2d", f"acc={ac}+coer={faixa}"
            intervalo_min = info_2d.get("p25"); intervalo_max = info_2d.get("p75")
            n_amostra = info_2d["n"]

    # 2f. Fallback: vizinhança de acertos (ac-1, ac, ac+1, ac±2)
    #     Mistura ponderada DENTRO DA MESMA PROVA. Restaurado como principal
    #     após teste mostrar que compacto_vizinhos_struct piorou Δméd em todos
    #     os tipos (PPL +50, reapl +25, regular +35) — protótipos não capturam
    #     bem o comportamento TRI em provas pequenas/extremos altos incoerentes.
    if nota_base is None:
        partes = []
        n_total = 0
        for offset, peso_acc in [(-2, 0.2), (-1, 0.5), (0, 1.0), (+1, 0.5), (+2, 0.2)]:
            ac_viz = ac + offset
            if ac_viz < 0: continue
            por_ac_viz = prova["por_acertos"].get(str(ac_viz))
            if not por_ac_viz: continue
            cubo_2d_viz = prova.get("por_acertos_coerencia", {}).get(str(ac_viz), {})
            info_viz = cubo_2d_viz.get(faixa)
            if info_viz and info_viz["n"] >= 10:
                partes.append((info_viz["media"], peso_acc, info_viz["n"]))
                n_total += info_viz["n"]
            else:
                base_media = por_ac_viz.get("smoothed_media", por_ac_viz["media"])
                ajuste = offset * -15
                partes.append((base_media + ajuste, peso_acc * 0.5,
                              por_ac_viz.get("n", 0)))
                n_total += por_ac_viz.get("n", 0)
        if partes:
            sp = sum(p for _, p, _ in partes)
            nota_base = sum(m*p for m, p, _ in partes) / sp
            metodo = "compacto_vizinhos_acc"
            anc = f"acc~{ac}+coer={faixa}"
            por_ac = prova["por_acertos"].get(str(ac), {})
            intervalo_min = por_ac.get("p25"); intervalo_max = por_ac.get("p75")
            n_amostra = n_total

    # 2g. Fallback extremo: global da prova
    if nota_base is None:
        nota_base = g["nota_media"]
        metodo, anc = "compacto_global", "global"
        intervalo_min = g["p25"]; intervalo_max = g["p75"]

    # ─── 3. Refinamento por bins de b
    nota_base, info_bins = _refinar_por_bins_b(prova, ac, mba, feats["media_b_erros"], nota_base)

    # ─── 4. Penalidades por incoerência (faixas v2)
    pen = prova["penalidades_calibradas"]
    # ─── 4. Motor de incoerência controlada ───────────────────────────
    # Acionado quando o padrão de respostas é estruturalmente incoerente.
    # Substitui o velho esquema de "penalidade fixa + teto" que somava
    # efeitos e derrubava nota em extremos altos (ex: 43/45 com coer=0.18
    # ia para 410 quando a regressão histórica indicava ~640).
    detalhes_pen = []
    nota = nota_base
    teto = piso = None
    aciona_incoerencia = (
        coer < 0.40
        or (em is not None and em < 0.0 and ac >= 10)
        or (mba is not None and mbe is not None and mba > mbe and coer < 0.60)
    )

    if aciona_incoerencia:
        nota_pre = nota_base
        nota, teto, piso, info = _motor_incoerencia_controlada(
            prova, feats, nota_base, n_amostra
        )
        metodo = "compacto_incoerencia_controlada"
        anc = (f"local_inco_model|nota_pre={nota_pre:.0f}"
               f"|faixa={info.get('faixa_usada','?')}"
               f"|fonte={info.get('fonte','?')}"
               f"|densidade={info.get('densidade_faixa',0)}"
               f"|suav={info.get('suavizado',False)}"
               f"|teto={teto}|piso={piso}|razao={info['razao']}")
        detalhes_pen = info.get("ajustes", [])
    elif coer < 0.60:
        # Faixa intermediária 0.40-0.60: penalidade leve, sem teto duro
        gap = (0.60 - coer) / 0.20
        pen = prova["penalidades_calibradas"]
        pen_060 = pen.get("coer_lt_0.60", 40.0)
        penalidade = gap * pen_060 * 0.5
        nota = nota_base - penalidade
        detalhes_pen.append(f"coer<0.60_leve({penalidade:.0f})")

    # ─── 5. Clamp ao intervalo histórico global
    nota = max(g["nota_min"], min(g["nota_max"], nota))

    # ─── 6. Confiança baseada em amostra + coerência
    if n_amostra >= 200 and coer >= 0.5:   confianca = "alta"
    elif n_amostra >= 50  and coer >= 0.3: confianca = "media"
    elif coer < 0.3:                       confianca = "muito_baixa"
    else:                                  confianca = "baixa"

    if detalhes_pen and anc.startswith("acc="):
        anc = anc + " | pen:" + "+".join(detalhes_pen)
    if info_bins not in ("sem_ref","sem_bins"):
        anc = anc + " | " + info_bins

    return _resultado(prova, feats, nota, metodo, anc,
                      confianca=confianca,
                      n_similares=n_amostra,
                      intervalo_min=intervalo_min, intervalo_max=intervalo_max,
                      area=area, ano=ano, tipo=tipo, cor=cor)


def _resultado(prova, feats, nota, metodo, ancoragem, confianca,
               n_similares=0, intervalo_min=None, intervalo_max=None,
               area=None, ano=None, tipo=None, cor=None):
    g = prova["global"]
    ac = feats["acertos"]
    media_ac = prova["por_acertos"].get(str(ac), {}).get("media")
    # Densidade local do bucket de acertos atual
    n_local_ac = prova["por_acertos"].get(str(ac), {}).get("n", 0)
    return {
        "nota_estimada":        round(nota, 1),
        "nota_mediana":         round(nota, 1),
        "intervalo_min":        intervalo_min if intervalo_min is not None else g["p10"],
        "intervalo_max":        intervalo_max if intervalo_max is not None else g["p90"],
        "desvio_padrao":        None,
        "confianca":            confianca,
        "candidatos_similares": n_similares,
        "distancia_max_usada":  None,
        "distancia_media":      None,
        "metodo":               metodo,
        "metodo_ancoragem":     ancoragem,
        "acertos":              feats["acertos"],
        "erros":                feats["erros"],
        "total_aplicaveis":     feats["total_aplicaveis"],
        "coerencia":            feats["coerencia"],
        "inversoes":            feats["inversoes"],
        "pares_possiveis":      feats["pares_possiveis"],
        "media_b_acertos":      feats["media_b_acertos"],
        "media_b_erros":        feats["media_b_erros"],
        "hardest_hit":          feats["hardest_hit"],
        "easiest_miss":         feats["easiest_miss"],
        "nota_minima_historica":  g["nota_min"],
        "nota_maxima_historica":  g["nota_max"],
        "nota_media_por_acertos": media_ac,
        # Metadados de localidade
        "modelo_localidade":      prova.get("modelo_localidade"),
        "densidade_amostral":     prova.get("densidade_amostral"),
        "confiabilidade_local":   prova.get("confiabilidade_local"),
        "densidade_acertos_atual": n_local_ac,
        "area": area, "ano": ano, "tipo": tipo, "cor": cor,
    }


# ───────────────────────────────────────────────────────────────────────
#  CLI rápido
# ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, sys
    p = argparse.ArgumentParser()
    p.add_argument("--modelo", default="modelo_compacto_tri.json")
    p.add_argument("--area", required=True)
    p.add_argument("--ano", type=int, required=True)
    p.add_argument("--tipo", default="regular")
    p.add_argument("--cor", required=True)
    p.add_argument("--vetor", required=True)
    p.add_argument("--mascara", default=None)
    args = p.parse_args()

    modelo = carregar_modelo(args.modelo)
    # Para teste rápido sem b reais, gera placeholder zerado (não recomendado)
    print("[AVISO] CLI exige b_por_posicao real para resultado correto. "
          "Use como módulo importável.", file=sys.stderr)
