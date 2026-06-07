"""
gerar_vetor_estrategico_v2.py
------------------------------
v2: usa mapeamento_canonico_v6.json para buscar itens (b, tp_ling, co_item, gab),
em vez de re-parsear ITENS_PROVA_<ano>.csv.

Mantém modos identicos: faceis, dificeis, aleatorio, intervalo-b, coerente, incoerente.

IMPORTANTE: o vetor produzido fica na ordem "da cor" daquele candidato,
para enviar para a API exatamente como o microdado vem (a API faz a
canonicalizacao internamente).
"""

import argparse, json, os, random, sys


_MAPEAMENTO_PATH = "mapeamento_canonico_v6.json"
_MAPEAMENTO_CACHE = None


def configurar(mapeamento_path="mapeamento_canonico_v6.json"):
    global _MAPEAMENTO_PATH, _MAPEAMENTO_CACHE
    _MAPEAMENTO_PATH = mapeamento_path
    _MAPEAMENTO_CACHE = None


def _carregar_mapeamento():
    global _MAPEAMENTO_CACHE
    if _MAPEAMENTO_CACHE is None:
        with open(_MAPEAMENTO_PATH, encoding="utf-8") as f:
            _MAPEAMENTO_CACHE = json.load(f)
    return _MAPEAMENTO_CACHE


def norm_tipo(t):
    t = t.lower()
    if t in ("reaplic", "reaplicacao", "reaplicação"): return "regular"
    if t == "ppl": return "ppl"
    return "regular"


def _resolver_chave(area, ano, tipo, lingua="ing"):
    tipo_n = norm_tipo(tipo)
    if area == "LC":
        ling = (lingua or "ing").lower()
        if ling not in ("ing", "esp"): ling = "ing"
        return f"{area}_{ano}_{tipo_n}_{ling}"
    return f"{area}_{ano}_{tipo_n}"


def carregar_itens_prova(area, ano, tipo, cor, lingua="ing"):
    """
    Retorna (itens_na_ordem_da_cor, mascara, erro).

    itens: lista [(b, tp_ling, co_item)] na ORDEM ORIGINAL do TX_RESPOSTAS
           da cor escolhida (45 ou 50 itens).
    mascara: string '0'/'1' do mesmo tamanho. Em LC ant 2016-2021 (50 itens),
             os 5 itens da outra lingua tem mascara=0.
    """
    try:
        mapeamento = _carregar_mapeamento()
    except FileNotFoundError as e:
        return None, None, str(e)

    chave = _resolver_chave(area, ano, tipo, lingua)
    if chave not in mapeamento:
        return None, None, f"chave {chave} sem mapeamento"

    mapa = mapeamento[chave]
    cores = mapa.get("cores", {})
    cor_upper = (cor or "").upper()
    info_cor = cores.get(cor_upper)
    if info_cor is None:
        # Procura case-insensitive
        for c_k, c_v in cores.items():
            if c_k.upper() == cor_upper:
                info_cor = c_v
                break
        if info_cor is None:
            return None, None, f"cor {cor} nao mapeada em {chave}. Disponiveis: {list(cores.keys())}"

    itens_canon = mapa["itens_canonicos"]
    perm = info_cor.get("perm_resp") or info_cor.get("perm")
    len_vetor = info_cor.get("len_vetor_microdado") or info_cor.get("n_itens_originais")

    # Reconstroi a lista de itens NA ORDEM DA COR (inversa da perm)
    # perm[i_canon] = posicao no vetor original da cor onde esta o item canonico i_canon
    # Logo: itens_na_ordem_cor[pos_vetor] = item_canonico que la esta
    itens_na_ordem_cor = [None] * len_vetor
    for i_canon, p in enumerate(perm):
        if p is None: continue
        if 0 <= p < len_vetor:
            itens_na_ordem_cor[p] = itens_canon[i_canon]

    # Constroi (b, tp_ling, co_item) por posicao
    itens_out = []
    mascara_out = []
    for pos, ic in enumerate(itens_na_ordem_cor):
        if ic is None:
            # Posicao da OUTRA lingua em LC anos antigos (resp=50 mas mapa filtra)
            itens_out.append((None, "0_ou_1", "?"))
            mascara_out.append("0")
        else:
            try: b = float(ic["b"]) if ic["b"] else None
            except (ValueError, TypeError): b = None
            itens_out.append((b, ic.get("tp_ling", ""), ic.get("co_item", "")))
            mascara_out.append("1")

    return itens_out, "".join(mascara_out), None


# ── Geradores de vetor (identicos ao antigo) ────────────────────────

def _indices_aplicaveis_com_b(itens, mascara):
    return [(i, b) for i, ((b, _, _), m) in enumerate(zip(itens, mascara))
            if m == "1" and b is not None]


def gerar_faceis(itens, mascara, n):
    aplic = _indices_aplicaveis_com_b(itens, mascara)
    ordenados = sorted(aplic, key=lambda x: x[1])
    return {idx for idx, _ in ordenados[:n]}


def gerar_dificeis(itens, mascara, n):
    aplic = _indices_aplicaveis_com_b(itens, mascara)
    ordenados = sorted(aplic, key=lambda x: -x[1])
    return {idx for idx, _ in ordenados[:n]}


def gerar_aleatorio(itens, mascara, n, seed=None):
    aplic = _indices_aplicaveis_com_b(itens, mascara)
    rng = random.Random(seed) if seed is not None else random
    escolhidos = rng.sample(aplic, min(n, len(aplic)))
    return {idx for idx, _ in escolhidos}


def gerar_intervalo_b(itens, mascara, b_min, b_max):
    aplic = _indices_aplicaveis_com_b(itens, mascara)
    return {idx for idx, b in aplic if b_min <= b <= b_max}


def gerar_coerente(itens, mascara, n):
    return gerar_faceis(itens, mascara, n)


def gerar_incoerente(itens, mascara, n, seed=None):
    aplic = _indices_aplicaveis_com_b(itens, mascara)
    aplic_ord = sorted(aplic, key=lambda x: -x[1])
    rng = random.Random(seed) if seed is not None else random
    n_dif = int(round(n * 0.8)); n_med = n - n_dif
    escolhidos = [idx for idx, _ in aplic_ord[:n_dif]]
    if n_med > 0:
        medianas = [idx for idx, b in aplic if 0.5 <= b <= 1.5 and idx not in escolhidos]
        if medianas:
            escolhidos.extend(rng.sample(medianas, min(n_med, len(medianas))))
    return set(escolhidos)


def montar_vetor(itens, marcados_idx):
    return "".join("1" if i in marcados_idx else "0" for i in range(len(itens)))


def resumo(itens, mascara, marcados_idx):
    bs = [itens[i][0] for i in marcados_idx if itens[i][0] is not None]
    media_b = round(sum(bs)/len(bs), 4) if bs else 0.0
    pos_1based = sorted(i+1 for i in marcados_idx)
    questoes = []
    for i in sorted(marcados_idx):
        b, tp_ling, ci = itens[i]
        questoes.append({
            "pos":     i+1,
            "b":       round(b, 4) if b is not None else None,
            "tp_ling": tp_ling or "comum",
            "co_item": ci,
        })
    return {
        "n_marcadas":     len(marcados_idx),
        "posicoes":       pos_1based,
        "questoes":       questoes,
        "media_b_acertos": media_b,
    }


# ── CLI (igual antigo, agora rapido) ────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Gera vetor estrategico v2")
    p.add_argument("--area",  required=True, choices=["MT","CN","CH","LC"])
    p.add_argument("--ano",   required=True, type=int)
    p.add_argument("--tipo",  required=True, choices=["regular","reaplicacao","ppl"])
    p.add_argument("--cor",   required=True)
    p.add_argument("--modo",  required=True,
                   choices=["faceis","dificeis","aleatorio","intervalo-b","coerente","incoerente"])
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--b-min", type=float)
    p.add_argument("--b-max", type=float)
    p.add_argument("--lingua", choices=["ing","esp"], default="ing")
    p.add_argument("--seed",  type=int)
    p.add_argument("--json",  action="store_true")
    args = p.parse_args()

    if args.modo == "intervalo-b" and (args.b_min is None or args.b_max is None):
        print("[ERRO] --b-min e --b-max obrigatorios"); sys.exit(1)
    if args.modo != "intervalo-b" and args.n is None:
        print(f"[ERRO] --n obrigatorio em {args.modo}"); sys.exit(1)

    itens, mascara, erro = carregar_itens_prova(
        args.area, args.ano, args.tipo, args.cor, args.lingua)
    if erro:
        print(f"[ERRO] {erro}"); sys.exit(1)

    fn = {
        "faceis":      lambda: gerar_faceis(itens, mascara, args.n),
        "dificeis":    lambda: gerar_dificeis(itens, mascara, args.n),
        "aleatorio":   lambda: gerar_aleatorio(itens, mascara, args.n, args.seed),
        "intervalo-b": lambda: gerar_intervalo_b(itens, mascara, args.b_min, args.b_max),
        "coerente":    lambda: gerar_coerente(itens, mascara, args.n),
        "incoerente":  lambda: gerar_incoerente(itens, mascara, args.n, args.seed),
    }[args.modo]
    marcados = fn()

    vetor = montar_vetor(itens, marcados)
    info = resumo(itens, mascara, marcados)

    saida = {
        "area": args.area, "ano": args.ano, "tipo": args.tipo, "cor": args.cor,
        "modo": args.modo, "tamanho_vetor": len(vetor),
        "vetor":            vetor,
        "mascara":          mascara,
        "n_marcadas":       info["n_marcadas"],
        "posicoes_marcadas": info["posicoes"],
        "questoes_marcadas": info["questoes"],
        "media_b_acertos":   info["media_b_acertos"],
    }
    if args.json:
        print(json.dumps(saida, ensure_ascii=False, indent=2)); return

    print(f"\n  Vetor estrategico v2 | {args.area} {args.ano} {args.tipo} {args.cor} | {args.modo}")
    print(f"  Tamanho={len(vetor)}  Marcadas={info['n_marcadas']}  media_b={info['media_b_acertos']:.3f}")
    print(f"  Vetor:   {vetor}")
    if args.area == "LC":
        print(f"  Mascara: {mascara}")
    print(f"  Posicoes: {info['posicoes']}")


if __name__ == "__main__":
    main()
