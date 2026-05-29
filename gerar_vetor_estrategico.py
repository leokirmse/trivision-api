"""
gerar_vetor_estrategico.py
──────────────────────────
Gera vetores binários simulando padrões estratégicos de resolução,
preservando a ordem oficial da prova.

Modos:
  faceis      → marca as n questões com MENOR b
  dificeis    → marca as n questões com MAIOR b
  aleatorio   → marca n questões aleatórias
  intervalo-b → marca questões dentro de uma faixa de b (--b-min / --b-max)
  coerente    → padrão coerente: prioriza b baixos, garante alta coerência TRI
  incoerente  → padrão incoerente: prioriza b altos com algumas inversões

Saída:
  - vetor binário (45 ou 50 chars) na ordem oficial da prova
  - posições marcadas (1..N)
  - parâmetro b de cada questão marcada
  - média de b dos acertos
  - comando pronto para o estimador_nota.py

LC: vetor de 50 com 5 inglês + 5 espanhol + 40 comuns.
Por padrão considera Inglês como língua escolhida (--lingua ing|esp).
A máscara é gerada automaticamente.

USO:
    python gerar_vetor_estrategico.py --area MT --ano 2023 --tipo regular --cor AMARELA --modo faceis --n 21
"""

import argparse, csv, json, os, random, sys


# ───────────────────────────────────────────────────────────────────────
#  CARREGAMENTO DOS ITENS DA PROVA
# ───────────────────────────────────────────────────────────────────────

def norm_tipo(t):
    t = t.lower()
    if t in ("reaplic","reaplicacao","reaplicação"): return "reaplicacao"
    if t == "ppl": return "ppl"
    return "regular"


def carregar_itens_prova(area, ano, tipo, cor, lingua="ing"):
    """
    Retorna lista [(pos_oficial, b, tp_ling, co_item)] na ordem oficial (1..N).
    Para LC, monta a estrutura de 50 itens (5 ing + 5 esp + 40 comuns).
    A máscara é construída em paralelo.
    """
    if not os.path.exists("provas_todas.json"):
        return None, None, "provas_todas.json não encontrado"
    with open("provas_todas.json", encoding="utf-8") as f:
        provas = json.load(f)

    # Encontra co_prova
    cores_ano = provas.get(str(ano), {}).get(area, {})
    co_prova = None
    for c, tipos in cores_ano.items():
        if c.upper() != cor.upper(): continue
        for t, info in tipos.items():
            if norm_tipo(t) == norm_tipo(tipo):
                co_prova = str(info.get("co_prova","")).strip()
                break
    if not co_prova:
        return None, None, f"co_prova não encontrado para {area} {ano} {tipo} {cor}"

    path = os.path.join("dados_inep", f"ITENS_PROVA_{ano}.csv")
    if not os.path.exists(path):
        return None, None, f"{path} não encontrado"

    with open(path, encoding="latin-1") as f:
        sep = ";" if ";" in f.readline() else ","

    itens_brutos = []  # (CO_POSICAO, b, tp_ling, co_item)
    with open(path, encoding="latin-1", newline="") as f:
        for row in csv.DictReader(f, delimiter=sep):
            if str(row.get("CO_PROVA","")).strip() != co_prova: continue
            try: pos = int(row.get("CO_POSICAO",0) or 0)
            except: pos = 0
            try:
                b = float(str(row.get("NU_PARAM_B","")).replace(",","."))
                if not (-5 <= b <= 6): b = None
            except: b = None
            tp_ling = row.get("TP_LINGUA","").strip()
            co_item = str(row.get("CO_ITEM","")).strip()
            itens_brutos.append((pos, b, tp_ling, co_item))

    if not itens_brutos:
        return None, None, "nenhum item encontrado para esse co_prova"

    # Monta lista oficial + máscara
    if area == "LC":
        ingles   = sorted([t for t in itens_brutos if t[2] == "0"], key=lambda x: x[0])[:5]
        espanhol = sorted([t for t in itens_brutos if t[2] == "1"], key=lambda x: x[0])[:5]
        comuns   = sorted([t for t in itens_brutos if t[2] not in ("0","1")], key=lambda x: x[0])[:40]

        itens_oficial = []
        mascara = []
        # Posições 1-5: Inglês — aplicáveis se lingua=='ing'
        for _, b, _, ci in ingles:
            itens_oficial.append((b, "0", ci))
            mascara.append("1" if lingua == "ing" else "0")
        # Posições 6-10: Espanhol — aplicáveis se lingua=='esp'
        for _, b, _, ci in espanhol:
            itens_oficial.append((b, "1", ci))
            mascara.append("1" if lingua == "esp" else "0")
        # Posições 11-50: comuns
        for _, b, _, ci in comuns:
            itens_oficial.append((b, "", ci))
            mascara.append("1")
        return itens_oficial, "".join(mascara), None
    else:
        itens_brutos.sort(key=lambda x: x[0])
        itens_oficial = [(b, "", ci) for _, b, _, ci in itens_brutos]
        mascara = "1" * len(itens_oficial)
        return itens_oficial, mascara, None


# ───────────────────────────────────────────────────────────────────────
#  GERAÇÃO DE VETOR POR MODO
# ───────────────────────────────────────────────────────────────────────

def _indices_aplicaveis_com_b(itens, mascara):
    """Retorna lista de (idx, b) onde máscara=1 e b é válido."""
    return [(i, b) for i, ((b, _, _), m) in enumerate(zip(itens, mascara))
            if m == "1" and b is not None]


def gerar_faceis(itens, mascara, n):
    aplicaveis = _indices_aplicaveis_com_b(itens, mascara)
    ordenados = sorted(aplicaveis, key=lambda x: x[1])  # menor b primeiro
    return {idx for idx, _ in ordenados[:n]}


def gerar_dificeis(itens, mascara, n):
    aplicaveis = _indices_aplicaveis_com_b(itens, mascara)
    ordenados = sorted(aplicaveis, key=lambda x: -x[1])  # maior b primeiro
    return {idx for idx, _ in ordenados[:n]}


def gerar_aleatorio(itens, mascara, n, seed=None):
    aplicaveis = _indices_aplicaveis_com_b(itens, mascara)
    rng = random.Random(seed) if seed is not None else random
    escolhidos = rng.sample(aplicaveis, min(n, len(aplicaveis)))
    return {idx for idx, _ in escolhidos}


def gerar_intervalo_b(itens, mascara, b_min, b_max):
    aplicaveis = _indices_aplicaveis_com_b(itens, mascara)
    return {idx for idx, b in aplicaveis if b_min <= b <= b_max}


def gerar_coerente(itens, mascara, n):
    """
    Padrão coerente: prioriza b baixos. Equivalente a 'faceis' puro,
    sem ruído — coerência TRI máxima.
    """
    return gerar_faceis(itens, mascara, n)


def gerar_incoerente(itens, mascara, n, seed=None):
    """
    Padrão incoerente: marca preferencialmente os b mais ALTOS,
    com algum ruído de questões médias para evidenciar inversões.
    Resultado: coerência TRI baixa.
    """
    aplicaveis = _indices_aplicaveis_com_b(itens, mascara)
    aplicaveis_ord = sorted(aplicaveis, key=lambda x: -x[1])  # difícil primeiro
    rng = random.Random(seed) if seed is not None else random

    # 80% dos n vêm das mais difíceis; 20% sorteados das medianas (b ∈ [0.5, 1.5])
    n_dif  = int(round(n * 0.8))
    n_med  = n - n_dif
    escolhidos = [idx for idx, _ in aplicaveis_ord[:n_dif]]

    if n_med > 0:
        medianas = [idx for idx, b in aplicaveis if 0.5 <= b <= 1.5 and idx not in escolhidos]
        if medianas:
            escolhidos.extend(rng.sample(medianas, min(n_med, len(medianas))))

    return set(escolhidos)


# ───────────────────────────────────────────────────────────────────────
#  MONTAGEM DO VETOR
# ───────────────────────────────────────────────────────────────────────

def montar_vetor(itens, marcados_idx):
    """Constrói vetor binário com 1 nos índices marcados."""
    return "".join("1" if i in marcados_idx else "0" for i in range(len(itens)))


def resumo(itens, mascara, marcados_idx):
    """Estatísticas do vetor gerado."""
    bs_marcados = [itens[i][0] for i in marcados_idx if itens[i][0] is not None]
    media_b = round(sum(bs_marcados)/len(bs_marcados), 4) if bs_marcados else 0.0
    posicoes_1based = sorted(i+1 for i in marcados_idx)
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
        "posicoes":       posicoes_1based,
        "questoes":       questoes,
        "media_b_acertos": media_b,
    }


# ───────────────────────────────────────────────────────────────────────
#  CLI
# ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Gera vetor binário estratégico")
    p.add_argument("--area",  required=True, choices=["MT","CN","CH","LC"])
    p.add_argument("--ano",   required=True, type=int)
    p.add_argument("--tipo",  required=True, choices=["regular","reaplicacao","ppl"])
    p.add_argument("--cor",   required=True)
    p.add_argument("--modo",  required=True,
                   choices=["faceis","dificeis","aleatorio","intervalo-b","coerente","incoerente"])
    p.add_argument("--n", type=int, default=None,
                   help="Nº de questões a marcar (obrigatório exceto em intervalo-b)")
    p.add_argument("--b-min", type=float, help="Limite inferior de b (intervalo-b)")
    p.add_argument("--b-max", type=float, help="Limite superior de b (intervalo-b)")
    p.add_argument("--lingua", choices=["ing","esp"], default="ing",
                   help="Língua estrangeira escolhida (LC)")
    p.add_argument("--seed",  type=int, help="Seed para aleatório")
    p.add_argument("--json",  action="store_true", help="Saída JSON")
    args = p.parse_args()

    if args.modo == "intervalo-b" and (args.b_min is None or args.b_max is None):
        print("[ERRO] --b-min e --b-max são obrigatórios em intervalo-b"); sys.exit(1)
    if args.modo != "intervalo-b" and args.n is None:
        print(f"[ERRO] --n é obrigatório no modo {args.modo}"); sys.exit(1)

    itens, mascara, erro = carregar_itens_prova(args.area, args.ano, args.tipo, args.cor, args.lingua)
    if erro:
        print(f"[ERRO] {erro}"); sys.exit(1)

    if args.modo == "faceis":
        marcados = gerar_faceis(itens, mascara, args.n)
    elif args.modo == "dificeis":
        marcados = gerar_dificeis(itens, mascara, args.n)
    elif args.modo == "aleatorio":
        marcados = gerar_aleatorio(itens, mascara, args.n, args.seed)
    elif args.modo == "intervalo-b":
        marcados = gerar_intervalo_b(itens, mascara, args.b_min, args.b_max)
    elif args.modo == "coerente":
        marcados = gerar_coerente(itens, mascara, args.n)
    elif args.modo == "incoerente":
        marcados = gerar_incoerente(itens, mascara, args.n, args.seed)
    else:
        print(f"[ERRO] modo desconhecido"); sys.exit(1)

    vetor = montar_vetor(itens, marcados)
    info = resumo(itens, mascara, marcados)

    cmd_estimador = (
        f"python estimador_nota.py --area {args.area} --ano {args.ano} "
        f"--tipo {args.tipo} --cor {args.cor} --vetor {vetor}"
    )
    if args.area == "LC":
        cmd_estimador += f" --mascara {mascara}"

    saida = {
        "area": args.area, "ano": args.ano, "tipo": args.tipo, "cor": args.cor,
        "modo": args.modo, "tamanho_vetor": len(vetor),
        "vetor":            vetor,
        "mascara":          mascara,
        "n_marcadas":       info["n_marcadas"],
        "posicoes_marcadas": info["posicoes"],
        "questoes_marcadas": info["questoes"],
        "media_b_acertos":   info["media_b_acertos"],
        "comando_estimador": cmd_estimador,
    }

    if args.json:
        print(json.dumps(saida, ensure_ascii=False, indent=2)); return

    print(f"\n{'='*68}")
    print(f"  Vetor estratégico — {args.area} {args.ano} {args.tipo} {args.cor} | modo: {args.modo}")
    print(f"{'='*68}\n")
    print(f"  Tamanho do vetor:  {len(vetor)}")
    print(f"  Questões marcadas: {info['n_marcadas']}")
    print(f"  Média b acertos:   {info['media_b_acertos']:.3f}")
    if args.area == "LC":
        print(f"  Língua escolhida:  {args.lingua}")
    print(f"\n  Vetor:")
    print(f"    {vetor}")
    if args.area == "LC":
        print(f"\n  Máscara:")
        print(f"    {mascara}")

    print(f"\n  Posições marcadas (1-based):")
    print(f"    {info['posicoes']}")

    print(f"\n  Detalhe das questões marcadas:")
    print(f"    {'Pos':>4}  {'b':>6}  {'TP_LING':<8}  CO_ITEM")
    for q in info["questoes"]:
        b_str = f"{q['b']:.3f}" if q['b'] is not None else " —  "
        print(f"    {q['pos']:>4}  {b_str:>6}  {q['tp_ling']:<8}  {q['co_item']}")

    print(f"\n  Comando para o estimador:")
    print(f"    {cmd_estimador}")
    print()


if __name__ == "__main__":
    main()
