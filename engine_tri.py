"""
engine_tri.py
─────────────
Camada unificada de motor TRI. Permite alternar entre:
  • motor "supervisionado"  (LightGBM/Ridge por prova — produção)
  • motor "compacto"        (cubo 3D + interpolação heurística — fallback)
  • motor "historico"       (kNN sobre features reais — referência)

Default: supervisionado.
Fallback automático: se .pkl não existir ou falhar → compacto → histórico.

Filosofia: cada (area, ano, tipo, cor) é um motor LOCAL.
Não há parametrização global. Não há interpolação entre provas.
"""

import os
import logging

# ── Imports lazy dos motores ────────────────────────────────────────
# Cada um pode falhar (dependência ausente, .pkl indisponível);
# tratamos individualmente para que o engine sempre consiga responder.

try:
    import estimador_supervisionado as e_super
    _SUPER_OK = True
except Exception as e:
    _SUPER_OK = False
    print(f"[ENGINE] estimador_supervisionado indisponível: {e}")

try:
    import estimador_compacto as e_compact
    _COMPACT_OK = True
    _MODELO_COMPACTO = None  # carregamento lazy
except Exception as e:
    _COMPACT_OK = False
    _MODELO_COMPACTO = None
    print(f"[ENGINE] estimador_compacto indisponível: {e}")

try:
    from estimador_nota import estimar_nota_tri as estimar_historico
    _HIST_OK = True
except Exception as e:
    _HIST_OK = False
    print(f"[ENGINE] estimador_historico indisponível: {e}")


# ── Cache global de modelos supervisionados ────────────────────────
# O estimador_supervisionado já tem seu próprio cache lazy (_CACHE),
# mas mantemos um registro local de quais provas têm modelo disponível
# para evitar tentativas repetidas de carregamento que falham.
_MODEL_AVAILABILITY_CACHE = {}   # chave → True/False
_DIR_MODELOS = "modelos_supervisionados"


def configurar(dir_modelos="modelos_supervisionados",
                path_modelo_compacto="modelo_compacto_tri.json"):
    """Configura caminhos. Chamado uma vez na inicialização da API."""
    global _DIR_MODELOS, _PATH_COMPACTO
    _DIR_MODELOS = dir_modelos
    _PATH_COMPACTO = path_modelo_compacto
    if _SUPER_OK:
        e_super.configurar_dir(dir_modelos)


_PATH_COMPACTO = "modelo_compacto_tri.json"


def _supervisionado_disponivel(area, ano, tipo, cor):
    chave = f"{area}_{ano}_{tipo}_{cor}"
    if chave in _MODEL_AVAILABILITY_CACHE:
        return _MODEL_AVAILABILITY_CACHE[chave]
    path = os.path.join(_DIR_MODELOS, f"{chave}.pkl")
    disp = os.path.exists(path)
    _MODEL_AVAILABILITY_CACHE[chave] = disp
    return disp


def _carregar_modelo_compacto():
    """Carrega o JSON do motor compacto uma única vez."""
    global _MODELO_COMPACTO
    if _MODELO_COMPACTO is None and _COMPACT_OK:
        try:
            _MODELO_COMPACTO = e_compact.carregar_modelo(_PATH_COMPACTO)
        except Exception as e:
            print(f"[ENGINE] falha carregando modelo compacto: {e}")
            _MODELO_COMPACTO = False  # sentinela para não tentar de novo
    return _MODELO_COMPACTO if _MODELO_COMPACTO not in (None, False) else None


# ── Padronização da resposta ───────────────────────────────────────

def _qualidade_de_rmse(rmse):
    if rmse is None:                  return None
    if rmse <= 15:                    return "alta"
    if rmse <= 30:                    return "media"
    return "baixa"


def _padronizar(resultado, motor, area, ano, tipo, cor, modelo_nome=None):
    """
    Garante que TODOS os motores retornem o mesmo formato mínimo
    sem remover campos extras específicos de cada um.
    """
    if not isinstance(resultado, dict): return resultado
    if "erro" in resultado: return resultado

    # Metadados do engine — sempre presentes
    resultado.setdefault("motor", motor)
    resultado.setdefault("modelo", modelo_nome or resultado.get("modelo_nome") or motor)
    resultado.setdefault("area", area)
    resultado.setdefault("ano", ano)
    resultado.setdefault("tipo", tipo)
    resultado.setdefault("cor", cor)

    # Qualidade baseada em RMSE local (se disponível)
    rmse_local = resultado.get("rmse_local")
    if rmse_local is not None and "qualidade_estimativa" not in resultado:
        resultado["qualidade_estimativa"] = _qualidade_de_rmse(rmse_local)

    return resultado


# ── Interface única ────────────────────────────────────────────────

def estimar_nota(vetor, b_por_posicao, area, ano, tipo, cor,
                  mascara=None, engine="supervisionado"):
    """
    Estima nota TRI.

    Parâmetros (mesmos do motor histórico, total compatibilidade):
        vetor           — string binária de respostas
        b_por_posicao   — lista de b's dos itens da prova
        area, ano, tipo, cor — identificadores
        mascara         — None ou string '0/1' de aplicabilidade
        engine          — "supervisionado" (default) | "compacto" | "historico"

    Retorna dict com:
        nota_estimada, acertos, erros, coerencia, inversoes,
        media_b_acertos, media_b_erros, hardest_hit, easiest_miss,
        intervalo_min, intervalo_max, confianca, qualidade_estimativa,
        metodo, metodo_ancoragem, motor, modelo, area, ano, tipo, cor
    """
    chave = f"{area}_{ano}_{tipo}_{cor}"

    # ── 1. Supervisionado (default) ────────────────────────────────
    if engine == "supervisionado":
        if _SUPER_OK and _supervisionado_disponivel(area, ano, tipo, cor):
            try:
                res = e_super.estimar_nota_supervisionado(
                    vetor, b_por_posicao, area, ano, tipo, cor, mascara=mascara
                )
                if res and "erro" not in res:
                    modelo_nome = res.get("modelo_nome", "lgb")
                    print(f"[ENGINE] supervisionado  [MODEL] {chave}.pkl  "
                          f"[RMSE] {res.get('rmse_local', '?')}")
                    return _padronizar(res, "supervisionado", area, ano, tipo, cor,
                                        modelo_nome=modelo_nome)
                else:
                    print(f"[WARN] supervisionado retornou erro: "
                          f"{res.get('erro') if res else 'None'} — fallback → compacto")
            except Exception as ex:
                print(f"[WARN] supervisionado exceção em {chave}: {ex} — fallback → compacto")
        else:
            # Modelo não disponível para esta prova
            if not _supervisionado_disponivel(area, ano, tipo, cor):
                print(f"[WARN] supervisionado sem modelo para {chave} → fallback → compacto")

        engine = "compacto"   # cai para o compacto

    # ── 2. Compacto ────────────────────────────────────────────────
    if engine == "compacto":
        if _COMPACT_OK:
            modelo = _carregar_modelo_compacto()
            if modelo is not None:
                try:
                    res = e_compact.estimar_nota_compacto(
                        vetor, b_por_posicao, area, ano, tipo, cor,
                        mascara=mascara, modelo=modelo,
                    )
                    if res and "erro" not in res:
                        print(f"[ENGINE] compacto  [PROVA] {chave}")
                        return _padronizar(res, "compacto", area, ano, tipo, cor)
                    else:
                        print(f"[WARN] compacto retornou erro: "
                              f"{res.get('erro') if res else 'None'} — fallback → historico")
                except Exception as ex:
                    print(f"[WARN] compacto exceção em {chave}: {ex} — fallback → historico")
            else:
                print(f"[WARN] modelo compacto indisponível — fallback → historico")
        engine = "historico"   # cai para o histórico

    # ── 3. Histórico (último recurso — kNN sobre features) ─────────
    if engine == "historico":
        if not _HIST_OK:
            return {"erro": "nenhum motor disponível",
                     "motor": "indisponivel",
                     "area": area, "ano": ano, "tipo": tipo, "cor": cor}
        try:
            res = estimar_historico(
                vetor=vetor, b_por_posicao=b_por_posicao,
                area=area, ano=ano, tipo=tipo, cor=cor, mascara=mascara,
            )
            print(f"[ENGINE] historico  [PROVA] {chave}")
            return _padronizar(res, "historico", area, ano, tipo, cor)
        except Exception as ex:
            return {"erro": f"falha no motor histórico: {ex}",
                     "motor": "historico",
                     "area": area, "ano": ano, "tipo": tipo, "cor": cor}

    return {"erro": f"engine desconhecido: {engine}", "motor": engine}


# ── Utilitários ─────────────────────────────────────────────────────

def status():
    """Diagnóstico: quais motores estão disponíveis e quantos modelos."""
    info = {
        "supervisionado": _SUPER_OK,
        "compacto":       _COMPACT_OK,
        "historico":      _HIST_OK,
        "dir_modelos":    _DIR_MODELOS,
        "n_pkl":          0,
    }
    if os.path.isdir(_DIR_MODELOS):
        info["n_pkl"] = sum(1 for f in os.listdir(_DIR_MODELOS) if f.endswith(".pkl"))
    return info


if __name__ == "__main__":
    import json
    print(json.dumps(status(), indent=2, ensure_ascii=False))
