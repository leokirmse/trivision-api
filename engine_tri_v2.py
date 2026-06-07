"""
engine_tri_v2.py
-----------------
Camada unificada de motor TRI v2.

Sem fallback complexo: usa estimador_lgb_v2 (LightGBM canonico).
Se o modelo nao existe ou falha, retorna erro estruturado (sem mascarar).
"""

import os

try:
    import estimador_lgb_v2 as e_lgb
    _LGB_OK = True
except Exception as e:
    _LGB_OK = False
    print(f"[ENGINE_V2] estimador_lgb_v2 indisponivel: {e}")


def configurar(dir_modelos="modelos_v2",
                mapeamento_path="mapeamento_canonico_v6.json"):
    if _LGB_OK:
        e_lgb.configurar_dir(dir_modelos, mapeamento_path)


def _qualidade_de_rmse(rmse):
    if rmse is None: return None
    if rmse <= 10: return "alta"
    if rmse <= 20: return "media"
    return "baixa"


def estimar_nota(vetor, b_por_posicao, area, ano, tipo, cor,
                  mascara=None, engine="supervisionado", lingua=None):
    """
    Interface compativel com engine_tri antigo.

    Parametros novos:
      lingua: "ing" ou "esp" (para LC)
    """
    if not _LGB_OK:
        return {"erro": "estimador LightGBM indisponivel", "motor": "lgb"}

    try:
        res = e_lgb.estimar_nota_supervisionado(
            vetor=vetor,
            b_por_posicao=b_por_posicao,
            area=area, ano=ano, tipo=tipo, cor=cor,
            mascara=mascara, lingua=lingua,
        )
    except Exception as ex:
        return {"erro": f"excecao no estimador: {ex}", "motor": "lgb",
                "area": area, "ano": ano, "tipo": tipo, "cor": cor}

    if "erro" in res:
        print(f"[ENGINE_V2] erro em {res.get('chave', '?')}: {res['erro']}")
        return res

    # Padronizacao (compativel com motor antigo)
    res.setdefault("motor", "lightgbm")
    res.setdefault("modelo", res.get("modelo_nome", "lgb"))
    res.setdefault("area", area)
    res.setdefault("ano",  ano)
    res.setdefault("tipo", tipo)
    res.setdefault("cor",  cor)
    rmse = res.get("rmse_local")
    if rmse is not None and "qualidade_estimativa" not in res:
        res["qualidade_estimativa"] = _qualidade_de_rmse(rmse)

    chave = res.get("chave", "?")
    print(f"[ENGINE_V2] lgb  [MODEL] {chave}.pkl  "
          f"[RMSE] {rmse}  [nota] {res.get('nota_estimada')}")
    return res


def status():
    if _LGB_OK:
        info = e_lgb.status()
        info["motor"] = "lightgbm"
        return info
    return {"motor": "indisponivel"}


if __name__ == "__main__":
    import json
    print(json.dumps(status(), indent=2, ensure_ascii=False))
