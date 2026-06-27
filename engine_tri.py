"""
engine_tri.py  (v3 - EAP primario)
----------------------------------
Motor TRI unificado.

Mudanca em relacao a v2:
  - Motor PRIMARIO agora e o EAP 3PL (estimador_eap), que reproduz o
    modelo do INEP e nao inverte o efeito coerencia nos perfis estrategicos.
  - O LightGBM (estimador_lgb) vira SEGUNDA OPINIAO: roda em paralelo, e a
    divergencia entre os dois e exposta em metadados (campo lgb_*). Nao
    altera a nota exibida.
  - A nota exibida e a faixa de incerteza vem do EAP (faixa = std real por
    nivel de acertos, nao mais +-RMSE do LightGBM).

Interface preservada: estimar_nota(...) devolve dict com nota_estimada,
intervalo_min/max, etc. A API e o frontend nao mudam.

Para reverter ao motor antigo, defina TRI_MOTOR=lgb no ambiente.
"""

import os

_MOTOR = os.environ.get("TRI_MOTOR", "eap").lower()   # "eap" | "lgb"

try:
    import estimador_eap as e_eap
    _EAP_OK = True
except Exception as e:
    _EAP_OK = False
    print(f"[ENGINE] estimador_eap indisponivel: {e}")

try:
    import estimador_lgb as e_lgb
    _LGB_OK = True
except Exception as e:
    _LGB_OK = False
    print(f"[ENGINE] estimador_lgb indisponivel: {e}")


def configurar(dir_modelos="modelos_v2",
               mapeamento_path="mapeamento_canonico_v6.json",
               params_eap="parametros_eap.json"):
    if _EAP_OK:
        e_eap.configurar_dir(params_eap, mapeamento_path)
    if _LGB_OK:
        e_lgb.configurar_dir(dir_modelos, mapeamento_path)


def _qualidade(std):
    if std is None: return None
    if std <= 8: return "alta"
    if std <= 18: return "media"
    return "baixa"


def estimar_nota(vetor, b_por_posicao, area, ano, tipo, cor,
                 mascara=None, engine="supervisionado", lingua=None):
    """
    Interface compativel. Decide o motor por _MOTOR (default eap).
    """
    primario = _MOTOR
    if primario == "eap" and not _EAP_OK:
        primario = "lgb"
    if primario == "lgb" and not _LGB_OK:
        primario = "eap"

    if primario == "eap":
        return _estimar_eap_primario(vetor, b_por_posicao, area, ano, tipo,
                                     cor, mascara, lingua)
    return _estimar_lgb_primario(vetor, b_por_posicao, area, ano, tipo,
                                 cor, mascara, lingua)


def _estimar_eap_primario(vetor, b_por_posicao, area, ano, tipo, cor,
                          mascara, lingua):
    try:
        res = e_eap.estimar_nota_supervisionado(
            vetor=vetor, b_por_posicao=b_por_posicao,
            area=area, ano=ano, tipo=tipo, cor=cor,
            mascara=mascara, lingua=lingua)
    except Exception as ex:
        return {"erro": f"excecao no EAP: {ex}", "motor": "eap",
                "area": area, "ano": ano, "tipo": tipo, "cor": cor}
    if "erro" in res:
        # fallback automatico para LGB se EAP falhar nessa chave
        if _LGB_OK:
            print(f"[ENGINE] EAP falhou em {res.get('chave','?')}: "
                  f"{res['erro']} -> fallback LGB")
            return _estimar_lgb_primario(vetor, b_por_posicao, area, ano,
                                         tipo, cor, mascara, lingua)
        return res

    res.setdefault("motor", "eap_3pl")
    res.setdefault("modelo", res.get("modelo_nome", "eap"))
    res["area"], res["ano"], res["tipo"], res["cor"] = area, ano, tipo, cor
    res.setdefault("qualidade_estimativa", _qualidade(res.get("std_faixa")))

    # segunda opiniao (LightGBM) - so metadados, nao muda a nota
    if _LGB_OK:
        try:
            r2 = e_lgb.estimar_nota_supervisionado(
                vetor=vetor, b_por_posicao=b_por_posicao,
                area=area, ano=ano, tipo=tipo, cor=cor,
                mascara=mascara, lingua=lingua)
            if "erro" not in r2:
                nota_lgb = r2.get("nota_estimada")
                res["lgb_nota"] = nota_lgb
                if nota_lgb is not None:
                    res["lgb_divergencia"] = round(
                        res["nota_estimada"] - nota_lgb, 2)
                res["lgb_coerencia"] = r2.get("coerencia")
        except Exception:
            pass

    chave = res.get("chave", "?")
    print(f"[ENGINE] eap  {chave}  theta={res.get('theta')}  "
          f"nota={res.get('nota_estimada')}  "
          f"(lgb={res.get('lgb_nota')})")
    return res


def _estimar_lgb_primario(vetor, b_por_posicao, area, ano, tipo, cor,
                          mascara, lingua):
    if not _LGB_OK:
        return {"erro": "estimador LightGBM indisponivel", "motor": "lgb"}
    try:
        res = e_lgb.estimar_nota_supervisionado(
            vetor=vetor, b_por_posicao=b_por_posicao,
            area=area, ano=ano, tipo=tipo, cor=cor,
            mascara=mascara, lingua=lingua)
    except Exception as ex:
        return {"erro": f"excecao no estimador: {ex}", "motor": "lgb"}
    if "erro" in res:
        return res
    res.setdefault("motor", "lightgbm")
    res.setdefault("modelo", res.get("modelo_nome", "lgb"))
    res["area"], res["ano"], res["tipo"], res["cor"] = area, ano, tipo, cor
    rmse = res.get("rmse_local")
    if rmse is not None and "qualidade_estimativa" not in res:
        res["qualidade_estimativa"] = ("alta" if rmse <= 10
                                       else "media" if rmse <= 20 else "baixa")
    return res


def status():
    info = {"motor_primario": _MOTOR}
    if _EAP_OK:
        info["eap"] = e_eap.status()
    if _LGB_OK:
        info["lgb"] = e_lgb.status()
    base = info.get("eap" if _MOTOR == "eap" and _EAP_OK else "lgb", {})
    info["motor"] = base.get("motor", "indisponivel")
    info["n_pkl"] = (e_lgb.status().get("n_pkl") if _LGB_OK else 0)
    info["aprovados"] = (e_lgb.status().get("aprovados") if _LGB_OK else 0)
    info["n_chaves_eap"] = (e_eap.status().get("n_chaves") if _EAP_OK else 0)
    return info


if __name__ == "__main__":
    import json
    print(json.dumps(status(), indent=2, ensure_ascii=False))
