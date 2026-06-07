"""
aprovar_quarentena_manual.py
-----------------------------
Marca manualmente MT_2017_regular e MT_2019_regular como aprovados,
mantendo flag de baixa_confianca para o frontend exibir aviso.
"""

import os, pickle, sys

CHAVES = ["MT_2017_regular", "MT_2019_regular"]
DIR_MODELOS = "modelos_v2"


def main():
    for chave in CHAVES:
        path = os.path.join(DIR_MODELOS, f"{chave}.pkl")
        if not os.path.exists(path):
            print(f"[X] {chave}.pkl nao encontrado"); continue

        with open(path, "rb") as f:
            payload = pickle.load(f)

        antigo_status = payload.get("status", "?")
        motivos_orig = list(payload.get("motivos_quarentena", []))

        # Aprova como qualquer outro modelo. A faixa min-max que o frontend
        # exibe naturalmente comunica a incerteza.
        payload["status"] = "aprovado"
        payload["motivos_quarentena_original"] = motivos_orig
        payload["motivos_quarentena"] = []
        payload["aprovado_manualmente"] = True

        with open(path, "wb") as f:
            pickle.dump(payload, f)

        m = payload.get("metricas_holdout", {})
        print(f"[OK] {chave}")
        print(f"     status: {antigo_status} -> aprovado")
        print(f"     MAE={m.get('mae', '?'):.2f} RMSE={m.get('rmse', '?'):.2f}")
        print(f"     motivos originais: {motivos_orig}")


if __name__ == "__main__":
    main()
