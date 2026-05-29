# TRI Vision — Backend (Render)

API FastAPI que serve o motor supervisionado TRI.

## Arquivos Python (já incluídos)
- `api.py`               — FastAPI app, rotas `/healthz`, `/estimar`, `/warmup`
- `engine_tri.py`        — router: supervisionado → compacto → histórico
- `estimador_supervisionado.py`
- `estimador_compacto.py`
- `estimador_nota.py`    — fallback kNN histórico
- `gerar_vetor_estrategico.py`
- `requirements.txt`, `render.yaml`

## Dados que VOCÊ precisa copiar para esta pasta antes do deploy

1. **`modelos_supervisionados/`** — pasta com os 376 `.pkl`
2. **`provas_todas.json`** — índice de provas (raiz)
3. **`modelo_compacto_tri.json`** — fallback compacto (raiz)
4. **`dados_inep/`** — pasta com `ITENS_PROVA_<ano>.csv`
5. **`features/`** — parquets `features_<area>_<ano>_<tipo>.parquet` *só se quiser fallback histórico funcional*

> O motor supervisionado **não precisa dos parquets em runtime**. Você só
> precisa deles se quiser que o histórico (kNN) funcione como segundo fallback.
> No plano free do Render (512 MB), os parquets podem estourar o espaço.
> **Recomendo deploy SEM `features/`** — supervisionado + compacto cobrem 100%.

## Não inclua nesta pasta
- `tri_models_browser/` — só serve para o motor browser
- `cartao_tri_enem.html` — frontend vai pro Netlify
- Validadores, treinos, debug scripts, CSVs gigantes, relatórios

## Variáveis de ambiente (Render)
- `ALLOWED_ORIGIN` — domínio do Netlify (ex: `https://trivision.netlify.app`).
  Deixe vazio durante testes (libera `*`).

## Teste local
```cmd
cd deploy_render
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
```
Verifica:
```cmd
curl http://localhost:8000/healthz
```

## Deploy no Render
1. Cria repo Git só com o conteúdo de `deploy_render/`
2. No Render: New → Web Service → conecta o repo
3. Render detecta `render.yaml` automaticamente
4. Anota a URL final (ex: `https://trivision-api.onrender.com`)
5. Configura `ALLOWED_ORIGIN` com o domínio do Netlify
