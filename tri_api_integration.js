/* ═══════════════════════════════════════════════════════════════════════
   TRI Vision — Integração API (v2 com fix de race + reset correto)

   Quando a API está conectada:
   - TODAS as métricas (nota, acertos, coerência) vêm exclusivamente da API
   - Vetor sempre reconstruído do zero a partir de `respostas` atuais
   - Race condition prevenida via AbortController + requestId
   - Vetor todo-zeros também é enviado (após carregar prova ou limpar)
   ═══════════════════════════════════════════════════════════════════════ */

(function() {
  // ─── Detecção automática de ambiente ───────────────────────────────────
  function detectarApiBase() {
    if (window.TRI_API_BASE) return window.TRI_API_BASE;
    const host = window.location.hostname || "";
    if (host.endsWith("netlify.app") || host.endsWith("trivision.com.br")) {
      return "https://trivision-api.onrender.com";
    }
    return "http://localhost:8000";
  }
  const API_BASE = detectarApiBase();
  const TOGGLE_KEY = "triVisionMostrarNota";
  const MIN_RESPOSTAS_TRI = 5;

  // ─── MODO BROWSER-ONLY ─────────────────────────────────────────────────
  // Quando true, NÃO chama API. Usa tri_engine_browser.js + tri_models_browser/.
  // Pode ser sobrescrito antes deste script: window.USE_BROWSER_ENGINE = true;
  const USE_BROWSER_ENGINE = (typeof window.USE_BROWSER_ENGINE !== "undefined")
    ? window.USE_BROWSER_ENGINE
    : true;   // DEFAULT: browser-only (sem servidor)
  const BROWSER_MODELS_URL = window.TRI_BROWSER_MODELS_URL || "tri_models_browser";

  // Estado global
  window.triApi = {
    online:           false,
    mostrarNota:      true,           // controla o olho aberto/fechado
    ultimaEstimativa: null,
    debounceTimer:    null,
    abortController:  null,
    requestId:        0,
    ultimoRequestAceito: 0,
    warmupId:         0,
    ultimoWarmupAceito: 0,
    warmupAbort:      null,
    warming:          false,   // true só durante loading inicial (não trava clique)
  };

  // Carrega preferência do olho
  try {
    const saved = localStorage.getItem(TOGGLE_KEY);
    if (saved !== null) window.triApi.mostrarNota = saved === "1";
  } catch(e) {}

  // ── 1. Verifica conexão com API ────────────────────────────────────────
  async function verificarApi() {
    try {
      const resp = await fetch(API_BASE + "/healthz", { method: "GET" });
      window.triApi.online = resp.ok;
    } catch(e) {
      window.triApi.online = false;
    }
    return window.triApi.online;
  }

  // ── 2. Monta vetor binário SEMPRE do zero ──────────────────────────────
  function montarVetorAtual() {
    if (!window.questoesAtivas || !window.questoesAtivas.length) return null;
    const respostas = window.respostas || {};
    // Convenção do INEP: branco conta como erro.
    // "Uma questão deixada em branco é corrigida como errada" — INEP.
    // Portanto: '1' se acertou, '0' caso contrário (incluindo branco).
    const vetor = window.questoesAtivas.map(q => {
      const r = respostas[q.pos];
      return (r !== undefined && r === q.gab) ? "1" : "0";
    });
    return vetor.join("");
  }

  // ── 3b. Estimativa via motor BROWSER (sem servidor) ───────────────────
  async function estimarViaBrowser() {
    const vetor = montarVetorAtual();
    if (!vetor) return null;

    const area = document.getElementById("sel-area")?.value || "MT";
    const ano  = parseInt(document.getElementById("sel-ano")?.value || "2024");
    const cor  = document.getElementById("sel-cor")?.value || "AMARELA";
    const tipo = document.getElementById("sel-tipo")?.value || "regular";

    // itens da prova: window.questoesAtivas tem .b por questão
    const itens   = (window.questoesAtivas || []).map(q => (q.b !== undefined ? q.b : 0));
    const mascara = window.triMascaraAtual || null;

    const reqId = ++window.triApi.requestId;
    window.triApi.warming = false;
    renderEstadoTRI(ESTADOS_TRI.ESTIMANDO);

    try {
      const data = await window.TriEngineBrowser.estimarNotaBrowser({
        vetor, area, ano, tipo, cor, itens, mascara,
      });
      if (reqId < window.triApi.ultimoRequestAceito) return null;
      window.triApi.ultimoRequestAceito = reqId;

      if (!data || data.erro) {
        console.warn(`[BROWSER ENGINE] #${reqId} ${data ? data.erro : "sem dados"}`);
        renderEstadoTRI(ESTADOS_TRI.ERRO, { msg: "Modelo indisponível" });
        return null;
      }
      // nota_minima_historica não existe no browser — usa intervalo_min como proxy
      if (data.nota_minima_historica == null) {
        data.nota_minima_historica = data.intervalo_min;
      }
      console.log(`[BROWSER ENGINE] #${reqId} ← nota=${data.nota_estimada} `
        + `acertos=${data.acertos} coer=${data.coerencia} modelo=${data.modelo} rmse=${data.rmse_local}`);
      renderEstadoTRI(ESTADOS_TRI.RESULTADO, data);
      return data;
    } catch (e) {
      console.warn(`[BROWSER ENGINE] #${reqId} falha:`, e.message);
      renderEstadoTRI(ESTADOS_TRI.ERRO, { msg: "Falha no motor local" });
      return null;
    }
  }

  // ── 3. Envia vetor para API com proteção contra race condition ─────────
  async function estimarViaApi() {
    // Desvia para o motor browser se ativo
    if (USE_BROWSER_ENGINE) return estimarViaBrowser();

    if (!window.triApi.online) return null;
    const vetor = montarVetorAtual();
    if (!vetor) return null;  // prova não carregada ainda

    const area = document.getElementById("sel-area")?.value || "MT";
    const ano  = parseInt(document.getElementById("sel-ano")?.value || "2024");
    const cor  = document.getElementById("sel-cor")?.value || "AMARELA";
    const tipo = document.getElementById("sel-tipo")?.value || "regular";

    // Aborta request anterior se ainda estiver pendente
    if (window.triApi.abortController) {
      try { window.triApi.abortController.abort(); } catch(e) {}
    }
    const ctrl = new AbortController();
    window.triApi.abortController = ctrl;

    const reqId = ++window.triApi.requestId;
    console.log(`[TRI API] #${reqId} → ${area}/${ano}/${tipo}/${cor} vetor=${vetor.length}c ${vetor.includes("1") ? `(${vetor.split("1").length-1} acertos)` : "(zero)"}`);

    // Estimativa real SEMPRE assume o card — cancela qualquer warmup em curso.
    window.triApi.warming = false;
    if (window.triApi.warmupAbort) {
      try { window.triApi.warmupAbort.abort(); } catch(e) {}
    }
    renderEstadoTRI(ESTADOS_TRI.ESTIMANDO);

    try {
      const resp = await fetch(API_BASE + "/estimar", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ area, ano, tipo, cor, vetor }),
        signal: ctrl.signal,
      });

      if (!resp.ok) {
        console.warn(`[TRI API] #${reqId} HTTP ${resp.status}`);
        renderEstadoTRI(ESTADOS_TRI.ERRO, { msg: `Erro ${resp.status}` });
        return null;
      }
      const data = await resp.json();

      // Ignora resposta stale: se outro request mais novo já chegou, descarta
      if (reqId < window.triApi.ultimoRequestAceito) {
        console.log(`[TRI API] #${reqId} stale ignorada (último aceito: #${window.triApi.ultimoRequestAceito})`);
        return null;
      }
      window.triApi.ultimoRequestAceito = reqId;
      window.triApi.ultimaEstimativa = data;

      console.log(`[TRI API] #${reqId} ← nota=${data.nota_estimada} acertos=${data.acertos} coer=${data.coerencia} motor=${data.motor}`);
      renderEstadoTRI(ESTADOS_TRI.RESULTADO, data);
      return data;
    } catch(e) {
      if (e.name === "AbortError") {
        console.log(`[TRI API] #${reqId} abortada (sucessora chegou)`);
      } else {
        console.warn(`[TRI API] #${reqId} falha:`, e.message);
        renderEstadoTRI(ESTADOS_TRI.ERRO, { msg: "Falha de conexão" });
      }
      return null;
    }
  }

  // ═════════════════════════════════════════════════════════════════════
  //  MÁQUINA DE ESTADOS DO CARD TRI (v2)
  //  Estados: 'idle' | 'warming' | 'estimando' | 'resultado' | 'erro'
  //  Toda renderização do card passa por renderEstadoTRI(estado, dados).
  //
  //  Regras semânticas:
  //   - acertos = 0  → badge "Mínimo histórico", sem coerência
  //   - acertos < 5  → badge "Estimativa inicial"
  //   - acertos >= 5 → badge derivado da confiança backend
  //   - coerência percentual SÓ aparece com acertos >= 2
  //   - campo θ vira "CONFIANÇA" quando API ativa
  // ═════════════════════════════════════════════════════════════════════

  const ESTADOS_TRI = {
    IDLE:       'idle',
    WARMING:    'warming',
    ESTIMANDO:  'estimando',
    RESULTADO:  'resultado',
    ERRO:       'erro',
  };

  // Estado atual (não derivado do DOM)
  window.triApi.estadoCard = ESTADOS_TRI.IDLE;
  window.triApi.ultimoResultado = null;

  // Helpers de acesso ao DOM, em um único lugar
  function _els() {
    return {
      card:       document.getElementById("tri-score-card"),
      nota:       document.getElementById("m-nota"),
      ac:         document.getElementById("m-acertos"),
      theta:      document.getElementById("m-theta"),
      thetaLbl:   document.getElementById("m-theta-label"),
      prof:       document.getElementById("tsc-profile"),
      badge:      document.getElementById("tsc-badge"),
      bar:        document.getElementById("nota-bar"),
    };
  }

  // Renomeia label "θ Proficiência" → "Faixa provável" quando API ativa.
  // (Substituímos o conceito de "confiança" — que soa artificial —
  // pela faixa provável da nota, que é informação concreta.)
  function _renomearLabelTheta() {
    const e = _els();
    if (e.thetaLbl && window.triApi.online) e.thetaLbl.textContent = "Faixa provável";
  }

  function _iconeConfianca(conf) {
    if (conf === "alta")        return "●●●";
    if (conf === "media")       return "●●";
    if (conf === "baixa")       return "●";
    if (conf === "muito_baixa") return "○";
    return "·";
  }

  // Mantido por compatibilidade. NÃO é exibido no card final.
  function _textoConfianca(conf) {
    if (conf === "alta")        return "Alta";
    if (conf === "media")       return "Média";
    if (conf === "baixa")       return "Baixa";
    if (conf === "muito_baixa") return "Muito baixa";
    return "—";
  }

  // Badge derivado dos dados — usa rótulos qualitativos sobre a coerência,
  // não sobre "confiança da estimativa".
  function _badgeFromData(data) {
    const coer = data.coerencia;
    if (coer == null) return { txt: "Estimada", bg: "rgba(99,102,241,0.18)" };
    if (coer >= 0.70) return { txt: "Consistente", bg: "rgba(16,185,129,0.18)" };
    if (coer >= 0.40) return { txt: "Parcial",     bg: "rgba(245,158,11,0.18)" };
    return                   { txt: "Inconsistente", bg: "rgba(239,68,68,0.18)" };
  }

  // Reset visual completo — chamado ao trocar de prova
  function resetVisualTRI() {
    const e = _els();
    if (e.card)  { e.card.classList.remove("has-nota","warming","estimando","erro"); }
    if (e.nota)  { e.nota.textContent  = "—"; e.nota.classList.add("empty"); }
    if (e.ac)    { e.ac.textContent    = "—"; }
    if (e.theta) { e.theta.textContent = "—"; }
    if (e.prof)  { e.prof.textContent  = ""; }
    if (e.badge) { e.badge.textContent = "Aguardando"; e.badge.style.cssText = ""; }
    if (e.bar)   { e.bar.style.width   = "0%"; }
    window.triApi.estadoCard = ESTADOS_TRI.IDLE;
    window.triApi.ultimoResultado = null;
  }

  // Renderização única
  function renderEstadoTRI(estado, dados) {
    const e = _els();
    if (!e.card) return;
    window.triApi.estadoCard = estado;
    e.card.classList.remove("warming","estimando","erro","has-nota");

    switch (estado) {
      case ESTADOS_TRI.IDLE:
        if (e.nota)  { e.nota.textContent = "—"; e.nota.classList.add("empty"); }
        if (e.ac)    { e.ac.textContent    = "—"; }
        if (e.theta) { e.theta.textContent = "—"; }
        if (e.prof)  { e.prof.textContent  = ""; }
        if (e.badge) { e.badge.textContent = "Aguardando"; e.badge.style.cssText = ""; }
        if (e.bar)   { e.bar.style.width   = "0%"; }
        break;

      case ESTADOS_TRI.WARMING:
        // Loading inicial: mostra animação de carregamento e badge "Carregando".
        // NÃO trava cliques — se o usuário clicar, a estimativa real assume.
        e.card.classList.add("warming");
        if (e.nota) {
          e.nota.textContent = "—";
          e.nota.classList.add("empty");
        }
        if (e.ac)    e.ac.textContent    = "—";
        if (e.theta) e.theta.textContent = "—";
        if (e.prof)  { e.prof.textContent = ""; e.prof.title = ""; }
        if (e.badge) {
          e.badge.textContent = "Carregando";
          e.badge.style.background = "rgba(99,102,241,0.20)";
        }
        if (e.bar) e.bar.style.width = "0%";
        break;

      case ESTADOS_TRI.ESTIMANDO:
        // Mantém valores antigos com classe sutil — não pisca
        e.card.classList.add("estimando");
        if (window.triApi.ultimoResultado) {
          e.card.classList.add("has-nota");
        } else {
          // Primeira estimativa em voo (não há valor anterior pra manter).
          // Mostra "Calculando…" pra o usuário saber que está processando,
          // mesmo que o backend leve segundos pra responder (cold start).
          if (e.badge) {
            e.badge.textContent = "Calculando…";
            e.badge.style.background = "rgba(99,102,241,0.28)";
          }
          if (e.nota)  { e.nota.textContent  = "—"; e.nota.classList.add("empty"); }
          if (e.theta) { e.theta.textContent = "—"; }
        }
        break;

      case ESTADOS_TRI.RESULTADO:
        _renderResultado(e, dados);
        break;

      case ESTADOS_TRI.ERRO:
        e.card.classList.add("erro");
        if (e.nota)  { e.nota.textContent  = "—"; e.nota.classList.add("empty"); }
        if (e.ac)    { e.ac.textContent    = "—"; }
        if (e.theta) { e.theta.textContent = "—"; }
        if (e.prof)  { e.prof.textContent  = (dados && dados.msg) || "Erro na estimativa"; }
        if (e.badge) { e.badge.textContent = "Erro"; e.badge.style.background = "rgba(239,68,68,0.18)"; }
        if (e.bar)   { e.bar.style.width   = "0%"; }
        break;
    }
  }

  // Frases explicativas fixas REMOVIDAS — barra de coerência + tooltips
  // assumem 100% da comunicação qualitativa.
  function _fraseDiagnostico(data) {
    return "";   // mantido por compatibilidade — não usado
  }

  // Renderização final (estado=resultado)
  function _renderResultado(e, data) {
    if (!data || data.nota_estimada === undefined || data.nota_estimada === null) {
      renderEstadoTRI(ESTADOS_TRI.ERRO, { msg: "Estimativa indisponível" });
      return;
    }
    window.triApi.ultimoResultado = data;
    e.card.classList.add("has-nota");
    e.card.classList.remove("warming");

    // Log diagnóstico no console (não exposto ao usuário)
    console.log("[TRI API RESULT]", {
      nota:          data.nota_estimada,
      acertos:       data.acertos,
      erros:         data.erros,
      coerencia:     data.coerencia,
      motor:         data.motor,
      modelo:        data.modelo,
      rmse_local:    data.rmse_local,
      qualidade:     data.qualidade_estimativa,
      payload:       data,
    });

    const ac     = data.acertos ?? 0;
    const erros  = data.erros ?? 0;
    // Conta cliques do USUÁRIO (respostas marcadas), não acertos+erros
    // do backend. Branco conta como erro na convenção INEP, então
    // ac+erros = 45 desde o primeiro clique. O número de cliques reais
    // (alternativas selecionadas) é o que define a fase do display.
    const respostasMarcadas = window.respostas ? Object.keys(window.respostas).length : 0;
    // Faixas de exibição do badge e da nota:
    //   • 0 marcações  → vazio "Aguardando"
    //   • 1-3 marcações → "Mínimo histórico" (mostra piso da prova)
    //   • 4+ marcações  → badge normal + nota real computada
    const semCliques   = respostasMarcadas === 0;
    const abaixoLimiar = respostasMarcadas < 4;

    // Caso especial: 0 cliques. Não exibe nota nenhuma — apenas o estado
    // visual indica que estamos aguardando interação.
    if (semCliques) {
      if (e.nota)  { e.nota.textContent  = "—"; e.nota.classList.add("empty"); }
      if (e.ac)    e.ac.textContent      = `0/${(data.total_aplicaveis ?? 45)}`;
      if (e.theta) { e.theta.textContent = "—"; e.theta.title = ""; e.theta.removeAttribute("data-tip"); }
      if (e.prof)  { e.prof.textContent  = ""; e.prof.title = ""; }
      if (e.badge) { e.badge.textContent = "Aguardando"; e.badge.style.background = "rgba(99,102,241,0.18)"; }
      if (e.bar)   e.bar.style.width     = "0%";
      _atualizarStickyCoerencia(data, true);
      return;
    }

    // ─── Contador interno de marcações ───────────────────────────────
    // Conta exatamente quantas alternativas o usuário tem marcadas AGORA.
    // Independente do que o backend retornar (acertos+erros sempre dá 45
    // porque branco conta como erro, conforme regra INEP). Esse contador é
    // a fonte de verdade para o display do mínimo histórico.
    const marcacoes = window.respostas ? Object.keys(window.respostas).length : 0;
    const usarMinHistorico = marcacoes < 4;

    // ─── Nota: contador < 4 → mínimo histórico; >= 4 → nota real ─────
    // Prioriza: 1) campo do payload, 2) cache do warmup, 3) nota estimada.
    const notaMinHistCached = window.triApi.notaMinHistorica;
    const notaExibida = usarMinHistorico
      ? Math.round(data.nota_minima_historica ?? notaMinHistCached ?? data.nota_estimada)
      : Math.round(data.nota_estimada);
    console.log(`[TRI API] marcações=${marcacoes} → ${usarMinHistorico ? "mínimo histórico" : "nota real"} (${notaExibida})`);

    if (e.nota) {
      if (window.triApi.mostrarNota) {
        e.nota.textContent = notaExibida;
        e.nota.classList.remove("empty");
      } else {
        e.nota.textContent = "•••";
        e.nota.classList.add("empty");
      }
    }

    // ─── Acertos / total ─────────────────────────────────────────────
    const totalAplic = (data.total_aplicaveis !== undefined && data.total_aplicaveis !== null)
      ? data.total_aplicaveis
      : ((data.acertos ?? 0) + (data.erros ?? 0));
    if (e.ac) e.ac.textContent = `${ac}/${totalAplic}`;

    // ─── Segundo metric: Faixa provável (intervalo) ──────────────────
    _renomearLabelTheta();   // muda label para "Faixa provável"
    if (e.theta) {
      if (abaixoLimiar || data.intervalo_min == null || data.intervalo_max == null) {
        e.theta.textContent = "—";
        e.theta.title = "";
        e.theta.removeAttribute("data-tip");
      } else {
        const iMin = Math.round(data.intervalo_min);
        const iMax = Math.round(data.intervalo_max);
        e.theta.textContent = `${iMin}–${iMax}`;
        // Tooltip técnico discreto, sem expor modelo/features/JSON
        const tipTxt = _tooltipFaixaProvavel(data);
        e.theta.title = tipTxt;
        e.theta.setAttribute("data-tip", tipTxt);
        _bindTip(e.theta);
      }
    }

    // ─── Frase explicativa fixa REMOVIDA ─────────────────────────────
    // A barra de coerência abaixo da nota assume o papel de indicador
    // qualitativo. O tooltip dela traz o resumo curto.
    if (e.prof) {
      e.prof.textContent = "";
      e.prof.title       = "";
    }

    // ─── Badge: muda conforme a quantidade de cliques ─────────────────
    // (caso semCliques já foi tratado e retornou antes deste ponto)
    if (e.badge) {
      if (abaixoLimiar) {
        // 1-3 cliques: ainda usando piso histórico
        e.badge.textContent = "Mínimo histórico";
        e.badge.style.background = "rgba(148,163,184,0.18)";
      } else {
        // 4+ cliques: badge normal
        const b = _badgeFromData(data);
        e.badge.textContent = b.txt;
        e.badge.style.background = b.bg;
      }
    }

    // ─── BARRA: agora representa COERÊNCIA TRI (não a nota normalizada)
    // Escala: 0% = inconsistente, 50% = parcial, 100% = consistente.
    // Mantém gradiente do CSS — só ajusta a largura.
    if (e.bar) {
      const coer = data.coerencia;
      const pct = (coer == null || abaixoLimiar) ? 0 : Math.max(0, Math.min(100, coer * 100));
      e.bar.style.width = pct.toFixed(1) + "%";
      // Tooltip elegante na barra (mobile + desktop)
      const parent = e.bar.closest(".tsc-bar-wrap") || e.bar.parentElement;
      if (parent) {
        const tipTxt = _tooltipBarraCoerencia(data, abaixoLimiar);
        parent.title = tipTxt;
        parent.setAttribute("data-tip", tipTxt);
        _bindTip(parent);
      }
    }

    // ─── Espelha a barra de coerência no card sticky (mini-tri) ──────
    // Mesma escala, mesmo gradiente. Acompanha o usuário ao rolar a página.
    _atualizarStickyCoerencia(data, abaixoLimiar);
  }

  function _atualizarStickyCoerencia(data, abaixoLimiar) {
    const barFill = document.getElementById("mini-coer-bar");
    const barWrap = document.getElementById("mini-coer-bar-wrap");
    if (!barFill) return;
    const coer = data.coerencia;
    const pct = (coer == null || abaixoLimiar) ? 0 : Math.max(0, Math.min(100, coer * 100));
    barFill.style.width = pct.toFixed(1) + "%";
    if (barWrap) {
      const tipTxt = _tooltipBarraCoerencia(data, abaixoLimiar);
      barWrap.title = tipTxt;
      barWrap.setAttribute("data-tip", tipTxt);
      _bindTip(barWrap);
    }
  }

  // Tooltip da BARRA DE COERÊNCIA — curto, elegante, sem dados técnicos.
  // ──────────────────────────────────────────────────────────────────────
  // SISTEMA DE TOOLTIPS MOBILE-FRIENDLY
  // O atributo `title` nativo NÃO funciona em mobile (não há hover).
  // Esta função instala um overlay customizado que aparece em:
  //   - hover (desktop)
  //   - tap (mobile)  → fica visível até clicar fora ou esperar 4s
  // Lê o texto de `data-tip` (preferido) ou `title` (fallback) do elemento.
  // ──────────────────────────────────────────────────────────────────────
  function _ensureTooltipOverlay() {
    let tip = document.getElementById("tri-tooltip-overlay");
    if (tip) return tip;
    tip = document.createElement("div");
    tip.id = "tri-tooltip-overlay";
    tip.style.cssText = `
      position: fixed; z-index: 99998; pointer-events: none;
      background: rgba(15,23,42,0.97); color: #e2e8f0;
      padding: 8px 12px; border-radius: 8px;
      font-size: 12px; line-height: 1.45;
      font-family: var(--font, system-ui, sans-serif);
      max-width: 280px; white-space: pre-line;
      border: 1px solid rgba(99,102,241,0.4);
      box-shadow: 0 6px 20px rgba(0,0,0,0.45);
      opacity: 0; transform: translateY(-4px);
      transition: opacity 0.15s, transform 0.15s;
    `;
    document.body.appendChild(tip);
    return tip;
  }

  function _showTip(el, evt) {
    const txt = el.getAttribute("data-tip") || el.getAttribute("title");
    if (!txt) return;
    const tip = _ensureTooltipOverlay();
    tip.textContent = txt;

    // Posiciona próximo ao elemento (ou ao ponto de toque)
    const rect = el.getBoundingClientRect();
    let x, y;
    if (evt && evt.touches && evt.touches[0]) {
      x = evt.touches[0].clientX; y = evt.touches[0].clientY - 12;
    } else {
      x = rect.left + rect.width / 2; y = rect.top - 8;
    }
    // Mede o tooltip
    tip.style.opacity = "0";
    tip.style.left = "0px"; tip.style.top = "0px";
    requestAnimationFrame(() => {
      const tw = tip.offsetWidth, th = tip.offsetHeight;
      let left = x - tw / 2;
      let top  = y - th;
      // Ajustes para não vazar da viewport
      const vw = window.innerWidth, vh = window.innerHeight;
      if (left < 6) left = 6;
      if (left + tw > vw - 6) left = vw - tw - 6;
      if (top < 6) top = rect.bottom + 8;   // se não couber acima, vai abaixo
      if (top + th > vh - 6) top = vh - th - 6;
      tip.style.left = left + "px";
      tip.style.top  = top + "px";
      tip.style.opacity = "1";
      tip.style.transform = "translateY(0)";
    });
  }

  function _hideTip() {
    const tip = document.getElementById("tri-tooltip-overlay");
    if (!tip) return;
    tip.style.opacity = "0";
    tip.style.transform = "translateY(-4px)";
  }

  // Instala handlers em um elemento que tenha (ou venha a ter) data-tip/title
  function _bindTip(el) {
    if (!el || el._triTipBound) return;
    el._triTipBound = true;
    el.style.cursor = "help";
    // Desktop: hover
    el.addEventListener("mouseenter", (e) => _showTip(el, e));
    el.addEventListener("mouseleave", _hideTip);
    // Mobile: tap mostra; auto-hide em 4s ou clique fora
    let autoHideTimer = null;
    el.addEventListener("touchstart", (e) => {
      _showTip(el, e);
      clearTimeout(autoHideTimer);
      autoHideTimer = setTimeout(_hideTip, 4000);
      // Não bloqueia o evento — o card ainda funciona normal
    }, { passive: true });
  }

  // Fechar tooltip ao tocar/clicar em qualquer outro lugar
  document.addEventListener("touchstart", (e) => {
    const tip = document.getElementById("tri-tooltip-overlay");
    if (!tip || tip.style.opacity !== "1") return;
    // Se o toque NÃO for sobre um elemento com data-tip, fecha
    if (!e.target.closest("[data-tip]")) _hideTip();
  }, { passive: true });
  document.addEventListener("click", (e) => {
    if (!e.target.closest("[data-tip]")) _hideTip();
  });

  function _tooltipBarraCoerencia(d, abaixoLimiar) {
    if (abaixoLimiar) return "Coerência TRI indisponível com poucos acertos";
    const coer = d.coerencia;
    if (coer == null) return "Coerência TRI: —";
    if (coer >= 0.70) return "Padrão TRI consistente";
    if (coer >= 0.40) return "Padrão TRI parcialmente consistente";
    return "Padrão TRI inconsistente";
  }

  // Tooltip da FAIXA PROVÁVEL — técnico mas elegante, sem expor modelo/JSON.
  function _tooltipFaixaProvavel(d) {
    // Erro médio histórico: derivado de rmse_local quando disponível, senão MAE
    let erro = null;
    if (typeof d.rmse_local === "number" && isFinite(d.rmse_local)) {
      erro = Math.round(d.rmse_local);
    } else if (typeof d.mae_local === "number" && isFinite(d.mae_local)) {
      erro = Math.round(d.mae_local);
    } else if (d.intervalo_min != null && d.intervalo_max != null && d.nota_estimada != null) {
      // Aproximação: meia-largura do intervalo
      erro = Math.max(1, Math.round((d.intervalo_max - d.intervalo_min) / 2));
    }
    if (erro != null) {
      return `Faixa historicamente observada para padrões semelhantes.\n`
           + `Precisão histórica estimada: ±${erro} pontos.`;
    }
    return "Faixa historicamente observada para padrões semelhantes.";
  }

  // Mantido SÓ por compatibilidade — não é mais aplicado ao DOM.
  function _tooltipDiagnostico(d) { return ""; }

  // ── Compat: nomes antigos delegam para a máquina de estados ───────────
  function atualizarUIComEstimativa(data) {
    renderEstadoTRI(ESTADOS_TRI.RESULTADO, data);
  }
  function _resetarDisplayApi() { resetVisualTRI(); }
  function mostrarLoading(msg)  { renderEstadoTRI(ESTADOS_TRI.WARMING, { msg }); }
  function esconderLoading()    { /* gestão via renderEstadoTRI(RESULTADO|ERRO) */ }

  // Expor pra debug
  window.triApi.renderEstadoTRI = renderEstadoTRI;
  window.triApi.resetVisualTRI  = resetVisualTRI;
  window.triApi.ESTADOS         = ESTADOS_TRI;

  // ── 5. Debounce — cálculo automático após cada clique ────────────────
  function dispararEstimativaDebounced() {
    if (!window.triApi.online) return;
    if (window.triApi.warming) {
      // Motor ainda aquecendo. O bloqueio em window.selecionar normalmente
      // evita chegar aqui, mas defendemos por segurança.
      _feedbackWarming();
      return;
    }
    clearTimeout(window.triApi.debounceTimer);
    window.triApi.debounceTimer = setTimeout(estimarViaApi, 200);
  }

  // ── 5c. Warmup — pré-carrega base ao trocar prova ─────────────────────
  async function warmupProva() {
    if (!window.triApi.online) return null;

    const area = document.getElementById("sel-area")?.value || "MT";
    const ano  = parseInt(document.getElementById("sel-ano")?.value || "2024");
    const cor  = document.getElementById("sel-cor")?.value || "AMARELA";
    const tipo = document.getElementById("sel-tipo")?.value || "regular";

    // Cancela warmup anterior se houver
    if (window.triApi.warmupAbort) {
      try { window.triApi.warmupAbort.abort(); } catch(e) {}
    }
    const ctrl = new AbortController();
    window.triApi.warmupAbort = ctrl;

    const wId = ++window.triApi.warmupId;
    window.triApi.warming = true;
    renderEstadoTRI(ESTADOS_TRI.WARMING);   // silencioso, sem mensagem
    console.log(`[TRI API] warmup #${wId} → ${area}/${ano}/${tipo}/${cor}`);

    try {
      const resp = await fetch(API_BASE + "/warmup", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ area, ano, tipo, cor }),
        signal:  ctrl.signal,
      });
      if (!resp.ok) {
        console.warn(`[TRI API] warmup #${wId} HTTP ${resp.status}`);
        // Só desligamos warming/erro se ainda for o warmup mais recente
        if (wId >= window.triApi.warmupId) {
          window.triApi.warming = false;
          renderEstadoTRI(ESTADOS_TRI.ERRO, { msg: `Falha no warmup (${resp.status})` });
        }
        return null;
      }
      const data = await resp.json();

      // Ignora warmup stale (outro mais novo já chegou)
      if (wId < window.triApi.ultimoWarmupAceito) {
        console.log(`[TRI API] warmup #${wId} stale ignorado (último: #${window.triApi.ultimoWarmupAceito})`);
        return null;
      }
      window.triApi.ultimoWarmupAceito = wId;
      window.triApi.warming = false;   // libera cliques

      console.log(`[TRI API] warmup #${wId} ok — base=${data.base_size} load=${data.load_time_ms}ms`);

      // Guarda a nota_zero do warmup como "mínimo histórico" da prova atual.
      // Será usada para exibir nos primeiros 3 cliques do usuário (o estimador
      // supervisionado não retorna nota_minima_historica diretamente, então
      // o warmup é a fonte de verdade desse valor).
      if (data.nota_zero && data.nota_zero.nota_estimada != null) {
        window.triApi.notaMinHistorica = Math.round(data.nota_zero.nota_estimada);
        console.log(`[TRI API] mínimo histórico salvo: ${window.triApi.notaMinHistorica}`);
      } else {
        window.triApi.notaMinHistorica = null;
      }

      // Transição WARMING → RESULTADO com nota_zero (mínimo histórico)
      if (data.nota_zero) {
        renderEstadoTRI(ESTADOS_TRI.RESULTADO, data.nota_zero);
      } else {
        renderEstadoTRI(ESTADOS_TRI.IDLE);
      }
      return data;
    } catch(e) {
      if (e.name === "AbortError") {
        console.log(`[TRI API] warmup #${wId} abortado`);
        // Aborto NÃO mexe na UI — quem abortou já mostrou outro estado
      } else {
        console.warn(`[TRI API] warmup #${wId} falha:`, e.message);
        if (wId === window.triApi.warmupId) {
          window.triApi.warming = false;
          renderEstadoTRI(ESTADOS_TRI.ERRO, { msg: "Falha de conexão" });
        }
      }
      return null;
    }
  }

  // ── 6. Hooks ──────────────────────────────────────────────────────────
  // selecionar() — disparado em cada clique. NÃO bloqueia mais durante
  // warmup; o debounce + aborts garantem que só o último clique conta.
  const _selecionarOriginal = window.selecionar;
  window.selecionar = function(pos, alt, gab) {
    if (typeof _selecionarOriginal === "function") {
      _selecionarOriginal(pos, alt, gab);
    }
    dispararEstimativaDebounced();
  };

  // calcular() — intercepta para impedir motor legado de pintar UI quando API ativa
  const _calcularOriginal = window.calcular;
  window.calcular = function(...args) {
    if (window.triApi.online) {
      // Quando API está ativa, o calcular legado NÃO atualiza nada —
      // a UI vem 100% das respostas da API
      return;
    }
    if (typeof _calcularOriginal === "function") {
      return _calcularOriginal.apply(this, args);
    }
  };

  // carregarProva() — ao trocar prova, dispara warmup
  const _carregarProvaOriginal = window.carregarProva;
  window.carregarProva = function(...args) {
    let r;
    if (typeof _carregarProvaOriginal === "function") {
      r = _carregarProvaOriginal.apply(this, args);
    }
    // Reset visual imediato — evita carry-over da prova anterior
    if (window.triApi.online) {
      resetVisualTRI();
      // Dispara warmup: carrega o .pkl e captura a nota_zero (mínimo histórico)
      // da prova selecionada. Sem isso, o display dos primeiros 3 cliques
      // não tem como saber qual é o piso histórico desta prova específica.
      setTimeout(() => warmupProva(), 50);
    }
    return r;
  };

  // ── 7. Ícone de olho — mostra/oculta a nota visualmente ───────────────
  function injetarOlho() {
    const card = document.getElementById("tri-score-card");
    if (!card || document.getElementById("tri-eye-toggle")) return;

    const btn = document.createElement("button");
    btn.id = "tri-eye-toggle";
    btn.type = "button";
    btn.setAttribute("aria-label", "Mostrar ou ocultar nota");
    btn.title = "Mostrar/ocultar nota";
    btn.style.cssText = `
      width: 26px; height: 26px;
      display: inline-flex; align-items: center; justify-content: center;
      background: transparent; border: none; cursor: pointer;
      padding: 0; margin-left: 8px; vertical-align: middle;
      opacity: 0.55; transition: opacity 0.15s;
      color: var(--text2, #6b7280);
    `;
    btn.addEventListener("mouseenter", () => btn.style.opacity = "1");
    btn.addEventListener("mouseleave", () => btn.style.opacity = "0.55");

    btn.innerHTML = _svgOlho(window.triApi.mostrarNota);

    btn.addEventListener("click", () => {
      window.triApi.mostrarNota = !window.triApi.mostrarNota;
      try { localStorage.setItem(TOGGLE_KEY, window.triApi.mostrarNota ? "1" : "0"); } catch(e) {}
      btn.innerHTML = _svgOlho(window.triApi.mostrarNota);
      // Re-renderiza com último resultado guardado
      if (window.triApi.ultimoResultado) {
        renderEstadoTRI(ESTADOS_TRI.RESULTADO, window.triApi.ultimoResultado);
      }
    });

    // Coloca o olho ao lado do label "NOTA TRI" (canto superior esquerdo),
    // dentro do header, longe do badge de coerência (canto direito).
    const header = card.querySelector(".tsc-header");
    const label  = header ? header.querySelector(".tsc-label") : null;
    if (label) {
      // Insere logo depois do texto "Nota TRI"
      label.insertAdjacentElement("afterend", btn);
    } else if (header) {
      header.insertBefore(btn, header.firstChild);
    } else {
      // Fallback: flutua no topo esquerdo
      btn.style.position = "absolute";
      btn.style.top = "10px";
      btn.style.left = "12px";
      const computed = window.getComputedStyle(card);
      if (computed.position === "static") card.style.position = "relative";
      card.appendChild(btn);
    }
  }

  function _svgOlho(aberto) {
    if (aberto) {
      return `<svg width="18" height="18" viewBox="0 0 24 24" fill="none"
        stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>
        <circle cx="12" cy="12" r="3"/></svg>`;
    }
    return `<svg width="18" height="18" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"/>
      <line x1="1" y1="1" x2="23" y2="23"/></svg>`;
  }

  // ── Inject CSS para estados visuais (warming/estimando/erro) ──────────
  function _injetarCSS() {
    if (document.getElementById("tri-api-styles")) return;
    const style = document.createElement("style");
    style.id = "tri-api-styles";
    style.textContent = `
      /* Loading inicial — pulse sutil enquanto o backend aquece.
         Os cliques continuam funcionando normalmente durante isso. */
      #tri-score-card.warming {
        position: relative;
        animation: triWarmingPulse 1.4s ease-in-out infinite;
      }
      #tri-score-card.warming #m-nota {
        opacity: 0.6;
        letter-spacing: 0.04em;
      }
      @keyframes triWarmingPulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(99,102,241,0.35); }
        50%      { box-shadow: 0 0 0 7px rgba(99,102,241,0.04); }
      }
      #tri-score-card.estimando { opacity: 0.92; transition: opacity 0.15s; }
      #tri-score-card.erro #tsc-badge {
        background: rgba(239,68,68,0.18) !important;
        color: #ef4444 !important;
      }
      /* Tooltips nativos do browser ficam disponíveis nestes elementos */
      #tri-score-card .tsc-bar-wrap   { cursor: help; }
      #tri-score-card #m-theta        { cursor: help; }
    `;
    document.head.appendChild(style);
  }

  // ── 8. Inicialização ──────────────────────────────────────────────────

  // Interceptação global: bloqueia cliques nas questões enquanto o motor
  // estiver "warming" (aquecendo). Mostra feedback visual e console.log,
  // para o usuário entender que NÃO é erro — só está esperando o motor.
  function _instalarBloqueioWarming() {
    if (window._triApiBloqueioInstalado) return;
    window._triApiBloqueioInstalado = true;

    // Salva a referência original e envolve com guarda
    const _selOrig = window.selecionar;
    if (typeof _selOrig !== "function") {
      // selecionar ainda não foi definido — tenta de novo depois
      setTimeout(_instalarBloqueioWarming, 50);
      return;
    }
    window.selecionar = function(pos, alt, gab) {
      if (window.triApi.warming) {
        _feedbackWarming();   // mostra "aguarde, motor aquecendo"
        return;
      }
      return _selOrig.apply(this, arguments);
    };

    // Mesma proteção para deixarBranco
    const _branOrig = window.deixarBranco;
    if (typeof _branOrig === "function") {
      window.deixarBranco = function(pos) {
        if (window.triApi.warming) { _feedbackWarming(); return; }
        return _branOrig.apply(this, arguments);
      };
    }
    console.log("[TRI API] bloqueio de cliques durante warming instalado");
  }

  // Pequeno toast inferior + flash no badge: deixa explícito que o motivo
  // do clique não funcionar é o aquecimento, não bug.
  let _ultimoFeedback = 0;
  function _feedbackWarming() {
    const agora = Date.now();
    if (agora - _ultimoFeedback < 700) return;   // não spammar
    _ultimoFeedback = agora;

    const e = _els();
    if (e.badge) {
      e.badge.textContent = "Motor aquecendo…";
      e.badge.style.background = "rgba(99,102,241,0.32)";
      e.badge.style.transition = "background 0.2s";
    }

    // Toast curto na parte inferior
    let toast = document.getElementById("tri-warming-toast");
    if (!toast) {
      toast = document.createElement("div");
      toast.id = "tri-warming-toast";
      toast.style.cssText = `
        position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%);
        background: rgba(15,23,42,0.96); color: #e2e8f0;
        padding: 10px 18px; border-radius: 999px; font-size: 13px;
        font-family: var(--font, system-ui, sans-serif);
        border: 1px solid rgba(99,102,241,0.4);
        box-shadow: 0 8px 24px rgba(0,0,0,0.4);
        z-index: 99999; pointer-events: none;
        opacity: 0; transition: opacity 0.2s;
      `;
      toast.textContent = "Motor TRI ainda aquecendo — aguarde um instante";
      document.body.appendChild(toast);
    }
    toast.style.opacity = "1";
    clearTimeout(window._triToastTimer);
    window._triToastTimer = setTimeout(() => { toast.style.opacity = "0"; }, 1500);
  }

  document.addEventListener("DOMContentLoaded", async () => {
    _injetarCSS();
    _instalarBloqueioWarming();

    if (USE_BROWSER_ENGINE) {
      // Modo browser-only — não há servidor.
      if (!window.TriEngineBrowser) {
        console.error("[BROWSER ENGINE] tri_engine_browser.js não carregado!");
        return;
      }
      try {
        window.triApi.warming = true;
        renderEstadoTRI(ESTADOS_TRI.WARMING);   // mostra "Carregando" enquanto baixa o índice
        await window.TriEngineBrowser.init(BROWSER_MODELS_URL);
        window.triApi.online = true;   // "online" no sentido de motor pronto
        window.triApi.warming = false;
        console.log("[BROWSER ENGINE] pronto — sem servidor");
        _renomearLabelTheta();
        injetarOlho();
        // Card vazio (IDLE) até o usuário escolher prova E começar a marcar.
        // Não disparamos estimativa automática — antes a chamada
        // estimarViaBrowser() no boot gerava uma nota "fantasma" do vetor
        // zero da primeira prova default.
        renderEstadoTRI(ESTADOS_TRI.IDLE);
      } catch (e) {
        console.error("[BROWSER ENGINE] falha init:", e.message);
        window.triApi.warming = false;
        renderEstadoTRI(ESTADOS_TRI.ERRO, { msg: "Falha ao carregar modelos" });
      }
      return;
    }

    // Modo API (legado/Render): também precisa marcar warming durante o
    // verificarApi, para o bloqueio de cliques funcionar.
    window.triApi.warming = true;
    renderEstadoTRI(ESTADOS_TRI.WARMING);

    await verificarApi();
    if (window.triApi.online) {
      console.log("[TRI API] conectado em", API_BASE);
      window.triApi.warming = false;
      _renomearLabelTheta();
      injetarOlho();
      // Card vazio (IDLE) — não disparamos warmup automático.
      // O warmupProva() roda apenas quando o usuário troca de prova
      // (chamado por carregarProva no gerar_html.py). Antes, o boot puxava
      // a primeira prova default e exibia o mínimo histórico dela, dando
      // a impressão de uma "nota fantasma" sem clique nenhum.
      renderEstadoTRI(ESTADOS_TRI.IDLE);
    } else {
      window.triApi.warming = false;
      console.log("[TRI API] offline — usando motor local");
    }
  });

  // Expor pra debug
  window.triApi.estimar     = estimarViaApi;
  window.triApi.warmup      = warmupProva;
  window.triApi.montarVetor = montarVetorAtual;
  window.triApi.resetar     = _resetarDisplayApi;
})();
