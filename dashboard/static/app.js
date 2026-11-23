/* BTC AI Trader dashboard - plain JS + Chart.js. */
"use strict";

const $ = (s) => document.querySelector(s);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const usd = (v, d = 2) => v == null ? "-" : "$" + Number(v).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (v, d = 1) => v == null ? "-" : (v > 0 ? "+" : "") + Number(v).toFixed(d) + "%";
const signCls = (v) => v > 0 ? "up-txt" : v < 0 ? "down-txt" : "";
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmtTime = (iso, withTime) => {
  const d = new Date(iso);
  return withTime ? d.toLocaleString(undefined, { year: "numeric", month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit" })
                  : d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "2-digit" });
};

/* ------------------------------------------------------------ theme + tabs */
const charts = {};
function themeColors() {
  return {
    text: css("--text"), text2: css("--text-2"), muted: css("--muted"), grid: css("--grid"),
    s1: css("--series-1"), s2: css("--series-2"), good: css("--good"), bad: css("--critical"),
    surface: css("--surface"), seq0: css("--seq-0"), seq1: css("--seq-1"),
  };
}
function applyChartDefaults() {
  const c = themeColors();
  Chart.defaults.color = c.text2;
  Chart.defaults.borderColor = c.grid;
  Chart.defaults.font.family = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
  Chart.defaults.font.size = 12;
  Chart.defaults.animation = false;
  Chart.defaults.plugins.legend.display = false;
  Chart.defaults.plugins.tooltip.backgroundColor = c.surface;
  Chart.defaults.plugins.tooltip.titleColor = c.text;
  Chart.defaults.plugins.tooltip.bodyColor = c.text2;
  Chart.defaults.plugins.tooltip.borderColor = c.grid;
  Chart.defaults.plugins.tooltip.borderWidth = 1;
  Chart.defaults.plugins.tooltip.padding = 10;
  Chart.defaults.elements.line.borderWidth = 2;
  Chart.defaults.elements.point.radius = 0;
  Chart.defaults.elements.point.hoverRadius = 4;
}
$("#themeToggle").addEventListener("click", () => {
  const html = document.documentElement;
  html.dataset.theme = html.dataset.theme === "dark" ? "light" : "dark";
  try { localStorage.setItem("theme", html.dataset.theme); } catch (e) { /* storage unavailable */ }
  applyChartDefaults();
  if (overviewData) renderOverview(overviewData);
  rebuildTradingCharts(true);
  renderFigures();
});
try { const t = localStorage.getItem("theme"); if (t) document.documentElement.dataset.theme = t; } catch (e) { /* ignore */ }

document.querySelectorAll(".tab").forEach((btn) => btn.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b === btn));
  document.querySelectorAll(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === "tab-" + btn.dataset.tab));
}));

function makeChart(id, config) {
  if (charts[id]) charts[id].destroy();
  charts[id] = new Chart(document.getElementById(id), config);
  return charts[id];
}
const timeScale = (unit) => ({
  type: "time", time: unit ? { unit } : {}, grid: { display: false },
  ticks: { maxRotation: 0, autoSkipPadding: 24 },
});
const valueScale = (opts = {}) => ({ grid: { color: themeColors().grid }, border: { display: false }, ...opts });
const kpi = (label, value, delta = "", cls = "") =>
  `<div class="kpi"><div class="label">${label}</div><div class="value ${cls}">${value}</div><div class="delta">${delta}</div></div>`;

/* Shades train / val / test regions behind the price history. */
const splitBands = {
  id: "splitBands",
  beforeDatasetsDraw(chart, _args, opts) {
    if (!opts || !opts.bands) return;
    const { ctx, chartArea: a, scales: { x } } = chart;
    ctx.save();
    opts.bands.forEach((b) => {
      const x0 = Math.max(x.getPixelForValue(+new Date(b.from)), a.left);
      const x1 = Math.min(x.getPixelForValue(+new Date(b.to)), a.right);
      ctx.fillStyle = b.color;
      ctx.fillRect(x0, a.top, x1 - x0, a.bottom - a.top);
      ctx.fillStyle = themeColors().muted;
      ctx.font = "12px system-ui";
      ctx.fillText(b.label, x0 + 6, a.top + 14);
    });
    ctx.restore();
  },
};
Chart.register(splitBands);

/* ------------------------------------------------------------ overview */
let overviewData = null;
async function loadOverview() {
  try {
    const r = await fetch("/api/overview");
    const d = await r.json();
    if (d.status === "loading") return setTimeout(loadOverview, 1500);
    if (d.status === "error") {
      $("#overviewLoading").textContent = "Could not build the overview: " + d.error;
      return;
    }
    overviewData = d;
    $("#overviewLoading").hidden = true;
    $("#overviewBody").hidden = false;
    renderOverview(d);
    renderFigures();
  } catch (e) {
    setTimeout(loadOverview, 2000);
  }
}

function renderOverview(d) {
  const c = themeColors();
  const ab = d.agent_backtest.summary;
  const rf = d.models.find((m) => m.name.startsWith("Random forest (calibrated"));
  const per = d.models.find((m) => m.name.startsWith("Persistence"));
  $("#overviewKpis").innerHTML = [
    kpi("Dataset (hourly candles)", (d.dataset.raw_rows || 0).toLocaleString(),
        `${fmtTime(d.dataset.first)} → ${fmtTime(d.dataset.last)}`),
    kpi("Data completeness", (d.dataset.completeness_pct ?? 0).toFixed(2) + "%",
        `${d.dataset.missing_rows} missing hours, longest gap ${d.dataset.longest_gap_hours} h`),
    kpi("Unseen test window", d.splits.test_days + " days",
        `${fmtTime(d.splits.test_start)} → ${fmtTime(d.splits.test_end)}`),
    kpi("AI agent on test", usd(ab.equity), `${pct(ab.pnl_pct)} vs buy-and-hold ${pct(ab.bh_pnl_pct)}`, signCls(ab.pnl_pct)),
    kpi("Agent trades / win rate", `${ab.n_trades} / ${ab.win_rate == null ? "-" : ab.win_rate.toFixed(0) + "%"}`,
        `max drawdown ${pct(ab.max_drawdown_pct)}, fees ${usd(ab.fees_paid)}`),
    kpi("Model macro-F1 (test)", rf.macro_f1.toFixed(3), `persistence baseline ${per.macro_f1.toFixed(3)}`,
        rf.macro_f1 > per.macro_f1 ? "up-txt" : "down-txt"),
  ].join("");

  const beats = rf.macro_f1 > per.macro_f1 && rf.roc_auc > per.roc_auc;
  const outcome = ab.pnl_pct >= 0
    ? `made ${pct(ab.pnl_pct)}`
    : ab.pnl_pct > ab.bh_pnl_pct ? `lost less than buy-and-hold (${pct(ab.pnl_pct)} vs ${pct(ab.bh_pnl_pct)}) - it did not make money`
    : `lost more than buy-and-hold (${pct(ab.pnl_pct)} vs ${pct(ab.bh_pnl_pct)})`;
  $("#verdict").innerHTML = `<strong>Honest read:</strong> on this unseen window the calibrated model ${beats ? "beats" : "does not beat"} both baselines
    (by a thin margin), and the AI agent ${outcome} in a falling market. The walk-forward check (<em>07_regime_robustness</em> in Report figures)
    shows the edge is not stable across bear, bull and range regimes. Treat this as a learning project, not a money-making system.`;

  // price history, log scale, split shading
  const s = d.splits;
  const first = d.price[0].t, last = d.price[d.price.length - 1].t;
  makeChart("priceHistory", {
    type: "line",
    data: { datasets: [{ data: d.price.map((p) => ({ x: p.t, y: p.v })), borderColor: c.s1 }] },
    options: {
      maintainAspectRatio: false, parsing: { xAxisKey: "x", yAxisKey: "y" },
      interaction: { mode: "index", intersect: false },
      scales: { x: timeScale("year"), y: valueScale({ type: "logarithmic", ticks: { callback: (v) => {
        const n = Number(v); return [1000, 3000, 10000, 30000, 100000].includes(n) ? "$" + n.toLocaleString() : ""; } } }) },
      plugins: {
        tooltip: { callbacks: { title: (it) => fmtTime(it[0].parsed.x), label: (it) => " Close " + usd(it.parsed.y, 0) } },
        splitBands: { bands: [
          { from: first, to: s.train_end, label: "train", color: "transparent" },
          { from: s.train_end, to: s.val_end, label: "validation", color: "rgba(127,127,127,0.10)" },
          { from: s.test_start, to: last, label: "test (unseen)", color: "rgba(127,127,127,0.20)" },
        ] },
      },
    },
  });

  // agent backtest
  const buys = d.agent_backtest.trades.filter((t) => t.action === "BUY").map((t) => ({ x: t.t, y: t.price }));
  const sells = d.agent_backtest.trades.filter((t) => t.action === "SELL").map((t) => ({ x: t.t, y: t.price }));
  makeChart("agentBacktest", {
    type: "line",
    data: { datasets: [
      { label: "AI agent", data: d.agent_backtest.equity.map((p) => ({ x: p.t, y: p.v })), borderColor: c.s1 },
      { label: "Buy-and-hold", data: d.agent_backtest.bh.map((p) => ({ x: p.t, y: p.v })), borderColor: c.s2 },
    ] },
    options: {
      maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
      scales: { x: timeScale("month"), y: valueScale({ ticks: { callback: (v) => "$" + v } }) },
      plugins: {
        legend: { display: true, position: "top", align: "end", labels: { boxWidth: 16, boxHeight: 2 } },
        tooltip: { callbacks: { title: (it) => fmtTime(it[0].parsed.x), label: (it) => ` ${it.dataset.label}: ${usd(it.parsed.y)}` } },
      },
    },
  });

  makeChart("volatility", {
    type: "line",
    data: { datasets: [{ data: d.volatility.map((p) => ({ x: p.t, y: p.v })), borderColor: c.s1 }] },
    options: {
      maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
      scales: { x: timeScale("year"), y: valueScale({ ticks: { callback: (v) => v + "%" } }) },
      plugins: { tooltip: { callbacks: { title: (it) => fmtTime(it[0].parsed.x), label: (it) => ` ${it.parsed.y.toFixed(1)}% annualized` } } },
    },
  });

  // model table with inline bars
  const maxF1 = Math.max(...d.models.map((m) => m.macro_f1));
  $("#modelTable").innerHTML = `<thead><tr><th>Model</th><th>macro-F1</th><th>ROC-AUC</th></tr></thead><tbody>` +
    d.models.map((m) => `<tr class="${m.name.includes("agent") ? "highlight" : ""}"><td>${esc(m.name)}</td>
      <td class="n bar-cell"><span class="b" style="width:${(m.macro_f1 / maxF1) * 100}%"></span><span class="t">${m.macro_f1.toFixed(3)}</span></td>
      <td class="n">${m.roc_auc == null ? "-" : m.roc_auc.toFixed(3)}</td></tr>`).join("") + "</tbody>";

  // confusion heatmap (sequential single hue)
  const m = d.confusion.matrix, labels = d.confusion.labels;
  const max = Math.max(...m.flat());
  let html = `<div class="hd"></div>` + labels.map((l) => `<div class="hd">pred ${l}</div>`).join("");
  m.forEach((row, i) => {
    html += `<div class="hd">${labels[i]}</div>`;
    row.forEach((v) => {
      const t = max ? v / max : 0;
      const bg = `color-mix(in srgb, ${c.seq1} ${Math.round(8 + t * 82)}%, ${c.seq0})`;
      html += `<div class="cell" style="background:${bg};color:${t > 0.5 ? "#fff" : c.text}">${v}</div>`;
    });
  });
  $("#confusion").innerHTML = html;

  makeChart("classMix", {
    type: "bar",
    data: { labels: ["down", "flat", "up"], datasets: [
      { label: "Predicted", data: ["down", "flat", "up"].map((k) => d.class_mix.predicted[k]), backgroundColor: c.s1, borderRadius: 4, borderSkipped: "start" },
      { label: "Actual", data: ["down", "flat", "up"].map((k) => d.class_mix.actual[k]), backgroundColor: c.s2, borderRadius: 4, borderSkipped: "start" },
    ] },
    options: {
      maintainAspectRatio: false, categoryPercentage: 0.7, barPercentage: 0.9,
      scales: { x: { grid: { display: false } }, y: valueScale({ ticks: { callback: (v) => v + "%" } }) },
      plugins: { legend: { display: true, position: "top", align: "end", labels: { boxWidth: 10, boxHeight: 10 } },
        tooltip: { callbacks: { label: (it) => ` ${it.dataset.label}: ${it.parsed.y.toFixed(1)}% of days` } } },
    },
  });

  const imp = Object.entries(d.feature_importance);
  makeChart("importance", {
    type: "bar",
    data: { labels: imp.map((e) => e[0]), datasets: [{ data: imp.map((e) => e[1] * 100), backgroundColor: c.s1, borderRadius: 4, borderSkipped: "start" }] },
    options: {
      indexAxis: "y", maintainAspectRatio: false,
      scales: { x: valueScale({ ticks: { callback: (v) => v + "%" } }), y: { grid: { display: false }, ticks: { autoSkip: false } } },
      plugins: { tooltip: { callbacks: { label: (it) => ` ${it.parsed.x.toFixed(1)}% of total importance` } } },
    },
  });
}

/* ------------------------------------------------------------ figures */
function renderFigures() {
  if (!overviewData) return;
  $("#gallery").innerHTML = overviewData.figures.map((f) =>
    `<figure data-src="/figures/${encodeURIComponent(f.file)}" data-cap="${esc(f.caption)}">
       <img loading="lazy" src="/figures/${encodeURIComponent(f.file)}" alt="${esc(f.caption)}">
       <figcaption><b>${esc(f.section)} · ${esc(f.file)}</b>${esc(f.caption)}</figcaption></figure>`).join("");
  document.querySelectorAll(".gallery figure").forEach((fig) => fig.addEventListener("click", () => {
    const lb = $("#lightbox");
    lb.querySelector("img").src = fig.dataset.src;
    lb.querySelector("p").textContent = fig.dataset.cap;
    lb.hidden = false;
  }));
}
$("#lightbox").addEventListener("click", () => { $("#lightbox").hidden = true; });

/* ------------------------------------------------------------ trading */
const T = { records: [], history: [], next: 0, polling: null, interval: "1D", mode: "replay" };

function rebuildTradingCharts(keepData) {
  const c = themeColors();
  const unit = T.mode === "live" ? undefined : (T.interval === "1h" ? "day" : "month");
  makeChart("liveChart", {
    type: "line",
    data: { datasets: [
      { label: "BTC price", data: [], borderColor: c.s1, order: 3 },
      { label: "AI forecast", data: [], borderColor: c.s2, borderDash: [6, 4], borderWidth: 1.5, order: 4 },
      { label: "Buy", type: "scatter", data: [], pointStyle: "triangle", pointRadius: 8, pointHoverRadius: 10,
        backgroundColor: c.good, borderColor: c.surface, borderWidth: 2, order: 1 },
      { label: "Sell", type: "scatter", data: [], pointStyle: "triangle", rotation: 180, pointRadius: 8, pointHoverRadius: 10,
        backgroundColor: c.bad, borderColor: c.surface, borderWidth: 2, order: 2 },
      { label: "Recent price (before start)", data: [], borderColor: c.muted, borderWidth: 1.5, order: 5 },
    ] },
    options: {
      maintainAspectRatio: false, interaction: { mode: "nearest", axis: "x", intersect: false },
      scales: { x: timeScale(unit), y: valueScale({ ticks: { callback: (v) => "$" + Number(v).toLocaleString() } }) },
      plugins: { tooltip: { callbacks: {
        title: (it) => fmtTime(it[0].parsed.x, T.mode === "live" || T.interval === "1h"),
        label: (it) => ` ${it.dataset.label}: ${usd(it.parsed.y)}`,
      } } },
    },
  });
  makeChart("walletChart", {
    type: "line",
    data: { datasets: [
      { label: "AI agent", data: [], borderColor: c.s1 },
      { label: "Buy-and-hold", data: [], borderColor: c.s2 },
    ] },
    options: {
      maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
      scales: { x: timeScale(unit), y: valueScale({ ticks: { callback: (v) => "$" + Number(v).toLocaleString() } }) },
      plugins: { tooltip: { callbacks: {
        title: (it) => fmtTime(it[0].parsed.x, T.mode === "live" || T.interval === "1h"),
        label: (it) => ` ${it.dataset.label}: ${usd(it.parsed.y)}`,
      } } },
    },
  });
  if (keepData) {
    const recs = T.records; T.records = [];
    $("#logTable tbody").innerHTML = "";
    addRecords(recs); setHistory(T.history);
  }
}

function setHistory(history) {
  T.history = history || [];
  if (!charts.liveChart) return;
  charts.liveChart.data.datasets[4].data = T.history.map((h) => ({ x: h.timestamp, y: h.price }));
  charts.liveChart.update("none");
}

function addRecords(recs) {
  if (!recs.length) return;
  const lc = charts.liveChart.data.datasets, wc = charts.walletChart.data.datasets;
  for (const r of recs) {
    T.records.push(r);
    lc[0].data.push({ x: r.timestamp, y: r.price });
    if (r.is_decision) lc[1].data.push({ x: r.forecast_for, y: r.predicted_price });
    if (r.action === "BUY") lc[2].data.push({ x: r.timestamp, y: r.price });
    if (r.action === "SELL") lc[3].data.push({ x: r.timestamp, y: r.price });
    wc[0].data.push({ x: r.timestamp, y: r.equity });
    wc[1].data.push({ x: r.timestamp, y: r.bh_equity });
  }
  charts.liveChart.update("none");
  charts.walletChart.update("none");
  renderLog(recs);
  const lastDecision = [...T.records].reverse().find((r) => r.is_decision);
  if (lastDecision) renderAiView(lastDecision, recs[recs.length - 1]);
}

function renderLog(newRecs) {
  const body = $("#logTable tbody");
  const rows = newRecs.filter((r) => r.is_decision).reverse().map((r) => `<tr>
      <td>${fmtTime(r.timestamp, T.mode === "live" || T.interval === "1h")}</td>
      <td class="n">${usd(r.price)}</td>
      <td><span class="tag ${r.action}">${r.action}</span></td>
      <td class="n">${(r.p_up * 100).toFixed(0)} / ${(r.p_flat * 100).toFixed(0)} / ${(r.p_down * 100).toFixed(0)}%</td>
      <td class="n ${signCls(r.expected_pct)}">${pct(r.expected_pct, 2)}</td>
      <td>${esc(r.reason)}</td>
      <td class="n">${usd(r.equity)}</td></tr>`).join("");
  body.insertAdjacentHTML("afterbegin", rows);
  while (body.rows.length > 400) body.deleteRow(-1);
}

function renderAiView(d, latest) {
  const cls = d.pred_class > 0 ? "UP" : d.pred_class < 0 ? "DOWN" : "FLAT";
  const bar = (name, p) => `<div class="prob"><span>${name}</span><div class="track"><div class="fill" style="width:${(p * 100).toFixed(1)}%"></div></div><span class="num">${(p * 100).toFixed(0)}%</span></div>`;
  $("#aiView").innerHTML = `
    <div class="muted">Last decision · ${fmtTime(d.timestamp, true)}</div>
    <div class="action-big"><span class="tag ${d.action}" style="font-size:20px;padding:2px 14px">${d.action}</span></div>
    <p class="reason">${esc(d.reason)}</p>
    ${bar("Up", d.p_up)}${bar("Flat", d.p_flat)}${bar("Down", d.p_down)}
    <dl>
      <dt>Model's call</dt><dd>${cls}</dd>
      <dt>Expected next move</dt><dd class="${signCls(d.expected_pct)}">${pct(d.expected_pct, 2)}</dd>
      <dt>Forecast price</dt><dd>${usd(d.predicted_price)}</dd>
      <dt>Current price</dt><dd>${usd(latest.price)}</dd>
    </dl>`;
}

function renderTradeKpis(s) {
  if (!s) return;
  $("#tradeKpis").innerHTML = [
    kpi("Wallet value", usd(s.equity), `${pct(s.pnl_pct, 2)} since start`, signCls(s.pnl_pct)),
    kpi("vs. buy-and-hold", pct(s.vs_bh_pct, 2), `buy-and-hold wallet ${usd(s.bh_equity)}`, signCls(s.vs_bh_pct)),
    kpi("Cash (USD)", usd(s.cash), `position: ${s.position}`),
    kpi("BTC held", Number(s.btc).toFixed(6), s.entry_price ? `entry ${usd(s.entry_price)}` : "no open position"),
    kpi("BTC price", usd(s.price), `${s.steps} steps`),
    kpi("Trades / win rate", `${s.n_trades} / ${s.win_rate == null ? "-" : s.win_rate.toFixed(0) + "%"}`,
        `${s.n_round_trips} closed round trips`),
    kpi("Max drawdown", pct(s.max_drawdown_pct), `fees paid ${usd(s.fees_paid)}`, s.max_drawdown_pct < 0 ? "down-txt" : ""),
  ].join("");
}

function setStatus(st) {
  const pill = $("#statusPill");
  pill.textContent = st.status;
  pill.className = "pill " + st.status;
  $("#statusMsg").textContent = st.message || "";
  const running = st.status === "running" || st.status === "starting";
  $("#startBtn").disabled = running;
  $("#stopBtn").disabled = !running;
  const showProg = st.mode === "replay" && st.total_steps;
  $("#progressWrap").hidden = !showProg;
  if (showProg) $("#progressBar").style.width = (100 * st.next_index / st.total_steps).toFixed(1) + "%";
}

let inFlight = false;
async function poll() {
  if (inFlight) return;  // never let two polls fetch the same records
  inFlight = true;
  try {
    const r = await fetch("/api/session/state?since=" + T.next);
    const st = await r.json();
    if (st.status === "idle") return;
    if (T.next === 0 && st.history && st.history.length) setHistory(st.history);
    addRecords(st.records || []);
    T.next = st.next_index ?? T.next;
    renderTradeKpis(st.summary);
    setStatus(st);
    if (["finished", "stopped", "error"].includes(st.status) && !(st.records || []).length) stopPolling();
  } catch (e) {
    $("#statusMsg").textContent = "Lost connection to the dashboard server…";
  } finally {
    inFlight = false;
  }
}
function startPolling() { stopPolling(); T.polling = setInterval(poll, 500); poll(); }
function stopPolling() { if (T.polling) clearInterval(T.polling); T.polling = null; }

/* Risk settings picked by Sharpe on the VALIDATION window (never the test window) -
   see docs/08_trading_agent.md. The user can still change any of them. */
const PRESETS = {
  "1D": { stopLoss: 5, takeProfit: "", minConf: 0, feeAware: false },
  "1h": { stopLoss: 5, takeProfit: "", minConf: 45, feeAware: false },
};
function syncControls() {
  const live = $("#mode").value === "live";
  $("#intervalField").hidden = live;
  $("#speedField").hidden = live;
  $("#decisionField").hidden = !live;
  const p = PRESETS[live ? "1h" : $("#interval").value];
  $("#stopLoss").value = p.stopLoss; $("#takeProfit").value = p.takeProfit;
  $("#minConf").value = p.minConf; $("#feeAware").checked = p.feeAware;
  $("#presetHint").textContent = "Risk settings pre-filled with the values that scored best on the validation window. " +
    (live || $("#interval").value === "1h"
      ? "Note: the hourly model's expected moves rarely cover the 0.2% round-trip fee, so with \"Fee-aware\" on it almost never trades."
      : "Leave take-profit empty to disable it.");
}
$("#mode").addEventListener("change", syncControls);
$("#interval").addEventListener("change", syncControls);
$("#speed").addEventListener("input", (e) => { $("#speedVal").textContent = e.target.value; });

$("#startBtn").addEventListener("click", async () => {
  const num = (id) => { const v = parseFloat($(id).value); return Number.isFinite(v) && v > 0 ? v : null; };
  const body = {
    mode: $("#mode").value,
    interval: $("#mode").value === "live" ? "1h" : $("#interval").value,
    speed: parseFloat($("#speed").value),
    initial_cash: num("#cash") || 1000,
    stop_loss_pct: num("#stopLoss"),
    take_profit_pct: num("#takeProfit"),
    min_confidence_pct: parseFloat($("#minConf").value) || 0,
    fee_aware: $("#feeAware").checked,
    decision_seconds: parseFloat($("#decisionSeconds").value),
  };
  T.mode = body.mode; T.interval = body.interval;
  T.records = []; T.history = []; T.next = 0;
  $("#logTable tbody").innerHTML = "";
  $("#aiView").innerHTML = `<p class="muted">Waiting for the first decision…</p>`;
  rebuildTradingCharts(false);
  setStatus({ status: "starting", message: body.mode === "live" ? "Connecting to Binance…" : "Loading model (the 1-hour model trains on first use, ~1 min)…" });
  const r = await fetch("/api/session/start", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) { setStatus({ status: "error", message: "Could not start: " + (await r.text()) }); return; }
  startPolling();
});
$("#stopBtn").addEventListener("click", async () => { await fetch("/api/session/stop", { method: "POST" }); poll(); });

/* ------------------------------------------------------------ boot */
applyChartDefaults();
syncControls();
rebuildTradingCharts(false);
renderTradeKpis({ equity: 1000, pnl_pct: 0, vs_bh_pct: 0, bh_equity: 1000, cash: 1000, position: "CASH", btc: 0,
                  entry_price: null, price: null, steps: 0, n_trades: 0, win_rate: null, n_round_trips: 0,
                  max_drawdown_pct: 0, fees_paid: 0 });
loadOverview();
// Resume a session that is already running (e.g. after a page refresh).
fetch("/api/session/state?since=0").then((r) => r.json()).then((st) => {
  if (st.status && st.status !== "idle") {
    T.mode = st.mode; T.interval = st.interval;
    rebuildTradingCharts(false);
    startPolling();
  }
}).catch(() => {});
