/* BTC Multi-Agent Trader dashboard - plain JS + Chart.js (vendored). */
"use strict";

const $ = (s) => document.querySelector(s);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const usd = (v, d = 2) => v == null ? "-" : "$" + Number(v).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (v, d = 1) => v == null ? "-" : (v > 0 ? "+" : "") + Number(v).toFixed(d) + "%";
const share = (v) => v == null ? "-" : (v * 100).toFixed(0) + "%";
const signed = (v, d = 2) => v == null ? "-" : (v > 0 ? "+" : "") + Number(v).toFixed(d);
const signCls = (v) => v > 0 ? "up-txt" : v < 0 ? "down-txt" : "";
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmtTime = (iso, withTime) => {
  if (!iso) return "-";
  const d = new Date(iso);
  return withTime ? d.toLocaleString(undefined, { year: "numeric", month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit" })
                  : d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "2-digit" });
};
const ago = (iso) => {
  if (!iso) return "never";
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 90) return "just now";
  if (s < 5400) return Math.round(s / 60) + " min ago";
  if (s < 129600) return Math.round(s / 3600) + " h ago";
  return Math.round(s / 86400) + " days ago";
};
const ANALYSTS = { trend: "Trend", momentum: "Momentum", volatility: "Volatility", volume: "Volume / flow", sentiment: "Sentiment", quant: "Quant (ML)", news: "News (LLM)" };
const kpi = (label, value, delta = "", cls = "") =>
  `<div class="kpi"><div class="label">${label}</div><div class="value ${cls}">${value}</div><div class="delta">${delta}</div></div>`;

/* ------------------------------------------------------------ theme, tabs, charts */
const charts = {};
function C() {
  return { text: css("--text"), text2: css("--text-2"), muted: css("--muted"), grid: css("--grid"),
           s1: css("--series-1"), s2: css("--series-2"), good: css("--good"), bad: css("--critical"),
           warn: css("--warning"), surface: css("--surface") };
}
function applyChartDefaults() {
  const c = C();
  Object.assign(Chart.defaults, { color: c.text2, borderColor: c.grid, animation: false });
  Chart.defaults.font.family = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
  Chart.defaults.font.size = 12;
  Chart.defaults.plugins.legend.display = false;
  Object.assign(Chart.defaults.plugins.tooltip, { backgroundColor: c.surface, titleColor: c.text, bodyColor: c.text2,
    borderColor: c.grid, borderWidth: 1, padding: 10 });
  Chart.defaults.elements.line.borderWidth = 2;
  Chart.defaults.elements.point.radius = 0;
  Chart.defaults.elements.point.hoverRadius = 4;
}
function makeChart(id, config) {
  if (charts[id]) charts[id].destroy();
  charts[id] = new Chart(document.getElementById(id), config);
  return charts[id];
}
const timeScale = (unit) => ({ type: "time", time: unit ? { unit } : {}, grid: { display: false }, ticks: { maxRotation: 0, autoSkipPadding: 24 } });
const valueScale = (opts = {}) => ({ grid: { color: C().grid }, border: { display: false }, ...opts });
const usdTicks = { callback: (v) => "$" + Number(v).toLocaleString() };

/* Shaded bands behind a time chart (used for "new data" and rating zones). */
Chart.register({
  id: "bands",
  beforeDatasetsDraw(chart, _a, opts) {
    if (!opts || !opts.items) return;
    const { ctx, chartArea: a, scales: { x, y } } = chart;
    ctx.save();
    for (const b of opts.items) {
      ctx.fillStyle = b.color;
      if (b.from !== undefined) {
        const x0 = Math.max(x.getPixelForValue(+new Date(b.from)), a.left), x1 = Math.min(x.getPixelForValue(+new Date(b.to)), a.right);
        if (x1 > x0) { ctx.fillRect(x0, a.top, x1 - x0, a.bottom - a.top); }
        if (b.label) { ctx.fillStyle = C().muted; ctx.font = "12px system-ui"; ctx.fillText(b.label, x0 + 6, a.top + 14); }
      } else {
        const y0 = Math.max(y.getPixelForValue(b.yTo), a.top), y1 = Math.min(y.getPixelForValue(b.yFrom), a.bottom);
        ctx.fillRect(a.left, y0, a.right - a.left, y1 - y0);
      }
    }
    ctx.restore();
  },
});

$("#themeToggle").addEventListener("click", () => {
  const html = document.documentElement;
  html.dataset.theme = html.dataset.theme === "dark" ? "light" : "dark";
  try { localStorage.setItem("theme", html.dataset.theme); } catch (e) { /* storage unavailable */ }
  applyChartDefaults();
  if (overviewData) renderOverview(overviewData);
  rebuildTradingCharts(true);
});
try { const t = localStorage.getItem("theme"); if (t) document.documentElement.dataset.theme = t; } catch (e) { /* ignore */ }

document.querySelectorAll(".tab").forEach((btn) => btn.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b === btn));
  document.querySelectorAll(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === "tab-" + btn.dataset.tab));
}));

/* ------------------------------------------------------------ data sync pill */
let lastSeenCandle = null;
async function pollData() {
  try {
    const st = await (await fetch("/api/data/status")).json();
    const dot = $("#dataDot");
    dot.className = "dot " + (st.syncing ? "busy" : st.ok ? "ok" : "err");
    const newC = st.live_candles ? ` · +${st.live_candles.toLocaleString()} new candles` : "";
    $("#dataText").textContent = st.syncing ? "Syncing with Binance…"
      : !st.last_sync && st.ok ? (st.auto_sync ? "Waiting for the first Binance sync…" : "Snapshot data only - press Sync now")
      : st.ok ? `Data up to ${fmtTime(st.last_candle, true)}${newC} · synced ${ago(st.last_sync)}`
              : `Sync failed (${ago(st.last_attempt)}) - using stored data`;
    if (!st.last_sync && st.ok) dot.className = "dot";
    $("#dataPill").title = st.ok ? `Auto-sync every ${Math.round(st.every_seconds / 60)} min` : (st.error || "");
    $("#syncBtn").disabled = !!st.syncing;
    if (lastSeenCandle && st.last_candle && st.last_candle !== lastSeenCandle) setTimeout(loadOverview, 4000);
    lastSeenCandle = st.last_candle;
  } catch (e) { $("#dataText").textContent = "Dashboard server unreachable"; }
}
$("#syncBtn").addEventListener("click", async () => {
  $("#dataDot").className = "dot busy";
  $("#dataText").textContent = "Syncing with Binance…";
  await fetch("/api/data/sync", { method: "POST" });
  setTimeout(pollData, 1500);
});

/* ------------------------------------------------------------ overview */
let overviewData = null;
async function loadOverview() {
  try {
    const d = await (await fetch("/api/overview")).json();
    if (d.status === "loading") return setTimeout(loadOverview, 1500);
    if (d.status === "error") { $("#overviewLoading").textContent = "Could not build the overview: " + d.error; return; }
    overviewData = d;
    $("#overviewLoading").hidden = true;
    $("#overviewBody").hidden = false;
    renderOverview(d);
    renderFigures(d.figures);
  } catch (e) { setTimeout(loadOverview, 2000); }
}

function ratingBadge(r) { return `<span class="rating ${esc(r)}">${esc(r)}</span>`; }

function renderOverview(d) {
  const c = C(), data = d.data;
  const test4 = d.backtests.find((b) => b.interval === "4h" && b.period.startsWith("Test"));
  const now = d.agent_now;
  $("#overviewKpis").innerHTML = [
    kpi("BTC price (last closed hour)", usd(data.last_price, 0), `${pct(data.change_24h_pct, 2)} in 24 h`, signCls(data.change_24h_pct)),
    kpi("Data coverage", (data.hourly_rows || 0).toLocaleString() + " h", `${fmtTime(data.first)} → ${fmtTime(data.last, true)}`),
    kpi("New since the snapshot", "+" + (data.new_candles || 0).toLocaleString(), `snapshot ends ${fmtTime(data.snapshot_end)}`),
    kpi("Agents now (4h)", now ? ratingBadge(now.plan.rating) : "-", now ? `conviction ${signed(now.plan.conviction)}` : ""),
    test4 ? kpi("Agents on test window", pct(test4.agent_pct), `buy-and-hold ${pct(test4.bh_pct)}`, signCls(test4.agent_pct - test4.bh_pct)) : "",
    test4 ? kpi("Max drawdown (test)", pct(test4.max_dd_pct), `${test4.trades} trades, fees ${usd(test4.fees, 0)}`, "down-txt") : "",
  ].join("");

  const beat = d.backtests.filter((b) => b.agent_pct > b.bh_pct).length;
  const profitable = d.backtests.filter((b) => b.agent_pct > 0).length;
  $("#verdict").innerHTML = `<strong>Honest read:</strong> the rule-based team beat buy-and-hold in ${beat} of ${d.backtests.length} backtests below
    and made money in ${profitable}. Its edge is mostly defensive: it cuts exposure when the debate turns bearish, which limits drawdowns in falling markets,
    and it lags in strong rallies. Fast candles (1h and below) lose to fees. This is a learning project, not financial advice.`;

  const first = d.price[0].t, last = d.price[d.price.length - 1].t;
  makeChart("priceHistory", {
    type: "line",
    data: { datasets: [{ data: d.price.map((p) => ({ x: p.t, y: p.v })), borderColor: c.s1 }] },
    options: { maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
      scales: { x: timeScale("year"), y: valueScale({ type: "logarithmic", ticks: { callback: (v) => {
        const n = Number(v); return [1000, 3000, 10000, 30000, 100000].includes(n) ? "$" + n.toLocaleString() : ""; } } }) },
      plugins: { tooltip: { callbacks: { title: (it) => fmtTime(it[0].parsed.x), label: (it) => " Close " + usd(it.parsed.y, 0) } },
        bands: { items: [{ from: data.snapshot_end, to: last > data.snapshot_end ? last : data.snapshot_end, color: "rgba(127,127,127,0.18)",
          label: data.new_candles ? "new data" : "" }] } } },
  });

  if (d.main_backtest) {
    makeChart("mainBacktest", {
      type: "line",
      data: { datasets: [
        { label: "Agent wallet", data: d.main_backtest.equity.map((p) => ({ x: p.t, y: p.v })), borderColor: c.s1 },
        { label: "Buy-and-hold", data: d.main_backtest.bh.map((p) => ({ x: p.t, y: p.v })), borderColor: c.s2 },
      ] },
      options: { maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
        scales: { x: timeScale("month"), y: valueScale({ ticks: usdTicks }) },
        plugins: { tooltip: { callbacks: { title: (it) => fmtTime(it[0].parsed.x, true), label: (it) => ` ${it.dataset.label}: ${usd(it.parsed.y)}` } } } },
    });
  }

  $("#backtestTable").innerHTML = `<thead><tr><th>Period</th><th>Candle</th><th>Agents</th><th>Buy &amp; hold</th><th>Sharpe</th><th>Max DD</th><th>Trades</th></tr></thead><tbody>` +
    d.backtests.map((b) => `<tr><td>${esc(b.period)}<div class="muted">${fmtTime(b.from)} → ${fmtTime(b.to)}</div></td><td>${b.interval}</td>
      <td class="n ${signCls(b.agent_pct)}">${pct(b.agent_pct)}</td><td class="n ${signCls(b.bh_pct)}">${pct(b.bh_pct)}</td>
      <td class="n">${b.agent_sharpe == null ? "-" : b.agent_sharpe.toFixed(2)} <span class="muted">vs ${b.bh_sharpe == null ? "-" : b.bh_sharpe.toFixed(2)}</span></td>
      <td class="n">${pct(b.max_dd_pct)}</td><td class="n">${b.trades}</td></tr>`).join("") + "</tbody>";

  $("#nowTime").textContent = now ? `decision at ${fmtTime(now.timestamp, true)} UTC candle close` : "";
  $("#agentNow").innerHTML = now ? `
    <div class="now-head"><span class="big">${ratingBadge(now.plan.rating)}</span>
      <span class="muted">${esc(now.decision.reasons.slice(-1)[0] || "")}</span></div>
    ${now.reports.map(analystRow).join("")}
    <p class="foot">${esc(now.plan.rationale)}</p>` : `<p class="muted">Not enough data.</p>`;

  const fg = d.fear_greed || [];
  $("#fgEmpty").hidden = fg.length > 0;
  $("#fearGreed").parentElement.hidden = fg.length === 0;
  makeChart("fearGreed", {
    type: "line",
    data: { datasets: [{ data: fg.slice(-730).map((p) => ({ x: p.t, y: p.v })), borderColor: c.s1 }] },
    options: { maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
      scales: { x: timeScale("month"), y: valueScale({ min: 0, max: 100 }) },
      plugins: { tooltip: { callbacks: { title: (it) => fmtTime(it[0].parsed.x), label: (it) => ` Fear & Greed: ${it.parsed.y}` } },
        bands: { items: [{ yFrom: 0, yTo: 25, color: "rgba(208,59,59,0.10)" }, { yFrom: 75, yTo: 100, color: "rgba(12,163,12,0.10)" }] } } },
  });

  makeChart("volatility", {
    type: "line",
    data: { datasets: [{ data: d.volatility.map((p) => ({ x: p.t, y: p.v })), borderColor: c.s1 }] },
    options: { maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
      scales: { x: timeScale("year"), y: valueScale({ ticks: { callback: (v) => v + "%" } }) },
      plugins: { tooltip: { callbacks: { title: (it) => fmtTime(it[0].parsed.x), label: (it) => ` ${it.parsed.y.toFixed(1)}% annualized` } } } },
  });

  const tm = d.team_model || {};
  const auc = tm.walk_forward_cal_sign_auc || {};
  $("#teamModelHint").textContent = tm.model ? `${tm.model} · ${tm.horizon} candles · ${tm.target} target · chosen ${tm.decided}` : "";
  $("#teamModelFeatures").textContent = tm.features ? "Features: " + tm.features.join(", ") + ". Sign-AUC 0.5 = coin flip." : "";
  makeChart("teamModel", {
    type: "bar",
    data: { labels: Object.keys(auc).map((k) => k.replace("_", " ")), datasets: [{ data: Object.values(auc), backgroundColor: c.s1, borderRadius: 4, borderSkipped: "start" }] },
    options: { maintainAspectRatio: false, scales: { x: { grid: { display: false } }, y: valueScale({ min: 0.45, max: 0.6 }) },
      plugins: { tooltip: { callbacks: { label: (it) => ` Sign-AUC ${it.parsed.y.toFixed(3)}` } },
        bands: { items: [{ yFrom: 0.45, yTo: 0.5, color: "rgba(127,127,127,0.15)" }] } } },
  });
}

/* ------------------------------------------------------------ figures */
function renderFigures(figs) {
  const groups = {};
  figs.forEach((f) => (groups[f.section] = groups[f.section] || []).push(f));
  $("#gallery").innerHTML = Object.entries(groups).map(([sec, items]) => `<div class="gallery-section"><h3>${esc(sec)}</h3><div class="gallery">` +
    items.map((f) => `<figure data-src="/figures/${encodeURIComponent(f.file)}" data-cap="${esc(f.caption)}">
      <img loading="lazy" src="/figures/${encodeURIComponent(f.file)}" alt="${esc(f.caption)}">
      <figcaption><b>${esc(f.file)}</b>${esc(f.caption)}</figcaption></figure>`).join("") + "</div></div>").join("");
  document.querySelectorAll(".gallery figure").forEach((fig) => fig.addEventListener("click", () => {
    const lb = $("#lightbox");
    lb.querySelector("img").src = fig.dataset.src;
    lb.querySelector("p").textContent = fig.dataset.cap;
    lb.hidden = false;
  }));
}
$("#lightbox").addEventListener("click", () => { $("#lightbox").hidden = true; });

/* ------------------------------------------------------------ pipeline rendering */
function scorebar(score) {
  const w = Math.min(Math.abs(score), 1) * 50;
  return `<div class="scorebar"><span class="${score >= 0 ? "pos" : "neg"}" style="width:${w}%"></span></div>`;
}
function analystRow(r) {
  const abst = r.confidence === 0;
  return `<div class="analyst">
    <div class="row"><span class="name">${esc(ANALYSTS[r.analyst] || r.title)}</span>${scorebar(r.score)}<span class="num ${signCls(r.score)}">${signed(r.score)}</span></div>
    <div class="muted-row">${abst ? "abstains" : `confidence ${(r.confidence * 100).toFixed(0)}% · weight ×${r.weight.toFixed(2)}`} - ${esc(r.summary)}</div>
    ${r.evidence.length ? `<details><summary>${r.evidence.length} argument(s)</summary><ul>${r.evidence.map((e) =>
      `<li class="${e.direction > 0 ? "ev-pos" : "ev-neg"}">${e.direction > 0 ? "▲" : "▼"} <span style="color:var(--text-2)">${esc(e.text)}</span>
       <span class="muted">(strength ${e.strength.toFixed(2)}${e.discounted < 1 ? `, rebutted ×${e.discounted.toFixed(2)}` : ""})</span></li>`).join("")}</ul></details>` : ""}
  </div>`;
}
function renderPipeline(s) {
  if (!s) return;
  const p = s.plan, tp = s.proposal, d = s.decision;
  const tot = p.bull_strength + p.bear_strength || 1;
  const lastRound = Math.max(...s.risk_views.map((v) => v.round));
  $("#pipeline").innerHTML = `
    <div class="stage"><h3><span>1 · Analyst team</span><span class="step">${s.reports.length} analysts</span></h3>${s.reports.map(analystRow).join("")}</div>
    <div class="stage"><h3><span>2 · Bull vs. bear debate</span><span class="step">${Math.max(...p.debate.map((t) => t.round))} round(s)</span></h3>
      ${p.debate.map((t) => `<div class="turn ${t.speaker.startsWith("Bull") ? "bull" : "bear"}"><b>${esc(t.speaker)} · round ${t.round}</b>${esc(t.text)}</div>`).join("")}
      <div class="verdict-box"><div>Research manager: bull ${p.bull_strength.toFixed(2)} vs bear ${p.bear_strength.toFixed(2)}</div>
        <div class="balance"><div class="b" style="width:${(p.bull_strength / tot * 100).toFixed(1)}%"></div><div class="r" style="width:${(p.bear_strength / tot * 100).toFixed(1)}%"></div></div>
        <div class="big">${ratingBadge(p.rating)}</div>${esc(p.rationale)}</div></div>
    <div class="stage"><h3><span>3 · Trader</span></h3>
      <dl class="kv"><dt>Proposal</dt><dd><b>${esc(tp.action)}</b></dd><dt>Target BTC share</dt><dd>${share(tp.target_exposure)}</dd>
        <dt>Current share</dt><dd>${share(tp.current_exposure)}</dd><dt>Entry</dt><dd>${usd(tp.entry_price, 0)}</dd>
        <dt>Stop-loss</dt><dd>${tp.stop_loss ? usd(tp.stop_loss, 0) : "-"}</dd><dt>Take-profit</dt><dd>${tp.take_profit ? usd(tp.take_profit, 0) : "-"}</dd></dl>
      <p>${esc(tp.reasoning)}</p></div>
    <div class="stage"><h3><span>4 · Risk team</span><span class="step">${lastRound} round(s)</span></h3>
      ${s.risk_views.map((v) => `<div class="riskview"><b>${esc(v.speaker)}${lastRound > 1 ? ` · r${v.round}` : ""}: ${share(v.target_exposure)}</b><br>${esc(v.argument)}</div>`).join("")}</div>
    <div class="stage"><h3><span>5 · Portfolio manager</span></h3>
      <div class="final"><span class="tag ${d.action}">${d.action}</span>${ratingBadge(d.rating)}</div>
      <dl class="kv"><dt>BTC share</dt><dd>${share(d.previous_exposure)} → ${share(d.target_exposure)}</dd><dt>Approved</dt><dd>${d.approved ? "yes" : "no change"}</dd></dl>
      <ul class="reasons">${d.reasons.map((r) => `<li>${esc(r)}</li>`).join("")}</ul></div>`;
}

/* ------------------------------------------------------------ trading charts */
const T = { records: [], history: [], next: 0, polling: null, mode: "replay", interval: "4h", selected: null };
const withTime = () => T.mode === "live" || T.interval !== "1D";

function rebuildTradingCharts(keepData) {
  const c = C();
  const unit = T.mode === "live" ? undefined : (T.interval === "1D" ? "month" : undefined);
  const tip = { title: (it) => fmtTime(it[0].parsed.x, withTime()), label: (it) => ` ${it.dataset.label}: ${usd(it.parsed.y)}` };
  makeChart("liveChart", {
    type: "line",
    data: { datasets: [
      { label: "BTC price", data: [], borderColor: c.s1, order: 3 },
      { label: "Stop-loss", data: [], borderColor: c.warn, borderDash: [5, 4], borderWidth: 1.5, spanGaps: false, order: 4 },
      { label: "Buy", type: "scatter", data: [], pointStyle: "triangle", pointRadius: 8, pointHoverRadius: 10, backgroundColor: c.good, borderColor: c.surface, borderWidth: 2, order: 1 },
      { label: "Sell", type: "scatter", data: [], pointStyle: "triangle", rotation: 180, pointRadius: 8, pointHoverRadius: 10, backgroundColor: c.bad, borderColor: c.surface, borderWidth: 2, order: 2 },
      { label: "Before start", data: [], borderColor: c.muted, borderWidth: 1.5, order: 5 },
    ] },
    options: { maintainAspectRatio: false, interaction: { mode: "nearest", axis: "x", intersect: false },
      scales: { x: timeScale(unit), y: valueScale({ ticks: usdTicks }) }, plugins: { tooltip: { callbacks: tip } } },
  });
  makeChart("walletChart", {
    type: "line",
    data: { datasets: [{ label: "Agent team", data: [], borderColor: c.s1 }, { label: "Buy-and-hold", data: [], borderColor: c.s2 }] },
    options: { maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
      scales: { x: timeScale(unit), y: valueScale({ ticks: usdTicks }) }, plugins: { tooltip: { callbacks: tip } } },
  });
  makeChart("convictionChart", {
    type: "line",
    data: { datasets: [{ label: "Smoothed", data: [], borderColor: c.s1 }, { label: "Per candle", data: [], borderColor: c.muted, borderWidth: 1 }] },
    options: { maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
      scales: { x: timeScale(unit), y: valueScale({ min: -1, max: 1 }) },
      plugins: { tooltip: { callbacks: { title: (it) => fmtTime(it[0].parsed.x, withTime()), label: (it) => ` ${it.dataset.label}: ${signed(it.parsed.y)}` } },
        bands: { items: [{ yFrom: 0.4, yTo: 1, color: "rgba(12,163,12,0.10)" }, { yFrom: 0.15, yTo: 0.4, color: "rgba(12,163,12,0.05)" },
          { yFrom: -0.4, yTo: -0.15, color: "rgba(208,59,59,0.05)" }, { yFrom: -1, yTo: -0.4, color: "rgba(208,59,59,0.10)" }] } } },
  });
  makeChart("exposureChart", {
    type: "line",
    data: { datasets: [{ label: "BTC share", data: [], borderColor: c.s1, stepped: true, fill: { target: "origin" }, backgroundColor: "rgba(57,135,229,0.15)" }] },
    options: { maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
      scales: { x: timeScale(unit), y: valueScale({ min: 0, max: 1, ticks: { callback: (v) => v * 100 + "%" } }) },
      plugins: { tooltip: { callbacks: { title: (it) => fmtTime(it[0].parsed.x, withTime()), label: (it) => ` BTC share ${share(it.parsed.y)}` } } } },
  });
  makeChart("memoryChart", {
    type: "bar",
    data: { labels: [], datasets: [{ data: [], backgroundColor: c.s1, borderRadius: 4, borderSkipped: "start" }] },
    options: { indexAxis: "y", maintainAspectRatio: false,
      scales: { x: valueScale({ min: 0.3, max: 0.7, ticks: { callback: (v) => (v * 100).toFixed(0) + "%" } }), y: { grid: { display: false } } },
      plugins: { tooltip: { callbacks: { label: (it) => ` hit rate ${(it.parsed.x * 100).toFixed(0)}%` } } } },
  });
  if (keepData) {
    const recs = T.records; T.records = [];
    $("#logTable tbody").innerHTML = "";
    addRecords(recs); setHistory(T.history);
  }
}

function setHistory(h) {
  T.history = h || [];
  if (!charts.liveChart) return;
  charts.liveChart.data.datasets[4].data = T.history.map((p) => ({ x: p.timestamp, y: p.price }));
  charts.liveChart.update("none");
}

function addRecords(recs) {
  if (!recs.length) return;
  const lc = charts.liveChart.data.datasets, wc = charts.walletChart.data.datasets;
  const cc = charts.convictionChart.data.datasets, ec = charts.exposureChart.data.datasets[0];
  for (const r of recs) {
    T.records.push(r);
    lc[0].data.push({ x: r.timestamp, y: r.price });
    lc[1].data.push({ x: r.timestamp, y: r.stop_loss ?? null });
    if (r.action === "BUY") lc[2].data.push({ x: r.timestamp, y: r.price });
    if (r.action === "SELL") lc[3].data.push({ x: r.timestamp, y: r.price });
    wc[0].data.push({ x: r.timestamp, y: r.equity });
    wc[1].data.push({ x: r.timestamp, y: r.bh_equity });
    ec.data.push({ x: r.timestamp, y: r.exposure });
    if (r.is_decision) {
      cc[0].data.push({ x: r.timestamp, y: r.smoothed });
      cc[1].data.push({ x: r.timestamp, y: r.conviction });
    }
  }
  for (const id of ["liveChart", "walletChart", "convictionChart", "exposureChart"]) charts[id].update("none");
  renderLog(recs);
}

function renderLog(recs) {
  const body = $("#logTable tbody");
  const rows = recs.filter((r) => r.is_decision || r.action !== "HOLD").reverse().map((r) => `<tr data-i="${r.i}">
      <td>${fmtTime(r.timestamp, withTime())}</td><td class="n">${usd(r.price, 0)}</td>
      <td>${r.rating ? ratingBadge(r.rating) : '<span class="muted">intrabar</span>'}</td>
      <td class="n ${signCls(r.smoothed)}">${r.is_decision ? signed(r.smoothed) : ""}</td>
      <td><span class="tag ${r.action}">${r.action}</span></td><td class="n">${share(r.exposure)}</td>
      <td class="n">${usd(r.equity)}</td><td>${esc(r.reason)}</td></tr>`).join("");
  body.insertAdjacentHTML("afterbegin", rows);
  while (body.rows.length > 500) body.deleteRow(-1);
}
$("#logTable tbody").addEventListener("click", async (e) => {
  const tr = e.target.closest("tr");
  if (!tr) return;
  const r = await fetch("/api/session/step/" + tr.dataset.i);
  if (!r.ok) { $("#pipelineHint").textContent = "That decision is no longer stored (only the latest 600 are kept)."; return; }
  document.querySelectorAll("#logTable tr.selected").forEach((x) => x.classList.remove("selected"));
  tr.classList.add("selected");
  T.selected = Number(tr.dataset.i);
  const s = await r.json();
  renderPipeline(s);
  $("#pipelineHint").innerHTML = `Showing the decision of ${fmtTime(s.timestamp, true)}. <a href="#" id="backLatest">Back to the latest</a>`;
  $("#backLatest").addEventListener("click", (ev) => { ev.preventDefault(); T.selected = null; tr.classList.remove("selected"); poll(true); });
  $("#pipeline").scrollIntoView({ behavior: "smooth", block: "start" });
});

function renderMemory(summary) {
  const hr = summary.hit_rates || {};
  const keys = Object.keys(hr).filter((k) => hr[k].calls >= 1).sort((a, b) => hr[b].hit_rate - hr[a].hit_rate);
  const ch = charts.memoryChart;
  ch.data.labels = keys.map((k) => `${ANALYSTS[k] || k} (${hr[k].hits}/${hr[k].calls})`);
  ch.data.datasets[0].data = keys.map((k) => hr[k].hit_rate);
  ch.update("none");
  $("#lessons").innerHTML = (summary.lessons || []).map((l) => `<li>${esc(l)}</li>`).join("") ||
    `<li class="muted">Lessons appear once decisions start to settle.</li>`;
}

function renderTradeKpis(s) {
  if (!s) return;
  $("#tradeKpis").innerHTML = [
    kpi("Wallet value", usd(s.equity), `${pct(s.pnl_pct, 2)} since start`, signCls(s.pnl_pct)),
    kpi("vs. buy-and-hold", pct(s.vs_bh_pct, 2), `buy-and-hold ${usd(s.bh_equity)}`, signCls(s.vs_bh_pct)),
    kpi("BTC share", share(s.exposure), `${Number(s.btc || 0).toFixed(5)} BTC · ${usd(s.cash)} cash`),
    kpi("BTC price", usd(s.price), s.avg_entry ? `avg entry ${usd(s.avg_entry, 0)}` : "no open position"),
    kpi("Stop-loss", s.stop_loss ? usd(s.stop_loss, 0) : "-", s.take_profit ? `take-profit ${usd(s.take_profit, 0)}` : ""),
    kpi("Trades / fees", `${s.n_trades}`, `fees paid ${usd(s.fees_paid)}`),
    kpi("Max drawdown", pct(s.max_drawdown_pct), s.rating_accuracy == null ? "" : `ratings right ${(s.rating_accuracy * 100).toFixed(0)}%`, s.max_drawdown_pct < 0 ? "down-txt" : ""),
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
  $("#notes").innerHTML = (st.notes || []).map((n) => `<div>${esc(n)}</div>`).join("");
}

let inFlight = false;
async function poll(forcePipeline) {
  if (inFlight) return;
  inFlight = true;
  try {
    const st = await (await fetch("/api/session/state?since=" + T.next)).json();
    if (st.status === "idle") return;
    if (T.next === 0 && st.history && st.history.length) setHistory(st.history);
    addRecords(st.records || []);
    T.next = st.next_index ?? T.next;
    renderTradeKpis(st.summary);
    renderMemory(st.summary);
    if (st.latest_state && (T.selected === null || forcePipeline === true)) renderPipeline(st.latest_state);
    if (forcePipeline === true && !st.latest_state) {
      const r = await fetch("/api/session/step/" + st.latest_state_index);
      if (r.ok) renderPipeline(await r.json());
      $("#pipelineHint").textContent = "The latest decision. Click any row of the decision log to inspect it.";
    }
    setStatus(st);
    if (["finished", "stopped", "error"].includes(st.status) && !(st.records || []).length) stopPolling();
  } catch (e) {
    $("#statusMsg").textContent = "Lost connection to the dashboard server…";
  } finally { inFlight = false; }
}
function startPolling() { stopPolling(); T.polling = setInterval(poll, 600); poll(); }
function stopPolling() { if (T.polling) clearInterval(T.polling); T.polling = null; }

/* ------------------------------------------------------------ controls */
const INTERVALS = { replay: [["4h", "4 hours (recommended)"], ["1D", "1 day"], ["1h", "1 hour (+ quant model)"]],
                    live: [["1m", "1 minute"], ["5m", "5 minutes"], ["15m", "15 minutes"], ["1h", "1 hour (+ quant model)"], ["4h", "4 hours"]] };
$("#analystChecks").innerHTML = Object.entries(ANALYSTS).map(([k, v]) =>
  `<label><input type="checkbox" value="${k}" checked> ${v}</label>`).join("");
function syncControls() {
  const mode = $("#mode").value, prev = $("#interval").value;
  $("#interval").innerHTML = INTERVALS[mode].map(([v, l]) => `<option value="${v}">${l}</option>`).join("");
  if (INTERVALS[mode].some(([v]) => v === prev)) $("#interval").value = prev;
  else $("#interval").value = mode === "live" ? "5m" : "4h";
  $("#periodField").hidden = mode === "live";
  $("#speedField").hidden = mode === "live";
  const iv = $("#interval").value;
  $("#presetHint").textContent = (mode === "live"
    ? "Live: the price updates every 3 s and the whole agent team decides each time a candle closes. "
    : "Replay: the agents trade stored Binance candles one by one, including every candle synced since the snapshot. ")
    + (["1m", "5m", "15m", "1h"].includes(iv) ? "Short candles flip often - the cooldown and smoothing settings keep fees in check, but in backtests 1h candles still lost to fees." : "")
    + (iv !== "1h" ? " The quant analyst abstains (the team's ML model is trained on 1h candles)." : "");
}
$("#mode").addEventListener("change", syncControls);
$("#interval").addEventListener("change", syncControls);
$("#speed").addEventListener("input", (e) => { $("#speedVal").textContent = e.target.value; });

$("#startBtn").addEventListener("click", async () => {
  const analysts = [...document.querySelectorAll("#analystChecks input:checked")].map((x) => x.value);
  if (!analysts.length) { setStatus({ status: "error", message: "Pick at least one analyst." }); return; }
  const body = {
    mode: $("#mode").value, interval: $("#interval").value, period: $("#period").value,
    speed: parseFloat($("#speed").value), initial_cash: parseFloat($("#cash").value) || 1000,
    risk_profile: $("#riskProfile").value, analysts,
    max_debate_rounds: parseInt($("#debateRounds").value, 10), max_risk_rounds: parseInt($("#riskRounds").value, 10),
  };
  T.mode = body.mode; T.interval = body.interval; T.records = []; T.history = []; T.next = 0; T.selected = null;
  $("#logTable tbody").innerHTML = "";
  $("#pipeline").innerHTML = `<p class="muted">Waiting for the first decision…</p>`;
  rebuildTradingCharts(false);
  setStatus({ status: "starting", message: body.mode === "live" ? "Connecting to Binance…" : "Preparing candles and indicators…" });
  const r = await fetch("/api/session/start", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) { setStatus({ status: "error", message: "Could not start: " + (await r.text()) }); return; }
  startPolling();
});
$("#stopBtn").addEventListener("click", async () => { await fetch("/api/session/stop", { method: "POST" }); poll(); });

/* ------------------------------------------------------------ boot */
applyChartDefaults();
syncControls();
rebuildTradingCharts(false);
renderTradeKpis({ equity: 1000, pnl_pct: 0, vs_bh_pct: 0, bh_equity: 1000, exposure: 0, btc: 0, cash: 1000, price: null,
                  n_trades: 0, fees_paid: 0, max_drawdown_pct: 0 });
loadOverview();
pollData();
setInterval(pollData, 15000);
fetch("/api/session/state?since=0").then((r) => r.json()).then((st) => {
  if (st.status && st.status !== "idle") { T.mode = st.mode; T.interval = st.interval; rebuildTradingCharts(false); startPolling(); }
}).catch(() => {});
