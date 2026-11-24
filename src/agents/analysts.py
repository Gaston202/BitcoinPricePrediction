"""Analyst team: each analyst reads one family of data and writes a scored report.

TradingAgents has market, sentiment, news and fundamentals analysts backed by an LLM.
Here every analyst is an explicit rule. The only LLM in the system sits *upstream* of the
news analyst: it turns each day's headlines into a cached score (src/agents/news_sentiment.py),
and the news analyst reads that score like any other indicator.

  trend       - moving-average alignment, long-term trend, MACD
  momentum    - RSI, stochastic, rate of change
  volatility  - Bollinger bands, squeezes and breakouts
  volume      - taker-buy pressure, on-balance volume, money flow, volume confirmation
  sentiment   - futures funding rate, crowd long/short ratio, Fear & Greed (contrarian)
  quant       - the project's LightGBM model (P(next candle up)), when it is available
  news        - LLM (or lexicon) sentiment of yesterday's Bitcoin headlines

Each evidence item is (text, direction +1/-1, strength 0-1, tag). The analyst's score is
the strength-weighted vote; its confidence is how much and how consistent the evidence is.
"""

from __future__ import annotations

import math

import numpy as np

from src.agents.state import AnalystReport, Evidence


def _ok(*values) -> bool:
    return all(v is not None and not (isinstance(v, float) and math.isnan(v)) for v in values)


def _report(analyst: str, title: str, evidence: list[Evidence], metrics: dict,
            max_strength: float, abstain_reason: str = "") -> AnalystReport:
    for e in evidence:
        e.analyst = analyst
    if not evidence:
        return AnalystReport(analyst, title, 0.0, 0.0,
                             abstain_reason or "No clear signal - stays neutral.", [], metrics)
    bull = sum(e.strength for e in evidence if e.direction > 0)
    bear = sum(e.strength for e in evidence if e.direction < 0)
    total = bull + bear
    score = float(np.clip((bull - bear) / max_strength, -1, 1))
    agreement = abs(bull - bear) / total if total else 0.0
    confidence = float(np.clip(0.35 + 0.65 * agreement * min(1.0, total / max_strength), 0, 1))
    lean = "bullish" if score > 0.1 else "bearish" if score < -0.1 else "neutral"
    strongest = max(evidence, key=lambda e: e.strength)
    summary = f"{lean.capitalize()} ({score:+.2f}). Main point: {strongest.text}"
    return AnalystReport(analyst, title, score, confidence, summary, evidence, metrics)


def trend_analyst(row: dict) -> AnalystReport:
    ev: list[Evidence] = []
    c, e20, e50, s200 = row.get("close"), row.get("ema20"), row.get("ema50"), row.get("sma200")
    atr = row.get("atr_pct")
    if _ok(e20, e50, atr) and atr > 0:
        gap = (e20 / e50 - 1) / atr  # separation of the averages in ATR units
        strength = float(np.clip(0.25 + 0.15 * abs(gap), 0.25, 0.6))
        if e20 > e50:
            ev.append(Evidence(f"EMA20 is above EMA50 ({gap:+.1f} ATR): short-term uptrend", 1, strength, "trend_up"))
        else:
            ev.append(Evidence(f"EMA20 is below EMA50 ({gap:+.1f} ATR): short-term downtrend", -1, strength, "trend_down"))
    if _ok(c, s200):
        d = c / s200 - 1
        if d > 0:
            ev.append(Evidence(f"Price is {d:+.1%} above the 200-candle average: long-term uptrend", 1, 0.3, "long_up"))
        else:
            ev.append(Evidence(f"Price is {d:+.1%} below the 200-candle average: long-term downtrend", -1, 0.3, "long_down"))
    h, hp = row.get("macd_hist"), row.get("macd_hist_prev")
    if _ok(h, hp):
        if h > 0 and h > hp:
            ev.append(Evidence("MACD histogram positive and rising: momentum accelerating up", 1, 0.3, "macd_up"))
        elif h < 0 and h < hp:
            ev.append(Evidence("MACD histogram negative and falling: momentum accelerating down", -1, 0.3, "macd_down"))
        elif h > 0:
            ev.append(Evidence("MACD histogram positive but fading", 1, 0.1, "macd_up"))
        elif h < 0:
            ev.append(Evidence("MACD histogram negative but recovering", -1, 0.1, "macd_down"))
    return _report("trend", "Trend analyst", ev, {"ema20": e20, "ema50": e50, "sma200": s200, "macd_hist": h}, 1.1)


def momentum_analyst(row: dict) -> AnalystReport:
    ev: list[Evidence] = []
    r, rp = row.get("rsi14"), row.get("rsi14_prev")
    if _ok(r, rp):
        if r < 30:
            ev.append(Evidence(f"RSI {r:.0f} is oversold (< 30): bounce likely", 1, 0.45, "oversold"))
        elif r > 70:
            ev.append(Evidence(f"RSI {r:.0f} is overbought (> 70): pullback likely", -1, 0.45, "overbought"))
        elif r > 50 and r > rp:
            ev.append(Evidence(f"RSI {r:.0f} above 50 and rising: buyers in control", 1, 0.2, "rsi_up"))
        elif r < 50 and r < rp:
            ev.append(Evidence(f"RSI {r:.0f} below 50 and falling: sellers in control", -1, 0.2, "rsi_down"))
    k, d = row.get("stoch_k"), row.get("stoch_d")
    if _ok(k, d):
        if k < 20 and k > d:
            ev.append(Evidence(f"Stochastic %K {k:.0f} crossed above %D in the oversold zone", 1, 0.3, "oversold"))
        elif k > 80 and k < d:
            ev.append(Evidence(f"Stochastic %K {k:.0f} crossed below %D in the overbought zone", -1, 0.3, "overbought"))
    roc = row.get("roc10")
    if _ok(roc) and abs(roc) > 0.002:
        ev.append(Evidence(f"Price {roc:+.1%} over the last 10 candles", 1 if roc > 0 else -1,
                           float(np.clip(abs(roc) * 8, 0.1, 0.3)), "roc_up" if roc > 0 else "roc_down"))
    return _report("momentum", "Momentum analyst", ev, {"rsi14": r, "stoch_k": k, "stoch_d": d, "roc10": roc}, 1.0)


def volatility_analyst(row: dict) -> AnalystReport:
    ev: list[Evidence] = []
    b, wp, vr = row.get("bb_pct_b"), row.get("bb_width_pctile"), row.get("vol_ratio")
    c, hi, lo = row.get("close"), row.get("high_20"), row.get("low_20")
    squeeze = _ok(wp) and wp < 0.2
    if _ok(c, hi) and c > hi:
        ev.append(Evidence(f"Breakout above the 20-candle high{' after a squeeze' if squeeze else ''}", 1,
                           0.45 if squeeze else 0.3, "breakout_up"))
    elif _ok(c, lo) and c < lo:
        ev.append(Evidence(f"Breakdown below the 20-candle low{' after a squeeze' if squeeze else ''}", -1,
                           0.45 if squeeze else 0.3, "breakout_down"))
    elif _ok(b):
        if b < 0:
            ev.append(Evidence(f"Close below the lower Bollinger band (%B {b:.2f}): stretched down", 1, 0.3, "oversold"))
        elif b > 1:
            ev.append(Evidence(f"Close above the upper Bollinger band (%B {b:.2f}): stretched up", -1, 0.3, "overbought"))
    turbulent = _ok(vr) and vr > 1.5
    summary_extra = f" Volatility is {'high' if turbulent else 'normal'} (7/30 ratio {vr:.2f})." if _ok(vr) else ""
    rep = _report("volatility", "Volatility analyst", ev,
                  {"bb_pct_b": b, "bb_width_pctile": wp, "vol_ratio": vr, "atr_pct": row.get("atr_pct"),
                   "turbulent": bool(turbulent)}, 0.8)
    rep.summary += summary_extra
    if turbulent:
        rep.confidence *= 0.7  # signals are noisier in turbulent markets
    return rep


def volume_analyst(row: dict) -> AnalystReport:
    ev: list[Evidence] = []
    t, obv, mfi = row.get("taker_buy_ratio"), row.get("obv_slope"), row.get("mfi14")
    vz, cd = row.get("volume_z"), row.get("candle_dir")
    if _ok(t):
        if t > 0.53:
            ev.append(Evidence(f"Aggressive buyers take {t:.0%} of volume", 1, float(np.clip((t - 0.5) * 8, 0.1, 0.4)), "flow_up"))
        elif t < 0.47:
            ev.append(Evidence(f"Aggressive sellers take {1 - t:.0%} of volume", -1, float(np.clip((0.5 - t) * 8, 0.1, 0.4)), "flow_down"))
    if _ok(obv) and abs(obv) > 0.1:
        ev.append(Evidence(f"On-balance volume {'accumulation' if obv > 0 else 'distribution'} ({obv:+.2f})",
                           1 if obv > 0 else -1, float(np.clip(abs(obv), 0.1, 0.35)), "obv"))
    if _ok(mfi):
        if mfi < 20:
            ev.append(Evidence(f"Money Flow Index {mfi:.0f}: money flowing out is exhausted", 1, 0.25, "oversold"))
        elif mfi > 80:
            ev.append(Evidence(f"Money Flow Index {mfi:.0f}: buying looks exhausted", -1, 0.25, "overbought"))
    if _ok(vz, cd) and vz > 1.5 and cd != 0:
        ev.append(Evidence(f"Volume spike ({vz:+.1f} sigma) confirms the {'up' if cd > 0 else 'down'} candle",
                           int(cd), 0.3, "volume_confirm"))
    return _report("volume", "Volume / order-flow analyst", ev,
                   {"taker_buy_ratio": t, "obv_slope": obv, "mfi14": mfi, "volume_z": vz}, 1.0)


def sentiment_analyst(row: dict) -> AnalystReport:
    """Contrarian on crowd extremes: crowded longs pay high funding and tend to get flushed."""
    ev: list[Evidence] = []
    f, fg, ls = row.get("funding_avg_3", row.get("funding_rate")), row.get("fear_greed"), row.get("long_short_ratio")
    if _ok(f):
        if f > 0.0003:
            ev.append(Evidence(f"Funding {f:.3%}/8h is high: longs are crowded", -1, float(np.clip(f / 0.001, 0.2, 0.45)), "crowded_long"))
        elif f < 0:
            ev.append(Evidence(f"Funding {f:.3%}/8h is negative: shorts are crowded", 1, float(np.clip(-f / 0.0005, 0.2, 0.45)), "crowded_short"))
    if _ok(fg):
        if fg <= 25:
            ev.append(Evidence(f"Fear & Greed {fg:.0f} (extreme fear): contrarian buy zone", 1, 0.35, "fear"))
        elif fg >= 75:
            ev.append(Evidence(f"Fear & Greed {fg:.0f} (extreme greed): contrarian caution", -1, 0.35, "greed"))
    if _ok(ls):
        if ls > 2.5:
            ev.append(Evidence(f"{ls:.1f} retail accounts long per short: one-sided crowd", -1, 0.25, "crowded_long"))
        elif ls < 0.9:
            ev.append(Evidence(f"Retail is net short ({ls:.2f} long/short)", 1, 0.25, "crowded_short"))
    has_data = any(_ok(v) for v in (f, fg, ls))
    rep = _report("sentiment", "Sentiment analyst", ev, {"funding_rate": f, "fear_greed": fg, "long_short_ratio": ls},
                  0.9, "" if has_data else "No sentiment data for this period - abstains.")
    if not has_data:
        rep.confidence = 0.0
    return rep


def quant_analyst(row: dict) -> AnalystReport:
    """Wraps the project's selected ML model (LightGBM on 1h candles, src/modeling/selected_model.json)."""
    p = row.get("p_up")
    if not _ok(p):
        return AnalystReport("quant", "Quant analyst (ML model)", 0.0, 0.0,
                             "Model not available for this candle size - abstains.", [], {"p_up": None})
    edge = p - 0.5
    ev = []
    if abs(edge) >= 0.02:
        ev.append(Evidence(f"LightGBM gives P(next candle up) = {p:.1%}", 1 if edge > 0 else -1,
                           float(np.clip(abs(edge) * 6, 0.1, 0.6)), "model"))
    rep = _report("quant", "Quant analyst (ML model)", ev, {"p_up": p}, 0.6,
                  f"P(up) = {p:.1%} is too close to a coin flip - no vote.")
    rep.confidence = float(np.clip(abs(edge) * 5, 0.05, 0.8)) if ev else 0.1
    return rep


NEWS_MIN_SCORE = 0.15  # |3-day news score| below this is treated as no news signal


def news_analyst(row: dict) -> AnalystReport:
    """Reads the daily news-sentiment score (3-day average, already lagged one day)."""
    s, c, n = row.get("news_score_3d", row.get("news_score")), row.get("news_confidence"), row.get("news_count")
    metrics = {"news_score": s, "news_confidence": c, "news_headlines": n, "news_reason": row.get("news_reason")}
    if not _ok(s):
        rep = AnalystReport("news", "News analyst (LLM sentiment)", 0.0, 0.0,
                            "No news score for this period - abstains.", [], metrics)
        return rep
    ev = []
    if abs(s) >= NEWS_MIN_SCORE:
        why = f": {row['news_reason']}" if row.get("news_reason") else ""
        ev.append(Evidence(f"News sentiment {s:+.2f} over the last 3 days{why}", 1 if s > 0 else -1,
                           float(np.clip(abs(s), 0.1, 0.5)), "news_up" if s > 0 else "news_down"))
    rep = _report("news", "News analyst (LLM sentiment)", ev, metrics, 0.5,
                  f"News sentiment {s:+.2f} is close to neutral - no vote.")
    if ev:
        rep.confidence = float(np.clip(c if _ok(c) else 0.5, 0.1, 0.9))
    return rep


ANALYSTS = {
    "trend": trend_analyst,
    "momentum": momentum_analyst,
    "volatility": volatility_analyst,
    "volume": volume_analyst,
    "sentiment": sentiment_analyst,
    "quant": quant_analyst,
    "news": news_analyst,
}
