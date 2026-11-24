"""Indicator table the analysts read: one row per candle, every value causal (uses data up to close[t]).

Built from the project's own indicator functions (src.features.engineering / indicators),
computed once for a whole series so a replay over thousands of candles stays fast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.engineering import average_true_range, rsi
from src.features.indicators import bollinger, macd, money_flow_index, obv_slope, stochastic
from src.features.returns import log_returns, taker_buy_ratio

WARMUP_BARS = 210  # SMA200 + margin: rows before this are not used for decisions


def compute_indicators(bars: pd.DataFrame) -> pd.DataFrame:
    """bars: OHLCV(+taker volume) candles of any interval, indexed by open time."""
    close = bars["close"]
    ret = log_returns(close)
    out = pd.DataFrame(index=bars.index)
    out["close"] = close
    out["ret_1"] = ret
    out["ema20"] = close.ewm(span=20, adjust=False).mean()
    out["ema50"] = close.ewm(span=50, adjust=False).mean()
    out["sma200"] = close.rolling(200).mean()
    out = out.join(macd(close))
    out["macd_hist_prev"] = out["macd_hist"].shift(1)
    out["rsi14"] = rsi(close, 14)
    out["rsi14_prev"] = out["rsi14"].shift(1)
    k, d = stochastic(bars)
    out["stoch_k"], out["stoch_d"] = k, d
    out["roc10"] = close.pct_change(10)
    pct_b, width = bollinger(close)
    out["bb_pct_b"], out["bb_width"] = pct_b, width
    out["bb_width_pctile"] = width.rolling(120, min_periods=60).rank(pct=True)
    out["atr"] = average_true_range(bars)
    out["atr_pct"] = out["atr"] / close
    vol7, vol30 = ret.rolling(7).std(), ret.rolling(30).std()
    out["vol_ratio"] = vol7 / vol30
    log_vol = np.log1p(bars["volume"])
    out["volume_z"] = (log_vol - log_vol.rolling(20).mean()) / log_vol.rolling(20).std()
    out["taker_buy_ratio"] = taker_buy_ratio(bars).rolling(3).mean()
    out["obv_slope"] = obv_slope(bars)
    out["mfi14"] = money_flow_index(bars)
    out["candle_dir"] = np.sign(close - bars["open"])
    out["high_20"] = bars["high"].rolling(20).max().shift(1)
    out["low_20"] = bars["low"].rolling(20).min().shift(1)
    return out.replace([np.inf, -np.inf], np.nan)


def attach_sentiment(ind: pd.DataFrame, funding: pd.Series | None = None,
                     fear_greed: pd.Series | None = None, long_short: pd.Series | None = None) -> pd.DataFrame:
    """As-of join of slower sentiment series onto the candles (only values already published)."""
    out = ind.copy()
    for name, series in (("funding_rate", funding), ("fear_greed", fear_greed), ("long_short_ratio", long_short)):
        if series is None or len(series) == 0:
            out[name] = np.nan
            continue
        s = series.sort_index()
        s = s[~s.index.duplicated(keep="last")]
        # value known at candle close: as-of the candle's open + 1 bar is approximated by its index
        out[name] = s.reindex(out.index.union(s.index)).ffill().reindex(out.index)
    if "funding_rate" in out:
        out["funding_avg_3"] = (funding.sort_index().rolling(3).mean()
                                .reindex(out.index.union(funding.index)).ffill().reindex(out.index)
                                if funding is not None and len(funding) else np.nan)
    return out


NEWS_SMOOTHING_DAYS = 3


def attach_news(ind: pd.DataFrame, sentiment: pd.DataFrame | None) -> pd.DataFrame:
    """As-of join of the daily news score onto the candles, without look-ahead.

    The headlines of day D are complete only at the end of D, so the score of D becomes
    usable at D+1 00:00 UTC. A candle opening at t gets the latest score usable at t.
    Adds news_score, news_score_3d (3-day average), news_confidence, news_count, news_reason.
    """
    out = ind.copy()
    cols = ["news_score", "news_score_3d", "news_confidence", "news_count", "news_reason"]
    if sentiment is None or len(sentiment) == 0:
        for c in cols:
            out[c] = np.nan
        return out
    daily = sentiment.copy()
    daily["day"] = pd.to_datetime(daily["day"], utc=True)
    daily = daily.sort_values("day").drop_duplicates("day", keep="last").set_index("day")
    # A day without any scored headline (confidence 0) carries no information.
    score = daily["score"].where(daily["confidence"] > 0)
    table = pd.DataFrame({
        "news_score": score,
        # Calendar window: days without headlines have no row, so a 3-row window could span a week.
        "news_score_3d": score.rolling(f"{NEWS_SMOOTHING_DAYS}D", min_periods=1).mean(),
        "news_confidence": daily["confidence"],
        "news_count": daily["n_headlines"].astype(float),
        "news_reason": daily["reason"],
    })
    table.index = table.index + pd.Timedelta("1D")  # usable from the next midnight
    # As-of join: each candle takes the latest table row as it is. Unlike ffill() on the union,
    # this keeps the NaN of an uninformative day instead of filling it with an older score.
    joined = table.reindex(out.index, method="ffill")
    # Do not carry a stale score across a long news outage (more than 3 days).
    age = pd.Series(table.index, index=table.index).reindex(out.index, method="ffill")
    stale = (out.index.to_series() - age) > pd.Timedelta("3D")
    joined.loc[stale.to_numpy()] = np.nan
    for c in cols:
        out[c] = joined[c]
    return out
