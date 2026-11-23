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
