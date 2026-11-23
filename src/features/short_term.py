"""Short-term candidate features for the multi-horizon iteration (decided 2026-11-21).

Three families, all scale-free and causal (the value at bar t uses data up to close[t] only):

- fast_indicators: short-lookback versions of daily indicators (2-8 bars).
- candle_features: the shape of the bar itself (body, wicks, where the close sits in the range)
  plus the current up/down streak.
- intraday_features: statistics of the 1h candles *inside* and just before each bar
  (realized volatility, last-hours return, hourly volume spikes). For a bar opening at t
  with length B hours, the last hourly candle in it opens at t + B - 1h and closes exactly
  when the bar closes, so features are read from that hourly row (no look-ahead).

Same-candle features are legitimate because every label in this project uses close[t+1]
or later (see src.features.targets). The leakage in Jung (2023) / Oktem (2025) came from
labeling the *same* candle (docs/09_literature_review.md, item 9).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.engineering import rsi
from src.features.indicators import bollinger, stochastic
from src.features.returns import log_returns

FAST_RSI_WINDOWS = (3, 5)
FAST_MOM_WINDOW = 3
FAST_BB_WINDOW = 5
FAST_STOCH_WINDOW = 5
EMA_FAST, EMA_SLOW = 3, 8
RV_WINDOWS_H = (4, 24, 168)  # hours: last 4h, last day, last week
SPIKE_WINDOW_H = 24
HOURS_PER_DAY = 24


def fast_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    out = {f"rsi_{w}": rsi(close, w) for w in FAST_RSI_WINDOWS}
    out[f"mom_{FAST_MOM_WINDOW}"] = log_returns(close, FAST_MOM_WINDOW)
    out[f"bb{FAST_BB_WINDOW}_pct_b"] = bollinger(close, FAST_BB_WINDOW)[0]
    out[f"stoch{FAST_STOCH_WINDOW}_k"] = stochastic(df, FAST_STOCH_WINDOW)[0]
    fast = close.ewm(span=EMA_FAST, adjust=False).mean()
    slow = close.ewm(span=EMA_SLOW, adjust=False).mean()
    out[f"ema{EMA_FAST}_{EMA_SLOW}_cross"] = fast / slow - 1
    return pd.DataFrame(out, index=df.index)


def up_down_streak(close: pd.Series) -> pd.Series:
    """Signed length of the current run of up (+) or down (-) closes; 0 after an unchanged close."""
    direction = np.sign(close.diff()).fillna(0)
    run_id = (direction != direction.shift()).cumsum()
    return (direction.groupby(run_id).cumcount() + 1) * direction


def candle_features(df: pd.DataFrame) -> pd.DataFrame:
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    top = df[["open", "close"]].max(axis=1)
    bottom = df[["open", "close"]].min(axis=1)
    return pd.DataFrame({
        "body_ratio": (df["close"] - df["open"]) / rng,
        "upper_wick": (df["high"] - top) / rng,
        "lower_wick": (bottom - df["low"]) / rng,
        "close_location": (df["close"] - df["low"]) / rng,
        "range_pct": (df["high"] - df["low"]) / df["close"],
        "streak": up_down_streak(df["close"]),
    }, index=df.index)


def hourly_statistics(hourly: pd.DataFrame) -> pd.DataFrame:
    """Rolling statistics on the 1h series, indexed by hourly open time (value known at its close)."""
    r = log_returns(hourly["close"])
    out = {f"rv_{w}h": np.sqrt((r ** 2).rolling(w).sum()) for w in RV_WINDOWS_H}
    out["rv_ratio_day_week"] = out["rv_24h"] / (out["rv_168h"] / np.sqrt(7))
    out["ret_last_1h"] = r
    out["ret_last_4h"] = r.rolling(4).sum()
    vol = hourly["volume"]
    out["hour_volume_spike"] = vol.rolling(SPIKE_WINDOW_H).max() / vol.rolling(SPIKE_WINDOW_H).mean()
    out["up_hours_share"] = (r > 0).astype(float).where(r.notna()).rolling(SPIKE_WINDOW_H).mean()
    abs_r = r.abs()
    out["max_hour_share"] = abs_r.rolling(SPIKE_WINDOW_H).max() / abs_r.rolling(SPIKE_WINDOW_H).sum()
    return pd.DataFrame(out, index=hourly.index)


def intraday_features(hourly: pd.DataFrame, bar_index: pd.DatetimeIndex, bar_hours: int) -> pd.DataFrame:
    """Hourly statistics read at the last hourly candle of each bar (closes when the bar closes)."""
    stats = hourly_statistics(hourly)
    last_hour = bar_index + pd.Timedelta(hours=bar_hours - 1)
    out = stats.reindex(last_hour)
    out.index = bar_index
    return out


def hour_of_day(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Cyclical hour of the bar's open (only informative for bars shorter than a day)."""
    angle = 2 * np.pi * index.hour / HOURS_PER_DAY
    return pd.DataFrame({"hour_sin": np.sin(angle), "hour_cos": np.cos(angle)}, index=index)


def short_term_features(bars: pd.DataFrame, hourly: pd.DataFrame, bar_hours: int) -> pd.DataFrame:
    parts = [fast_indicators(bars), candle_features(bars),
             intraday_features(hourly, bars.index, bar_hours)]
    if bar_hours < HOURS_PER_DAY:
        parts.append(hour_of_day(bars.index))
    return pd.concat(parts, axis=1)


BAR_HOURS = {"1h": 1, "4h": 4, "1D": 24}


def candidate_pool(bars: pd.DataFrame, hourly: pd.DataFrame, bar_hours: int) -> pd.DataFrame:
    """Every candidate feature for one bar size: the original 1D set, the 13 paper indicators and the
    short-term families. Targets are added per walk-forward window (src.features.targets), not here."""
    from src.features.indicators import build_features_with_indicators  # local: avoids an import cycle

    base = build_features_with_indicators(bars).drop(columns="label")
    out = base.join(short_term_features(bars, hourly, bar_hours).reindex(base.index))
    return out.replace([np.inf, -np.inf], np.nan).dropna()
