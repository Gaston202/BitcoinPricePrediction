"""Returns, resampling and direction labels.

Timing convention (important to avoid look-ahead leakage):
  A row indexed at open_time t describes the candle [t, t + interval).
  Its close is known only at t + interval. The label at row t is the return
  from close[t] to close[t + horizon], so features for row t may use data up
  to and including close[t], never later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

UP, FLAT, DOWN = 1, 0, -1

_OHLCV_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
    "quote_asset_volume": "sum",
    "number_of_trades": "sum",
    "taker_buy_base_volume": "sum",
    "taker_buy_quote_volume": "sum",
}


def log_returns(close: pd.Series, periods: int = 1) -> pd.Series:
    """Backward-looking log return over `periods` rows: ln(close_t / close_{t-periods})."""
    return np.log(close / close.shift(periods))


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate hourly candles to a coarser interval (UTC-aligned buckets).

    A bucket containing any unfilled missing hour is set to NaN entirely,
    so long outages never produce a fake partial candle.
    """
    hours_per_bucket = int(pd.Timedelta(rule) / pd.Timedelta(config.BASE_INTERVAL))
    grouped = df.resample(rule, label="left", closed="left")
    out = grouped.agg(_OHLCV_AGG)
    valid_hours = grouped["close"].count()
    out["filled_hours"] = grouped["filled"].sum() if "filled" in df else 0
    incomplete = valid_hours < hours_per_bucket
    out.loc[incomplete, list(_OHLCV_AGG)] = np.nan
    out["is_complete"] = ~incomplete
    return out


def forward_log_return(close: pd.Series, periods: int = 1) -> pd.Series:
    """Forward-looking log return: ln(close_{t+periods} / close_t). This is the TARGET basis."""
    return np.log(close.shift(-periods) / close)


def direction_label(fwd_return: pd.Series, flat_threshold: float) -> pd.Series:
    """3-class label: UP if r > thr, DOWN if r < -thr, FLAT otherwise; NaN if r unknown."""
    labels = pd.Series(FLAT, index=fwd_return.index, dtype="float64")
    labels[fwd_return > flat_threshold] = UP
    labels[fwd_return < -flat_threshold] = DOWN
    labels[fwd_return.isna()] = np.nan
    return labels


def taker_buy_ratio(df: pd.DataFrame) -> pd.Series:
    """Share of volume initiated by aggressive buyers (0.5 = balanced)."""
    return (df["taker_buy_base_volume"] / df["volume"]).where(df["volume"] > 0)
