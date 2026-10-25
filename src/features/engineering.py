"""Candidate feature matrix and train/val/test split for the 1D direction model
(CRISP-DM Data Preparation).

This module only *creates* candidate features (mathematical transformations,
domain-knowledge ratios, date/time and interaction features).
Deciding which ones the model actually uses is feature selection, which lives in
src.features.selection and must run on the train split only.

Timing convention: every feature for row t uses data up to and including
close[t] (see src.features.returns). The label still looks forward, so a
plain positional split would leak the tail of train into val/test through
overlapping label windows; chronological_split() purges those rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.returns import direction_label, forward_log_return, log_returns, taker_buy_ratio

LAG_PERIODS = (1, 2, 3)
MOMENTUM_WINDOWS = (7, 14, 30)
VOL_WINDOWS = (7, 14, 30)
MA_WINDOWS = (20, 50)
ATR_WINDOW = 14
RSI_WINDOW = 14
VOLUME_Z_WINDOW = 14
VOLUME_RATIO_WINDOW = 30
WEEKEND_DAYS = (5, 6)  # pandas dayofweek: Monday = 0 ... Sunday = 6
DAYS_PER_WEEK = 7

# Team-recommended label (EDA item 11): 1D horizon, theta = 1%.
LABEL_HORIZON = 1
FLAT_THRESHOLD = 0.01


def lagged_returns(close: pd.Series, lags: tuple[int, ...] = LAG_PERIODS) -> pd.DataFrame:
    """True lags of the 1-period log return: ret_lag{n} is the return of the candle n-1 rows back.

    ret_lag1 is the most recent completed return (close[t-1] -> close[t]), ret_lag2 the
    one before it, and so on. Unlike multi-period returns these do not overlap, so they
    carry independent information (multi-period trends are covered by momentum()).
    """
    ret_1 = log_returns(close)
    return pd.DataFrame({f"ret_lag{n}": ret_1.shift(n - 1) for n in lags})


def momentum(close: pd.Series, windows: tuple[int, ...] = MOMENTUM_WINDOWS) -> pd.DataFrame:
    """Cumulative log return over the last n rows (trend strength), named mom_{n}d."""
    return pd.DataFrame({f"mom_{n}d": log_returns(close, n) for n in windows})


def rolling_volatility(returns: pd.Series, windows: tuple[int, ...] = VOL_WINDOWS) -> pd.DataFrame:
    """Rolling standard deviation of 1-period log returns (backward-looking)."""
    return pd.DataFrame({f"vol_{w}d": returns.rolling(w).std() for w in windows})


def true_range(df: pd.DataFrame) -> pd.Series:
    """Largest of today's high-low range and the gap from yesterday's close."""
    prev_close = df["close"].shift(1)
    ranges = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1)
    return ranges.max(axis=1)


def average_true_range(df: pd.DataFrame, window: int = ATR_WINDOW) -> pd.Series:
    return true_range(df).rolling(window).mean()


def rsi(close: pd.Series, window: int = RSI_WINDOW) -> pd.Series:
    """Wilder's Relative Strength Index in [0, 100] (exponential smoothing with alpha = 1/window)."""
    delta = close.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    return 100 - 100 / (1 + avg_gain / avg_loss)


def volume_features(df: pd.DataFrame, z_window: int = VOLUME_Z_WINDOW,
                    ratio_window: int = VOLUME_RATIO_WINDOW) -> pd.DataFrame:
    """Relative volume measures plus the taker-buy ratio.

    The raw (log) volume level is deliberately not a feature: it trends upward over
    the years, so levels learned on 2017-2024 never reappear in later data. Only
    measures relative to recent history (z-score, ratio to the rolling mean) transfer.
    """
    log_volume = np.log1p(df["volume"])
    roll_mean = log_volume.rolling(z_window).mean()
    roll_std = log_volume.rolling(z_window).std()
    return pd.DataFrame({
        "volume_z": (log_volume - roll_mean) / roll_std,
        "volume_ratio": df["volume"] / df["volume"].rolling(ratio_window).mean(),
        "taker_buy_ratio": taker_buy_ratio(df),
    })


def trend_features(close: pd.Series, ma_windows: tuple[int, ...] = MA_WINDOWS) -> pd.DataFrame:
    """Distance of the close from its moving averages (close / MA - 1) and the RSI."""
    out = pd.DataFrame({f"dist_ma{w}": close / close.rolling(w).mean() - 1 for w in ma_windows})
    out[f"rsi_{RSI_WINDOW}"] = rsi(close)
    return out


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Day of week as a cyclical (sin, cos) pair plus a weekend flag.

    A plain 0-6 integer would tell the model that Sunday (6) is far from Monday (0);
    the sin/cos encoding places the days on a circle so adjacent days stay adjacent.
    """
    dow = index.dayofweek
    angle = 2 * np.pi * dow / DAYS_PER_WEEK
    return pd.DataFrame({
        "dow_sin": np.sin(angle),
        "dow_cos": np.cos(angle),
        "is_weekend": dow.isin(WEEKEND_DAYS).astype(int),
    }, index=index)


def interaction_features(features: pd.DataFrame) -> pd.DataFrame:
    """Products of existing features that express a joint effect neither shows alone.

    ret1_x_volz: yesterday's move weighted by how unusual its volume was.
    mom7_x_volratio: the 7-day trend weighted by whether volatility is rising or falling.
    """
    return pd.DataFrame({
        "ret1_x_volz": features["ret_lag1"] * features["volume_z"],
        "mom7_x_volratio": features["mom_7d"] * features["vol_ratio"],
    })


def build_features(df: pd.DataFrame, horizon: int = LABEL_HORIZON,
                   flat_threshold: float = FLAT_THRESHOLD) -> pd.DataFrame:
    """Assemble every candidate feature + the label for one clean OHLCV frame (e.g. 1D candles).

    Rows with any NaN or infinite value (warm-up window, trailing horizon,
    unfilled data gaps, division by a zero rolling std) are dropped, so every
    remaining row is model-ready.
    """
    close = df["close"]
    ret_1 = log_returns(close)
    vol = rolling_volatility(ret_1)
    features = pd.concat([
        lagged_returns(close),
        momentum(close),
        vol,
        pd.DataFrame({
            f"atr_{ATR_WINDOW}": average_true_range(df) / close,
            "vol_ratio": vol["vol_7d"] / vol["vol_30d"],
        }),
        trend_features(close),
        volume_features(df),
        calendar_features(df.index),
    ], axis=1)
    features = pd.concat([features, interaction_features(features)], axis=1)
    fwd = forward_log_return(close, horizon)
    features["label"] = direction_label(fwd, flat_threshold)
    return features.replace([np.inf, -np.inf], np.nan).dropna()


def chronological_split(
    df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15, embargo: int = LABEL_HORIZON,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time-ordered frame into train/val/test by position, never shuffled.

    The label at row t depends on close[t + horizon], so the last `embargo`
    rows of train and of val are dropped (purged): without this, their label
    would be computed from a price that falls inside the following split,
    leaking outcome information across the boundary.
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    train = df.iloc[: max(train_end - embargo, 0)]
    val = df.iloc[train_end: max(val_end - embargo, train_end)]
    test = df.iloc[val_end:]
    return train, val, test
