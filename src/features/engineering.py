"""Feature matrix and train/val/test split for the 1D direction model
(CRISP-DM Data Preparation, following EDA item 11 in notebooks/03_eda.ipynb).

Timing convention: every feature for row t uses data up to and including
close[t] (see src.features.returns). The label still looks forward, so a
plain positional split would leak the tail of train into val/test through
overlapping label windows; chronological_split() purges those rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.returns import direction_label, forward_log_return, log_returns, taker_buy_ratio

LAG_PERIODS = (1, 2, 3, 5, 7, 14)
VOL_WINDOWS = (7, 14, 30)

# Team-recommended label (EDA item 11): 1D horizon, theta = 1%.
LABEL_HORIZON = 1
FLAT_THRESHOLD = 0.01


def lagged_returns(close: pd.Series, lags: tuple[int, ...] = LAG_PERIODS) -> pd.DataFrame:
    """Backward-looking log returns at each lag, named ret_lag{n}."""
    return pd.DataFrame({f"ret_lag{n}": log_returns(close, n) for n in lags})


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


def average_true_range(df: pd.DataFrame, window: int = 14) -> pd.Series:
    return true_range(df).rolling(window).mean()


def volume_features(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Log volume, its rolling z-score ("volume surprise"), and taker-buy ratio."""
    log_volume = np.log1p(df["volume"])
    roll_mean = log_volume.rolling(window).mean()
    roll_std = log_volume.rolling(window).std()
    return pd.DataFrame({
        "log_volume": log_volume,
        "volume_z": (log_volume - roll_mean) / roll_std,
        "taker_buy_ratio": taker_buy_ratio(df),
    })


def build_features(df: pd.DataFrame, horizon: int = LABEL_HORIZON,
                    flat_threshold: float = FLAT_THRESHOLD) -> pd.DataFrame:
    """Assemble the feature matrix + label for one clean OHLCV frame (e.g. 1D candles).

    Rows with any NaN (warm-up window, trailing horizon, unfilled data gaps)
    are dropped, so every remaining row is model-ready.
    """
    ret_1 = log_returns(df["close"])
    features = pd.concat([
        lagged_returns(df["close"]),
        rolling_volatility(ret_1),
        pd.DataFrame({"atr_14": average_true_range(df) / df["close"]}),
        volume_features(df),
    ], axis=1)
    fwd = forward_log_return(df["close"], horizon)
    features["label"] = direction_label(fwd, flat_threshold)
    return features.dropna()


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
