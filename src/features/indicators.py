"""Technical indicators that the reviewed papers commonly use (docs/09_literature_review.md, item 7).

They exist for the feature-set comparison in notebooks/05: the main runs use the 12 train-selected
features, and a second run adds these to measure whether the "standard" indicator set helps.

Two rules, both covered by tests/test_indicators.py:
- No look-ahead: the value at row t uses candles up to and including t only. The label is
  built from close[t+1], so even same-candle indicators like BOP are legitimate here. The
  leakage in Jung (2023) and Oktem (2025) came from labeling the *same* candle.
- Scale-free: anything in price units is divided by close, because BTC traded at ~4k and
  ~100k in different years and a price-level feature would mostly encode the year.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.engineering import FLAT_THRESHOLD, LABEL_HORIZON, build_features

MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
STOCH_WINDOW, STOCH_SMOOTH = 14, 3
BOLLINGER_WINDOW, BOLLINGER_STD = 20, 2.0
ROC_WINDOW = 10
WILLIAMS_WINDOW = 14
CCI_WINDOW = 20
CCI_CONSTANT = 0.015  # Lambert's constant: scales ~70-80% of CCI values into [-100, 100]
OBV_WINDOW = 20
MFI_WINDOW = 14

PAPER_INDICATOR_COLUMNS = [
    "macd", "macd_signal", "macd_hist",
    "stoch_k", "stoch_d",
    "bb_pct_b", "bb_width",
    "roc_10",
    "williams_r",
    "cci_20",
    "obv_slope",
    "mfi_14",
    "bop",
]


def macd(close: pd.Series) -> pd.DataFrame:
    """MACD line, signal and histogram, divided by close to make them scale-free."""
    line = close.ewm(span=MACD_FAST, adjust=False).mean() - close.ewm(span=MACD_SLOW, adjust=False).mean()
    signal = line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    return pd.DataFrame({"macd": line / close, "macd_signal": signal / close,
                         "macd_hist": (line - signal) / close})


def _rolling_range(df: pd.DataFrame, window: int) -> tuple[pd.Series, pd.Series]:
    return df["high"].rolling(window).max(), df["low"].rolling(window).min()


def stochastic(df: pd.DataFrame, window: int = STOCH_WINDOW,
               smooth: int = STOCH_SMOOTH) -> tuple[pd.Series, pd.Series]:
    """%K = where close sits in the window's high-low range (0-100); %D = its moving average."""
    high, low = _rolling_range(df, window)
    k = 100 * (df["close"] - low) / (high - low)
    return k, k.rolling(smooth).mean()


def williams_r(df: pd.DataFrame, window: int = WILLIAMS_WINDOW) -> pd.Series:
    """Williams %R in [-100, 0]: the stochastic %K measured from the top of the range."""
    high, low = _rolling_range(df, window)
    return -100 * (high - df["close"]) / (high - low)


def bollinger(close: pd.Series, window: int = BOLLINGER_WINDOW,
              n_std: float = BOLLINGER_STD) -> tuple[pd.Series, pd.Series]:
    """%B (position inside the bands: 0 = lower, 1 = upper) and band width relative to the mean."""
    mean = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper, lower = mean + n_std * std, mean - n_std * std
    return (close - lower) / (upper - lower), (upper - lower) / mean


def commodity_channel_index(df: pd.DataFrame, window: int = CCI_WINDOW) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3
    mean = typical.rolling(window).mean()
    mean_dev = typical.rolling(window).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (typical - mean) / (CCI_CONSTANT * mean_dev)


def obv_slope(df: pd.DataFrame, window: int = OBV_WINDOW) -> pd.Series:
    """Net signed volume over the window as a share of total volume, in [-1, 1].

    Raw OBV is a running total that grows without bound and depends on where the series
    starts, so we use its change over the window, normalized by volume traded in that window.
    """
    signed = np.sign(df["close"].diff()).fillna(0) * df["volume"]
    return signed.rolling(window).sum() / df["volume"].rolling(window).sum()


def money_flow_index(df: pd.DataFrame, window: int = MFI_WINDOW) -> pd.Series:
    """Volume-weighted RSI on the typical price, in [0, 100]."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    flow = typical * df["volume"]
    direction = np.sign(typical.diff())
    positive = flow.where(direction > 0, 0.0).rolling(window).sum()
    negative = flow.where(direction < 0, 0.0).rolling(window).sum()
    return 100 * positive / (positive + negative)


def balance_of_power(df: pd.DataFrame) -> pd.Series:
    """(close - open) / (high - low) of the candle itself. NaN when the candle has no range."""
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    return (df["close"] - df["open"]) / rng


def paper_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """All indicators in PAPER_INDICATOR_COLUMNS order for one OHLCV frame."""
    close = df["close"]
    k, d = stochastic(df)
    pct_b, width = bollinger(close)
    out = pd.concat([
        macd(close),
        pd.DataFrame({
            "stoch_k": k, "stoch_d": d,
            "bb_pct_b": pct_b, "bb_width": width,
            "roc_10": close.pct_change(ROC_WINDOW),
            "williams_r": williams_r(df),
            "cci_20": commodity_channel_index(df),
            "obv_slope": obv_slope(df),
            "mfi_14": money_flow_index(df),
            "bop": balance_of_power(df),
        }),
    ], axis=1)
    return out[PAPER_INDICATOR_COLUMNS]


def build_features_with_indicators(df: pd.DataFrame, horizon: int = LABEL_HORIZON,
                                   flat_threshold: float = FLAT_THRESHOLD) -> pd.DataFrame:
    """build_features() plus the paper indicators, with the same NaN/inf row dropping."""
    base = build_features(df, horizon=horizon, flat_threshold=flat_threshold)
    combined = base.join(paper_indicators(df), how="left")
    return combined.replace([np.inf, -np.inf], np.nan).dropna()
