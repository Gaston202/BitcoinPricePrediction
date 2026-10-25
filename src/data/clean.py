"""Data-quality checks and the gap policy (CRISP-DM Data Understanding -> Preparation).

Gap policy (decided in Business/Data Understanding):
  1. Reindex to a complete hourly grid so "next row" always means "next hour".
  2. Flag every inserted row with is_missing=True.
  3. Gaps of <= MAX_FFILL_GAP_HOURS consecutive candles: O=H=L=C = previous close,
     volume columns = 0 ("no trading happened"), filled=True.
  4. Longer gaps stay NaN and are excluded downstream.
"""

from __future__ import annotations

import pandas as pd

from src import config


def quality_report(df: pd.DataFrame, freq: str = config.BASE_INTERVAL) -> dict:
    """Summarize completeness, uniqueness and validity of a raw kline frame."""
    idx = df.index
    expected = pd.date_range(idx.min(), idx.max(), freq=freq)
    gaps = find_gaps(df, freq)
    ohlc_violations = (
        (df["high"] < df[["open", "close"]].max(axis=1))
        | (df["low"] > df[["open", "close"]].min(axis=1))
        | (df["high"] < df["low"])
    )
    return {
        "rows": len(df),
        "expected_rows": len(expected),
        "missing_rows": len(expected.difference(idx)),
        "completeness_pct": round(100 * idx.nunique() / len(expected), 4),
        "duplicate_timestamps": int(idx.duplicated().sum()),
        "null_values": int(df.isna().sum().sum()),
        "non_positive_prices": int((df[config.PRICE_COLUMNS] <= 0).sum().sum()),
        "ohlc_violations": int(ohlc_violations.sum()),
        "zero_volume_candles": int((df["volume"] == 0).sum()),
        "n_gaps": len(gaps),
        "longest_gap_hours": int(gaps["missing_candles"].max()) if len(gaps) else 0,
    }


def find_gaps(df: pd.DataFrame, freq: str = config.BASE_INTERVAL) -> pd.DataFrame:
    """List each run of missing candles with its first missing timestamp and length."""
    step = pd.Timedelta(freq)
    deltas = df.index.to_series().diff()
    after_gap = deltas[deltas > step]
    return pd.DataFrame({
        "gap_start": after_gap.index - after_gap + step,
        "gap_end": after_gap.index - step,
        "missing_candles": (after_gap / step - 1).astype(int).to_numpy(),
    }).reset_index(drop=True)


def _run_lengths(mask: pd.Series) -> pd.Series:
    """For each True in mask, the length of the consecutive True-run it belongs to."""
    run_id = (mask != mask.shift()).cumsum()
    lengths = mask.groupby(run_id).transform("size")
    return lengths.where(mask, 0)


def fill_gaps(df: pd.DataFrame, max_gap: int = config.MAX_FFILL_GAP_HOURS,
              freq: str = config.BASE_INTERVAL) -> pd.DataFrame:
    """Apply the gap policy. Returns a new frame on a complete time grid."""
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="first")]
    grid = pd.date_range(df.index.min(), df.index.max(), freq=freq, name=df.index.name)
    out = df.reindex(grid)
    out["is_missing"] = out["close"].isna()
    short_gap = out["is_missing"] & (_run_lengths(out["is_missing"]) <= max_gap)

    prev_close = out["close"].ffill()
    for col in config.PRICE_COLUMNS:
        out.loc[short_gap, col] = prev_close[short_gap]
    for col in config.VOLUME_COLUMNS:
        out.loc[short_gap, col] = 0
    out.loc[short_gap, "close_time"] = (
        out.index[short_gap] + pd.Timedelta(freq) - pd.Timedelta(milliseconds=1)
    )
    out["filled"] = short_gap
    return out


def robust_zscore(series: pd.Series) -> pd.Series:
    """Distance from the median in units of scaled MAD (≈ std for normal data, robust to fat tails)."""
    median = series.median()
    mad = (series - median).abs().median() * 1.4826
    return (series - median) / mad


def flag_outliers(series: pd.Series, threshold: float = config.OUTLIER_Z) -> pd.Series:
    """True where |robust z-score| exceeds `threshold`; NaN inputs are never flagged."""
    return robust_zscore(series).abs() > threshold
