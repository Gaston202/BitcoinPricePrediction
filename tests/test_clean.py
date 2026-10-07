import numpy as np
import pandas as pd
import pytest

from src import config
from src.data.clean import fill_gaps, find_gaps, quality_report


def make_candles(hours: list[int]) -> pd.DataFrame:
    """Hourly candles at the given hour offsets; close = 100 + offset."""
    idx = pd.Timestamp("2020-01-01", tz="UTC") + pd.to_timedelta(hours, unit="h")
    close = 100.0 + np.array(hours, dtype=float)
    df = pd.DataFrame({
        "open": close, "high": close + 1, "low": close - 1, "close": close,
        "close_time": idx + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1),
        **{col: 1.0 for col in config.VOLUME_COLUMNS},
    }, index=pd.DatetimeIndex(idx, name="open_time"))
    return df


def test_find_gaps_reports_start_and_length():
    df = make_candles([0, 1, 4, 5])  # hours 2 and 3 missing
    gaps = find_gaps(df)
    assert len(gaps) == 1
    assert gaps.loc[0, "missing_candles"] == 2
    assert gaps.loc[0, "gap_start"] == df.index[0] + pd.Timedelta(hours=2)
    assert gaps.loc[0, "gap_end"] == df.index[0] + pd.Timedelta(hours=3)


def test_short_gap_is_forward_filled_with_zero_volume():
    df = make_candles([0, 1, 4, 5])
    out = fill_gaps(df, max_gap=3)
    assert len(out) == 6
    filled = out[out["filled"]]
    assert len(filled) == 2
    assert (filled[config.PRICE_COLUMNS] == 101.0).all().all()  # previous close
    assert (filled[config.VOLUME_COLUMNS] == 0).all().all()


def test_long_gap_stays_nan():
    df = make_candles([0, 1, 6, 7])  # 4 missing > max_gap of 3
    out = fill_gaps(df, max_gap=3)
    assert out["is_missing"].sum() == 4
    assert not out["filled"].any()
    assert out["close"].isna().sum() == 4


def test_gap_exactly_at_threshold_is_filled():
    df = make_candles([0, 4])  # 3 missing
    out = fill_gaps(df, max_gap=3)
    assert out["filled"].sum() == 3


def test_duplicate_timestamps_are_dropped():
    df = make_candles([0, 1, 2])
    df = pd.concat([df, df.iloc[[1]]]).sort_index()
    out = fill_gaps(df)
    assert out.index.is_unique and len(out) == 3


def test_quality_report_flags_ohlc_violation():
    df = make_candles([0, 1, 2])
    df.loc[df.index[1], "high"] = 50.0  # below open/close
    report = quality_report(df)
    assert report["ohlc_violations"] == 1
    assert report["missing_rows"] == 0
    assert report["completeness_pct"] == pytest.approx(100.0)
