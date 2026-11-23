import numpy as np
import pandas as pd
import pytest

from src.data.clean import fill_gaps
from src.features.returns import (DOWN, FLAT, UP, direction_label, forward_log_return,
                                  log_returns, resample_ohlcv)
from tests.test_clean import make_candles


def test_forward_return_looks_ahead_and_log_return_looks_back():
    close = pd.Series([100.0, 110.0, 99.0])
    fwd = forward_log_return(close)
    back = log_returns(close)
    assert fwd.iloc[0] == pytest.approx(np.log(110 / 100))
    assert np.isnan(fwd.iloc[-1])  # no future for the last row
    assert np.isnan(back.iloc[0])  # no past for the first row
    # The label at t must equal the backward return at t+1 (no off-by-one).
    assert fwd.iloc[0] == pytest.approx(back.iloc[1])


def test_direction_label_thresholds():
    r = pd.Series([0.02, 0.001, -0.001, -0.02, np.nan])
    labels = direction_label(r, flat_threshold=0.005)
    assert labels.iloc[:4].tolist() == [UP, FLAT, FLAT, DOWN]
    assert np.isnan(labels.iloc[4])


def test_resample_to_4h_aggregates_ohlcv():
    df = fill_gaps(make_candles(list(range(8))))
    out = resample_ohlcv(df, "4h")
    assert len(out) == 2
    first = out.iloc[0]
    assert first["open"] == 100.0 and first["close"] == 103.0
    assert first["high"] == 104.0 and first["low"] == 99.0
    assert first["volume"] == 4.0
    assert out["is_complete"].all()


def test_resample_marks_bucket_with_long_gap_incomplete():
    df = fill_gaps(make_candles([0, 5, 6, 7]), max_gap=3)  # hours 1-4 missing (4 > 3)
    out = resample_ohlcv(df, "4h")
    assert not out["is_complete"].iloc[0]
    assert np.isnan(out["close"].iloc[0])
