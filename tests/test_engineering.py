import numpy as np
import pandas as pd
import pytest

from src.features.engineering import (average_true_range, build_features, chronological_split,
                                      lagged_returns, rolling_volatility, volume_features)
from src.features.returns import log_returns
from tests.test_clean import make_candles


def test_lagged_returns_shifts_match_log_returns():
    close = pd.Series([100.0, 110.0, 121.0, 108.9])
    out = lagged_returns(close, lags=(1, 2))
    assert out["ret_lag1"].iloc[1] == pytest.approx(np.log(110 / 100))
    assert out["ret_lag2"].iloc[2] == pytest.approx(np.log(121 / 100))
    assert np.isnan(out["ret_lag1"].iloc[0])


def test_rolling_volatility_window_warmup():
    returns = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01])
    out = rolling_volatility(returns, windows=(3,))
    assert np.isnan(out["vol_3d"].iloc[1])  # fewer than 3 observations yet
    assert out["vol_3d"].iloc[2] == pytest.approx(returns.iloc[:3].std())


def test_average_true_range_uses_gap_from_prior_close():
    df = make_candles([0, 1])  # row 0: high-low = (100+1)-(100-1) = 2, no prior close
    df.loc[df.index[1], ["high", "low", "close"]] = [120.0, 115.0, 118.0]  # gap up from close=100
    atr = average_true_range(df, window=2)
    # true range at row 1 = max(high-low, |high-prev_close|, |low-prev_close|) = max(5, 20, 15) = 20
    assert atr.iloc[1] == pytest.approx((2.0 + 20.0) / 2)


def test_volume_features_ratio_undefined_when_no_volume():
    df = make_candles([0, 1, 2])
    df["volume"] = [1.0, 0.0, 1.0]
    df["taker_buy_base_volume"] = [0.5, 0.0, 0.5]
    out = volume_features(df, window=2)
    assert np.isnan(out["taker_buy_ratio"].iloc[1])
    assert out["taker_buy_ratio"].iloc[0] == pytest.approx(0.5)


def test_build_features_drops_warmup_and_trailing_rows_and_has_no_nan():
    df = make_candles(list(range(40)))
    df["volume"] = 1.0 + np.arange(40)  # varying volume so the rolling z-score isn't 0/0
    df["taker_buy_base_volume"] = df["volume"] / 2
    out = build_features(df, horizon=1, flat_threshold=0.0)
    assert len(out) > 0
    assert len(out) < len(df)
    assert not out.isna().any().any()
    assert set(out["label"].unique()) <= {-1.0, 0.0, 1.0}


def test_chronological_split_is_ordered_and_purges_boundary():
    df = pd.DataFrame({"x": range(100)})
    train, val, test = chronological_split(df, train_frac=0.7, val_frac=0.15, embargo=2)
    # Splits stay in time order with no overlap or shuffling.
    assert train.index.max() < val.index.min() < val.index.max() < test.index.min()
    # Embargo rows are removed, not reassigned: the frame shrinks by exactly 2*embargo.
    assert len(train) + len(val) + len(test) == len(df) - 2 * 2


def test_chronological_split_respects_fractions_before_embargo():
    df = pd.DataFrame({"x": range(1000)})
    train, val, test = chronological_split(df, train_frac=0.7, val_frac=0.15, embargo=1)
    assert len(train) == pytest.approx(700, abs=2)
    assert len(val) == pytest.approx(150, abs=2)
    assert len(test) == pytest.approx(150, abs=2)
