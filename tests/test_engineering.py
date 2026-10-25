import numpy as np
import pandas as pd
import pytest

from src.features.engineering import (average_true_range, build_features, calendar_features,
                                      chronological_split, interaction_features, lagged_returns,
                                      momentum, rolling_volatility, rsi, trend_features,
                                      volume_features)
from tests.test_clean import make_candles


def _daily_candles(n: int, seed: int = 0) -> pd.DataFrame:
    """n daily candles with a noisy random-walk close and varying volume."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC", name="open_time")
    volume = rng.uniform(50, 150, n)
    return pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
        "volume": volume, "taker_buy_base_volume": volume * rng.uniform(0.4, 0.6, n),
    }, index=idx)


def test_lagged_returns_are_true_lags_of_the_one_day_return():
    close = pd.Series([100.0, 110.0, 121.0, 108.9])
    out = lagged_returns(close, lags=(1, 2))
    assert out["ret_lag1"].iloc[2] == pytest.approx(np.log(121 / 110))
    # ret_lag2 at row 2 is the 1-day return one candle earlier, not the 2-day return.
    assert out["ret_lag2"].iloc[2] == pytest.approx(np.log(110 / 100))
    assert np.isnan(out["ret_lag1"].iloc[0])


def test_momentum_is_cumulative_return_over_window():
    close = pd.Series([100.0, 110.0, 121.0, 108.9])
    out = momentum(close, windows=(2,))
    assert out["mom_2d"].iloc[2] == pytest.approx(np.log(121 / 100))


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


def test_rsi_is_100_in_a_pure_uptrend_and_bounded():
    assert rsi(pd.Series(np.arange(1.0, 31.0)), window=14).iloc[-1] == pytest.approx(100.0)
    values = rsi(_daily_candles(200)["close"]).dropna()
    assert values.between(0, 100).all()


def test_trend_features_distance_from_moving_average():
    close = pd.Series([10.0, 10.0, 13.0])
    out = trend_features(close, ma_windows=(3,))
    assert out["dist_ma3"].iloc[2] == pytest.approx(13 / 11 - 1)


def test_volume_features_ratio_undefined_when_no_volume():
    df = make_candles([0, 1, 2])
    df["volume"] = [1.0, 0.0, 1.0]
    df["taker_buy_base_volume"] = [0.5, 0.0, 0.5]
    out = volume_features(df, z_window=2, ratio_window=2)
    assert np.isnan(out["taker_buy_ratio"].iloc[1])
    assert out["taker_buy_ratio"].iloc[0] == pytest.approx(0.5)
    assert out["volume_ratio"].iloc[2] == pytest.approx(1.0 / 0.5)
    assert "log_volume" not in out  # non-stationary level, deliberately excluded


def test_calendar_features_are_cyclical_and_flag_weekends():
    idx = pd.date_range("2024-01-01", periods=7, freq="D", tz="UTC")  # Monday .. Sunday
    out = calendar_features(idx)
    assert out["is_weekend"].tolist() == [0, 0, 0, 0, 0, 1, 1]
    # On the unit circle Sunday sits next to Monday, unlike the raw 6 vs 0.
    monday, sunday = out.iloc[0], out.iloc[6]
    tuesday = out.iloc[1]
    dist = lambda a, b: np.hypot(a["dow_sin"] - b["dow_sin"], a["dow_cos"] - b["dow_cos"])
    assert dist(monday, sunday) == pytest.approx(dist(monday, tuesday))


def test_interaction_features_are_products():
    base = pd.DataFrame({"ret_lag1": [0.02], "volume_z": [3.0], "mom_7d": [0.1], "vol_ratio": [2.0]})
    out = interaction_features(base)
    assert out["ret1_x_volz"].iloc[0] == pytest.approx(0.06)
    assert out["mom7_x_volratio"].iloc[0] == pytest.approx(0.2)


def test_build_features_is_model_ready():
    df = _daily_candles(120)
    out = build_features(df, horizon=1, flat_threshold=0.01)
    assert 0 < len(out) < len(df)
    assert np.isfinite(out.to_numpy()).all()
    assert set(out["label"].unique()) <= {-1.0, 0.0, 1.0}
    assert "log_volume" not in out


def test_build_features_uses_no_future_data():
    # Changing prices after day t must not change any feature at day t (only the label may).
    df = _daily_candles(120)
    shocked = df.copy()
    shocked.iloc[100:, :4] *= 1.5
    before = build_features(df).drop(columns="label")
    after = build_features(shocked).drop(columns="label")
    common = before.index[before.index < df.index[100]]
    pd.testing.assert_frame_equal(before.loc[common], after.loc[common])


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
