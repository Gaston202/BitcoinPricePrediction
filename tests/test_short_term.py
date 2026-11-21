import numpy as np
import pandas as pd

from src.features.returns import resample_ohlcv
from src.features.short_term import candle_features, short_term_features, up_down_streak


def _hourly(n=24 * 60, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="h", tz="UTC", name="open_time")
    close = 30_000 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.003, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.003, n))
    vol = rng.uniform(100, 200, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol,
                         "quote_asset_volume": vol * close, "number_of_trades": vol,
                         "taker_buy_base_volume": vol / 2, "taker_buy_quote_volume": vol * close / 2,
                         "filled": False}, index=idx)


def _bars(hourly, rule):
    return hourly if rule == "1h" else resample_ohlcv(hourly, rule)


def test_no_look_ahead_for_every_bar_size():
    hourly = _hourly()
    for rule, hours in (("1h", 1), ("4h", 4), ("1D", 24)):
        bars = _bars(hourly, rule)
        full = short_term_features(bars, hourly, hours)
        cut = bars.index[len(bars) // 2]
        # Destroy everything after the bar at `cut` closes.
        h2 = hourly.copy()
        later = h2.index >= cut + pd.Timedelta(hours=hours)
        h2.loc[later, ["open", "high", "low", "close", "volume"]] *= 3.0
        b2 = _bars(h2, rule)
        partial = short_term_features(b2, h2, hours)
        pd.testing.assert_frame_equal(full.loc[:cut], partial.loc[:cut])


def test_features_are_scale_free():
    hourly = _hourly()
    scaled = hourly.copy()
    scaled[["open", "high", "low", "close"]] *= 10
    a = short_term_features(_bars(hourly, "4h"), hourly, 4)
    b = short_term_features(_bars(scaled, "4h"), scaled, 4)
    pd.testing.assert_frame_equal(a, b, rtol=1e-9)


def test_hour_of_day_only_for_intraday_bars():
    hourly = _hourly()
    assert "hour_sin" in short_term_features(_bars(hourly, "4h"), hourly, 4)
    assert "hour_sin" not in short_term_features(_bars(hourly, "1D"), hourly, 24)


def test_candle_shape_parts_sum_to_one():
    c = candle_features(_hourly())
    total = c["body_ratio"].abs() + c["upper_wick"] + c["lower_wick"]
    np.testing.assert_allclose(total.dropna(), 1.0)


def test_streak_counts_signed_runs():
    close = pd.Series([1, 2, 3, 2, 1, 1, 2.0])
    assert up_down_streak(close).tolist() == [0, 1, 2, -1, -2, 0, 1]
