import numpy as np
import pandas as pd
import pytest

from src.features.indicators import (
    PAPER_INDICATOR_COLUMNS,
    balance_of_power,
    bollinger,
    build_features_with_indicators,
    paper_indicators,
    stochastic,
    williams_r,
)
from tests.test_engineering import _daily_candles


def _ohlcv(n=300, seed=0):
    """Random-walk candles with a real intraday range, so high/low-based indicators aren't degenerate."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    open_ = close * np.exp(rng.normal(0, 0.005, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.02, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.02, n))
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC", name="open_time")
    volume = rng.uniform(50, 150, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume,
                         "taker_buy_base_volume": volume * rng.uniform(0.4, 0.6, n)}, index=idx)


def test_paper_indicators_has_documented_columns_and_no_nans_after_warmup():
    out = paper_indicators(_ohlcv())
    assert list(out.columns) == PAPER_INDICATOR_COLUMNS
    assert out.iloc[60:].notna().all().all()


def test_paper_indicators_never_look_ahead():
    # Leakage guard (docs/09 items 1 and 9): changing every candle after day t must leave
    # the indicators at t unchanged. Jung (2023) and Oktem (2025) failed exactly this.
    df = _ohlcv()
    t = 200
    base = paper_indicators(df)
    shocked = df.copy()
    shocked.iloc[t + 1:, :] *= 3.0
    after = paper_indicators(shocked)
    pd.testing.assert_frame_equal(base.iloc[: t + 1], after.iloc[: t + 1])


def test_paper_indicators_are_scale_free():
    # BTC went from ~4k to ~100k; features in price units would drift with the price level.
    df = _ohlcv()
    scaled = df.copy()
    scaled[["open", "high", "low", "close"]] *= 25.0
    pd.testing.assert_frame_equal(paper_indicators(df), paper_indicators(scaled), rtol=1e-9)


def test_stochastic_and_williams_r_are_bounded_and_mirror_each_other():
    df = _ohlcv()
    k, _ = stochastic(df)
    wr = williams_r(df)
    valid = k.dropna().index
    assert k[valid].between(0, 100).all()
    assert wr[valid].between(-100, 0).all()
    # Same 14-day window: %R = %K - 100.
    np.testing.assert_allclose(wr[valid], k[valid] - 100)


def test_stochastic_k_is_100_at_the_window_high():
    df = _ohlcv(n=30)
    df.loc[df.index[-1], ["high", "close"]] = df["high"].max() * 1.5
    k, _ = stochastic(df)
    assert k.iloc[-1] == pytest.approx(100.0)


def test_bollinger_percent_b_is_half_on_a_flat_moving_average():
    close = pd.Series([100.0, 102.0] * 20)
    pct_b, width = bollinger(close, window=20)
    # An alternating series ending on 102 sits above the 101 mean, so %B > 0.5; one step earlier < 0.5.
    assert pct_b.iloc[-1] > 0.5 > pct_b.iloc[-2]
    assert (width.dropna() > 0).all()


def test_balance_of_power_uses_only_its_own_candle():
    df = pd.DataFrame({"open": [10.0], "high": [12.0], "low": [8.0], "close": [11.0]})
    assert balance_of_power(df).iloc[0] == pytest.approx(0.25)  # (11 - 10) / (12 - 8)


def test_build_features_with_indicators_keeps_label_and_adds_indicator_columns():
    df = _daily_candles(200)
    out = build_features_with_indicators(df)
    assert "label" in out.columns
    assert set(PAPER_INDICATOR_COLUMNS) <= set(out.columns)
    assert np.isfinite(out.drop(columns="label").to_numpy()).all()
