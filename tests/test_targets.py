import numpy as np
import pandas as pd
import pytest

from src.features.returns import DOWN, FLAT, UP, resample_ohlcv
from src.features.targets import TargetBuilder
from tests.test_short_term import _hourly


def _builder(rule="4h", hours=4, hourly=None):
    hourly = _hourly() if hourly is None else hourly
    bars = hourly if rule == "1h" else resample_ohlcv(hourly, rule)
    return TargetBuilder(bars, hourly, hours), bars


def test_binary_is_sign_of_next_bar_return():
    b, bars = _builder()
    y = b.binary(bars.index)
    nxt = np.log(bars["close"].shift(-1) / bars["close"])
    mask = nxt.notna()
    assert (y[mask] == np.where(nxt[mask] > 0, UP, DOWN)).all()
    assert y[~mask].isna().all()


def test_vol3_flat_share_grows_with_k():
    b, bars = _builder()
    shares = [(b.vol3(bars.index, k).dropna() == FLAT).mean() for k in (0.2, 0.6, 1.5)]
    assert shares[0] < shares[1] < shares[2]


def test_choose_k_targets_a_third_flat_on_train_rows():
    b, bars = _builder()
    train = bars.index[: len(bars) // 2]
    for target in ("vol3", "triple_barrier"):
        k = b.choose_k(target, train)
        share = (b.labels(target, train, k).dropna() == FLAT).mean()
        assert abs(share - 1 / 3) < 0.08
    assert b.choose_k("binary", train) is None


def test_triple_barrier_reads_the_next_bar_path_in_order():
    """Hand-built path: the 2nd hour of the next bar touches the lower barrier, the 3rd the upper one."""
    hourly = _hourly(n=24 * 40)
    b, bars = _builder("4h", 4, hourly)
    t = bars.index[200]
    c0 = bars.loc[t, "close"]
    sigma = b.sigma.loc[t]
    nxt = [t + pd.Timedelta(hours=4 + j) for j in range(4)]
    hourly.loc[nxt, "high"] = c0 * 1.0001
    hourly.loc[nxt, "low"] = c0 * 0.9999
    hourly.loc[nxt[1], "low"] = c0 * np.exp(-2 * sigma)
    hourly.loc[nxt[2], "high"] = c0 * np.exp(2 * sigma)
    b2 = TargetBuilder(bars, hourly, 4)
    assert b2.triple_barrier(pd.Index([t]), 1.0).iloc[0] == DOWN
    # With a wider band nothing is touched: FLAT (vertical barrier).
    assert b2.triple_barrier(pd.Index([t]), 3.0).iloc[0] == FLAT


def test_triple_barrier_uses_no_data_beyond_the_next_bar():
    hourly = _hourly(n=24 * 40)
    b, bars = _builder("4h", 4, hourly)
    idx = bars.index[100:200]
    before = b.triple_barrier(idx, 0.8)
    h2 = hourly.copy()
    after_next = h2.index >= idx[-1] + pd.Timedelta(hours=8)
    h2.loc[after_next, ["high", "low", "close"]] *= 2
    b2 = TargetBuilder(bars, h2, 4)
    pd.testing.assert_series_equal(before, b2.triple_barrier(idx, 0.8))


def test_unknown_target_raises():
    b, bars = _builder()
    with pytest.raises(ValueError):
        b.labels("nope", bars.index, 1.0)
