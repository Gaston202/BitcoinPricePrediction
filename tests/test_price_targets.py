import numpy as np
import pandas as pd

from src.features.returns import DOWN, UP, resample_ohlcv
from src.features.targets import TargetBuilder
from src.modeling.multi_horizon import baselines
from src.modeling.price_targets import (
    band_scores,
    block_bootstrap_mae_pvalue,
    har_columns,
    pinball,
    price_targets,
    range_features,
    select_regression_features,
)
from tests.test_short_term import _hourly


def test_targets_are_log_distances_from_close():
    bars = resample_ohlcv(_hourly(), "4h")
    t = price_targets(bars, 4)
    i = 50
    assert np.isclose(t["high_next"].iloc[i], np.log(bars["high"].iloc[i + 1] / bars["close"].iloc[i]))
    assert np.isclose(t["low_next"].iloc[i], np.log(bars["low"].iloc[i + 1] / bars["close"].iloc[i]))
    assert np.isclose(t["ret_5d"].iloc[i], np.log(bars["close"].iloc[i + 30] / bars["close"].iloc[i]))
    assert (t["high_next"].dropna() >= -1e-12).mean() > 0.99 and (t["low_next"].dropna() <= 1e-12).mean() > 0.99


def test_range_features_use_no_future_data_and_cover_har_columns():
    hourly = _hourly()
    bars = resample_ohlcv(hourly, "4h")
    full = range_features(bars, 4)
    shocked = bars.copy()
    shocked.iloc[200:, :4] *= 2
    pd.testing.assert_frame_equal(full.iloc[:200], range_features(shocked, 4).iloc[:200])
    for target in ("high_next", "low_next"):
        assert set(har_columns(target, 4)) <= set(full.columns)


def test_five_day_binary_baselines_use_the_last_fully_realized_label():
    hourly = _hourly(n=24 * 60)
    b = TargetBuilder(hourly, hourly, 1, horizon=120)
    last = np.log(b.close / b.close.shift(120))
    y = b.binary(hourly.index).dropna().loc[last.dropna().index[0]:]  # warm-up rows are dropped in the pipeline too
    base = baselines(y, b.fwd.reindex(y.index), last.reindex(y.index), [DOWN, UP], shift=120)
    # the momentum (persistence) label at t is the label of t-120, whose outcome is known at t
    assert (base["persistence_pred"][120:] == y.shift(120).to_numpy()[120:]).all()


def test_regression_selection_finds_the_informative_feature():
    rng = np.random.default_rng(0)
    n = 1500
    X = pd.DataFrame(rng.normal(size=(n, 20)), columns=[f"f{i}" for i in range(20)])
    y = pd.Series(0.8 * X["f7"] + rng.normal(0, 0.5, n))
    res = select_regression_features(X.iloc[:1200], y.iloc[:1200], X.iloc[1200:], y.iloc[1200:], counts=(5, 10))
    assert res["features"][0] == "f7"


def test_pinball_and_band_scores():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    assert np.isclose(pinball(y, np.full(4, 1.5), 0.5), 0.5)
    q = {0.1: np.full(4, 0.5), 0.5: np.full(4, 1.5), 0.9: np.full(4, 2.5)}
    s = band_scores("ret_5d", y, q)
    assert s["breach_above_q90"] == 0.25 and s["breach_below_q10"] == 0.25 and s["coverage_10_90"] == 0.5


def test_mae_bootstrap_pvalue_detects_a_better_model():
    rng = np.random.default_rng(0)
    y = rng.normal(size=2000)
    good, bad = y + rng.normal(0, 0.1, 2000), y + rng.normal(0, 1.0, 2000)
    assert block_bootstrap_mae_pvalue(y, good, bad, block=20, n_boot=200) < 0.01
    assert block_bootstrap_mae_pvalue(y, bad, good, block=20, n_boot=200) > 0.99
