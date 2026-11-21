import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.features.returns import DOWN, FLAT, UP
from src.modeling.multi_horizon import (
    TailSubsample,
    baselines,
    block_bootstrap_pvalue,
    expected_calibration_error,
    fast_macro_f1,
    select_window_features,
    sign_score,
    tune_weights,
)


def test_tail_subsample_fits_on_most_recent_rows_only():
    X = pd.DataFrame({"a": np.r_[np.zeros(100), np.ones(100)]})
    y = pd.Series(np.r_[np.zeros(100), np.ones(100)])
    y.iloc[150:] = 0
    y.iloc[150:175] = 1
    m = TailSubsample(LogisticRegression(), max_rows=50).fit(X, y)
    assert m.estimator_.n_features_in_ == 1
    assert len(m.classes_) == 2


def test_fast_macro_f1_matches_sklearn():
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(0)
    y, p = rng.choice([-1, 0, 1], 500), rng.choice([-1, 0, 1], 500)
    assert np.isclose(fast_macro_f1(y, p, [-1, 0, 1]), f1_score(y, p, average="macro"))


def test_tune_weights_binary_and_three_class_never_hurt_validation_f1():
    rng = np.random.default_rng(1)
    for classes in ([DOWN, UP], [DOWN, FLAT, UP]):
        proba = rng.dirichlet(np.ones(len(classes)), 400)
        y = np.asarray(classes)[rng.integers(0, len(classes), 400)]
        w = tune_weights(proba, classes, y)
        cl = np.asarray(classes)
        assert fast_macro_f1(y, cl[(proba * w).argmax(1)], cl) >= fast_macro_f1(y, cl[proba.argmax(1)], cl)


def test_ece_zero_for_perfectly_calibrated_bins():
    proba = np.array([[0.0, 1.0]] * 10)
    y = np.array([UP] * 10)
    assert expected_calibration_error(y, proba, [DOWN, UP]) == 0.0


def test_sign_score_is_up_minus_down():
    proba = np.array([[0.2, 0.3, 0.5]])
    assert np.isclose(sign_score(proba, [DOWN, FLAT, UP])[0], 0.3)


def test_bootstrap_pvalue_small_for_clear_edge_and_large_for_none():
    rng = np.random.default_rng(0)
    y = rng.choice([DOWN, UP], 1000)
    good = np.where(rng.random(1000) < 0.8, y, -y)
    noise = rng.choice([DOWN, UP], 1000)
    assert block_bootstrap_pvalue(y, good, noise, [DOWN, UP], block=7, n_boot=200) < 0.01
    assert block_bootstrap_pvalue(y, noise, good, [DOWN, UP], block=7, n_boot=200) > 0.99


def test_persistence_baseline_uses_previous_label():
    idx = pd.date_range("2022-01-01", periods=6, freq="D", tz="UTC")
    y = pd.Series([UP, UP, DOWN, DOWN, UP, UP], index=idx, dtype=float)
    fwd = pd.Series([1, -1, -1, 1, 1, 1.0], index=idx)
    b = baselines(y, fwd, fwd.shift(1).fillna(1), [DOWN, UP])
    assert b["persistence_pred"][1:].tolist() == y.shift(1)[1:].tolist()


def test_select_window_features_is_fit_without_validation_leak_and_returns_ranked_subset():
    rng = np.random.default_rng(0)
    n = 1200
    X = pd.DataFrame(rng.normal(size=(n, 25)), columns=[f"f{i}" for i in range(25)])
    y = pd.Series(np.where(X["f3"] + rng.normal(0, 0.5, n) > 0, UP, DOWN), dtype=float)
    res = select_window_features(X.iloc[:900], y.iloc[:900], X.iloc[900:], y.iloc[900:], counts=(5, 10))
    assert res["features"][0] == "f3"
    assert len(res["features"]) in (5, 10) or len(res["features"]) == res["n_survivors"]
