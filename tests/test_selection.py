import json

import numpy as np
import pandas as pd
import pytest

from src.features.returns import DOWN, FLAT, UP
from src.features.selection import (correlated_pairs, drop_correlated, fit_pca, load_selected_features,
                                    low_variance, near_constant, pca_transform, relevance,
                                    save_manifest, select_features)


def _frame(n=600, seed=1):
    """A label driven by `signal`, a near-duplicate of it, pure noise, and degenerate columns.

    The noise test is statistical: by construction ~5% of pure-noise columns beat the
    95% shuffled-label baseline by chance (seed 0 is one of them), so the seed is fixed.
    """
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    X = pd.DataFrame({
        "signal": signal,
        "signal_copy": signal + rng.normal(scale=0.05, size=n),
        "noise": rng.normal(size=n),
        "constant": np.ones(n),
        "almost_constant": np.r_[np.zeros(n - 5), np.ones(5)],
    })
    y = pd.Series(np.where(signal > 0.5, UP, np.where(signal < -0.5, DOWN, FLAT)))
    return X, y


def test_near_constant_and_low_variance_detect_degenerate_columns():
    X, _ = _frame()
    assert set(near_constant(X)) == {"constant", "almost_constant"}
    assert low_variance(X) == ["constant"]


def test_relevance_separates_signal_from_noise():
    X, y = _frame()
    scores = relevance(X[["signal", "noise"]], y, n_permutations=5)
    assert not scores.loc["signal", "is_noise"]
    assert scores.loc["noise", "is_noise"]
    assert scores.index[0] == "signal"  # sorted by MI, strongest first


def test_drop_correlated_keeps_higher_mi_member_of_a_pair():
    corr = pd.DataFrame([[1.0, 0.95, 0.1], [0.95, 1.0, 0.1], [0.1, 0.1, 1.0]],
                        index=list("abc"), columns=list("abc"))
    dropped = drop_correlated(corr, pd.Series({"a": 0.01, "b": 0.05, "c": 0.02}), threshold=0.85)
    assert list(dropped) == ["a"]
    assert "correlated with b" in dropped["a"]


def test_drop_correlated_keeps_one_member_of_a_three_way_cluster():
    corr = pd.DataFrame(np.full((3, 3), 0.9) + 0.1 * np.eye(3), index=list("abc"), columns=list("abc"))
    dropped = drop_correlated(corr, pd.Series({"a": 0.3, "b": 0.2, "c": 0.1}), threshold=0.85)
    assert set(dropped) == {"b", "c"}


def test_correlated_pairs_respects_threshold_and_sign():
    corr = pd.DataFrame([[1.0, -0.9], [-0.9, 1.0]], index=list("ab"), columns=list("ab"))
    assert len(correlated_pairs(corr, threshold=0.85)) == 1  # |-0.9| counts as correlated
    assert correlated_pairs(corr, threshold=0.95).empty


def test_select_features_end_to_end():
    X, y = _frame()
    manifest = select_features(X, y)
    assert manifest["selected"] == ["signal"] or manifest["selected"] == ["signal_copy"]
    assert set(manifest["dropped"]) == {"signal", "signal_copy", "noise", "constant",
                                        "almost_constant"} - set(manifest["selected"])
    assert manifest["params"]["fit_on"] == "train"


def test_pca_explains_requested_variance_and_collapses_duplicates():
    X, _ = _frame()
    X = X[["signal", "signal_copy", "noise"]]
    pipeline = fit_pca(X, variance=0.9)
    pcs = pca_transform(pipeline, X)
    assert pipeline.named_steps["pca"].explained_variance_ratio_.sum() >= 0.9
    # Two near-identical columns + one independent column -> two components suffice.
    assert list(pcs.columns) == ["pc1", "pc2"]
    assert pcs.index.equals(X.index)


def test_manifest_round_trip(tmp_path):
    path = tmp_path / "selected.json"
    save_manifest({"selected": ["a", "b"], "dropped": {}, "params": {}}, path)
    assert load_selected_features(path) == ["a", "b"]
    assert json.loads(path.read_text(encoding="utf-8"))["dropped"] == {}
