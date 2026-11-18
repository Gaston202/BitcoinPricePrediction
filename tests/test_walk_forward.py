import numpy as np
import pandas as pd
import pytest

from src.features.returns import DOWN, FLAT, UP
from src.modeling.train import FEATURE_COLUMNS
from src.modeling.walk_forward import (
    CLASSES,
    WINDOWS,
    baseline_scores,
    evaluate_window,
    run_walk_forward,
    window_splits,
)


def _daily_feature_frame(start="2020-01-01", n=900, seed=0):
    """Daily rows with FEATURE_COLUMNS and a label partly driven by the first feature."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="D", tz="UTC", name="open_time")
    data = {col: rng.normal(size=n) for col in FEATURE_COLUMNS}
    s = data[FEATURE_COLUMNS[0]] + rng.normal(scale=0.8, size=n)
    data["label"] = np.where(s > 0.4, UP, np.where(s < -0.4, DOWN, FLAT)).astype(float)
    return pd.DataFrame(data, index=idx)


def test_windows_match_docs_07_regimes():
    assert list(WINDOWS) == ["original", "2022_bear", "2023_bull", "2025_range"]
    for start, end in WINDOWS.values():
        assert start < end


def test_window_splits_are_chronological_with_embargo_and_no_overlap():
    df = _daily_feature_frame()
    start, end = pd.Timestamp("2021-06-01", tz="UTC"), pd.Timestamp("2021-12-31", tz="UTC")
    train, val, window = window_splits(df, start, end, val_frac=0.15, embargo=1)

    assert train.index.max() < val.index.min()
    assert val.index.max() < start <= window.index.min()
    assert window.index.max() <= end
    # Only data strictly before the window feeds training and tuning.
    # Only embargo rows (end of train, end of val) and one rounding row are lost.
    lost = (df.index < start).sum() - (len(train) + len(val))
    assert 0 < lost <= 3


def test_baseline_scores_on_a_known_sequence():
    y = pd.Series([UP, UP, DOWN, DOWN, FLAT, FLAT, UP, UP], dtype=float)
    scores = baseline_scores(y)
    assert set(scores) == {"majority_f1", "majority_auc", "persistence_f1", "persistence_auc"}
    # Majority predicts one class for everything, so its AUC is exactly 0.5.
    assert scores["majority_auc"] == pytest.approx(0.5)
    assert scores["persistence_f1"] > scores["majority_f1"]


def test_evaluate_window_reports_raw_and_calibrated_scores():
    df = _daily_feature_frame()
    train, val, window = window_splits(df, pd.Timestamp("2021-06-01", tz="UTC"),
                                       pd.Timestamp("2021-12-31", tz="UTC"))
    from src.modeling.train import build_model

    model = build_model("logistic_regression")
    row = evaluate_window(model, train, val, window, FEATURE_COLUMNS)
    for key in ("raw_f1", "raw_auc", "cal_f1", "cal_auc", "persistence_f1", "beats_persistence_raw",
                "beats_persistence_cal", "dm_met_raw", "dm_met_cal", "n_window"):
        assert key in row
    # Calibration only re-weights the argmax; probabilities and therefore AUC are unchanged.
    assert row["cal_auc"] == pytest.approx(row["raw_auc"])
    assert row["raw_f1"] > row["majority_f1"]  # planted signal is learnable


def test_run_walk_forward_without_tuning_returns_one_row_per_window_and_model():
    df = _daily_feature_frame(start="2019-01-01", n=1300)
    windows = {
        "w1": (pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2021-06-30", tz="UTC")),
        "w2": (pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2022-06-30", tz="UTC")),
    }
    out = run_walk_forward(df, FEATURE_COLUMNS, ["logistic_regression", "soft_voting"],
                           windows=windows, n_trials=0)
    assert len(out) == 4
    assert set(out["window"]) == {"w1", "w2"}
    assert set(out["model"]) == {"logistic_regression", "soft_voting"}
    assert CLASSES == [DOWN, FLAT, UP]


def test_rank_models_orders_by_windows_won_then_mean_f1_then_auc():
    from src.modeling.walk_forward import rank_models

    rows = []
    # a: wins 2 windows, mean F1 0.36 | b: wins 1 window, higher mean F1 0.40 | c: ties a, lower AUC
    for model, f1s, aucs, per in [
        ("a", [0.37, 0.35], [0.56, 0.56], [0.36, 0.34]),
        ("b", [0.45, 0.35], [0.60, 0.60], [0.36, 0.40]),
        ("c", [0.37, 0.35], [0.52, 0.52], [0.36, 0.34]),
    ]:
        for w, (f1, auc, p) in enumerate(zip(f1s, aucs, per)):
            rows.append({"window": f"w{w}", "model": model, "raw_f1": f1, "raw_auc": auc,
                         "persistence_f1": p, "beats_persistence_raw": f1 > p, "dm_met_raw": False})
    ranked = rank_models(pd.DataFrame(rows), variant="raw")
    assert list(ranked.index) == ["a", "c", "b"]
    assert ranked.loc["a", "windows_beating_persistence"] == 2
    assert ranked.loc["b", "mean_f1"] == pytest.approx(0.40)
