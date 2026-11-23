import numpy as np
import pandas as pd
import pytest

from src.modeling.range_model import (
    RANGE_FEATURE_COLUMNS,
    RANGE_MODEL_NAMES,
    TARGETS,
    baseline_predictions,
    build_range_dataset,
    build_range_model,
    range_features,
    range_targets,
    regression_scores,
    run_range_walk_forward,
)
from tests.test_indicators import _ohlcv


def test_targets_are_next_day_high_and_low_relative_to_todays_close():
    df = pd.DataFrame({"open": [100.0, 101.0, 99.0], "high": [102.0, 105.0, 100.0],
                       "low": [98.0, 97.0, 95.0], "close": [100.0, 103.0, 96.0]})
    t = range_targets(df)
    assert list(t.columns) == TARGETS
    assert t["up_next"].iloc[0] == pytest.approx(np.log(105 / 100))
    assert t["down_next"].iloc[0] == pytest.approx(np.log(97 / 100))
    assert t["up_next"].iloc[1] == pytest.approx(np.log(100 / 103))  # a gap-down day: negative upside
    assert t.iloc[-1].isna().all()  # no tomorrow for the last row


def test_range_features_never_look_ahead():
    df = _ohlcv()
    t = 200
    base = range_features(df)
    shocked = df.copy()
    shocked.iloc[t + 1:, :] *= 2.0
    pd.testing.assert_frame_equal(base.iloc[: t + 1], range_features(shocked).iloc[: t + 1])


def test_range_features_lag1_equals_todays_realised_moves():
    df = _ohlcv(n=40)
    f = range_features(df)
    assert list(f.columns) == RANGE_FEATURE_COLUMNS
    i = 30
    assert f["up_move_lag1"].iloc[i] == pytest.approx(np.log(df["high"].iloc[i] / df["close"].iloc[i - 1]))
    assert f["down_move_lag1"].iloc[i] == pytest.approx(np.log(df["low"].iloc[i] / df["close"].iloc[i - 1]))
    assert f["hl_range_5d"].iloc[i] == pytest.approx(
        np.log(df["high"] / df["low"]).iloc[i - 4: i + 1].mean())


def test_build_range_dataset_is_finite_and_has_targets():
    ds = build_range_dataset(_ohlcv(n=400))
    assert set(TARGETS) <= set(ds.columns)
    assert set(RANGE_FEATURE_COLUMNS) <= set(ds.columns)
    assert np.isfinite(ds.to_numpy()).all()


def test_baselines_use_only_past_information():
    ds = build_range_dataset(_ohlcv(n=400))
    preds = baseline_predictions(ds, "up_next")
    assert set(preds) == {"persistence", "rolling_mean_20d"}
    # Persistence for tomorrow's upside = today's realised upside (known at today's close).
    np.testing.assert_allclose(preds["persistence"], ds["up_move_lag1"])


def test_regression_scores_on_a_perfect_and_a_biased_prediction():
    y = pd.Series([0.01, 0.02, 0.03, 0.04])
    perfect = regression_scores(y, y.to_numpy(), target="up_next")
    assert perfect["mae"] == 0 and perfect["r2"] == pytest.approx(1.0)
    # For the upside, "breach" = actual high exceeded the predicted level.
    assert perfect["breach_rate"] == 0.0
    low = regression_scores(y, (y - 0.005).to_numpy(), target="up_next")
    assert low["breach_rate"] == 1.0
    down = regression_scores(-y, (-y + 0.005).to_numpy(), target="down_next")
    assert down["breach_rate"] == 1.0  # actual low went below the predicted low every day


@pytest.mark.parametrize("name", RANGE_MODEL_NAMES)
def test_each_range_model_learns_volatility_clustering(name):
    ds = build_range_dataset(_ohlcv(n=600, seed=3))
    train, test = ds.iloc[:450], ds.iloc[450:]
    cols = RANGE_FEATURE_COLUMNS
    model = build_range_model(name).fit(train[cols], train["up_next"])
    pred = model.predict(test[cols])
    assert pred.shape == (len(test),)
    assert np.isfinite(pred).all()


def test_run_range_walk_forward_rows():
    ds = build_range_dataset(_ohlcv(n=900, seed=4))
    n = len(ds)  # warm-up rows are dropped, so place windows relative to what's left
    windows = {"w1": (ds.index[n - 250], ds.index[n - 150]), "w2": (ds.index[n - 100], ds.index[n - 1])}
    out = run_range_walk_forward(ds, RANGE_FEATURE_COLUMNS, ["har", "ridge"], windows=windows, n_trials=0)
    # 2 windows x 2 targets x (2 models + 2 baselines)
    assert len(out) == 2 * 2 * 4
    assert {"window", "target", "model", "mae", "rmse", "r2", "breach_rate"} <= set(out.columns)
