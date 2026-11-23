"""Train, cache and query the direction models the trading agent relies on.

One model per candle interval:
  * "1D" - the project's selected model (random forest, 1D horizon, theta = 1%),
            trained on `train`, decision weights calibrated on `val`.
  * "1h" - the same recipe on hourly candles (theta = 0.25%), used by the live
            agent because a daily model would only make one decision per day.

Both use the chronological split, so the `test` window the dashboard replays
was never seen during training or calibration.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

from src import config
from src.features.engineering import build_features, chronological_split, feature_matrix
from src.features.returns import DOWN, FLAT, UP
from src.modeling.train import FEATURE_COLUMNS, make_models, tune_class_weights

logger = logging.getLogger(__name__)

MODELS_DIR = config.PROJECT_ROOT / "models"
MODEL_VERSION = 1  # bump to invalidate cached .joblib files after a recipe change

INTERVALS = {
    "1D": {"file": "btcusdt_1D_clean.parquet", "flat_threshold": 0.01, "n_estimators": 300},
    "1h": {"file": "btcusdt_1h_clean.parquet", "flat_threshold": 0.0025, "n_estimators": 120},
}


@dataclass
class ModelBundle:
    """A fitted model plus everything needed to turn its output into a decision."""

    interval: str
    model: object
    weights: dict[float, float]
    class_mean_return: dict[float, float]  # mean forward log return per true class (train)
    flat_threshold: float
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    val_macro_f1: float
    feature_importance: dict[str, float] = field(default_factory=dict)
    version: int = MODEL_VERSION

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-row class probabilities, calibrated class, expected log return."""
        proba = self.model.predict_proba(X[FEATURE_COLUMNS])
        classes = [float(c) for c in self.model.classes_]
        full = pd.DataFrame(0.0, index=X.index, columns=[DOWN, FLAT, UP], dtype=float)
        for i, c in enumerate(classes):
            full[c] = proba[:, i]
        w = np.array([self.weights.get(c, 1.0) for c in (DOWN, FLAT, UP)])
        pred_idx = (full.to_numpy() * w).argmax(axis=1)
        expected = full.to_numpy() @ np.array([self.class_mean_return.get(c, 0.0)
                                               for c in (DOWN, FLAT, UP)])
        return pd.DataFrame({
            "p_down": full[DOWN], "p_flat": full[FLAT], "p_up": full[UP],
            "pred_class": np.array([DOWN, FLAT, UP])[pred_idx],
            "expected_log_return": expected,
        }, index=X.index)


def load_clean(interval: str) -> pd.DataFrame:
    path = config.INTERIM_DIR / INTERVALS[interval]["file"]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run notebooks/02_data_quality.ipynb first.")
    return pd.read_parquet(path)


def train_bundle(interval: str, random_state: int = 0) -> ModelBundle:
    spec = INTERVALS[interval]
    clean = load_clean(interval)
    features = build_features(clean, horizon=1, flat_threshold=spec["flat_threshold"])
    train, val, test = chronological_split(features)

    model = make_models(random_state=random_state)["random_forest"]
    model.set_params(n_estimators=spec["n_estimators"], n_jobs=-1)
    model.fit(train[FEATURE_COLUMNS], train["label"])
    tuned = tune_class_weights(model, val[FEATURE_COLUMNS], val["label"])

    fwd = np.log(clean["close"].shift(-1) / clean["close"]).reindex(train.index)
    class_mean = {float(c): float(fwd[train["label"] == c].mean()) for c in (DOWN, FLAT, UP)}
    importance = dict(sorted(zip(FEATURE_COLUMNS, model.feature_importances_.tolist()),
                             key=lambda kv: -kv[1]))
    return ModelBundle(
        interval=interval, model=model,
        weights={float(k): float(v) for k, v in tuned["weights"].items()},
        class_mean_return=class_mean, flat_threshold=spec["flat_threshold"],
        train_end=train.index.max(), val_end=val.index.max(), test_start=test.index.min(),
        val_macro_f1=float(tuned["macro_f1"]), feature_importance=importance,
    )


_cache: dict[str, ModelBundle] = {}
_lock = threading.Lock()


def get_bundle(interval: str) -> ModelBundle:
    """Return the model for `interval`, loading it from disk or training it once."""
    if interval not in INTERVALS:
        raise ValueError(f"interval must be one of {list(INTERVALS)}")
    with _lock:
        if interval in _cache:
            return _cache[interval]
        path = MODELS_DIR / f"agent_rf_{interval}.joblib"
        bundle = None
        if path.exists():
            try:
                bundle = joblib.load(path)
                if getattr(bundle, "version", None) != MODEL_VERSION:
                    bundle = None
            except Exception:  # corrupted / incompatible pickle -> retrain
                logger.warning("Could not load %s, retraining", path)
                bundle = None
        if bundle is None:
            logger.info("Training %s agent model (first run only)...", interval)
            bundle = train_bundle(interval)
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(bundle, path)
        _cache[interval] = bundle
        return bundle


def test_window(interval: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(clean candles, model-ready feature rows) for the unseen test window."""
    bundle = get_bundle(interval)
    clean = load_clean(interval)
    feats = feature_matrix(clean).dropna()
    feats = feats.loc[feats.index >= bundle.test_start]
    return clean.loc[feats.index], feats
