"""The quant analyst's model: the project's selected classifier, trained walk-forward.

src/modeling/selected_model.json (chosen by the team in notebooks 05-07) names the model:
LightGBM, binary target (next 1h candle up/down), 10 features from the candidate pool,
Platt calibration on the last 15% of the training data. Here it is retrained using only
candles strictly before `cutoff`, so a replay never sees its own future.
"""

from __future__ import annotations

import json
import logging
import threading

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from src import config
from src.features.returns import forward_log_return

logger = logging.getLogger(__name__)

CHOICE_PATH = config.PROJECT_ROOT / "src" / "modeling" / "selected_model.json"
MODELS_DIR = config.PROJECT_ROOT / "models"
CAL_SHARE = 0.15
MAX_TRAIN_ROWS = 40_000  # most recent ~4.5 years of hourly candles keeps training to seconds
MIN_TRAIN_ROWS = 2_000
_lock = threading.Lock()
_cache: dict[str, "QuantModel"] = {}


def load_choice() -> dict:
    return json.loads(CHOICE_PATH.read_text())


def hourly_features(hourly: pd.DataFrame) -> pd.DataFrame:
    """Candidate-pool features for 1h bars (needs the selected columns only)."""
    from src.features.short_term import candidate_pool
    pool = candidate_pool(hourly.dropna(subset=["close"]), hourly, 1)
    return pool[load_choice()["features"]]


class QuantModel:
    def __init__(self, model, calibrated, features: list[str], cutoff: pd.Timestamp, n_train: int):
        self.model, self.calibrated, self.features = model, calibrated, features
        self.cutoff, self.n_train = cutoff, n_train

    def p_up(self, X: pd.DataFrame) -> pd.Series:
        X = X[self.features]
        ok = X.notna().all(axis=1)
        out = pd.Series(np.nan, index=X.index)
        if ok.any():
            proba = self.calibrated.predict_proba(X[ok])
            up_col = list(self.calibrated.classes_).index(1.0)
            out[ok] = proba[:, up_col]
        return out


def train(hourly: pd.DataFrame, cutoff: pd.Timestamp, features: pd.DataFrame | None = None) -> QuantModel:
    from lightgbm import LGBMClassifier

    choice = load_choice()
    feats = hourly_features(hourly) if features is None else features
    fwd = forward_log_return(hourly["close"], 1).reindex(feats.index)
    y = pd.Series(np.where(fwd > 0, 1.0, -1.0), index=feats.index).where(fwd.notna())
    # Only rows whose label is known before the cutoff (one-candle embargo).
    keep = (feats.index < cutoff - pd.Timedelta("1h")) & y.notna() & feats.notna().all(axis=1)
    X, y = feats[keep].iloc[-MAX_TRAIN_ROWS:], y[keep].iloc[-MAX_TRAIN_ROWS:]
    if len(X) < MIN_TRAIN_ROWS:
        raise ValueError(f"only {len(X)} training rows before {cutoff}")
    split = int(len(X) * (1 - CAL_SHARE))
    model = LGBMClassifier(**choice["params"], random_state=0, verbosity=-1)
    model.fit(X.iloc[:split], y.iloc[:split])
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid").fit(X.iloc[split:], y.iloc[split:])
    return QuantModel(model, calibrated, list(choice["features"]), cutoff, len(X))


def get_model(hourly: pd.DataFrame, cutoff: pd.Timestamp, features: pd.DataFrame | None = None) -> QuantModel:
    """Train once per cutoff day and cache in memory and in models/ (gitignored)."""
    key = cutoff.floor("D").strftime("%Y%m%d")
    with _lock:
        if key in _cache:
            return _cache[key]
        path = MODELS_DIR / f"quant_lgbm_{key}.joblib"
        model = None
        if path.exists():
            try:
                model = joblib.load(path)
            except Exception:
                model = None
        if model is None:
            logger.info("Training the quant analyst's LightGBM on candles before %s", cutoff.floor("D"))
            model = train(hourly, cutoff.floor("D"), features)
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, path)
        _cache[key] = model
        return model
