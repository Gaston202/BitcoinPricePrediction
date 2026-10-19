"""Candidate models and evaluation for the 1D direction classifier
(CRISP-DM Phase 4 — Modeling).

Model selection must happen on `val`, never `test` (see docs/04_data_preparation.md):
the chronological split exists specifically so `test` can give one honest,
unbiased read of generalization at the end.
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.baselines import macro_f1

FEATURE_COLUMNS = [
    "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag5", "ret_lag7", "ret_lag14",
    "vol_7d", "vol_14d", "vol_30d", "atr_14",
    "log_volume", "volume_z", "taker_buy_ratio",
]


def split_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the feature matrix from the label column, column order fixed by FEATURE_COLUMNS."""
    return df[FEATURE_COLUMNS], df["label"]


def make_models(random_state: int = 0) -> dict[str, object]:
    """Candidate estimators. Logistic regression is scaled; trees are not (scale-invariant)."""
    return {
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                        random_state=random_state)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=20,
            class_weight="balanced", random_state=random_state,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=2, learning_rate=0.05, random_state=random_state,
        ),
    }


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict:
    """Accuracy and macro-F1 of `model` on (X, y). Does not fit the model."""
    pred = model.predict(X)
    return {"accuracy": float((pred == y.to_numpy()).mean()), "macro_f1": macro_f1(y, pred)}
