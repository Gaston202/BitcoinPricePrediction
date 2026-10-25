"""Candidate models and evaluation for the 1D direction classifier
(CRISP-DM Phase 4 — Modeling).

Model selection must happen on `val`, never `test` (see docs/04_data_preparation.md):
the chronological split exists specifically so `test` can give one honest,
unbiased read of generalization at the end.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.baselines import macro_f1
from src.features.selection import load_selected_features

# Chosen on train by the feature selection pass in notebooks/04_feature_engineering.ipynb,
# which writes src/features/selected_features.json. Never edit this list by hand.
FEATURE_COLUMNS = load_selected_features()


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


def weighted_predict(model, X: pd.DataFrame, weights: dict[float, float]) -> np.ndarray:
    """Predict argmax(proba[c] * weights[c]) instead of the model's default argmax(proba[c]).

    A plain argmax over class probabilities collapses toward whichever class the
    model is least uncertain about when the signal is weak (see docs/05_modeling.md:
    this model defaults to "flat" on 89% of test days vs. an actual 45% flat rate).
    Upweighting the under-predicted classes corrects that bias without retraining.
    """
    proba = model.predict_proba(X)
    class_order = list(model.classes_)
    w = np.array([weights[c] for c in class_order])
    idx = (proba * w).argmax(axis=1)
    return np.array(class_order)[idx]


DEFAULT_WEIGHT_GRID = tuple(round(0.5 + 0.25 * i, 2) for i in range(11))  # 0.5, 0.75, ..., 3.0


def tune_class_weights(
    model, X_val: pd.DataFrame, y_val: pd.Series, grid: tuple[float, ...] = DEFAULT_WEIGHT_GRID,
) -> dict:
    """Grid-search per-class weights that maximize macro-F1 on `X_val`/`y_val`.

    Must only be called with validation data, never test (see FEATURE_COLUMNS docstring
    in this module and docs/04_data_preparation.md on the chronological split).
    One class is held at weight 1.0 as a reference; only the other two vary, since
    only the ratios between weights affect the argmax.
    """
    class_order = list(model.classes_)
    proba = model.predict_proba(X_val)
    y = y_val.to_numpy()

    best = {"weights": {c: 1.0 for c in class_order}, "macro_f1": macro_f1(y_val, model.predict(X_val))}
    for w0, w2 in itertools.product(grid, grid):
        w = np.array([w0, 1.0, w2])
        pred = np.array(class_order)[(proba * w).argmax(axis=1)]
        f1 = macro_f1(y_val, pred)
        if f1 > best["macro_f1"]:
            best = {"weights": dict(zip(class_order, w.tolist())), "macro_f1": f1}
    return best
