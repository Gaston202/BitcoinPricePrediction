"""Candidate models and evaluation for the 1D direction classifier
(CRISP-DM Phase 4 — Modeling).

Model selection must happen on `val`, never `test` (see docs/04_data_preparation.md):
the chronological split exists specifically so `test` can give one honest,
unbiased read of generalization at the end.
"""

from __future__ import annotations

import itertools
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.evaluation.baselines import macro_f1
from src.features.selection import load_selected_features

# Chosen on train by the feature selection pass in notebooks/04_feature_engineering.ipynb,
# which writes src/features/selected_features.json. Never edit this list by hand.
FEATURE_COLUMNS = load_selected_features()

# Threads per model. Set BTC_N_JOBS when several experiment processes share the machine.
N_JOBS = int(os.environ.get("BTC_N_JOBS", "-1"))


def split_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the feature matrix from the label column, column order fixed by FEATURE_COLUMNS."""
    return df[FEATURE_COLUMNS], df["label"]


class LabelEncodedClassifier(ClassifierMixin, BaseEstimator):
    """Wrap a classifier that only accepts labels 0..K-1 (XGBoost) so it speaks -1/0/1.

    The rest of the pipeline (weighted_predict, macro_roc_auc, the baselines) indexes
    probabilities by `model.classes_`, so every candidate must expose the original labels.
    """

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y, **fit_params):
        self.encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.encoder_.classes_
        est = clone(self.estimator)
        if len(self.classes_) == 2 and est.get_params().get("objective") == "multi:softprob":
            # XGBoost's multi-class objective rejects a binary target; use the binary equivalent.
            est.set_params(objective="binary:logistic", eval_metric="logloss")
        self.estimator_ = est.fit(X, self.encoder_.transform(y), **fit_params)
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def _scaled(clf) -> Pipeline:
    """Distance/margin-based models need standardized inputs; fitted on train only by the pipeline."""
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


# Defaults are deliberately conservative (shallow trees, strong regularization): with ~2k daily
# rows, simple models beat complex ones in Chen et al. (2020) -- docs/09_literature_review.md, item 8.
# Optuna (src/modeling/tuning.py) overrides these per walk-forward window.
def _logistic_regression(p: dict, rs: int):
    return _scaled(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=rs,
                                      **{"C": 1.0, **p}))


def _random_forest(p: dict, rs: int):
    return RandomForestClassifier(class_weight="balanced", random_state=rs, n_jobs=N_JOBS,
                                  **{"n_estimators": 300, "max_depth": 5, "min_samples_leaf": 20, **p})


def _gradient_boosting(p: dict, rs: int):
    return GradientBoostingClassifier(random_state=rs,
                                      **{"n_estimators": 200, "max_depth": 2, "learning_rate": 0.05, **p})


def _xgboost(p: dict, rs: int):
    return LabelEncodedClassifier(XGBClassifier(
        objective="multi:softprob", eval_metric="mlogloss", random_state=rs, n_jobs=N_JOBS, verbosity=0,
        **{"n_estimators": 200, "max_depth": 2, "learning_rate": 0.05, "subsample": 0.8,
           "colsample_bytree": 0.8, **p},
    ))


def _lightgbm(p: dict, rs: int):
    return LGBMClassifier(class_weight="balanced", random_state=rs, n_jobs=N_JOBS, verbosity=-1,
                          **{"n_estimators": 200, "num_leaves": 7, "learning_rate": 0.05,
                             "min_child_samples": 30, "subsample": 0.8, "subsample_freq": 1,
                             "colsample_bytree": 0.8, **p})


def _svm(p: dict, rs: int):
    # SVC has no native probabilities. Platt (sigmoid) calibration is fitted with an internal
    # unshuffled CV on the training rows only (SVC(probability=True) is deprecated in sklearn 1.9).
    svc = SVC(kernel="rbf", class_weight="balanced", random_state=rs, **{"C": 1.0, "gamma": "scale", **p})
    return _scaled(CalibratedClassifierCV(svc, method="sigmoid", ensemble=False,
                                          cv=KFold(n_splits=5, shuffle=False)))


def _knn(p: dict, rs: int):
    # A large k smooths the noisy daily label; k=5 would mostly memorize noise.
    return _scaled(KNeighborsClassifier(**{"n_neighbors": 50, "weights": "uniform", **p}))


_FACTORIES = {
    "logistic_regression": _logistic_regression,
    "random_forest": _random_forest,
    "gradient_boosting": _gradient_boosting,
    "xgboost": _xgboost,
    "lightgbm": _lightgbm,
    "svm": _svm,
    "knn": _knn,
}
BASE_MODEL_NAMES = tuple(_FACTORIES)
ENSEMBLE_NAMES = ("soft_voting", "stacking")

# Diverse members: a linear model, a bagged tree model, a boosted tree model and a kernel model.
ENSEMBLE_MEMBERS = ("logistic_regression", "random_forest", "lightgbm", "svm")


def build_model(name: str, params: dict | None = None, random_state: int = 0):
    """One base estimator with `params` overriding its defaults (unprefixed hyperparameter names)."""
    if name not in _FACTORIES:
        raise ValueError(f"unknown model {name!r}; expected one of {BASE_MODEL_NAMES}")
    return _FACTORIES[name](params or {}, random_state)


def make_ensembles(members: dict[str, object], random_state: int = 0) -> dict[str, object]:
    """Soft-voting and stacking ensembles over already-configured (e.g. tuned) base models.

    Stacking uses unshuffled contiguous KFold to build out-of-fold meta-features. Some folds
    predict earlier rows from later ones, but only inside the training window, so nothing
    from validation or test is ever seen. TimeSeriesSplit can't be used here because
    cross_val_predict needs every training row to be predicted exactly once.
    """
    estimators = [(name, clone(model)) for name, model in members.items()]
    return {
        "soft_voting": VotingClassifier(estimators, voting="soft", n_jobs=1),
        "stacking": StackingClassifier(
            estimators,
            final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced",
                                               random_state=random_state),
            stack_method="predict_proba",
            cv=KFold(n_splits=5, shuffle=False),
            n_jobs=1,
        ),
    }


def make_models(random_state: int = 0) -> dict[str, object]:
    """Every candidate from docs/09_literature_review.md with default hyperparameters."""
    base = {name: build_model(name, random_state=random_state) for name in BASE_MODEL_NAMES}
    members = {name: base[name] for name in ENSEMBLE_MEMBERS}
    return {**base, **make_ensembles(members, random_state)}


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
    One class is held at weight 1.0 as a reference (FLAT when present, otherwise the
    first class); the others vary, since only the ratios between weights affect the
    argmax. Works for any number of classes, e.g. a training window that never saw
    a DOWN day (the previous version assumed exactly [DOWN, FLAT, UP] and crashed).
    """
    class_order = list(model.classes_)
    proba = model.predict_proba(X_val)
    ref = class_order.index(0.0) if 0.0 in class_order else 0
    free = [i for i in range(len(class_order)) if i != ref]

    best = {"weights": {c: 1.0 for c in class_order}, "macro_f1": macro_f1(y_val, model.predict(X_val))}
    for combo in itertools.product(grid, repeat=len(free)):
        w = np.ones(len(class_order))
        w[free] = combo
        pred = np.array(class_order)[(proba * w).argmax(axis=1)]
        f1 = macro_f1(y_val, pred)
        if f1 > best["macro_f1"]:
            best = {"weights": dict(zip(class_order, w.tolist())), "macro_f1": f1}
    return best


MODEL_CHOICE_PATH = Path(__file__).with_name("selected_model.json")


def save_model_choice(choice: dict, path: Path = MODEL_CHOICE_PATH) -> None:
    """Persist the model picked on `val` in notebooks/05 so 06 can rebuild it without re-tuning.

    `choice` = {"model": name, "feature_set": tag, "params": {...}, "members": {member: params}}.
    `members` is only used by the ensembles (each member's tuned hyperparameters).
    """
    path.write_text(json.dumps(choice, indent=2, sort_keys=True), encoding="utf-8")


def load_model_choice(path: Path = MODEL_CHOICE_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def model_from_choice(choice: dict, random_state: int = 0):
    """Unfitted estimator described by a saved choice (base model or ensemble)."""
    name = choice["model"]
    if name in ENSEMBLE_NAMES:
        members = {m: build_model(m, p, random_state) for m, p in choice["members"].items()}
        return make_ensembles(members, random_state)[name]
    return build_model(name, choice.get("params"), random_state)
