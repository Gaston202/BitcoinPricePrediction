"""Optuna hyperparameter search for each base model (CRISP-DM Phase 4 — Modeling).

The objective is macro-F1 on a *chronologically later* validation slice. It never sees the
walk-forward test window, because callers pass only data from before the window start
(see src.modeling.walk_forward.window_splits).

Search spaces lean toward strong regularization (shallow trees, many samples per leaf),
because with ~2k daily rows simpler models won in Chen et al. (2020)
(docs/09_literature_review.md, item 8).
"""

from __future__ import annotations

from typing import Callable

import optuna
import pandas as pd

from src.evaluation.baselines import macro_f1
from src.modeling.train import build_model

DEFAULT_N_TRIALS = 50


def _logistic_regression(t: optuna.Trial) -> dict:
    return {"C": t.suggest_float("C", 1e-3, 10.0, log=True)}


def _random_forest(t: optuna.Trial) -> dict:
    return {
        "n_estimators": t.suggest_int("n_estimators", 100, 500, step=100),
        "max_depth": t.suggest_int("max_depth", 2, 8),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 5, 60),
        "max_features": t.suggest_categorical("max_features", ["sqrt", 0.5, 1.0]),
    }


def _gradient_boosting(t: optuna.Trial) -> dict:
    return {
        "n_estimators": t.suggest_int("n_estimators", 50, 400, step=50),
        "max_depth": t.suggest_int("max_depth", 1, 4),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": t.suggest_float("subsample", 0.5, 1.0),
    }


def _xgboost(t: optuna.Trial) -> dict:
    return {
        "n_estimators": t.suggest_int("n_estimators", 50, 400, step=50),
        "max_depth": t.suggest_int("max_depth", 1, 4),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": t.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": t.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        "reg_lambda": t.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
    }


def _lightgbm(t: optuna.Trial) -> dict:
    return {
        "n_estimators": t.suggest_int("n_estimators", 50, 400, step=50),
        "num_leaves": t.suggest_int("num_leaves", 3, 15),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_child_samples": t.suggest_int("min_child_samples", 10, 80),
        "subsample": t.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": t.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
    }


def _svm(t: optuna.Trial) -> dict:
    return {
        "C": t.suggest_float("C", 1e-2, 10.0, log=True),
        "gamma": t.suggest_float("gamma", 1e-3, 1.0, log=True),
    }


def _knn(t: optuna.Trial) -> dict:
    return {
        "n_neighbors": t.suggest_int("n_neighbors", 10, 150, step=5),
        "weights": t.suggest_categorical("weights", ["uniform", "distance"]),
    }


SEARCH_SPACES: dict[str, Callable[[optuna.Trial], dict]] = {
    "logistic_regression": _logistic_regression,
    "random_forest": _random_forest,
    "gradient_boosting": _gradient_boosting,
    "xgboost": _xgboost,
    "lightgbm": _lightgbm,
    "svm": _svm,
    "knn": _knn,
}


def suggest_params(name: str, trial: optuna.Trial) -> dict:
    return SEARCH_SPACES[name](trial)


def tune_model(name: str, X_train: pd.DataFrame, y_train: pd.Series,
               X_val: pd.DataFrame, y_val: pd.Series,
               n_trials: int = DEFAULT_N_TRIALS, seed: int = 0) -> dict:
    """TPE search maximizing validation macro-F1. Returns the best params and their score."""

    def objective(trial: optuna.Trial) -> float:
        model = build_model(name, suggest_params(name, trial), random_state=seed)
        model.fit(X_train, y_train)
        return macro_f1(y_val, model.predict(X_val))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return {"params": study.best_params, "val_macro_f1": float(study.best_value),
            "n_trials": len(study.trials)}
