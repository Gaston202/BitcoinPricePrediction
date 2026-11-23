"""Multi-regime walk-forward comparison of every candidate model (CRISP-DM Phase 4/5).

For each test window, everything strictly before the window start is split chronologically
into an inner train (fit) and inner validation (Optuna tuning + class-weight calibration).
The window itself is scored once. This is the protocol docs/07 introduced, moved into tested
code so the notebooks only orchestrate it.

Known caveat (decision 2026-11-18): FEATURE_COLUMNS were selected on the canonical train
split (2017-10 -> 2024-06), which overlaps the 2022 and 2023 windows. Results for those two
windows are therefore somewhat optimistic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.evaluation.baselines import macro_f1, majority_baseline, persistence_baseline
from src.evaluation.criteria import data_mining_criterion_met, hard_prediction_scores, macro_roc_auc
from src.features.engineering import LABEL_HORIZON, chronological_split
from src.features.returns import DOWN, FLAT, UP
from src.modeling.train import (
    BASE_MODEL_NAMES,
    ENSEMBLE_MEMBERS,
    ENSEMBLE_NAMES,
    build_model,
    make_ensembles,
    tune_class_weights,
    weighted_predict,
)
from src.modeling.tuning import DEFAULT_N_TRIALS, tune_model

CLASSES = [DOWN, FLAT, UP]
INNER_VAL_FRAC = 0.15


def _utc(day: str) -> pd.Timestamp:
    return pd.Timestamp(day, tz="UTC")


# Regimes picked from the full-year buy-and-hold table (notebooks/03_eda.ipynb, item 12).
WINDOWS: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {
    "original": (_utc("2025-07-09"), _utc("2026-09-29")),
    "2022_bear": (_utc("2022-01-01"), _utc("2022-12-31")),
    "2023_bull": (_utc("2023-01-01"), _utc("2023-12-31")),
    "2025_range": (_utc("2025-01-01"), _utc("2025-12-31")),
}


def window_splits(features: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
                  val_frac: float = INNER_VAL_FRAC, embargo: int = LABEL_HORIZON,
                  ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(inner_train, inner_val, window). Only rows before `start` are used for train/val."""
    pool = features.loc[features.index < start]
    inner_train, inner_val, _ = chronological_split(pool, train_frac=1 - val_frac,
                                                    val_frac=val_frac, embargo=embargo)
    return inner_train, inner_val, features.loc[start:end]


def baseline_scores(y: pd.Series) -> dict:
    """Macro-F1 and ROC-AUC of the majority and persistence baselines on one window."""
    maj = majority_baseline(y)
    per = persistence_baseline(y)
    maj_pred = pd.Series(maj["class"], index=y.index)
    per_pred = y.shift(1)
    mask = per_pred.notna()
    return {
        "majority_f1": maj["macro_f1"],
        "majority_auc": macro_roc_auc(y, hard_prediction_scores(maj_pred, CLASSES), CLASSES),
        "persistence_f1": per["macro_f1"],
        "persistence_auc": macro_roc_auc(y[mask], hard_prediction_scores(per_pred[mask], CLASSES),
                                         CLASSES),
    }


def evaluate_window(model, train: pd.DataFrame, val: pd.DataFrame, window: pd.DataFrame,
                    feature_cols: list[str]) -> dict:
    """Fit on inner train, calibrate on inner val, score raw and calibrated on the window."""
    X_tr, y_tr = train[feature_cols], train["label"]
    X_val, y_val = val[feature_cols], val["label"]
    X_win, y_win = window[feature_cols], window["label"]

    fitted = clone(model).fit(X_tr, y_tr)
    weights = tune_class_weights(fitted, X_val, y_val)["weights"]
    proba = fitted.predict_proba(X_win)
    auc = macro_roc_auc(y_win, proba, CLASSES)
    raw_pred = fitted.predict(X_win)
    cal_pred = weighted_predict(fitted, X_win, weights)

    base = baseline_scores(y_win)
    row = {
        "raw_f1": macro_f1(y_win, raw_pred), "raw_auc": auc,
        "cal_f1": macro_f1(y_win, cal_pred), "cal_auc": auc,
        "raw_flat_share": float(np.mean(raw_pred == FLAT)),
        "cal_flat_share": float(np.mean(cal_pred == FLAT)),
        "n_window": len(window), **base,
    }
    for variant in ("raw", "cal"):
        row[f"beats_persistence_{variant}"] = bool(row[f"{variant}_f1"] > base["persistence_f1"])
        row[f"dm_met_{variant}"] = data_mining_criterion_met(
            row[f"{variant}_f1"], row[f"{variant}_auc"], base["majority_f1"], base["majority_auc"],
            base["persistence_f1"], base["persistence_auc"])
    return row


def configure_models(names: list[str], train: pd.DataFrame, val: pd.DataFrame,
                       feature_cols: list[str], n_trials: int, seed: int) -> tuple[dict, dict]:
    """Tuned (or default, if n_trials == 0) base models plus ensembles built from them."""
    needed = set(names) | (set(ENSEMBLE_MEMBERS) if set(names) & set(ENSEMBLE_NAMES) else set())
    params, models = {}, {}
    for name in BASE_MODEL_NAMES:
        if name not in needed:
            continue
        if n_trials > 0:
            result = tune_model(name, train[feature_cols], train["label"],
                                val[feature_cols], val["label"], n_trials=n_trials, seed=seed)
            params[name] = result["params"]
        models[name] = build_model(name, params.get(name), random_state=seed)
    members = {n: models[n] for n in ENSEMBLE_MEMBERS if n in models}
    if set(names) & set(ENSEMBLE_NAMES):
        models.update(make_ensembles(members, random_state=seed))
    return {n: models[n] for n in names}, params


def run_walk_forward(features: pd.DataFrame, feature_cols: list[str], model_names: list[str],
                     windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = WINDOWS,
                     n_trials: int = DEFAULT_N_TRIALS, seed: int = 0,
                     return_params: bool = False, log=None):
    """One row per (window, model). Tuning and calibration only ever see pre-window data."""
    rows, all_params = [], {}
    for window_name, (start, end) in windows.items():
        if log:
            log(f"window {window_name}: start")
        train, val, window = window_splits(features, start, end)
        models, params = configure_models(model_names, train, val, feature_cols, n_trials, seed)
        all_params[window_name] = params
        for model_name, model in models.items():
            row = evaluate_window(model, train, val, window, feature_cols)
            rows.append({"window": window_name, "model": model_name, **row})
    out = pd.DataFrame(rows)
    return (out, all_params) if return_params else out


def rank_models(results: pd.DataFrame, variant: str = "cal") -> pd.DataFrame:
    """Agreed selection rule (2026-11-18): most windows beating persistence, then mean
    macro-F1 across windows, then mean ROC-AUC as the tiebreaker. `variant` is "raw" or "cal"."""
    grouped = results.groupby("model")
    table = pd.DataFrame({
        "windows_beating_persistence": grouped[f"beats_persistence_{variant}"].sum().astype(int),
        "windows_meeting_criterion": grouped[f"dm_met_{variant}"].sum().astype(int),
        "mean_f1": grouped[f"{variant}_f1"].mean(),
        "mean_auc": grouped[f"{variant}_auc"].mean(),
        "mean_f1_minus_persistence": grouped.apply(
            lambda g: (g[f"{variant}_f1"] - g["persistence_f1"]).mean(), include_groups=False),
    })
    return table.sort_values(["windows_beating_persistence", "mean_f1", "mean_auc"], ascending=False)
