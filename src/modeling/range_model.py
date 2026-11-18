"""Next-day high/low range model: the risk leg of the portfolio simulation.

The direction classifier (src.modeling.train) says *which way*. This model says *how far*
price is likely to move: tomorrow's high and low as log distances from today's close.
The portfolio uses that for position sizing (trade smaller when a wide range is expected)
and for stop-loss / take-profit levels.

Targets are relative to close[t], never price levels. A price-level regression scores
R^2 ~ 0.99 just by copying today's price, which says nothing (docs/09, Gyamerah 2019).

Volatility clusters, so this target is far more predictable than direction. The HAR model
(Corsi 2009: linear regression on 1-, 5- and 22-day average ranges) is the standard benchmark
in volatility forecasting and is included alongside the ML models.
"""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.features.engineering import build_features
from src.modeling.walk_forward import WINDOWS, window_splits

TARGETS = ["up_next", "down_next"]
SHORT_WINDOW, LONG_WINDOW = 5, 22  # one trading week / one month: the HAR horizons
BASELINE_WINDOW = 20

RANGE_FEATURE_COLUMNS = [
    "up_move_lag1", "down_move_lag1", "hl_range_lag1",
    f"up_move_{SHORT_WINDOW}d", f"down_move_{SHORT_WINDOW}d", f"hl_range_{SHORT_WINDOW}d",
    f"up_move_{LONG_WINDOW}d", f"down_move_{LONG_WINDOW}d", f"hl_range_{LONG_WINDOW}d",
    "close_position",
]
HAR_COLUMNS = RANGE_FEATURE_COLUMNS[:-1]
# Which realised move is "yesterday's value" of each target (the persistence baseline).
PERSISTENCE_COLUMN = {"up_next": "up_move_lag1", "down_next": "down_move_lag1"}


def range_targets(df: pd.DataFrame) -> pd.DataFrame:
    """log(high[t+1] / close[t]) and log(low[t+1] / close[t])."""
    close = df["close"]
    return pd.DataFrame({
        "up_next": np.log(df["high"].shift(-1) / close),
        "down_next": np.log(df["low"].shift(-1) / close),
    })[TARGETS]


def range_features(df: pd.DataFrame) -> pd.DataFrame:
    """Realised range measures up to and including day t (no look-ahead)."""
    prev_close = df["close"].shift(1)
    up = np.log(df["high"] / prev_close)
    down = np.log(df["low"] / prev_close)
    hl = np.log(df["high"] / df["low"])
    out = {"up_move_lag1": up, "down_move_lag1": down, "hl_range_lag1": hl}
    for w in (SHORT_WINDOW, LONG_WINDOW):
        out[f"up_move_{w}d"] = up.rolling(w).mean()
        out[f"down_move_{w}d"] = down.rolling(w).mean()
        out[f"hl_range_{w}d"] = hl.rolling(w).mean()
    # Where today closed inside its own range: 0 = at the low, 1 = at the high.
    out["close_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
    return pd.DataFrame(out)[RANGE_FEATURE_COLUMNS]


def build_range_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Direction features + range features + the two range targets; finite rows only."""
    combined = build_features(df).join([range_features(df), range_targets(df)], how="left")
    return combined.replace([np.inf, -np.inf], np.nan).dropna()


def baseline_predictions(ds: pd.DataFrame, target: str) -> dict[str, pd.Series]:
    """Naive forecasts that use only information available at close[t]."""
    realised = ds[PERSISTENCE_COLUMN[target]]
    return {
        "persistence": realised,
        f"rolling_mean_{BASELINE_WINDOW}d": realised.rolling(BASELINE_WINDOW, min_periods=1).mean(),
    }


def regression_scores(y: pd.Series, pred: np.ndarray, target: str) -> dict:
    """MAE / RMSE / R^2 plus the breach rate a stop or target level would see.

    breach_rate: share of days where the actual high went above the predicted high
    (up_next) or the actual low went below the predicted low (down_next).
    """
    y_arr = np.asarray(y)
    breach = y_arr > pred if target == "up_next" else y_arr < pred
    return {
        "mae": float(mean_absolute_error(y_arr, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_arr, pred))),
        "r2": float(r2_score(y_arr, pred)),
        "breach_rate": float(np.mean(breach)),
    }


def _har():
    select = ColumnTransformer([("har", "passthrough", HAR_COLUMNS)], remainder="drop")
    return Pipeline([("select", select), ("ols", LinearRegression())])


def _ridge(p: dict, rs: int):
    return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(**{"alpha": 1.0, **p}))])


def _random_forest(p: dict, rs: int):
    return RandomForestRegressor(random_state=rs, n_jobs=-1,
                                 **{"n_estimators": 300, "max_depth": 6, "min_samples_leaf": 20, **p})


def _lightgbm(p: dict, rs: int):
    return LGBMRegressor(random_state=rs, n_jobs=-1, verbosity=-1,
                         **{"n_estimators": 300, "num_leaves": 7, "learning_rate": 0.03,
                            "min_child_samples": 30, "subsample": 0.8, "subsample_freq": 1,
                            "colsample_bytree": 0.8, **p})


def _xgboost(p: dict, rs: int):
    return XGBRegressor(random_state=rs, n_jobs=-1, verbosity=0,
                        **{"n_estimators": 300, "max_depth": 2, "learning_rate": 0.03,
                           "subsample": 0.8, "colsample_bytree": 0.8, **p})


_FACTORIES = {"ridge": _ridge, "random_forest": _random_forest, "lightgbm": _lightgbm,
              "xgboost": _xgboost}
RANGE_MODEL_NAMES = ("har", *_FACTORIES)

_SPACES = {
    "ridge": lambda t: {"alpha": t.suggest_float("alpha", 1e-3, 100.0, log=True)},
    "random_forest": lambda t: {
        "max_depth": t.suggest_int("max_depth", 2, 10),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 5, 80),
        "max_features": t.suggest_categorical("max_features", ["sqrt", 0.5, 1.0]),
    },
    "lightgbm": lambda t: {
        "n_estimators": t.suggest_int("n_estimators", 100, 600, step=100),
        "num_leaves": t.suggest_int("num_leaves", 3, 31),
        "learning_rate": t.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "min_child_samples": t.suggest_int("min_child_samples", 10, 100),
        "reg_lambda": t.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
    },
    "xgboost": lambda t: {
        "n_estimators": t.suggest_int("n_estimators", 100, 600, step=100),
        "max_depth": t.suggest_int("max_depth", 1, 5),
        "learning_rate": t.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "min_child_weight": t.suggest_float("min_child_weight", 1.0, 30.0, log=True),
        "reg_lambda": t.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
    },
}


def build_range_model(name: str, params: dict | None = None, random_state: int = 0):
    if name == "har":
        return _har()
    if name not in _FACTORIES:
        raise ValueError(f"unknown range model {name!r}; expected one of {RANGE_MODEL_NAMES}")
    return _FACTORIES[name](params or {}, random_state)


def tune_range_model(name: str, X_tr, y_tr, X_val, y_val, n_trials: int, seed: int = 0) -> dict:
    """TPE search minimizing validation MAE. HAR has no hyperparameters."""
    if name == "har" or n_trials <= 0:
        return {}

    def objective(trial):
        model = build_range_model(name, _SPACES[name](trial), seed).fit(X_tr, y_tr)
        return mean_absolute_error(y_val, model.predict(X_val))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def run_range_walk_forward(ds: pd.DataFrame, feature_cols: list[str], model_names: list[str],
                           windows=WINDOWS, n_trials: int = 30, seed: int = 0,
                           return_predictions: bool = False):
    """One row per (window, target, model or baseline). Fit/tune only on pre-window data."""
    rows, predictions = [], {}
    for window_name, (start, end) in windows.items():
        train, val, window = window_splits(ds, start, end)
        for target in TARGETS:
            y_win = window[target]
            candidates = {name: series.loc[window.index].to_numpy()
                          for name, series in baseline_predictions(ds, target).items()}
            for name in model_names:
                params = tune_range_model(name, train[feature_cols], train[target],
                                          val[feature_cols], val[target], n_trials, seed)
                model = clone(build_range_model(name, params, seed)).fit(train[feature_cols], train[target])
                candidates[name] = model.predict(window[feature_cols])
            for name, pred in candidates.items():
                kind = "baseline" if name not in model_names else "model"
                rows.append({"window": window_name, "target": target, "model": name, "kind": kind,
                             **regression_scores(y_win, pred, target)})
                predictions[(window_name, target, name)] = pd.Series(pred, index=window.index)
    out = pd.DataFrame(rows)
    return (out, predictions) if return_predictions else out
