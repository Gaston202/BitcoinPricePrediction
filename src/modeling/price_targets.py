"""Price-level targets on 1h / 4h bars (agreed 2026-11-22): next-bar high/low and the 5-day return.

Targets are log distances from close[t]. They are stationary and comparable across a 4k and a 100k
bitcoin, and are turned back into **price levels** for the output: level = close[t] * exp(prediction).
Trees cannot extrapolate beyond the prices they were trained on, so modelling levels directly fails
whenever BTC makes new highs.

- high_next: log(high[t+1] / close[t])   (>= 0 up to rounding)
- low_next:  log(low[t+1] / close[t])    (<= 0 up to rounding)
- ret_5d:    log(close[t+h] / close[t]), h = 120 bars (1h) or 30 bars (4h)

Every window: features re-selected on inner train (regression-aware filter, count chosen on
inner-validation MAE), mean models tuned with Optuna on validation MAE, LightGBM quantile models
(10 / 50 / 90 %) and a split-conformal shift of the bands on inner validation, so that the 90 % level
is breached about 10 % of the time. The window is scored once.
"""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.features.selection import ANOVA_ALPHA, CORR_THRESHOLD, NOISE_QUANTILE, drop_correlated, low_variance, near_constant
from src.modeling.train import N_JOBS
from src.modeling.walk_forward import WINDOWS, window_splits

QUANTILES = (0.1, 0.5, 0.9)
FEATURE_COUNTS = (10, 15, 20, 30)
SELECTION_MAX_ROWS = 20_000
N_PERMUTATIONS = 5
FIVE_DAYS_BARS = {1: 120, 4: 30}
# HAR horizons in bars: last bar, last day, last week.
HAR_WINDOWS = {1: (24, 168), 4: (6, 42)}
BASELINE_WINDOW = 20
N_BOOT = 500


# --- targets and range features -----------------------------------------------------------------
def price_targets(bars: pd.DataFrame, bar_hours: int) -> pd.DataFrame:
    close = bars["close"]
    h = FIVE_DAYS_BARS[bar_hours]
    return pd.DataFrame({
        "high_next": np.log(bars["high"].shift(-1) / close),
        "low_next": np.log(bars["low"].shift(-1) / close),
        "ret_5d": np.log(close.shift(-h) / close),
    }, index=bars.index)


def range_features(bars: pd.DataFrame, bar_hours: int) -> pd.DataFrame:
    """Realized up/down/range moves up to close[t] and their HAR averages (last bar, day, week)."""
    prev = bars["close"].shift(1)
    up, down = np.log(bars["high"] / prev), np.log(bars["low"] / prev)
    hl = np.log(bars["high"] / bars["low"])
    out = {"up_move_lag1": up, "down_move_lag1": down, "hl_range_lag1": hl}
    for w in HAR_WINDOWS[bar_hours]:
        out[f"up_move_{w}b"] = up.rolling(w).mean()
        out[f"down_move_{w}b"] = down.rolling(w).mean()
        out[f"hl_range_{w}b"] = hl.rolling(w).mean()
    return pd.DataFrame(out, index=bars.index)


def har_columns(target: str, bar_hours: int) -> list[str]:
    stem = {"high_next": "up_move", "low_next": "down_move"}[target]
    return [f"{stem}_lag1", "hl_range_lag1"] + [f"{s}_{w}b" for w in HAR_WINDOWS[bar_hours] for s in (stem, "hl_range")]


# --- models ---------------------------------------------------------------------------------------
def _ridge(p, rs):
    return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(**{"alpha": 1.0, **p}))])


def _random_forest(p, rs):
    return RandomForestRegressor(random_state=rs, n_jobs=N_JOBS,
                                 **{"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 50, **p})


def _lightgbm(p, rs):
    return LGBMRegressor(random_state=rs, n_jobs=N_JOBS, verbosity=-1,
                         **{"n_estimators": 300, "num_leaves": 15, "learning_rate": 0.03, "min_child_samples": 50,
                            "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, **p})


def _xgboost(p, rs):
    return XGBRegressor(random_state=rs, n_jobs=N_JOBS, verbosity=0,
                        **{"n_estimators": 300, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.8,
                           "colsample_bytree": 0.8, **p})


FACTORIES = {"ridge": _ridge, "random_forest": _random_forest, "xgboost": _xgboost, "lightgbm": _lightgbm}
SPACES = {
    "ridge": lambda t: {"alpha": t.suggest_float("alpha", 1e-3, 1000.0, log=True)},
    "random_forest": lambda t: {
        "max_depth": t.suggest_int("max_depth", 3, 12),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 10, 300, log=True),
        "max_features": t.suggest_categorical("max_features", ["sqrt", 0.5, 1.0]),
    },
    "xgboost": lambda t: {
        "n_estimators": t.suggest_int("n_estimators", 100, 600, step=100),
        "max_depth": t.suggest_int("max_depth", 1, 6),
        "learning_rate": t.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "min_child_weight": t.suggest_float("min_child_weight", 1.0, 100.0, log=True),
        "reg_lambda": t.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
    },
    "lightgbm": lambda t: {
        "n_estimators": t.suggest_int("n_estimators", 100, 600, step=100),
        "num_leaves": t.suggest_int("num_leaves", 3, 63, log=True),
        "learning_rate": t.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "min_child_samples": t.suggest_int("min_child_samples", 20, 500, log=True),
        "reg_lambda": t.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
    },
}
MODEL_NAMES = ("har", *FACTORIES)


def build(name: str, params: dict | None = None, seed: int = 0, har_cols: list[str] | None = None):
    if name == "har":
        return Pipeline([("select", ColumnTransformer([("har", "passthrough", har_cols)], remainder="drop")),
                         ("ols", LinearRegression())])
    return FACTORIES[name](params or {}, seed)


def tune(name, X_tr, y_tr, X_val, y_val, n_trials: int, seed: int = 0) -> dict:
    """TPE search minimizing validation MAE. HAR has no hyperparameters."""
    if name == "har" or n_trials <= 0:
        return {}

    def objective(trial):
        m = build(name, SPACES[name](trial), seed).fit(X_tr, y_tr)
        return mean_absolute_error(y_val, m.predict(X_val))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def quantile_model(alpha: float, params: dict, seed: int = 0) -> LGBMRegressor:
    p = {k: v for k, v in params.items()}
    return LGBMRegressor(objective="quantile", alpha=alpha, random_state=seed, n_jobs=N_JOBS, verbosity=-1,
                         **{"n_estimators": 300, "num_leaves": 15, "learning_rate": 0.03, "min_child_samples": 50,
                            "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, **p})


# --- feature selection for a continuous target -------------------------------------------------
def select_regression_features(X_tr, y_tr, X_val, y_val, counts=FEATURE_COUNTS, seed: int = 0) -> dict:
    """Same rules as src.features.selection, with regression tests: a feature is noise if it fails
    the F-test (p > 0.05) AND its MI is <= the 95th percentile of MI against shuffled targets."""
    Xs, ys = X_tr.iloc[-SELECTION_MAX_ROWS:], y_tr.iloc[-SELECTION_MAX_ROWS:]
    drop = set(near_constant(Xs)) | set(low_variance(Xs))
    cols = [c for c in Xs.columns if c not in drop]
    _, pval = f_regression(Xs[cols], ys)
    mi = mutual_info_regression(Xs[cols], ys, random_state=seed)
    rng = np.random.default_rng(seed)
    null = np.array([mutual_info_regression(Xs[cols], rng.permutation(ys.to_numpy()), random_state=seed)
                     for _ in range(N_PERMUTATIONS)])
    noise = (pval > ANOVA_ALPHA) & (mi <= np.quantile(null, NOISE_QUANTILE, axis=0))
    mi_s = pd.Series(mi, index=cols)
    keep = [c for c, n in zip(cols, noise) if not n]
    corr = Xs[keep].corr(method="spearman")
    keep = [c for c in keep if c not in drop_correlated(corr, mi_s, CORR_THRESHOLD)]
    ranked = list(mi_s[keep].sort_values(ascending=False).index)
    scores = {}
    for n in sorted({min(c, len(ranked)) for c in counts}):
        m = _lightgbm({}, seed).fit(X_tr[ranked[:n]], y_tr)
        scores[n] = float(mean_absolute_error(y_val, m.predict(X_val[ranked[:n]])))
    best = min(scores, key=scores.get)
    return {"features": ranked[:best], "n_survivors": len(ranked), "val_mae_by_count": scores}


# --- scoring ---------------------------------------------------------------------------------------
def pinball(y, q, alpha: float) -> float:
    d = np.asarray(y) - np.asarray(q)
    return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d)))


def block_bootstrap_mae_pvalue(y, pred_model, pred_base, block: int, n_boot: int = N_BOOT, seed: int = 0) -> float:
    """Circular block bootstrap of MAE(baseline) - MAE(model). p = share of resamples <= 0 (model not better)."""
    e_m, e_b = np.abs(np.asarray(y) - pred_model), np.abs(np.asarray(y) - pred_base)
    n = len(e_m)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    d = np.empty(n_boot)
    for i in range(n_boot):
        idx = ((rng.integers(0, n, nb)[:, None] + np.arange(block)[None, :]) % n).ravel()[:n]
        d[i] = e_b[idx].mean() - e_m[idx].mean()
    return float(np.mean(d <= 0))


def point_scores(y, pred, close) -> dict:
    y, pred, close = map(np.asarray, (y, pred, close))
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
        "mae_price": float(np.mean(np.abs(close * np.exp(y) - close * np.exp(pred)))),
        "mape_price": float(np.mean(np.abs(np.exp(y - pred) - 1))),
        "sign_acc": float(np.mean(np.sign(y) == np.sign(pred))),
    }


def band_scores(target: str, y, q: dict) -> dict:
    """Pinball loss per quantile, plus the breach rate the agent's stop / take-profit would see."""
    y = np.asarray(y)
    out = {f"pinball_{int(a * 100)}": pinball(y, q[a], a) for a in QUANTILES}
    out["breach_above_q90"] = float(np.mean(y > q[0.9]))
    out["breach_below_q10"] = float(np.mean(y < q[0.1]))
    out["coverage_10_90"] = float(np.mean((y >= q[0.1]) & (y <= q[0.9])))
    out["band_width"] = float(np.mean(q[0.9] - q[0.1]))
    return out


# --- walk-forward -------------------------------------------------------------------------------------
def run_price_target(features: pd.DataFrame, targets: pd.DataFrame, close: pd.Series, target: str,
                     bar_hours: int, model_names=MODEL_NAMES, n_trials: int = 25, windows=WINDOWS,
                     quarterly: bool = True, seed: int = 0, log=None):
    """Returns (window_rows, quarter_rows, predictions, meta) for one target on one bar size."""
    h = 1 if target != "ret_5d" else FIVE_DAYS_BARS[bar_hours]
    block = max(FIVE_DAYS_BARS[bar_hours], h)
    names = [m for m in model_names if not (m == "har" and target == "ret_5d")]
    data = features.join(targets[[target]]).dropna(subset=[target])
    rows, qrows, preds, meta = [], [], [], {}
    for wname, (start, end) in windows.items():
        if log:
            log(f"{target} window {wname}: start")
        tr, va, win = window_splits(data, start, end, embargo=h)
        feat_cols = [c for c in features.columns]
        sel = select_regression_features(tr[feat_cols], tr[target], va[feat_cols], va[target], seed=seed)
        cols = sel["features"]
        har_cols = har_columns(target, bar_hours) if target != "ret_5d" else None
        params, fitted = {}, {}
        for name in names:
            use = list(dict.fromkeys(cols + (har_cols or []))) if name == "har" else cols
            params[name] = tune(name, tr[use], tr[target], va[use], va[target], n_trials, seed)
            fitted[name] = (build(name, params[name], seed, har_cols).fit(tr[use], tr[target]), use)
        # quantile bands with the tuned LightGBM settings, shifted on inner validation (split conformal)
        lgb_p = params.get("lightgbm", {})
        qmodels = {a: quantile_model(a, lgb_p, seed).fit(tr[cols], tr[target]) for a in QUANTILES}
        shift = {}
        for a in QUANTILES:
            resid = va[target].to_numpy() - qmodels[a].predict(va[cols])
            shift[a] = float(np.quantile(resid, a))  # empirical a-quantile of residuals -> coverage a on val
        meta[wname] = {"features": cols, "n_survivors": sel["n_survivors"], "val_mae_by_count": sel["val_mae_by_count"],
                       "params": params, "quantile_shift": shift}
        y, c = win[target], close.reindex(win.index)
        base = baseline_preds(data, tr, win, target, bar_hours)
        cand = {**{f"baseline:{k}": v for k, v in base.items()},
                **{n: m.predict(win[use]) for n, (m, use) in fitted.items()}}
        ref_name = "baseline:zero" if target == "ret_5d" else f"baseline:rolling_mean_{BASELINE_WINDOW}"
        for name, pred in cand.items():
            row = {"window": wname, "target": target, "model": name, "kind": "baseline" if name.startswith("baseline") else "model",
                   "n_window": len(win), "n_features": len(cols), **point_scores(y, pred, c)}
            if not name.startswith("baseline"):
                row["mae_pvalue_vs_ref"] = block_bootstrap_mae_pvalue(y, pred, cand[ref_name], block)
            rows.append(row)
        qraw = {a: qmodels[a].predict(win[cols]) for a in QUANTILES}
        qcal = {a: qraw[a] + shift[a] for a in QUANTILES}
        for variant, q in (("raw", qraw), ("cal", qcal)):
            rows.append({"window": wname, "target": target, "model": f"lgbm_quantile_{variant}", "kind": "quantile",
                         "n_window": len(win), "n_features": len(cols), **point_scores(y, q[0.5], c),
                         **band_scores(target, y, q)})
        pdf = pd.DataFrame({"y": y, "close": c, **{f"pred_{n}": cand[n] for n in fitted},
                            **{f"q{int(a * 100)}": qcal[a] for a in QUANTILES}}, index=win.index)
        pdf["window"], pdf["target"] = wname, target
        preds.append(pdf)
        if quarterly:
            qrows += quarterly_scores(data, close, target, cols, params, shift, bar_hours, start, end, wname, h, seed)
        if log:
            best = min((r for r in rows if r["window"] == wname and r["kind"] == "model"), key=lambda r: r["mae"])
            ref = next(r for r in rows if r["window"] == wname and r["model"] == ref_name)
            cal = next(r for r in rows if r["window"] == wname and r["model"] == "lgbm_quantile_cal")
            log(f"{target} window {wname}: done ({len(cols)} features; best MAE {best['mae']:.5f} {best['model']} vs "
                f"{ref_name} {ref['mae']:.5f}; band coverage {cal['coverage_10_90']:.3f})")
    return pd.DataFrame(rows), pd.DataFrame(qrows), pd.concat(preds), meta


def baseline_preds(data, tr, win, target, bar_hours) -> dict[str, np.ndarray]:
    """Naive forecasts using only information at close[t] (and train means)."""
    if target == "ret_5d":
        return {"zero": np.zeros(len(win)), "drift": np.full(len(win), tr[target].mean())}
    col = {"high_next": "up_move_lag1", "low_next": "down_move_lag1"}[target]
    realised = data[col]
    roll = realised.rolling(BASELINE_WINDOW, min_periods=1).mean()
    return {"persistence": realised.reindex(win.index).to_numpy(),
            f"rolling_mean_{BASELINE_WINDOW}": roll.reindex(win.index).to_numpy()}


def quarterly_scores(data, close, target, cols, params, shift, bar_hours, start, end, wname, h, seed) -> list[dict]:
    """Refit LightGBM (mean + quantiles, window settings) before each quarter and score that quarter."""
    out = []
    edges = list(pd.date_range(start, end, freq="QS", tz="UTC"))
    if not edges or edges[0] > start:
        edges = [start] + edges
    edges.append(end + pd.Timedelta(seconds=1))
    for q0, q1 in zip(edges[:-1], edges[1:]):
        tr, _, _ = window_splits(data, q0, end, embargo=h)
        q = data.loc[(data.index >= q0) & (data.index < q1)]
        if len(q) < 50:
            continue
        m = build("lightgbm", params.get("lightgbm"), seed).fit(tr[cols], tr[target])
        qp = {a: quantile_model(a, params.get("lightgbm", {}), seed).fit(tr[cols], tr[target]).predict(q[cols]) + shift[a]
              for a in (0.1, 0.9)}
        y = q[target].to_numpy()
        base = baseline_preds(data, tr, q, target, bar_hours)
        ref = base["zero"] if target == "ret_5d" else base[f"rolling_mean_{BASELINE_WINDOW}"]
        pred = m.predict(q[cols])
        out.append({"window": wname, "quarter": str(q0.date()), "target": target, "n": len(q),
                    "mae": float(mean_absolute_error(y, pred)), "ref_mae": float(mean_absolute_error(y, ref)),
                    "coverage_10_90": float(np.mean((y >= qp[0.1]) & (y <= qp[0.9])))})
    return out
