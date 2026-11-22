"""Multi-horizon, multi-target walk-forward (iteration agreed 2026-11-21, docs/09 section 4).

For each horizon (1h / 4h / 1D bars), target (binary / vol3 / triple_barrier) and test window:

1. Inner train / inner validation come from rows strictly before the window (one-bar embargo).
2. The target's k is chosen on inner-train rows only (src.features.targets).
3. Features are re-selected on inner train (src.features.selection filter: noise and |rho| > 0.85
   removed), ranked by mutual information. The number kept is chosen on inner validation by
   the log-loss of a default LightGBM. This removes the docs/07 caveat that the fixed 12
   features were chosen on data overlapping the 2022 and 2023 windows.
4. Every base model is tuned with Optuna (validation macro-F1). The ensembles are built from
   the tuned members.
5. Each model is fit on inner train. Its probabilities are then Platt-calibrated on inner
   validation, and class weights are tuned there too.
6. The window is scored once, on two variants: "raw" (uncalibrated argmax) and "cal" (Platt +
   weights). Metrics: macro-F1, accuracy, ROC-AUC, log-loss, Brier, ECE, the shared sign-AUC,
   and a block-bootstrap p-value for the macro-F1 lift over persistence.

Slow single-threaded models (SVM, kNN, sklearn GB) train on the most recent rows only for
intraday bars (TailSubsample). That is the agreed runtime limit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import mutual_info_classif
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import log_loss, roc_auc_score

from src.features.returns import DOWN, FLAT, UP
from src.features.selection import select_features
from src.features.targets import TargetBuilder
from src.modeling.train import BASE_MODEL_NAMES, ENSEMBLE_MEMBERS, ENSEMBLE_NAMES, build_model, make_ensembles
from src.modeling.tuning import tune_model
from src.modeling.walk_forward import WINDOWS, window_splits

SLOW_MODEL_ROWS = {"svm": 10_000, "knn": 20_000, "gradient_boosting": 20_000}
SELECTION_MAX_ROWS = 20_000
FEATURE_COUNTS = (10, 15, 20, 30)
WEIGHT_GRID = tuple(round(0.5 + 0.25 * i, 2) for i in range(11))  # 0.5 ... 3.0
ECE_BINS = 10
N_BOOT = 1000
# Bootstrap block = about one week of bars, so weekly dependence is kept inside a block.
BOOT_BLOCK = {24: 7, 4: 42, 1: 168}


def classes_for(target: str) -> list[float]:
    return [DOWN, UP] if target == "binary" else [DOWN, FLAT, UP]


def _tail(X, n):
    return X.iloc[-n:] if hasattr(X, "iloc") else X[-n:]


class TailSubsample(ClassifierMixin, BaseEstimator):
    """Fit `estimator` on the last `max_rows` rows only (the most recent data)."""

    def __init__(self, estimator=None, max_rows: int | None = None):
        self.estimator = estimator
        self.max_rows = max_rows

    def fit(self, X, y):
        if self.max_rows and len(y) > self.max_rows:
            X, y = _tail(X, self.max_rows), _tail(y, self.max_rows)
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def slow_limit(name: str, intraday: bool) -> int | None:
    return SLOW_MODEL_ROWS.get(name) if intraday else None


def make_base(name: str, params: dict | None, intraday: bool, seed: int = 0):
    model = build_model(name, params, random_state=seed)
    limit = slow_limit(name, intraday)
    return TailSubsample(model, limit) if limit else model


# --- feature selection inside the window -------------------------------------------------------
def select_window_features(X_tr: pd.DataFrame, y_tr: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
                           counts: tuple[int, ...] = FEATURE_COUNTS, seed: int = 0) -> dict:
    Xs, ys = _tail(X_tr, SELECTION_MAX_ROWS), _tail(y_tr, SELECTION_MAX_ROWS)
    survivors = select_features(Xs, ys, random_state=seed)["selected"]
    mi = pd.Series(mutual_info_classif(Xs[survivors], ys, random_state=seed), index=survivors)
    ranked = list(mi.sort_values(ascending=False).index)
    options = sorted({min(c, len(ranked)) for c in counts})
    scores = {}
    for n in options:
        cols = ranked[:n]
        m = LGBMClassifier(n_estimators=200, num_leaves=7, learning_rate=0.05, min_child_samples=30,
                           class_weight="balanced", random_state=seed, verbosity=-1).fit(X_tr[cols], y_tr)
        scores[n] = log_loss(y_val, m.predict_proba(X_val[cols]), labels=m.classes_)
    best = min(scores, key=scores.get)
    return {"features": ranked[:best], "n_survivors": len(ranked), "val_logloss_by_count": scores}


# --- weights, metrics, baselines ---------------------------------------------------------------
def fast_macro_f1(y: np.ndarray, pred: np.ndarray, classes) -> float:
    f1s = []
    for c in classes:
        tp = np.sum((pred == c) & (y == c))
        fp, fn = np.sum((pred == c) & (y != c)), np.sum((pred != c) & (y == c))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def tune_weights(proba: np.ndarray, classes, y: np.ndarray) -> np.ndarray:
    """Per-class multipliers on the probabilities that maximise validation macro-F1."""
    classes = np.asarray(classes)
    best_w = np.ones(len(classes))
    best = fast_macro_f1(y, classes[proba.argmax(1)], classes)
    if len(classes) == 2:
        grid = [np.array([1.0, w]) for w in (0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.1, 1.33, 1.67, 2.0, 2.5, 4.0)]
    else:
        grid = [np.array([a, 1.0, b]) for a in WEIGHT_GRID for b in WEIGHT_GRID]
    for w in grid:
        f1 = fast_macro_f1(y, classes[(proba * w).argmax(1)], classes)
        if f1 > best:
            best, best_w = f1, w
    return best_w


def one_hot(y, classes) -> np.ndarray:
    return (np.asarray(y)[:, None] == np.asarray(classes)[None, :]).astype(float)


def expected_calibration_error(y, proba, classes, n_bins: int = ECE_BINS) -> float:
    """Top-label ECE: |accuracy - confidence| averaged over confidence bins, weighted by bin size."""
    conf = proba.max(1)
    correct = np.asarray(classes)[proba.argmax(1)] == np.asarray(y)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def class_auc(y, scores, classes) -> float:
    y = np.asarray(y)
    if len(classes) == 2:
        return float(roc_auc_score(y == classes[1], scores[:, 1]))
    return float(roc_auc_score(one_hot(y, classes), scores, average="macro", multi_class="ovr"))


def sign_score(proba, classes) -> np.ndarray:
    """P(up) - P(down): the shared score used to compare targets (sign-AUC)."""
    cl = list(classes)
    return proba[:, cl.index(UP)] - proba[:, cl.index(DOWN)]


def prob_metrics(y, proba, classes, fwd) -> dict:
    return {
        "auc": class_auc(y, proba, classes),
        "logloss": float(log_loss(y, np.clip(proba, 1e-12, 1), labels=classes)),
        "brier": float(np.mean(np.sum((one_hot(y, classes) - proba) ** 2, axis=1))),
        "ece": expected_calibration_error(y, proba, classes),
        "sign_auc": float(roc_auc_score(np.asarray(fwd) > 0, sign_score(proba, classes))),
    }


def baselines(y: pd.Series, fwd: pd.Series, last_ret: pd.Series, classes, shift: int = 1) -> dict:
    """Majority (window mode), persistence (previous bar's label) and reversal (its opposite).

    Reversal was added on 2026-11-22: short-term BTC returns are negatively autocorrelated, so at
    1h/4h "bet against the last bar" beats persistence by itself and is the fairer bar to clear.
    """
    yv = y.to_numpy()
    maj = y.mode().iloc[0]
    # Persistence = the last *fully realized* label. For an h-bar target that is the label h bars back:
    # the label one bar back still depends on prices after t (for h > 1 it would leak the future).
    per = y.shift(shift).to_numpy().copy()
    per[:shift] = maj  # first rows have no realized label inside the window: fall back to the majority class
    rev = -per  # UP <-> DOWN, FLAT stays FLAT (labels are -1/0/1)
    rev[:shift] = maj
    return {
        "reversal_f1": fast_macro_f1(yv, rev, classes),
        "reversal_acc": float(np.mean(yv == rev)),
        "reversal_auc": class_auc(yv, one_hot(rev, classes), classes),
        "reversal_sign_auc": float(roc_auc_score(fwd.to_numpy() > 0, -np.sign(last_ret.to_numpy()))),
        "reversal_pred": rev,
        "majority_f1": fast_macro_f1(yv, np.full(len(yv), maj), classes),
        "majority_acc": float(np.mean(yv == maj)),
        "persistence_f1": fast_macro_f1(yv, per, classes),
        "persistence_acc": float(np.mean(yv == per)),
        "persistence_auc": class_auc(yv, one_hot(per, classes), classes),
        "persistence_sign_auc": float(roc_auc_score(fwd.to_numpy() > 0, np.sign(last_ret.to_numpy()))),
        "persistence_pred": per,
    }


def block_bootstrap_pvalue(y, pred_model, pred_base, classes, block: int,
                           n_boot: int = N_BOOT, seed: int = 0) -> float:
    """Circular block bootstrap of macro-F1(model) - macro-F1(baseline). p = share of resamples <= 0."""
    y, pm, pb = map(np.asarray, (y, pred_model, pred_base))
    n = len(y)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = ((rng.integers(0, n, n_blocks)[:, None] + np.arange(block)[None, :]) % n).ravel()[:n]
        diffs[i] = fast_macro_f1(y[idx], pm[idx], classes) - fast_macro_f1(y[idx], pb[idx], classes)
    return float(np.mean(diffs <= 0))


# --- one fit / score ----------------------------------------------------------------------------
def fit_calibrated(model, X_tr, y_tr, X_val, y_val, classes):
    fitted = clone(model).fit(X_tr, y_tr)
    calibrated = CalibratedClassifierCV(FrozenEstimator(fitted), method="sigmoid").fit(X_val, y_val)
    weights = tune_weights(calibrated.predict_proba(X_val), classes, y_val.to_numpy())
    return fitted, calibrated, weights


def score_split(fitted, calibrated, weights, X, y, fwd, base: dict, classes, block: int) -> tuple[dict, np.ndarray]:
    cl = np.asarray(classes)
    yv = y.to_numpy()
    raw_p, cal_p = fitted.predict_proba(X), calibrated.predict_proba(X)
    raw_pred, cal_pred = cl[raw_p.argmax(1)], cl[(cal_p * weights).argmax(1)]
    row = {}
    for v, p, pred in (("raw", raw_p, raw_pred), ("cal", cal_p, cal_pred)):
        row[f"{v}_f1"] = fast_macro_f1(yv, pred, cl)
        row[f"{v}_acc"] = float(np.mean(yv == pred))
        row[f"{v}_flat_share"] = float(np.mean(pred == FLAT))
        row.update({f"{v}_{k}": val for k, val in prob_metrics(yv, p, cl, fwd).items()})
        row[f"{v}_beats_persistence"] = bool(row[f"{v}_f1"] > base["persistence_f1"])
        row[f"{v}_beats_reversal"] = bool(row[f"{v}_f1"] > base["reversal_f1"])
    row["cal_lift_pvalue"] = block_bootstrap_pvalue(yv, cal_pred, base["persistence_pred"], cl, block)
    row["cal_lift_vs_reversal_pvalue"] = block_bootstrap_pvalue(yv, cal_pred, base["reversal_pred"], cl, block)
    row["cal_beats_all_baselines"] = beats_all_baselines(row["cal_f1"], row["cal_auc"], base)
    return row, cal_p, cal_pred


def beats_all_baselines(f1: float, auc: float, base: dict) -> bool:
    """docs/01 data-mining criterion as revised 2026-11-22: beat majority, persistence AND reversal
    on macro-F1, and persistence and reversal on ROC-AUC (a constant majority prediction has AUC 0.5)."""
    return (f1 > base["majority_f1"] and f1 > base["persistence_f1"] and f1 > base["reversal_f1"]
            and auc > max(0.5, base["persistence_auc"], base["reversal_auc"]))


# --- the walk-forward -----------------------------------------------------------------------------
def prepare_window(features: pd.DataFrame, builder: TargetBuilder, target: str, start, end):
    train, val, window = window_splits(features, start, end, embargo=builder.horizon)
    k = builder.choose_k(target, train.index)
    out = []
    for part in (train, val, window):
        y = builder.labels(target, part.index, k)
        keep = y.notna()
        out.append((part[keep], y[keep]))
    return out, k


def tune_models(names, X_tr, y_tr, X_val, y_val, intraday: bool, n_trials: int, seed: int):
    needed = set(names) | (set(ENSEMBLE_MEMBERS) if set(names) & set(ENSEMBLE_NAMES) else set())
    params, models = {}, {}
    for name in BASE_MODEL_NAMES:
        if name not in needed:
            continue
        if n_trials > 0:
            limit = slow_limit(name, intraday) or len(y_tr)
            res = tune_model(name, _tail(X_tr, limit), _tail(y_tr, limit), X_val, y_val, n_trials=n_trials, seed=seed)
            params[name] = res["params"]
        models[name] = make_base(name, params.get(name), intraday, seed)
    if set(names) & set(ENSEMBLE_NAMES):
        models.update(make_ensembles({n: models[n] for n in ENSEMBLE_MEMBERS}, random_state=seed))
    return {n: models[n] for n in names}, params


def run_target(features: pd.DataFrame, builder: TargetBuilder, target: str, model_names: list[str],
               n_trials: int, windows: dict = WINDOWS, quarterly: bool = False, seed: int = 0, log=None):
    """Returns (window_rows, quarter_rows, predictions, meta) for one horizon and target."""
    classes = classes_for(target)
    bar_hours = builder.bar_hours
    intraday = bar_hours < 24
    h = builder.horizon
    block = max(BOOT_BLOCK[bar_hours], h)
    last_ret = np.log(builder.close / builder.close.shift(h))  # last fully realized h-bar return
    rows, qrows, preds, meta = [], [], [], {}
    for wname, (start, end) in windows.items():
        if log:
            log(f"{target} window {wname}: start")
        ((tr, y_tr), (va, y_va), (win, y_win)), k = prepare_window(features, builder, target, start, end)
        feat_cols = list(features.columns)
        sel = select_window_features(tr[feat_cols], y_tr, va[feat_cols], y_va, seed=seed)
        cols = sel["features"]
        models, params = tune_models(model_names, tr[cols], y_tr, va[cols], y_va, intraday, n_trials, seed)
        meta[wname] = {"k": k, "features": cols, "n_survivors": sel["n_survivors"],
                       "val_logloss_by_count": sel["val_logloss_by_count"], "params": params,
                       "label_shares_train": y_tr.value_counts(normalize=True).round(4).to_dict()}
        fwd, lr = builder.fwd.reindex(win.index), last_ret.reindex(win.index)
        base = baselines(y_win, fwd, lr, classes, shift=h)
        base_cols = {k2: v for k2, v in base.items() if not k2.endswith("_pred")}
        for mname, model in models.items():
            fitted, calibrated, w = fit_calibrated(model, tr[cols], y_tr, va[cols], y_va, classes)
            row, cal_p, cal_pred = score_split(fitted, calibrated, w, win[cols], y_win, fwd, base, classes, block)
            rows.append({"window": wname, "model": mname, "target": target, "k": k, "n_features": len(cols),
                         "n_window": len(win), **row, **base_cols})
            pdf = pd.DataFrame(cal_p, index=win.index, columns=[f"p_{int(c)}" for c in classes])
            pdf["y"], pdf["fwd"], pdf["window"], pdf["model"], pdf["target"] = y_win, fwd, wname, mname, target
            pdf["cal_pred"] = cal_pred
            preds.append(pdf)
            if quarterly:
                qrows += quarterly_scores(features, builder, target, k, cols, model, start, end, classes, wname, mname)
        if log:
            best = max((r for r in rows if r["window"] == wname), key=lambda r: r["cal_f1"])
            log(f"{target} window {wname}: done (k={k}, {len(cols)} features, best cal_f1 "
                f"{best['cal_f1']:.3f} {best['model']} vs persistence {base['persistence_f1']:.3f})")
    return pd.DataFrame(rows), pd.DataFrame(qrows), pd.concat(preds) if preds else pd.DataFrame(), meta


def quarterly_scores(features, builder, target, k, cols, model, start, end, classes, wname, mname) -> list[dict]:
    """Refit (tuned params, same features and k) before each quarter of the window and score it.

    No re-tuning: this is a robustness view of the window result, not a new model selection.
    """
    out = []
    edges = list(pd.date_range(start, end, freq="QS", tz="UTC"))
    if not edges or edges[0] > start:
        edges = [start] + edges
    edges.append(end + pd.Timedelta(seconds=1))
    h = builder.horizon
    last_ret = np.log(builder.close / builder.close.shift(h))
    for q0, q1 in zip(edges[:-1], edges[1:]):
        tr, va, _ = window_splits(features, q0, end, embargo=h)
        q = features.loc[(features.index >= q0) & (features.index < q1)]
        parts = []
        for part in (tr, va, q):
            y = builder.labels(target, part.index, k)
            parts.append((part[y.notna()], y[y.notna()]))
        (tr, y_tr), (va, y_va), (q, y_q) = parts
        if len(y_q) < 50:
            continue
        fitted, calibrated, w = fit_calibrated(model, tr[cols], y_tr, va[cols], y_va, classes)
        fwd = builder.fwd.reindex(q.index)
        base = baselines(y_q, fwd, last_ret.reindex(q.index), classes, shift=h)
        cl = np.asarray(classes)
        cal_p = calibrated.predict_proba(q[cols])
        pred = cl[(cal_p * w).argmax(1)]
        out.append({"window": wname, "quarter": str(q0.date()), "model": mname, "target": target,
                    "n": len(q), "cal_f1": fast_macro_f1(y_q.to_numpy(), pred, cl),
                    "cal_sign_auc": float(roc_auc_score(fwd.to_numpy() > 0, sign_score(cal_p, cl))),
                    "persistence_f1": base["persistence_f1"]})
    return out


def rank_models(results: pd.DataFrame, variant: str = "cal") -> pd.DataFrame:
    """Agreed rule + log-loss tiebreaker: windows beating persistence -> mean macro-F1 -> mean ROC-AUC
    -> lower mean log-loss. Also reports significant wins (bootstrap p < 0.05) and the shared sign-AUC."""
    g = results.groupby("model")
    table = pd.DataFrame({
        "windows_beating_persistence": g[f"{variant}_beats_persistence"].sum().astype(int),
        "windows_significant": g["cal_lift_pvalue"].apply(lambda p: int((p < 0.05).sum())) if variant == "cal" else 0,
        "mean_f1": g[f"{variant}_f1"].mean(),
        "mean_auc": g[f"{variant}_auc"].mean(),
        "mean_logloss": g[f"{variant}_logloss"].mean(),
        "mean_sign_auc": g[f"{variant}_sign_auc"].mean(),
        "mean_ece": g[f"{variant}_ece"].mean(),
        "mean_f1_lift": g.apply(lambda d: (d[f"{variant}_f1"] - d["persistence_f1"]).mean(), include_groups=False),
    })
    if f"{variant}_beats_reversal" in results:
        table["windows_beating_reversal"] = g[f"{variant}_beats_reversal"].sum().astype(int)
        table["mean_f1_lift_vs_reversal"] = g.apply(
            lambda d: (d[f"{variant}_f1"] - d["reversal_f1"]).mean(), include_groups=False)
    if variant == "cal" and "cal_lift_vs_reversal_pvalue" in results:
        table["windows_significant_vs_reversal"] = g["cal_lift_vs_reversal_pvalue"].apply(lambda p: int((p < 0.05).sum()))
        table["windows_all_baselines"] = g["cal_beats_all_baselines"].sum().astype(int)
    table["neg_logloss"] = -table["mean_logloss"]
    table = table.sort_values(["windows_beating_persistence", "mean_f1", "mean_auc", "neg_logloss"], ascending=False)
    return table.drop(columns="neg_logloss")
