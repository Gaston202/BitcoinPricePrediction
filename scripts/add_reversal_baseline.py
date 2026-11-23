"""Add the reversal baseline (decided 2026-11-22) to the finished multi-horizon results, without re-running.

The 2026-11-21 run saved calibrated probabilities but not the class-weighted hard predictions. This
script recovers them by trying every weight vector in the tuning grid and keeping the one that
reproduces the saved cal_f1 exactly (the count of exact matches is reported). It then:

- adds reversal_* baseline columns, beats/p-value vs reversal and cal_beats_all_baselines to <h>_<t>_wf.csv
- rewrites <h>_<t>_ranking.csv
- writes predictions_slim.parquet (selected models only, with cal_pred) for the notebooks

New runs of scripts/run_multi_horizon_experiment.py produce these columns directly.

    python scripts/add_reversal_baseline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.modeling.multi_horizon import (  # noqa: E402
    BOOT_BLOCK, WEIGHT_GRID, beats_all_baselines, block_bootstrap_pvalue, class_auc, classes_for,
    fast_macro_f1, one_hot, rank_models,
)

RES = ROOT / "reports" / "results" / "multi_horizon"
BAR_HOURS = {"1h": 1, "4h": 4, "1D": 24}
# Notebooks need the top binary models and LightGBM for the 3-class targets.
SLIM_MODELS = {"binary": ("lightgbm", "stacking", "logistic_regression"),
               "vol3": ("lightgbm",), "triple_barrier": ("lightgbm",)}


def weight_grid(n_classes: int) -> list[np.ndarray]:
    if n_classes == 2:
        return [np.array([1.0, 1.0])] + [np.array([1.0, w]) for w in
                                          (0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.1, 1.33, 1.67, 2.0, 2.5, 4.0)]
    return [np.ones(3)] + [np.array([a, 1.0, b]) for a in WEIGHT_GRID for b in WEIGHT_GRID]


def recover_pred(proba: np.ndarray, y: np.ndarray, classes, target_f1: float) -> tuple[np.ndarray, int]:
    cl = np.asarray(classes)
    matches = [cl[(proba * w).argmax(1)] for w in weight_grid(len(cl))]
    matches = [p for p in matches if abs(fast_macro_f1(y, p, cl) - target_f1) < 1e-12]
    if not matches:
        raise RuntimeError("no weight vector reproduces the saved cal_f1")
    distinct = {p.tobytes() for p in matches}  # different weights often give the same argmax
    return matches[0], len(distinct)


def main() -> None:
    slim = []
    for h, hours in BAR_HOURS.items():
        for t in ("binary", "vol3", "triple_barrier"):
            wf = pd.read_csv(RES / f"{h}_{t}_wf.csv")
            preds = pd.read_parquet(RES / f"{h}_{t}_predictions.parquet")
            classes = classes_for(t)
            pcols = [f"p_{int(c)}" for c in classes]
            extra, ambiguous = [], 0
            for i, r in wf.iterrows():
                g = preds[(preds.window == r.window) & (preds.model == r.model)].sort_index()
                y = g["y"].to_numpy()
                pred, n_match = recover_pred(g[pcols].to_numpy(), y, classes, r.cal_f1)
                ambiguous += n_match > 1
                per = g["y"].shift(1).to_numpy().copy()
                maj = g["y"].mode().iloc[0]
                per[0] = maj
                rev = -per
                rev[0] = maj
                base = {"majority_f1": r.majority_f1, "persistence_f1": r.persistence_f1,
                        "persistence_auc": r.persistence_auc,
                        "reversal_f1": fast_macro_f1(y, rev, classes),
                        "reversal_auc": class_auc(y, one_hot(rev, classes), classes)}
                extra.append({
                    "reversal_f1": base["reversal_f1"], "reversal_acc": float(np.mean(y == rev)),
                    "reversal_auc": base["reversal_auc"],
                    "reversal_sign_auc": 1 - r.persistence_sign_auc,  # AUC of a negated score = 1 - AUC
                    "raw_beats_reversal": bool(r.raw_f1 > base["reversal_f1"]),
                    "cal_beats_reversal": bool(r.cal_f1 > base["reversal_f1"]),
                    "cal_lift_vs_reversal_pvalue": block_bootstrap_pvalue(y, pred, rev, classes, BOOT_BLOCK[hours]),
                    "cal_beats_all_baselines": beats_all_baselines(r.cal_f1, r.cal_auc, base),
                })
                if r.model in SLIM_MODELS[t]:
                    s = g[pcols + ["y", "fwd", "window", "model", "target"]].copy()
                    s[pcols + ["fwd"]] = s[pcols + ["fwd"]].astype("float32")
                    s["cal_pred"], s["horizon"] = pred.astype("int8"), h
                    slim.append(s)
            wf = pd.concat([wf.drop(columns=[c for c in extra[0] if c in wf]), pd.DataFrame(extra)], axis=1)
            wf.to_csv(RES / f"{h}_{t}_wf.csv", index=False)
            ranking = pd.concat({v: rank_models(wf, v) for v in ("cal", "raw")}, names=["variant"])
            ranking.to_csv(RES / f"{h}_{t}_ranking.csv")
            print(f"{h} {t}: {len(wf)} rows updated, {ambiguous} with >1 distinct recovered prediction", flush=True)
    out = pd.concat(slim)
    out.index.name = "open_time"
    out.to_parquet(RES / "predictions_slim.parquet")
    print(f"predictions_slim.parquet: {len(out)} rows, {(RES / 'predictions_slim.parquet').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
