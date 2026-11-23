"""Walk-forward experiment for the price-level targets on 1h / 4h bars (agreed 2026-11-22).

Tasks:
    highlow  next-bar high and low (as % from close, output as price levels) with 10/50/90% bands
    ret5d    5-day log return (120 bars on 1h, 30 on 4h) with 10/50/90% bands
    dir5d    5-day up/down classifier: the full classification pipeline (9 models, calibration, bootstrap)

    python scripts/run_price_targets_experiment.py --horizon 1h --task highlow
    python scripts/run_price_targets_experiment.py --horizon 4h --task dir5d --smoke

Run one process per (horizon, task). Set BTC_N_JOBS to limit threads per model.
Writes to reports/results/price_targets/<horizon>_<task-or-target>_{wf,quarterly,meta,predictions}.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.features.short_term import BAR_HOURS, candidate_pool  # noqa: E402
from src.features.targets import TargetBuilder  # noqa: E402
from src.modeling.multi_horizon import rank_models, run_target  # noqa: E402
from src.modeling.price_targets import FIVE_DAYS_BARS, MODEL_NAMES, price_targets, range_features, run_price_target  # noqa: E402
from src.modeling.train import BASE_MODEL_NAMES, ENSEMBLE_NAMES  # noqa: E402
from src.modeling.walk_forward import WINDOWS  # noqa: E402

OUT_DIR = ROOT / "reports" / "results" / "price_targets"


def load(rule: str) -> pd.DataFrame:
    return pd.read_parquet(config.INTERIM_DIR / f"{config.SYMBOL.lower()}_{rule}_clean.parquet")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--horizon", required=True, choices=["1h", "4h"])
    p.add_argument("--task", required=True, choices=["highlow", "ret5d", "dir5d"])
    p.add_argument("--trials", type=int, default=25)
    p.add_argument("--smoke", action="store_true", help="2 trials, 3 models, first window only, no quarterly folds")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    bar_hours = BAR_HOURS[a.horizon]
    hourly = load("1h")
    bars = hourly if a.horizon == "1h" else load(a.horizon)
    t0 = time.time()
    features = candidate_pool(bars, hourly, bar_hours).join(range_features(bars, bar_hours))
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    windows = dict(list(WINDOWS.items())[:1]) if a.smoke else WINDOWS
    trials = 2 if a.smoke else a.trials
    quarterly = not a.smoke
    suffix = "_smoke" if a.smoke else ""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{a.horizon} {a.task}: {len(features)} rows x {features.shape[1]} candidate features, trials={trials}", flush=True)

    def log(msg: str) -> None:
        print(f"[{a.horizon} {a.task} +{time.time() - t0:6.0f}s] {msg}", flush=True)

    if a.task == "dir5d":
        models = ["logistic_regression", "lightgbm", "soft_voting"] if a.smoke else [*BASE_MODEL_NAMES, *ENSEMBLE_NAMES]
        builder = TargetBuilder(bars, hourly, bar_hours, horizon=FIVE_DAYS_BARS[bar_hours])
        rows, qrows, preds, meta = run_target(features, builder, "binary", models, trials, windows,
                                              quarterly=quarterly, seed=a.seed, log=log)
        stem = OUT_DIR / f"{a.horizon}_dir5d{suffix}"
        ranking = pd.concat({v: rank_models(rows, v) for v in ("cal", "raw")}, names=["variant"])
        ranking.to_csv(f"{stem}_ranking.csv")
        _save(stem, rows, qrows, preds, meta)
        print(ranking.loc["cal"].round(3).to_string(), flush=True)
        return

    models = ["har", "ridge", "lightgbm"] if a.smoke else list(MODEL_NAMES)
    targets = price_targets(bars, bar_hours)
    for target in (["high_next", "low_next"] if a.task == "highlow" else ["ret_5d"]):
        rows, qrows, preds, meta = run_price_target(features, targets, bars["close"], target, bar_hours, models,
                                                    trials, windows, quarterly=quarterly, seed=a.seed, log=log)
        stem = OUT_DIR / f"{a.horizon}_{target}{suffix}"
        _save(stem, rows, qrows, preds, meta)
        summary = rows.groupby(["model", "kind"])[["mae", "r2", "mae_price", "sign_acc"]].mean().sort_values("mae")
        print(summary.round(5).to_string(), flush=True)
        log(f"{target} done")


def _save(stem, rows, qrows, preds, meta) -> None:
    rows.to_csv(f"{stem}_wf.csv", index=False)
    if len(qrows):
        qrows.to_csv(f"{stem}_quarterly.csv", index=False)
    preds.to_parquet(f"{stem}_predictions.parquet")
    Path(f"{stem}_meta.json").write_text(json.dumps(meta, indent=1, default=str))


if __name__ == "__main__":
    main()
