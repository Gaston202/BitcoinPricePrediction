"""Walk-forward comparison of the next-day high/low range models against naive baselines.

Run from the project root:

    python scripts/run_range_experiment.py              # 30 Optuna trials per model per window
    python scripts/run_range_experiment.py --trials 10

Writes to reports/results/:
    range_wf.csv           one row per (window, target, model or baseline)
    range_predictions.csv  every window's predictions, for plots and the portfolio simulation
    range_summary.csv      mean MAE / R^2 / breach rate per (target, model) across windows
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.modeling.range_model import (  # noqa: E402
    RANGE_FEATURE_COLUMNS,
    RANGE_MODEL_NAMES,
    build_range_dataset,
    run_range_walk_forward,
)
from src.modeling.train import FEATURE_COLUMNS  # noqa: E402

OUT_DIR = ROOT / "reports" / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--models", nargs="+", default=list(RANGE_MODEL_NAMES))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(config.INTERIM_DIR / f"{config.SYMBOL.lower()}_1D_clean.parquet")
    ds = build_range_dataset(raw)
    print(f"{len(ds)} rows", flush=True)
    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[range +{time.time() - t0:5.0f}s] {msg}", flush=True)

    results, preds = run_range_walk_forward(ds, FEATURE_COLUMNS + RANGE_FEATURE_COLUMNS, args.models,
                                            n_trials=args.trials, seed=args.seed,
                                            return_predictions=True, log=log)
    results.to_csv(OUT_DIR / "range_wf.csv", index=False)
    long = pd.concat({k: v for k, v in preds.items()}, names=["window", "target", "model", "open_time"])
    long.rename("prediction").reset_index().to_csv(OUT_DIR / "range_predictions.csv", index=False)
    summary = (results.groupby(["target", "model", "kind"])[["mae", "rmse", "r2", "breach_rate"]]
               .mean().sort_values(["target", "mae"]))
    summary.to_csv(OUT_DIR / "range_summary.csv")
    log(f"done, saved to {OUT_DIR}")
    print(summary.round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
