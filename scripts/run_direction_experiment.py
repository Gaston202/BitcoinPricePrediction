"""Walk-forward comparison of every direction classifier on both feature sets.

Run from the project root (data/interim must exist; run notebooks 01-02 first if it doesn't):

    python scripts/run_direction_experiment.py                 # 50 Optuna trials (agreed budget)
    python scripts/run_direction_experiment.py --trials 25     # faster
    python scripts/run_direction_experiment.py --feature-sets selected12

Writes to reports/results/:
    direction_wf_<feature_set>.csv       one row per (window, model)
    direction_params_<feature_set>.json  tuned hyperparameters per window
    direction_ranking_<feature_set>.csv  rank_models() for the calibrated and raw variants
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
from src.features.indicators import PAPER_INDICATOR_COLUMNS, build_features_with_indicators  # noqa: E402
from src.features.short_term import short_term_features  # noqa: E402
from src.modeling.train import BASE_MODEL_NAMES, ENSEMBLE_NAMES, FEATURE_COLUMNS  # noqa: E402
from src.modeling.walk_forward import rank_models, run_walk_forward  # noqa: E402

FEATURE_SETS = {
    "selected12": FEATURE_COLUMNS,
    "plus_paper": FEATURE_COLUMNS + PAPER_INDICATOR_COLUMNS,
}
OUT_DIR = ROOT / "reports" / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=50, help="Optuna trials per model per window")
    parser.add_argument("--feature-sets", nargs="+", default=list(FEATURE_SETS), choices=list(FEATURE_SETS))
    parser.add_argument("--models", nargs="+", default=[*BASE_MODEL_NAMES, *ENSEMBLE_NAMES])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(config.INTERIM_DIR / f"{config.SYMBOL.lower()}_1D_clean.parquet")
    hourly = pd.read_parquet(config.INTERIM_DIR / f"{config.SYMBOL.lower()}_1h_clean.parquet")
    # The 1D manifest includes short-term indicators since the 2026-11-22 re-selection (notebooks/04).
    features = build_features_with_indicators(raw).join(short_term_features(raw, hourly, bar_hours=24))
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    print(f"{len(features)} rows, {features.index.min().date()} -> {features.index.max().date()}", flush=True)

    for tag in args.feature_sets:
        t0 = time.time()

        def log(msg: str, tag: str = tag, t0: float = t0) -> None:
            print(f"[{tag} +{time.time() - t0:5.0f}s] {msg}", flush=True)

        results, params = run_walk_forward(features, FEATURE_SETS[tag], args.models, n_trials=args.trials,
                                           seed=args.seed, return_params=True, log=log)
        results.to_csv(OUT_DIR / f"direction_wf_{tag}.csv", index=False)
        (OUT_DIR / f"direction_params_{tag}.json").write_text(json.dumps(params, indent=1, default=str))
        ranking = pd.concat({v: rank_models(results, v) for v in ("cal", "raw")}, names=["variant"])
        ranking.to_csv(OUT_DIR / f"direction_ranking_{tag}.csv")
        log(f"done, saved to {OUT_DIR}")
        print(ranking.loc["cal"].round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
