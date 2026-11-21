"""Walk-forward comparison across horizons (1h / 4h / 1D) and targets (binary / vol3 / triple_barrier).

Run from the project root (data/interim must exist: execute notebooks/02 first):

    python scripts/run_multi_horizon_experiment.py --horizon 1D
    python scripts/run_multi_horizon_experiment.py --horizon 1h --targets binary --trials 25
    python scripts/run_multi_horizon_experiment.py --horizon 4h --smoke      # 2 trials, 3 models, 1 window

Run one process per (horizon, target) to use all cores. Set BTC_N_JOBS to limit threads per model.

Writes to reports/results/multi_horizon/:
    <horizon>_<target>_wf.csv          one row per (window, model)
    <horizon>_<target>_quarterly.csv   intraday only: refit-per-quarter robustness view
    <horizon>_<target>_ranking.csv     rank_models() for cal and raw
    <horizon>_<target>_meta.json       k, selected features, tuned params per window
    <horizon>_<target>_predictions.parquet  calibrated out-of-sample probabilities (input for the agent)
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
from src.features.indicators import build_features_with_indicators  # noqa: E402
from src.features.short_term import short_term_features  # noqa: E402
from src.features.targets import TARGET_NAMES, TargetBuilder  # noqa: E402
from src.modeling.multi_horizon import rank_models, run_target  # noqa: E402
from src.modeling.train import BASE_MODEL_NAMES, ENSEMBLE_NAMES  # noqa: E402
from src.modeling.walk_forward import WINDOWS  # noqa: E402

BAR_HOURS = {"1h": 1, "4h": 4, "1D": 24}
DEFAULT_TRIALS = {"1h": 25, "4h": 25, "1D": 50}
OUT_DIR = ROOT / "reports" / "results" / "multi_horizon"


def load(rule: str) -> pd.DataFrame:
    return pd.read_parquet(config.INTERIM_DIR / f"{config.SYMBOL.lower()}_{rule}_clean.parquet")


def candidate_features(bars: pd.DataFrame, hourly: pd.DataFrame, bar_hours: int) -> pd.DataFrame:
    """Every candidate feature for one bar size. Targets are added per window, not here."""
    base = build_features_with_indicators(bars).drop(columns="label")
    short = short_term_features(bars, hourly, bar_hours).reindex(base.index)
    out = base.join(short)
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--horizon", required=True, choices=list(BAR_HOURS))
    p.add_argument("--targets", nargs="+", default=list(TARGET_NAMES), choices=list(TARGET_NAMES))
    p.add_argument("--trials", type=int, default=None)
    p.add_argument("--models", nargs="+", default=[*BASE_MODEL_NAMES, *ENSEMBLE_NAMES])
    p.add_argument("--windows", nargs="+", default=list(WINDOWS), choices=list(WINDOWS))
    p.add_argument("--smoke", action="store_true", help="2 trials, LR/LightGBM/soft voting, first window only")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    trials = a.trials if a.trials is not None else DEFAULT_TRIALS[a.horizon]
    models, windows = a.models, {w: WINDOWS[w] for w in a.windows}
    suffix = ""
    if a.smoke:
        trials, models, windows, suffix = 2, ["logistic_regression", "lightgbm", "soft_voting"], dict(list(windows.items())[:1]), "_smoke"

    bar_hours = BAR_HOURS[a.horizon]
    hourly = load("1h")
    bars = hourly if a.horizon == "1h" else load(a.horizon)
    t0 = time.time()
    features = candidate_features(bars, hourly, bar_hours)
    builder = TargetBuilder(bars, hourly, bar_hours)
    print(f"{a.horizon}: {len(features)} rows x {features.shape[1]} candidate features, "
          f"{features.index.min()} -> {features.index.max()}, trials={trials}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for target in a.targets:
        def log(msg: str) -> None:
            print(f"[{a.horizon} +{time.time() - t0:6.0f}s] {msg}", flush=True)

        rows, qrows, preds, meta = run_target(features, builder, target, models, trials, windows,
                                              quarterly=bar_hours < 24, seed=a.seed, log=log)
        stem = OUT_DIR / f"{a.horizon}_{target}{suffix}"
        rows.to_csv(f"{stem}_wf.csv", index=False)
        if len(qrows):
            qrows.to_csv(f"{stem}_quarterly.csv", index=False)
        preds.to_parquet(f"{stem}_predictions.parquet")
        Path(f"{stem}_meta.json").write_text(json.dumps(meta, indent=1, default=str))
        ranking = pd.concat({v: rank_models(rows, v) for v in ("cal", "raw")}, names=["variant"])
        ranking.to_csv(f"{stem}_ranking.csv")
        log(f"{target} done")
        print(ranking.loc["cal"].round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
