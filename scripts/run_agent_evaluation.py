"""Run the agent experiments and save them to reports/results/agent/.

    python scripts/run_agent_evaluation.py tune          # 1. presets, development period only
    python scripts/run_agent_evaluation.py select        # 2. analyst team, development period only
    python scripts/run_agent_evaluation.py evaluate      # 3. all windows + baselines + bootstrap
    python scripts/run_agent_evaluation.py ablation      # 4. one component removed at a time
    python scripts/run_agent_evaluation.py sensitivity   # 5. costs, delay and rule parameters
    python scripts/run_agent_evaluation.py all           # 2-5 in order (tune is run separately,
                                                          #   because its result is copied into
                                                          #   src/agents/config.py INTERVAL_PRESETS)

Everything reads frozen inputs only (price snapshot + data/external/), so reruns give the same
numbers. Runtime on a laptop: tune ~15 min, the rest ~20 min.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.config import INTERVAL_PRESETS  # noqa: E402
from src.evaluation import agent_experiments as ex  # noqa: E402
from src.evaluation.agent_backtest import MarketData  # noqa: E402

logger = logging.getLogger("agent_evaluation")


def step_tune(data: MarketData) -> None:
    grid = ex.tune_presets(data)
    ex.save(grid, "preset_tuning.csv")
    best = ex.best_presets(grid)
    print("Best presets on the development period (copy into INTERVAL_PRESETS if they differ):")
    print(json.dumps(best, indent=2))
    for iv, params in best.items():
        current = {k: INTERVAL_PRESETS[iv][k] for k in params}
        print(f"  {iv}: {'matches' if current == params else 'DIFFERS from'} src/agents/config.py {current}")


def step_select(data: MarketData) -> dict:
    table, chosen = ex.select_team(data)
    ex.save(table, "team_selection.csv")
    ex.save_selection(chosen, {iv: INTERVAL_PRESETS[iv] for iv in chosen})
    print("Selected team per interval (development Sharpe):", chosen)
    return chosen


def step_evaluate(data: MarketData, chosen: dict) -> None:
    metrics, tests, curves = ex.evaluate(data, chosen)
    ex.save(ex.sharpe_rank(metrics), "metrics.csv")
    ex.save(tests, "sharpe_tests.csv")
    ex.save(curves, "equity_curves.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["tune", "select", "evaluate", "ablation", "sensitivity", "all"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    t0 = time.time()
    data = MarketData.build()
    if args.step == "tune":
        step_tune(data)
    else:
        chosen = step_select(data) if args.step in ("select", "all") else ex.load_selection()["team"]
        if args.step in ("evaluate", "all"):
            step_evaluate(data, chosen)
        if args.step in ("ablation", "all"):
            ex.save(ex.ablation(data, chosen), "ablation.csv")
        if args.step in ("sensitivity", "all"):
            ex.save(ex.sensitivity(data, chosen), "sensitivity.csv")
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {ex.RESULTS_DIR}")


if __name__ == "__main__":
    main()
