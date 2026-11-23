"""Robustness check for the quant analyst's model choice (audit item 2.4).

src/modeling/selected_model.json was chosen with all four walk-forward windows, including
"original" (2025-07-09 -> 2026-09-29), which is also the agent's test window. This script
re-applies the same ranking rule *without* that window and reports whether the choice
(1h horizon, binary target, LightGBM among the top models) still holds.

    python scripts/check_model_selection_without_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modeling.multi_horizon import rank_models  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "reports" / "results"
OUT = RESULTS / "agent" / "model_selection_without_test.csv"


def main() -> None:
    rows = []
    for path in sorted((RESULTS / "multi_horizon").glob("*_wf.csv")):
        horizon, target = path.stem.split("_", 1)[0], path.stem.split("_", 1)[1].removesuffix("_wf")
        wf = pd.read_csv(path)
        for scope, data in (("all 4 windows", wf), ("without the test window", wf[wf["window"] != "original"])):
            ranking = rank_models(data).reset_index()
            lgbm = ranking.index[ranking["model"] == "lightgbm"][0] + 1
            rows.append({"horizon": horizon, "target": target, "scope": scope,
                         "best_model": ranking["model"].iloc[0], "lightgbm_rank": int(lgbm),
                         "n_models": len(ranking), "best_mean_sign_auc": ranking["mean_sign_auc"].max(),
                         "lightgbm_mean_sign_auc": float(ranking.loc[ranking["model"] == "lightgbm", "mean_sign_auc"].iloc[0])})
    table = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False, float_format="%.4f")
    print(table.to_string(index=False))
    wo = table[table["scope"] == "without the test window"]
    best = wo.loc[wo["best_mean_sign_auc"].idxmax()]
    lgbm_rank = int(wo[(wo["horizon"] == "1h") & (wo["target"] == "binary")]["lightgbm_rank"].iloc[0])
    print(f"\nWithout the test window the highest sign-AUC is {best.horizon} {best.target} "
          f"({best.best_mean_sign_auc:.4f}); LightGBM ranks {lgbm_rank} of {int(best.n_models)} on 1h binary.")

if __name__ == "__main__":
    main()
