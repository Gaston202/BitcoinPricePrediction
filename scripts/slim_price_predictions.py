"""Write reports/results/price_targets/predictions_slim.parquet for the notebooks.

Keeps the high/low predictions only (HAR point forecast and the calibrated 10/50/90% band), as
float32, for 1h and 4h bars. The full prediction files (about 20 MB) stay local.

    python scripts/slim_price_predictions.py
"""

from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parents[1] / "reports" / "results" / "price_targets"

parts = []
for h in ("1h", "4h"):
    for t in ("high_next", "low_next"):
        p = pd.read_parquet(RES / f"{h}_{t}_predictions.parquet")
        keep = p[["y", "close", "pred_har", "q10", "q50", "q90"]].astype("float32")
        parts.append(keep.assign(window=p["window"], target=t, horizon=h))
out = pd.concat(parts)
out.index.name = "open_time"
out.to_parquet(RES / "predictions_slim.parquet")
print(f"{len(out):,} rows, {(RES / 'predictions_slim.parquet').stat().st_size / 1e6:.1f} MB")
