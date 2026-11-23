"""Does daily news sentiment carry information about Bitcoin's next move? (notebook 09)

Alignment: the score of day D is known at the end of D, so it is compared with the return
of day D+1 (`next_ret`). The same-day return (`same_ret`) is shown too: news that only
*describes* today's move is not a forecast.

Three tests, each a standard choice for this question:
* Spearman rank correlation of the score with the next-day return and absolute return
  (robust to the fat tails found in docs/02).
* Granger causality in both directions (score -> return, return -> score), lags 1..3.
* Event study: average cumulative return over the 5 days after the most bullish and most
  bearish 10% of news days, with a bootstrap confidence interval.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

EVENT_QUANTILE = 0.10
EVENT_HORIZON = 5
N_BOOT = 2_000


def align(sentiment: pd.DataFrame, daily_close: pd.Series) -> pd.DataFrame:
    """One row per news day: score, confidence, same-day / next-day / next-5-day returns."""
    s = sentiment.copy()
    s["day"] = pd.to_datetime(s["day"], utc=True)
    s = s.drop_duplicates("day", keep="last").set_index("day").sort_index()
    log_close = np.log(daily_close)
    ret = log_close.diff()
    out = pd.DataFrame({
        "score": s["score"].where(s["confidence"] > 0),
        "confidence": s["confidence"],
        "n_headlines": s["n_headlines"],
    })
    out["same_ret"] = ret.reindex(out.index)
    out["next_ret"] = ret.shift(-1).reindex(out.index)
    out["next_abs_ret"] = out["next_ret"].abs()
    out[f"next_{EVENT_HORIZON}d_ret"] = (log_close.shift(-EVENT_HORIZON) - log_close).reindex(out.index)
    return out.dropna(subset=["score", "next_ret"])


def correlations(aligned: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in ("same_ret", "next_ret", "next_abs_ret", f"next_{EVENT_HORIZON}d_ret"):
        d = aligned[["score", target]].dropna()
        rho, p = stats.spearmanr(d["score"], d[target])
        rows.append({"compared with": target, "spearman_rho": float(rho), "p_value": float(p), "days": len(d)})
    return pd.DataFrame(rows)


def granger(aligned: pd.DataFrame, max_lag: int = 3) -> pd.DataFrame:
    """F-test p-values: does the past of X improve the forecast of Y beyond Y's own past?"""
    from statsmodels.tsa.stattools import grangercausalitytests

    d = aligned[["score", "same_ret"]].dropna()
    rows = []
    for cause, effect in (("score", "same_ret"), ("same_ret", "score")):
        res = grangercausalitytests(d[[effect, cause]], maxlag=max_lag)
        for lag, (tests, _) in res.items():
            rows.append({"direction": f"{cause} -> {effect}", "lag_days": lag,
                         "f_stat": float(tests["ssr_ftest"][0]), "p_value": float(tests["ssr_ftest"][1])})
    return pd.DataFrame(rows)


def event_study(aligned: pd.DataFrame, quantile: float = EVENT_QUANTILE, seed: int = 0) -> pd.DataFrame:
    """Mean 5-day forward return after the most bullish / bearish news days vs. all days."""
    col = f"next_{EVENT_HORIZON}d_ret"
    d = aligned.dropna(subset=[col])
    lo, hi = d["score"].quantile(quantile), d["score"].quantile(1 - quantile)
    groups = {"most bullish news": d[d["score"] >= hi], "most bearish news": d[d["score"] <= lo], "all days": d}
    rng = np.random.default_rng(seed)
    rows = []
    for name, g in groups.items():
        x = g[col].to_numpy()
        boots = rng.choice(x, size=(N_BOOT, len(x)), replace=True).mean(axis=1) if len(x) else np.array([np.nan])
        rows.append({"group": name, "days": len(x), f"mean_{EVENT_HORIZON}d_return": float(np.mean(x)) if len(x) else np.nan,
                     "ci_low": float(np.quantile(boots, 0.025)), "ci_high": float(np.quantile(boots, 0.975)),
                     "hit_rate_up": float(np.mean(x > 0)) if len(x) else np.nan})
    return pd.DataFrame(rows)


def scorer_agreement(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """How similar two scorers' daily scores are (Spearman) and how often their signs agree."""
    x = a.assign(day=pd.to_datetime(a["day"], utc=True)).set_index("day")["score"]
    y = b.assign(day=pd.to_datetime(b["day"], utc=True)).set_index("day")["score"]
    x, y = x.align(y, join="inner")
    mask = (x != 0) & (y != 0)
    rho, p = stats.spearmanr(x, y)
    return {"days": int(len(x)), "spearman_rho": float(rho), "p_value": float(p),
            "sign_agreement": float((np.sign(x[mask]) == np.sign(y[mask])).mean()) if mask.any() else np.nan}
