"""Everything the dashboard's Overview tab shows, computed from the up-to-date data."""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import pandas as pd

from src import config
from src.agents.config import config_for
from src.agents.graph import run_backtest
from src.agents.market_data import attach_sentiment, compute_indicators
from src.agents.session import period_start, sentiment_series
from src.data import update

PERIODS_PER_YEAR = {"1h": 8760, "4h": 2190, "1D": 365}

FIGURE_CAPTIONS = {
    "02_gaps": "Missing hourly candles in the raw snapshot and how the gap policy fills them.",
    "02_outliers": "Robust z-score outlier check on returns.",
    "03_price_vs_returns": "Price is non-stationary; log returns are stationary.",
    "03_return_distribution": "Fat-tailed returns compared with a normal curve.",
    "03_acf": "Autocorrelation: direction barely predictable, volatility clusters.",
    "03_correlation": "Correlation of candidate predictors with the forward return.",
    "03_class_balance": "Up / flat / down balance for candidate horizons and thresholds.",
    "03_bh_sharpe_by_year": "Buy-and-hold Sharpe ratio by year.",
    "03_volume_test": "Does volume carry information about the next move?",
    "04_selected_features": "Features kept after train-only selection.",
    "04_selection_funnel": "How many candidate features survive each selection step.",
    "04_split_timeline": "Chronological train / validation / test split.",
    "05_walk_forward": "Walk-forward evaluation windows.",
    "06_sign_auc_heatmap": "Sign-AUC of every model, horizon and window.",
    "06_model_vs_baselines_1h": "1h model vs. the baselines.",
    "07_pipeline": "The final modeling pipeline.",
    "07_lift_by_iteration": "Improvement over the baselines, iteration by iteration.",
}
SECTIONS = {"02": "Data quality", "03": "EDA", "04": "Features", "05": "Modeling",
            "06": "Evaluation", "07": "Iteration"}


def _series(s: pd.Series, digits: int = 2) -> list[dict]:
    return [{"t": t.isoformat(), "v": round(float(v), digits)} for t, v in s.dropna().items()]


def _sharpe(equity: pd.Series, interval: str) -> float | None:
    r = equity.pct_change().dropna()
    if len(r) < 2 or r.std() == 0:
        return None
    return float(r.mean() / r.std() * np.sqrt(PERIODS_PER_YEAR[interval]))


def backtest_row(ind: pd.DataFrame, interval: str, start: pd.Timestamp, label: str) -> tuple[dict, pd.DataFrame, Any]:
    seg = ind.loc[ind.index >= start]
    graph, df = run_backtest(seg, config_for(interval, analysts=["trend", "momentum", "volatility", "volume", "sentiment"]),
                             interval)
    s = graph.summary()
    return ({"period": label, "interval": interval, "from": seg.index.min().isoformat(), "to": seg.index.max().isoformat(),
             "candles": len(seg), "agent_pct": s["pnl_pct"], "bh_pct": s["bh_pnl_pct"],
             "agent_sharpe": _sharpe(df["equity"], interval), "bh_sharpe": _sharpe(df["bh_equity"], interval),
             "trades": s["n_trades"], "fees": s["fees_paid"], "max_dd_pct": s["max_drawdown_pct"],
             "rating_accuracy": s["rating_accuracy"]}, df, graph)


def figures() -> list[dict]:
    out = []
    for p in sorted(config.FIGURES_DIR.glob("*.png")):
        m = re.match(r"(\d\d)_(.*)", p.stem)
        section = SECTIONS.get(m.group(1), "Other") if m else "Other"
        caption = FIGURE_CAPTIONS.get(p.stem, (m.group(2) if m else p.stem).replace("_", " ").capitalize())
        out.append({"file": p.name, "section": f"{m.group(1) if m else ''} {section}".strip(), "caption": caption})
    return out


def team_model() -> dict:
    path = config.PROJECT_ROOT / "src" / "modeling" / "selected_model.json"
    try:
        choice = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return {k: choice.get(k) for k in ("model", "horizon", "target", "features", "decided",
                                         "walk_forward_cal_sign_auc", "walk_forward_cal_f1", "calibration")}


def build_overview() -> dict[str, Any]:
    status = update.read_status()
    hourly = update.load_latest("1h")
    daily = update.load_latest("1D")
    four = update.load_latest("4h").dropna(subset=["close"])
    close_d = daily["close"]
    log_ret = np.log(close_d / close_d.shift(1))
    sent = sentiment_series()

    ind4 = attach_sentiment(compute_indicators(four), sent["funding"], sent["fear_greed"])
    ind1d = attach_sentiment(compute_indicators(daily.dropna(subset=["close"])), sent["funding"], sent["fear_greed"])
    rows = []
    main_df = None
    for interval, ind in (("4h", ind4), ("1D", ind1d)):
        for label, period in (("Test window (unseen)", "test"), ("Last 90 days", "90d")):
            row, df, _g = backtest_row(ind, interval, period_start(period, ind.index), label)
            rows.append(row)
            if interval == "4h" and period == "test":
                main_df = df
    new_start = config.CUTOFF_DATE + pd.Timedelta("1h")
    if (ind4.index >= new_start).sum() >= 6:
        row, _df, _g = backtest_row(ind4, "4h", new_start, "New data since the snapshot")
        rows.append(row)

    # The agent's current view: warm up on the last 500 4h candles, report the final state.
    graph, _ = run_backtest(ind4.iloc[-500:], config_for("4h", analysts=["trend", "momentum", "volatility",
                                                                         "volume", "sentiment"]), "4h")
    now_state = graph.last_state.to_dict() if graph.last_state else None
    last = hourly.dropna(subset=["close"])
    day_ago = last["close"].asof(last.index.max() - pd.Timedelta("24h"))

    return {
        "data": {
            "snapshot_end": config.CUTOFF_DATE.isoformat(),
            "first": hourly.index.min().isoformat(), "last": last.index.max().isoformat(),
            "hourly_rows": int(len(hourly)), "new_candles": int((hourly.index > config.CUTOFF_DATE).sum()),
            "last_price": float(last["close"].iloc[-1]),
            "change_24h_pct": float((last["close"].iloc[-1] / day_ago - 1) * 100) if day_ago else None,
            "status": status.__dict__,
            "funding_rows": int(len(sent["funding"])), "fear_greed_rows": int(len(sent["fear_greed"])),
        },
        "price": _series(close_d),
        "volatility": _series(log_ret.rolling(30).std() * np.sqrt(365) * 100, 1),
        "fear_greed": _series(sent["fear_greed"], 0),
        "backtests": rows,
        "main_backtest": {
            "equity": [{"t": t.isoformat(), "v": round(v, 2)} for t, v in main_df["equity"].items()],
            "bh": [{"t": t.isoformat(), "v": round(v, 2)} for t, v in main_df["bh_equity"].items()],
            "exposure": [{"t": t.isoformat(), "v": round(v, 3)} for t, v in main_df["exposure"].items()],
        } if main_df is not None else None,
        "agent_now": now_state,
        "agent_now_memory": {"hit_rates": graph.memory.hit_rates(), "lessons": graph.memory.lessons},
        "team_model": team_model(),
        "figures": figures(),
    }
