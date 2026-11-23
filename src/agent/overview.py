"""Numbers and series for the dashboard's Overview tab, computed from the project's own data."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from src import config
from src.agent import model_service
from src.agent.trader import AgentConfig, TradingAgent
from src.data.clean import quality_report
from src.evaluation.baselines import buy_and_hold, macro_f1, majority_baseline, persistence_baseline
from src.evaluation.criteria import hard_prediction_scores, macro_roc_auc
from src.features.engineering import build_features
from src.features.returns import DOWN, FLAT, UP
from src.modeling.train import FEATURE_COLUMNS

CLASSES = [DOWN, FLAT, UP]
CLASS_NAMES = {DOWN: "down", FLAT: "flat", UP: "up"}

FIGURE_CAPTIONS = {
    "02_gaps": ("Data quality", "Missing hourly candles in the raw Binance snapshot and how the gap policy fills them."),
    "03_price_vs_returns": ("EDA", "Price is non-stationary; log returns are stationary - the reason the model predicts returns."),
    "03_return_distribution": ("EDA", "Fat-tailed return distribution compared with a normal curve."),
    "03_acf": ("EDA", "Autocorrelation: returns are barely predictable, volatility clusters strongly."),
    "03_correlation": ("EDA", "Correlation between candidate predictors and the forward return."),
    "03_class_balance": ("EDA", "Up / flat / down label balance for the candidate horizons and thresholds."),
    "03_bh_sharpe_by_year": ("EDA", "Buy-and-hold Sharpe ratio by year - the bar the agent must beat."),
    "04_feature_correlation": ("Features", "Correlation between the engineered features."),
    "04_label_balance": ("Features", "Label balance in the train / validation / test splits."),
    "05_val_macro_f1": ("Modeling", "Candidate models vs. baselines on validation macro-F1."),
    "05_test_confusion_matrix": ("Modeling", "Selected random forest on the test window."),
    "07_regime_robustness": ("Evaluation", "Walk-forward check across bear, bull and range markets."),
}


def _series(s: pd.Series, digits: int = 2) -> list[dict[str, Any]]:
    return [{"t": t.isoformat(), "v": round(float(v), digits)} for t, v in s.dropna().items()]


def _metadata() -> dict:
    files = sorted(config.RAW_DIR.glob("*_metadata.json"))
    return json.loads(files[-1].read_text()) if files else {}


def build_overview() -> dict[str, Any]:
    bundle = model_service.get_bundle("1D")
    clean = model_service.load_clean("1D")
    raw_files = sorted(config.RAW_DIR.glob("*.parquet"))
    raw = pd.read_parquet(raw_files[-1]) if raw_files else None
    quality = quality_report(raw) if raw is not None else {}

    # --- model on the unseen test window ------------------------------------
    feats = build_features(clean, horizon=1, flat_threshold=bundle.flat_threshold)
    test = feats.loc[feats.index >= bundle.test_start]
    y = test["label"]
    pred = bundle.predict(test[FEATURE_COLUMNS])
    proba = pred[["p_down", "p_flat", "p_up"]].to_numpy()
    plain = bundle.model.predict(test[FEATURE_COLUMNS])

    maj, per = majority_baseline(y), persistence_baseline(y)
    per_pred = y.shift(1)
    mask = per_pred.notna()
    model_rows = [
        {"name": "Majority baseline", "macro_f1": maj["macro_f1"],
         "roc_auc": macro_roc_auc(y, hard_prediction_scores(pd.Series(maj["class"], index=y.index), CLASSES), CLASSES)},
        {"name": "Persistence baseline", "macro_f1": per["macro_f1"],
         "roc_auc": macro_roc_auc(y[mask], hard_prediction_scores(per_pred[mask], CLASSES), CLASSES)},
        {"name": "Random forest (argmax)", "macro_f1": macro_f1(y, plain),
         "roc_auc": macro_roc_auc(y, proba, CLASSES)},
        {"name": "Random forest (calibrated, used by agent)", "macro_f1": macro_f1(y, pred["pred_class"]),
         "roc_auc": macro_roc_auc(y, proba, CLASSES)},
    ]
    confusion = pd.crosstab(y.map(CLASS_NAMES), pred["pred_class"].map(CLASS_NAMES)).reindex(
        index=["down", "flat", "up"], columns=["down", "flat", "up"], fill_value=0)
    mix = {
        "predicted": {CLASS_NAMES[c]: float((pred["pred_class"] == c).mean() * 100) for c in CLASSES},
        "actual": {CLASS_NAMES[c]: float((y == c).mean() * 100) for c in CLASSES},
    }

    # --- agent backtest on the same window -----------------------------------
    agent = TradingAgent(AgentConfig())
    close = clean["close"]
    for t, row in pred.iterrows():
        agent.step(t + pd.Timedelta("1D"), close.at[t], row.to_dict())
    eq = pd.DataFrame(agent.decisions)
    window_close = close.loc[test.index.min():test.index.max()]
    bh = buy_and_hold(window_close, "1D")

    # --- data understanding series -------------------------------------------
    log_ret = np.log(close / close.shift(1))
    vol_30 = log_ret.rolling(30).std() * np.sqrt(365) * 100

    return {
        "dataset": {
            "symbol": config.SYMBOL,
            "first": str(raw.index.min()) if raw is not None else None,
            "last": str(raw.index.max()) if raw is not None else None,
            "raw_rows": int(len(raw)) if raw is not None else None,
            "daily_rows": int(len(clean)),
            "completeness_pct": quality.get("completeness_pct"),
            "missing_rows": quality.get("missing_rows"),
            "longest_gap_hours": quality.get("longest_gap_hours"),
            "metadata": _metadata(),
        },
        "splits": {
            "train_end": bundle.train_end.isoformat(), "val_end": bundle.val_end.isoformat(),
            "test_start": bundle.test_start.isoformat(), "test_end": test.index.max().isoformat(),
            "test_days": int(len(test)),
        },
        "price": _series(close),
        "volatility": _series(vol_30, 1),
        "models": model_rows,
        "confusion": {"labels": ["down", "flat", "up"], "matrix": confusion.to_numpy().tolist()},
        "class_mix": mix,
        "feature_importance": bundle.feature_importance,
        "class_weights": {CLASS_NAMES[k]: v for k, v in bundle.weights.items()},
        "agent_backtest": {
            "equity": [{"t": r.timestamp, "v": round(r.equity, 2)} for r in eq.itertuples()],
            "bh": [{"t": r.timestamp, "v": round(r.bh_equity, 2)} for r in eq.itertuples()],
            "trades": [{"t": pd.Timestamp(t["timestamp"]).isoformat(), "action": t["action"],
                        "price": t["price"]} for t in agent.trades],
            "summary": agent.summary(),
            "bh_sharpe": bh["sharpe"],
        },
        "figures": [{"file": f"{k}.png", "section": s, "caption": c}
                    for k, (s, c) in FIGURE_CAPTIONS.items()
                    if (config.FIGURES_DIR / f"{k}.png").exists()]
                   + [{"file": p.name, "section": "Other", "caption": p.stem.replace("_", " ")}
                      for p in sorted(config.FIGURES_DIR.glob("*.png")) if p.stem not in FIGURE_CAPTIONS],
    }
