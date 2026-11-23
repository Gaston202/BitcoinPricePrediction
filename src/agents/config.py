"""Every tunable of the agent engine in one place (the counterpart of TradingAgents' default_config)."""

from __future__ import annotations

from copy import deepcopy

from src import config as project_config

DEFAULT_CONFIG: dict = {
    # --- which analysts take part (TradingAgents: market, social, news, fundamentals)
    "analysts": ["trend", "momentum", "volatility", "volume", "sentiment", "quant", "news"],
    # base weight of each analyst's vote before the memory re-weights it
    "analyst_weights": {"trend": 1.2, "momentum": 1.0, "volatility": 0.8,
                        "volume": 0.9, "sentiment": 0.7, "quant": 1.0, "news": 0.8},

    # --- debates (TradingAgents: max_debate_rounds / max_risk_discuss_rounds)
    "max_debate_rounds": 2,
    "max_risk_rounds": 1,

    # --- research manager: net conviction in [-1, 1] -> 5-tier rating
    "conviction_smoothing": 6,     # EMA span (candles) of the conviction the rating is based on
    "rating_thresholds": {"buy": 0.40, "overweight": 0.15, "underweight": -0.15, "sell": -0.40},

    # --- trader: target share of the wallet in BTC for each rating (long-only spot)
    "target_exposure": {"Buy": 1.0, "Overweight": 0.6, "Underweight": 0.25, "Sell": 0.0},
    "stop_atr_mult": 2.0,          # stop-loss = entry - 2 x ATR
    "take_profit_atr_mult": 4.0,   # take-profit = entry + 4 x ATR

    # --- risk team
    "risk_profile": "neutral",     # whose view the portfolio manager adopts: aggressive / neutral / conservative
    "high_vol_ratio": 1.5,         # 7-bar / 30-bar volatility above this = turbulent market
    "conservative_cap": 0.5,       # max exposure the conservative analyst accepts in risky conditions

    # --- portfolio manager hard limits
    "min_rebalance": 0.15,         # ignore exposure changes smaller than this (fees)
    "min_hold_bars": 2,            # cooldown after a trade, except risk exits
    "max_drawdown": 0.25,          # kill switch: wallet drawdown beyond 25% -> cash
    "pause_bars_after_kill": 24,

    # --- memory / reflection (TradingAgents: decision log + settlement)
    "holding_bars": 6,             # a decision is scored this many bars later
    "memory_window": 200,          # settled decisions used for each analyst's hit rate
    "memory_reweighting": True,    # False = fixed analyst weights (used by the ablation study)

    # --- wallet
    "initial_cash": 1_000.0,
    "fee": project_config.TRADING_FEE,
}


# Hysteresis per candle size, picked by Sharpe on 2023-01 -> 2025-06 (NOT the 2025-07+ test window).
# Fast candles flip constantly, so they need a smoother conviction and a longer cooldown to
# keep fees under control; see docs/11_rule_based_agents.md.
INTERVAL_PRESETS: dict[str, dict] = {
    "1m": {"conviction_smoothing": 12, "min_hold_bars": 6, "min_rebalance": 0.35},
    "5m": {"conviction_smoothing": 12, "min_hold_bars": 6, "min_rebalance": 0.35},
    "15m": {"conviction_smoothing": 12, "min_hold_bars": 6, "min_rebalance": 0.35},
    "1h": {"conviction_smoothing": 12, "min_hold_bars": 6, "min_rebalance": 0.35},
    "4h": {"conviction_smoothing": 12, "min_hold_bars": 2, "min_rebalance": 0.35},
    "1D": {"conviction_smoothing": 6, "min_hold_bars": 6, "min_rebalance": 0.15},
}


def config_for(interval: str, **overrides) -> dict:
    """DEFAULT_CONFIG + the interval's preset + caller overrides."""
    return make_config(**{**INTERVAL_PRESETS.get(interval, {}), **overrides})


def make_config(**overrides) -> dict:
    """Copy of DEFAULT_CONFIG with top-level keys (or nested dict keys) overridden."""
    cfg = deepcopy(DEFAULT_CONFIG)
    for key, value in overrides.items():
        if key not in cfg:
            raise KeyError(f"unknown config key: {key}")
        if isinstance(cfg[key], dict) and isinstance(value, dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    unknown = set(cfg["analysts"]) - set(cfg["analyst_weights"])
    if unknown:
        raise ValueError(f"unknown analysts: {sorted(unknown)}")
    if cfg["risk_profile"] not in ("aggressive", "neutral", "conservative"):
        raise ValueError("risk_profile must be aggressive, neutral or conservative")
    return cfg
