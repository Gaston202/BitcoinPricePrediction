"""The agent experiments behind docs/13_agent_evaluation.md and notebooks/08_agent_evaluation.ipynb.

Order matters, because only the development period may be used to choose anything:

1. tune_presets      grid-search the hysteresis settings per interval on DEVELOPMENT (by Sharpe)
2. select_team       pick the analyst team per interval on DEVELOPMENT (by Sharpe)
3. evaluate          every team + every baseline on every window, with bootstrap tests
4. ablation          remove one component at a time from the selected team
5. sensitivity       fees, slippage, execution delay and the main rule parameters

All results go to reports/results/agent/ (run with scripts/run_agent_evaluation.py).
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pandas as pd

from src import config
from src.agents.config import INTERVAL_PRESETS
from src.evaluation.agent_backtest import (
    DEVELOPMENT, INTERVALS, TEAMS, WINDOWS, MarketData, available_teams, run_agent, trade_targets,
    window_table,
)
from src.evaluation.portfolio import (
    STRATEGIES, bar_returns, performance, run_baseline, sharpe_difference_test, simulate_exposure,
)

RESULTS_DIR = config.PROJECT_ROOT / "reports" / "results" / "agent"
SELECTION_FILE = RESULTS_DIR / "selection.json"
DEV = "development"

PRESET_GRID = {
    "conviction_smoothing": (3, 6, 12, 24),
    "min_hold_bars": (1, 2, 6, 12),
    "min_rebalance": (0.15, 0.25, 0.35),
}


def _segment(data: MarketData, interval: str, window: str, team: str) -> pd.DataFrame:
    return window_table(data, interval, window, team)


# --------------------------------------------------------------------------- 1. presets
def tune_presets(data: MarketData, intervals=INTERVALS, team: str = "core") -> pd.DataFrame:
    """Sharpe of every grid point on the development period, per interval."""
    rows = []
    for iv in intervals:
        seg = _segment(data, iv, DEV, team)
        for values in itertools.product(*PRESET_GRID.values()):
            params = dict(zip(PRESET_GRID, values))
            _, sim = run_agent(seg, iv, team, **params)
            rows.append({"interval": iv, **params, **performance(sim, iv)})
    return pd.DataFrame(rows)


def best_presets(grid: pd.DataFrame) -> dict[str, dict]:
    """The highest-Sharpe grid point per interval, in INTERVAL_PRESETS format."""
    best = grid.loc[grid.groupby("interval")["sharpe"].idxmax()]
    return {r.interval: {"conviction_smoothing": int(r.conviction_smoothing),
                         "min_hold_bars": int(r.min_hold_bars),
                         "min_rebalance": float(r.min_rebalance)} for r in best.itertuples()}


# --------------------------------------------------------------------------- 2. team
def select_team(data: MarketData, intervals=INTERVALS, teams=None) -> tuple[pd.DataFrame, dict]:
    teams = teams or available_teams()
    rows = []
    for iv, team in itertools.product(intervals, teams):
        _, sim = run_agent(_segment(data, iv, DEV, team), iv, team)
        rows.append({"interval": iv, "team": team, **performance(sim, iv)})
    table = pd.DataFrame(rows)
    chosen = table.loc[table.groupby("interval")["sharpe"].idxmax()].set_index("interval")["team"].to_dict()
    return table, chosen


def save_selection(chosen: dict, presets: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SELECTION_FILE.write_text(json.dumps({
        "chosen_on": f"{DEVELOPMENT[0]:%Y-%m-%d} -> {DEVELOPMENT[1]:%Y-%m-%d} (development period)",
        "rule": "highest Sharpe on the development period",
        "team": chosen, "presets": presets,
    }, indent=2))


def load_selection() -> dict:
    return json.loads(SELECTION_FILE.read_text())


# --------------------------------------------------------------------------- 3. evaluation
def evaluate(data: MarketData, chosen: dict, intervals=INTERVALS, windows=tuple(WINDOWS),
             teams=None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(metrics, sharpe tests, equity curves) for every team and baseline on every window."""
    teams = teams or available_teams()
    metrics, tests, curves = [], [], []
    for iv, win in itertools.product(intervals, windows):
        sample = WINDOWS[win][2]
        returns = {}
        for team in teams:
            seg = _segment(data, iv, win, team)
            _, sim = run_agent(seg, iv, team)
            name = f"agent:{team}"
            metrics.append({"interval": iv, "window": win, "sample": sample, "strategy": name,
                            "selected": team == chosen.get(iv), **performance(sim, iv)})
            returns[name] = bar_returns(sim["equity"])
            curves.append(sim["equity"].rename("equity").to_frame().assign(interval=iv, window=win, strategy=name))
        seg = _segment(data, iv, win, "core")
        for base in STRATEGIES:
            sim = run_baseline(base, seg, iv)
            if sim is None:
                continue
            metrics.append({"interval": iv, "window": win, "sample": sample, "strategy": base,
                            "selected": False, **performance(sim, iv)})
            returns[base] = bar_returns(sim["equity"])
            curves.append(sim["equity"].rename("equity").to_frame().assign(interval=iv, window=win, strategy=base))
        agent = f"agent:{chosen.get(iv, 'core')}"
        for other in ("buy_and_hold", "sma200_filter", "ema20_50_cross", "static_50", "cash", "agent:core"):
            if other in returns and other != agent:
                tests.append({"interval": iv, "window": win, "sample": sample, "agent": agent, "versus": other,
                              **sharpe_difference_test(returns[agent], returns[other], iv)})
    curves_df = pd.concat(curves).rename_axis("timestamp").reset_index()
    return pd.DataFrame(metrics), pd.DataFrame(tests), curves_df


# --------------------------------------------------------------------------- 4. ablation
def ablation_variants(team: list[str]) -> dict[str, dict]:
    """name -> config overrides. Each variant removes exactly one component."""
    variants: dict[str, dict] = {"full": {}}
    for a in team:
        variants[f"without {a}"] = {"analysts": [x for x in team if x != a]}
    variants.update({
        "without debate rebuttals": {"max_debate_rounds": 1},
        "without memory re-weighting": {"memory_reweighting": False},
        "without risk team": {"max_risk_rounds": 0},
        "without conviction smoothing": {"conviction_smoothing": 1},
        "without stop-loss / take-profit": {"stop_atr_mult": 1e6, "take_profit_atr_mult": 1e6},
        "without kill switch": {"max_drawdown": 1.0},
    })
    return variants


def ablation(data: MarketData, chosen: dict, intervals=INTERVALS, windows=(DEV, "2022_bear", "test")) -> pd.DataFrame:
    rows = []
    for iv in intervals:
        team = chosen.get(iv, "core")
        analysts = list(TEAMS[team]) + (["quant"] if iv == "1h" else [])
        for win in windows:
            seg = _segment(data, iv, win, team)
            for name, overrides in ablation_variants(analysts).items():
                _, sim = run_agent(seg, iv, team, **overrides)
                rows.append({"interval": iv, "window": win, "team": team, "variant": name, **performance(sim, iv)})
    out = pd.DataFrame(rows)
    full = out[out["variant"] == "full"].set_index(["interval", "window"])["sharpe"]
    out["sharpe_change"] = out["sharpe"] - out.set_index(["interval", "window"]).index.map(full)
    return out


# --------------------------------------------------------------------------- 5. sensitivity
EXECUTION_GRID = {"fee": (0.0005, 0.001, 0.002), "slippage_bps": (0.0, 5.0, 10.0), "delay_bars": (0, 1)}
PARAMETER_GRID = {
    "stop_atr_mult": (1.5, 2.0, 3.0),
    "take_profit_atr_mult": (3.0, 4.0, 6.0),
    "risk_profile": ("aggressive", "neutral", "conservative"),
    "threshold_scale": (0.75, 1.0, 1.25),
    "smoothing_scale": (0.5, 1.0, 2.0),
}


def _parameter_override(name: str, value, interval: str) -> dict:
    from src.agents.config import DEFAULT_CONFIG
    if name == "threshold_scale":
        return {"rating_thresholds": {k: v * value for k, v in DEFAULT_CONFIG["rating_thresholds"].items()}}
    if name == "smoothing_scale":
        base = INTERVAL_PRESETS[interval]["conviction_smoothing"]
        return {"conviction_smoothing": max(1, int(round(base * value)))}
    return {name: value}


def sensitivity(data: MarketData, chosen: dict, intervals=INTERVALS, windows=(DEV, "test")) -> pd.DataFrame:
    rows = []
    for iv, win in itertools.product(intervals, windows):
        team = chosen.get(iv, "core")
        seg = _segment(data, iv, win, team)
        graph, _ = run_agent(seg, iv, team)
        targets = trade_targets(graph, seg.index)
        # Execution assumptions: re-price the agent's exact trades under other costs and delays.
        for fee, slip, delay in itertools.product(*EXECUTION_GRID.values()):
            sim = simulate_exposure(seg["close"], targets, fee=fee, slippage_bps=slip, delay_bars=delay)
            rows.append({"interval": iv, "window": win, "kind": "execution",
                         "setting": f"fee={fee:.2%} slip={slip:g}bp delay={delay}",
                         "fee": fee, "slippage_bps": slip, "delay_bars": delay, **performance(sim, iv)})
        # Rule parameters: rerun the agent with one parameter changed at a time.
        for name, values in PARAMETER_GRID.items():
            for value in values:
                _, sim = run_agent(seg, iv, team, **_parameter_override(name, value, iv))
                rows.append({"interval": iv, "window": win, "kind": "parameter", "parameter": name,
                             "value": str(value), "setting": f"{name}={value}", **performance(sim, iv)})
    return pd.DataFrame(rows)


def save(df: pd.DataFrame, name: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    if name.endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False, float_format="%.6g")
    return path


def sharpe_rank(metrics: pd.DataFrame) -> pd.DataFrame:
    """Rank of every strategy by Sharpe within each interval x window (1 = best)."""
    m = metrics.copy()
    m["sharpe_rank"] = m.groupby(["interval", "window"])["sharpe"].rank(ascending=False, method="min").astype(int)
    return m

