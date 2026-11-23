"""Backtest the agent team on fixed windows, from frozen inputs only (CRISP-DM Phase 5).

Everything here is reproducible: prices come from the frozen Binance snapshot, sentiment and
news from data/external/, and the quant analyst's LightGBM is retrained on candles strictly
before each window starts.

Evaluation protocol
-------------------
* DEVELOPMENT (2023-01-01 -> 2025-06-30) is the only period used to choose anything: the
  interval presets (scripts/tune_agent_presets.py) and which analyst team is "the" agent.
* The four walk-forward windows of the modeling phase are then reported. Two of them overlap
  the development period, so they are labelled in-sample; only 2022_bear and the test window
  are genuinely out-of-sample for the agent.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src import config
from src.agents.config import config_for
from src.agents.graph import run_backtest
from src.agents.market_data import attach_news, attach_sentiment, compute_indicators
from src.agents.news_sentiment import load_sentiment
from src.data.clean import fill_gaps
from src.data.external import load_fear_greed, load_funding
from src.data.update import load_snapshot
from src.features.returns import resample_ohlcv
from src.modeling.walk_forward import WINDOWS as MODEL_WINDOWS

DEVELOPMENT = (pd.Timestamp("2023-01-01", tz="UTC"), pd.Timestamp("2025-06-30 23:59", tz="UTC"))

# name -> (start, end, how it relates to the development period)
WINDOWS: dict[str, tuple[pd.Timestamp, pd.Timestamp, str]] = {
    "2022_bear": (*MODEL_WINDOWS["2022_bear"], "out-of-sample"),
    "2023_bull": (*MODEL_WINDOWS["2023_bull"], "in-sample (development)"),
    "2025_range": (*MODEL_WINDOWS["2025_range"], "half in development"),
    "test": (MODEL_WINDOWS["original"][0], MODEL_WINDOWS["original"][1] + pd.Timedelta("23h59min"),
             "out-of-sample (test)"),
}
INTERVALS = ("1h", "4h", "1D")

# The analyst teams compared. "core" is the team reported on the marouen branch.
CORE = ["trend", "momentum", "volatility", "volume"]
TEAMS: dict[str, list[str]] = {
    "core": CORE,
    "core+sentiment": CORE + ["sentiment"],
    "hybrid_llm": CORE + ["sentiment", "news"],
    "hybrid_lexicon": CORE + ["sentiment", "news"],
}
TEAM_NEWS_SCORER = {"hybrid_llm": "llm", "hybrid_lexicon": "lexicon"}


def available_teams() -> tuple[str, ...]:
    """Every team whose news scores exist (without LLM scores, hybrid_llm would silently
    equal core+sentiment, so it is left out instead of reported twice)."""
    return tuple(t for t in TEAMS if t not in TEAM_NEWS_SCORER or len(load_sentiment(TEAM_NEWS_SCORER[t])))


def price_frames() -> dict[str, pd.DataFrame]:
    """Cleaned 1h / 4h / 1D candles of the frozen snapshot (never the live updates)."""
    hourly = fill_gaps(load_snapshot())
    frames = {"1h": hourly}
    for rule in config.RESAMPLE_INTERVALS:
        frames[rule] = resample_ohlcv(hourly, rule)
    return frames


@dataclass
class MarketData:
    """Indicator tables per interval, with sentiment and both news scores attached."""

    frames: dict[str, pd.DataFrame]
    tables: dict[tuple[str, str], pd.DataFrame]   # (interval, news scorer) -> indicators

    @classmethod
    def build(cls, intervals=INTERVALS, scorers=("llm", "lexicon")) -> "MarketData":
        frames = price_frames()
        funding, fear_greed = load_funding(), load_fear_greed()
        tables = {}
        for iv in intervals:
            base = attach_sentiment(compute_indicators(frames[iv].dropna(subset=["close"])), funding, fear_greed)
            for scorer in scorers:
                tables[(iv, scorer)] = attach_news(base, load_sentiment(scorer))
        return cls(frames, tables)

    def table(self, interval: str, scorer: str = "llm") -> pd.DataFrame:
        return self.tables[(interval, scorer)]


def with_quant(ind: pd.DataFrame, hourly: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    """Add the LightGBM's P(up) to a 1h table, trained only on candles before `start`."""
    from src.agents.session import add_quant
    out = ind.copy()
    add_quant(out, start, hourly)
    return out


def window_bounds(window: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Start and end of a named window ("development" or a key of WINDOWS)."""
    return DEVELOPMENT if window == "development" else WINDOWS[window][:2]


def window_table(data: MarketData, interval: str, window: str, team: str = "core",
                 use_quant: bool = True) -> pd.DataFrame:
    """The indicator rows of one window, ready for the agent and the baselines."""
    start, end = window_bounds(window)
    ind = data.table(interval, TEAM_NEWS_SCORER.get(team, "llm"))
    if interval == "1h" and use_quant:
        ind = with_quant(ind, data.frames["1h"], start)
    return ind.loc[start:end]


def agent_config(interval: str, team: str, **overrides) -> dict:
    """Interval preset + the team's analysts (+ the quant on 1h) + any overrides.

    An explicit `analysts` override (used by the ablation) is taken as is.
    """
    if "analysts" not in overrides:
        overrides["analysts"] = list(TEAMS[team]) + (["quant"] if interval == "1h" else [])
    return config_for(interval, **overrides)


def run_agent(seg: pd.DataFrame, interval: str, team: str = "core", **overrides):
    """Run the agent on one window. Returns (graph, per-candle table like simulate_exposure's)."""
    graph, steps = run_backtest(seg, agent_config(interval, team, **overrides), interval)
    sim = steps[["price", "exposure", "equity", "fees", "trades"]].copy()
    return graph, sim


def trade_targets(graph, index: pd.DatetimeIndex) -> pd.Series:
    """The agent's executed targets as a sparse series (NaN = no trade on that candle)."""
    trades = pd.DataFrame(graph.wallet.trades)
    out = pd.Series(float("nan"), index=index)
    if len(trades):
        ts = pd.to_datetime(trades["timestamp"], utc=True)
        last = trades.assign(ts=ts).groupby("ts")["target"].last()
        out.loc[last.index] = last.to_numpy()
    return out
