"""Rule-based multi-agent trading engine for BTC/USDT.

Same organisation as TradingAgents (github.com/TauricResearch/TradingAgents):

    analysts  ->  bull / bear researchers (debate)  ->  research manager (5-tier rating)
              ->  trader (proposal)  ->  risk team (aggressive / conservative / neutral)
              ->  portfolio manager (final, approved decision)  ->  wallet
    + memory: every decision is settled later and each analyst's hit rate re-weights it.

The difference: no LLM anywhere. Every agent is a deterministic set of rules over
market data, so every decision is reproducible and explained line by line.
Everything is paper trading.
"""

from src.agents.config import DEFAULT_CONFIG
from src.agents.graph import TradingAgentsGraph

__all__ = ["DEFAULT_CONFIG", "TradingAgentsGraph"]
