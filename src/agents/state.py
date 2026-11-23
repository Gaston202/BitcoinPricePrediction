"""Structured messages passed between agents (TradingAgents' schemas.py, without the LLM)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

RATINGS = ("Buy", "Overweight", "Hold", "Underweight", "Sell")  # TradingAgents' 5-tier scale


@dataclass
class Evidence:
    """One argument an analyst makes: +1 bullish / -1 bearish, with a strength in (0, 1]."""

    text: str
    direction: int
    strength: float
    tag: str = ""          # machine-readable kind, used by debate rebuttal rules
    analyst: str = ""
    discounted: float = 1.0  # set by the debate when the other side rebuts it

    @property
    def weight(self) -> float:
        return self.strength * self.discounted


@dataclass
class AnalystReport:
    analyst: str
    title: str
    score: float           # -1 (strongly bearish) ... +1 (strongly bullish)
    confidence: float      # 0 = abstains / no data, 1 = very sure
    summary: str
    evidence: list[Evidence] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0    # final weight after memory re-weighting


@dataclass
class DebateTurn:
    speaker: str           # "Bull Researcher" / "Bear Researcher"
    round: int
    text: str


@dataclass
class ResearchPlan:
    rating: str
    conviction: float      # net bull - bear in [-1, 1]
    bull_strength: float
    bear_strength: float
    rationale: str
    debate: list[DebateTurn] = field(default_factory=list)


@dataclass
class TraderProposal:
    action: str            # Buy / Hold / Sell
    target_exposure: float
    current_exposure: float
    entry_price: float
    stop_loss: float | None
    take_profit: float | None
    reasoning: str


@dataclass
class RiskView:
    speaker: str           # Aggressive / Conservative / Neutral Analyst
    round: int
    target_exposure: float
    argument: str


@dataclass
class FinalDecision:
    rating: str
    action: str            # BUY / SELL / HOLD actually executed
    approved: bool
    target_exposure: float
    previous_exposure: float
    reasons: list[str]
    stop_loss: float | None = None
    take_profit: float | None = None


@dataclass
class AgentState:
    timestamp: str
    price: float
    interval: str
    reports: list[AnalystReport]
    plan: ResearchPlan
    proposal: TraderProposal
    risk_views: list[RiskView]
    decision: FinalDecision
    market: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        for r, rep in zip(d["reports"], self.reports):
            for e, ev in zip(r["evidence"], rep.evidence):
                e["weight"] = ev.weight
        return d
