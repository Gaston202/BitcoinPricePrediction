"""Decision memory and reflection (TradingAgents: decision log -> settlement -> lessons).

Every decision is logged with each analyst's vote. Once `holding_bars` candles have passed it is
settled against the realized return. Each analyst's hit rate on its settled, non-neutral votes
(Laplace-smoothed, last `memory_window` decisions) becomes a multiplier on its weight:
an analyst that has been right more often than a coin flip gets more say, one that has been
wrong gets less. That is the rule-based equivalent of TradingAgents' reflections.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

VOTE_THRESHOLD = 0.1  # |score| below this is not a directional call


@dataclass
class Pending:
    step: int
    timestamp: str
    price: float
    rating: str
    votes: dict[str, float]


@dataclass
class DecisionMemory:
    holding_bars: int = 6
    window: int = 200
    log_path: Path | None = None
    pending: deque = field(default_factory=deque)
    settled: deque = field(default_factory=deque)
    lessons: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.settled = deque(self.settled, maxlen=self.window)
        self._rates: dict[str, dict] | None = None  # cache, invalidated whenever a decision settles

    def record(self, step: int, timestamp: str, price: float, rating: str, votes: dict[str, float]) -> None:
        self.pending.append(Pending(step, timestamp, price, rating, dict(votes)))

    def settle(self, step: int, price: float, timestamp: str) -> list[dict]:
        done = []
        while self.pending and step - self.pending[0].step >= self.holding_bars:
            p = self.pending.popleft()
            ret = price / p.price - 1
            hits = {a: bool(np.sign(v) == np.sign(ret)) for a, v in p.votes.items() if abs(v) >= VOTE_THRESHOLD}
            rating_dir = {"Buy": 1, "Overweight": 1, "Underweight": -1, "Sell": -1}.get(p.rating, 0)
            rec = {"timestamp": p.timestamp, "settled_at": timestamp, "rating": p.rating,
                   "return": ret, "hits": hits,
                   "rating_correct": None if rating_dir == 0 else bool(np.sign(ret) == rating_dir)}
            self.settled.append(rec)
            done.append(rec)
            if self.log_path:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")
        if done:
            self._rates = None
            self._reflect()
        return done

    def hit_rates(self) -> dict[str, dict]:
        """Laplace-smoothed hit rate of each analyst over the last `window` settled decisions."""
        if self._rates is None:
            self._rates = self._count_hits()
        return self._rates

    def _count_hits(self) -> dict[str, dict]:
        stats: dict[str, list[int]] = {}
        for rec in self.settled:
            for a, hit in rec["hits"].items():
                stats.setdefault(a, [0, 0])
                stats[a][0] += int(hit)
                stats[a][1] += 1
        return {a: {"hits": h, "calls": n, "hit_rate": (h + 2) / (n + 4)} for a, (h, n) in stats.items()}

    def weight_multiplier(self, analyst: str) -> float:
        s = self.hit_rates().get(analyst)
        if not s or s["calls"] < 10:
            return 1.0
        return float(np.clip(1 + (s["hit_rate"] - 0.5) * 4, 0.5, 1.5))

    def rating_accuracy(self) -> float | None:
        calls = [r["rating_correct"] for r in self.settled if r["rating_correct"] is not None]
        return float(np.mean(calls)) if calls else None

    def _reflect(self) -> None:
        rates = self.hit_rates()
        lessons = []
        for a, s in sorted(rates.items(), key=lambda kv: -kv[1]["hit_rate"]):
            if s["calls"] < 10:
                continue
            verdict = ("trusted more" if s["hit_rate"] > 0.53 else "trusted less" if s["hit_rate"] < 0.47
                       else "kept at normal weight")
            lessons.append(f"{a}: right {s['hits']}/{s['calls']} times ({s['hit_rate']:.0%}) -> {verdict}.")
        acc = self.rating_accuracy()
        if acc is not None:
            lessons.append(f"Directional ratings were right {acc:.0%} of the time after {self.holding_bars} candles.")
        self.lessons = lessons
