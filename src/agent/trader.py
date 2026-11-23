"""The trading agent's decision policy, on top of PortfolioSimulator (paper trading only).

Policy (long-only spot wallet: USD cash or BTC, never both partially):
  BUY   when flat (in cash), the calibrated model predicts UP, and the expected
        move covers the round-trip fee (no point trading for less than it costs).
  SELL  when holding BTC and (a) the model predicts DOWN, (b) the stop-loss is
        hit, or (c) the take-profit is hit.
  HOLD  otherwise. Every decision carries a human-readable reason.

`min_confidence` additionally requires the predicted class probability to be at
least that high before acting on it (risk-management stops always fire).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from src import config
from src.features.returns import DOWN, UP
from src.simulation import PortfolioSimulator


@dataclass
class AgentConfig:
    """Defaults = the best of a small grid scored by Sharpe on the VALIDATION window
    for the 1D model (min_confidence 0/45/55/65%, stop-loss off/5/10%, fee-aware on/off).
    They were not chosen by looking at the test window. See docs/08_trading_agent.md.
    """

    initial_cash: float = 1_000.0
    fee: float = config.TRADING_FEE
    stop_loss: float | None = 0.05      # sell if price falls 5% below entry
    take_profit: float | None = None    # e.g. 0.15 = sell if price rises 15% above entry
    min_confidence: float = 0.0         # minimum P(predicted class) to act on a signal
    fee_aware: bool = False             # require expected move > round-trip fee to buy

    def validate(self) -> "AgentConfig":
        if not np.isfinite(self.initial_cash) or self.initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        for name in ("stop_loss", "take_profit"):
            v = getattr(self, name)
            if v is not None and not 0 < v < 1:
                raise ValueError(f"{name} must be between 0 and 1 (or None to disable)")
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("min_confidence must be in [0, 1]")
        return self


def _iso(timestamp: Any) -> str:
    try:
        return pd.Timestamp(timestamp).isoformat()
    except (ValueError, TypeError):
        return str(timestamp)


class TradingAgent:
    """Turns model predictions into wallet actions and keeps a full decision log."""

    def __init__(self, cfg: AgentConfig | None = None) -> None:
        self.cfg = (cfg or AgentConfig()).validate()
        self.sim = PortfolioSimulator(initial_cash=self.cfg.initial_cash, fee=self.cfg.fee)
        self.entry_price: float | None = None
        self.first_price: float | None = None
        self.decisions: list[dict[str, Any]] = []
        self.trades: list[dict[str, Any]] = []
        self.fees_paid = 0.0
        self._peak = self.cfg.initial_cash
        self.max_drawdown = 0.0

    # ------------------------------------------------------------------ policy
    def _choose(self, price: float, pred: dict[str, float]) -> tuple[str, str]:
        holding = self.sim.btc_holding > 0
        cls = pred["pred_class"]
        conf = {UP: pred["p_up"], DOWN: pred["p_down"]}.get(cls, pred["p_flat"])
        exp_pct = float(np.expm1(pred["expected_log_return"])) * 100
        round_trip = 2 * self.cfg.fee * 100

        if holding and self.entry_price:
            change = price / self.entry_price - 1
            if self.cfg.stop_loss and change <= -self.cfg.stop_loss:
                return "SELL", f"Stop-loss: price {change:+.1%} vs entry"
            if self.cfg.take_profit and change >= self.cfg.take_profit:
                return "SELL", f"Take-profit: price {change:+.1%} vs entry"

        if conf < self.cfg.min_confidence and cls in (UP, DOWN):
            return "HOLD", f"Signal too weak ({conf:.0%} < {self.cfg.min_confidence:.0%} confidence)"

        if cls == UP and not holding:
            if self.cfg.fee_aware and exp_pct <= round_trip:
                return "HOLD", f"UP predicted but expected move {exp_pct:+.2f}% doesn't cover {round_trip:.2f}% fees"
            return "BUY", f"Model predicts UP ({conf:.0%}), expected {exp_pct:+.2f}%"
        if cls == DOWN and holding:
            return "SELL", f"Model predicts DOWN ({conf:.0%}), expected {exp_pct:+.2f}%"
        if cls == UP:
            return "HOLD", f"Already in BTC, model still UP ({conf:.0%})"
        if cls == DOWN:
            return "HOLD", f"Already in cash, model predicts DOWN ({conf:.0%})"
        return "HOLD", f"Model predicts FLAT ({pred['p_flat']:.0%}) - no edge"

    # ------------------------------------------------------------------ step
    def step(self, timestamp: Any, price: float, pred: dict[str, float],
             allow_trade: bool = True) -> dict[str, Any]:
        """Decide and execute at `price`. allow_trade=False only marks to market."""
        price = float(price)
        if self.first_price is None:
            self.first_price = price
        action, reason = self._choose(price, pred) if allow_trade else ("HOLD", "Price update")

        value_before = self.sim.cash + self.sim.btc_holding * price
        row = self.sim.step(timestamp, price, action)
        if action in ("BUY", "SELL"):
            fee = value_before * self.cfg.fee
            self.fees_paid += fee
            trade = {"timestamp": timestamp, "action": action, "price": price, "fee": fee}
            if action == "BUY":
                self.entry_price = price
            else:
                trade["pnl_pct"] = ((price * (1 - self.cfg.fee)) /
                                    (self.entry_price / (1 - self.cfg.fee)) - 1) * 100
                self.entry_price = None
            self.trades.append(trade)

        equity = row["total_portfolio_value"]
        self._peak = max(self._peak, equity)
        self.max_drawdown = min(self.max_drawdown, equity / self._peak - 1)
        bh_equity = self.cfg.initial_cash * (1 - self.cfg.fee) * price / self.first_price
        predicted_price = price * float(np.exp(pred.get("expected_log_return", 0.0)))

        record = {
            "timestamp": _iso(timestamp),
            "price": price, "action": action, "reason": reason,
            "p_up": float(pred.get("p_up", np.nan)), "p_flat": float(pred.get("p_flat", np.nan)),
            "p_down": float(pred.get("p_down", np.nan)),
            "pred_class": float(pred.get("pred_class", np.nan)),
            "expected_pct": float(np.expm1(pred.get("expected_log_return", 0.0)) * 100),
            "predicted_price": predicted_price,
            "cash": row["cash"], "btc": row["btc_holding"], "equity": equity,
            "bh_equity": bh_equity,
        }
        self.decisions.append(record)
        return record

    # ------------------------------------------------------------------ stats
    def summary(self) -> dict[str, Any]:
        last = self.decisions[-1] if self.decisions else None
        closed = [t for t in self.trades if "pnl_pct" in t]
        equity = last["equity"] if last else self.cfg.initial_cash
        bh = last["bh_equity"] if last else self.cfg.initial_cash
        return {
            "equity": equity,
            "cash": self.sim.cash,
            "btc": self.sim.btc_holding,
            "price": last["price"] if last else None,
            "pnl_pct": (equity / self.cfg.initial_cash - 1) * 100,
            "bh_equity": bh,
            "bh_pnl_pct": (bh / self.cfg.initial_cash - 1) * 100,
            "vs_bh_pct": (equity - bh) / self.cfg.initial_cash * 100,
            "n_trades": len(self.trades),
            "n_round_trips": len(closed),
            "win_rate": (sum(t["pnl_pct"] > 0 for t in closed) / len(closed) * 100) if closed else None,
            "fees_paid": self.fees_paid,
            "max_drawdown_pct": self.max_drawdown * 100,
            "position": "BTC" if self.sim.btc_holding > 0 else "CASH",
            "entry_price": self.entry_price,
            "steps": len(self.decisions),
            "config": asdict(self.cfg),
        }
