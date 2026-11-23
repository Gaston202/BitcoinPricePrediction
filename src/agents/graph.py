"""TradingAgentsGraph: runs the whole agent pipeline for one candle and executes the decision.

    graph = TradingAgentsGraph(make_config(risk_profile="conservative"))
    state = graph.propagate(row, timestamp)     # row = one line of market_data.compute_indicators
    print(state.decision.rating, state.decision.action)

Same call shape as TradingAgents' propagate(), but every step is a rule.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.agents.analysts import ANALYSTS
from src.agents.config import make_config
from src.agents.execution import ManagerState, Wallet, portfolio_manager, risk_debate, trader
from src.agents.memory import DecisionMemory
from src.agents.researchers import rate, run_debate
from src.agents.state import AgentState


class TradingAgentsGraph:
    def __init__(self, cfg: dict | None = None, interval: str = "1h", memory_log=None) -> None:
        self.cfg = make_config(**(cfg or {}))
        self.interval = interval
        self.wallet = Wallet(cash=self.cfg["initial_cash"], fee=self.cfg["fee"])
        self.memory = DecisionMemory(self.cfg["holding_bars"], self.cfg["memory_window"], memory_log)
        self.ms = ManagerState()
        self.peak = self.cfg["initial_cash"]
        self.step = -1
        self.max_dd = 0.0
        self.smoothed: float | None = None
        self.first_price: float | None = None
        self.last_state: AgentState | None = None

    # ------------------------------------------------------------------ one decision
    def propagate(self, row: dict, timestamp: Any) -> AgentState:
        self.step += 1
        ts = pd.Timestamp(timestamp).isoformat()
        price = float(row["close"])
        if self.first_price is None:
            self.first_price = price
        self.memory.settle(self.step, price, ts)

        reports = []
        for name in self.cfg["analysts"]:
            rep = ANALYSTS[name](row)
            multiplier = self.memory.weight_multiplier(name) if self.cfg["memory_reweighting"] else 1.0
            rep.weight = self.cfg["analyst_weights"][name] * multiplier
            reports.append(rep)

        plan = run_debate(reports, row, self.cfg["max_debate_rounds"], self.cfg["rating_thresholds"])
        # The research manager rates the conviction trend, not one candle's noise.
        span = self.cfg["conviction_smoothing"]
        alpha = 2 / (span + 1)
        raw = plan.conviction
        self.smoothed = raw if self.smoothed is None else alpha * raw + (1 - alpha) * self.smoothed
        if span > 1:
            plan.rating = rate(self.smoothed, self.cfg["rating_thresholds"])
            plan.rationale += (f" Smoothed over ~{span} candles the conviction is {self.smoothed:+.2f} "
                               f"-> final rating {plan.rating}.")
        if self.step == self.ms.paused_until:
            # The kill-switch pause is over: measure the drawdown from today's equity again.
            # Keeping the old peak would re-fire the kill switch on the first candle back in
            # the market (the wallet is still ~25% below it) and sell again at a loss.
            self.peak = self.wallet.value(price)
        current = self.wallet.exposure(price)
        proposal = trader(plan, row, current, self.cfg)
        drawdown = self.wallet.value(price) / self.peak - 1
        views = risk_debate(proposal, plan, row, drawdown, self.cfg)
        decision = portfolio_manager(plan, proposal, views, row, self.wallet, self.peak, self.step, self.ms, self.cfg)

        if decision.action != "HOLD":
            before = self.wallet.btc
            self.wallet.rebalance(decision.target_exposure, price, ts, "; ".join(decision.reasons[-2:]))
            self.ms.last_trade_step = self.step
            if self.wallet.btc > before:  # opened / added: arm fresh exits
                self.ms.stop_loss, self.ms.take_profit = decision.stop_loss, decision.take_profit
            if self.wallet.btc == 0:
                self.ms.stop_loss = self.ms.take_profit = None
            if decision.exit_reason == "take_profit":
                self.ms.take_profit = None      # half banked; the rest runs with the stop
                self.ms.reentry_blocked = True

        equity = self.wallet.value(price)
        self.peak = max(self.peak, equity)
        self.max_dd = min(self.max_dd, equity / self.peak - 1)
        self.memory.record(self.step, ts, price, plan.rating, {r.analyst: r.score for r in reports if r.confidence > 0})
        state = AgentState(
            timestamp=ts, price=price, interval=self.interval, reports=reports, plan=plan,
            proposal=proposal, risk_views=views, decision=decision,
            market={k: row.get(k) for k in ("atr_pct", "vol_ratio", "rsi14", "ema20", "ema50", "sma200",
                                            "funding_rate", "fear_greed", "long_short_ratio", "p_up",
                                            "news_score_3d")},
        )
        self.last_state = state
        return state

    # ------------------------------------------------------------------ between decisions
    def mark(self, price: float, timestamp: Any) -> dict:
        """Mark the wallet to a live price and enforce the stop-loss intrabar."""
        exit_trade = None
        if self.wallet.btc > 0 and self.ms.stop_loss and price <= self.ms.stop_loss:
            exit_trade = self.wallet.rebalance(0.0, price, pd.Timestamp(timestamp).isoformat(),
                                               f"Intrabar stop-loss at {self.ms.stop_loss:,.0f}")
            self.ms.stop_loss = self.ms.take_profit = None
            self.ms.last_trade_step = self.step
        self.peak = max(self.peak, self.wallet.value(price))
        self.max_dd = min(self.max_dd, self.wallet.value(price) / self.peak - 1)
        return {"trade": exit_trade}

    # ------------------------------------------------------------------ reporting
    def summary(self, price: float | None = None) -> dict:
        price = price if price is not None else (self.last_state.price if self.last_state else None)
        init = self.cfg["initial_cash"]
        equity = self.wallet.value(price) if price else init
        bh = init * (1 - self.cfg["fee"]) * price / self.first_price if price and self.first_price else init
        sells = [t for t in self.wallet.trades if t["side"] == "SELL"]
        return {
            "equity": equity, "cash": self.wallet.cash, "btc": self.wallet.btc,
            "exposure": self.wallet.exposure(price) if price else 0.0,
            "price": price, "pnl_pct": (equity / init - 1) * 100,
            "bh_equity": bh, "bh_pnl_pct": (bh / init - 1) * 100, "vs_bh_pct": (equity - bh) / init * 100,
            "n_trades": len(self.wallet.trades), "n_sells": len(sells),
            "fees_paid": self.wallet.fees_paid, "avg_entry": self.wallet.avg_entry,
            "stop_loss": self.ms.stop_loss, "take_profit": self.ms.take_profit,
            "max_drawdown_pct": self.max_dd * 100, "steps": self.step + 1,
            "hit_rates": self.memory.hit_rates(), "lessons": self.memory.lessons,
            "rating_accuracy": self.memory.rating_accuracy(),
        }


def run_backtest(ind: pd.DataFrame, cfg: dict | None = None, interval: str = "1h") -> tuple[TradingAgentsGraph, pd.DataFrame]:
    """Run the graph over every row of an indicator table; returns the graph and a per-step table."""
    graph = TradingAgentsGraph(cfg, interval)
    rows = []
    peak = graph.cfg["initial_cash"]
    for ts, row in zip(ind.index, ind.to_dict("records")):
        s = graph.propagate(row, ts)
        eq = graph.wallet.value(s.price)
        peak = max(peak, eq)
        rows.append({"timestamp": ts, "price": s.price, "rating": s.plan.rating, "conviction": s.plan.conviction,
                     "action": s.decision.action, "exposure": graph.wallet.exposure(s.price), "equity": eq,
                     "drawdown": eq / peak - 1, "fees": graph.wallet.fees_paid,
                     "trades": len(graph.wallet.trades)})
    df = pd.DataFrame(rows).set_index("timestamp")
    if len(df):
        df["bh_equity"] = graph.cfg["initial_cash"] * (1 - graph.cfg["fee"]) * df["price"] / df["price"].iloc[0]
    return graph, df
