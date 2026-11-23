"""Trader -> risk team -> portfolio manager (rule-based), plus the fractional paper wallet.

Trader            turns the 5-tier rating into a target BTC share of the wallet, with ATR-based
                  stop-loss and take-profit levels.
Risk team         three fixed personalities review the proposal for `max_risk_rounds` rounds:
                  aggressive (lean in when trend and conviction agree), conservative (cap risk in
                  turbulent, losing or low-conviction situations), neutral (meets in the middle).
Portfolio manager adopts the configured profile's view, then enforces hard limits: stop-loss /
                  take-profit exits, a drawdown kill switch, a cooldown after each trade and a
                  minimum rebalance size so fees don't eat the wallet.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.agents.state import FinalDecision, ResearchPlan, RiskView, TraderProposal


# --------------------------------------------------------------------------- wallet
@dataclass
class Wallet:
    """USD + BTC paper wallet that can hold any BTC share between 0 and 100%."""

    cash: float
    fee: float
    btc: float = 0.0
    avg_entry: float | None = None
    fees_paid: float = 0.0
    trades: list[dict] = field(default_factory=list)

    def value(self, price: float) -> float:
        return self.cash + self.btc * price

    def exposure(self, price: float) -> float:
        v = self.value(price)
        return self.btc * price / v if v > 0 else 0.0

    def rebalance(self, target: float, price: float, timestamp: str, reason: str = "") -> dict | None:
        """Trade so that BTC is `target` of the wallet. Fee is charged on the traded notional."""
        target = min(max(target, 0.0), 1.0)
        value = self.value(price)
        delta_usd = target * value - self.btc * price
        if abs(delta_usd) < 1e-9 * max(value, 1):
            return None
        if delta_usd > 0:  # buy
            spend = min(delta_usd, self.cash)
            fee = spend * self.fee
            qty = (spend - fee) / price
            new_btc = self.btc + qty
            self.avg_entry = ((self.avg_entry or price) * self.btc + price * qty) / new_btc if new_btc else None
            self.btc, self.cash = new_btc, self.cash - spend
            side, notional = "BUY", spend
        else:  # sell
            qty = min(-delta_usd / price, self.btc)
            proceeds = qty * price
            fee = proceeds * self.fee
            self.btc -= qty
            self.cash += proceeds - fee
            if self.btc <= 1e-12:
                self.btc, self.avg_entry = 0.0, None
            side, notional = "SELL", proceeds
        self.fees_paid += fee
        trade = {"timestamp": timestamp, "side": side, "price": price, "notional": notional,
                 "qty": qty, "fee": fee, "exposure_after": self.exposure(price), "reason": reason}
        self.trades.append(trade)
        return trade


# --------------------------------------------------------------------------- trader
def trader(plan: ResearchPlan, row: dict, current_exposure: float, cfg: dict) -> TraderProposal:
    price, atr = row["close"], row.get("atr") or 0.0
    rating = plan.rating
    if rating == "Hold":
        target = current_exposure
    elif rating == "Underweight":
        target = min(current_exposure, cfg["target_exposure"]["Underweight"])
    else:
        target = cfg["target_exposure"][rating]
    action = "Buy" if target > current_exposure + 1e-9 else "Sell" if target < current_exposure - 1e-9 else "Hold"
    stop = price - cfg["stop_atr_mult"] * atr if atr and target > 0 else None
    tp = price + cfg["take_profit_atr_mult"] * atr if atr and target > 0 else None
    reasoning = (f"Research rating {rating} (conviction {plan.conviction:+.2f}) -> target {target:.0%} of the "
                 f"wallet in BTC (now {current_exposure:.0%}).")
    if stop:
        reasoning += f" Stop-loss {stop:,.0f} ({cfg['stop_atr_mult']:g} ATR), take-profit {tp:,.0f}."
    return TraderProposal(action, float(target), float(current_exposure), float(price),
                          stop, tp, reasoning)


# --------------------------------------------------------------------------- risk team
def risk_debate(proposal: TraderProposal, plan: ResearchPlan, row: dict, drawdown: float,
                cfg: dict) -> list[RiskView]:
    views: list[RiskView] = []
    target = proposal.target_exposure
    turbulent = (row.get("vol_ratio") or 1) > cfg["high_vol_ratio"]
    below_long_trend = row.get("sma200") is not None and row["close"] < (row.get("sma200") or 0)
    trend_up = (row.get("ema20") or 0) > (row.get("ema50") or 0)
    cap = cfg["conservative_cap"]
    for rnd in range(1, cfg["max_risk_rounds"] + 1):
        # Aggressive: follow the rating, and press it when trend and conviction agree.
        agg = target
        a_arg = "Agrees with the trader's size."
        if plan.conviction > 0.25 and trend_up and target < 1.0 and proposal.action != "Sell":
            agg = min(1.0, max(target, proposal.current_exposure) + 0.2)
            a_arg = f"Trend and conviction ({plan.conviction:+.2f}) agree - size up to {agg:.0%}."
        views.append(RiskView("Aggressive Analyst", rnd, agg, a_arg))

        # Conservative: cap exposure when something is off.
        cons, worries = target, []
        if turbulent:
            worries.append(f"volatility is high (ratio {row.get('vol_ratio'):.2f})")
        if drawdown < -0.10:
            worries.append(f"the wallet is {drawdown:.0%} below its peak")
        if abs(plan.conviction) < 0.3 and target > proposal.current_exposure:
            worries.append(f"conviction is only {plan.conviction:+.2f}")
        if below_long_trend and target > proposal.current_exposure:
            worries.append("price is below its 200-candle average")
        if worries and cons > cap:
            cons = cap
        c_arg = (f"Caps exposure at {cons:.0%} because " + ", ".join(worries) + "."
                 if worries and cons < target else "No objection to the trader's size.")
        views.append(RiskView("Conservative Analyst", rnd, cons, c_arg))

        # Neutral: midpoint of the two.
        neu = (agg + cons) / 2
        views.append(RiskView("Neutral Analyst", rnd, neu,
                              f"Splits the difference between {agg:.0%} and {cons:.0%}: {neu:.0%}."))
        target = neu  # the next round starts from the neutral compromise
    return views


# --------------------------------------------------------------------------- portfolio manager
@dataclass
class ManagerState:
    last_trade_step: int = -10**9
    paused_until: int = -1
    stop_loss: float | None = None
    take_profit: float | None = None


def portfolio_manager(plan: ResearchPlan, proposal: TraderProposal, views: list[RiskView], row: dict,
                      wallet: Wallet, peak: float, step: int, ms: ManagerState, cfg: dict) -> FinalDecision:
    price = row["close"]
    current = wallet.exposure(price)
    reasons: list[str] = []
    final_round = max((v.round for v in views), default=1)
    chosen = {v.speaker.split()[0].lower(): v for v in views if v.round == final_round}
    profile = cfg["risk_profile"]
    target = chosen[profile].target_exposure if profile in chosen else proposal.target_exposure
    reasons.append(f"Adopts the {profile} risk view: target {target:.0%}.")
    forced = False

    value = wallet.value(price)
    drawdown = value / peak - 1 if peak > 0 else 0.0
    if wallet.btc > 0 and ms.stop_loss and price <= ms.stop_loss:
        target, forced = 0.0, True
        reasons.append(f"STOP-LOSS hit: price {price:,.0f} <= stop {ms.stop_loss:,.0f}. Exit everything.")
    elif wallet.btc > 0 and ms.take_profit and price >= ms.take_profit:
        target, forced = min(target, current / 2), True
        reasons.append(f"TAKE-PROFIT hit at {price:,.0f}: bank half the position.")
    if drawdown <= -cfg["max_drawdown"] and current > 0:
        target, forced = 0.0, True
        ms.paused_until = step + cfg["pause_bars_after_kill"]
        reasons.append(f"KILL SWITCH: wallet {drawdown:.0%} below peak. Go to cash and pause "
                       f"{cfg['pause_bars_after_kill']} candles.")
    elif step < ms.paused_until and target > current:
        target = current
        reasons.append("Trading paused after the kill switch - no new buying yet.")

    change = target - current
    approved = True
    if not forced:
        if abs(change) < cfg["min_rebalance"] - 1e-9:
            approved = False
            reasons.append(f"Change of {change:+.0%} is below the {cfg['min_rebalance']:.0%} minimum - not worth the fees.")
        elif step - ms.last_trade_step < cfg["min_hold_bars"]:
            approved = False
            reasons.append(f"Cooldown: last trade was {step - ms.last_trade_step} candle(s) ago "
                           f"(minimum {cfg['min_hold_bars']}).")
    if not approved:
        target = current
    action = "BUY" if target > current + 1e-9 else "SELL" if target < current - 1e-9 else "HOLD"
    if approved and action != "HOLD":
        reasons.append(f"Approved: {action} to {target:.0%} BTC.")
    elif action == "HOLD" and approved:
        reasons.append("Nothing to change.")
    return FinalDecision(plan.rating, action, approved, float(target), float(current), reasons,
                         proposal.stop_loss, proposal.take_profit)
