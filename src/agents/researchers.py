"""Bull vs. bear debate and the research manager's verdict (rule-based).

Round 1: each researcher opens with its side's evidence, weighted by
analyst weight x analyst confidence x evidence strength.
Rounds 2..max_debate_rounds: each side attacks the other's strongest argument that
has not been rebutted yet. A rebuttal only lands if one of the explicit rules below
applies to the current market; a landed rebuttal discounts that argument.
The research manager then rates the net conviction on TradingAgents' 5-tier scale.
"""

from __future__ import annotations

from src.agents.state import AnalystReport, DebateTurn, Evidence, ResearchPlan

# (tag of the attacked argument, condition on the market, discount, how the attacker says it)
BEAR_REBUTTALS = [
    ("oversold", lambda m: m["downtrend"], 0.5,
     "Oversold readings in a downtrend are a falling knife - they can stay oversold."),
    ("breakout_up", lambda m: not m["volume_confirms_up"], 0.5,
     "The breakout came without a volume spike - many such breakouts fail."),
    ("trend_up", lambda m: m["overbought"], 0.6,
     "The uptrend is real but already overbought - late entries carry poor reward/risk."),
    ("flow_up", lambda m: m["downtrend"], 0.7,
     "Buy pressure inside a downtrend is usually short covering."),
    ("model", lambda m: True, 0.8,
     "The ML model's edge is thin (AUC ~0.55 in the project's walk-forward) - weigh it lightly."),
    ("*", lambda m: m["turbulent"], 0.8,
     "Volatility is elevated, so every bullish signal is noisier than usual."),
]
BULL_REBUTTALS = [
    ("overbought", lambda m: m["uptrend"], 0.5,
     "Overbought in a strong uptrend just means strength - trends stay overbought."),
    ("breakout_down", lambda m: not m["volume_confirms_down"], 0.5,
     "The breakdown had no volume behind it - sellers are not committed."),
    ("trend_down", lambda m: m["oversold"], 0.6,
     "The downtrend is stretched and oversold - selling here means selling the low."),
    ("flow_down", lambda m: m["uptrend"], 0.7,
     "Sell pressure inside an uptrend is profit taking, not distribution."),
    ("crowded_long", lambda m: m["uptrend"], 0.7,
     "Crowded longs matter less while the trend keeps pushing up."),
    ("model", lambda m: True, 0.8,
     "The ML model's edge is thin (AUC ~0.55 in the project's walk-forward) - weigh it lightly."),
    ("*", lambda m: m["turbulent"], 0.8,
     "Volatility is elevated, so every bearish signal is noisier than usual."),
]


def market_conditions(row: dict) -> dict:
    def val(k, default=float("nan")):
        v = row.get(k)
        return default if v is None else v
    e20, e50, rsi = val("ema20"), val("ema50"), val("rsi14", 50)
    vz, cd = val("volume_z", 0), val("candle_dir", 0)
    return {
        "uptrend": e20 > e50,
        "downtrend": e20 < e50,
        "overbought": rsi > 70 or val("bb_pct_b", 0.5) > 1,
        "oversold": rsi < 30 or val("bb_pct_b", 0.5) < 0,
        "turbulent": val("vol_ratio", 1) > 1.5,
        "volume_confirms_up": vz > 1 and cd > 0,
        "volume_confirms_down": vz > 1 and cd < 0,
    }


def _weighted(reports: list[AnalystReport], direction: int) -> list[tuple[Evidence, float]]:
    out = []
    for r in reports:
        for e in r.evidence:
            if e.direction == direction:
                out.append((e, r.weight * max(r.confidence, 0.05)))
    return out


def _strength(items: list[tuple[Evidence, float]]) -> float:
    return sum(e.weight * w for e, w in items)


def _rebut(attacker: str, items: list[tuple[Evidence, float]], rules, market: dict, used: set) -> str:
    candidates = sorted((x for x in items if id(x[0]) not in used), key=lambda x: -x[0].weight * x[1])
    for ev, _w in candidates:
        for tag, cond, discount, line in rules:
            if (tag == ev.tag or tag == "*") and cond(market):
                used.add(id(ev))
                ev.discounted *= discount
                return f"On \"{ev.text}\" ({ev.analyst}): {line} (argument weight x{discount})"
        used.add(id(ev))
        return f"Accepts \"{ev.text}\" ({ev.analyst}) - no counter-argument applies."
    return "Has no further points to attack."


def rate(conviction: float, t: dict) -> str:
    """Research manager's 5-tier scale (TradingAgents: Buy / Overweight / Hold / Underweight / Sell)."""
    return ("Buy" if conviction >= t["buy"] else "Overweight" if conviction >= t["overweight"]
            else "Sell" if conviction <= t["sell"] else "Underweight" if conviction <= t["underweight"]
            else "Hold")


def run_debate(reports: list[AnalystReport], row: dict, max_rounds: int, thresholds: dict) -> ResearchPlan:
    market = market_conditions(row)
    bulls, bears = _weighted(reports, 1), _weighted(reports, -1)
    debate: list[DebateTurn] = []

    def opening(items, side):
        if not items:
            return f"{side}: no supporting evidence from the analysts."
        top = sorted(items, key=lambda x: -x[0].weight * x[1])[:3]
        pts = "; ".join(f"{e.text} [{e.analyst}]" for e, _ in top)
        return f"{len(items)} supporting signal(s), total weight {_strength(items):.2f}. Key points: {pts}."

    debate.append(DebateTurn("Bull Researcher", 1, opening(bulls, "Bull")))
    debate.append(DebateTurn("Bear Researcher", 1, opening(bears, "Bear")))
    used_bull, used_bear = set(), set()
    for rnd in range(2, max_rounds + 1):
        debate.append(DebateTurn("Bull Researcher", rnd, _rebut("bull", bears, BULL_REBUTTALS, market, used_bear)))
        debate.append(DebateTurn("Bear Researcher", rnd, _rebut("bear", bulls, BEAR_REBUTTALS, market, used_bull)))

    bull_s, bear_s = _strength(bulls), _strength(bears)
    total = bull_s + bear_s
    # Net conviction: direction from the balance, magnitude damped when there is little evidence.
    conviction = 0.0 if total == 0 else (bull_s - bear_s) / total * min(1.0, total / 0.8)
    rating = rate(conviction, thresholds)
    winner = "bull" if bull_s > bear_s else "bear" if bear_s > bull_s else "neither"
    rationale = (f"After {max_rounds} round(s), bull case {bull_s:.2f} vs bear case {bear_s:.2f} "
                 f"-> net conviction {conviction:+.2f}: the {winner} side carried the debate. Rating: {rating}.")
    return ResearchPlan(rating, float(conviction), float(bull_s), float(bear_s), rationale, debate)
