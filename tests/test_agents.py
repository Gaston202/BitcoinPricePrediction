import numpy as np
import pandas as pd
import pytest

from src.agents.analysts import ANALYSTS, momentum_analyst, quant_analyst, sentiment_analyst, trend_analyst
from src.agents.config import DEFAULT_CONFIG, INTERVAL_PRESETS, config_for, make_config
from src.agents.execution import ManagerState, Wallet, portfolio_manager, risk_debate, trader
from src.agents.graph import TradingAgentsGraph, run_backtest
from src.agents.market_data import compute_indicators
from src.agents.memory import DecisionMemory
from src.agents.researchers import rate, run_debate
from src.agents.state import RATINGS, AnalystReport, Evidence
from src.features.engineering import _build_features_reference, build_features, feature_matrix


def _bars(n=400, drift=0.0, seed=0, freq="4h"):
    rng = np.random.default_rng(seed)
    close = 30_000 * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))
    idx = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    vol = rng.uniform(100, 200, n)
    return pd.DataFrame({"open": close * (1 - rng.normal(0, 0.002, n)), "high": close * 1.006,
                         "low": close * 0.994, "close": close, "volume": vol,
                         "taker_buy_base_volume": vol * rng.uniform(0.4, 0.6, n)}, index=idx)


# ------------------------------------------------------------------ config
def test_make_config_overrides_and_validates():
    cfg = make_config(risk_profile="conservative", analyst_weights={"trend": 2.0})
    assert cfg["risk_profile"] == "conservative" and cfg["analyst_weights"]["trend"] == 2.0
    assert cfg["analyst_weights"]["momentum"] == DEFAULT_CONFIG["analyst_weights"]["momentum"]
    with pytest.raises(KeyError):
        make_config(nope=1)
    with pytest.raises(ValueError):
        make_config(risk_profile="yolo")
    assert config_for("4h")["min_rebalance"] == INTERVAL_PRESETS["4h"]["min_rebalance"]


# ------------------------------------------------------------------ analysts
def test_trend_analyst_reads_direction():
    up = trend_analyst({"close": 110, "ema20": 105, "ema50": 100, "sma200": 90, "atr_pct": 0.02,
                        "macd_hist": 0.001, "macd_hist_prev": 0.0005})
    down = trend_analyst({"close": 90, "ema20": 95, "ema50": 100, "sma200": 110, "atr_pct": 0.02,
                          "macd_hist": -0.001, "macd_hist_prev": -0.0005})
    assert up.score > 0.5 and down.score < -0.5
    assert all(e.direction == 1 for e in up.evidence)


def test_momentum_flags_oversold_and_overbought():
    assert momentum_analyst({"rsi14": 22, "rsi14_prev": 25}).score > 0
    assert momentum_analyst({"rsi14": 80, "rsi14_prev": 78}).score < 0


def test_analysts_abstain_without_data():
    assert sentiment_analyst({}).confidence == 0
    assert quant_analyst({"p_up": float("nan")}).confidence == 0
    assert sentiment_analyst({"funding_rate": 0.001}).score < 0  # crowded longs -> contrarian bearish
    assert quant_analyst({"p_up": 0.6}).score > 0


def test_every_analyst_handles_an_empty_row():
    for fn in ANALYSTS.values():
        rep = fn({})
        assert -1 <= rep.score <= 1 and 0 <= rep.confidence <= 1


# ------------------------------------------------------------------ debate
def _rep(name, direction, strength=0.5, tag="x", conf=1.0):
    return AnalystReport(name, name, direction * strength, conf, "", [Evidence("e", direction, strength, tag, name)])


def test_debate_rating_follows_the_evidence():
    row = {"ema20": 1, "ema50": 1, "rsi14": 50, "vol_ratio": 1}
    bull = run_debate([_rep("a", 1, 0.6), _rep("b", 1, 0.6)], row, 2, DEFAULT_CONFIG["rating_thresholds"])
    bear = run_debate([_rep("a", -1, 0.6), _rep("b", -1, 0.6)], row, 2, DEFAULT_CONFIG["rating_thresholds"])
    assert bull.rating == "Buy" and bear.rating == "Sell"
    assert {t.speaker for t in bull.debate} == {"Bull Researcher", "Bear Researcher"}
    assert max(t.round for t in bull.debate) == 2


def test_rebuttal_discounts_oversold_signal_in_a_downtrend():
    row = {"ema20": 90, "ema50": 100, "rsi14": 25, "vol_ratio": 1}
    rep = _rep("momentum", 1, 0.5, "oversold")
    plan = run_debate([rep], row, 2, DEFAULT_CONFIG["rating_thresholds"])
    assert rep.evidence[0].discounted == pytest.approx(0.5)
    assert "falling knife" in " ".join(t.text for t in plan.debate)


def test_rate_covers_the_five_tiers():
    t = DEFAULT_CONFIG["rating_thresholds"]
    assert [rate(c, t) for c in (0.9, 0.2, 0.0, -0.2, -0.9)] == list(RATINGS)


# ------------------------------------------------------------------ wallet / trader / risk / PM
def test_wallet_rebalances_fractionally_with_fees():
    w = Wallet(cash=1000, fee=0.001)
    w.rebalance(0.5, 100.0, "t0")
    assert w.cash == pytest.approx(500) and w.btc == pytest.approx(500 * 0.999 / 100)
    w.rebalance(0.0, 200.0, "t1")
    assert w.btc == 0 and w.cash == pytest.approx(500 + 4.995 * 200 * 0.999)
    assert w.fees_paid == pytest.approx(0.5 + 4.995 * 200 * 0.001)


def test_conservative_risk_analyst_caps_turbulent_markets():
    cfg = make_config()
    plan = run_debate([_rep("a", 1, 0.6), _rep("b", 1, 0.6)], {"ema20": 2, "ema50": 1}, 1, cfg["rating_thresholds"])
    row = {"close": 100, "atr": 2, "vol_ratio": 2.0, "ema20": 2, "ema50": 1, "sma200": 90}
    prop = trader(plan, row, 0.0, cfg)
    assert prop.target_exposure == 1.0 and prop.stop_loss == pytest.approx(96)
    views = {v.speaker.split()[0]: v.target_exposure for v in risk_debate(prop, plan, row, 0.0, cfg)}
    assert views["Conservative"] == cfg["conservative_cap"] and views["Aggressive"] >= views["Neutral"] >= views["Conservative"]


def test_portfolio_manager_enforces_stop_loss_and_min_rebalance():
    cfg = make_config()
    plan = run_debate([_rep("a", 1, 0.6)], {}, 1, cfg["rating_thresholds"])
    w = Wallet(cash=1000, fee=0.001)
    w.rebalance(1.0, 100, "t0")
    ms = ManagerState(last_trade_step=0, stop_loss=95)
    row = {"close": 94, "atr": 1}
    prop = trader(plan, row, w.exposure(94), cfg)
    d = portfolio_manager(plan, prop, risk_debate(prop, plan, row, 0, cfg), row, w, 1000, 1, ms, cfg)
    assert d.action == "SELL" and d.target_exposure == 0 and "STOP-LOSS" in d.reasons[1]
    # tiny change is refused
    w2 = Wallet(cash=1000, fee=0.001)
    prop2 = trader(plan, {"close": 100, "atr": 1}, 0.0, cfg)
    prop2.target_exposure = 0.05
    d2 = portfolio_manager(plan, prop2, [], {"close": 100}, w2, 1000, 10, ManagerState(), cfg)
    assert d2.action == "HOLD" and not d2.approved


# ------------------------------------------------------------------ memory
def test_memory_settles_and_reweights_analysts():
    m = DecisionMemory(holding_bars=1, window=100)
    price = 100.0
    for step in range(30):
        m.record(step, f"t{step}", price, "Buy", {"good": 1.0, "bad": -1.0})
        price *= 1.01
        m.settle(step + 1, price, f"t{step + 1}")
    assert m.hit_rates()["good"]["hit_rate"] > 0.9
    assert m.weight_multiplier("good") > 1 > m.weight_multiplier("bad")
    assert m.lessons and m.rating_accuracy() == 1.0


# ------------------------------------------------------------------ whole graph
def test_graph_propagate_returns_a_full_state():
    ind = compute_indicators(_bars())
    g = TradingAgentsGraph({"analysts": ["trend", "momentum", "volatility", "volume"]}, "4h")
    state = None
    for ts, row in zip(ind.index[-50:], ind.iloc[-50:].to_dict("records")):
        state = g.propagate(row, ts)
    d = state.to_dict()
    assert d["plan"]["rating"] in RATINGS and d["decision"]["action"] in ("BUY", "SELL", "HOLD")
    assert len(d["reports"]) == 4 and d["risk_views"] and d["plan"]["debate"]
    assert g.summary()["equity"] > 0


def test_uptrend_gets_bought_and_downtrend_sold():
    up, _ = run_backtest(compute_indicators(_bars(drift=0.004, seed=1)).iloc[220:],
                         {"analysts": ["trend", "momentum", "volume"]})
    down, _ = run_backtest(compute_indicators(_bars(drift=-0.004, seed=1)).iloc[220:],
                           {"analysts": ["trend", "momentum", "volume"]})
    assert up.summary()["exposure"] > down.summary()["exposure"]
    assert up.summary()["pnl_pct"] > down.summary()["pnl_pct"]


def test_backtest_has_no_lookahead():
    """Changing future candles must not change earlier decisions."""
    bars = _bars(seed=3)
    alt = bars.copy()
    alt.iloc[-20:, alt.columns.get_loc("close")] *= 1.5
    _, a = run_backtest(compute_indicators(bars).iloc[220:])
    _, b = run_backtest(compute_indicators(alt).iloc[220:])
    cut = a.index[-21]
    pd.testing.assert_series_equal(a.loc[:cut, "action"], b.loc[:cut, "action"])


def test_feature_matrix_refactor_keeps_build_features_identical():
    bars = _bars(300, freq="1D")
    pd.testing.assert_frame_equal(build_features(bars), _build_features_reference(bars))
    assert feature_matrix(bars).index.equals(bars.index)


# ------------------------------------------------------------------ hard-limit regressions
def _flat_uptrend_row(price: float) -> dict:
    """A row on which the trend analyst is always bullish, so the agent always wants to buy."""
    return {"close": price, "atr": price * 0.01, "atr_pct": 0.01, "vol_ratio": 1.0,
            "ema20": price * 1.02, "ema50": price, "sma200": price * 0.9,
            "macd_hist": 1.0, "macd_hist_prev": 0.0}


def test_kill_switch_fires_once_and_resets_the_drawdown_after_the_pause():
    # Stops so wide they never trigger, so only the kill switch can protect the wallet.
    cfg = config_for("4h", analysts=["trend"], stop_atr_mult=100.0, take_profit_atr_mult=100.0)
    g = TradingAgentsGraph(cfg, "4h")
    prices = list(np.linspace(100, 70, 40)) + [70.0 - 0.01 * i for i in range(80)]  # crash, then drift lower
    ts = pd.date_range("2024-01-01", periods=len(prices), freq="4h", tz="UTC")
    kills = []
    for t, p in zip(ts, prices):
        s = g.propagate(_flat_uptrend_row(p), t)
        if any(r.startswith("KILL SWITCH") for r in s.decision.reasons):
            kills.append(t)
    # Before the fix the old peak was kept, so every re-entry after the pause was killed again.
    assert len(kills) == 1
    assert g.wallet.btc > 0  # it is allowed back into the market after the pause


def test_take_profit_is_not_bought_back_on_the_next_candle():
    cfg = config_for("4h", analysts=["trend"], take_profit_atr_mult=2.0, min_hold_bars=1)
    g = TradingAgentsGraph(cfg, "4h")
    prices = [100.0] * 3 + list(np.linspace(100, 110, 20))  # steady rally: take-profit hits
    ts = pd.date_range("2024-01-01", periods=len(prices), freq="4h", tz="UTC")
    for t, p in zip(ts, prices):
        g.propagate(_flat_uptrend_row(p), t)
    sides = [t["side"] for t in g.wallet.trades]
    first_tp = next(i for i, t in enumerate(g.wallet.trades) if "TAKE-PROFIT" in t["reason"])
    # The rating stays Buy the whole time, so nothing after the take-profit may be a BUY.
    assert "BUY" not in sides[first_tp + 1:]
