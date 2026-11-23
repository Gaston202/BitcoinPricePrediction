import numpy as np
import pandas as pd
import pytest

from src.agent import market
from src.agent.trader import AgentConfig, TradingAgent
from src.features.engineering import build_features, feature_matrix
from src.features.returns import DOWN, FLAT, UP


def _pred(cls, p=0.6, exp=0.02):
    probs = {UP: (0.2, 0.2, p), DOWN: (p, 0.2, 0.2), FLAT: (0.2, p, 0.2)}[cls]
    return {"p_down": probs[0], "p_flat": probs[1], "p_up": probs[2],
            "pred_class": cls, "expected_log_return": exp if cls == UP else -exp if cls == DOWN else 0.0}


def test_agent_buys_on_up_and_sells_on_down_with_fees():
    agent = TradingAgent(AgentConfig(fee=0.001, stop_loss=None, take_profit=None))
    assert agent.step("t0", 100.0, _pred(UP))["action"] == "BUY"
    assert agent.step("t1", 105.0, _pred(UP))["action"] == "HOLD"
    rec = agent.step("t2", 110.0, _pred(DOWN))
    assert rec["action"] == "SELL"
    assert rec["cash"] == pytest.approx(1000 * 0.999 / 100 * 110 * 0.999)
    s = agent.summary()
    assert s["n_trades"] == 2 and s["n_round_trips"] == 1 and s["win_rate"] == 100
    assert s["fees_paid"] == pytest.approx(1.0 + 9.99 * 110 * 0.001)


def test_fee_aware_agent_skips_trades_that_cannot_cover_fees():
    agent = TradingAgent(AgentConfig(fee=0.001, fee_aware=True))
    rec = agent.step("t0", 100.0, _pred(UP, exp=0.001))  # +0.1% expected < 0.2% round trip
    assert rec["action"] == "HOLD" and "fees" in rec["reason"]


def test_stop_loss_and_take_profit_fire_regardless_of_signal():
    agent = TradingAgent(AgentConfig(stop_loss=0.05, take_profit=0.10))
    agent.step("t0", 100.0, _pred(UP))
    assert agent.step("t1", 94.0, _pred(UP))["action"] == "SELL"
    agent.step("t2", 100.0, _pred(UP))
    rec = agent.step("t3", 111.0, _pred(UP))
    assert rec["action"] == "SELL" and "Take-profit" in rec["reason"]


def test_min_confidence_blocks_weak_signals():
    agent = TradingAgent(AgentConfig(min_confidence=0.7))
    assert agent.step("t0", 100.0, _pred(UP, p=0.6))["action"] == "HOLD"
    assert agent.step("t1", 100.0, _pred(UP, p=0.8))["action"] == "BUY"


def test_buy_and_hold_benchmark_and_drawdown():
    agent = TradingAgent(AgentConfig(fee=0.0, stop_loss=None, take_profit=None))
    agent.step("t0", 100.0, _pred(UP))
    agent.step("t1", 50.0, _pred(FLAT))
    rec = agent.step("t2", 100.0, _pred(FLAT))
    assert rec["bh_equity"] == pytest.approx(1000.0)
    assert agent.summary()["max_drawdown_pct"] == pytest.approx(-50.0)


def test_invalid_config_is_rejected():
    with pytest.raises(ValueError):
        AgentConfig(stop_loss=1.5).validate()
    with pytest.raises(ValueError):
        AgentConfig(initial_cash=0).validate()


def _candles(n=60, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    idx = pd.date_range("2025-01-01", periods=n, freq="1D", tz="UTC")
    vol = rng.uniform(10, 20, n)
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
                         "volume": vol, "taker_buy_base_volume": vol / 2}, index=idx)


def test_feature_matrix_matches_build_features_and_keeps_unlabeled_last_row():
    df = _candles()
    fm = feature_matrix(df)
    bf = build_features(df)
    pd.testing.assert_frame_equal(fm.loc[bf.index], bf.drop(columns="label"))
    assert not fm.iloc[-1].isna().any()  # the newest candle is usable live


def test_provisional_frame_extrapolates_volume_of_the_open_candle():
    df = _candles(3)
    df["quote_asset_volume"] = df["volume"]
    df["taker_buy_quote_volume"] = df["taker_buy_base_volume"]
    df["close_time"] = df.index + pd.Timedelta("1D")
    now = df.index[-1] + pd.Timedelta("6h")  # a quarter into the last candle
    out = market.provisional_frame(df, now=now)
    assert out["volume"].iloc[-1] == pytest.approx(df["volume"].iloc[-1] * 4)
    assert out["volume"].iloc[0] == df["volume"].iloc[0]
