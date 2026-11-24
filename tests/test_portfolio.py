import numpy as np
import pandas as pd
import pytest

from src.agents.graph import run_backtest
from src.agents.market_data import compute_indicators
from src.evaluation.agent_backtest import agent_config, trade_targets
from src.evaluation.portfolio import (
    INITIAL_CASH, STRATEGIES, bar_returns, performance, run_baseline, sharpe, sharpe_difference_test,
    simulate_exposure,
)


def _bars(n=500, drift=0.0, seed=0, freq="4h"):
    rng = np.random.default_rng(seed)
    close = 30_000 * np.exp(np.cumsum(rng.normal(drift, 0.01, n)))
    idx = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    vol = rng.uniform(100, 200, n)
    return pd.DataFrame({"open": close, "high": close * 1.006, "low": close * 0.994, "close": close,
                         "volume": vol, "taker_buy_base_volume": vol * 0.5}, index=idx)


def test_buy_and_hold_pays_one_fee_and_tracks_the_price():
    close = pd.Series([100.0, 110.0, 121.0], index=pd.date_range("2024", periods=3, freq="1D", tz="UTC"))
    sim = simulate_exposure(close, pd.Series(1.0, index=close.index), fee=0.001)
    assert sim["trades"].iloc[-1] == 1
    assert sim["equity"].iloc[-1] == pytest.approx(INITIAL_CASH * 0.999 * 1.21)


def test_nan_target_means_no_trade_and_delay_shifts_execution():
    idx = pd.date_range("2024", periods=4, freq="1D", tz="UTC")
    close = pd.Series([100.0, 200.0, 200.0, 200.0], index=idx)
    target = pd.Series([1.0, np.nan, np.nan, np.nan], index=idx)
    now = simulate_exposure(close, target, fee=0.0)
    late = simulate_exposure(close, target, fee=0.0, delay_bars=1)
    assert now["equity"].iloc[-1] == pytest.approx(2 * INITIAL_CASH)   # bought at 100
    assert late["equity"].iloc[-1] == pytest.approx(INITIAL_CASH)      # bought at 200, one day late


def test_slippage_is_charged_like_an_extra_fee():
    close = pd.Series([100.0] * 3, index=pd.date_range("2024", periods=3, freq="1D", tz="UTC"))
    sim = simulate_exposure(close, pd.Series([1.0, 0.0, np.nan], index=close.index), fee=0.0, slippage_bps=10)
    assert sim["fees"].iloc[-1] == pytest.approx(1.0 + 0.999 * 1.0, rel=1e-3)


def test_performance_of_cash_is_all_zero():
    close = pd.Series(np.linspace(100, 50, 50), index=pd.date_range("2024", periods=50, freq="1D", tz="UTC"))
    m = performance(simulate_exposure(close, pd.Series(0.0, index=close.index)), "1D")
    assert m["total_return"] == 0 and m["sharpe"] == 0 and m["max_drawdown"] == 0 and m["trades"] == 0


def test_drawdown_and_sharpe_of_a_known_curve():
    idx = pd.date_range("2024", periods=4, freq="1D", tz="UTC")
    sim = pd.DataFrame({"equity": [1100.0, 880.0, 990.0, 1210.0], "exposure": 1.0, "fees": 0.0, "trades": 1}, index=idx)
    m = performance(sim, "1D")
    assert m["max_drawdown"] == pytest.approx(880 / 1100 - 1)
    r = bar_returns(sim["equity"]).to_numpy()
    assert r[0] == pytest.approx(0.1)
    assert m["sharpe"] == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(365))


def test_every_baseline_only_uses_past_data():
    """Changing the last candles must not change any baseline's earlier positions."""
    ind = compute_indicators(_bars(seed=2)).iloc[220:]
    alt = ind.copy()
    alt.iloc[-10:, alt.columns.get_loc("close")] *= 2
    for name in STRATEGIES:
        a, b = run_baseline(name, ind, "4h"), run_baseline(name, alt, "4h")
        if a is None:
            continue
        pd.testing.assert_series_equal(a["exposure"].iloc[:-11], b["exposure"].iloc[:-11], check_names=False)


def test_model_only_baseline_skips_without_probabilities():
    ind = compute_indicators(_bars()).iloc[220:]
    assert run_baseline("model_only", ind, "4h") is None
    ind["p_up"] = np.where(np.arange(len(ind)) % 2, 0.6, 0.4)
    sim = run_baseline("model_only", ind, "4h")
    # Alternating confident up/down calls: long, cash, long, ... one trade per candle.
    assert sim["trades"].iloc[-1] == len(ind) - 1
    assert sim["exposure"].max() <= 1.0


def test_replaying_the_agents_trades_reproduces_its_equity():
    """The baselines and the agent share one wallet, so the accounting is identical."""
    ind = compute_indicators(_bars(seed=5)).iloc[220:]
    graph, steps = run_backtest(ind, agent_config("4h", "core"), "4h")
    replay = simulate_exposure(ind["close"], trade_targets(graph, ind.index))
    np.testing.assert_allclose(replay["equity"].to_numpy(), steps["equity"].to_numpy(), rtol=1e-12)


def test_sharpe_difference_test_finds_a_clear_winner():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024", periods=2000, freq="4h", tz="UTC")
    market = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    better = market + 0.002
    res = sharpe_difference_test(better, market, "4h", n_boot=300)
    assert res["sharpe_diff"] > 0 and res["ci_low"] > 0 and res["p_value"] < 0.01
    same = sharpe_difference_test(market, market, "4h", n_boot=100)
    assert same["sharpe_diff"] == 0


def test_sharpe_of_constant_returns_is_zero():
    assert sharpe(np.zeros(10), 365) == 0.0


def test_agent_config_adds_the_quant_only_on_1h_and_respects_explicit_teams():
    assert "quant" in agent_config("1h", "core")["analysts"]
    assert "quant" not in agent_config("4h", "core")["analysts"]
    assert agent_config("1h", "core", analysts=["trend"])["analysts"] == ["trend"]
