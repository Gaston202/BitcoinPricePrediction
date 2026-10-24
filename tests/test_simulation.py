import pandas as pd
import pytest

from src.simulation import PortfolioSimulator


def test_buy_hold_sell_tracks_cash_btc_and_fees():
    data = pd.DataFrame({"close": [100.0, 110.0, 120.0]}, index=pd.date_range("2025-01-01", periods=3))
    simulator = PortfolioSimulator(fee=0.001)

    history = simulator.run(data, pd.Series([1, "HOLD", -1], index=data.index))

    assert history.iloc[0]["cash"] == pytest.approx(0.0)
    assert history.iloc[0]["btc_holding"] == pytest.approx(9.99)
    assert history.iloc[-1]["cash"] == pytest.approx(1197.6012)
    assert history.iloc[-1]["btc_holding"] == pytest.approx(0.0)


def test_zero_is_sell_and_hold_is_distinct():
    data = pd.DataFrame({"close": [100.0, 110.0, 90.0]}, index=pd.date_range("2025-01-01", periods=3))
    simulator = PortfolioSimulator(fee=0.0)

    history = simulator.run(data, pd.Series([1, "HOLD", 0], index=data.index))

    assert history.iloc[1]["btc_holding"] == pytest.approx(10.0)
    assert history.iloc[2]["cash"] == pytest.approx(900.0)


def test_metrics_include_fee_adjusted_buy_and_hold():
    data = pd.DataFrame({"close": [100.0, 200.0]}, index=pd.date_range("2025-01-01", periods=2))
    simulator = PortfolioSimulator(fee=0.001)
    simulator.run(data, pd.Series(["HOLD", "HOLD"], index=data.index))

    metrics = simulator.performance_metrics()

    assert metrics["strategy_total_return_pct"] == pytest.approx(0.0)
    assert metrics["benchmark_total_return_pct"] == pytest.approx(99.8)
    assert metrics["strategy_max_drawdown_pct"] == pytest.approx(0.0)


def test_invalid_signal_and_price_are_rejected():
    simulator = PortfolioSimulator()
    with pytest.raises(ValueError, match="signal"):
        simulator.step("t", 100.0, 2)
    with pytest.raises(ValueError, match="price"):
        simulator.step("t", 0.0, "HOLD")
