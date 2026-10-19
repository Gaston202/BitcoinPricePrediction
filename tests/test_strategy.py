import numpy as np
import pandas as pd
import pytest

from src.evaluation.strategy import backtest
from src.features.returns import DOWN, FLAT, UP


def test_backtest_always_up_matches_buy_and_hold_before_fees():
    close = pd.Series([100.0, 110.0, 99.0, 120.0])
    labels = pd.Series([UP, UP, UP, UP])
    result = backtest(labels, close, fee=0.0)
    # With fee=0 and always-long, strategy return equals buy-and-hold return.
    expected_total_return = close.iloc[-1] / close.iloc[0] - 1
    assert result["total_return"] == pytest.approx(expected_total_return, rel=1e-6)


def test_backtest_flat_position_has_zero_return_and_one_entry_fee():
    close = pd.Series([100.0, 110.0, 99.0])
    labels = pd.Series([FLAT, FLAT, FLAT])
    result = backtest(labels, close, fee=0.001)
    # Flat position never changes after the (costless, since position is 0) opening trade.
    assert result["total_return"] == pytest.approx(0.0, abs=1e-9)


def test_backtest_charges_fee_on_each_position_change():
    close = pd.Series([100.0, 100.0, 100.0])  # flat prices isolate the fee drag
    labels = pd.Series([UP, DOWN, UP])  # position flips every step
    result = backtest(labels, close, fee=0.01)
    # Row 2 has no forward return (last close) and is dropped, so only rows 0-1 count:
    # open to +1 (turnover=1), +1 to -1 (turnover=2).
    turnover = [1.0, 2.0]
    expected_log_fee = sum(np.log(1 - 0.01) * t for t in turnover)
    assert result["total_return"] == pytest.approx(np.expm1(expected_log_fee), rel=1e-6)


def test_backtest_short_position_profits_from_a_falling_price():
    close = pd.Series([100.0, 90.0])
    labels = pd.Series([DOWN, DOWN])
    result = backtest(labels, close, fee=0.0)
    assert result["total_return"] > 0
