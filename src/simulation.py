"""Trade-by-trade portfolio simulation using an already-loaded local price frame."""

from __future__ import annotations

from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from src import config


class PortfolioSimulator:
    """Simulate an all-in/all-out long-only BTC portfolio.

    Signals are interpreted as ``1`` = buy, ``-1`` or ``0`` = sell, and
    ``"HOLD"`` (case-insensitive) = keep the current position. A numeric
    ``0`` is therefore a sell signal to match the requested interface; use
    ``"HOLD"`` when the model wants to stay invested or in cash.
    """

    def __init__(self, initial_cash: float = 1_000.0, fee: float = config.TRADING_FEE) -> None:
        if not np.isfinite(initial_cash) or initial_cash <= 0:
            raise ValueError("initial_cash must be a finite positive number")
        if not np.isfinite(fee) or not 0 <= fee < 1:
            raise ValueError("fee must be finite and in the range [0, 1)")
        self.initial_cash = float(initial_cash)
        self.fee = float(fee)
        self.cash = self.initial_cash
        self.btc_holding = 0.0
        self._history: list[dict[str, Any]] = []

    @staticmethod
    def _validate_price(price: Real) -> float:
        value = float(price)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("price must be a finite positive number")
        return value

    @staticmethod
    def _signal_kind(signal: Any) -> str:
        if isinstance(signal, str):
            normalized = signal.strip().upper()
            if normalized in {"HOLD", "H"}:
                return "hold"
            if normalized in {"BUY", "B"}:
                return "buy"
            if normalized in {"SELL", "S"}:
                return "sell"
        elif isinstance(signal, Real) and not isinstance(signal, bool):
            if float(signal) == 1:
                return "buy"
            if float(signal) in {-1, 0}:
                return "sell"
        raise ValueError("signal must be 1, -1, 0, BUY, SELL, or HOLD")

    def step(self, timestamp: Any, price: Real, signal: Any) -> dict[str, Any]:
        """Apply one signal at ``price`` and return the resulting ledger row."""
        price_value = self._validate_price(price)
        kind = self._signal_kind(signal)

        if kind == "buy" and self.cash > 0:
            self.btc_holding += self.cash * (1 - self.fee) / price_value
            self.cash = 0.0
        elif kind == "sell" and self.btc_holding > 0:
            self.cash += self.btc_holding * price_value * (1 - self.fee)
            self.btc_holding = 0.0

        total_value = self.cash + self.btc_holding * price_value
        row = {
            "timestamp": timestamp,
            "price": price_value,
            "signal": signal,
            "cash": self.cash,
            "btc_holding": self.btc_holding,
            "total_portfolio_value": total_value,
        }
        self._history.append(row)
        return row

    def run(
        self,
        data: pd.DataFrame,
        signals: pd.Series | str,
        *,
        timestamp_column: str | None = None,
        price_column: str = "close",
    ) -> pd.DataFrame:
        """Run the simulation over a time-ordered local DataFrame.

        ``signals`` may be a Series aligned to ``data`` or the name of a
        signal column in ``data``. No data is fetched or inferred externally.
        """
        if price_column not in data.columns:
            raise ValueError(f"data must contain a {price_column!r} column")
        signal_values = data[signals] if isinstance(signals, str) else signals
        if len(signal_values) != len(data):
            raise ValueError("signals and data must have the same length")
        if not signal_values.index.equals(data.index):
            signal_values = signal_values.reindex(data.index)
        if signal_values.isna().any():
            raise ValueError("signals must not contain missing values")

        timestamps = data[timestamp_column] if timestamp_column else data.index
        for timestamp, price, signal in zip(timestamps, data[price_column], signal_values):
            self.step(timestamp, price, signal)
        return self.history

    @property
    def history(self) -> pd.DataFrame:
        """Return the ledger with a timestamp index and stable column order."""
        if not self._history:
            return pd.DataFrame(
                columns=["timestamp", "price", "signal", "cash", "btc_holding", "total_portfolio_value"]
            ).set_index("timestamp")
        return pd.DataFrame(self._history).set_index("timestamp")

    def benchmark(self, prices: pd.Series) -> pd.DataFrame:
        """Return the fee-adjusted $1,000 buy-and-hold equity curve."""
        clean_prices = prices.astype(float)
        if clean_prices.empty:
            raise ValueError("prices must contain at least one value")
        if clean_prices.isna().any() or (clean_prices <= 0).any():
            raise ValueError("prices must contain only finite positive values")
        btc = self.initial_cash * (1 - self.fee) / clean_prices.iloc[0]
        equity = btc * clean_prices
        return pd.DataFrame({"price": clean_prices, "total_portfolio_value": equity})

    @staticmethod
    def _drawdown(equity: pd.Series) -> float:
        return float((equity / equity.cummax() - 1).min())

    def performance_metrics(self, prices: pd.Series | None = None) -> dict[str, float]:
        """Compare strategy and buy-and-hold returns and maximum drawdowns."""
        history = self.history
        if history.empty:
            raise ValueError("run the simulator before calculating performance metrics")
        benchmark_prices = history["price"] if prices is None else prices
        benchmark = self.benchmark(benchmark_prices)
        strategy_equity = history["total_portfolio_value"]
        benchmark_equity = benchmark["total_portfolio_value"]
        return {
            "strategy_total_return_pct": float((strategy_equity.iloc[-1] / self.initial_cash - 1) * 100),
            "benchmark_total_return_pct": float((benchmark_equity.iloc[-1] / self.initial_cash - 1) * 100),
            "strategy_max_drawdown_pct": self._drawdown(strategy_equity) * 100,
            "benchmark_max_drawdown_pct": self._drawdown(benchmark_equity) * 100,
            "strategy_minus_benchmark_pct": float(
                (strategy_equity.iloc[-1] - benchmark_equity.iloc[-1]) / self.initial_cash * 100
            ),
        }
