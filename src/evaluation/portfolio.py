"""Portfolio-level evaluation (CRISP-DM Phase 5, business criterion).

The business question in docs/01 §3 is whether a strategy earns a higher Sharpe ratio than
buy-and-hold, net of fees. This module answers it for any strategy expressed as a
*target BTC share of the wallet* per candle:

* `simulate_exposure` runs a target-exposure path through the same paper `Wallet` the agent
  uses, so the agent and every baseline pay exactly the same fees.
* `STRATEGIES` holds the simple baselines the agent must beat: cash, buy-and-hold, a static
  50/50 wallet, a 200-candle moving-average filter, an EMA 20/50 cross, volatility targeting
  and "the ML model alone".
* `performance` turns an equity curve into Sharpe, Sortino, drawdown, Calmar, exposure and
  turnover.
* `sharpe_difference_test` is a paired circular block bootstrap of Sharpe(a) - Sharpe(b),
  so "the agent beats X" comes with a confidence interval and a p-value.

Timing convention (shared with the agent): the target for candle t is computed from data up to
close[t] and executed at close[t]; it earns the return from close[t] to close[t+1].
`delay_bars=1` executes one candle later instead, as a robustness check.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src import config
from src.agents.execution import Wallet
from src.evaluation.baselines import periods_per_year

INITIAL_CASH = 1_000.0
BOOT_SAMPLES = 2_000
BOOT_BLOCK_HOURS = 168          # one week of data per bootstrap block keeps weekly dependence
REBALANCE_BAND = 0.15           # same minimum rebalance as the agent's default
VOL_TARGET = 0.50               # annualised volatility the vol-targeting baseline aims for
MODEL_LONG, MODEL_FLAT = 0.55, 0.45


# --------------------------------------------------------------------------- simulation
def simulate_exposure(close: pd.Series, target: pd.Series, fee: float = config.TRADING_FEE,
                      band: float = 0.0, delay_bars: int = 0, slippage_bps: float = 0.0,
                      initial_cash: float = INITIAL_CASH) -> pd.DataFrame:
    """Trade a target-exposure path with the agent's wallet.

    band          skip rebalances smaller than this share of the wallet (fee control)
    delay_bars    execute each target this many candles after it was decided
    slippage_bps  extra cost per trade on top of the fee, in basis points
    Returns one row per candle: price, exposure after trading, equity, cumulative fees.
    """
    target = target.reindex(close.index).shift(delay_bars).clip(0, 1)
    wallet = Wallet(cash=initial_cash, fee=fee + slippage_bps / 1e4)
    rows = []
    for ts, price, tgt in zip(close.index, close.to_numpy(), target.to_numpy()):
        if not np.isnan(tgt) and abs(tgt - wallet.exposure(price)) > max(band, 1e-9):
            wallet.rebalance(float(tgt), float(price), ts.isoformat())
        rows.append((ts, price, wallet.exposure(price), wallet.value(price), wallet.fees_paid, len(wallet.trades)))
    return pd.DataFrame(rows, columns=["timestamp", "price", "exposure", "equity", "fees", "trades"]).set_index("timestamp")


# --------------------------------------------------------------------------- baselines
def _sma200(ind: pd.DataFrame) -> pd.Series:
    return (ind["close"] > ind["sma200"]).astype(float).where(ind["sma200"].notna())


def _ema_cross(ind: pd.DataFrame) -> pd.Series:
    return (ind["ema20"] > ind["ema50"]).astype(float)


def _vol_target(ind: pd.DataFrame, interval: str) -> pd.Series:
    realised = ind["ret_1"].rolling(30).std() * np.sqrt(periods_per_year(interval))
    return (VOL_TARGET / realised).clip(0, 1)


def _model_only(ind: pd.DataFrame) -> pd.Series:
    """Long when the LightGBM is confident up, cash when confident down, otherwise keep."""
    if "p_up" not in ind or ind["p_up"].isna().all():
        return pd.Series(np.nan, index=ind.index)
    p = ind["p_up"]
    signal = pd.Series(np.where(p > MODEL_LONG, 1.0, np.where(p < MODEL_FLAT, 0.0, np.nan)), index=ind.index)
    return signal.ffill().fillna(0.0).where(p.notna())


# name -> (target builder, rebalance band). Each builder only reads columns known at close[t].
STRATEGIES: dict[str, tuple[Callable[[pd.DataFrame, str], pd.Series], float]] = {
    "cash": (lambda ind, iv: pd.Series(0.0, index=ind.index), 0.0),
    "buy_and_hold": (lambda ind, iv: pd.Series(1.0, index=ind.index), 0.0),
    "static_50": (lambda ind, iv: pd.Series(0.5, index=ind.index), REBALANCE_BAND),
    "sma200_filter": (lambda ind, iv: _sma200(ind), 0.0),
    "ema20_50_cross": (lambda ind, iv: _ema_cross(ind), 0.0),
    "vol_target": (_vol_target, REBALANCE_BAND),
    "model_only": (lambda ind, iv: _model_only(ind), 0.0),
}


def run_baseline(name: str, ind: pd.DataFrame, interval: str, **sim_kwargs) -> pd.DataFrame | None:
    """Simulate one baseline on an indicator table; None if it has no signal (e.g. no model)."""
    builder, band = STRATEGIES[name]
    target = builder(ind, interval)
    if target.isna().all():
        return None
    return simulate_exposure(ind["close"], target, band=band, **sim_kwargs)


# --------------------------------------------------------------------------- metrics
def bar_returns(equity: pd.Series, initial_cash: float = INITIAL_CASH) -> pd.Series:
    """Per-candle simple returns, the first one measured from the starting cash."""
    return pd.concat([pd.Series([initial_cash]), equity.reset_index(drop=True)]).pct_change().iloc[1:].set_axis(equity.index)


def sharpe(returns: np.ndarray, ppy: float) -> float:
    sd = returns.std(ddof=1) if len(returns) > 1 else 0.0
    return float(returns.mean() / sd * np.sqrt(ppy)) if sd > 0 else 0.0


def performance(sim: pd.DataFrame, interval: str, initial_cash: float = INITIAL_CASH) -> dict:
    """Headline metrics of one simulated equity curve."""
    ppy = periods_per_year(interval)
    eq = sim["equity"]
    r = bar_returns(eq, initial_cash).to_numpy()
    downside = np.sqrt(np.mean(np.minimum(r, 0) ** 2))
    years = len(eq) / ppy
    total = eq.iloc[-1] / initial_cash - 1
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else -1.0
    drawdown = (eq / eq.cummax().clip(lower=initial_cash) - 1).min()
    return {
        "total_return": float(total),
        "cagr": float(cagr),
        "volatility": float(r.std(ddof=1) * np.sqrt(ppy)) if len(r) > 1 else 0.0,
        "sharpe": sharpe(r, ppy),
        "sortino": float(r.mean() / downside * np.sqrt(ppy)) if downside > 0 else 0.0,
        "max_drawdown": float(min(drawdown, 0.0)),
        "calmar": float(cagr / abs(drawdown)) if drawdown < 0 else 0.0,
        "avg_exposure": float(sim["exposure"].mean()),
        "turnover_per_year": float(sim["exposure"].diff().abs().sum() / years) if years > 0 else 0.0,
        "trades": int(sim["trades"].iloc[-1]),
        "fees": float(sim["fees"].iloc[-1]),
    }


def sharpe_difference_test(returns_a: pd.Series, returns_b: pd.Series, interval: str,
                           n_boot: int = BOOT_SAMPLES, seed: int = 0) -> dict:
    """Paired circular block bootstrap of Sharpe(a) - Sharpe(b).

    Both strategies are resampled with the same blocks, so their common market exposure
    cancels out. p_value = share of resamples where a does not beat b (one-sided).
    """
    a, b = returns_a.align(returns_b, join="inner")
    a, b = a.to_numpy(), b.to_numpy()
    n, ppy = len(a), periods_per_year(interval)
    block = max(2, int(round(BOOT_BLOCK_HOURS / (pd.Timedelta(interval) / pd.Timedelta("1h")))))
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    offsets = np.arange(block)
    for i in range(n_boot):
        idx = ((rng.integers(0, n, n_blocks)[:, None] + offsets) % n).ravel()[:n]
        diffs[i] = sharpe(a[idx], ppy) - sharpe(b[idx], ppy)
    return {
        "sharpe_diff": sharpe(a, ppy) - sharpe(b, ppy),
        "ci_low": float(np.quantile(diffs, 0.025)),
        "ci_high": float(np.quantile(diffs, 0.975)),
        "p_value": float(np.mean(diffs <= 0)),
    }
