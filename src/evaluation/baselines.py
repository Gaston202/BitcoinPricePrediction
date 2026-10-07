"""Baselines the ML model and the trading agent must beat (CRISP-DM success criteria)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src import config

HOURS_PER_YEAR = 24 * 365  # crypto trades 24/7


def macro_f1(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def periods_per_year(rule: str) -> float:
    return HOURS_PER_YEAR / (pd.Timedelta(rule) / pd.Timedelta("1h"))


def majority_baseline(labels: pd.Series) -> dict:
    """Always predict the most frequent class."""
    y = labels.dropna()
    majority = y.mode().iloc[0]
    pred = pd.Series(majority, index=y.index)
    return {"class": int(majority), "accuracy": float((pred == y).mean()),
            "macro_f1": macro_f1(y, pred)}


def persistence_baseline(labels: pd.Series, horizon: int = 1) -> dict:
    """Predict that the next move repeats the last *fully realized* move.

    The label at t-horizon is the first one whose outcome is known at time t.
    """
    pred = labels.shift(horizon)
    mask = labels.notna() & pred.notna()
    y, pred = labels[mask], pred[mask]
    return {"accuracy": float((pred == y).mean()), "macro_f1": macro_f1(y, pred)}


def sharpe_ratio(log_rets: pd.Series, rule: str) -> float:
    """Annualized Sharpe of simple returns, risk-free rate assumed 0."""
    simple = np.expm1(log_rets.dropna())
    return float(simple.mean() / simple.std() * np.sqrt(periods_per_year(rule)))


def max_drawdown(log_rets: pd.Series) -> float:
    equity = np.exp(log_rets.dropna().cumsum())
    return float((equity / equity.cummax() - 1).min())


def buy_and_hold(close: pd.Series, rule: str) -> dict:
    """Buy at the first close, hold to the last, paying one entry and one exit fee."""
    rets = np.log(close / close.shift(1))
    total = close.dropna().iloc[-1] / close.dropna().iloc[0] * (1 - config.TRADING_FEE) ** 2 - 1
    return {"total_return": float(total), "sharpe": sharpe_ratio(rets, rule),
            "max_drawdown": max_drawdown(rets)}
