"""Turn predicted direction labels into a simple long/flat/short strategy and backtest it
(the business-criterion evaluation that EDA item 12 set the buy-and-hold bar for).

Simplifying assumption: the fee cost of a position change is approximated as
`log(1 - fee) * turnover`, i.e. one full round-trip fee per unit of position
changed. This is a linear approximation, not an exact trade-by-trade ledger,
but is good enough to compare strategies at this stage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.evaluation.baselines import max_drawdown, sharpe_ratio
from src.features.returns import DOWN, FLAT, UP

POSITION = {UP: 1.0, FLAT: 0.0, DOWN: -1.0}


def backtest(pred_label: pd.Series, close: pd.Series, rule: str = "1D",
             fee: float = config.TRADING_FEE) -> dict:
    """Hold position POSITION[pred_label[t]] from close[t] to close[t+1].

    `pred_label` must be predictions made using only data available at t (the
    caller's responsibility — this function has no visibility into how they
    were produced). It only applies the resulting position to the realized
    next-period return and charges a fee whenever the position changes.
    """
    position = pred_label.map(POSITION)
    next_log_return = np.log(close.shift(-1) / close).reindex(position.index)
    gross = position * next_log_return

    turnover = position.diff().abs()
    turnover.iloc[0] = abs(position.iloc[0])  # opening the initial position also costs a trade
    fee_drag = np.log(1 - fee) * turnover

    net = (gross + fee_drag).dropna()
    return {
        "total_return": float(np.expm1(net.sum())),
        "sharpe": sharpe_ratio(net, rule),
        "max_drawdown": max_drawdown(net),
    }
