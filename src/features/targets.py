"""Direction targets for the multi-horizon iteration (decided 2026-11-21, docs/09 section 4).

Every target looks exactly one bar ahead (bar t's label uses data from bar t+1 only), so the
existing one-bar embargo between train / validation / test still purges every overlap.

- binary:          UP if the next bar's log return > 0, else DOWN.
- vol3:            3-class, flat band = +-k * sigma_t (sigma_t = rolling std of past bar returns).
- triple_barrier:  3-class on the *path* of the next bar, read from its 1h candles. UP if the
                   upper barrier close[t] * exp(k * sigma_t) is touched first, DOWN if the lower
                   one is, FLAT if neither is touched before the bar ends (vertical barrier = horizon).
                   If one hourly candle touches both, its close decides (the order inside an hour is unknown).

The multiplier k is never tuned for profit (Fu et al. 2024 show that overfits the label itself,
docs/09 #24). It is chosen on the inner *training* rows only, so that about a third of them are
FLAT, which keeps the classes balanced the way Jaquart (2021) did with training-set quantiles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.returns import DOWN, FLAT, UP, forward_log_return, log_returns

TARGET_NAMES = ("binary", "vol3", "triple_barrier")
TARGET_FLAT_SHARE = 1 / 3
K_GRID = tuple(np.round(np.arange(0.05, 3.01, 0.05), 2))
# Rolling window for sigma_t: about a month of daily bars, a week of 4h or 1h bars.
SIGMA_WINDOW = {24: 30, 4: 42, 1: 168}


class TargetBuilder:
    """Holds the raw ingredients (next-bar return, sigma, next-bar hourly path) for one bar size."""

    def __init__(self, bars: pd.DataFrame, hourly: pd.DataFrame, bar_hours: int):
        self.bar_hours = bar_hours
        close = bars["close"]
        self.close = close
        self.fwd = forward_log_return(close, 1)
        self.sigma = log_returns(close).rolling(SIGMA_WINDOW[bar_hours]).std()
        # Hourly highs/lows of the *next* bar, one column per hour inside it.
        offsets = [pd.Timedelta(hours=bar_hours + j) for j in range(bar_hours)]
        self.path_high = np.column_stack([hourly["high"].reindex(bars.index + o).to_numpy() for o in offsets])
        self.path_low = np.column_stack([hourly["low"].reindex(bars.index + o).to_numpy() for o in offsets])
        self.path_close = np.column_stack([hourly["close"].reindex(bars.index + o).to_numpy() for o in offsets])
        self.index = bars.index

    # --- labels -----------------------------------------------------------------
    def binary(self, index: pd.Index) -> pd.Series:
        fwd = self.fwd.reindex(index)
        return pd.Series(np.where(fwd > 0, UP, DOWN), index=index, dtype="float64").where(fwd.notna())

    def vol3(self, index: pd.Index, k: float) -> pd.Series:
        fwd, thr = self.fwd.reindex(index), k * self.sigma.reindex(index)
        lab = pd.Series(FLAT, index=index, dtype="float64")
        lab[fwd > thr] = UP
        lab[fwd < -thr] = DOWN
        return lab.where(fwd.notna() & thr.notna())

    def triple_barrier(self, index: pd.Index, k: float) -> pd.Series:
        pos = self.index.get_indexer(index)
        c0 = self.close.to_numpy()[pos]
        band = k * self.sigma.to_numpy()[pos]
        upper, lower = c0 * np.exp(band), c0 * np.exp(-band)
        hi, lo, cl = self.path_high[pos], self.path_low[pos], self.path_close[pos]
        hit_up, hit_dn = hi >= upper[:, None], lo <= lower[:, None]
        any_hit = hit_up | hit_dn
        first = np.where(any_hit.any(axis=1), any_hit.argmax(axis=1), -1)
        rows = np.arange(len(pos))
        col = np.clip(first, 0, None)
        up_first, dn_first = hit_up[rows, col], hit_dn[rows, col]
        both = up_first & dn_first
        lab = np.full(len(pos), float(FLAT))
        lab[(first >= 0) & up_first & ~both] = UP
        lab[(first >= 0) & dn_first & ~both] = DOWN
        lab[(first >= 0) & both] = np.where(cl[rows, col][(first >= 0) & both] > c0[(first >= 0) & both], UP, DOWN)
        valid = ~np.isnan(hi).any(axis=1) & ~np.isnan(lo).any(axis=1) & ~np.isnan(band)
        return pd.Series(np.where(valid, lab, np.nan), index=index)

    # --- k chosen on training rows only ------------------------------------------
    def choose_k(self, target: str, train_index: pd.Index, flat_share: float = TARGET_FLAT_SHARE) -> float | None:
        if target == "binary":
            return None
        label_fn = self.vol3 if target == "vol3" else self.triple_barrier
        best_k, best_gap = K_GRID[0], np.inf
        for k in K_GRID:
            share = (label_fn(train_index, k).dropna() == FLAT).mean()
            if abs(share - flat_share) < best_gap:
                best_k, best_gap = float(k), abs(share - flat_share)
        return best_k

    def labels(self, target: str, index: pd.Index, k: float | None) -> pd.Series:
        if target == "binary":
            return self.binary(index)
        if target == "vol3":
            return self.vol3(index, k)
        if target == "triple_barrier":
            return self.triple_barrier(index, k)
        raise ValueError(f"unknown target {target!r}; expected one of {TARGET_NAMES}")
