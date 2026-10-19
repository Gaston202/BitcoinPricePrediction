"""Formal checks against the success criteria set in docs/01_business_understanding.md
(CRISP-DM Phase 5 — Evaluation).

These wrap the two criteria exactly as written, so Evaluation is a yes/no check
against a decision made before any data was seen, not a post-hoc judgment call.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def macro_roc_auc(y_true: pd.Series, scores: np.ndarray, classes: list[float]) -> float:
    """One-vs-rest macro-average ROC-AUC for a 3-class problem.

    `scores` may be predicted probabilities (a real classifier) or one-hot hard
    predictions (a baseline), both scored the same way so they are comparable.
    """
    y_bin = pd.get_dummies(y_true).reindex(columns=classes, fill_value=0).to_numpy()
    return float(roc_auc_score(y_bin, scores, average="macro", multi_class="ovr"))


def hard_prediction_scores(pred: pd.Series, classes: list[float]) -> np.ndarray:
    """One-hot-encode a hard (non-probabilistic) prediction so macro_roc_auc can score it."""
    return pd.get_dummies(pred).reindex(columns=classes, fill_value=0).to_numpy()


def data_mining_criterion_met(model_f1: float, model_auc: float,
                               majority_f1: float, majority_auc: float,
                               persistence_f1: float, persistence_auc: float) -> bool:
    """docs/01 Sec.4: beat BOTH baselines on BOTH macro-F1 and ROC-AUC."""
    return (model_f1 > majority_f1 and model_f1 > persistence_f1
            and model_auc > majority_auc and model_auc > persistence_auc)


def business_criterion_met(strategy_sharpe: float, buy_and_hold_sharpe: float) -> bool:
    """docs/01 Sec.2: beat buy-and-hold Sharpe over the same test window, net of fees."""
    return strategy_sharpe > buy_and_hold_sharpe
