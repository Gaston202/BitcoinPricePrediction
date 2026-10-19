import numpy as np
import pandas as pd
import pytest

from src.evaluation.criteria import (business_criterion_met, data_mining_criterion_met,
                                     hard_prediction_scores, macro_roc_auc)

CLASSES = [-1.0, 0.0, 1.0]


def test_macro_roc_auc_is_half_for_a_constant_prediction():
    y_true = pd.Series([1.0, 0.0, -1.0, 1.0, 0.0, -1.0])
    constant_scores = hard_prediction_scores(pd.Series([0.0] * len(y_true)), CLASSES)
    assert macro_roc_auc(y_true, constant_scores, CLASSES) == pytest.approx(0.5)


def test_macro_roc_auc_is_perfect_for_exact_probabilistic_predictions():
    y_true = pd.Series([1.0, 0.0, -1.0, 1.0])
    proba = hard_prediction_scores(y_true, CLASSES)  # the "model" predicts the truth exactly
    assert macro_roc_auc(y_true, proba, CLASSES) == pytest.approx(1.0)


def test_hard_prediction_scores_is_one_hot():
    pred = pd.Series([1.0, -1.0])
    scores = hard_prediction_scores(pred, CLASSES)
    assert scores.tolist() == [[0, 0, 1], [1, 0, 0]]


def test_data_mining_criterion_requires_beating_both_baselines_on_both_metrics():
    # Beats both on F1 but not on AUC -> criterion not met.
    assert not data_mining_criterion_met(
        model_f1=0.40, model_auc=0.50,
        majority_f1=0.20, majority_auc=0.50,
        persistence_f1=0.35, persistence_auc=0.55,
    )
    # Beats both baselines on both metrics -> criterion met.
    assert data_mining_criterion_met(
        model_f1=0.40, model_auc=0.60,
        majority_f1=0.20, majority_auc=0.50,
        persistence_f1=0.35, persistence_auc=0.55,
    )


def test_business_criterion_is_a_strict_sharpe_comparison():
    assert business_criterion_met(strategy_sharpe=0.1, buy_and_hold_sharpe=-0.3)
    assert not business_criterion_met(strategy_sharpe=-0.5, buy_and_hold_sharpe=-0.3)
    assert not business_criterion_met(strategy_sharpe=0.5, buy_and_hold_sharpe=0.5)  # tie doesn't count
