import numpy as np
import pandas as pd
import pytest

from src.evaluation.baselines import (majority_baseline, max_drawdown,
                                      periods_per_year, persistence_baseline)


def test_majority_baseline_picks_most_common_class():
    labels = pd.Series([1, 1, 1, -1, 0])
    result = majority_baseline(labels)
    assert result["class"] == 1
    assert result["accuracy"] == pytest.approx(0.6)


def test_persistence_uses_previous_label():
    labels = pd.Series([1.0, 1.0, -1.0, -1.0])
    result = persistence_baseline(labels)
    # predictions: [1, 1, -1] vs truth [1, -1, -1] -> 2/3 correct
    assert result["accuracy"] == pytest.approx(2 / 3)


def test_max_drawdown_of_up_then_halving():
    log_rets = pd.Series(np.log([2.0, 0.5]))  # 1 -> 2 -> 1
    assert max_drawdown(log_rets) == pytest.approx(-0.5)


def test_periods_per_year():
    assert periods_per_year("1h") == 8760
    assert periods_per_year("1D") == 365
