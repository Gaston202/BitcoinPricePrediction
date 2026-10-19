import numpy as np
import pandas as pd
import pytest

from src.features.returns import DOWN, FLAT, UP
from src.modeling.train import FEATURE_COLUMNS, evaluate, make_models, split_X_y


def _toy_frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    data = {col: rng.normal(size=n) for col in FEATURE_COLUMNS}
    # Make the label partly learnable from ret_lag1 so a classifier has something to fit,
    # not purely random noise that would make macro-F1 meaningless to assert on.
    data["label"] = np.where(data["ret_lag1"] > 0.3, UP, np.where(data["ret_lag1"] < -0.3, DOWN, FLAT))
    return pd.DataFrame(data)


def test_split_X_y_separates_label_and_keeps_column_order():
    df = _toy_frame(n=10)
    X, y = split_X_y(df)
    assert list(X.columns) == FEATURE_COLUMNS
    assert "label" not in X.columns
    assert y.tolist() == df["label"].tolist()


def test_make_models_returns_expected_candidates():
    models = make_models()
    assert set(models) == {"logistic_regression", "random_forest", "gradient_boosting"}


@pytest.mark.parametrize("name", ["logistic_regression", "random_forest", "gradient_boosting"])
def test_each_candidate_fits_and_predicts_valid_classes(name):
    train, val = _toy_frame(seed=1), _toy_frame(seed=2)
    X_train, y_train = split_X_y(train)
    X_val, y_val = split_X_y(val)

    model = make_models()[name]
    model.fit(X_train, y_train)
    pred = model.predict(X_val)

    assert set(pred) <= {DOWN, FLAT, UP}
    assert len(pred) == len(X_val)


def test_evaluate_matches_manual_macro_f1():
    from sklearn.metrics import f1_score

    class StubModel:
        def predict(self, X):
            return np.array([UP] * len(X))

    y = pd.Series([UP, UP, DOWN, FLAT])
    result = evaluate(StubModel(), pd.DataFrame(index=y.index), y)
    expected_f1 = f1_score(y, [UP] * len(y), average="macro")
    assert result["macro_f1"] == pytest.approx(expected_f1)
    assert result["accuracy"] == pytest.approx(0.5)  # 2 of 4 correctly predicted UP
