import numpy as np
import pandas as pd
import pytest

from src.features.returns import DOWN, FLAT, UP
from src.modeling.train import FEATURE_COLUMNS, evaluate, make_models, split_X_y, tune_class_weights, weighted_predict


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


class _StubProbaModel:
    """Fixed per-row class probabilities, for deterministic weighted_predict/tune_class_weights tests."""

    classes_ = [DOWN, FLAT, UP]

    def __init__(self, proba):
        self._proba = np.asarray(proba)

    def predict_proba(self, X):
        return self._proba

    def predict(self, X):
        return np.array(self.classes_)[self._proba.argmax(axis=1)]


def test_weighted_predict_can_override_the_default_argmax():
    # Every row: P(down)=0.3, P(flat)=0.4, P(up)=0.3 -> default argmax always picks flat.
    model = _StubProbaModel([[0.3, 0.4, 0.3]] * 3)
    X = pd.DataFrame(index=range(3))
    assert list(model.predict(X)) == [FLAT, FLAT, FLAT]
    # Upweighting up enough flips the decision without retraining anything.
    pred = weighted_predict(model, X, weights={DOWN: 1.0, FLAT: 1.0, UP: 2.0})
    assert list(pred) == [UP, UP, UP]


def test_tune_class_weights_fixes_a_flat_biased_model():
    rng = np.random.default_rng(0)
    n = 90
    true_labels = rng.choice([DOWN, FLAT, UP], size=n, p=[0.3, 0.4, 0.3])
    # The model is biased toward flat for every row, exactly like the real random forest
    # (docs/05_modeling.md: 89% predicted flat vs. an actual 45% flat rate) -- but the true
    # label still nudges the probabilities, so there is real signal a better threshold can use.
    proba = np.array([
        [0.35, 0.45, 0.20] if t == DOWN else [0.20, 0.45, 0.35] if t == UP else [0.25, 0.50, 0.25]
        for t in true_labels
    ])
    model = _StubProbaModel(proba)
    X = pd.DataFrame(index=range(n))
    y = pd.Series(true_labels)

    baseline_f1 = evaluate(model, X, y)["macro_f1"]
    assert set(model.predict(X)) == {FLAT}  # default argmax collapses to flat every time

    best = tune_class_weights(model, X, y, grid=(0.5, 1.0, 1.5, 2.0, 3.0))
    assert best["macro_f1"] > baseline_f1

    pred = weighted_predict(model, X, best["weights"])
    assert set(pred) == {DOWN, FLAT, UP}  # the tuned rule actually uses all three classes
