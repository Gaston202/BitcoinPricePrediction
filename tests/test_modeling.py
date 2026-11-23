import numpy as np
import pandas as pd
import pytest

from src.features.returns import DOWN, FLAT, UP
from src.modeling.train import (
    BASE_MODEL_NAMES,
    ENSEMBLE_NAMES,
    FEATURE_COLUMNS,
    LabelEncodedClassifier,
    build_model,
    evaluate,
    load_model_choice,
    model_from_choice,
    save_model_choice,
    make_models,
    split_X_y,
    tune_class_weights,
    weighted_predict,
)


def _toy_frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    data = {col: rng.normal(size=n) for col in FEATURE_COLUMNS}
    # Make the label partly learnable from one feature so a classifier has something to fit,
    # not purely random noise that would make macro-F1 meaningless to assert on.
    signal = data[FEATURE_COLUMNS[0]]
    data["label"] = np.where(signal > 0.3, UP, np.where(signal < -0.3, DOWN, FLAT))
    return pd.DataFrame(data)


def test_split_X_y_separates_label_and_keeps_column_order():
    df = _toy_frame(n=10)
    X, y = split_X_y(df)
    assert list(X.columns) == FEATURE_COLUMNS
    assert "label" not in X.columns
    assert y.tolist() == df["label"].tolist()


ALL_CANDIDATES = set(BASE_MODEL_NAMES) | set(ENSEMBLE_NAMES)


def test_make_models_returns_literature_candidates():
    # docs/09_literature_review.md section 2: base learners + two ensembles.
    assert set(BASE_MODEL_NAMES) == {
        "logistic_regression", "random_forest", "gradient_boosting",
        "xgboost", "lightgbm", "svm", "knn",
    }
    assert set(ENSEMBLE_NAMES) == {"soft_voting", "stacking"}
    assert set(make_models()) == ALL_CANDIDATES


@pytest.mark.parametrize("name", sorted(ALL_CANDIDATES))
def test_each_candidate_fits_and_predicts_valid_classes(name):
    train, val = _toy_frame(seed=1), _toy_frame(seed=2)
    X_train, y_train = split_X_y(train)
    X_val, _ = split_X_y(val)

    model = make_models()[name]
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    proba = model.predict_proba(X_val)

    # Every candidate must speak the project's label space (-1/0/1), not an encoded 0/1/2,
    # because weighted_predict and macro_roc_auc index probabilities by model.classes_.
    assert list(model.classes_) == [DOWN, FLAT, UP]
    assert set(pred) <= {DOWN, FLAT, UP}
    assert proba.shape == (len(X_val), 3)
    assert np.allclose(proba.sum(axis=1), 1.0)


@pytest.mark.parametrize("name", sorted(ALL_CANDIDATES))
def test_each_candidate_learns_a_planted_signal(name):
    # The toy label is a deterministic function of one feature, so every model should beat
    # the 1/3 macro-F1 of guessing. Guards against a wrapper that silently scrambles labels.
    train, val = _toy_frame(n=400, seed=1), _toy_frame(n=200, seed=2)
    model = make_models()[name].fit(*split_X_y(train))
    assert evaluate(model, *split_X_y(val))["macro_f1"] > 0.5


def test_build_model_applies_params():
    model = build_model("xgboost", {"max_depth": 2, "n_estimators": 17})
    params = model.get_params()
    assert params["estimator__max_depth"] == 2
    assert params["estimator__n_estimators"] == 17


def test_build_model_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown model"):
        build_model("deep_magic")


def test_label_encoded_classifier_round_trips_original_labels():
    from sklearn.tree import DecisionTreeClassifier

    X = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]})
    y = pd.Series([UP, DOWN, FLAT, UP, DOWN, FLAT])
    model = LabelEncodedClassifier(DecisionTreeClassifier(random_state=0)).fit(X, y)
    assert list(model.classes_) == [DOWN, FLAT, UP]
    assert list(model.predict(X)) == y.tolist()


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


def test_model_choice_round_trips_through_json(tmp_path):
    path = tmp_path / "selected_model.json"
    choice = {"model": "xgboost", "feature_set": "selected12",
              "params": {"max_depth": 2, "learning_rate": 0.05}, "members": {}}
    save_model_choice(choice, path)
    assert load_model_choice(path) == choice


def test_model_from_choice_rebuilds_base_and_ensemble_models():
    base = model_from_choice({"model": "random_forest", "params": {"max_depth": 3}, "members": {}})
    assert base.get_params()["max_depth"] == 3
    ens = model_from_choice({"model": "soft_voting", "params": {},
                             "members": {"logistic_regression": {"C": 0.5}, "random_forest": {},
                                         "lightgbm": {}, "svm": {}}})
    assert ens.get_params()["logistic_regression__clf__C"] == 0.5
