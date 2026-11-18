import optuna
import pytest

from src.modeling.train import BASE_MODEL_NAMES, build_model, split_X_y
from src.modeling.tuning import SEARCH_SPACES, suggest_params, tune_model
from tests.test_modeling import _toy_frame


def test_every_base_model_has_a_search_space():
    assert set(SEARCH_SPACES) == set(BASE_MODEL_NAMES)


@pytest.mark.parametrize("name", BASE_MODEL_NAMES)
def test_suggested_params_build_a_valid_model(name):
    # A FixedTrial-free check: sample once from a real study and make sure the
    # sampled values are accepted by the model factory and the model fits.
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    trial = study.ask()
    params = suggest_params(name, trial)
    model = build_model(name, params).fit(*split_X_y(_toy_frame(seed=1)))
    assert len(model.predict(split_X_y(_toy_frame(seed=2))[0])) == 200


def test_tune_model_returns_params_and_validation_score():
    X_tr, y_tr = split_X_y(_toy_frame(n=300, seed=1))
    X_val, y_val = split_X_y(_toy_frame(n=150, seed=2))
    result = tune_model("logistic_regression", X_tr, y_tr, X_val, y_val, n_trials=4, seed=0)
    assert set(result) == {"params", "val_macro_f1", "n_trials"}
    assert 0.0 <= result["val_macro_f1"] <= 1.0
    assert result["n_trials"] == 4
    assert "C" in result["params"]


def test_tune_model_is_reproducible_with_a_seed():
    X_tr, y_tr = split_X_y(_toy_frame(n=300, seed=1))
    X_val, y_val = split_X_y(_toy_frame(n=150, seed=2))
    a = tune_model("random_forest", X_tr, y_tr, X_val, y_val, n_trials=3, seed=7)
    b = tune_model("random_forest", X_tr, y_tr, X_val, y_val, n_trials=3, seed=7)
    assert a == b
