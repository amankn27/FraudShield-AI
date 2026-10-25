"""Unit tests for FraudShield AI imbalance strategies."""
import numpy as np
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline as ImbPipeline

from src.models.imbalance_experiments import make_experiment_pipeline


@pytest.mark.parametrize("strategy", ["baseline", "class_weight", "undersampling", "smote"])
def test_factory_returns_pipeline(strategy):
    model = make_experiment_pipeline(strategy, random_state=7, max_iter=100)
    assert isinstance(model, ImbPipeline)
    assert model.steps[0][0] == "imputer"
    assert model.steps[1][0] == "scaler"
    assert model.steps[-1][0] == "classifier"


def test_class_weight_does_not_add_sampler():
    model = make_experiment_pipeline("class_weight")
    assert "sampler" not in model.named_steps
    assert model.named_steps["classifier"].class_weight == "balanced"


def test_resampling_pipeline_fits_and_predicts():
    rng = np.random.default_rng(10)
    X_major = rng.normal(0, 1, size=(120, 4))
    X_minor = rng.normal(1.5, 1, size=(20, 4))
    X = pd.DataFrame(np.vstack([X_major, X_minor]), columns=list("abcd"))
    y = pd.Series([0] * len(X_major) + [1] * len(X_minor))
    for strategy in ["undersampling", "smote"]:
        model = make_experiment_pipeline(strategy, random_state=7, max_iter=200)
        model.fit(X, y)
        probabilities = model.predict_proba(X[:10])[:, 1]
        assert probabilities.shape == (10,)
        assert np.isfinite(probabilities).all()


def test_invalid_strategy_fails_fast():
    with pytest.raises(ValueError, match="Unknown strategy"):
        make_experiment_pipeline("random-magic")
