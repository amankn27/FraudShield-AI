import numpy as np
import pandas as pd
import pytest

from src.evaluation.shap_explainer import FraudShapService


def test_validate_features_rejects_missing_and_extra_without_loading_service():
    service = FraudShapService.__new__(FraudShapService)
    service.feature_names = ["Time", "V1", "Amount"]
    with pytest.raises(ValueError, match="Missing required features"):
        service.validate_features({"Time": 1, "V1": 2})
    with pytest.raises(ValueError, match="Unexpected features"):
        service.validate_features({"Time": 1, "V1": 2, "Amount": 3, "Class": 0})


def test_validate_features_rejects_non_finite_values():
    service = FraudShapService.__new__(FraudShapService)
    service.feature_names = ["Time", "V1", "Amount"]
    with pytest.raises(ValueError, match="finite"):
        service.validate_features({"Time": 1, "V1": float("nan"), "Amount": 3})


def test_validate_features_orders_columns_to_model_feature_order():
    service = FraudShapService.__new__(FraudShapService)
    service.feature_names = ["Time", "V1", "Amount"]
    frame = service.validate_features({"Amount": 3, "V1": 2, "Time": 1})
    assert frame.columns.tolist() == ["Time", "V1", "Amount"]
    assert frame.iloc[0].tolist() == [1.0, 2.0, 3.0]


def test_risk_levels_follow_policy_boundaries():
    service = FraudShapService.__new__(FraudShapService)
    service.policy = {"risk_bands": {
        "LOW": {"upper_exclusive": 0.1},
        "MEDIUM": {"upper_exclusive": 0.3},
        "HIGH": {"upper_exclusive": 0.8},
        "CRITICAL": {"lower_inclusive": 0.8, "upper_inclusive": 1.0},
    }}
    assert service.risk_level(0.05) == "LOW"
    assert service.risk_level(0.1) == "MEDIUM"
    assert service.risk_level(0.3) == "HIGH"
    assert service.risk_level(0.8) == "CRITICAL"
