"""Request and response schemas for the FraudShield HTTP API."""
from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TransactionFeatures(BaseModel):
    """Complete feature vector required by the Phase 5 XGBoost model."""

    model_config = ConfigDict(extra="forbid")

    Time: float = Field(ge=0, description="Seconds elapsed since the dataset began")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(ge=0, description="Transaction amount in dataset units")

    @field_validator("Time", "Amount", *[f"V{i}" for i in range(1, 29)])
    @classmethod
    def values_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Feature values must be finite numeric values")
        return value


class FeatureContribution(BaseModel):
    feature: str
    feature_value: float
    shap_value: float
    direction: Literal["increases_model_score", "decreases_model_score", "neutral"]


class PredictionResponse(BaseModel):
    request_id: str
    model_version: str
    policy_version: str
    model_type: str
    feature_count: int
    model_score: float = Field(ge=0, le=1)
    score_interpretation: str
    risk_score_0_100: float = Field(ge=0, le=100)
    decision_threshold: float = Field(ge=0, le=1)
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    predicted_fraud: bool
    decision: str
    latency_ms: float = Field(ge=0)
    shap_available: bool
    shap_output_space: str | None = None
    shap_base_value: float | None = None
    shap_reconstructed_score: float | None = None
    shap_additivity_error: float | None = None
    top_positive_contributions: list[FeatureContribution] = Field(default_factory=list)
    top_negative_contributions: list[FeatureContribution] = Field(default_factory=list)
    interpretation_note: str | None = None


class ModelInfoResponse(BaseModel):
    service: str
    model_version: str
    model_type: str
    policy_version: str
    feature_count: int
    feature_names: list[str]
    decision_threshold: float
    risk_bands: dict
    shap_available: bool
    score_semantics: str
    libraries: dict[str, str]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    service: str
    ready: bool
    model_loaded: bool
    missing_artifacts: list[str] = Field(default_factory=list)
