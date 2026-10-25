"""FastAPI endpoint for transaction-level SHAP explanations.

This focused endpoint is an explainability slice for Phase 7. Phase 10 can
expand it with general prediction, health/model metadata, persistence, and auth.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, field_validator

from src.evaluation.shap_explainer import FraudShapService


class TransactionFeatures(BaseModel):
    """All numeric inputs required by the saved XGBoost model."""

    model_config = ConfigDict(extra="forbid")

    Time: float
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
    Amount: float

    @field_validator("Time", "Amount", *[f"V{i}" for i in range(1, 29)])
    @classmethod
    def values_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Feature values must be finite numeric values")
        return value


app = FastAPI(
    title="FraudShield AI Explanation API",
    version="0.7.0",
    description=(
        "Returns the XGBoost model score, Phase 6 decision, and SHAP feature contributions. "
        "The model score is not guaranteed to be calibrated as a real-world fraud probability."
    ),
)


@lru_cache(maxsize=1)
def get_explanation_service() -> FraudShapService:
    """Lazily create the model/explainer so /health stays lightweight."""
    return FraudShapService()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "fraudshield-explanation-api"}


@app.post("/explain")
def explain_transaction(payload: TransactionFeatures, top_k: int = 5) -> dict[str, Any]:
    """Explain a single transaction using positive and negative SHAP drivers."""
    if not 1 <= top_k <= 30:
        raise HTTPException(status_code=422, detail="top_k must be between 1 and 30")
    try:
        service = get_explanation_service()
        result = service.explain(payload.model_dump(), top_k=top_k)
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # Return a generic service error, not internals to clients.
        raise HTTPException(status_code=500, detail="Unable to create SHAP explanation") from exc
