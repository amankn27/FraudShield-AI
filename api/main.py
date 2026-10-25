"""FastAPI service for FraudShield scoring, SHAP, and PostgreSQL prediction logging.

Run from the repository root:
    uvicorn api.main:app --reload --host 127.0.0.1 --port 8000

Set DATABASE_URL to enable persistence. If a database URL is configured,
DATABASE_REQUIRED defaults to true so failed audit writes are not silently lost.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from functools import lru_cache
import logging
from time import perf_counter
from typing import Any, AsyncIterator
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from api.schemas import (
    HealthResponse,
    InvestigationStatus,
    InvestigationUpdateRequest,
    ModelInfoResponse,
    PredictionDetailResponse,
    PredictionListResponse,
    PredictionResponse,
    RiskLevel,
    StoredPredictionResponse,
    TransactionFeatures,
)
from src.config import PROJECT_ROOT
from src.evaluation.shap_explainer import FraudShapService
from src.persistence.database import DatabaseManager, get_database_manager
from src.persistence.repository import (
    create_prediction,
    get_prediction as get_stored_prediction,
    list_predictions,
    update_investigation,
)

SERVICE_NAME = "fraudshield-ai-api"
logger = logging.getLogger(__name__)
MODEL_ARTIFACTS = (
    PROJECT_ROOT / "models" / "xgboost_best.joblib",
    PROJECT_ROOT / "models" / "shap_background.csv",
    PROJECT_ROOT / "models" / "decision_policy.json",
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load required model artifacts once during process startup."""
    try:
        get_explanation_service()
        logger.info("FraudShield model artifacts loaded successfully")
    except Exception:
        logger.exception("FraudShield model artifacts could not be loaded at startup")
    manager = get_database_manager()
    if manager.required and not manager.enabled:
        logger.error("Database is required but its engine could not be initialized")
    if manager.enabled and not manager.ping():
        logger.error("DATABASE_URL is configured, but database health check failed")
    yield


app = FastAPI(
    lifespan=lifespan,
    title="FraudShield AI API",
    version="1.1.0",
    description=(
        "Transaction scoring API with optional SHAP explanations and PostgreSQL prediction history. "
        "Scores are not calibrated real-world fraud probabilities."
    ),
    contact={"name": "FraudShield AI project"},
)


@lru_cache(maxsize=1)
def get_explanation_service() -> FraudShapService:
    """Load XGBoost, policy, and SHAP background once per Python process."""
    return FraudShapService()


def clear_service_cache() -> None:
    """Clear the cached service (useful in tests and after local model replacement)."""
    get_explanation_service.cache_clear()


def _load_service_or_503() -> FraudShapService:
    try:
        return get_explanation_service()
    except Exception as exc:
        logger.exception("FraudShield model artifact loading failed")
        raise HTTPException(
            status_code=503,
            detail="Model service unavailable: required model artifacts could not be loaded.",
        ) from exc


def _database_or_503(*, require_enabled: bool = True) -> DatabaseManager:
    manager = get_database_manager()
    if require_enabled and not manager.enabled:
        raise HTTPException(
            status_code=503,
            detail="Prediction persistence is not configured. Set DATABASE_URL and apply database migrations.",
        )
    if manager.enabled and not manager.ping() and manager.required:
        raise HTTPException(status_code=503, detail="Database is unavailable; request cannot be persisted.")
    return manager


def _shap_summary(result: dict[str, Any]) -> dict[str, Any] | None:
    """Store compact attribution metadata without copying feature values into the DB."""
    if not result.get("shap_available"):
        return None

    def compact(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "feature": item.get("feature"),
                "shap_value": item.get("shap_value"),
                "direction": item.get("direction"),
            }
            for item in items
        ]

    return {
        "output_space": result.get("shap_output_space"),
        "base_value": result.get("shap_base_value"),
        "reconstructed_score": result.get("shap_reconstructed_score"),
        "additivity_error": result.get("shap_additivity_error"),
        "top_positive_contributions": compact(result.get("top_positive_contributions", [])),
        "top_negative_contributions": compact(result.get("top_negative_contributions", [])),
        "interpretation_note": result.get("interpretation_note"),
    }


def _predict_response(
    payload: TransactionFeatures,
    *,
    include_shap: bool,
    top_k: int,
) -> PredictionResponse:
    request_id = str(uuid4())
    started = perf_counter()
    service = _load_service_or_503()
    features = payload.model_dump()

    try:
        if include_shap:
            result: dict[str, Any] = service.explain(features, top_k=top_k)
        else:
            result = service.predict(features)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Transaction features could not be validated.") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Model explanation artifacts are unavailable.") from exc
    except Exception as exc:
        logger.exception("Prediction failed for request_id=%s", request_id)
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check service logs using the request ID.",
        ) from exc

    result.setdefault("shap_available", False)
    result.setdefault("shap_output_space", None)
    result.setdefault("shap_base_value", None)
    result.setdefault("shap_reconstructed_score", None)
    result.setdefault("shap_additivity_error", None)
    result.setdefault("top_positive_contributions", [])
    result.setdefault("top_negative_contributions", [])
    result.setdefault("interpretation_note", None)
    result.update({
        "request_id": request_id,
        "latency_ms": round((perf_counter() - started) * 1000.0, 3),
        "persistence_status": "disabled",
        "prediction_id": None,
    })

    try:
        response = PredictionResponse.model_validate(result)
    except ValidationError as exc:
        logger.exception("Prediction service returned invalid output for request_id=%s", request_id)
        raise HTTPException(status_code=500, detail="Prediction service returned an invalid response.") from exc

    manager = get_database_manager()
    if not manager.enabled:
        if manager.required:
            raise HTTPException(status_code=503, detail="Database is required but unavailable; request was not completed.")
        return response

    persistence_payload = {
        "request_id": response.request_id,
        "created_at": response.created_at,
        "model_version": response.model_version,
        "policy_version": response.policy_version,
        "model_type": response.model_type,
        "feature_count": response.feature_count,
        "model_score": response.model_score,
        "risk_score_0_100": response.risk_score_0_100,
        "decision_threshold": response.decision_threshold,
        "risk_level": response.risk_level,
        "predicted_fraud": response.predicted_fraud,
        "decision": response.decision,
        "prediction_latency_ms": response.latency_ms,
        "shap_available": response.shap_available,
        "shap_summary": _shap_summary(result),
    }
    try:
        saved = create_prediction(manager, persistence_payload)
        result["prediction_id"] = saved["prediction_id"]
        result["persistence_status"] = "saved"
    except Exception as exc:
        logger.exception("Prediction persistence failed for request_id=%s", request_id)
        if manager.required:
            raise HTTPException(
                status_code=503,
                detail="Prediction could not be durably recorded; request was not completed.",
            ) from exc
        result["persistence_status"] = "failed"

    try:
        return PredictionResponse.model_validate(result)
    except ValidationError as exc:
        logger.exception("Response validation failed after persistence for request_id=%s", request_id)
        raise HTTPException(status_code=500, detail="Prediction service returned an invalid response.") from exc


@app.get("/health", response_model=HealthResponse, tags=["operations"])
def health() -> HealthResponse:
    """Liveness/readiness summary for model artifacts and configured database."""
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in MODEL_ARTIFACTS if not path.is_file()]
    loaded = get_explanation_service.cache_info().currsize > 0
    artifacts_available = not missing
    manager = get_database_manager()
    db_ready = manager.ping() if manager.enabled else False
    ready = artifacts_available and loaded and (not manager.required or db_ready)
    return HealthResponse(
        status="ok" if ready else "degraded",
        service=SERVICE_NAME,
        ready=ready,
        model_loaded=loaded,
        missing_artifacts=missing,
        database_enabled=manager.enabled,
        database_ready=db_ready,
        database_required=manager.required,
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["model"])
def model_info() -> ModelInfoResponse:
    service = _load_service_or_503()
    info = service.model_info()
    info["service"] = SERVICE_NAME
    try:
        return ModelInfoResponse.model_validate(info)
    except ValidationError as exc:
        raise HTTPException(status_code=500, detail="Model metadata is invalid.") from exc


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(
    payload: TransactionFeatures,
    include_shap: bool = Query(default=True, description="Return local SHAP contributions"),
    top_k: int = Query(default=5, ge=1, le=30, description="Maximum positive and negative contributors"),
) -> PredictionResponse:
    """Score a transaction, return optional SHAP and persist when configured."""
    return _predict_response(payload, include_shap=include_shap, top_k=top_k)


@app.post("/explain", response_model=PredictionResponse, tags=["explainability"], deprecated=True)
def explain_transaction(
    payload: TransactionFeatures,
    top_k: int = Query(default=5, ge=1, le=30, description="Maximum positive and negative contributors"),
) -> PredictionResponse:
    """Backward-compatible alias for POST /predict?include_shap=true."""
    return _predict_response(payload, include_shap=True, top_k=top_k)


@app.get("/predictions", response_model=PredictionListResponse, tags=["history"])
def prediction_history(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    investigation_status: InvestigationStatus | None = Query(default=None),
    risk_level: RiskLevel | None = Query(default=None),
) -> PredictionListResponse:
    """List persisted predictions with pagination and optional filters."""
    manager = _database_or_503()
    try:
        items, total = list_predictions(
            manager, limit=limit, offset=offset,
            investigation_status=investigation_status, risk_level=risk_level,
        )
    except Exception as exc:
        logger.exception("Could not list prediction records")
        raise HTTPException(status_code=503, detail="Prediction history is temporarily unavailable.") from exc
    return PredictionListResponse(items=items, total=total, limit=limit, offset=offset)


@app.get("/predictions/{prediction_id}", response_model=PredictionDetailResponse, tags=["history"])
def prediction_detail(prediction_id: UUID) -> PredictionDetailResponse:
    """Return a persisted prediction and its investigation status history."""
    manager = _database_or_503()
    try:
        record = get_stored_prediction(manager, str(prediction_id))
    except Exception as exc:
        logger.exception("Could not load prediction_id=%s", prediction_id)
        raise HTTPException(status_code=503, detail="Prediction history is temporarily unavailable.") from exc
    if record is None:
        raise HTTPException(status_code=404, detail="Prediction record not found.")
    try:
        return PredictionDetailResponse.model_validate(record)
    except ValidationError as exc:
        logger.exception("Stored prediction record failed response validation")
        raise HTTPException(status_code=500, detail="Stored prediction record is invalid.") from exc


@app.patch("/predictions/{prediction_id}/investigation", response_model=PredictionDetailResponse, tags=["investigation"])
def update_prediction_investigation(
    prediction_id: UUID,
    payload: InvestigationUpdateRequest,
) -> PredictionDetailResponse:
    """Update investigation status and append an audit-history event."""
    manager = _database_or_503()
    try:
        record = update_investigation(
            manager, str(prediction_id), new_status=payload.status, note=payload.note
        )
    except Exception as exc:
        logger.exception("Could not update investigation for prediction_id=%s", prediction_id)
        raise HTTPException(status_code=503, detail="Investigation status could not be updated.") from exc
    if record is None:
        raise HTTPException(status_code=404, detail="Prediction record not found.")
    try:
        return PredictionDetailResponse.model_validate(record)
    except ValidationError as exc:
        raise HTTPException(status_code=500, detail="Updated investigation record is invalid.") from exc


@app.exception_handler(Exception)
async def unexpected_exception_handler(request, exc: Exception) -> JSONResponse:
    """Avoid leaking stack traces, database URLs, or transaction features to clients."""
    logger.exception("Unhandled API error for %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
