"""Production-style FastAPI service for FraudShield AI inference.

Run from the repository root with:
    uvicorn api.main:app --reload

The model and policy are loaded lazily and cached once per process. SHAP's
TreeExplainer is itself initialized lazily only when explanation is requested.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from functools import lru_cache
import logging
from pathlib import Path
from time import perf_counter
from typing import Any, AsyncIterator
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from api.schemas import HealthResponse, ModelInfoResponse, PredictionResponse, TransactionFeatures
from src.config import PROJECT_ROOT
from src.evaluation.shap_explainer import FraudShapService

SERVICE_NAME = "fraudshield-ai-api"
logger = logging.getLogger(__name__)
MODEL_ARTIFACTS = (
    PROJECT_ROOT / "models" / "xgboost_best.joblib",
    PROJECT_ROOT / "models" / "shap_background.csv",
    PROJECT_ROOT / "models" / "decision_policy.json",
)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Preload model artifacts once during normal server startup.

    If artifacts are unavailable, keep the HTTP process alive so /health can
    report degraded readiness and clients receive a clear 503 from inference.
    """
    try:
        get_explanation_service()
        logger.info("FraudShield model artifacts loaded successfully")
    except Exception:
        logger.exception("FraudShield model artifacts could not be loaded at startup")
    yield


app = FastAPI(
    lifespan=lifespan,
    title="FraudShield AI API",
    version="1.0.0",
    description=(
        "Transaction scoring API for the FraudShield XGBoost model. Optional SHAP "
        "contributions explain the model output; scores are not calibrated real-world fraud probabilities."
    ),
    contact={"name": "FraudShield AI project"},
)


@lru_cache(maxsize=1)
def get_explanation_service() -> FraudShapService:
    """Load XGBoost, policy, and SHAP background once per Python process."""
    return FraudShapService()


def clear_service_cache() -> None:
    """Clear the cached service (useful for tests and local model replacement)."""
    get_explanation_service.cache_clear()


def _load_service_or_503() -> FraudShapService:
    try:
        return get_explanation_service()
    except (FileNotFoundError, ValueError, OSError) as exc:
        # Do not expose local paths or internal artifact details in HTTP responses.
        logger.exception("FraudShield model artifact loading failed")
        raise HTTPException(
            status_code=503,
            detail="Model service unavailable: required model artifacts could not be loaded.",
        ) from exc


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
        result: dict[str, Any]
        if include_shap:
            result = service.explain(features, top_k=top_k)
        else:
            result = service.predict(features)
    except ValueError as exc:
        # Schema validation usually catches these; this guards model-order/contract mismatches.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Explanation artifacts are unavailable.") from exc
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
    })
    try:
        return PredictionResponse.model_validate(result)
    except ValidationError as exc:
        # A broken model/service response is an internal error, not bad client input.
        raise HTTPException(status_code=500, detail="Prediction service returned an invalid response.") from exc


@app.get("/health", response_model=HealthResponse, tags=["operations"])
def health() -> HealthResponse:
    """Liveness/readiness summary without forcing expensive model loading."""
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in MODEL_ARTIFACTS if not path.is_file()]
    loaded = get_explanation_service.cache_info().currsize > 0
    artifacts_available = not missing
    ready = artifacts_available and loaded
    return HealthResponse(
        status="ok" if ready else "degraded",
        service=SERVICE_NAME,
        ready=ready,
        model_loaded=loaded,
        missing_artifacts=missing,
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["model"])
def model_info() -> ModelInfoResponse:
    """Return model, input-schema, threshold, policy, and library metadata."""
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
    """Score a transaction and optionally return its local SHAP explanation."""
    return _predict_response(payload, include_shap=include_shap, top_k=top_k)


@app.post("/explain", response_model=PredictionResponse, tags=["explainability"], deprecated=True)
def explain_transaction(
    payload: TransactionFeatures,
    top_k: int = Query(default=5, ge=1, le=30, description="Maximum positive and negative contributors"),
) -> PredictionResponse:
    """Backward-compatible alias for POST /predict?include_shap=true."""
    return _predict_response(payload, include_shap=True, top_k=top_k)


@app.exception_handler(Exception)
async def unexpected_exception_handler(request, exc: Exception) -> JSONResponse:
    """Avoid leaking stack traces, model paths, or input data to API clients."""
    logger.exception("Unhandled API error for %s %s", request.method, request.url.path)
    # FastAPI's HTTPException and request validation handlers are handled by FastAPI
    # before this catch-all in normal routing; this is a final defensive response.
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
