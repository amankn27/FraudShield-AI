"""Persistence operations for prediction and investigation records."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.persistence.database import DatabaseManager
from src.persistence.models import InvestigationEvent, PredictionRecord

INVESTIGATION_STATUSES = {"NEW", "IN_REVIEW", "CONFIRMED_FRAUD", "FALSE_POSITIVE", "CLOSED"}


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _event_dict(event: InvestigationEvent) -> dict[str, Any]:
    return {
        "event_id": event.id,
        "previous_status": event.previous_status,
        "new_status": event.new_status,
        "changed_at": event.changed_at,
        "note": event.note,
    }


def _record_dict(row: PredictionRecord, include_events: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {
        "prediction_id": row.id,
        "request_id": row.request_id,
        "created_at": row.created_at,
        "model_version": row.model_version,
        "policy_version": row.policy_version,
        "model_type": row.model_type,
        "feature_count": row.feature_count,
        "model_score": row.model_score,
        "risk_score_0_100": row.risk_score_0_100,
        "decision_threshold": row.decision_threshold,
        "risk_level": row.risk_level,
        "predicted_fraud": row.predicted_fraud,
        "decision": row.decision,
        "prediction_latency_ms": row.prediction_latency_ms,
        "shap_available": row.shap_available,
        "shap_summary": row.shap_summary,
        "investigation_status": row.investigation_status,
        "investigation_updated_at": row.investigation_updated_at,
        "investigation_note": row.investigation_note,
    }
    if include_events:
        result["investigation_events"] = [_event_dict(event) for event in row.investigation_events]
    return result


def create_prediction(manager: DatabaseManager, data: dict[str, Any]) -> dict[str, Any]:
    """Persist one prediction and its initial NEW investigation event atomically."""
    with manager.session() as session:
        row = PredictionRecord(
            id=str(uuid4()),
            request_id=data["request_id"],
            created_at=data.get("created_at") or _now_utc(),
            model_version=data["model_version"],
            policy_version=data["policy_version"],
            model_type=data["model_type"],
            feature_count=data["feature_count"],
            model_score=data["model_score"],
            risk_score_0_100=data["risk_score_0_100"],
            decision_threshold=data["decision_threshold"],
            risk_level=data["risk_level"],
            predicted_fraud=data["predicted_fraud"],
            decision=data["decision"],
            prediction_latency_ms=data["prediction_latency_ms"],
            shap_available=data.get("shap_available", False),
            shap_summary=data.get("shap_summary"),
            investigation_status="NEW",
        )
        session.add(row)
        session.flush()
        session.add(InvestigationEvent(
            id=str(uuid4()), prediction_id=row.id, previous_status=None,
            new_status="NEW", changed_at=row.created_at, note="Prediction created",
        ))
        session.commit()
        session.refresh(row)
        return _record_dict(row)


def get_prediction(manager: DatabaseManager, prediction_id: str) -> dict[str, Any] | None:
    with manager.session() as session:
        row = session.get(PredictionRecord, prediction_id)
        if row is None:
            return None
        # Touch relationship before session closes and create detached response data.
        return _record_dict(row, include_events=True)


def list_predictions(
    manager: DatabaseManager,
    *,
    limit: int = 50,
    offset: int = 0,
    investigation_status: str | None = None,
    risk_level: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    with manager.session() as session:
        query = select(PredictionRecord)
        count_query = select(func.count()).select_from(PredictionRecord)
        if investigation_status:
            query = query.where(PredictionRecord.investigation_status == investigation_status)
            count_query = count_query.where(PredictionRecord.investigation_status == investigation_status)
        if risk_level:
            query = query.where(PredictionRecord.risk_level == risk_level)
            count_query = count_query.where(PredictionRecord.risk_level == risk_level)
        total = int(session.scalar(count_query) or 0)
        rows = session.scalars(
            query.order_by(PredictionRecord.created_at.desc(), PredictionRecord.id.desc())
            .limit(limit).offset(offset)
        ).all()
        return [_record_dict(row) for row in rows], total


def update_investigation(
    manager: DatabaseManager,
    prediction_id: str,
    *,
    new_status: str,
    note: str | None = None,
) -> dict[str, Any] | None:
    if new_status not in INVESTIGATION_STATUSES:
        raise ValueError(f"Unsupported investigation status: {new_status}")
    with manager.session() as session:
        row = session.get(PredictionRecord, prediction_id)
        if row is None:
            return None
        previous_status = row.investigation_status
        changed_at = _now_utc()
        row.investigation_status = new_status
        row.investigation_updated_at = changed_at
        if note is not None:
            row.investigation_note = note
        session.add(InvestigationEvent(
            id=str(uuid4()), prediction_id=row.id, previous_status=previous_status,
            new_status=new_status, changed_at=changed_at, note=note,
        ))
        session.commit()
        session.refresh(row)
        return _record_dict(row, include_events=True)
