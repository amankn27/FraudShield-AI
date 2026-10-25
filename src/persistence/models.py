"""SQLAlchemy models for prediction logs and investigation history."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    Uuid,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    """Return an aware UTC timestamp."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Base for project SQLAlchemy models."""


class PredictionRecord(Base):
    """Persisted output of a single inference request (not raw transaction features)."""

    __tablename__ = "prediction_records"
    __table_args__ = (
        CheckConstraint("model_score >= 0 AND model_score <= 1", name="ck_prediction_score_range"),
        CheckConstraint("risk_score_0_100 >= 0 AND risk_score_0_100 <= 100", name="ck_prediction_risk_score_range"),
        CheckConstraint("decision_threshold >= 0 AND decision_threshold <= 1", name="ck_prediction_threshold_range"),
        CheckConstraint("prediction_latency_ms >= 0", name="ck_prediction_latency_nonnegative"),
        CheckConstraint(
            "risk_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')",
            name="ck_prediction_risk_level",
        ),
        CheckConstraint(
            "investigation_status IN ('NEW', 'IN_REVIEW', 'CONFIRMED_FRAUD', 'FALSE_POSITIVE', 'CLOSED')",
            name="ck_prediction_investigation_status",
        ),
        Index("ix_prediction_created_at_id", "created_at", "id"),
    )

    id: Mapped[str] = mapped_column(Uuid(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    request_id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now, index=True)

    model_version: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    policy_version: Mapped[str] = mapped_column(String(100), nullable=False)
    model_type: Mapped[str] = mapped_column(String(100), nullable=False)
    feature_count: Mapped[int] = mapped_column(Integer, nullable=False)

    model_score: Mapped[float] = mapped_column(Float, nullable=False)
    risk_score_0_100: Mapped[float] = mapped_column(Float, nullable=False)
    decision_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    predicted_fraud: Mapped[bool] = mapped_column(Boolean, nullable=False)
    decision: Mapped[str] = mapped_column(String(32), nullable=False)
    prediction_latency_ms: Mapped[float] = mapped_column(Float, nullable=False)

    shap_available: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    shap_summary: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    investigation_status: Mapped[str] = mapped_column(String(32), nullable=False, default="NEW", index=True)
    investigation_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    investigation_note: Mapped[str | None] = mapped_column(Text, nullable=True)

    investigation_events: Mapped[list["InvestigationEvent"]] = relationship(
        back_populates="prediction", cascade="all, delete-orphan", order_by="InvestigationEvent.changed_at"
    )


class InvestigationEvent(Base):
    """Append-only history of investigation status changes."""

    __tablename__ = "investigation_events"
    __table_args__ = (
        CheckConstraint(
            "new_status IN ('NEW', 'IN_REVIEW', 'CONFIRMED_FRAUD', 'FALSE_POSITIVE', 'CLOSED')",
            name="ck_investigation_event_new_status",
        ),
        Index("ix_investigation_event_prediction_changed", "prediction_id", "changed_at"),
    )

    id: Mapped[str] = mapped_column(Uuid(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    prediction_id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False), ForeignKey("prediction_records.id", ondelete="CASCADE"), nullable=False
    )
    previous_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    new_status: Mapped[str] = mapped_column(String(32), nullable=False)
    changed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)

    prediction: Mapped[PredictionRecord] = relationship(back_populates="investigation_events")
