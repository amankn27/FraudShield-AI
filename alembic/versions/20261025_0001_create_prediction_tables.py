"""Create prediction_records and investigation_events.

Revision ID: 20261025_0001
Revises:
Create Date: 2026-10-25
"""
from alembic import op
import sqlalchemy as sa

revision = "20261025_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "prediction_records",
        sa.Column("id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("request_id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("model_version", sa.String(length=100), nullable=False),
        sa.Column("policy_version", sa.String(length=100), nullable=False),
        sa.Column("model_type", sa.String(length=100), nullable=False),
        sa.Column("feature_count", sa.Integer(), nullable=False),
        sa.Column("model_score", sa.Float(), nullable=False),
        sa.Column("risk_score_0_100", sa.Float(), nullable=False),
        sa.Column("decision_threshold", sa.Float(), nullable=False),
        sa.Column("risk_level", sa.String(length=16), nullable=False),
        sa.Column("predicted_fraud", sa.Boolean(), nullable=False),
        sa.Column("decision", sa.String(length=32), nullable=False),
        sa.Column("prediction_latency_ms", sa.Float(), nullable=False),
        sa.Column("shap_available", sa.Boolean(), nullable=False),
        sa.Column("shap_summary", sa.JSON(), nullable=True),
        sa.Column("investigation_status", sa.String(length=32), nullable=False),
        sa.Column("investigation_updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("investigation_note", sa.Text(), nullable=True),
        sa.CheckConstraint("model_score >= 0 AND model_score <= 1", name="ck_prediction_score_range"),
        sa.CheckConstraint("risk_score_0_100 >= 0 AND risk_score_0_100 <= 100", name="ck_prediction_risk_score_range"),
        sa.CheckConstraint("decision_threshold >= 0 AND decision_threshold <= 1", name="ck_prediction_threshold_range"),
        sa.CheckConstraint("prediction_latency_ms >= 0", name="ck_prediction_latency_nonnegative"),
        sa.CheckConstraint("risk_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')", name="ck_prediction_risk_level"),
        sa.CheckConstraint("investigation_status IN ('NEW', 'IN_REVIEW', 'CONFIRMED_FRAUD', 'FALSE_POSITIVE', 'CLOSED')", name="ck_prediction_investigation_status"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_prediction_records_request_id", "prediction_records", ["request_id"], unique=True)
    op.create_index("ix_prediction_records_created_at", "prediction_records", ["created_at"], unique=False)
    op.create_index("ix_prediction_records_model_version", "prediction_records", ["model_version"], unique=False)
    op.create_index("ix_prediction_records_risk_level", "prediction_records", ["risk_level"], unique=False)
    op.create_index("ix_prediction_records_investigation_status", "prediction_records", ["investigation_status"], unique=False)
    op.create_index("ix_prediction_created_at_id", "prediction_records", ["created_at", "id"], unique=False)

    op.create_table(
        "investigation_events",
        sa.Column("id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("prediction_id", sa.Uuid(as_uuid=False), nullable=False),
        sa.Column("previous_status", sa.String(length=32), nullable=True),
        sa.Column("new_status", sa.String(length=32), nullable=False),
        sa.Column("changed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.CheckConstraint("new_status IN ('NEW', 'IN_REVIEW', 'CONFIRMED_FRAUD', 'FALSE_POSITIVE', 'CLOSED')", name="ck_investigation_event_new_status"),
        sa.ForeignKeyConstraint(["prediction_id"], ["prediction_records.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_investigation_event_prediction_changed", "investigation_events", ["prediction_id", "changed_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_investigation_event_prediction_changed", table_name="investigation_events")
    op.drop_table("investigation_events")
    op.drop_index("ix_prediction_created_at_id", table_name="prediction_records")
    op.drop_index("ix_prediction_records_investigation_status", table_name="prediction_records")
    op.drop_index("ix_prediction_records_risk_level", table_name="prediction_records")
    op.drop_index("ix_prediction_records_model_version", table_name="prediction_records")
    op.drop_index("ix_prediction_records_created_at", table_name="prediction_records")
    op.drop_index("ix_prediction_records_request_id", table_name="prediction_records")
    op.drop_table("prediction_records")
