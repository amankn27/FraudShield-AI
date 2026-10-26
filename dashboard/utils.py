"""Pure helpers for rendering saved prediction records in Streamlit."""
from __future__ import annotations

from typing import Any

import pandas as pd


DISPLAY_COLUMNS = [
    "created_at", "prediction_id", "model_version", "model_score", "risk_score_0_100",
    "risk_level", "predicted_fraud", "decision", "prediction_latency_ms", "investigation_status",
]


def records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert API records into a clean, display-ready dataframe."""
    if not records:
        return pd.DataFrame(columns=DISPLAY_COLUMNS)
    frame = pd.DataFrame(records)
    for column in DISPLAY_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    frame["created_at"] = pd.to_datetime(frame["created_at"], errors="coerce", utc=True)
    for column in ("model_score", "risk_score_0_100", "prediction_latency_ms"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["predicted_fraud"] = frame["predicted_fraud"].astype("boolean")
    return frame[DISPLAY_COLUMNS].copy()


def shap_contributions_to_frame(result: dict[str, Any]) -> pd.DataFrame:
    """Flatten positive and negative SHAP contributions for charting."""
    positive = result.get("top_positive_contributions") or []
    negative = result.get("top_negative_contributions") or []
    rows = []
    for item in positive + negative:
        rows.append({
            "feature": item.get("feature", "unknown"),
            "feature_value": item.get("feature_value"),
            "shap_value": item.get("shap_value", 0.0),
            "direction": item.get("direction", "neutral"),
        })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["shap_value"] = pd.to_numeric(frame["shap_value"], errors="coerce")
        frame = frame.sort_values("shap_value", key=lambda col: col.abs(), ascending=True)
    return frame


def daily_frame(daily_predictions: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a chronologically sorted daily monitoring frame."""
    if not daily_predictions:
        return pd.DataFrame(columns=["date", "total_predictions", "flagged_predictions", "average_latency_ms"])
    frame = pd.DataFrame(daily_predictions)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ("total_predictions", "flagged_predictions", "average_latency_ms"):
        if column not in frame:
            frame[column] = 0
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("date")
