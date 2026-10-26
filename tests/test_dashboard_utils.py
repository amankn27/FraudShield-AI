from dashboard.utils import daily_frame, records_to_frame, shap_contributions_to_frame


def test_records_to_frame_empty_has_expected_columns():
    frame = records_to_frame([])
    assert "prediction_id" in frame.columns
    assert "risk_level" in frame.columns
    assert frame.empty


def test_records_to_frame_coerces_numeric_values_and_timestamps():
    frame = records_to_frame([{
        "prediction_id": "p1", "created_at": "2026-10-25T10:00:00Z", "model_version": "xgb-v1",
        "model_score": "0.82", "risk_score_0_100": "82", "risk_level": "CRITICAL",
        "predicted_fraud": True, "decision": "FLAG_FOR_REVIEW", "prediction_latency_ms": "12.3",
        "investigation_status": "NEW",
    }])
    assert frame.iloc[0]["model_score"] == 0.82
    assert frame.iloc[0]["risk_score_0_100"] == 82.0
    assert str(frame.iloc[0]["created_at"].tz) == "UTC"


def test_shap_contributions_combine_positive_and_negative():
    frame = shap_contributions_to_frame({
        "top_positive_contributions": [{"feature": "V14", "shap_value": 0.4, "direction": "increases_model_score"}],
        "top_negative_contributions": [{"feature": "V4", "shap_value": -0.2, "direction": "decreases_model_score"}],
    })
    assert set(frame["feature"]) == {"V14", "V4"}
    assert len(frame) == 2


def test_daily_frame_sorts_dates_and_converts_numbers():
    frame = daily_frame([
        {"date": "2026-10-26", "total_predictions": "4", "flagged_predictions": "1", "average_latency_ms": "10"},
        {"date": "2026-10-25", "total_predictions": "2", "flagged_predictions": "0", "average_latency_ms": "6"},
    ])
    assert frame.iloc[0]["date"].date().isoformat() == "2026-10-25"
    assert frame.iloc[0]["total_predictions"] == 2
