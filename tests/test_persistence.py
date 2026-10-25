from __future__ import annotations

from sqlalchemy import select, func
from fastapi.testclient import TestClient

from api import main
from src.persistence.database import DatabaseManager
from src.persistence.models import InvestigationEvent, PredictionRecord
from src.persistence.repository import create_prediction, get_prediction, list_predictions, update_investigation


def make_manager() -> DatabaseManager:
    manager = DatabaseManager("sqlite+pysqlite:///:memory:", required=True)
    manager.create_schema_for_local_dev()
    return manager


def prediction_payload(request_id="request-test-001"):
    from datetime import datetime, timezone
    return {
        "request_id": request_id,
        "created_at": datetime.now(timezone.utc),
        "model_version": "xgboost_best",
        "policy_version": "1.0.0",
        "model_type": "XGBClassifier",
        "feature_count": 30,
        "model_score": 0.82,
        "risk_score_0_100": 82.0,
        "decision_threshold": 0.7853,
        "risk_level": "CRITICAL",
        "predicted_fraud": True,
        "decision": "FLAG_FOR_REVIEW",
        "prediction_latency_ms": 12.34,
        "shap_available": True,
        "shap_summary": {"top_positive_contributions": [{"feature": "V14", "shap_value": 0.3}]},
    }


def test_prediction_record_and_initial_investigation_event_are_persisted():
    manager = make_manager()
    saved = create_prediction(manager, prediction_payload())
    assert saved["prediction_id"]
    assert saved["investigation_status"] == "NEW"
    assert saved["model_version"] == "xgboost_best"
    assert saved["risk_score_0_100"] == 82.0
    assert saved["prediction_latency_ms"] == 12.34
    assert saved["shap_summary"]["top_positive_contributions"][0]["feature"] == "V14"
    detail = get_prediction(manager, saved["prediction_id"])
    assert detail is not None
    assert len(detail["investigation_events"]) == 1
    assert detail["investigation_events"][0]["new_status"] == "NEW"
    manager.dispose()


def test_prediction_list_filters_and_pagination():
    manager = make_manager()
    one = create_prediction(manager, prediction_payload("request-a"))
    payload = prediction_payload("request-b")
    payload["risk_level"] = "HIGH"
    payload["predicted_fraud"] = False
    create_prediction(manager, payload)
    rows, total = list_predictions(manager, limit=1, offset=0, risk_level="HIGH")
    assert total == 1
    assert len(rows) == 1
    assert rows[0]["risk_level"] == "HIGH"
    manager.dispose()


def test_investigation_update_appends_status_history():
    manager = make_manager()
    saved = create_prediction(manager, prediction_payload())
    updated = update_investigation(
        manager, saved["prediction_id"], new_status="IN_REVIEW", note="Analyst review started"
    )
    assert updated is not None
    assert updated["investigation_status"] == "IN_REVIEW"
    assert updated["investigation_note"] == "Analyst review started"
    assert len(updated["investigation_events"]) == 2
    assert updated["investigation_events"][1]["previous_status"] == "NEW"
    assert updated["investigation_events"][1]["new_status"] == "IN_REVIEW"
    manager.dispose()


def test_update_unknown_prediction_returns_none():
    manager = make_manager()
    assert update_investigation(manager, "00000000-0000-0000-0000-000000000000", new_status="CLOSED") is None
    manager.dispose()


def test_database_manager_without_url_is_disabled():
    manager = DatabaseManager("")
    assert manager.enabled is False
    assert manager.ping() is False


def test_api_persists_prediction_and_updates_investigation(monkeypatch):
    manager = make_manager()
    monkeypatch.setattr(main, "get_database_manager", lambda: manager)

    class FakeService:
        def predict(self, payload):
            assert len(payload) == 30
            return {
                "model_version": "xgboost_best", "policy_version": "1.0.0",
                "model_type": "XGBClassifier", "feature_count": 30,
                "model_score": 0.82, "score_interpretation": "uncalibrated score",
                "risk_score_0_100": 82.0, "decision_threshold": 0.7853,
                "risk_level": "CRITICAL", "predicted_fraud": True,
                "decision": "FLAG_FOR_REVIEW", "shap_available": False,
            }

    monkeypatch.setattr(main, "get_explanation_service", lambda: FakeService())
    payload = {"Time": 12345.0, "Amount": 149.5}
    payload.update({f"V{i}": 0.01 * i for i in range(1, 29)})
    client = TestClient(main.app)
    response = client.post("/predict?include_shap=false", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["persistence_status"] == "saved"
    prediction_id = body["prediction_id"]
    assert prediction_id

    history = client.get("/predictions?limit=10")
    assert history.status_code == 200, history.text
    assert history.json()["total"] == 1
    assert history.json()["items"][0]["prediction_id"] == prediction_id

    updated = client.patch(
        f"/predictions/{prediction_id}/investigation",
        json={"status": "CONFIRMED_FRAUD", "note": "Test review"},
    )
    assert updated.status_code == 200, updated.text
    detail = updated.json()
    assert detail["investigation_status"] == "CONFIRMED_FRAUD"
    assert detail["investigation_events"][-1]["new_status"] == "CONFIRMED_FRAUD"

    detail_response = client.get(f"/predictions/{prediction_id}")
    assert detail_response.status_code == 200
    assert detail_response.json()["investigation_status"] == "CONFIRMED_FRAUD"
    manager.dispose()


def test_shap_summary_persistence_omits_raw_feature_values():
    from api.main import _shap_summary
    summary = _shap_summary({
        "shap_available": True,
        "shap_output_space": "probability output scale",
        "shap_base_value": 0.1,
        "shap_reconstructed_score": 0.8,
        "shap_additivity_error": 0.0,
        "top_positive_contributions": [{
            "feature": "V14", "feature_value": -2.5, "shap_value": 0.3,
            "direction": "increases_model_score",
        }],
        "top_negative_contributions": [],
        "interpretation_note": "SHAP is not causal.",
    })
    assert summary is not None
    contributor = summary["top_positive_contributions"][0]
    assert contributor["feature"] == "V14"
    assert contributor["shap_value"] == 0.3
    assert "feature_value" not in contributor
