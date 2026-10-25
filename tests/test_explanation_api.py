from fastapi.testclient import TestClient

from api import main

client = TestClient(main.app)


def valid_payload():
    payload = {"Time": 12345.0, "Amount": 149.5}
    payload.update({f"V{i}": 0.01 * i for i in range(1, 29)})
    return payload


def fake_prediction_result(include_shap=True):
    result = {
        "model_version": "xgboost_best",
        "policy_version": "1.0.0",
        "model_type": "XGBClassifier",
        "feature_count": 30,
        "model_score": 0.82,
        "score_interpretation": "Uncalibrated XGBoost score.",
        "risk_score_0_100": 82.0,
        "decision_threshold": 0.78530025,
        "risk_level": "CRITICAL",
        "predicted_fraud": True,
        "decision": "FLAG_FOR_REVIEW",
    }
    if include_shap:
        result.update({
            "shap_available": True,
            "shap_output_space": "model probability output scale",
            "shap_base_value": 0.01,
            "shap_reconstructed_score": 0.82,
            "shap_additivity_error": 0.0,
            "top_positive_contributions": [{
                "feature": "V14", "feature_value": -2.0,
                "shap_value": 0.4, "direction": "increases_model_score",
            }],
            "top_negative_contributions": [],
            "interpretation_note": "SHAP is not causal.",
        })
    return result


def test_health_endpoint_is_lightweight_and_reports_artifact_state(monkeypatch):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["service"] == "fraudshield-ai-api"
    assert response.json()["missing_artifacts"] == []
    assert isinstance(response.json()["ready"], bool)
    assert response.json()["ready"] == response.json()["model_loaded"]
    assert response.json()["status"] == ("ok" if response.json()["ready"] else "degraded")


def test_predict_endpoint_returns_prediction_and_shap(monkeypatch):
    class FakeService:
        def explain(self, payload, top_k=5):
            assert len(payload) == 30
            assert top_k == 3
            return fake_prediction_result(include_shap=True)

    monkeypatch.setattr(main, "get_explanation_service", lambda: FakeService())
    response = client.post("/predict?top_k=3", json=valid_payload())
    assert response.status_code == 200
    data = response.json()
    assert data["model_score"] == 0.82
    assert data["predicted_fraud"] is True
    assert data["shap_available"] is True
    assert data["top_positive_contributions"][0]["feature"] == "V14"
    assert data["request_id"]
    assert data["latency_ms"] >= 0


def test_predict_can_skip_shap_for_lower_latency(monkeypatch):
    class FakeService:
        def predict(self, payload):
            assert len(payload) == 30
            return fake_prediction_result(include_shap=False)

    monkeypatch.setattr(main, "get_explanation_service", lambda: FakeService())
    response = client.post("/predict?include_shap=false", json=valid_payload())
    assert response.status_code == 200
    data = response.json()
    assert data["model_score"] == 0.82
    assert data["shap_available"] is False
    assert data["top_positive_contributions"] == []


def test_explain_legacy_alias_returns_shap(monkeypatch):
    class FakeService:
        def explain(self, payload, top_k=5):
            return fake_prediction_result(include_shap=True)

    monkeypatch.setattr(main, "get_explanation_service", lambda: FakeService())
    response = client.post("/explain?top_k=3", json=valid_payload())
    assert response.status_code == 200
    assert response.json()["shap_available"] is True


def test_model_info_endpoint_returns_schema_and_policy(monkeypatch):
    class FakeService:
        def model_info(self):
            return {
                "model_version": "xgboost_best",
                "model_type": "XGBClassifier",
                "policy_version": "1.0.0",
                "feature_count": 30,
                "feature_names": ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"],
                "decision_threshold": 0.78530025,
                "risk_bands": {},
                "shap_available": True,
                "score_semantics": "uncalibrated score",
                "libraries": {"xgboost": "3.0", "scikit_learn": "1.7", "shap": "0.50"},
            }

    monkeypatch.setattr(main, "get_explanation_service", lambda: FakeService())
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["feature_count"] == 30
    assert data["decision_threshold"] == 0.78530025
    assert data["service"] == "fraudshield-ai-api"


def test_missing_feature_is_rejected_before_inference(monkeypatch):
    payload = valid_payload()
    payload.pop("V28")
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_extra_feature_is_rejected():
    payload = valid_payload()
    payload["Class"] = 1
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_non_finite_feature_is_rejected():
    payload = valid_payload()
    payload["Amount"] = "NaN"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_negative_amount_is_rejected():
    payload = valid_payload()
    payload["Amount"] = -1
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_top_k_is_bounded():
    response = client.post("/predict?top_k=31", json=valid_payload())
    assert response.status_code == 422


def test_missing_artifacts_return_service_unavailable(monkeypatch):
    def fail_load():
        raise FileNotFoundError("internal path must not be revealed")

    monkeypatch.setattr(main, "get_explanation_service", fail_load)
    response = client.get("/model-info")
    assert response.status_code == 503
    assert "internal path" not in response.text
