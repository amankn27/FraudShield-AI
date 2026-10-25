from fastapi.testclient import TestClient

from api import main

client = TestClient(main.app)


def _valid_payload():
    payload = {"Time": 12345.0, "Amount": 149.5}
    payload.update({f"V{i}": 0.01 * i for i in range(1, 29)})
    return payload


def test_health_endpoint_is_lightweight():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_explain_endpoint_validates_missing_features_before_service(monkeypatch):
    payload = _valid_payload()
    payload.pop("V28")
    response = client.post("/explain", json=payload)
    assert response.status_code == 422


def test_explain_endpoint_returns_service_payload(monkeypatch):
    class FakeService:
        def explain(self, payload, top_k=5):
            assert len(payload) == 30
            return {"model_score": 0.2, "risk_level": "MEDIUM", "top_positive_contributions": []}

    monkeypatch.setattr(main, "get_explanation_service", lambda: FakeService())
    response = client.post("/explain?top_k=3", json=_valid_payload())
    assert response.status_code == 200
    assert response.json()["model_score"] == 0.2


def test_explain_endpoint_rejects_non_finite_feature(monkeypatch):
    payload = _valid_payload()
    payload["Amount"] = "NaN"
    response = client.post("/explain", json=payload)
    assert response.status_code == 422
