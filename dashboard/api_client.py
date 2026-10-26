"""Small synchronous client for the FraudShield FastAPI service."""
from __future__ import annotations

from typing import Any
from urllib.parse import urljoin

import httpx


class APIClientError(RuntimeError):
    """Friendly API error with status code for dashboard display."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class FraudShieldAPIClient:
    def __init__(self, base_url: str, timeout: float = 15.0):
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> Any:
        url = urljoin(self.base_url, path.lstrip("/"))
        try:
            response = httpx.request(method, url, params=params, json=json, timeout=self.timeout)
        except httpx.TimeoutException as exc:
            raise APIClientError(f"API request timed out after {self.timeout:g} seconds.") from exc
        except httpx.HTTPError as exc:
            raise APIClientError(f"Cannot reach the FraudShield API at {self.base_url}. Start FastAPI and retry.") from exc

        try:
            body = response.json()
        except ValueError:
            body = {"detail": response.text[:500]}

        if response.is_error:
            detail = body.get("detail", "Request failed") if isinstance(body, dict) else "Request failed"
            if isinstance(detail, list):
                detail = "; ".join(str(item.get("msg", item)) for item in detail)
            raise APIClientError(f"API returned HTTP {response.status_code}: {detail}", response.status_code)
        return body

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def model_info(self) -> dict[str, Any]:
        return self._request("GET", "/model-info")

    def predict(self, payload: dict[str, float], *, include_shap: bool = True, top_k: int = 8) -> dict[str, Any]:
        return self._request(
            "POST", "/predict", params={"include_shap": include_shap, "top_k": top_k}, json=payload
        )

    def list_predictions(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        risk_level: str | None = None,
        investigation_status: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if risk_level and risk_level != "ALL":
            params["risk_level"] = risk_level
        if investigation_status and investigation_status != "ALL":
            params["investigation_status"] = investigation_status
        return self._request("GET", "/predictions", params=params)

    def prediction_detail(self, prediction_id: str) -> dict[str, Any]:
        return self._request("GET", f"/predictions/{prediction_id}")

    def update_investigation(self, prediction_id: str, status: str, note: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"status": status}
        if note and note.strip():
            payload["note"] = note.strip()
        return self._request("PATCH", f"/predictions/{prediction_id}/investigation", json=payload)

    def monitoring_summary(self) -> dict[str, Any]:
        return self._request("GET", "/monitoring/summary")
