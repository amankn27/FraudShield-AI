# Phase 10 — FastAPI Prediction Service

## Outcome

The XGBoost model is exposed through a typed FastAPI service. The API uses the Phase 5 model artifact (`models/xgboost_best.joblib`), the Phase 6 decision policy (`models/decision_policy.json`), and the Phase 7 SHAP background (`models/shap_background.csv`). The model/policy/background service is cached once per process and normally preloaded during FastAPI lifespan startup. SHAP's `TreeExplainer` is initialized only when an explanation is requested.

## API routes

| Method and route | Purpose | Expected success status |
|---|---|---:|
| `GET /health` | Liveness, artifact availability, and model-loaded status | 200 |
| `GET /model-info` | Model type/version, ordered feature names, policy threshold, risk bands, library versions | 200 |
| `POST /predict` | Validate and score one complete transaction; return SHAP by default | 200 |
| `POST /predict?include_shap=false` | Score only, avoiding SHAP computation | 200 |
| `POST /explain` | Backwards-compatible deprecated alias for prediction with SHAP | 200 |
| `GET /docs` | Interactive OpenAPI/Swagger UI | 200 |

## Input validation

`api/schemas.py` defines a Pydantic request schema with `Time`, `V1`–`V28`, and `Amount`. It rejects extra fields such as `Class`, missing fields, non-finite numeric inputs, negative `Time`/`Amount`, and `top_k` values outside 1–30. Values are reordered to the exact feature order stored on the trained model before inference.

## Prediction pipeline

1. Validate request data via Pydantic.
2. Retrieve the cached model service (or return 503 if artifacts cannot be loaded).
3. Convert the input to a one-row DataFrame ordered to match training.
4. Call `XGBClassifier.predict_proba` and obtain the class-1 score.
5. Apply the saved Phase 6 threshold and risk-band policy.
6. If requested, calculate SHAP contributions against the saved background and return top positive/negative features.
7. Return model/policy versions, request ID, and measured latency.

The service returns `model_score` and `risk_score_0_100 = 100 * model_score`. The score is uncalibrated and must not be described as a calibrated real-world fraud probability. SHAP values describe model attribution relative to its background and do not prove causation or interpret anonymized PCA features semantically.

## Error behavior

- Invalid request bodies and feature values: HTTP 422.
- Missing/unreadable model, policy, or explanation artifacts: HTTP 503.
- Unhandled inference or response-construction failures: HTTP 500 with a generic client message; details are written to server logs rather than exposed to clients.
- `/health` remains lightweight and reports artifact availability, whether model service is loaded, and readiness. `status` is `ok` only when the service is ready; missing files or failed loading yields `degraded`. During normal Uvicorn startup, the lifespan handler preloads model artifacts; the SHAP explainer remains lazy.

## Smoke-test result

Executed `TestClient` against the actual artifacts in the current workspace:

- `/health`: HTTP 200, required artifacts available.
- `/model-info`: HTTP 200, `XGBClassifier`, 30 input features, policy threshold `0.78530025`.
- `/predict?include_shap=true&top_k=5`: HTTP 200, SHAP contributions returned, additivity error approximately `-1.51e-08` for the sampled record.
- `/predict?include_shap=false`: HTTP 200, same model score and policy result without local SHAP attribution.
- `/predict` returns HTTP 422 for a missing feature, an extra `Class` field, non-finite input, negative amount, and invalid `top_k`.
- Automated tests: **48 passed** after Phase 10 changes.

The sampled API request/response are in `reports/api/example_request.json` and `reports/api/example_response.json`. These examples use one row of the public credit-card dataset and are for API demonstration only.

## Run locally on Windows

From the repository root with the project environment activated:

```powershell
python -m pip install -r requirements.txt
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs`. Send the full 30-feature numeric request to `POST /predict`. `Class` is the evaluation label and must never be sent as an input feature.

Run tests in another terminal:

```powershell
python -m pytest -q
```

## Operational limitations

This is a portfolio/development service. It does not yet have authentication/authorization, rate limiting, database logging, deployment TLS, request quotas, or a calibrated probability model. Do not expose the API publicly until appropriate security and operational controls are implemented. Because the historical test period has been examined in earlier phases, API smoke tests validate functionality rather than providing a fresh, independent model-performance claim.
