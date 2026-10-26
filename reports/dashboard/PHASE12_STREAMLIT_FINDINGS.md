# Phase 12 — Streamlit Analyst Dashboard

## Scope delivered

The dashboard is a separate Streamlit frontend and calls the FastAPI service over HTTP. It does not open PostgreSQL connections directly. This keeps API validation, model inference, SHAP generation, and persistence in the backend.

### Dashboard workspaces

1. **Fraud overview** — count of logged predictions, flagged decisions, high/critical risk records, mean model score, p95 inference latency, analyst-confirmed fraud and false-positive review statuses, daily activity, risk distribution, status distribution, and model-version counts.
2. **Transaction analyzer** — all 30 required features (`Time`, `V1`–`V28`, and `Amount`), calls `POST /predict`, and renders the returned score, decision, persistence status, latency, and local SHAP attribution chart.
3. **High-risk transactions** — loads recent HIGH and CRITICAL records, filters by investigation state, loads a selected record and its audit history, and submits investigator status updates with notes.
4. **Historical predictions** — paginated/filterable history by risk and investigation status, plus detail and status update view.
5. **Monitoring dashboard** — health/readiness, model metadata, daily throughput, latency, risk counts, investigation queue and model version counts.

## Architecture

`Streamlit UI -> dashboard/api_client.py -> FastAPI -> model/SHAP service -> PostgreSQL persistence`

The frontend never bypasses the API to query the database. Prediction history and monitoring pages require a configured and reachable database. The transaction analyzer can still call the prediction API when persistence is disabled, if the API is configured to allow it.

## Monitoring API addition

`GET /monitoring/summary` was added to the FastAPI service. It returns aggregate counts, mean score, average/p50/p95 latency, risk/status/model-version counts, latest prediction time and daily prediction groups. It intentionally does not expose raw transaction features.

The current portable implementation reads only the columns required for monitoring and aggregates them in Python. For a high-volume production service, move daily grouping and percentile computation to database-side aggregate queries or an analytics store.

## Run on Windows PowerShell

Install the dashboard dependencies in the existing `.venv`:

```powershell
python -m pip install -r requirements.txt
```

Start PostgreSQL (if using persisted history) and migrate it:

```powershell
docker compose up -d postgres
alembic upgrade head
```

Terminal 1 — API:

```powershell
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Terminal 2 — dashboard:

```powershell
streamlit run dashboard/app.py --server.port 8501
```

Open `http://localhost:8501`. The sidebar API URL defaults to `http://127.0.0.1:8000` and can be changed for a different API location.

## Validation performed

- Existing tests plus new dashboard helper and monitoring-summary tests: **61 passed**.
- Python syntax compilation for `dashboard/`, `api/main.py`, and the persistence repository succeeded.
- A live Streamlit runtime smoke test could not be run in this execution environment: Streamlit is not installed and package installation could not reach the package index. The exact dependency is included in `requirements.txt` for local installation.
- PostgreSQL was not available for live end-to-end validation in this environment. Existing persistence tests use in-memory SQLite; the actual PostgreSQL startup/migration procedure remains a local validation step.

## Interpretation and limitations

- “Flagged by model” counts predictions, not confirmed fraudulent transactions.
- Confirmed fraud and false positive counts reflect analyst status values. They should not be treated as ground truth until reviewed outcomes are trustworthy.
- Input-feature drift is not calculated from prediction logs because the raw 30-feature vector is intentionally not stored in PostgreSQL by default.
- Model scores and 0–100 risk scores are uncalibrated ranking/decision values, not verified probabilities of fraud.
- The high-risk worklist displays at most 200 recent records per risk level per request; the history page uses API pagination.
- Add authentication, authorization, CSRF protections as appropriate, TLS, rate limiting, secrets management and audit retention controls before exposing this development UI to external users.
