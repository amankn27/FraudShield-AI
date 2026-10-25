# Phase 11 — PostgreSQL Prediction Logging

## Delivered

The API now supports optional database-backed prediction history using SQLAlchemy 2.x. PostgreSQL is the intended runtime database; SQLite is used for isolated automated integration tests so test runs do not need a running database service.

### Tables

`prediction_records` stores:

- A UUID prediction ID and unique request ID
- UTC creation timestamp
- Model version, policy version, model type, and feature count
- Model score, 0–100 display score, decision threshold, risk level, decision, and predicted-fraud flag
- Prediction/inference latency in milliseconds
- Whether SHAP was calculated and a compact JSON summary of top contributions
- Current investigation status, its latest update timestamp, and optional note

`investigation_events` stores an append-only record of investigation state transitions. Creating a prediction creates an initial `NEW` event. Updating status preserves history by storing previous status, new status, timestamp, and optional note.

Raw transaction feature vectors are not saved by default. This reduces unnecessary retention of transaction-level data. If the project later requires complete input auditing, add an explicit privacy-reviewed retention configuration instead of silently persisting all request bodies.

## Status lifecycle

Allowed states:

- `NEW`
- `IN_REVIEW`
- `CONFIRMED_FRAUD`
- `FALSE_POSITIVE`
- `CLOSED`

The update endpoint validates these states through Pydantic and the database schema constraints. Authentication/authorization is not implemented yet; therefore, the endpoint is for local development only until role-based access controls are added.

## API contract

- `POST /predict` — saves the prediction when `DATABASE_URL` is configured and returns `prediction_id` plus `persistence_status`.
- `GET /predictions?limit=50&offset=0` — paginated list with optional `investigation_status` and `risk_level` filters.
- `GET /predictions/{prediction_id}` — prediction details and all investigation events.
- `PATCH /predictions/{prediction_id}/investigation` — update the status and append an event.
- `GET /health` — model-artifact and database availability/readiness.

`latency_ms` is measured for model inference and optional SHAP before database persistence. This keeps database-write time separate from the reported inference latency. If persistence is configured and `DATABASE_REQUIRED=true`, a failed write returns HTTP 503 rather than silently losing the prediction log. With no `DATABASE_URL`, the prediction API remains usable and returns `persistence_status="disabled"`; history routes return 503.

## Validation performed in this environment

- **55 automated tests passed** across the project after adding Phase 11 tests.
- SQLAlchemy repository integration tested with in-memory SQLite.
- Alembic migration applied successfully to a temporary SQLite database and both expected tables were inspected.
- API test exercised prediction persistence, history query, status update, and event history.

A live PostgreSQL server was not running in the execution environment, so a live PostgreSQL connection was not claimed. Use Docker Compose steps in the README to validate the PostgreSQL driver and runtime on the user's machine.

## Local setup

1. Install dependencies: `python -m pip install -r requirements.txt`.
2. Merge the database variables from `.env.example` into your local `.env` (do not overwrite other keys or commit `.env`).
3. Start PostgreSQL: `docker compose up -d postgres`.
4. Apply schema migrations: `alembic upgrade head`.
5. Start API: `uvicorn api.main:app --reload --host 127.0.0.1 --port 8000`.
6. Check `http://127.0.0.1:8000/health` and use `/docs` to submit a prediction.

The compose password is for local development only. Replace it and use secret management before any shared or production deployment.
