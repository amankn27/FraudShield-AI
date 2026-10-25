# FraudShield AI database schema reference

## `prediction_records`

| Column | Type | Purpose |
|---|---|---|
| `id` | UUID | Primary prediction ID |
| `request_id` | VARCHAR(36), unique | Correlates API request/log |
| `created_at` | TIMESTAMPTZ | UTC prediction timestamp |
| `model_version` | VARCHAR(100) | Model artifact version/name |
| `policy_version` | VARCHAR(100) | Decision policy version |
| `model_type` | VARCHAR(100) | Estimator type |
| `feature_count` | INTEGER | Number of prediction inputs |
| `model_score` | DOUBLE PRECISION | Raw, uncalibrated model score |
| `risk_score_0_100` | DOUBLE PRECISION | Display score, model score × 100 |
| `decision_threshold` | DOUBLE PRECISION | Threshold applied to this prediction |
| `risk_level` | VARCHAR(16) | LOW/MEDIUM/HIGH/CRITICAL |
| `predicted_fraud` | BOOLEAN | Binary decision at saved threshold |
| `decision` | VARCHAR(32) | Decision-policy label |
| `prediction_latency_ms` | DOUBLE PRECISION | Inference and optional SHAP latency |
| `shap_available` | BOOLEAN | Whether explanation was returned |
| `shap_summary` | JSONB-compatible JSON | Compact top-contributor summary |
| `investigation_status` | VARCHAR(32) | Current analyst workflow state |
| `investigation_updated_at` | TIMESTAMPTZ nullable | Most recent status update |
| `investigation_note` | TEXT nullable | Most recent note |

## `investigation_events`

| Column | Type | Purpose |
|---|---|---|
| `id` | UUID | Event identifier |
| `prediction_id` | UUID foreign key | Parent prediction |
| `previous_status` | VARCHAR(32) nullable | State before update |
| `new_status` | VARCHAR(32) | State after update |
| `changed_at` | TIMESTAMPTZ | UTC event timestamp |
| `note` | TEXT nullable | Note for this change |

The event table preserves status history rather than overwriting it. The schema deliberately excludes raw input features by default.
