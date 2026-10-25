# Phase 7 — Explainable AI with SHAP

## What this phase does

SHAP (SHapley Additive exPlanations) attributes a model output to input features relative to a chosen background dataset. Here, `shap.TreeExplainer` explains the saved XGBoost model's `predict_proba` output on the probability-output scale.

SHAP values explain model behavior; they do not prove causation, validate calibration, or recover the real-world business meaning of anonymized PCA components `V1`–`V28`.

## Reproducibility and sample design

- Background rows: 100
- Explanation rows: 376
- Fraud cases in explanation sample: 75
- Legitimate cases in explanation sample: 301
- Maximum absolute SHAP additivity error: 1.1016658e-07

The interpretability sample includes all fraud cases from the historical test partition plus a deterministic random sample of legitimate cases. This case-enriched design makes minority-class explanations visible, but global mean-absolute SHAP values are **not** prevalence-weighted population estimates. The model and threshold are unchanged by this analysis.

## Global explanations

`global_shap_importance.csv` ranks features by mean absolute SHAP contribution over the interpretability sample. `global_shap_bar.png` visualizes global magnitude. `global_shap_beeswarm.png` shows contribution direction and spread per feature. These are associations with the model's output, not causal claims.

### Top 10 features by mean absolute SHAP

| Rank | Feature | Mean absolute SHAP | Mean SHAP |
|---:|---|---:|---:|
| 1 | V14 | 0.06434913 | 0.06004759 |
| 2 | V4 | 0.03881071 | 0.03069369 |
| 3 | V12 | 0.03057884 | 0.02524602 |
| 4 | V10 | 0.01950955 | 0.01840513 |
| 5 | V17 | 0.01091083 | 0.00964315 |
| 6 | V11 | 0.00355149 | 0.00288345 |
| 7 | V3 | 0.00322269 | 0.00303028 |
| 8 | V21 | 0.00287232 | 0.00239933 |
| 9 | V16 | 0.00285772 | 0.00250251 |
| 10 | V8 | 0.00246617 | -0.00077913 |

## Local explanations

Waterfall plots are generated for a true positive, a false positive, and a false negative where these cases exist under the Phase 6 threshold. The true positive shows why the model scored a correctly detected fraud highly; the false positive is a legitimate transaction incorrectly flagged; the false negative is a fraud case the decision threshold did not flag.

Files: `local_explanations.json`, `local_top_contributions.csv`, and `local/waterfall_*.png`.

## Explanation API

Run `uvicorn api.main:app --reload` and send a `POST /explain` request with all 30 numeric model features (`Time`, `V1`–`V28`, and `Amount`). The response includes the model score, Phase 6 risk tier and decision, plus the largest positive and negative SHAP contributions.

The returned `model_score` is an uncalibrated XGBoost score and must not be interpreted as a verified real-world fraud probability. Avoid exposing the API publicly without authentication, rate limits, request logging policies, and deployment security controls.

## Re-run

```powershell
python -m src.evaluation.shap_explanations
python -m pytest -q
```
