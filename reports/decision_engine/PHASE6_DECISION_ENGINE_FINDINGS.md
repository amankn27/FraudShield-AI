# Phase 6 — Fraud Decision Engine Findings

## Objective

Convert XGBoost scores into a decision policy with an explicit cost assumption and ordinal risk tiers.

## Probability vs classification

`fraud_score` is the XGBoost `predict_proba[:, 1]` output. Because the model was trained with class weighting and has not been calibrated, this score must not be interpreted as a verified real-world fraud probability. `risk_score_0_100 = 100 × fraud_score` is a convenient score scale, not a probability percentage.

A **score** is continuous; a **classification** is produced by applying a threshold. Changing the threshold changes precision/recall without retraining the model.

## Policy assumptions

- False-positive cost: **1 relative units**.
- False-negative cost: **100 relative units**.
- These are illustrative assumptions, not currency, chargeback amounts, investigation costs, or measured financial losses.
- The threshold is selected by minimum assumed cost on validation data only. The test set is not used to choose the threshold.

## Threshold selection results

| Partition / policy | Threshold | Precision | Recall | F1 | FP | FN | Assumed total cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation, default | 0.5000 | 16.31% | 80.70% | 0.2714 | 236 | 11 | 1336 |
| Validation, selected | 0.785300 | 60.53% | 80.70% | 0.6917 | 30 | 11 | 1130 |
| Test, default | 0.5000 | 26.32% | 80.00% | 0.3960 | 168 | 15 | 1668 |
| Test, validation-selected | 0.785300 | 76.32% | 77.33% | 0.7682 | 18 | 17 | 1718 |

### Important result

The selected threshold reduced validation cost from 1336 to 1130 relative units. On the later test period, its cost was 1718 versus 1668 at threshold 0.5. **It did not improve held-out test cost under the same assumed ratio.** This is an important generalization warning: the validation set has only 57 fraud cases, and score/rate patterns change over time.

## Risk bands

Risk tiers are relative model-score bands derived from validation cutoffs; they are not calibrated probability intervals.

| Risk level | Rule | Interpretation in this prototype |
|---|---|---|
| LOW | score < 0.027193 | Below the validation score cutoff needed to reach 95% recall |
| MEDIUM | 0.027193 ≤ score < 0.293915 | Between the 95%- and 85%-recall score cutoffs |
| HIGH | 0.293915 ≤ score < 0.785300 | Elevated score, below cost-selected classification threshold |
| CRITICAL | score ≥ 0.785300 | Meets the cost-selected threshold and is flagged for review |

These labels support prioritization in a demo. Actual financial workflows should set band boundaries and actions with a business owner, analyst-capacity constraints, and independently validated costs.

## Cost sensitivity

See `cost_sensitivity.csv` for the threshold selected on validation for several assumed false-negative costs and its test-period outcome. Different cost ratios can produce materially different alert volumes.

## Limitations and next steps

1. The assumed cost ratio is illustrative; the dataset does not provide operational bank costs.
2. The scores are not calibrated. Assess calibration using temporally appropriate data before displaying them as literal probabilities.
3. The validation block has few fraud cases, so threshold choice is noisy.
4. The validation-selected threshold increased the test cost under this assumed cost ratio; perform broader temporal validation before adopting a policy.
5. Continue to monitor precision, recall, alert volume, and delayed ground-truth labels after deployment.

## Generated artifacts

- `validation_threshold_sweep.csv`
- `validation_policy_comparison.csv`
- `test_policy_comparison.csv`
- `cost_sensitivity.csv`
- `validation_decisions.csv`
- `test_decisions.csv`
- `confusion_matrix_test_cost_threshold.png`
- `validation_cost_vs_threshold.png`
- `test_policy_comparison.png`
- `risk_band_distribution.png`
