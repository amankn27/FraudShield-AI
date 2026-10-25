# FraudShield AI — Phase 4: Imbalanced learning experiments

## Experimental protocol

All approaches use the same chronological train/validation/test split and Logistic Regression classifier. Models are ranked using validation Average Precision. Validation and test data are never resampled. The selected strategy is evaluated on the test partition once; the fixed 0.5 threshold is not optimized in this phase.

## Split summary

| Partition | Rows | Fraud cases | Legitimate cases | Fraud rate | Time range |
|---|---:|---:|---:|---:|---|
| Train | 170,888 | 360 | 170,528 | 0.2107% | 0–120396 |
| Validation | 56,958 | 57 | 56,901 | 0.1001% | 120397–145248 |
| Test | 56,961 | 75 | 56,886 | 0.1317% | 145249–172792 |

## Why accuracy fails here

The validation partition has 56,901 legitimate rows and 57 fraud rows. A trivial classifier that predicts every row as legitimate would achieve **99.8999% accuracy while catching 0 frauds**. That is why accuracy is reported but not used to select the strategy.

The `class_weight='balanced'` heuristic assigns weights of approximately **0.501** to Class 0 and **237.344** to Class 1 in this training partition. This large difference tells the loss function to pay much more attention to fraud errors.

## Validation comparison

All fixed-threshold metrics below use threshold 0.5. Average Precision (AP) is the selection metric. Precision/recall/F1 and confusion counts depend on this threshold; ROC-AUC and AP evaluate ranking across thresholds.

| Approach | Train rows after strategy | Accuracy | Precision | Recall | F1 | ROC-AUC | Average Precision | FP | FN | Fit seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SMOTE oversampling | 341,056 | 0.9733 | 0.0320 | 0.8772 | 0.0618 | 0.9652 | 0.7735 | 1512 | 7 | 2.7 |
| Class weights (balanced) | 170,888 | 0.9677 | 0.0271 | 0.8947 | 0.0525 | 0.9712 | 0.7720 | 1834 | 6 | 1.9 |
| Baseline (no resampling) | 170,888 | 0.9992 | 0.6429 | 0.4737 | 0.5455 | 0.9678 | 0.6504 | 15 | 30 | 1.5 |
| Random undersampling | 720 | 0.9699 | 0.0290 | 0.8947 | 0.0562 | 0.9675 | 0.4595 | 1707 | 6 | 0.7 |

## Validation-selected approach: test evaluation

Selected by validation AP: **SMOTE oversampling**. After strategy selection, that pipeline was refitted on the combined training + validation data (227,846 rows, including 417 fraud cases). The test partition has 56,961 rows and 75 fraud cases.

| Metric | Test result |
|---|---:|
| Accuracy | 0.98371 |
| Precision | 0.06701 |
| Recall | 0.88000 |
| F1-score | 0.12453 |
| ROC-AUC | 0.98595 |
| Average Precision | 0.77290 |
| True positives | 66 |
| False positives | 919 |
| False negatives | 9 |
| True negatives | 55,967 |

## How confidently should we choose a winner?

The selected strategy's validation AP is 0.77347; the runner-up AP is 0.77203 (gap 0.00144). This difference is small relative to the uncertainty from only 57 validation fraud cases, so treat it as a near-tie rather than proof that SMOTE is generally superior. Class weighting is simpler operationally because it does not synthesize or discard rows. The fixed-threshold results show a separate precision/recall trade-off that must be addressed with threshold selection and, if needed, calibration.

## What each approach does

- **Baseline:** learns from the original skewed training distribution.
- **Class weights:** assigns larger loss weight to the minority class (`class_weight='balanced'`); no rows are created or discarded.
- **Random undersampling:** randomly discards majority-class training cases until classes are balanced; fast, but removes legitimate examples.
- **SMOTE:** synthesizes minority-class training examples using nearest neighbours; can help decision-boundary learning but cannot create new real-world evidence and may synthesize ambiguous points.

## Leakage prevention and interpretation

- Imputation and scaling are fitted on training data only.
- Samplers are placed inside an `imblearn.pipeline.Pipeline`, so they run only during `fit` on the training partition.
- Validation/test rows retain the natural, imbalanced distribution.
- Model selection is based on validation Average Precision, not test results.
- All approaches use a fixed 0.5 threshold for this comparison. Rebalancing can shift predicted probabilities; Phase 6 will study business thresholds, and probability calibration may need separate evaluation.
- Validation has a small number of frauds, so validation scores are noisy. Read the confusion counts and precision-recall curves alongside scalar metrics.

## Generated artifacts

- `validation_experiment_comparison.csv` — validation metric table for all approaches.
- `validation_predictions_all_experiments.csv` — validation probabilities and labels.
- `validation_metric_comparison.png` — metric visualization.
- `validation_precision_recall_curves.png` — validation PR curves.
- `selected_model_test_metrics.csv` — test metrics for the validation-selected approach only.
- `selected_model_test_predictions.csv` — test predictions for that approach.
- `imbalance_experiment_results.json` — machine-readable experiment details.
