# FraudShield AI — Phase 3 Baseline Findings

## Scope

This report records the first Logistic Regression baseline trained on the bundled project's expected `creditcard.csv` dataset. It is a benchmark, not a tuned or production-ready fraud model.

## Split design

The source rows were stably sorted by `Time` and split chronologically. The nominal target proportions were 60% train, 20% validation, and 20% test. Split boundaries were moved forward where needed so that equal `Time` values were kept in the same partition. Since exact duplicate rows have the same `Time`, exact duplicate copies do not straddle partitions.

| Partition | Rows | Fraud | Legitimate | Fraud rate | Time range |
|---|---:|---:|---:|---:|---|
| Train | 170,888 | 360 | 170,528 | 0.2107% | 0–120,396 |
| Validation | 56,958 | 57 | 56,901 | 0.1001% | 120,397–145,248 |
| Test | 56,961 | 75 | 56,886 | 0.1317% | 145,249–172,792 |

The prevalence changes across periods. This is a meaningful property of the chronological test, and metrics must be interpreted in that context. This dataset spans a short historical window, so it cannot guarantee performance on future production data.

## Preprocessing and model

- Target: `Class` (`0` legitimate, `1` fraud).
- Input features: all columns other than `Class`, including `Time`, `Amount`, and `V1`–`V28`.
- Pipeline: median imputation → `StandardScaler` → `LogisticRegression`.
- Logistic Regression configuration: `max_iter=1000`, `solver=lbfgs`, no class weighting, threshold `0.5`.
- Imputer and scaler are inside the pipeline and are fit only when `model.fit(X_train, y_train)` runs. Validation/test values do not fit preprocessing.
- No resampling, threshold tuning, or hyperparameter search was performed in this phase.

## Metrics at threshold 0.5

| Metric | Train | Validation | Test |
|---|---:|---:|---:|
| Accuracy | 99.8935% | 99.9210% | 99.9175% |
| Precision | 0.8321 | 0.6429 | 0.7500 |
| Recall | 0.6194 | 0.4737 | 0.5600 |
| F1 | 0.7102 | 0.5455 | 0.6412 |
| ROC-AUC | 0.9796 | 0.9678 | 0.9731 |
| Average Precision (AP) | 0.7828 | 0.6504 | 0.6926 |
| Trapezoidal PR-AUC | 0.7826 | 0.6487 | 0.6910 |

Average Precision is the primary precision-recall ranking metric in this report. Trapezoidal PR-AUC is also reported because the two quantities are not always identical.

## Test confusion matrix

Rows are actual labels, columns are predicted labels, ordered as `[legitimate (0), fraud (1)]`.

| | Predicted legitimate | Predicted fraud |
|---|---:|---:|
| Actual legitimate | 56,872 (TN) | 14 (FP) |
| Actual fraud | 33 (FN) | 42 (TP) |

At threshold 0.5, the model flags 56 of 56,961 test transactions; 42 are fraud and 14 are false alarms. It catches 42 of the 75 fraud transactions and misses 33. The test accuracy is very high mainly because legitimate transactions dominate the data, so accuracy is not a sufficient measure of usefulness.

## Leakage prevention

1. The data is partitioned before fitting learned preprocessing.
2. `Class` is excluded from model inputs.
3. Imputation and scaling are fit only on the training partition through an sklearn `Pipeline`.
4. Resampling is not applied to validation or test.
5. The chronological split keeps equal timestamps in one partition, which also keeps exact duplicate rows together for this dataset.
6. The initial model and 0.5 threshold were fixed without using test results for tuning. Use validation data for choices in later phases; reserve test for final comparisons.

## Interpretation and limitations

- ROC-AUC indicates good ranking separation in this time slice, but does not determine a useful operating threshold by itself.
- Recall of 0.56 means 44% of test fraud cases were missed at threshold 0.5. Threshold and imbalance handling are priority follow-up experiments.
- Precision of 0.75 means one quarter of the transactions flagged as fraud were false positives in this test partition.
- Fraud prevalence differs across partitions, which affects precision and AP. Report the split distribution with the metrics.
- The baseline retains the original rows, including duplicate copies, but equal timestamps do not cross split boundaries. A deduplicated sensitivity experiment should be documented separately before claiming robustness to duplicate records.
- `V1`–`V28` are anonymized PCA components, so their model influence cannot be translated directly into merchant/location/business causes.
- This is historical offline evaluation, not live validation or a claim of financial savings.

## Generated artifacts

- `baseline_metrics.json` — complete structured results.
- `baseline_metrics.csv` — tabular metrics.
- `split_summary.csv` — data split counts and time ranges.
- `validation_test_predictions.csv` — per-row labels and probabilities for analysis.
- `confusion_matrix_*.png` — confusion matrices at threshold 0.5.
- `roc_curve_*.png` and `precision_recall_curve_*.png` — validation/test ranking plots.
- `models/logistic_regression_baseline.joblib` — fitted end-to-end preprocessing/model pipeline (software-version compatibility should be considered when loading it elsewhere).
