# FraudShield AI — Phase 2 Data Findings

**Dataset analyzed:** `creditcard.csv` from the user's uploaded `archive.zip`  
**Analysis generated:** 25 October 2026  
**Raw data policy:** This report summarizes the raw dataset. The raw CSV was not changed.

## Executive summary

- The dataset has **284,807 rows and 31 columns**.
- The schema matches the expected columns: `Time`, `V1`–`V28`, `Amount`, and `Class`.
- All 31 columns are numeric: 30 `float64` columns and integer target column `Class`.
- There are **0 missing cells**.
- The dataset contains **1,081 exact duplicate extra copies**, occurring in **773 repeated row groups**. In total, 1,854 rows belong to groups containing repeated exact rows. No identical input-feature group has conflicting target labels.
- The fraud class is only **0.1727%** of records. A classifier that predicts every row as legitimate would still obtain **99.8273% accuracy**, while catching **zero fraud cases**.
- `Amount` is strongly right-skewed. The median amount is lower for fraud cases than for legitimate cases, while the mean is higher; a single amount threshold will therefore not cleanly separate the classes.
- The largest absolute Pearson correlations with `Class` are `V17`, `V14`, `V12`, and `V10`. These are descriptive relationships, not causal explanations or guaranteed model feature importance.

## 1. Dataset structure

| Check | Result |
|---|---:|
| Rows | 284,807 |
| Columns | 31 |
| Expected columns missing | 0 |
| Unexpected columns | 0 |
| Numeric columns | 31 |
| Missing cells | 0 |
| Time minimum / maximum | 0 / 172,792 seconds |
| Amount minimum / maximum | 0 / 25,691.16 |
| Zero-amount records | 1,825 |
| Negative-amount records | 0 |
| Time values sorted non-decreasingly in file | Yes |

`Class` is the target: `0` means legitimate and `1` means fraud. `Time` is elapsed relative time in seconds, not a calendar timestamp. `V1`–`V28` are anonymized PCA components; their original business meanings are not available in this dataset.

## 2. Class imbalance

| Class | Meaning | Count | Percentage |
|---:|---|---:|---:|
| 0 | Legitimate | 284,315 | 99.827251% |
| 1 | Fraud | 492 | 0.172749% |
| **Total** | | **284,807** | **100%** |

There are approximately **578 legitimate transactions for every fraud transaction**. Accuracy is not an adequate standalone metric. In the next phase, report at least precision, recall, F1, PR-AUC / Average Precision, ROC-AUC, and the confusion matrix.

## 3. Missing values and data types

Pandas detected **no missing values** in any column. All columns have numeric data types. This means no missing-value imputation is required for this raw file at this stage, but it does not mean the dataset has no other quality issues; the duplicate review below remains important.

## 4. Duplicate analysis

| Duplicate check | Count |
|---|---:|
| Exact duplicate extra copies (`df.duplicated().sum()`) | 1,081 |
| Groups of identical full rows | 773 |
| Rows belonging to those repeated-row groups | 1,854 |
| Duplicate extra copies by target: class 0 / class 1 | 1,062 / 19 |
| Identical feature groups with conflicting labels | 0 |

If all exact duplicates were removed, the dataset would contain **283,726 rows**, including **283,253 legitimate rows and 473 fraud rows**. That would remove 19 duplicate fraud records, so the choice should be documented rather than made silently.

**Recommendation for Phase 3:** keep the uploaded raw dataset immutable. For evaluation, do not allow identical records to leak across training and validation/test partitions. A chronological split based on `Time` is a sensible primary evaluation for this dataset, with boundaries chosen by time value rather than blindly splitting row indices. Also run a deduplicated-data sensitivity experiment and report the change in class counts and metrics. Do not remove duplicates automatically without recording the decision.

## 5. Amount distribution by class

| Statistic | Legitimate (`Class=0`) | Fraud (`Class=1`) |
|---|---:|---:|
| Count | 284,315 | 492 |
| Mean | 88.29 | 122.21 |
| Median | 22.00 | 9.25 |
| 25th percentile | 5.65 | 1.00 |
| 75th percentile | 77.05 | 105.89 |
| Maximum | 25,691.16 | 2,125.87 |

The mean is larger than the median in both classes, indicating right-skewed amount distributions. The fraud median is lower than the legitimate median, even though the fraud mean is higher; that is consistent with a distribution that contains many small fraud amounts alongside a smaller number of larger amounts. Amount alone does not give a simple separation rule. For plots, `log1p(Amount)` improves visibility of lower amounts while preserving zero values.

## 6. Time distribution by class

| Statistic | Legitimate (`Class=0`) | Fraud (`Class=1`) |
|---|---:|---:|
| Mean time (seconds) | 94,838.20 | 80,746.81 |
| Median time (seconds) | 84,711.00 | 75,568.50 |
| Minimum | 0 | 406 |
| Maximum | 172,792 | 170,348 |

The fraud and legitimate distributions overlap substantially. The Pearson correlation between `Time` and `Class` is approximately **-0.0123**, which is close to zero. `Time` may still carry nonlinear or temporal-pattern information, but this univariate correlation is weak and is not enough to conclude that Time is useless.

## 7. Correlation analysis

### Highest absolute correlations with `Class`

| Feature | Pearson correlation with `Class` |
|---|---:|
| `V17` | -0.3265 |
| `V14` | -0.3025 |
| `V12` | -0.2606 |
| `V10` | -0.2169 |
| `V16` | -0.1965 |
| `V3` | -0.1930 |
| `V7` | -0.1873 |
| `V11` | +0.1549 |
| `V4` | +0.1334 |
| `V18` | -0.1115 |
| `Amount` | +0.0056 |
| `Time` | -0.0123 |

The signs indicate direction of *linear association* with the binary label, not causation. Because `V1`–`V28` are anonymized PCA components, these correlations cannot be translated directly into business explanations such as location, merchant, or customer intent. Also, univariate correlation does not capture all nonlinear interactions used by tree-based models.

### Strongest correlations among input columns

| Feature pair | Absolute Pearson correlation |
|---|---:|
| `V2` – `Amount` | 0.5314 |
| `V3` – `Time` | 0.4196 |
| `V7` – `Amount` | 0.3973 |
| `V5` – `Amount` | 0.3864 |
| `V20` – `Amount` | 0.3394 |

No pair of input columns has an absolute Pearson correlation above 0.9 in this file. This does not prove the absence of nonlinear dependencies.

## 8. Files produced by the EDA run

- `eda_summary.json` — machine-readable executive summary.
- `class_distribution.csv` — target counts and percentages.
- `missing_values.csv` — missing count and percentage per column.
- `duplicate_summary.csv` — duplicate row and duplicate feature-vector checks.
- `descriptive_statistics.csv` — numerical summary statistics.
- `column_types.csv` — column data types.
- `correlation_matrix.csv` — numeric correlation matrix.
- `correlations_with_class.csv` — correlations ranked by absolute strength.
- `01_class_imbalance.png` — class count chart.
- `02_missing_values.png` — missing-value check visualization.
- `03_amount_distribution_log1p.png` and `04_amount_boxplot_log1p.png` — amount distributions.
- `05_time_distribution.png` — relative time distributions.
- `06_correlation_heatmap.png` — full numeric correlation heatmap.
- `07_top_feature_distributions.png` — distributions for the six features most correlated with `Class`.
- `08_correlations_with_class.png` — all feature-wise correlations with `Class`.

## 9. Modeling implications for Phase 3

1. Use a time-aware evaluation as the primary realism check, because the `Time` column is ordered in the source CSV. Choose boundaries by `Time` value so rows with the same timestamp do not land on opposite sides of a boundary.
2. Keep a final test partition untouched until model selection and threshold decisions are complete.
3. Because fraud is extremely rare, verify the number of positive cases in every split. A validation or test set with too few fraud cases will make recall and PR-AUC uncertain.
4. Start with a Logistic Regression baseline. Scale the numerical inputs using parameters fitted only on the training data; tree-based models such as XGBoost generally do not require standardization.
5. Compare class weighting and resampling only within training folds. Never apply SMOTE or other resampling to the full dataset before splitting.
6. Preserve this report and compare future models against a documented baseline; do not optimize accuracy alone.

**Phase 2 result:** schema, missingness, imbalance, duplicates, class-specific distributions, and correlations have been inspected. The next step is to design and implement a leakage-aware evaluation split before fitting the first model.
