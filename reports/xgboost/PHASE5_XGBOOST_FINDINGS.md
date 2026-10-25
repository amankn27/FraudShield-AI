# FraudShield AI — Phase 5: XGBoost findings

## Objective

Replace a linear-only baseline with gradient-boosted trees, tune hyperparameters using `RandomizedSearchCV`, preserve the chronological evaluation strategy, handle the rare fraud class via `scale_pos_weight`, persist the model, and compare its test results to the existing baselines.

## What gradient boosting does

A boosted-tree ensemble builds shallow trees sequentially. Each new tree improves the ensemble's objective using gradient information from the current model. XGBoost adds regularization and sampling controls to manage model complexity. Unlike Logistic Regression, trees can learn nonlinear relationships and feature interactions without standardizing the input columns.

## Data split and leakage controls

The source dataset was split chronologically using the existing Phase 3 function, preserving equal `Time` groups:

| Partition | Rows | Fraud cases | Fraud rate |
|---|---:|---:|---:|
| Train | 170,888 | 360 | 0.2107% |
| Validation | 56,958 | 57 | 0.1001% |
| Test | 56,961 | 75 | 0.1317% |

Hyperparameter search saw only the training partition. Three expanding-window CV folds were built inside training; each fold trains on earlier rows and validates on a later contiguous block. Equal `Time` groups are not split across a fold boundary. `RandomizedSearchCV` scored each candidate using Average Precision. The top five CV-ranked candidates (four were searched in this run) were then fit on training and compared by validation Average Precision because chronological fold scores and the later validation block showed distribution shift. The selected candidate was refit on train + validation, and XGBoost test results were generated only after selection. The same test period has already been reported in the earlier learning phases; while it was not used to choose XGBoost parameters, it has been repeatedly observed across the overall project. For a strictly independent final performance claim after all future iteration, define a new later-time holdout or use nested temporal evaluation.

Validation participates in model selection, so it is not a final unbiased evaluation. The test partition is reserved for final reporting. The public dataset is historical and anonymized; results do not establish production-bank performance.

## Search configuration

- XGBoost version used for this run: 3.1.3
- Search iterations: 4 randomized parameter configurations
- Cross-validation: 3 expanding-window chronological folds
- Primary search score: `average_precision`
- `tree_method="hist"`, `objective="binary:logistic"`
- XGBoost CPU workers per fit: 2

Searched parameters included `n_estimators`, `max_depth`, `learning_rate`, `min_child_weight`, `subsample`, `colsample_bytree`, `gamma`, `reg_alpha`, `reg_lambda`, and `scale_pos_weight`.

The training ratio of legitimate to fraud cases was 170,528 / 360 = **473.69**. This common ratio was included as a candidate, not hard-coded as the best choice. The selected candidate uses `scale_pos_weight=100`, which was supported by validation Average Precision rather than blindly using the full ratio.

## Selected hyperparameters

| Parameter | Selected value | Practical interpretation |
|---|---:|---|
| `n_estimators` | 142 | Number of boosting trees |
| `max_depth` | 2 | Limits individual tree complexity |
| `learning_rate` | 0.03051 | Step size for each new tree |
| `min_child_weight` | 5.9775 | Minimum summed instance weight required in a child |
| `subsample` | 0.8604 | Fraction of training rows sampled per tree |
| `colsample_bytree` | 0.7498 | Fraction of features sampled per tree |
| `gamma` | 1.0 | Minimum loss reduction needed to add a split |
| `reg_alpha` | 0.01 | L1 regularization |
| `reg_lambda` | 5.0 | L2 regularization |
| `scale_pos_weight` | 100 | Additional positive-class weighting |

### Important selection observation

The highest mean CV Average Precision candidate had CV AP **0.80385** but mean training AP of approximately **0.99797**; its validation AP was only **0.60541**. This gap suggested that simply choosing the highest CV score did not generalize well to the next chronological period. Among the four searched candidates, the selected candidate was CV-ranked third, with CV AP **0.75587** and validation AP **0.77951**. This is why the implementation keeps a chronological validation partition for model selection instead of assuming that the highest CV score always wins.

Because only four parameter configurations were searched, this is a compact first tuning pass, not an exhaustive search. Increase `--n-iter` to explore more configurations after inspecting runtime and stability.

## Final held-out test metrics

The selected model was refit on train + validation and evaluated on the test partition at the default threshold 0.5:

| Metric | Result |
|---|---:|
| Accuracy | 99.6787% |
| Precision | 26.32% |
| Recall | 80.00% |
| F1 | 0.3960 |
| ROC-AUC | 0.98568 |
| Average Precision | 0.78718 |
| Trapezoidal PR-AUC | 0.78683 |

Confusion matrix: **TN 56,718; FP 168; FN 15; TP 60**.

At this threshold, 60 of 75 fraud cases were detected, but 168 legitimate transactions were also flagged. Thus the model emphasizes recall and ranking quality; the fixed 0.5 threshold is not a business-ready decision rule. Phase 6 will choose a threshold on validation data according to a clearly stated objective or cost function. Scores produced by a class-weighted model are not guaranteed to be calibrated real-world fraud probabilities.

## Test-set comparison

All listed metrics are at threshold 0.5. Historical rows preserve the results recorded during prior phases; the Phase 3 original baseline used training-only data while XGBoost and the refit Logistic Regression used train + validation. Use the row-specific training protocol when interpreting model differences.

| Model | Precision | Recall | F1 | ROC-AUC | Average Precision | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression baseline (Phase 3, train only) | 75.00% | 56.00% | 0.6412 | 0.97309 | 0.69258 | 14 | 33 | 42 |
| Phase 4 Logistic Regression + SMOTE | 6.70% | 88.00% | 0.1245 | 0.98595 | 0.77290 | 919 | 9 | 66 |
| Logistic Regression refit on train + validation | 89.36% | 56.00% | 0.6885 | 0.97620 | 0.74040 | 5 | 33 | 42 |
| XGBoost tuned, refit on train + validation | 26.32% | 80.00% | 0.3960 | 0.98568 | 0.78718 | 168 | 15 | 60 |

XGBoost gives the best Average Precision in this set of recorded runs, and it detects more fraud cases than the baseline at threshold 0.5. It does **not** dominate all threshold-dependent metrics: its fixed-threshold precision and F1 are lower than the refit Logistic Regression model. Compared with SMOTE, XGBoost has slightly higher Average Precision and far fewer false positives at this threshold, while missing six more fraud cases. These are trade-offs, not a universal winner claim.

## Feature importance

The leading gain-based features for this run are `V17`, `V14`, `V11`, `V10`, `V12`, and `V4`. These are anonymized PCA components. Importance indicates the model's use of features, not causation or a direct business explanation.

## Model persistence

The fitted model is saved in two formats:

- `models/xgboost_best.joblib`: the scikit-learn estimator wrapper, convenient for loading in this project.
- `models/xgboost_best.json`: XGBoost's native model format.

The experiment also saves randomized-search CV results, validation candidate comparisons, predictions, metrics, feature importance, confusion matrices, precision-recall curves, and the test comparison chart under `reports/xgboost/`.

## Reproduce

From the project root, after installing `requirements.txt` and placing the dataset at `data/raw/creditcard.csv`:

```powershell
python -m src.models.xgboost_model
python -m pytest -q
```

Increase the search budget on a machine with enough CPU time:

```powershell
python -m src.models.xgboost_model --n-iter 10 --n-jobs 2
```

Do not tune the threshold using test labels. The selected threshold and cost function belong in Phase 6.
