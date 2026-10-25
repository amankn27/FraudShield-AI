# FraudShield AI

An end-to-end learning project for imbalanced financial fraud detection. The planned system will compare supervised ML and deep-learning approaches, explain predictions, expose a prediction API, and add monitoring and reproducible experiments.

> **Status:** Phase 5 complete — Logistic Regression baseline, imbalance experiments, and a tuned XGBoost model have been evaluated. See the phase reports for measured results and limitations.

## Dataset

Download `creditcard.csv` from the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at `data/raw/creditcard.csv`.

The dataset contains anonymized PCA components (`V1`–`V28`), `Time`, `Amount`, and binary target `Class`. The raw dataset is not included in this repository. Check the dataset page for its current terms of use. Do not commit the CSV.

## Requirements

- Windows 10/11
- Python 3.11 recommended for broad compatibility with the planned ML stack
- Git
- PowerShell

## Setup (PowerShell)

```powershell
py -3.11 --version
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

If PowerShell blocks activation, either use `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` for the current terminal session and activate again, or run the environment's Python directly: `\.venv\Scripts\python.exe`.

## Validate the dataset

```powershell
python -m src.data.validate_dataset
```

Or pass a different CSV path:

```powershell
python -m src.data.validate_dataset --data-path "D:\path\to\creditcard.csv"
```

The validator reports schema differences, target distribution, missing values, duplicates, and non-numeric columns. It does not modify the source CSV.

## Roadmap

1. Project foundation and data validation
2. Exploratory data analysis
3. Logistic Regression baseline
4. Imbalance handling
5. XGBoost and hyperparameter tuning
6. Threshold and cost-sensitive evaluation
7. SHAP explainability
8. PyTorch autoencoder and anomaly detection
9. Hybrid risk engine (only if validation supports it)
10. FastAPI, PostgreSQL, Streamlit
11. MLflow, monitoring, tests, Docker, and deployment

## Reproducibility and limitations

- Training, validation, and test data must be separated before any resampling.
- Keep the held-out test set out of model selection and threshold tuning.
- The public dataset is historical and anonymized; its features do not represent all information available in a real banking system.
- Add results only after executing experiments; do not put unmeasured metrics in the README.

## Phase 2 — Exploratory data analysis

After placing `creditcard.csv` in `data/raw/`, you can explore the data interactively:

```powershell
jupyter lab
```

Open `notebooks/01_eda.ipynb` and run the cells from top to bottom. The notebook covers dataset structure, class imbalance, missing values, duplicate detection, amount/time distributions, feature distributions, and correlation analysis.

For a repeatable command-line report that writes CSV tables, a JSON summary, and PNG plots to `reports/eda/`, run:

```powershell
python -m src.data.eda
```

To supply a different dataset path or report destination:

```powershell
python -m src.data.eda --data-path "D:\path\to\creditcard.csv" --output-dir reports/eda
```

The EDA script never edits the raw data or drops records. Duplicate counts are investigation signals, not an instruction to delete rows. `Time` is a relative elapsed-time feature, not a wall-clock timestamp, and `V1`–`V28` are anonymized PCA components without directly available business meanings.

Run the tests with:

```powershell
python -m pytest -q
```

## Phase 3 — Logistic Regression baseline

The first baseline uses a chronological 60/20/20 train/validation/test split ordered by `Time`. The split code keeps equal timestamp values in one partition. The model is an sklearn pipeline: median imputation, standard scaling, and Logistic Regression. All preprocessing is fitted on train only. The default threshold is 0.5 and no oversampling or class weighting is applied in this phase.

With the analyzed dataset, the test partition contains 56,961 rows and 75 fraud cases. At threshold 0.5, the first run achieved precision 0.7500, recall 0.5600, F1 0.6412, ROC-AUC 0.9731, and Average Precision 0.6926. The test confusion matrix is TN=56,872, FP=14, FN=33, TP=42. Accuracy is 99.9175%, but is not sufficient for an imbalanced dataset.

Run the baseline from the repository root:

```powershell
python -m src.models.baseline
```

The outputs are written to `reports/baseline/`, and the fitted pipeline is saved to `models/logistic_regression_baseline.joblib`. The raw CSV is excluded from Git. Read `reports/baseline/PHASE3_BASELINE.md` for split details, metric definitions, limitations, and leakage controls.

Open `notebooks/02_baseline_logistic_regression.ipynb` with JupyterLab for an interactive walkthrough.

## Phase 4 — Imbalanced learning experiments

Install the additional dependency with `pip install -r requirements.txt`, then run:

```powershell
python -m src.models.imbalance_experiments
```

This compares the unweighted Logistic Regression baseline, `class_weight="balanced"`, random undersampling, and SMOTE. All strategies use the same chronological 60/20/20 split. Imputation/scaling and resampling are fitted within the training pipeline only; validation and test retain their natural class ratios. Strategies are ranked by validation Average Precision, and only the validation-selected strategy is evaluated on the holdout test partition. The decision threshold is fixed at 0.5 in this phase; threshold optimization is a later step.

Reports are written to `reports/imbalance/`. The raw dataset is not committed to Git.


## Phase 5 — XGBoost and hyperparameter tuning

Install dependencies with `pip install -r requirements.txt`, then run:

```powershell
python -m src.models.xgboost_model
```

The experiment builds expanding-window chronological CV folds inside the training partition and runs `RandomizedSearchCV` using Average Precision. It searches tree complexity, learning rate, row/column subsampling, regularization, and `scale_pos_weight`. It uses CV to rank candidates, checks the top candidates on validation Average Precision because the chronological periods exhibit distribution shift, then refits the selected configuration on train + validation before evaluating the untouched test partition. The test set is not used during this XGBoost search or threshold tuning. Because the same time block has been reviewed in earlier learning phases, treat it as a repeated holdout across project history; for a strictly independent final claim after further iteration, create a new later-time holdout.

The selected model is saved to `models/xgboost_best.joblib` and the native XGBoost model is saved to `models/xgboost_best.json`. Reports, randomized-search results, candidate validation metrics, test predictions, confusion matrices, feature importance, and a model comparison are saved under `reports/xgboost/`. Read `reports/xgboost/PHASE5_XGBOOST_FINDINGS.md` before using results in a resume or project description.

The current tuning budget is four randomized configurations across three chronological CV folds to keep the run practical; use `python -m src.models.xgboost_model --n-iter 10 --n-jobs 2` for a broader search. The selected validation metric is Average Precision, so threshold-0.5 precision and F1 may remain low. Threshold optimization and cost-sensitive decision-making are Phase 6.

## Phase 6 — Fraud decision engine

Phase 6 converts the Phase 5 XGBoost score into a thresholded decision and relative risk bands.

```powershell
python -m src.evaluation.decision_engine
python -m pytest -q
```

Policy assumptions live in `configs/decision_policy.yaml`; outputs are written to `reports/decision_engine/` and `models/decision_policy.json`. The default false-negative/false-positive cost values are illustrative relative units, not financial loss estimates. The output score is uncalibrated and must not be presented as a verified fraud probability. Threshold selection uses validation labels only; test metrics are reported only after the threshold has been selected.

Open `notebooks/05_fraud_decision_engine.ipynb` for the step-by-step explanation.
