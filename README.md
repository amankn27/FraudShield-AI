# FraudShield AI

An end-to-end learning project for imbalanced financial fraud detection. The planned system will compare supervised ML and deep-learning approaches, explain predictions, expose a prediction API, and add monitoring and reproducible experiments.

> **Status:** Phase 1 — project foundation. No model performance claims are made yet.

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
