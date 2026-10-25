"""Train and evaluate FraudShield AI's first Logistic Regression baseline."""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_curve, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import load_config
from src.data.splitting import chronological_train_validation_test_split
from src.evaluation.metrics import classification_metrics


def _save_confusion_matrix(cm: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    sns.heatmap(
        cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax,
        xticklabels=["Predicted legitimate", "Predicted fraud"],
        yticklabels=["Actual legitimate", "Actual fraud"],
    )
    ax.set_title(title)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Actual label")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_curves(y_true: pd.Series, probabilities: np.ndarray, output_dir: Path, partition: str) -> None:
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    fig, ax = plt.subplots(figsize=(6.3, 5.0))
    ax.plot(fpr, tpr, label="Logistic Regression")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random ranking")
    ax.set(title=f"ROC curve — {partition}", xlabel="False positive rate", ylabel="True positive rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"roc_curve_{partition.lower()}.png", dpi=160)
    plt.close(fig)

    # Precision-recall curve is especially useful with rare positive cases.
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    baseline_prevalence = float(np.mean(y_true))
    fig, ax = plt.subplots(figsize=(6.3, 5.0))
    ax.plot(recall, precision, label="Logistic Regression")
    ax.axhline(baseline_prevalence, linestyle="--", label=f"Positive prevalence ({baseline_prevalence:.3%})")
    ax.set(title=f"Precision-recall curve — {partition}", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"precision_recall_curve_{partition.lower()}.png", dpi=160)
    plt.close(fig)


def run_baseline(
    data_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    model_path: str | Path | None = None,
    threshold: float = 0.5,
) -> dict:
    """Fit the baseline on the earliest time block, evaluate validation and test.

    The model and threshold are fixed a priori. Test metrics are calculated once
    for this initial baseline and should not be used for model selection.
    """
    config = load_config()
    project_root = Path(__file__).resolve().parents[2]
    data_path = Path(data_path) if data_path else project_root / config["data"]["raw_path"]
    output_dir = Path(output_dir) if output_dir else project_root / "reports" / "baseline"
    model_path = Path(model_path) if model_path else project_root / "models" / "logistic_regression_baseline.joblib"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}. Put creditcard.csv in data/raw/.")
    df = pd.read_csv(data_path)
    target = config["data"].get("target_column", "Class")
    time_column = "Time"
    expected_columns = config["data"].get("expected_columns", [])
    missing_columns = sorted(set(expected_columns) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Dataset missing required columns: {missing_columns}")
    if df[target].isna().any() or not set(df[target].unique()).issubset({0, 1}):
        raise ValueError(f"Target '{target}' must contain only non-missing 0/1 labels")
    if df.isna().any().any():
        raise ValueError("Dataset contains missing values; inspect Phase 2 report before baseline training")

    splits = chronological_train_validation_test_split(
        df,
        time_column=time_column,
        train_fraction=0.60,
        validation_fraction=0.20,
    )
    feature_columns = [c for c in df.columns if c != target]
    X_train, y_train = splits.train[feature_columns], splits.train[target].astype(int)
    X_val, y_val = splits.validation[feature_columns], splits.validation[target].astype(int)
    X_test, y_test = splits.test[feature_columns], splits.test[target].astype(int)

    # All learned preprocessing is part of the pipeline, and is fitted on training data only.
    # Time, Amount, and PCA components are standardized for Logistic Regression.
    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=int(config.get("project", {}).get("random_seed", 42)),
        )),
    ])
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(X_train, y_train)
    model_warnings = [str(w.message) for w in captured_warnings]

    results: dict[str, dict] = {}
    partitions = {
        "train": (X_train, y_train),
        "validation": (X_val, y_val),
        "test": (X_test, y_test),
    }
    prediction_rows = []
    for name, (X, y) in partitions.items():
        probabilities = model.predict_proba(X)[:, 1]
        metrics, cm = classification_metrics(y, probabilities, threshold=threshold)
        results[name] = metrics
        _save_confusion_matrix(cm, output_dir / f"confusion_matrix_{name}.png", f"Confusion matrix — {name.title()} (threshold={threshold:g})")
        if name in {"validation", "test"}:
            _save_curves(y, probabilities, output_dir, name)
            prediction_rows.append(pd.DataFrame({
                "partition": name,
                "Time": X[time_column].to_numpy(),
                "actual_class": y.to_numpy(),
                "fraud_probability": probabilities,
                "predicted_class": (probabilities >= threshold).astype(int),
            }))

    split_summary = {
        name: {
            "rows": int(len(partition)),
            "fraud_count": int(partition[target].sum()),
            "legitimate_count": int((partition[target] == 0).sum()),
            "fraud_rate_percent": float(partition[target].mean() * 100),
            "time_min": float(partition[time_column].min()),
            "time_max": float(partition[time_column].max()),
        }
        for name, partition in [
            ("train", splits.train),
            ("validation", splits.validation),
            ("test", splits.test),
        ]
    }
    report = {
        "project": "FraudShield AI",
        "model": "LogisticRegression",
        "model_configuration": {
            "class_weight": None,
            "max_iter": 1000,
            "solver": "lbfgs",
            "threshold": threshold,
            "preprocessing": ["median imputer (fitted on train only)", "StandardScaler (fitted on train only)"],
        },
        "split_strategy": "chronological 60/20/20 by row order, preserving equal-Time groups",
        "split_cutoffs": splits.cutoffs,
        "split_summary": split_summary,
        "metrics": results,
        "warnings": model_warnings,
        "interpretation_notes": [
            "Accuracy is not a sufficient metric because fraud is rare.",
            "Precision, recall, F1, ROC-AUC, and average precision are reported.",
            "Average precision is the primary PR ranking metric; trapezoidal PR-AUC is also reported for transparency.",
            "Threshold 0.5 is an initial reference, not an optimized business threshold.",
            "The test set is for final evaluation only and must not be used to tune the model.",
            "This historical chronological split tests later records, but this dataset spans a limited time window.",
        ],
    }
    # Save the single end-to-end pipeline so inference uses identical preprocessing.
    joblib.dump(model, model_path)
    try:
        report["model_artifact"] = model_path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        report["model_artifact"] = str(model_path)
    with (output_dir / "baseline_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    pd.DataFrame([
        {"partition": partition, **metrics} for partition, metrics in results.items()
    ]).to_csv(output_dir / "baseline_metrics.csv", index=False)
    pd.DataFrame(split_summary).T.to_csv(output_dir / "split_summary.csv", index=True, index_label="partition")
    pd.concat(prediction_rows, ignore_index=True).to_csv(output_dir / "validation_test_predictions.csv", index=False)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=None, help="Path to creditcard.csv")
    parser.add_argument("--output-dir", default=None, help="Directory for reports and plots")
    parser.add_argument("--model-path", default=None, help="Where to save the trained pipeline")
    parser.add_argument("--threshold", type=float, default=0.5, help="Initial prediction threshold (default: 0.5)")
    args = parser.parse_args()
    report = run_baseline(args.data, args.output_dir, args.model_path, args.threshold)
    print("\nFraudShield AI — Logistic Regression baseline")
    print("=" * 52)
    print("Split summary:")
    for name, metrics in report["split_summary"].items():
        print(f"  {name:10s}: rows={metrics['rows']:>7,} | fraud={metrics['fraud_count']:>3,} | fraud_rate={metrics['fraud_rate_percent']:.4f}% | time={metrics['time_min']:g}..{metrics['time_max']:g}")
    print(f"\nMetrics at threshold={report['model_configuration']['threshold']:g}:")
    headers = ["partition", "precision", "recall", "f1", "roc_auc", "pr_auc_average_precision", "true_positive", "false_positive", "false_negative", "true_negative"]
    print(pd.DataFrame([{ "partition": k, **v } for k, v in report["metrics"].items()])[headers].to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"\nReports: {Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parents[2] / 'reports' / 'baseline'}")
    print(f"Model: {Path(args.model_path) if args.model_path else Path(__file__).resolve().parents[2] / 'models' / 'logistic_regression_baseline.joblib'}")


if __name__ == "__main__":
    main()
