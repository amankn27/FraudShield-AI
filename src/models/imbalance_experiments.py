"""Compare imbalance-handling strategies for FraudShield AI.

All experiments fit on the same chronological training partition. Model-selection
metrics are computed on the untouched, naturally imbalanced validation partition.
Only the validation-selected strategy is evaluated on the holdout test partition.
Resampling happens inside an imbalanced-learn Pipeline, during fit on training data
only, preventing validation/test leakage.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler

from src.config import load_config
from src.data.splitting import chronological_train_validation_test_split
from src.evaluation.metrics import classification_metrics


SELECTION_METRIC = "pr_auc_average_precision"
DEFAULT_THRESHOLD = 0.5


def make_experiment_pipeline(
    strategy: str,
    *,
    random_state: int = 42,
    max_iter: int = 1500,
) -> ImbPipeline:
    """Create a consistent Logistic Regression pipeline for one strategy.

    Supported strategies: baseline, class_weight, undersampling, smote.
    The scaler precedes SMOTE because SMOTE uses nearest-neighbour distances.
    Samplers are executed only in fit(), never on validation/test rows.
    """
    steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    if strategy == "undersampling":
        steps.append(("sampler", RandomUnderSampler(random_state=random_state)))
        class_weight = None
    elif strategy == "smote":
        steps.append(("sampler", SMOTE(random_state=random_state, k_neighbors=5)))
        class_weight = None
    elif strategy == "class_weight":
        class_weight = "balanced"
    elif strategy == "baseline":
        class_weight = None
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    steps.append((
        "classifier",
        LogisticRegression(
            class_weight=class_weight,
            max_iter=max_iter,
            solver="lbfgs",
            random_state=random_state,
        ),
    ))
    return ImbPipeline(steps=steps)


def _save_confusion_matrix(cm: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.7, 4.8))
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


def _save_validation_comparison(results: pd.DataFrame, path: Path) -> None:
    metrics = ["precision", "recall", "f1", "roc_auc", "pr_auc_average_precision"]
    labels = {
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1-score",
        "roc_auc": "ROC-AUC",
        "pr_auc_average_precision": "Average Precision",
    }
    fig, ax = plt.subplots(figsize=(12, 6.5))
    x = np.arange(len(results))
    width = 0.15
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - (len(metrics) - 1) / 2) * width, results[metric], width, label=labels[metric])
    ax.set_xticks(x)
    ax.set_xticklabels(results["experiment"], rotation=18, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Validation comparison — threshold metrics at 0.5")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, loc="upper center")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_pr_curves(
    y_true: pd.Series,
    validation_probability_by_name: dict[str, np.ndarray],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for name, probability in validation_probability_by_name.items():
        precision, recall, _ = precision_recall_curve(y_true, probability)
        ap = classification_metrics(y_true, probability, threshold=DEFAULT_THRESHOLD)[0][SELECTION_METRIC]
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    prevalence = float(np.mean(y_true))
    ax.axhline(prevalence, linestyle="--", color="gray", label=f"Validation prevalence={prevalence:.3%}")
    ax.set(title="Precision–recall curves — validation", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _training_class_counts(y: pd.Series) -> dict[str, int]:
    counts = y.value_counts().to_dict()
    return {"legitimate_class_0": int(counts.get(0, 0)), "fraud_class_1": int(counts.get(1, 0))}


def _estimated_resampled_counts(strategy: str, y_train: pd.Series) -> dict[str, int]:
    counts = _training_class_counts(y_train)
    negatives = counts["legitimate_class_0"]
    positives = counts["fraud_class_1"]
    if strategy == "undersampling":
        return {"legitimate_class_0": min(negatives, positives), "fraud_class_1": min(negatives, positives)}
    if strategy == "smote":
        maximum = max(negatives, positives)
        return {"legitimate_class_0": maximum, "fraud_class_1": maximum}
    return counts


def run_imbalance_experiments(
    data_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    model_path: str | Path | None = None,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    random_state: int | None = None,
    max_iter: int = 1500,
) -> dict[str, Any]:
    """Run validation comparisons; test only the approach selected on validation AP."""
    config = load_config()
    root = Path(__file__).resolve().parents[2]
    data_path = Path(data_path) if data_path else root / config["data"]["raw_path"]
    output_dir = Path(output_dir) if output_dir else root / "reports" / "imbalance"
    model_path = Path(model_path) if model_path else root / "models" / "logistic_regression_imbalance_best.joblib"
    random_state = int(random_state if random_state is not None else config.get("project", {}).get("random_seed", 42))
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}. Put creditcard.csv in data/raw/.")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    df = pd.read_csv(data_path)
    target = config["data"].get("target_column", "Class")
    expected = config["data"].get("expected_columns", [])
    missing_columns = sorted(set(expected) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Dataset missing required columns: {missing_columns}")
    if df[target].isna().any() or not set(df[target].unique()).issubset({0, 1}):
        raise ValueError(f"Target '{target}' must contain only non-missing 0/1 labels")
    if df.isna().any().any():
        raise ValueError("Dataset has missing values; check the Phase 2 report before running experiments")

    splits = chronological_train_validation_test_split(df, time_column="Time", train_fraction=0.60, validation_fraction=0.20)
    feature_columns = [column for column in df.columns if column != target]
    X_train = splits.train[feature_columns]
    y_train = splits.train[target].astype(int)
    X_val = splits.validation[feature_columns]
    y_val = splits.validation[target].astype(int)
    X_test = splits.test[feature_columns]
    y_test = splits.test[target].astype(int)

    train_counts = _training_class_counts(y_train)
    experiment_specs = [
        ("baseline", "Baseline (no resampling)"),
        ("class_weight", "Class weights (balanced)"),
        ("undersampling", "Random undersampling"),
        ("smote", "SMOTE oversampling"),
    ]
    validation_rows: list[dict[str, Any]] = []
    validation_predictions: list[pd.DataFrame] = []
    validation_probabilities: dict[str, np.ndarray] = {}

    for strategy, display_name in experiment_specs:
        model = make_experiment_pipeline(strategy, random_state=random_state, max_iter=max_iter)
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # Keep convergence warnings visible in the result report; do not silently hide them.
            model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start
        probabilities = model.predict_proba(X_val)[:, 1]
        metrics, cm = classification_metrics(y_val, probabilities, threshold=threshold)
        row: dict[str, Any] = {
            "strategy": strategy,
            "experiment": display_name,
            "fit_seconds": round(float(fit_seconds), 3),
            "train_rows_original": int(len(y_train)),
            "train_legitimate_original": train_counts["legitimate_class_0"],
            "train_fraud_original": train_counts["fraud_class_1"],
            "train_rows_after_strategy": int(sum(_estimated_resampled_counts(strategy, y_train).values())),
            **metrics,
            "warnings": " | ".join(str(w.message) for w in caught),
        }
        validation_rows.append(row)
        validation_probabilities[display_name] = probabilities
        validation_predictions.append(pd.DataFrame({
            "experiment": display_name,
            "Time": X_val["Time"].to_numpy(),
            "actual_class": y_val.to_numpy(),
            "fraud_probability": probabilities,
            "predicted_class": (probabilities >= threshold).astype(int),
        }))
        _save_confusion_matrix(
            cm,
            output_dir / f"confusion_matrix_validation_{strategy}.png",
            f"Validation confusion matrix — {display_name} (threshold={threshold:g})",
        )
        print(f"Finished {display_name}: validation AP={metrics[SELECTION_METRIC]:.5f}, fit={fit_seconds:.1f}s")

    results_df = pd.DataFrame(validation_rows).sort_values(SELECTION_METRIC, ascending=False).reset_index(drop=True)
    # Model selection uses validation Average Precision only. Test labels are not consulted here.
    selected_strategy = str(results_df.iloc[0]["strategy"])
    selected_name = str(results_df.iloc[0]["experiment"])

    results_df.to_csv(output_dir / "validation_experiment_comparison.csv", index=False)
    pd.concat(validation_predictions, ignore_index=True).to_csv(output_dir / "validation_predictions_all_experiments.csv", index=False)
    _save_validation_comparison(results_df, output_dir / "validation_metric_comparison.png")
    _save_pr_curves(y_val, validation_probabilities, output_dir / "validation_precision_recall_curves.png")

    # After selecting a strategy using validation AP, refit that strategy on all
    # development data (train + validation), then evaluate it on the test partition.
    # The test set is not used to choose the strategy, fit preprocessing, or tune thresholds.
    X_development = pd.concat([X_train, X_val], axis=0, ignore_index=True)
    y_development = pd.concat([y_train, y_val], axis=0, ignore_index=True)
    selected_model = make_experiment_pipeline(selected_strategy, random_state=random_state, max_iter=max_iter)
    with warnings.catch_warnings(record=True) as final_fit_warnings:
        warnings.simplefilter("always")
        selected_model.fit(X_development, y_development)
    test_probabilities = selected_model.predict_proba(X_test)[:, 1]
    test_metrics, test_cm = classification_metrics(y_test, test_probabilities, threshold=threshold)
    _save_confusion_matrix(
        test_cm,
        output_dir / "confusion_matrix_test_selected.png",
        f"Selected model test confusion matrix — {selected_name} (threshold={threshold:g})",
    )
    pd.DataFrame([{
        "strategy": selected_strategy,
        "experiment": selected_name,
        **test_metrics,
    }]).to_csv(output_dir / "selected_model_test_metrics.csv", index=False)
    pd.DataFrame({
        "Time": X_test["Time"].to_numpy(),
        "actual_class": y_test.to_numpy(),
        "fraud_probability": test_probabilities,
        "predicted_class": (test_probabilities >= threshold).astype(int),
    }).to_csv(output_dir / "selected_model_test_predictions.csv", index=False)
    joblib.dump(selected_model, model_path)

    split_summary = {
        name: {
            "rows": int(len(part)),
            "fraud_count": int(part[target].sum()),
            "legitimate_count": int((part[target] == 0).sum()),
            "fraud_rate_percent": float(part[target].mean() * 100),
            "time_min": float(part["Time"].min()),
            "time_max": float(part["Time"].max()),
        }
        for name, part in [("train", splits.train), ("validation", splits.validation), ("test", splits.test)]
    }
    try:
        artifact_rel_path = model_path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        artifact_rel_path = str(model_path)

    naive_validation_accuracy = float((y_val == 0).mean())
    balanced_class_weights = {
        int(class_id): float(len(y_train) / (2 * int(count)))
        for class_id, count in y_train.value_counts().items()
    }
    second_best_ap = float(results_df.iloc[1][SELECTION_METRIC]) if len(results_df) > 1 else None
    best_ap = float(results_df.iloc[0][SELECTION_METRIC])
    report: dict[str, Any] = {
        "project": "FraudShield AI",
        "phase": 4,
        "model_family": "LogisticRegression",
        "selection_metric": SELECTION_METRIC,
        "selected_strategy_by_validation_ap": selected_strategy,
        "selected_experiment": selected_name,
        "best_validation_average_precision": best_ap,
        "runner_up_validation_average_precision": second_best_ap,
        "validation_ap_gap_to_runner_up": (best_ap - second_best_ap) if second_best_ap is not None else None,
        "naive_predict_all_legitimate_validation_accuracy": naive_validation_accuracy,
        "naive_predict_all_legitimate_validation_recall": 0.0,
        "balanced_class_weights_from_training_partition": balanced_class_weights,
        "threshold_used_for_fixed_threshold_metrics": float(threshold),
        "resampling_policy": "All resampling is performed inside an imbalanced-learn pipeline fitted on train only; validation and test keep their natural class distributions.",
        "split_strategy": "chronological 60/20/20 by Time, preserving equal-Time groups",
        "split_cutoffs": splits.cutoffs,
        "split_summary": split_summary,
        "original_training_class_counts": train_counts,
        "validation_results": results_df.to_dict(orient="records"),
        "final_fit_rows_train_plus_validation": int(len(y_development)),
        "final_fit_fraud_cases_train_plus_validation": int(y_development.sum()),
        "final_fit_warnings": [str(w.message) for w in final_fit_warnings],
        "selected_model_test_metrics": test_metrics,
        "selected_model_test_confusion_matrix_labels_0_1": test_cm.tolist(),
        "model_artifact": artifact_rel_path,
        "interpretation_notes": [
            "Accuracy can be misleading because fraud is rare; prioritize precision, recall, F1 and Average Precision alongside ROC-AUC.",
            "Class weights change the training loss; undersampling discards majority examples; SMOTE synthesizes minority examples using nearest neighbours.",
            "Model selection used validation Average Precision only. The fixed threshold is 0.5 for this experiment and has not been optimized.",
            "Class weights and resampling can change probability calibration; threshold tuning and calibration checks belong to later phases.",
            "The selected model's test metrics are reported once here. Do not use the test result to tune the model or threshold.",
            "The validation set contains relatively few positive cases; validation metrics can be noisy and should be interpreted with counts and PR curves.",
        ],
    }
    (output_dir / "imbalance_experiment_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_report(report, results_df, output_dir / "PHASE4_IMBALANCE_FINDINGS.md")
    return report


def _write_report(report: dict[str, Any], results_df: pd.DataFrame, path: Path) -> None:
    split_rows = report["split_summary"]
    lines = [
        "# FraudShield AI — Phase 4: Imbalanced learning experiments",
        "",
        "## Experimental protocol",
        "",
        "All approaches use the same chronological train/validation/test split and Logistic Regression classifier. "
        "Models are ranked using validation Average Precision. Validation and test data are never resampled. "
        "The selected strategy is evaluated on the test partition once; the fixed 0.5 threshold is not optimized in this phase.",
        "",
        "## Split summary",
        "",
        "| Partition | Rows | Fraud cases | Legitimate cases | Fraud rate | Time range |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, vals in split_rows.items():
        lines.append(f"| {name.title()} | {vals['rows']:,} | {vals['fraud_count']:,} | {vals['legitimate_count']:,} | {vals['fraud_rate_percent']:.4f}% | {vals['time_min']:g}–{vals['time_max']:g} |")
    lines += [
        "",
        "## Why accuracy fails here",
        "",
        f"The validation partition has {split_rows['validation']['legitimate_count']:,} legitimate rows and {split_rows['validation']['fraud_count']:,} fraud rows. A trivial classifier that predicts every row as legitimate would achieve **{report['naive_predict_all_legitimate_validation_accuracy']:.4%} accuracy while catching 0 frauds**. That is why accuracy is reported but not used to select the strategy.",
        "",
        "The `class_weight='balanced'` heuristic assigns weights of approximately "
        f"**{report['balanced_class_weights_from_training_partition'].get(0, 0):.3f}** to Class 0 and "
        f"**{report['balanced_class_weights_from_training_partition'].get(1, 0):.3f}** to Class 1 in this training partition. "
        "This large difference tells the loss function to pay much more attention to fraud errors.",
        "",
        "## Validation comparison",
        "",
        "All fixed-threshold metrics below use threshold 0.5. Average Precision (AP) is the selection metric. Precision/recall/F1 and confusion counts depend on this threshold; ROC-AUC and AP evaluate ranking across thresholds.",
        "",
        "| Approach | Train rows after strategy | Accuracy | Precision | Recall | F1 | ROC-AUC | Average Precision | FP | FN | Fit seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in results_df.iterrows():
        lines.append(
            f"| {r['experiment']} | {int(r['train_rows_after_strategy']):,} | {r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['roc_auc']:.4f} | {r['pr_auc_average_precision']:.4f} | {int(r['false_positive'])} | {int(r['false_negative'])} | {r['fit_seconds']:.1f} |"
        )
    test = report["selected_model_test_metrics"]
    lines += [
        "",
        "## Validation-selected approach: test evaluation",
        "",
        f"Selected by validation AP: **{report['selected_experiment']}**. After strategy selection, that pipeline was refitted on the combined training + validation data ({report['final_fit_rows_train_plus_validation']:,} rows, including {report['final_fit_fraud_cases_train_plus_validation']} fraud cases). The test partition has {test['n_samples']:,} rows and {test['n_positive_actual']} fraud cases.",
        "",
        "| Metric | Test result |",
        "|---|---:|",
    ]
    for key, label in [
        ("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"),
        ("f1", "F1-score"), ("roc_auc", "ROC-AUC"), ("pr_auc_average_precision", "Average Precision"),
        ("true_positive", "True positives"), ("false_positive", "False positives"),
        ("false_negative", "False negatives"), ("true_negative", "True negatives"),
    ]:
        val = test[key]
        lines.append(f"| {label} | {val:.5f} |" if isinstance(val, float) else f"| {label} | {val:,} |")
    lines += [
        "",
        "## How confidently should we choose a winner?",
        "",
        f"The selected strategy's validation AP is {report['best_validation_average_precision']:.5f}; the runner-up AP is {report['runner_up_validation_average_precision']:.5f} (gap {report['validation_ap_gap_to_runner_up']:.5f}). This difference is small relative to the uncertainty from only 57 validation fraud cases, so treat it as a near-tie rather than proof that SMOTE is generally superior. Class weighting is simpler operationally because it does not synthesize or discard rows. The fixed-threshold results show a separate precision/recall trade-off that must be addressed with threshold selection and, if needed, calibration.",
        "",
        "## What each approach does",
        "",
        "- **Baseline:** learns from the original skewed training distribution.",
        "- **Class weights:** assigns larger loss weight to the minority class (`class_weight='balanced'`); no rows are created or discarded.",
        "- **Random undersampling:** randomly discards majority-class training cases until classes are balanced; fast, but removes legitimate examples.",
        "- **SMOTE:** synthesizes minority-class training examples using nearest neighbours; can help decision-boundary learning but cannot create new real-world evidence and may synthesize ambiguous points.",
        "",
        "## Leakage prevention and interpretation",
        "",
        "- Imputation and scaling are fitted on training data only.",
        "- Samplers are placed inside an `imblearn.pipeline.Pipeline`, so they run only during `fit` on the training partition.",
        "- Validation/test rows retain the natural, imbalanced distribution.",
        "- Model selection is based on validation Average Precision, not test results.",
        "- All approaches use a fixed 0.5 threshold for this comparison. Rebalancing can shift predicted probabilities; Phase 6 will study business thresholds, and probability calibration may need separate evaluation.",
        "- Validation has a small number of frauds, so validation scores are noisy. Read the confusion counts and precision-recall curves alongside scalar metrics.",
        "",
        "## Generated artifacts",
        "",
        "- `validation_experiment_comparison.csv` — validation metric table for all approaches.",
        "- `validation_predictions_all_experiments.csv` — validation probabilities and labels.",
        "- `validation_metric_comparison.png` — metric visualization.",
        "- `validation_precision_recall_curves.png` — validation PR curves.",
        "- `selected_model_test_metrics.csv` — test metrics for the validation-selected approach only.",
        "- `selected_model_test_predictions.csv` — test predictions for that approach.",
        "- `imbalance_experiment_results.json` — machine-readable experiment details.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=None, help="Path to creditcard.csv")
    parser.add_argument("--output-dir", default=None, help="Output directory for experiment reports")
    parser.add_argument("--model-path", default=None, help="Path to save validation-selected pipeline")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=1500)
    args = parser.parse_args()
    report = run_imbalance_experiments(
        data_path=args.data,
        output_dir=args.output_dir,
        model_path=args.model_path,
        threshold=args.threshold,
        random_state=args.random_state,
        max_iter=args.max_iter,
    )
    print("\nFraudShield AI — Phase 4 imbalanced learning")
    print("=" * 56)
    print(pd.DataFrame(report["validation_results"])[[
        "experiment", "precision", "recall", "f1", "roc_auc",
        "pr_auc_average_precision", "false_positive", "false_negative", "fit_seconds",
    ]].to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"\nSelected by validation AP: {report['selected_experiment']}")
    print("Selected model test metrics:")
    for k in ["precision", "recall", "f1", "roc_auc", "pr_auc_average_precision", "true_positive", "false_positive", "false_negative", "true_negative"]:
        print(f"  {k}: {report['selected_model_test_metrics'][k]}")
    print(f"\nReports: {Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parents[2] / 'reports' / 'imbalance'}")


if __name__ == "__main__":
    main()
