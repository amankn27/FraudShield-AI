"""Tune, train, evaluate, and persist FraudShield AI's XGBoost classifier.

Hyperparameter selection is performed using expanding-window chronological CV
inside the training partition only. Validation is used for a post-selection
check, and the held-out test partition is evaluated after refitting the chosen
configuration on train + validation.
"""
from __future__ import annotations

import argparse
import json
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
from scipy.stats import randint, loguniform, uniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import load_config
from src.data.splitting import chronological_train_validation_test_split
from src.evaluation.metrics import classification_metrics


def _boundary(time_values: pd.Series, proposed: int) -> int:
    """Move a split point to the end of an equal-Time group."""
    n = len(time_values)
    proposed = min(max(int(proposed), 1), n - 1)
    while proposed < n and time_values.iloc[proposed] == time_values.iloc[proposed - 1]:
        proposed += 1
    return proposed


def make_chronological_cv_splits(
    X_train: pd.DataFrame,
    *,
    time_column: str = "Time",
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build three expanding-window folds without splitting equal-Time groups.

    Fold 1 trains on the first ~50% and validates on the next ~16.7%; fold 2
    trains on the first ~66.7% and validates on the next ~16.7%; fold 3 trains
    on the first ~83.3% and validates on the latest ~16.7% of the training block.
    """
    if time_column not in X_train:
        raise ValueError(f"Missing time column {time_column!r} for chronological CV")
    X = X_train.reset_index(drop=True)
    if not X[time_column].is_monotonic_increasing:
        raise ValueError("X_train must be sorted chronologically before CV")

    n = len(X)
    raw = [round(n * 0.5), round(n * (2 / 3)), round(n * (5 / 6)), n]
    boundaries = [_boundary(X[time_column], value) if value < n else n for value in raw]
    if not (0 < boundaries[0] < boundaries[1] < boundaries[2] < boundaries[3] == n):
        raise ValueError(f"Could not form valid chronological CV blocks: {boundaries}")

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    starts = [0, boundaries[0], boundaries[1]]
    train_ends = boundaries[:3]
    val_ends = boundaries[1:]
    for fold, (train_end, val_end) in enumerate(zip(train_ends, val_ends), start=1):
        train_idx = np.arange(0, train_end, dtype=np.int64)
        val_idx = np.arange(train_end, val_end, dtype=np.int64)
        if len(train_idx) == 0 or len(val_idx) == 0:
            raise ValueError(f"Fold {fold} is empty")
        if X.iloc[train_idx][time_column].max() >= X.iloc[val_idx][time_column].min():
            raise AssertionError(f"Chronological leakage in fold {fold}")
        if X.iloc[val_idx][time_column].nunique() == 0:
            raise ValueError(f"Fold {fold} has no validation timestamps")
        splits.append((train_idx, val_idx))
    return splits


def _save_confusion_matrix(cm: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax,
        xticklabels=["Predicted legitimate", "Predicted fraud"],
        yticklabels=["Actual legitimate", "Actual fraud"],
    )
    ax.set(title=title, xlabel="Prediction", ylabel="Actual label")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_feature_importance(model: XGBClassifier, feature_names: list[str], out_path: Path) -> pd.DataFrame:
    importances = pd.DataFrame({
        "feature": feature_names,
        "importance_gain": model.feature_importances_,
    }).sort_values("importance_gain", ascending=False)
    importances.to_csv(out_path.with_suffix(".csv"), index=False)
    top = importances.head(20).sort_values("importance_gain", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top["feature"], top["importance_gain"])
    ax.set(title="XGBoost feature importance (top 20)", xlabel="Importance", ylabel="Feature")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return importances


def _save_precision_recall_curve(y_true: pd.Series, scores: np.ndarray, out_path: Path, title: str) -> None:
    from sklearn.metrics import precision_recall_curve

    precision, recall, _ = precision_recall_curve(y_true, scores)
    prevalence = float(np.mean(y_true))
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.plot(recall, precision, label="XGBoost")
    ax.axhline(prevalence, linestyle="--", label=f"Positive prevalence ({prevalence:.3%})")
    ax.set(title=title, xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_comparison_chart(comparison_df: pd.DataFrame, out_path: Path) -> None:
    metric_columns = [
        ("pr_auc_average_precision", "Average Precision"),
        ("precision", "Precision @ 0.5"),
        ("recall", "Recall @ 0.5"),
        ("f1", "F1 @ 0.5"),
    ]
    fig, axes = plt.subplots(1, len(metric_columns), figsize=(15, max(4.8, 0.72 * len(comparison_df))))
    for ax, (column, label) in zip(axes, metric_columns):
        rows = comparison_df.sort_values(column, ascending=True)
        ax.barh(rows["model"], rows[column])
        ax.set_title(label)
        ax.set_xlim(0, 1)
        ax.grid(axis="x", alpha=0.25)
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("Held-out test comparison (threshold-dependent metrics at 0.5)", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _make_model(params: dict[str, Any], *, random_seed: int, n_jobs: int = 4) -> XGBClassifier:
    base = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "random_state": random_seed,
        "n_jobs": n_jobs,
        "verbosity": 0,
    }
    base.update(params)
    return XGBClassifier(**base)


def run_xgboost_experiment(
    data_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    model_path: str | Path | None = None,
    *,
    n_iter: int = 4,
    cv_splits: int = 3,
    n_jobs: int = 2,
    random_seed: int | None = None,
) -> dict[str, Any]:
    """Run RandomizedSearchCV on train only, refit on train+validation, test once."""
    project_root = Path(__file__).resolve().parents[2]
    config = load_config()
    data_path = Path(data_path) if data_path else project_root / config["data"]["raw_path"]
    output_dir = Path(output_dir) if output_dir else project_root / "reports" / "xgboost"
    model_path = Path(model_path) if model_path else project_root / "models" / "xgboost_best.joblib"
    native_model_path = model_path.with_suffix(".json")
    random_seed = int(random_seed if random_seed is not None else config.get("project", {}).get("random_seed", 42))
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}. Put creditcard.csv in data/raw/.")
    if n_iter < 1:
        raise ValueError("n_iter must be at least 1")
    if cv_splits != 3:
        raise ValueError("This implementation currently defines exactly three chronological CV folds")

    df = pd.read_csv(data_path)
    target = config["data"].get("target_column", "Class")
    expected_columns = config["data"].get("expected_columns", [])
    missing_columns = sorted(set(expected_columns) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Dataset missing required columns: {missing_columns}")
    if df[target].isna().any() or not set(df[target].unique()).issubset({0, 1}):
        raise ValueError(f"Target {target!r} must contain only non-missing 0/1 labels")
    if df.isna().any().any():
        raise ValueError("Dataset contains missing values; review Phase 2 before training")

    splits = chronological_train_validation_test_split(df, time_column="Time", train_fraction=0.60, validation_fraction=0.20)
    feature_columns = [col for col in df.columns if col != target]
    X_train = splits.train[feature_columns].reset_index(drop=True)
    y_train = splits.train[target].astype(int).reset_index(drop=True)
    X_val = splits.validation[feature_columns].reset_index(drop=True)
    y_val = splits.validation[target].astype(int).reset_index(drop=True)
    X_test = splits.test[feature_columns].reset_index(drop=True)
    y_test = splits.test[target].astype(int).reset_index(drop=True)

    cv = make_chronological_cv_splits(X_train, time_column="Time")
    cv_summary = []
    for fold_no, (train_idx, val_idx) in enumerate(cv, start=1):
        cv_summary.append({
            "fold": fold_no,
            "train_rows": int(len(train_idx)),
            "train_fraud": int(y_train.iloc[train_idx].sum()),
            "validation_rows": int(len(val_idx)),
            "validation_fraud": int(y_train.iloc[val_idx].sum()),
            "train_time_max": float(X_train.iloc[train_idx]["Time"].max()),
            "validation_time_min": float(X_train.iloc[val_idx]["Time"].min()),
            "validation_time_max": float(X_train.iloc[val_idx]["Time"].max()),
        })
        if y_train.iloc[val_idx].nunique() < 2:
            raise ValueError(f"CV fold {fold_no} validation block must contain both classes")

    # The conventional ratio is a search candidate, not assumed to be optimal.
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    class_ratio = negative_count / max(positive_count, 1)
    parameter_distributions = {
        "n_estimators": randint(60, 181),
        "max_depth": [2, 3, 4],
        "learning_rate": loguniform(0.02, 0.20),
        "min_child_weight": loguniform(1.0, 20.0),
        "subsample": uniform(0.65, 0.35),
        "colsample_bytree": uniform(0.60, 0.40),
        "gamma": [0.0, 0.01, 0.1, 0.3, 1.0],
        "reg_alpha": [0.0, 0.001, 0.01, 0.1, 1.0],
        "reg_lambda": [1.0, 3.0, 5.0, 10.0],
        "scale_pos_weight": [1.0, 10.0, 25.0, 50.0, 100.0, 200.0, float(class_ratio)],
    }
    estimator = _make_model({}, random_seed=random_seed, n_jobs=n_jobs)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=parameter_distributions,
        n_iter=n_iter,
        scoring="average_precision",
        cv=cv,
        refit=False,
        random_state=random_seed,
        n_jobs=1,  # Avoid nested CPU oversubscription; XGBoost itself uses n_jobs.
        verbose=1,
        return_train_score=True,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    # sklearn embeds parameter dictionaries; stringify them for robust CSV export.
    csv_results = cv_results.copy()
    csv_results["params"] = csv_results["params"].map(lambda p: json.dumps(p, sort_keys=True, default=float))
    csv_results.to_csv(output_dir / "randomized_search_cv_results.csv", index=False)

    cv_best_params = {key: (value.item() if isinstance(value, np.generic) else value) for key, value in search.best_params_.items()}
    cv_best_ap = float(search.best_score_)

    # Fraud patterns change across time blocks. Use CV to shortlist candidates,
    # then choose among the top five CV-ranked candidates using validation AP.
    # The test partition remains untouched during both steps.
    candidate_rows = []
    candidate_models = []
    for _, row in cv_results.head(min(5, len(cv_results))).iterrows():
        params = {key: (value.item() if isinstance(value, np.generic) else value) for key, value in row["params"].items()}
        candidate_model = _make_model(params, random_seed=random_seed, n_jobs=n_jobs)
        candidate_model.fit(X_train, y_train)
        candidate_scores = candidate_model.predict_proba(X_val)[:, 1]
        candidate_metrics, _ = classification_metrics(y_val, candidate_scores, threshold=0.5)
        candidate_record = {
            "cv_rank": int(row["rank_test_score"]),
            "mean_cv_average_precision": float(row["mean_test_score"]),
            "mean_cv_train_average_precision": float(row["mean_train_score"]),
            "validation_average_precision": float(candidate_metrics["pr_auc_average_precision"]),
            "validation_precision_at_0_5": float(candidate_metrics["precision"]),
            "validation_recall_at_0_5": float(candidate_metrics["recall"]),
            "validation_f1_at_0_5": float(candidate_metrics["f1"]),
            "scale_pos_weight": float(params["scale_pos_weight"]),
            "n_estimators": int(params["n_estimators"]),
            "max_depth": int(params["max_depth"]),
            "learning_rate": float(params["learning_rate"]),
            "params": json.dumps(params, sort_keys=True, default=float),
        }
        candidate_rows.append(candidate_record)
        candidate_models.append((candidate_record, params, candidate_model, candidate_scores, candidate_metrics))
    pd.DataFrame(candidate_rows).sort_values("validation_average_precision", ascending=False).to_csv(
        output_dir / "validation_candidate_comparison.csv", index=False
    )
    selected_candidate = max(candidate_models, key=lambda item: (item[0]["validation_average_precision"], item[0]["mean_cv_average_precision"]))
    selected_record, best_params, validation_model, val_scores, val_metrics = selected_candidate
    selected_cv_ap = float(selected_record["mean_cv_average_precision"])
    val_metrics, val_cm = classification_metrics(y_val, val_scores, threshold=0.5)
    _save_confusion_matrix(val_cm, output_dir / "confusion_matrix_validation.png", "XGBoost validation confusion matrix (threshold=0.5)")
    pd.DataFrame({
        "Time": X_val["Time"].to_numpy(),
        "actual_class": y_val.to_numpy(),
        "fraud_score": val_scores,
        "predicted_class_at_0_5": (val_scores >= 0.5).astype(int),
    }).to_csv(output_dir / "validation_predictions.csv", index=False)

    # Once configuration is chosen without accessing test labels, train final artifact on train + validation.
    X_train_val = pd.concat([X_train, X_val], ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], ignore_index=True)
    final_model = _make_model(best_params, random_seed=random_seed, n_jobs=n_jobs)
    final_model.fit(X_train_val, y_train_val)
    test_scores = final_model.predict_proba(X_test)[:, 1]
    test_metrics, test_cm = classification_metrics(y_test, test_scores, threshold=0.5)
    _save_confusion_matrix(test_cm, output_dir / "confusion_matrix_test.png", "XGBoost test confusion matrix (threshold=0.5)")
    pd.DataFrame({
        "Time": X_test["Time"].to_numpy(),
        "actual_class": y_test.to_numpy(),
        "fraud_score": test_scores,
        "predicted_class_at_0_5": (test_scores >= 0.5).astype(int),
    }).to_csv(output_dir / "test_predictions.csv", index=False)
    _save_precision_recall_curve(y_val, val_scores, output_dir / "precision_recall_curve_validation.png", "XGBoost precision-recall curve — validation")
    _save_precision_recall_curve(y_test, test_scores, output_dir / "precision_recall_curve_test.png", "XGBoost precision-recall curve — test")
    importance_df = _save_feature_importance(final_model, feature_columns, output_dir / "feature_importance.png")

    # Refit the simple baseline on the same train+validation rows for a fairer
    # model-family comparison; the historical Phase 3 train-only score is also retained.
    lr_final = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, solver="lbfgs", random_state=random_seed)),
    ])
    lr_final.fit(X_train_val, y_train_val)
    lr_test_scores = lr_final.predict_proba(X_test)[:, 1]
    lr_test_metrics, lr_test_cm = classification_metrics(y_test, lr_test_scores, threshold=0.5)
    _save_confusion_matrix(lr_test_cm, output_dir / "confusion_matrix_logistic_refit_test.png", "Logistic Regression refit test confusion matrix (threshold=0.5)")
    joblib.dump(lr_final, output_dir / "logistic_regression_train_validation.joblib")
    pd.DataFrame({
        "Time": X_test["Time"].to_numpy(),
        "actual_class": y_test.to_numpy(),
        "fraud_score": lr_test_scores,
        "predicted_class_at_0_5": (lr_test_scores >= 0.5).astype(int),
    }).to_csv(output_dir / "logistic_regression_train_validation_test_predictions.csv", index=False)

    # Compare the untouched test partition with prior recorded experiments.
    comparison_rows = []
    baseline_json = project_root / "reports" / "baseline" / "baseline_metrics.json"
    if baseline_json.exists():
        baseline = json.loads(baseline_json.read_text(encoding="utf-8"))
        m = baseline["metrics"]["test"]
        comparison_rows.append({"model": "Logistic Regression baseline (Phase 3)", **m})
    phase4_csv = project_root / "reports" / "imbalance" / "selected_model_test_metrics.csv"
    if phase4_csv.exists():
        phase4 = pd.read_csv(phase4_csv)
        if not phase4.empty:
            m = phase4.iloc[0].to_dict()
            comparison_rows.append({
                "model": f"Phase 4 {m.get('strategy', 'selected imbalance model')}",
                **{key: m[key] for key in ["threshold", "n_samples", "n_positive_actual", "n_negative_actual", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc_average_precision", "pr_auc_trapezoidal", "true_negative", "false_positive", "false_negative", "true_positive"] if key in m},
            })
    comparison_rows.append({"model": "Logistic Regression (refit train + validation)", **lr_test_metrics})
    comparison_rows.append({"model": "XGBoost tuned (train + validation refit)", **test_metrics})
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(output_dir / "test_model_comparison.csv", index=False)
    _save_comparison_chart(comparison_df, output_dir / "test_model_comparison.png")

    split_summary = {
        name: {
            "rows": int(len(partition)),
            "fraud_count": int(partition[target].sum()),
            "legitimate_count": int((partition[target] == 0).sum()),
            "fraud_rate_percent": float(partition[target].mean() * 100),
            "time_min": float(partition["Time"].min()),
            "time_max": float(partition["Time"].max()),
        }
        for name, partition in [("train", splits.train), ("validation", splits.validation), ("test", splits.test)]
    }
    report = {
        "project": "FraudShield AI",
        "model": "XGBClassifier",
        "xgboost_version": __import__("xgboost").__version__,
        "selection_metric": "RandomizedSearchCV Average Precision for CV ranking; validation Average Precision selects among the top five CV-ranked candidates",
        "search": {
            "n_iter": n_iter,
            "cv_splits": cv_summary,
            "best_cv_average_precision": cv_best_ap,
            "best_cv_params": cv_best_params,
            "selected_candidate_cv_rank": int(selected_record["cv_rank"]),
            "selected_candidate_cv_average_precision": selected_cv_ap,
            "selected_candidate_validation_average_precision": float(selected_record["validation_average_precision"]),
            "selected_params": best_params,
            "train_negative_count": negative_count,
            "train_positive_count": positive_count,
            "negative_to_positive_ratio_candidate": float(class_ratio),
            "scoring": "average_precision",
            "test_touched_during_search": False,
        },
        "split_summary": split_summary,
        "validation_metrics_threshold_0_5": val_metrics,
        "test_metrics_threshold_0_5": test_metrics,
        "artifacts": {
            "model_path_joblib": str(model_path.relative_to(project_root)) if model_path.is_relative_to(project_root) else str(model_path),
            "model_path_native_json": str(native_model_path.relative_to(project_root)) if native_model_path.is_relative_to(project_root) else str(native_model_path),
            "cv_results": "reports/xgboost/randomized_search_cv_results.csv",
            "comparison": "reports/xgboost/test_model_comparison.csv",
            "comparison_chart": "reports/xgboost/test_model_comparison.png",
            "feature_importance": "reports/xgboost/feature_importance.csv",
        },
        "interpretation_notes": [
            "RandomizedSearchCV ranks parameter candidates using chronological CV Average Precision inside the training partition only.",
            "The top five CV-ranked candidates (or fewer if fewer were searched) are then compared using validation Average Precision because CV and validation time blocks show distribution shift.",
            "The validation partition is used for model selection, so it is not an unbiased final evaluation; the test partition remains untouched until final evaluation.",
            "CV folds are expanding windows made from contiguous chronological blocks; equal-Time groups are not split.",
            "The final XGBoost artifact is refit on training + validation after the model configuration is selected.",
            "The test partition is held out from hyperparameter selection and evaluated after final refit.",
            "At threshold 0.5, weighted model scores are not guaranteed to be calibrated real-world fraud probabilities.",
            "scale_pos_weight negative/positive ratio is included as one search candidate, not assumed to be optimal.",
            "Feature importance describes model usage, not causality; V1-V28 are anonymized PCA components.",
            "Prior Phase 3/4 test metrics are included for historical comparison. Training protocols differ across those earlier experiments and should be noted when interpreting differences.",
        ],
    }
    joblib.dump(final_model, model_path)
    final_model.save_model(str(native_model_path))
    with (output_dir / "xgboost_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)
    pd.DataFrame([
        {"partition": "validation", **val_metrics},
        {"partition": "test", **test_metrics},
    ]).to_csv(output_dir / "xgboost_metrics.csv", index=False)
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=None, help="Path to creditcard.csv")
    parser.add_argument("--output-dir", default=None, help="Output report directory")
    parser.add_argument("--model-path", default=None, help="Path for saved XGBoost model")
    parser.add_argument("--n-iter", type=int, default=4, help="Number of RandomizedSearchCV configurations (default: 4; increase for a broader search)")
    parser.add_argument("--n-jobs", type=int, default=2, help="CPU workers used by each XGBoost fit (default: 2)")
    args = parser.parse_args()
    report = run_xgboost_experiment(args.data, args.output_dir, args.model_path, n_iter=args.n_iter, n_jobs=args.n_jobs)
    print("\nFraudShield AI — XGBoost experiment")
    print("=" * 52)
    print(f"XGBoost version: {report['xgboost_version']}")
    print(f"Highest CV Average Precision: {report['search']['best_cv_average_precision']:.6f}")
    print(f"Selected candidate validation Average Precision: {report['search']['selected_candidate_validation_average_precision']:.6f}")
    print("Selected parameters:")
    for key, value in report["search"]["selected_params"].items():
        print(f"  {key}: {value}")
    print("\nValidation metrics at threshold=0.5:")
    print(json.dumps(report["validation_metrics_threshold_0_5"], indent=2))
    print("\nHeld-out test metrics at threshold=0.5:")
    print(json.dumps(report["test_metrics_threshold_0_5"], indent=2))
    print("\nComparison: reports/xgboost/test_model_comparison.csv")
    print("Saved sklearn model: models/xgboost_best.joblib")
    print("Saved native XGBoost model: models/xgboost_best.json")


if __name__ == "__main__":
    main()
