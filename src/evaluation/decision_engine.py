"""Threshold selection, cost-sensitive decisions, and risk-band assignment.

Inputs are Phase 5 out-of-time score exports:
  reports/xgboost/validation_predictions.csv  (scores from a model fit on train)
  reports/xgboost/test_predictions.csv       (scores from final train+validation refit)

The threshold is selected only on validation labels. The test partition is used
only to report the generalization of that already-selected policy.

IMPORTANT: XGBoost was trained with class weighting. Its predict_proba output is
therefore treated as an uncalibrated model score until calibration is assessed.
The displayed 0-100 risk score is score*100, not a claim of real-world fraud
probability.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, roc_auc_score
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "decision_policy.yaml"
DEFAULT_VALIDATION = PROJECT_ROOT / "reports" / "xgboost" / "validation_predictions.csv"
DEFAULT_TEST = PROJECT_ROOT / "reports" / "xgboost" / "test_predictions.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "decision_engine"
DEFAULT_POLICY_OUT = PROJECT_ROOT / "models" / "decision_policy.json"


def load_policy_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Policy config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    costs = config.get("costs", {})
    cost_fp = float(costs.get("false_positive", 1.0))
    cost_fn = float(costs.get("false_negative", 100.0))
    if cost_fp < 0 or cost_fn < 0 or (cost_fp == 0 and cost_fn == 0):
        raise ValueError("Cost values must be non-negative and at least one must be positive")
    return config


def load_prediction_file(path: str | Path, partition: str) -> pd.DataFrame:
    """Load saved labels and scores without rebuilding the model or touching test labels."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {partition} predictions: {path}. Run Phase 5 first: python -m src.models.xgboost_model"
        )
    df = pd.read_csv(path)
    required = {"actual_class", "fraud_score"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} must contain columns {sorted(required)}")
    if df["actual_class"].isna().any() or not set(df["actual_class"].unique()).issubset({0, 1}):
        raise ValueError(f"{partition} actual_class must contain non-missing binary labels 0/1")
    scores = df["fraud_score"].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError(f"{partition} fraud_score values must be finite and between 0 and 1")
    if df["actual_class"].nunique() < 2:
        raise ValueError(f"{partition} must contain both classes")
    return df.reset_index(drop=True)


def threshold_sweep(y_true: np.ndarray, scores: np.ndarray, cost_fp: float, cost_fn: float) -> pd.DataFrame:
    """Evaluate every distinct score threshold in O(n log n), including no-alert option."""
    y = np.asarray(y_true, dtype=np.int8)
    s = np.asarray(scores, dtype=float)
    if y.ndim != 1 or s.ndim != 1 or len(y) != len(s) or len(y) == 0:
        raise ValueError("y_true and scores must be non-empty one-dimensional arrays of equal length")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("y_true must contain only 0/1 labels")
    if not np.isfinite(s).all() or ((s < 0) | (s > 1)).any():
        raise ValueError("scores must be finite and lie in [0, 1]")

    order = np.argsort(-s, kind="mergesort")
    sorted_scores = s[order]
    sorted_y = y[order]
    group_ends = np.flatnonzero(np.r_[sorted_scores[:-1] != sorted_scores[1:], True])
    thresholds = sorted_scores[group_ends]
    tp = np.cumsum(sorted_y, dtype=np.int64)[group_ends]
    predicted_positive = group_ends + 1
    fp = predicted_positive - tp
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    fn = positives - tp
    tn = negatives - fp

    # Add a threshold strictly above the maximum score, representing no alerts.
    thresholds = np.r_[np.nextafter(float(s.max()), np.inf), thresholds]
    tp = np.r_[0, tp]
    fp = np.r_[0, fp]
    fn = np.r_[positives, fn]
    tn = np.r_[negatives, tn]
    predicted_positive = tp + fp
    precision = np.divide(tp, predicted_positive, out=np.zeros(len(tp), dtype=float), where=predicted_positive > 0)
    recall = tp / positives if positives else np.zeros(len(tp), dtype=float)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(len(tp), dtype=float), where=(precision + recall) > 0)
    cost = cost_fp * fp + cost_fn * fn
    return pd.DataFrame({
        "threshold": thresholds,
        "true_negative": tn.astype(int),
        "false_positive": fp.astype(int),
        "false_negative": fn.astype(int),
        "true_positive": tp.astype(int),
        "predicted_positive": predicted_positive.astype(int),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cost_fp_weighted": cost_fp * fp,
        "cost_fn_weighted": cost_fn * fn,
        "total_cost_units": cost,
    })


def choose_cost_threshold(sweep: pd.DataFrame) -> pd.Series:
    """Select minimum-cost validation threshold; ties prefer higher threshold/fewer alerts."""
    minimum = float(sweep["total_cost_units"].min())
    tied = sweep[np.isclose(sweep["total_cost_units"], minimum)]
    return tied.sort_values("threshold", ascending=False).iloc[0]


def threshold_for_target_recall(sweep: pd.DataFrame, target_recall: float) -> pd.Series:
    """Highest threshold that attains at least the requested recall on validation."""
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be in (0, 1]")
    eligible = sweep[sweep["recall"] >= target_recall]
    if eligible.empty:
        raise ValueError(f"Could not reach target recall {target_recall:.3f}")
    return eligible.sort_values("threshold", ascending=False).iloc[0]


def evaluate_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float,
                       cost_fp: float, cost_fn: float) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=np.int8)
    s = np.asarray(scores, dtype=float)
    pred = (s >= float(threshold)).astype(np.int8)
    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "n_samples": int(len(y)),
        "n_fraud_actual": int(y.sum()),
        "n_legitimate_actual": int(len(y) - y.sum()),
        "accuracy": float((tp + tn) / len(y)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y, s)),
        "average_precision": float(average_precision_score(y, s)),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "predicted_positive": int(tp + fp),
        "cost_false_positives": float(cost_fp * fp),
        "cost_false_negatives": float(cost_fn * fn),
        "total_cost_units": float(cost_fp * fp + cost_fn * fn),
    }


def assign_risk_level(scores: np.ndarray, low_boundary: float, medium_boundary: float,
                       critical_boundary: float) -> np.ndarray:
    """Assign ordinal, validation-derived relative score tiers.

    Boundaries are not calibrated-probability cutoffs. Current policy uses:
      LOW:      score < threshold meeting >=95% validation recall
      MEDIUM:   that boundary <= score < threshold meeting >=85% validation recall
      HIGH:     that boundary <= score < cost-sensitive classification threshold
      CRITICAL: score >= cost-sensitive classification threshold
    """
    if not (0 <= low_boundary < medium_boundary < critical_boundary <= 1.0000001):
        raise ValueError("Risk thresholds must satisfy 0 <= low < medium < critical <= 1")
    s = np.asarray(scores, dtype=float)
    return np.select(
        [s < low_boundary, s < medium_boundary, s < critical_boundary],
        ["LOW", "MEDIUM", "HIGH"],
        default="CRITICAL",
    )


def _save_confusion_matrix(metrics: dict[str, Any], out_path: Path, title: str) -> None:
    cm = np.array([[metrics["true_negative"], metrics["false_positive"]],
                   [metrics["false_negative"], metrics["true_positive"]]])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Predicted legitimate", "Predicted fraud"],
                yticklabels=["Actual legitimate", "Actual fraud"])
    ax.set(title=title, xlabel="Prediction", ylabel="Actual label")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_cost_sweep(sweep: pd.DataFrame, selected: float, out_path: Path) -> None:
    plotted = sweep[np.isfinite(sweep["threshold"])].sort_values("threshold")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(plotted["threshold"], plotted["total_cost_units"], linewidth=1.5)
    ax.axvline(selected, linestyle="--", label=f"Selected threshold = {selected:.4f}")
    ax.set(title="Validation cost across decision thresholds", xlabel="Model score threshold",
           ylabel="Assumed cost (relative units)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_policy_comparison(comparison: pd.DataFrame, out_path: Path) -> None:
    cols = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1-score")]
    fig, axes = plt.subplots(1, len(cols), figsize=(13, 4.8))
    for ax, (col, label) in zip(axes, cols):
        ax.bar(comparison["policy"], comparison[col])
        ax.set_title(label)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Held-out test results at different fixed thresholds")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_risk_band_counts(validation_bands: pd.Series, test_bands: pd.Series, out_path: Path) -> None:
    order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    df = pd.DataFrame({
        "Validation": validation_bands.value_counts().reindex(order, fill_value=0),
        "Test": test_bands.value_counts().reindex(order, fill_value=0),
    })
    ax = df.plot(kind="bar", figsize=(8, 5))
    ax.set(title="Transaction counts by relative risk band", xlabel="Risk band", ylabel="Transactions")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Partition")
    ax.figure.tight_layout()
    ax.figure.savefig(out_path, dpi=160)
    plt.close(ax.figure)


def run_decision_engine(
    validation_path: str | Path = DEFAULT_VALIDATION,
    test_path: str | Path = DEFAULT_TEST,
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT,
    policy_out: str | Path = DEFAULT_POLICY_OUT,
) -> dict[str, Any]:
    config = load_policy_config(config_path)
    costs = config.get("costs", {})
    cost_fp = float(costs.get("false_positive", 1.0))
    cost_fn = float(costs.get("false_negative", 100.0))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(policy_out).parent.mkdir(parents=True, exist_ok=True)

    val_df = load_prediction_file(validation_path, "validation")
    test_df = load_prediction_file(test_path, "test")
    y_val = val_df["actual_class"].to_numpy(dtype=np.int8)
    s_val = val_df["fraud_score"].to_numpy(dtype=float)
    y_test = test_df["actual_class"].to_numpy(dtype=np.int8)
    s_test = test_df["fraud_score"].to_numpy(dtype=float)

    # All policy choices are made using validation labels only.
    sweep = threshold_sweep(y_val, s_val, cost_fp, cost_fn)
    selected = choose_cost_threshold(sweep)
    selected_threshold = float(selected["threshold"])
    risk_cfg = config.get("risk_bands", {})
    target_recall_low_medium = float(risk_cfg.get("target_recall_low_medium", 0.95))
    target_recall_medium_high = float(risk_cfg.get("target_recall_medium_high", 0.85))
    val95 = threshold_for_target_recall(sweep, target_recall_low_medium)
    val85 = threshold_for_target_recall(sweep, target_recall_medium_high)
    if not (float(val95["threshold"]) < float(val85["threshold"]) < selected_threshold):
        raise ValueError(
            "Risk-band thresholds are not ordered for this cost configuration. "
            "Adjust target-recall anchors or review cost assumptions in configs/decision_policy.yaml."
        )

    val_default = evaluate_threshold(y_val, s_val, 0.5, cost_fp, cost_fn)
    val_cost = evaluate_threshold(y_val, s_val, selected_threshold, cost_fp, cost_fn)
    test_default = evaluate_threshold(y_test, s_test, 0.5, cost_fp, cost_fn)
    test_cost = evaluate_threshold(y_test, s_test, selected_threshold, cost_fp, cost_fn)
    val_default["policy"] = "Default threshold 0.5"
    val_cost["policy"] = "Validation-selected cost threshold"
    test_default["policy"] = "Default threshold 0.5"
    test_cost["policy"] = "Validation-selected cost threshold"

    sweep.to_csv(output_dir / "validation_threshold_sweep.csv", index=False)
    pd.DataFrame([val_default, val_cost]).to_csv(output_dir / "validation_policy_comparison.csv", index=False)
    pd.DataFrame([test_default, test_cost]).to_csv(output_dir / "test_policy_comparison.csv", index=False)

    sensitivity_rows = []
    for ratio in config.get("cost_sensitivity", {}).get("false_negative_costs", [1, 5, 10, 25, 50, 100, 250, 500]):
        ratio = float(ratio)
        ratio_sweep = threshold_sweep(y_val, s_val, cost_fp=cost_fp, cost_fn=ratio)
        ratio_selected = choose_cost_threshold(ratio_sweep)
        ratio_threshold = float(ratio_selected["threshold"])
        val_m = evaluate_threshold(y_val, s_val, ratio_threshold, cost_fp, ratio)
        test_m = evaluate_threshold(y_test, s_test, ratio_threshold, cost_fp, ratio)
        sensitivity_rows.append({
            "assumed_false_negative_cost": ratio,
            "assumed_false_positive_cost": cost_fp,
            "selected_validation_threshold": ratio_threshold,
            "validation_total_cost_units": val_m["total_cost_units"],
            "validation_precision": val_m["precision"],
            "validation_recall": val_m["recall"],
            "validation_false_positives": val_m["false_positive"],
            "validation_false_negatives": val_m["false_negative"],
            "test_total_cost_units_at_selected_threshold": test_m["total_cost_units"],
            "test_precision": test_m["precision"],
            "test_recall": test_m["recall"],
            "test_false_positives": test_m["false_positive"],
            "test_false_negatives": test_m["false_negative"],
        })
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(output_dir / "cost_sensitivity.csv", index=False)

    # Risk tiers are ordinal score bands, not calibrated probability bands.
    risk_low = float(val95["threshold"])
    risk_medium = float(val85["threshold"])
    risk_critical = selected_threshold
    val_bands = assign_risk_level(s_val, risk_low, risk_medium, risk_critical)
    test_bands = assign_risk_level(s_test, risk_low, risk_medium, risk_critical)

    val_decisions = val_df.copy()
    val_decisions["risk_score_0_100"] = val_decisions["fraud_score"] * 100.0
    val_decisions["risk_level"] = val_bands
    val_decisions["decision_threshold"] = selected_threshold
    val_decisions["decision"] = np.where(val_decisions["fraud_score"] >= selected_threshold,
                                          "FLAG_FOR_REVIEW", "NOT_FLAGGED")
    test_decisions = test_df.copy()
    test_decisions["risk_score_0_100"] = test_decisions["fraud_score"] * 100.0
    test_decisions["risk_level"] = test_bands
    test_decisions["decision_threshold"] = selected_threshold
    test_decisions["decision"] = np.where(test_decisions["fraud_score"] >= selected_threshold,
                                           "FLAG_FOR_REVIEW", "NOT_FLAGGED")
    val_decisions.to_csv(output_dir / "validation_decisions.csv", index=False)
    test_decisions.to_csv(output_dir / "test_decisions.csv", index=False)

    risk_counts = pd.DataFrame({
        "validation_count": pd.Series(val_bands).value_counts(),
        "test_count": pd.Series(test_bands).value_counts(),
    }).reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"], fill_value=0).fillna(0).astype(int)
    risk_counts.to_csv(output_dir / "risk_band_distribution.csv", index_label="risk_level")
    _save_confusion_matrix(test_cost, output_dir / "confusion_matrix_test_cost_threshold.png",
                           f"Test confusion matrix @ {selected_threshold:.4f}")
    _plot_cost_sweep(sweep, selected_threshold, output_dir / "validation_cost_vs_threshold.png")
    _plot_policy_comparison(pd.DataFrame([test_default, test_cost]), output_dir / "test_policy_comparison.png")
    _plot_risk_band_counts(pd.Series(val_bands), pd.Series(test_bands), output_dir / "risk_band_distribution.png")

    policy = {
        "policy_version": "1.0.0",
        "model_artifact": "models/xgboost_best.joblib",
        "score_definition": "risk_score_0_100 = 100 * XGBoost predict_proba[:, 1]; uncalibrated model score, not verified real-world probability",
        "decision": {
            "flag_label": "FLAG_FOR_REVIEW",
            "not_flagged_label": "NOT_FLAGGED",
            "threshold": selected_threshold,
            "threshold_selection": "minimum assumed cost on validation data only; ties prefer higher threshold/fewer alerts",
            "cost_false_positive_units": cost_fp,
            "cost_false_negative_units": cost_fn,
            "cost_unit_note": "Relative illustrative units, not currency or measured bank loss",
        },
        "risk_bands": {
            "LOW": {"lower_inclusive": 0.0, "upper_exclusive": risk_low},
            "MEDIUM": {"lower_inclusive": risk_low, "upper_exclusive": risk_medium},
            "HIGH": {"lower_inclusive": risk_medium, "upper_exclusive": risk_critical},
            "CRITICAL": {"lower_inclusive": risk_critical, "upper_inclusive": 1.0},
            "boundary_method": "validation score thresholds anchored to >=95% and >=85% recall, plus selected cost threshold",
            "note": "Relative triage bands, not calibrated probability ranges; review policy against false-alarm capacity before operational use",
        },
        "validation_summary": {"default_0_5": val_default, "selected_policy": val_cost},
        "test_summary": {"default_0_5": test_default, "selected_policy": test_cost},
        "risk_threshold_anchors": {
            "target_recall_low_medium": target_recall_low_medium,
            "target_recall_low_medium_threshold": float(val95["threshold"]),
            "achieved_validation_recall_at_low_medium_anchor": float(val95["recall"]),
            "target_recall_medium_high": target_recall_medium_high,
            "target_recall_medium_high_threshold": float(val85["threshold"]),
            "achieved_validation_recall_at_medium_high_anchor": float(val85["recall"]),
        },
        "artifacts": {
            "threshold_sweep": "reports/decision_engine/validation_threshold_sweep.csv",
            "cost_sensitivity": "reports/decision_engine/cost_sensitivity.csv",
            "validation_decisions": "reports/decision_engine/validation_decisions.csv",
            "test_decisions": "reports/decision_engine/test_decisions.csv",
        },
        "warning": "Threshold was selected on a small validation fraud sample and did not improve the assumed cost on the held-out test period; do not treat it as a production policy without further temporal validation and cost validation.",
    }
    Path(policy_out).write_text(json.dumps(policy, indent=2), encoding="utf-8")

    result = {
        "policy_path": str(policy_out),
        "output_dir": str(output_dir),
        "cost_false_positive": cost_fp,
        "cost_false_negative": cost_fn,
        "selected_threshold": selected_threshold,
        "validation_default": val_default,
        "validation_selected": val_cost,
        "test_default": test_default,
        "test_selected": test_cost,
        "risk_band_thresholds": {"low_medium": risk_low, "medium_high": risk_medium,
                                 "high_critical": risk_critical},
    }
    (output_dir / "decision_engine_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_report(result, sensitivity, risk_counts, output_dir / "PHASE6_DECISION_ENGINE_FINDINGS.md")
    return result


def _write_report(result: dict[str, Any], sensitivity: pd.DataFrame, risk_counts: pd.DataFrame, path: Path) -> None:
    def pct(x: float) -> str: return f"{x:.2%}"
    vd, vs = result["validation_default"], result["validation_selected"]
    td, ts = result["test_default"], result["test_selected"]
    lines = [
        "# Phase 6 — Fraud Decision Engine Findings", "",
        "## Objective", "",
        "Convert XGBoost scores into a decision policy with an explicit cost assumption and ordinal risk tiers.", "",
        "## Probability vs classification", "",
        "`fraud_score` is the XGBoost `predict_proba[:, 1]` output. Because the model was trained with class weighting and has not been calibrated, this score must not be interpreted as a verified real-world fraud probability. `risk_score_0_100 = 100 × fraud_score` is a convenient score scale, not a probability percentage.", "",
        "A **score** is continuous; a **classification** is produced by applying a threshold. Changing the threshold changes precision/recall without retraining the model.", "",
        "## Policy assumptions", "",
        f"- False-positive cost: **{result['cost_false_positive']:g} relative units**.",
        f"- False-negative cost: **{result['cost_false_negative']:g} relative units**.",
        "- These are illustrative assumptions, not currency, chargeback amounts, investigation costs, or measured financial losses.",
        "- The threshold is selected by minimum assumed cost on validation data only. The test set is not used to choose the threshold.", "",
        "## Threshold selection results", "",
        "| Partition / policy | Threshold | Precision | Recall | F1 | FP | FN | Assumed total cost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Validation, default | 0.5000 | {pct(vd['precision'])} | {pct(vd['recall'])} | {vd['f1']:.4f} | {vd['false_positive']} | {vd['false_negative']} | {vd['total_cost_units']:.0f} |",
        f"| Validation, selected | {vs['threshold']:.6f} | {pct(vs['precision'])} | {pct(vs['recall'])} | {vs['f1']:.4f} | {vs['false_positive']} | {vs['false_negative']} | {vs['total_cost_units']:.0f} |",
        f"| Test, default | 0.5000 | {pct(td['precision'])} | {pct(td['recall'])} | {td['f1']:.4f} | {td['false_positive']} | {td['false_negative']} | {td['total_cost_units']:.0f} |",
        f"| Test, validation-selected | {ts['threshold']:.6f} | {pct(ts['precision'])} | {pct(ts['recall'])} | {ts['f1']:.4f} | {ts['false_positive']} | {ts['false_negative']} | {ts['total_cost_units']:.0f} |",
        "", "### Important result", "",
        f"The selected threshold reduced validation cost from {vd['total_cost_units']:.0f} to {vs['total_cost_units']:.0f} relative units. On the later test period, its cost was {ts['total_cost_units']:.0f} versus {td['total_cost_units']:.0f} at threshold 0.5. **It did not improve held-out test cost under the same assumed ratio.** This is an important generalization warning: the validation set has only 57 fraud cases, and score/rate patterns change over time.", "",
        "## Risk bands", "",
        "Risk tiers are relative model-score bands derived from validation cutoffs; they are not calibrated probability intervals.", "",
        "| Risk level | Rule | Interpretation in this prototype |",
        "|---|---|---|",
        f"| LOW | score < {result['risk_band_thresholds']['low_medium']:.6f} | Below the validation score cutoff needed to reach 95% recall |",
        f"| MEDIUM | {result['risk_band_thresholds']['low_medium']:.6f} ≤ score < {result['risk_band_thresholds']['medium_high']:.6f} | Between the 95%- and 85%-recall score cutoffs |",
        f"| HIGH | {result['risk_band_thresholds']['medium_high']:.6f} ≤ score < {result['risk_band_thresholds']['high_critical']:.6f} | Elevated score, below cost-selected classification threshold |",
        f"| CRITICAL | score ≥ {result['risk_band_thresholds']['high_critical']:.6f} | Meets the cost-selected threshold and is flagged for review |",
        "", "These labels support prioritization in a demo. Actual financial workflows should set band boundaries and actions with a business owner, analyst-capacity constraints, and independently validated costs.", "",
        "## Cost sensitivity", "",
        "See `cost_sensitivity.csv` for the threshold selected on validation for several assumed false-negative costs and its test-period outcome. Different cost ratios can produce materially different alert volumes.", "",
        "## Limitations and next steps", "",
        "1. The assumed cost ratio is illustrative; the dataset does not provide operational bank costs.",
        "2. The scores are not calibrated. Assess calibration using temporally appropriate data before displaying them as literal probabilities.",
        "3. The validation block has few fraud cases, so threshold choice is noisy.",
        "4. The validation-selected threshold increased the test cost under this assumed cost ratio; perform broader temporal validation before adopting a policy.",
        "5. Continue to monitor precision, recall, alert volume, and delayed ground-truth labels after deployment.", "",
        "## Generated artifacts", "",
        "- `validation_threshold_sweep.csv`", "- `validation_policy_comparison.csv`", "- `test_policy_comparison.csv`",
        "- `cost_sensitivity.csv`", "- `validation_decisions.csv`", "- `test_decisions.csv`",
        "- `confusion_matrix_test_cost_threshold.png`", "- `validation_cost_vs_threshold.png`", "- `test_policy_comparison.png`", "- `risk_band_distribution.png`", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", default=str(DEFAULT_VALIDATION))
    parser.add_argument("--test", default=str(DEFAULT_TEST))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--policy-out", default=str(DEFAULT_POLICY_OUT))
    args = parser.parse_args()
    result = run_decision_engine(args.validation, args.test, args.config, args.output_dir, args.policy_out)
    print(json.dumps({
        "selected_threshold": result["selected_threshold"],
        "validation_cost_default": result["validation_default"]["total_cost_units"],
        "validation_cost_selected": result["validation_selected"]["total_cost_units"],
        "test_cost_default": result["test_default"]["total_cost_units"],
        "test_cost_selected": result["test_selected"]["total_cost_units"],
        "policy_path": result["policy_path"],
        "output_dir": result["output_dir"],
    }, indent=2))


if __name__ == "__main__":
    main()
