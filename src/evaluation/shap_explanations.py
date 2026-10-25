"""Generate global and local SHAP explanations for FraudShield AI.

All model fitting and threshold selection occurred in prior phases. This module
uses the saved XGBoost artifact and the Phase 6 policy; SHAP analysis is an
interpretability pass, not an additional model-training step.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import PROJECT_ROOT, load_config
from src.data.splitting import chronological_train_validation_test_split
from src.evaluation.shap_explainer import FraudShapService


def _ensure_additivity(scores: np.ndarray, base_values: np.ndarray, values: np.ndarray) -> float:
    reconstructed = np.asarray(base_values).reshape(-1) + np.asarray(values).sum(axis=1)
    return float(np.max(np.abs(np.asarray(scores).reshape(-1) - reconstructed)))


def run_shap_analysis(
    data_path: str | Path | None = None,
    model_path: str | Path | None = None,
    policy_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    background_size: int = 100,
    legitimate_sample_size: int = 300,
    random_seed: int = 42,
) -> dict[str, Any]:
    project_root = PROJECT_ROOT
    config = load_config()
    data_path = Path(data_path or project_root / config["data"]["raw_path"])
    model_path = Path(model_path or project_root / "models" / "xgboost_best.joblib")
    policy_path = Path(policy_path or project_root / "models" / "decision_policy.json")
    output_dir = Path(output_dir or project_root / "reports" / "explainability")
    output_dir.mkdir(parents=True, exist_ok=True)
    (project_root / "models").mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Trained XGBoost model not found: {model_path}. Run Phase 5 first.")
    if not policy_path.exists():
        raise FileNotFoundError(f"Decision policy not found: {policy_path}. Run Phase 6 first.")
    if background_size < 10:
        raise ValueError("background_size must be at least 10 for a useful reference sample")
    if legitimate_sample_size < 10:
        raise ValueError("legitimate_sample_size must be at least 10")

    df = pd.read_csv(data_path)
    target = config["data"].get("target_column", "Class")
    splits = chronological_train_validation_test_split(
        df, time_column="Time", train_fraction=0.60, validation_fraction=0.20
    )
    model = joblib.load(model_path)
    feature_names = list(getattr(model, "feature_names_in_", []))
    if not feature_names:
        feature_names = [col for col in df.columns if col != target]
    missing = sorted(set(feature_names) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset missing model features: {missing}")

    fit_data = pd.concat([splits.train, splits.validation], ignore_index=True)
    X_fit = fit_data[feature_names].astype(float)
    X_test = splits.test[feature_names].reset_index(drop=True).astype(float)
    y_test = splits.test[target].astype(int).reset_index(drop=True).to_numpy()
    scores = model.predict_proba(X_test)[:, 1]
    with Path(policy_path).open("r", encoding="utf-8") as f:
        policy = json.load(f)
    threshold = float(policy["decision"]["threshold"])

    # Persist a compact, deterministic reference set from the model-fit period.
    bg_n = min(background_size, len(X_fit))
    background = X_fit.sample(n=bg_n, random_state=random_seed).reset_index(drop=True)
    background_path = project_root / "models" / "shap_background.csv"
    background.to_csv(background_path, index=False)

    fraud_idx = np.flatnonzero(y_test == 1)
    legitimate_idx = np.flatnonzero(y_test == 0)
    if len(fraud_idx) < 1 or len(legitimate_idx) < 1:
        raise ValueError("Test split must contain both legitimate and fraudulent examples")
    rng = np.random.default_rng(random_seed)
    n_legit = min(legitimate_sample_size, len(legitimate_idx))
    sampled_legit = rng.choice(legitimate_idx, size=n_legit, replace=False)

    flagged = scores >= threshold
    tp = np.flatnonzero((y_test == 1) & flagged)
    fp = np.flatnonzero((y_test == 0) & flagged)
    fn = np.flatnonzero((y_test == 1) & ~flagged)
    local_indices: dict[str, int] = {}
    if len(tp):
        local_indices["true_positive"] = int(tp[np.argmax(scores[tp])])
    if len(fp):
        local_indices["false_positive"] = int(fp[np.argmax(scores[fp])])
    if len(fn):
        # Show the missed-fraud case closest to the threshold, which is most
        # informative for illustrating a decision boundary miss.
        local_indices["false_negative"] = int(fn[np.argmax(scores[fn])])

    explain_indices = sorted(set(map(int, fraud_idx)) | set(map(int, sampled_legit)) | set(local_indices.values()))
    X_explain = X_test.iloc[explain_indices].reset_index(drop=True)
    y_explain = y_test[np.asarray(explain_indices, dtype=int)]
    score_explain = scores[np.asarray(explain_indices, dtype=int)]

    explainer = shap.TreeExplainer(
        model,
        data=background,
        feature_perturbation="interventional",
        model_output="probability",
    )
    explanation = explainer(X_explain)
    shap_values = np.asarray(explanation.values, dtype=float)
    base_values = np.asarray(explanation.base_values, dtype=float).reshape(-1)
    if shap_values.shape != X_explain.shape:
        raise RuntimeError(f"Unexpected SHAP output shape {shap_values.shape}; expected {X_explain.shape}")
    max_additivity_error = _ensure_additivity(score_explain, base_values, shap_values)
    if max_additivity_error > 5e-4:
        raise RuntimeError(f"SHAP additivity check failed; maximum error={max_additivity_error:.6g}")

    # The interpretability subset is case-enriched (all fraud + sampled
    # legitimate records) to illuminate the rare positive class. Its global
    # mean-absolute SHAP values are not population-prevalence-weighted.
    global_importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap_value": np.abs(shap_values).mean(axis=0),
        "mean_shap_value": shap_values.mean(axis=0),
        "fraction_positive_contribution": (shap_values > 0).mean(axis=0),
        "fraction_negative_contribution": (shap_values < 0).mean(axis=0),
    }).sort_values("mean_abs_shap_value", ascending=False)
    global_importance.to_csv(output_dir / "global_shap_importance.csv", index=False)
    pd.DataFrame(shap_values, columns=feature_names).assign(
        test_position=explain_indices,
        actual_class=y_explain,
        model_score=score_explain,
    ).to_csv(output_dir / "sample_shap_values.csv", index=False)

    # Global bar plot.
    plt.figure(figsize=(9, 7))
    shap.plots.bar(explanation, max_display=20, show=False)
    plt.title("Global SHAP importance — case-enriched test sample")
    plt.tight_layout()
    plt.savefig(output_dir / "global_shap_bar.png", dpi=170, bbox_inches="tight")
    plt.close("all")

    # Beeswarm plot: direction and distribution of feature contributions.
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(explanation, max_display=20, show=False)
    plt.title("SHAP summary — case-enriched test sample")
    plt.tight_layout()
    plt.savefig(output_dir / "global_shap_beeswarm.png", dpi=170, bbox_inches="tight")
    plt.close("all")

    # Save local, individual explanations for one TP, FP, and FN where present.
    inverse_lookup = {test_pos: sample_pos for sample_pos, test_pos in enumerate(explain_indices)}
    local_results: dict[str, Any] = {}
    local_plot_dir = output_dir / "local"
    local_plot_dir.mkdir(parents=True, exist_ok=True)
    local_rows: list[dict[str, Any]] = []
    for case_name, test_pos in local_indices.items():
        sample_pos = inverse_lookup[test_pos]
        row_exp = explanation[sample_pos]
        row_values = shap_values[sample_pos]
        row_features = X_test.iloc[test_pos]
        score = float(scores[test_pos])
        base_value = float(base_values[sample_pos])
        ranked = [
            {
                "feature": feature_names[i],
                "feature_value": float(row_features.iloc[i]),
                "shap_value": float(row_values[i]),
                "direction": "increases_model_score" if row_values[i] > 0 else (
                    "decreases_model_score" if row_values[i] < 0 else "neutral"
                ),
            }
            for i in range(len(feature_names))
        ]
        top = sorted(ranked, key=lambda x: abs(x["shap_value"]), reverse=True)[:10]
        local_results[case_name] = {
            "test_position": int(test_pos),
            "actual_class": int(y_test[test_pos]),
            "model_score": score,
            "decision_threshold": threshold,
            "risk_level": _risk_level(score, policy),
            "decision": policy["decision"]["flag_label"] if score >= threshold else policy["decision"]["not_flagged_label"],
            "shap_base_value": base_value,
            "shap_reconstructed_score": float(base_value + row_values.sum()),
            "top_features_by_absolute_contribution": top,
            "note": "Feature contributions explain the model output relative to its background; they are not causal claims.",
        }
        for feature in top:
            local_rows.append({"case": case_name, **{k: v for k, v in local_results[case_name].items() if k in ["test_position", "actual_class", "model_score", "decision_threshold", "risk_level", "decision"]}, **feature})
        plt.figure(figsize=(9, 6))
        shap.plots.waterfall(row_exp, max_display=15, show=False)
        plt.title(f"Local SHAP explanation — {case_name.replace('_', ' ').title()}")
        plt.tight_layout()
        plt.savefig(local_plot_dir / f"waterfall_{case_name}.png", dpi=170, bbox_inches="tight")
        plt.close("all")

    with (output_dir / "local_explanations.json").open("w", encoding="utf-8") as f:
        json.dump(local_results, f, indent=2)
    pd.DataFrame(local_rows).to_csv(output_dir / "local_top_contributions.csv", index=False)

    summary = {
        "model_artifact": str(model_path),
        "policy_artifact": str(policy_path),
        "background_artifact": str(background_path),
        "explainer": "shap.TreeExplainer",
        "model_output": "probability",
        "background_rows": int(len(background)),
        "test_rows_in_explanation_sample": int(len(X_explain)),
        "test_fraud_rows_in_explanation_sample": int((y_explain == 1).sum()),
        "test_legitimate_rows_in_explanation_sample": int((y_explain == 0).sum()),
        "sampling_note": "Interpretability sample contains all test fraud cases and a random sample of legitimate transactions; global mean-absolute SHAP values are case-enriched, not prevalence-weighted population estimates.",
        "decision_threshold_from_phase6_policy": threshold,
        "max_absolute_shap_additivity_error": max_additivity_error,
        "global_top_10_features": global_importance.head(10).to_dict(orient="records"),
        "local_cases": local_results,
    }
    with (output_dir / "shap_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_findings(summary, global_importance, output_dir)
    return summary


def _risk_level(score: float, policy: dict[str, Any]) -> str:
    bands = policy["risk_bands"]
    if score < float(bands["LOW"]["upper_exclusive"]):
        return "LOW"
    if score < float(bands["MEDIUM"]["upper_exclusive"]):
        return "MEDIUM"
    if score < float(bands["HIGH"]["upper_exclusive"]):
        return "HIGH"
    return "CRITICAL"


def write_findings(summary: dict[str, Any], importance: pd.DataFrame, output_dir: Path) -> None:
    rows = [
        "# Phase 7 — Explainable AI with SHAP",
        "",
        "## What this phase does",
        "",
        "SHAP (SHapley Additive exPlanations) attributes a model output to input features relative to a chosen background dataset. Here, `shap.TreeExplainer` explains the saved XGBoost model's `predict_proba` output on the probability-output scale.",
        "",
        "SHAP values explain model behavior; they do not prove causation, validate calibration, or recover the real-world business meaning of anonymized PCA components `V1`–`V28`.",
        "",
        "## Reproducibility and sample design",
        "",
        f"- Background rows: {summary['background_rows']}",
        f"- Explanation rows: {summary['test_rows_in_explanation_sample']}",
        f"- Fraud cases in explanation sample: {summary['test_fraud_rows_in_explanation_sample']}",
        f"- Legitimate cases in explanation sample: {summary['test_legitimate_rows_in_explanation_sample']}",
        f"- Maximum absolute SHAP additivity error: {summary['max_absolute_shap_additivity_error']:.8g}",
        "",
        "The interpretability sample includes all fraud cases from the historical test partition plus a deterministic random sample of legitimate cases. This case-enriched design makes minority-class explanations visible, but global mean-absolute SHAP values are **not** prevalence-weighted population estimates. The model and threshold are unchanged by this analysis.",
        "",
        "## Global explanations",
        "",
        "`global_shap_importance.csv` ranks features by mean absolute SHAP contribution over the interpretability sample. `global_shap_bar.png` visualizes global magnitude. `global_shap_beeswarm.png` shows contribution direction and spread per feature. These are associations with the model's output, not causal claims.",
        "",
        "### Top 10 features by mean absolute SHAP",
        "",
        "| Rank | Feature | Mean absolute SHAP | Mean SHAP |",
        "|---:|---|---:|---:|",
    ]
    for i, row in enumerate(importance.head(10).itertuples(index=False), 1):
        rows.append(f"| {i} | {row.feature} | {row.mean_abs_shap_value:.8f} | {row.mean_shap_value:.8f} |")
    rows += [
        "",
        "## Local explanations",
        "",
        "Waterfall plots are generated for a true positive, a false positive, and a false negative where these cases exist under the Phase 6 threshold. The true positive shows why the model scored a correctly detected fraud highly; the false positive is a legitimate transaction incorrectly flagged; the false negative is a fraud case the decision threshold did not flag.",
        "",
        "Files: `local_explanations.json`, `local_top_contributions.csv`, and `local/waterfall_*.png`.",
        "",
        "## Explanation API",
        "",
        "Run `uvicorn api.main:app --reload` and send a `POST /explain` request with all 30 numeric model features (`Time`, `V1`–`V28`, and `Amount`). The response includes the model score, Phase 6 risk tier and decision, plus the largest positive and negative SHAP contributions.",
        "",
        "The returned `model_score` is an uncalibrated XGBoost score and must not be interpreted as a verified real-world fraud probability. Avoid exposing the API publicly without authentication, rate limits, request logging policies, and deployment security controls.",
        "",
        "## Re-run",
        "",
        "```powershell",
        "python -m src.evaluation.shap_explanations",
        "python -m pytest -q",
        "```",
        "",
    ]
    (output_dir / "PHASE7_SHAP_FINDINGS.md").write_text("\n".join(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--background-size", type=int, default=100)
    parser.add_argument("--legitimate-sample-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run_shap_analysis(
        background_size=args.background_size,
        legitimate_sample_size=args.legitimate_sample_size,
        random_seed=args.seed,
    )
    print(json.dumps({
        "status": "completed",
        "background_rows": result["background_rows"],
        "explanation_rows": result["test_rows_in_explanation_sample"],
        "max_additivity_error": result["max_absolute_shap_additivity_error"],
        "top_features": result["global_top_10_features"][:5],
        "local_cases": {k: {"model_score": v["model_score"], "actual_class": v["actual_class"], "risk_level": v["risk_level"], "decision": v["decision"]} for k, v in result["local_cases"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
