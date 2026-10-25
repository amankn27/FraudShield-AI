"""SHAP-based explanations for the FraudShield XGBoost model.

SHAP values explain the trained model's output relative to a reference
background dataset. They are not causal explanations and do not establish the
real-world probability or cause of fraud.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import shap

from src.config import PROJECT_ROOT


class FraudShapService:
    """Load the production model, policy, SHAP background, and explain rows."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        background_path: str | Path | None = None,
        policy_path: str | Path | None = None,
    ) -> None:
        self.model_path = Path(model_path or PROJECT_ROOT / "models" / "xgboost_best.joblib")
        self.background_path = Path(background_path or PROJECT_ROOT / "models" / "shap_background.csv")
        self.policy_path = Path(policy_path or PROJECT_ROOT / "models" / "decision_policy.json")
        for path in (self.model_path, self.background_path, self.policy_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Required artifact not found: {path}. Run `python -m src.evaluation.shap_explanations` first."
                )

        self.model = joblib.load(self.model_path)
        self.feature_names = list(getattr(self.model, "feature_names_in_", []))
        if not self.feature_names:
            raise ValueError("Saved XGBoost model does not expose feature_names_in_.")
        self.background = pd.read_csv(self.background_path)
        missing_background = sorted(set(self.feature_names) - set(self.background.columns))
        if missing_background:
            raise ValueError(f"SHAP background missing model features: {missing_background}")
        self.background = self.background[self.feature_names].astype(float)
        with self.policy_path.open("r", encoding="utf-8") as f:
            self.policy = json.load(f)

        # Probability-output SHAP values are additive to the model's output
        # score. The score is not necessarily calibrated as a real-world risk.
        self.explainer = shap.TreeExplainer(
            self.model,
            data=self.background,
            feature_perturbation="interventional",
            model_output="probability",
        )
        self.threshold = float(self.policy["decision"]["threshold"])

    def validate_features(self, features: Mapping[str, Any]) -> pd.DataFrame:
        missing = sorted(set(self.feature_names) - set(features))
        extras = sorted(set(features) - set(self.feature_names))
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        if extras:
            raise ValueError(f"Unexpected features: {extras}")
        values: dict[str, float] = {}
        for name in self.feature_names:
            try:
                value = float(features[name])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Feature {name!r} must be numeric") from exc
            if not math.isfinite(value):
                raise ValueError(f"Feature {name!r} must be finite")
            values[name] = value
        return pd.DataFrame([values], columns=self.feature_names)

    def risk_level(self, score: float) -> str:
        bands = self.policy["risk_bands"]
        if score < float(bands["LOW"]["upper_exclusive"]):
            return "LOW"
        if score < float(bands["MEDIUM"]["upper_exclusive"]):
            return "MEDIUM"
        if score < float(bands["HIGH"]["upper_exclusive"]):
            return "HIGH"
        return "CRITICAL"

    def explain(self, features: Mapping[str, Any], top_k: int = 5) -> dict[str, Any]:
        """Return score, policy decision, and ranked SHAP contributions."""
        if not 1 <= top_k <= len(self.feature_names):
            raise ValueError(f"top_k must be between 1 and {len(self.feature_names)}")
        row = self.validate_features(features)
        score = float(self.model.predict_proba(row)[:, 1][0])
        explanation = self.explainer(row)
        values = np.asarray(explanation.values, dtype=float).reshape(-1)
        base_value = float(np.asarray(explanation.base_values, dtype=float).reshape(-1)[0])
        contribution_sum = float(base_value + values.sum())
        additive_error = float(score - contribution_sum)
        if not np.isfinite(values).all() or not np.isfinite([base_value, score]).all():
            raise RuntimeError("SHAP returned non-finite explanation values")

        contributions = [
            {
                "feature": name,
                "feature_value": float(row.iloc[0][name]),
                "shap_value": float(values[i]),
                "direction": "increases_model_score" if values[i] > 0 else (
                    "decreases_model_score" if values[i] < 0 else "neutral"
                ),
            }
            for i, name in enumerate(self.feature_names)
        ]
        positive = sorted(
            (item for item in contributions if item["shap_value"] > 0),
            key=lambda item: item["shap_value"], reverse=True,
        )[:top_k]
        negative = sorted(
            (item for item in contributions if item["shap_value"] < 0),
            key=lambda item: item["shap_value"],
        )[:top_k]

        return {
            "model_version": self.model_path.stem,
            "policy_version": self.policy.get("policy_version", "unknown"),
            "model_score": score,
            "score_interpretation": (
                "Uncalibrated XGBoost output score; not a verified real-world probability of fraud."
            ),
            "risk_score_0_100": 100.0 * score,
            "decision_threshold": self.threshold,
            "risk_level": self.risk_level(score),
            "decision": (
                self.policy["decision"]["flag_label"]
                if score >= self.threshold
                else self.policy["decision"]["not_flagged_label"]
            ),
            "shap_output_space": "model probability output scale; contributions sum to model_score",
            "shap_base_value": base_value,
            "shap_reconstructed_score": contribution_sum,
            "shap_additivity_error": additive_error,
            "top_positive_contributions": positive,
            "top_negative_contributions": negative,
            "interpretation_note": (
                "SHAP describes how input features contribute to this model output relative to its background; "
                "it does not prove causation or identify the real-world meaning of anonymized PCA features."
            ),
        }
