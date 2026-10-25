"""SHAP-based explanations and inference helpers for FraudShield XGBoost.

SHAP values explain the fitted model output relative to a background dataset.
They are not causal explanations, and model scores are not guaranteed to be
calibrated real-world fraud probabilities.
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
import sklearn
import xgboost

from src.config import PROJECT_ROOT


class FraudShapService:
    """Load model artifacts once and provide predictions and optional SHAP data."""

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
                    f"Required artifact not found: {path}. Ensure Phase 5–7 model artifacts exist."
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
        if self.background.empty:
            raise ValueError("SHAP background dataset is empty.")

        with self.policy_path.open("r", encoding="utf-8") as f:
            self.policy = json.load(f)
        try:
            self.threshold = float(self.policy["decision"]["threshold"])
            self.policy_version = str(self.policy.get("policy_version", "unknown"))
            if not 0.0 <= self.threshold <= 1.0:
                raise ValueError("Decision threshold must be between 0 and 1.")
        except (KeyError, TypeError) as exc:
            raise ValueError("Decision policy is missing a valid decision threshold.") from exc

        # SHAP can be relatively expensive to initialize. Build it only when an
        # explanation is requested; score-only API calls stay lightweight.
        self._explainer = None

    @property
    def explainer(self):
        """Build the TreeExplainer lazily and reuse it for subsequent requests."""
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(
                self.model,
                data=self.background,
                feature_perturbation="interventional",
                model_output="probability",
            )
        return self._explainer

    def validate_features(self, features: Mapping[str, Any]) -> pd.DataFrame:
        """Validate complete feature schema and preserve training feature order."""
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

    def _prediction_from_row(self, row: pd.DataFrame) -> dict[str, Any]:
        score = float(self.model.predict_proba(row)[:, 1][0])
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise RuntimeError("Model returned an invalid prediction score.")
        predicted_fraud = bool(score >= self.threshold)
        return {
            "model_version": self.model_path.stem,
            "policy_version": self.policy_version,
            "model_type": type(self.model).__name__,
            "feature_count": len(self.feature_names),
            "model_score": score,
            "score_interpretation": (
                "Uncalibrated XGBoost output score; not a verified real-world probability of fraud."
            ),
            "risk_score_0_100": 100.0 * score,
            "decision_threshold": self.threshold,
            "risk_level": self.risk_level(score),
            "predicted_fraud": predicted_fraud,
            "decision": (
                self.policy["decision"]["flag_label"]
                if predicted_fraud
                else self.policy["decision"]["not_flagged_label"]
            ),
        }

    def predict(self, features: Mapping[str, Any]) -> dict[str, Any]:
        """Run the validated prediction pipeline without generating SHAP values."""
        row = self.validate_features(features)
        return self._prediction_from_row(row)

    def explain(self, features: Mapping[str, Any], top_k: int = 5) -> dict[str, Any]:
        """Return prediction plus ranked positive/negative SHAP contributions."""
        if not 1 <= top_k <= len(self.feature_names):
            raise ValueError(f"top_k must be between 1 and {len(self.feature_names)}")
        row = self.validate_features(features)
        result = self._prediction_from_row(row)

        explanation = self.explainer(row)
        values = np.asarray(explanation.values, dtype=float).reshape(-1)
        base_value = float(np.asarray(explanation.base_values, dtype=float).reshape(-1)[0])
        contribution_sum = float(base_value + values.sum())
        additive_error = float(result["model_score"] - contribution_sum)
        if not np.isfinite(values).all() or not np.isfinite([base_value, result["model_score"]]).all():
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

        result.update({
            "shap_available": True,
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
        })
        return result

    def model_info(self) -> dict[str, Any]:
        """Return safe, relevant metadata without exposing filesystem paths."""
        return {
            "model_version": self.model_path.stem,
            "model_type": type(self.model).__name__,
            "policy_version": self.policy_version,
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "decision_threshold": self.threshold,
            "risk_bands": self.policy.get("risk_bands", {}),
            "shap_available": True,
            "score_semantics": (
                "Uncalibrated XGBoost predict_proba output score; do not interpret as a calibrated "
                "real-world fraud probability."
            ),
            "libraries": {
                "xgboost": xgboost.__version__,
                "scikit_learn": sklearn.__version__,
                "shap": shap.__version__,
            },
        }
