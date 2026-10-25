"""Classification metrics for the FraudShield AI baseline."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, probabilities, *, threshold: float = 0.5) -> tuple[dict[str, Any], np.ndarray]:
    """Compute fixed-threshold metrics plus threshold-independent ranking metrics.

    PR-AUC is reported both as Average Precision (AP) and as trapezoidal area,
    since these two conventions are sometimes conflated in ML reports.
    """
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(y_true) != len(probabilities):
        raise ValueError("y_true and probabilities must have the same length")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities contains NaN or infinite values")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    predictions = (probabilities >= threshold).astype(int)
    cm = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision, recall, _ = precision_recall_curve(y_true, probabilities)

    metrics: dict[str, Any] = {
        "threshold": float(threshold),
        "n_samples": int(len(y_true)),
        "n_positive_actual": int((y_true == 1).sum()),
        "n_negative_actual": int((y_true == 0).sum()),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) == 2 else None,
        "pr_auc_average_precision": float(average_precision_score(y_true, probabilities))
            if len(np.unique(y_true)) == 2 else None,
        "pr_auc_trapezoidal": float(auc(recall, precision)) if len(np.unique(y_true)) == 2 else None,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
    return metrics, cm
