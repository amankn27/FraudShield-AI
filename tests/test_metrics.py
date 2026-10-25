import numpy as np

from src.evaluation.metrics import classification_metrics


def test_metrics_confusion_counts_and_fixed_threshold():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.10, 0.70, 0.80, 0.20])
    metrics, cm = classification_metrics(y, p, threshold=0.5)
    assert cm.tolist() == [[1, 1], [1, 1]]
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert 0 <= metrics["roc_auc"] <= 1
    assert 0 <= metrics["pr_auc_average_precision"] <= 1
