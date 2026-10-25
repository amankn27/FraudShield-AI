"""Unit tests for Phase 9 hybrid score transformations and metrics."""
import numpy as np
import pandas as pd

from src.models.hybrid_system import _logit_transform, _metrics, _normalize_signals, _threshold_index


def test_logit_transform_is_finite_and_monotonic():
    scores = _logit_transform(np.array([0.0, 0.5, 1.0]))
    assert np.isfinite(scores).all()
    assert scores[0] < scores[1] < scores[2]


def test_signal_normalization_preserves_feature_shape_and_order():
    xgb = np.array([0.1, 0.5, 0.9])
    anomaly = np.array([0.0, 1.0, 10.0])
    signals = _normalize_signals(xgb, anomaly)
    assert signals.shape == (3, 2)
    assert np.all(np.diff(signals[:, 0]) > 0)
    assert np.all(np.diff(signals[:, 1]) > 0)


def test_metrics_counts_and_average_precision():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.8, 0.9, 0.95])
    result = _metrics(y, scores, threshold=0.75)
    assert result["true_negative"] == 1
    assert result["false_positive"] == 1
    assert result["false_negative"] == 0
    assert result["true_positive"] == 2
    assert result["recall"] == 1.0
    assert 0.0 <= result["average_precision"] <= 1.0


def test_threshold_index_does_not_split_equal_timestamp_group():
    times = pd.Series([1, 1, 2, 2, 2, 3, 4, 4, 5, 6])
    idx = _threshold_index(times, 0.4)
    assert idx > 0 and idx < len(times)
    assert times.iloc[idx - 1] != times.iloc[idx]


def test_threshold_index_rejects_empty_tail():
    times = pd.Series([1, 2, 3, 4])
    try:
        _threshold_index(times, 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected a ValueError for an empty tail split")
