import numpy as np
import pandas as pd
import pytest

from src.evaluation.decision_engine import (
    assign_risk_level,
    choose_cost_threshold,
    evaluate_threshold,
    threshold_for_target_recall,
    threshold_sweep,
)


def test_threshold_sweep_counts_and_cost():
    y = np.array([0, 1, 1, 0])
    scores = np.array([0.1, 0.9, 0.8, 0.2])
    sweep = threshold_sweep(y, scores, cost_fp=1, cost_fn=10)
    at_08 = sweep[np.isclose(sweep["threshold"], 0.8)].iloc[0]
    assert at_08["true_positive"] == 2
    assert at_08["false_positive"] == 0
    assert at_08["false_negative"] == 0
    assert at_08["total_cost_units"] == 0


def test_threshold_sweep_includes_no_alert_policy():
    y = np.array([0, 1, 0, 1])
    scores = np.array([0.2, 0.6, 0.4, 0.8])
    sweep = threshold_sweep(y, scores, cost_fp=1, cost_fn=10)
    no_alert = sweep.iloc[0]
    assert no_alert["true_positive"] == 0
    assert no_alert["false_positive"] == 0
    assert no_alert["false_negative"] == 2
    assert no_alert["total_cost_units"] == 20


def test_cost_threshold_selects_minimum_cost():
    y = np.array([0, 1, 1, 0])
    scores = np.array([0.1, 0.9, 0.8, 0.2])
    sweep = threshold_sweep(y, scores, cost_fp=1, cost_fn=10)
    chosen = choose_cost_threshold(sweep)
    assert chosen["total_cost_units"] == sweep["total_cost_units"].min()


def test_target_recall_threshold():
    y = np.array([0, 1, 1, 0])
    scores = np.array([0.1, 0.9, 0.8, 0.2])
    sweep = threshold_sweep(y, scores, cost_fp=1, cost_fn=10)
    chosen = threshold_for_target_recall(sweep, 1.0)
    assert chosen["recall"] == 1.0
    assert chosen["threshold"] == 0.8


def test_risk_bands_are_ordered_and_complete():
    scores = np.array([0.01, 0.1, 0.4, 0.9])
    bands = assign_risk_level(scores, 0.05, 0.3, 0.8)
    assert bands.tolist() == ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def test_risk_bands_reject_unordered_thresholds():
    with pytest.raises(ValueError):
        assign_risk_level(np.array([0.1]), 0.4, 0.3, 0.8)


def test_evaluate_threshold_separates_score_from_classification():
    y = np.array([0, 1, 0, 1])
    scores = np.array([0.2, 0.8, 0.3, 0.7])
    m = evaluate_threshold(y, scores, threshold=0.75, cost_fp=1, cost_fn=5)
    assert m["true_positive"] == 1
    assert m["false_negative"] == 1
    assert m["false_positive"] == 0
    assert m["precision"] == 1.0
    assert m["recall"] == 0.5
