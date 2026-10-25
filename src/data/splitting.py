"""Leakage-aware chronological splitting for FraudShield AI."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class DatasetSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    cutoffs: dict[str, Any]


def _advance_to_timestamp_boundary(time_values: pd.Series, split_index: int) -> int:
    """Move a row split forward so equal timestamps are not split across partitions."""
    n_rows = len(time_values)
    split_index = min(max(int(split_index), 1), n_rows - 1)
    while split_index < n_rows and time_values.iloc[split_index] == time_values.iloc[split_index - 1]:
        split_index += 1
    return split_index


def chronological_train_validation_test_split(
    df: pd.DataFrame,
    *,
    time_column: str = "Time",
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> DatasetSplits:
    """Split by time into earlier training, middle validation, and later test data.

    The input is stably sorted by time. Split points are adjusted to avoid placing
    records with the same timestamp in different partitions. Equal timestamps
    can include exact duplicate rows, so this also helps prevent exact duplicate
    records from crossing partition boundaries for this dataset.
    """
    if time_column not in df.columns:
        raise ValueError(f"Missing time column: {time_column}")
    if df[time_column].isna().any():
        raise ValueError(f"'{time_column}' contains missing values; resolve before splitting")
    if not (0 < train_fraction < 1):
        raise ValueError("train_fraction must be between 0 and 1")
    if not (0 < validation_fraction < 1):
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be less than 1")
    if len(df) < 10:
        raise ValueError("At least 10 rows are required for a three-way split")

    ordered = df.sort_values(time_column, kind="mergesort").reset_index(drop=True)
    n_rows = len(ordered)
    train_end = _advance_to_timestamp_boundary(ordered[time_column], round(n_rows * train_fraction))
    validation_end = _advance_to_timestamp_boundary(
        ordered[time_column], round(n_rows * (train_fraction + validation_fraction))
    )
    if validation_end <= train_end or validation_end >= n_rows:
        raise ValueError("Unable to form non-empty chronological partitions; inspect timestamp distribution")

    train = ordered.iloc[:train_end].copy()
    validation = ordered.iloc[train_end:validation_end].copy()
    test = ordered.iloc[validation_end:].copy()

    # Hard assertions make split mistakes fail early.
    if train[time_column].max() >= validation[time_column].min():
        raise AssertionError("Training and validation time ranges overlap")
    if validation[time_column].max() >= test[time_column].min():
        raise AssertionError("Validation and test time ranges overlap")

    cutoffs = {
        "train_fraction_actual": len(train) / n_rows,
        "validation_fraction_actual": len(validation) / n_rows,
        "test_fraction_actual": len(test) / n_rows,
        "train_time_max": float(train[time_column].max()),
        "validation_time_min": float(validation[time_column].min()),
        "validation_time_max": float(validation[time_column].max()),
        "test_time_min": float(test[time_column].min()),
    }
    return DatasetSplits(train=train, validation=validation, test=test, cutoffs=cutoffs)
