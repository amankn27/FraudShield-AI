"""Validate the downloaded FraudShield CSV without modifying it.

Run from the repository root:
    python -m src.data.validate_dataset

Optional path override:
    python -m src.data.validate_dataset --data-path data/raw/creditcard.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.config import load_config, resolve_data_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the FraudShield transaction dataset.")
    parser.add_argument(
        "--data-path",
        default=None,
        help="Optional CSV path. Relative paths are resolved from the project root.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        config = load_config()
        data_path = resolve_data_path(config, args.data_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"CONFIGURATION ERROR: {exc}", file=sys.stderr)
        return 2

    if not data_path.is_file():
        print(f"DATASET NOT FOUND: {data_path}", file=sys.stderr)
        print(
            "Download creditcard.csv from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            "and place it in data/raw/.",
            file=sys.stderr,
        )
        return 2

    print(f"Loading dataset: {data_path}")
    try:
        df = pd.read_csv(data_path)
    except Exception as exc:  # Pandas may raise several parsing/I/O exception types.
        print(f"CSV READ ERROR: {exc}", file=sys.stderr)
        return 2

    data_config = config.get("data", {})
    expected_columns = data_config.get("expected_columns", [])
    target_column = data_config.get("target_column", "Class")
    errors: list[str] = []
    warnings: list[str] = []

    print("\n=== Dataset overview ===")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MiB")
    print(f"Duplicate rows: {int(df.duplicated().sum()):,}")

    duplicated_columns = df.columns[df.columns.duplicated()].tolist()
    if duplicated_columns:
        errors.append(f"Duplicate column names: {duplicated_columns}")

    missing_columns = [column for column in expected_columns if column not in df.columns]
    extra_columns = [column for column in df.columns if column not in expected_columns]
    if missing_columns:
        errors.append(f"Missing expected columns: {missing_columns}")
    if extra_columns:
        warnings.append(f"Unexpected columns found: {extra_columns}")

    if target_column not in df.columns:
        errors.append(f"Target column '{target_column}' is missing.")
    else:
        if not pd.api.types.is_numeric_dtype(df[target_column]):
            errors.append(f"Target column '{target_column}' must be numeric.")
        else:
            target_values = set(df[target_column].dropna().unique().tolist())
            allowed_values = {
                data_config.get("negative_class", 0),
                data_config.get("positive_class", 1),
            }
            unexpected_labels = target_values - allowed_values
            if unexpected_labels:
                errors.append(f"Unexpected target labels: {sorted(unexpected_labels)}")
            if df[target_column].isna().any():
                errors.append(f"Target column '{target_column}' contains missing values.")
            if not df[target_column].dropna().isin(allowed_values).all():
                errors.append("Target labels must be binary 0/1.")

            print("\n=== Target distribution ===")
            counts = df[target_column].value_counts(dropna=False).sort_index()
            total = len(df)
            for label, count in counts.items():
                pct = (count / total * 100) if total else 0.0
                print(f"Class {label}: {count:,} ({pct:.4f}%)")
            if len(target_values) < 2:
                warnings.append("Only one target class appears in this file; verify that the correct dataset was downloaded.")

    missing_counts = df.isna().sum()
    columns_with_missing = missing_counts[missing_counts > 0]
    print("\n=== Missing values ===")
    if columns_with_missing.empty:
        print("No missing values found.")
    else:
        for column, count in columns_with_missing.items():
            print(f"{column}: {count:,}")
        warnings.append(f"Missing values detected in {len(columns_with_missing)} column(s); investigate before training.")

    non_numeric = [column for column in df.columns if not pd.api.types.is_numeric_dtype(df[column])]
    if non_numeric:
        warnings.append(f"Non-numeric columns detected: {non_numeric}")

    duplicate_count = int(df.duplicated().sum())
    if duplicate_count:
        warnings.append(
            f"{duplicate_count:,} duplicate row(s) detected. Do not drop them automatically; inspect them first."
        )

    print("\n=== Checks ===")
    print(f"Expected columns configured: {len(expected_columns)}")
    print(f"Missing expected columns: {len(missing_columns)}")
    print(f"Unexpected columns: {len(extra_columns)}")
    print(f"Non-numeric columns: {non_numeric if non_numeric else 'None'}")

    if warnings:
        print("\n=== Warnings ===")
        for warning in warnings:
            print(f"WARNING: {warning}")

    if errors:
        print("\n=== Validation failed ===", file=sys.stderr)
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print("\n=== Validation result ===")
    print("PASS: required schema and target checks completed.")
    if warnings:
        print("Review the warnings before using this data for model training.")
    else:
        print("No warnings detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
