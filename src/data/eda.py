\
"""Exploratory data analysis for FraudShield AI.

Run from the repository root:
    python -m src.data.eda

Optional overrides:
    python -m src.data.eda --data-path data/raw/creditcard.csv --output-dir reports/eda

This script reads the raw CSV without changing it and writes tables/plots to reports/eda.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Save figures reliably in scripts and headless environments.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_config, resolve_data_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exploratory data analysis for FraudShield AI.")
    parser.add_argument("--data-path", default=None, help="Optional CSV path; relative paths use the project root.")
    parser.add_argument("--output-dir", default="reports/eda", help="Output directory; relative paths use the project root.")
    return parser.parse_args()


def resolve_output_dir(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def safe_float(value: Any) -> float | None:
    """Convert numerical values to JSON-safe floats, replacing non-finite values with None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def validate_columns(df: pd.DataFrame, expected_columns: list[str], target: str) -> None:
    missing = [column for column in expected_columns if column not in df.columns]
    unexpected = [column for column in df.columns if column not in expected_columns]
    duplicated_names = df.columns[df.columns.duplicated()].tolist()
    if duplicated_names:
        raise ValueError(f"CSV has duplicate column names: {duplicated_names}")
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' is missing.")
    if unexpected:
        print(f"WARNING: unexpected columns will be present in the EDA: {unexpected}")


def build_duplicate_summary(df: pd.DataFrame, target: str) -> pd.DataFrame:
    feature_columns = [column for column in df.columns if column != target]
    full_extra = int(df.duplicated().sum())
    feature_extra = int(df.duplicated(subset=feature_columns).sum())
    feature_mask = df.duplicated(subset=feature_columns, keep=False)
    rows_in_feature_duplicate_groups = int(feature_mask.sum())
    conflicting_label_groups = 0
    if feature_extra and target in df.columns:
        duplicate_candidates = df.loc[feature_mask, feature_columns + [target]]
        group_label_counts = duplicate_candidates.groupby(
            feature_columns, dropna=False, sort=False
        )[target].nunique(dropna=False)
        conflicting_label_groups = int((group_label_counts > 1).sum())
    return pd.DataFrame([
        {"check": "exact_duplicate_rows_extra_copies", "count": full_extra,
         "meaning": "Rows identical across every column, excluding each group's first occurrence."},
        {"check": "duplicate_feature_rows_extra_copies", "count": feature_extra,
         "meaning": "Rows identical across input features while ignoring the target label."},
        {"check": "rows_in_duplicate_feature_groups", "count": rows_in_feature_duplicate_groups,
         "meaning": "All rows belonging to groups with identical input features."},
        {"check": "duplicate_feature_groups_with_conflicting_labels", "count": conflicting_label_groups,
         "meaning": "Identical input-feature groups that contain more than one target label."},
    ])


def class_distribution(df: pd.DataFrame, target: str) -> pd.DataFrame:
    counts = df[target].value_counts(dropna=False).sort_index()
    total = len(df)
    table = pd.DataFrame({"class": counts.index.astype(str), "count": counts.values})
    table["percentage"] = table["count"] / total * 100 if total else 0.0
    return table


def make_class_plot(distribution: pd.DataFrame, target: str, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Legitimate (0)" if label == "0" else "Fraud (1)" if label == "1" else f"Class {label}"
              for label in distribution["class"]]
    bars = ax.bar(labels, distribution["count"])
    ax.set_title("Class imbalance: legitimate vs fraudulent transactions")
    ax.set_ylabel("Number of transactions")
    ax.set_xlabel("Target class")
    ax.ticklabel_format(axis="y", style="plain")
    for bar, (_, row) in zip(bars, distribution.iterrows()):
        ax.annotate(f"{int(row['count']):,}\n{row['percentage']:.3f}%",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "01_class_imbalance.png", dpi=160)
    plt.close(fig)


def make_missing_plot(missing_table: pd.DataFrame, output_dir: Path) -> None:
    table = missing_table[missing_table["missing_count"] > 0].sort_values("missing_count")
    fig, ax = plt.subplots(figsize=(10, 5))
    if table.empty:
        ax.text(0.5, 0.5, "No missing values found", ha="center", va="center", fontsize=16)
        ax.set_axis_off()
        ax.set_title("Missing values by column")
    else:
        ax.barh(table["column"], table["missing_count"])
        ax.set_title("Missing values by column")
        ax.set_xlabel("Missing value count")
        ax.set_ylabel("Column")
    fig.tight_layout()
    fig.savefig(output_dir / "02_missing_values.png", dpi=160)
    plt.close(fig)


def make_amount_plots(df: pd.DataFrame, target: str, output_dir: Path) -> None:
    if "Amount" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, name in [(0, "Legitimate (0)"), (1, "Fraud (1)")]:
        values = pd.to_numeric(df.loc[df[target] == label, "Amount"], errors="coerce").dropna()
        # log1p helps display a long-tailed amount distribution without dropping zero-valued transactions.
        values = np.log1p(values.clip(lower=0))
        if not values.empty:
            ax.hist(values, bins=70, density=True, histtype="step", linewidth=1.7, label=name)
    ax.set_title("Transaction amount distributions by class")
    ax.set_xlabel("log(1 + Amount)")
    ax.set_ylabel("Density within class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "03_amount_distribution_log1p.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    grouped = [np.log1p(pd.to_numeric(df.loc[df[target] == label, "Amount"], errors="coerce").dropna().clip(lower=0))
               for label in (0, 1)]
    ax.boxplot(grouped, showfliers=False)
    ax.set_xticks([1, 2], ["Legitimate (0)", "Fraud (1)"])
    ax.set_title("Transaction amount by class (log1p scale; outliers hidden for readability)")
    ax.set_ylabel("log(1 + Amount)")
    fig.tight_layout()
    fig.savefig(output_dir / "04_amount_boxplot_log1p.png", dpi=160)
    plt.close(fig)


def make_time_plot(df: pd.DataFrame, target: str, output_dir: Path) -> None:
    if "Time" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, name in [(0, "Legitimate (0)"), (1, "Fraud (1)")]:
        values = pd.to_numeric(df.loc[df[target] == label, "Time"], errors="coerce").dropna() / 3600.0
        if not values.empty:
            ax.hist(values, bins=60, density=True, histtype="step", linewidth=1.7, label=name)
    ax.set_title("Transaction time distribution by class")
    ax.set_xlabel("Time from start of dataset (hours; relative, not wall-clock time)")
    ax.set_ylabel("Density within class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "05_time_distribution.png", dpi=160)
    plt.close(fig)


def make_correlation_plots(df: pd.DataFrame, target: str, output_dir: Path) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr(numeric_only=True)
    corr.to_csv(output_dir / "correlation_matrix.csv", index=True)

    if target in corr.columns:
        target_corr = corr[target].drop(labels=[target], errors="ignore").dropna()
        target_corr = target_corr.reindex(target_corr.abs().sort_values(ascending=False).index)
        target_corr.rename("correlation_with_Class").to_csv(output_dir / "correlations_with_class.csv", header=True)

        top_features = target_corr.head(6).index.tolist()
        if top_features:
            fig, axes = plt.subplots(2, 3, figsize=(15, 9))
            for ax, feature in zip(axes.flat, top_features):
                for label, name in [(0, "Legitimate (0)"), (1, "Fraud (1)")]:
                    values = pd.to_numeric(df.loc[df[target] == label, feature], errors="coerce").dropna()
                    if values.empty:
                        continue
                    # Clip only chart display at the 1st/99th percentiles so extreme values do not hide the shape.
                    lower, upper = values.quantile([0.01, 0.99])
                    values = pd.Series(np.clip(values.to_numpy(dtype=float), float(lower), float(upper)))
                    ax.hist(values, bins=40, density=True, histtype="step", linewidth=1.5, label=name)
                ax.set_title(f"{feature} (corr={target_corr[feature]:.3f})")
                ax.set_ylabel("Density within class")
                ax.legend(fontsize=8)
            for ax in axes.flat[len(top_features):]:
                ax.set_visible(False)
            fig.suptitle("Distributions of the six features most correlated with Class", y=1.01)
            fig.tight_layout()
            fig.savefig(output_dir / "07_top_feature_distributions.png", dpi=160, bbox_inches="tight")
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 7))
        sorted_corr = target_corr.sort_values()
        ax.barh(sorted_corr.index, sorted_corr.values)
        ax.axvline(0, linewidth=0.8)
        ax.set_title("Feature-wise correlation with Class (descriptive, not causal)")
        ax.set_xlabel("Pearson correlation with binary Class")
        ax.set_ylabel("Feature")
        fig.tight_layout()
        fig.savefig(output_dir / "08_correlations_with_class.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, 13))
    image = ax.imshow(corr.to_numpy(), aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.index, fontsize=7)
    ax.set_title("Correlation matrix for numeric columns")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="Pearson correlation")
    fig.tight_layout()
    fig.savefig(output_dir / "06_correlation_heatmap.png", dpi=160)
    plt.close(fig)
    return corr


def run_eda(df: pd.DataFrame, target: str, expected_columns: list[str], output_dir: Path) -> dict[str, Any]:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' is missing.")
    if not pd.api.types.is_numeric_dtype(df[target]):
        raise ValueError(f"Target column '{target}' must be numeric.")
    if df[target].isna().any():
        raise ValueError(f"Target column '{target}' contains missing values; resolve this before EDA.")

    unexpected_labels = sorted(set(df[target].dropna().unique().tolist()) - {0, 1})
    if unexpected_labels:
        raise ValueError(f"Expected binary target labels 0/1; found: {unexpected_labels}")

    rows, columns = df.shape
    memory_mib = float(df.memory_usage(deep=True).sum() / (1024 ** 2))
    missing = df.isna().sum().rename("missing_count").to_frame()
    missing["missing_percentage"] = missing["missing_count"] / max(rows, 1) * 100
    missing.index.name = "column"
    missing = missing.reset_index()
    missing.to_csv(output_dir / "missing_values.csv", index=False)

    duplicate_summary = build_duplicate_summary(df, target)
    duplicate_summary.to_csv(output_dir / "duplicate_summary.csv", index=False)

    distribution = class_distribution(df, target)
    distribution.to_csv(output_dir / "class_distribution.csv", index=False)

    df.describe(include="all").transpose().to_csv(output_dir / "descriptive_statistics.csv", index=True)
    dtype_table = pd.DataFrame({"column": df.columns, "dtype": [str(dtype) for dtype in df.dtypes]})
    dtype_table.to_csv(output_dir / "column_types.csv", index=False)

    make_class_plot(distribution, target, output_dir)
    make_missing_plot(missing, output_dir)
    make_amount_plots(df, target, output_dir)
    make_time_plot(df, target, output_dir)
    correlation = make_correlation_plots(df, target, output_dir)

    class_counts = df[target].value_counts(dropna=False).to_dict()
    fraud_count = int((df[target] == 1).sum())
    legitimate_count = int((df[target] == 0).sum())
    numeric = df.select_dtypes(include=[np.number])
    summary = {
        "dataset": "FraudShield AI transaction dataset",
        "shape": {"rows": int(rows), "columns": int(columns)},
        "memory_usage_mib": round(memory_mib, 3),
        "expected_columns_missing": [col for col in expected_columns if col not in df.columns],
        "extra_columns": [col for col in df.columns if col not in expected_columns],
        "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values_total": int(df.isna().sum().sum()),
        "columns_with_missing_values": int((df.isna().sum() > 0).sum()),
        "exact_duplicate_rows_extra_copies": int(df.duplicated().sum()),
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "legitimate_count": legitimate_count,
        "fraud_count": fraud_count,
        "fraud_percentage": round(fraud_count / rows * 100, 6) if rows else 0.0,
        "legitimate_percentage": round(legitimate_count / rows * 100, 6) if rows else 0.0,
        "amount_summary_by_class": {},
        "time_summary_by_class": {},
        "top_absolute_correlations_with_class": {},
        "notes": [
            "Time is a relative elapsed-time feature, not a calendar timestamp.",
            "V1-V28 are anonymized PCA components; do not assign unsupported business meanings to them.",
            "Correlation is descriptive and does not establish causation or model importance.",
            "Duplicate rows are reported for investigation; the source dataset is never modified by this script.",
            "If a test run used synthetic data, do not interpret its values as actual dataset results."
        ],
    }
    for feature in ["Amount", "Time"]:
        if feature in df.columns:
            summary_key = "amount_summary_by_class" if feature == "Amount" else "time_summary_by_class"
            for label, label_name in [(0, "legitimate_0"), (1, "fraud_1")]:
                values = pd.to_numeric(df.loc[df[target] == label, feature], errors="coerce").dropna()
                summary[summary_key][label_name] = {
                    "count": int(values.size),
                    "mean": safe_float(values.mean()),
                    "median": safe_float(values.median()),
                    "std": safe_float(values.std()),
                    "min": safe_float(values.min()),
                    "max": safe_float(values.max()),
                    "q25": safe_float(values.quantile(0.25)),
                    "q75": safe_float(values.quantile(0.75)),
                }
    if target in correlation.columns:
        target_corr = correlation[target].drop(labels=[target], errors="ignore").dropna()
        for feature, value in target_corr.reindex(target_corr.abs().sort_values(ascending=False).index).head(10).items():
            summary["top_absolute_correlations_with_class"][str(feature)] = safe_float(value)

    save_json(summary, output_dir / "eda_summary.json")
    return summary


def main() -> int:
    args = parse_args()
    try:
        config = load_config()
        data_path = resolve_data_path(config, args.data_path)
        output_dir = resolve_output_dir(args.output_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"CONFIGURATION ERROR: {exc}", file=sys.stderr)
        return 2

    if not data_path.is_file():
        print(f"DATASET NOT FOUND: {data_path}", file=sys.stderr)
        print("Download creditcard.csv and place it in data/raw/ or pass --data-path.", file=sys.stderr)
        return 2

    print(f"Loading: {data_path}")
    try:
        df = pd.read_csv(data_path)
        data_config = config.get("data", {})
        expected = data_config.get("expected_columns", [])
        target = data_config.get("target_column", "Class")
        validate_columns(df, expected, target)
        summary = run_eda(df, target, expected, output_dir)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"EDA FAILED: {exc}", file=sys.stderr)
        return 1

    print("\n=== EDA summary ===")
    print(f"Rows: {summary['shape']['rows']:,}")
    print(f"Columns: {summary['shape']['columns']:,}")
    print(f"Missing values: {summary['missing_values_total']:,}")
    print(f"Exact duplicate extra rows: {summary['exact_duplicate_rows_extra_copies']:,}")
    print(f"Fraud transactions: {summary['fraud_count']:,} ({summary['fraud_percentage']:.4f}%)")
    print(f"Legitimate transactions: {summary['legitimate_count']:,} ({summary['legitimate_percentage']:.4f}%)")
    print(f"\nOutputs written to: {output_dir}")
    print("Review eda_summary.json, CSV tables, and the PNG plots. No rows were removed or changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
