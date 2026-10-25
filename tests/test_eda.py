import pandas as pd

from src.data.eda import build_duplicate_summary, class_distribution


def test_class_distribution_counts_and_percentages():
    frame = pd.DataFrame({"Class": [0, 0, 0, 1]})
    result = class_distribution(frame, "Class").set_index("class")

    assert int(result.loc["0", "count"]) == 3
    assert int(result.loc["1", "count"]) == 1
    assert round(float(result.loc["0", "percentage"]), 2) == 75.00
    assert round(float(result.loc["1", "percentage"]), 2) == 25.00


def test_duplicate_summary_detects_feature_duplicates_with_conflicting_labels():
    frame = pd.DataFrame({
        "Time": [1.0, 1.0, 2.0],
        "Amount": [10.0, 10.0, 25.0],
        "Class": [0, 1, 0],
    })
    result = build_duplicate_summary(frame, "Class").set_index("check")

    assert int(result.loc["exact_duplicate_rows_extra_copies", "count"]) == 0
    assert int(result.loc["duplicate_feature_rows_extra_copies", "count"]) == 1
    assert int(result.loc["rows_in_duplicate_feature_groups", "count"]) == 2
    assert int(result.loc["duplicate_feature_groups_with_conflicting_labels", "count"]) == 1
