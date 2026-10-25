import pandas as pd

from src.data.splitting import chronological_train_validation_test_split


def test_chronological_split_does_not_split_equal_timestamps():
    df = pd.DataFrame({
        "Time": [0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 8, 9],
        "feature": list(range(12)),
        "Class": [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    })
    splits = chronological_train_validation_test_split(
        df, train_fraction=0.5, validation_fraction=0.25
    )
    assert splits.train["Time"].max() < splits.validation["Time"].min()
    assert splits.validation["Time"].max() < splits.test["Time"].min()
    assert set(splits.train["Time"]).isdisjoint(set(splits.validation["Time"]))
    assert set(splits.validation["Time"]).isdisjoint(set(splits.test["Time"]))
    assert len(splits.train) + len(splits.validation) + len(splits.test) == len(df)


def test_split_rejects_invalid_fraction_sum():
    df = pd.DataFrame({"Time": range(10), "Class": [0] * 9 + [1]})
    try:
        chronological_train_validation_test_split(
            df, train_fraction=0.8, validation_fraction=0.3
        )
    except ValueError as exc:
        assert "less than 1" in str(exc)
    else:
        raise AssertionError("Expected invalid fractions to be rejected")
