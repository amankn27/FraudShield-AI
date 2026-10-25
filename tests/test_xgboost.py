import numpy as np
import pandas as pd
import pytest

from src.models.xgboost_model import make_chronological_cv_splits


def test_chronological_cv_splits_are_time_ordered_and_nonoverlapping():
    df = pd.DataFrame({"Time": np.repeat(np.arange(100), 3), "x": np.arange(300)})
    splits = make_chronological_cv_splits(df, time_column="Time")
    assert len(splits) == 3
    for train_idx, val_idx in splits:
        assert len(train_idx) > 0 and len(val_idx) > 0
        assert df.iloc[train_idx].Time.max() < df.iloc[val_idx].Time.min()
        assert not set(train_idx).intersection(set(val_idx))


def test_cv_requires_sorted_time():
    df = pd.DataFrame({"Time": [0, 1, 2, 3, 5, 4, 6, 7, 8, 9], "x": range(10)})
    with pytest.raises(ValueError, match="sorted chronologically"):
        make_chronological_cv_splits(df)
