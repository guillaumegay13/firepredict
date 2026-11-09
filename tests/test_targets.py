import numpy as np
import pandas as pd

from firepredict.targets import RiskClasses, split_of


def test_zero_is_class_0_and_fires_are_at_least_class_1():
    counts = pd.Series([0, 1, 1, 1, 2, 2, 3, 3, 4, 6, 7, 0, np.nan])
    classes = RiskClasses().fit(counts.dropna()).predict(counts)
    assert classes.iloc[0] == 0 and classes.iloc[11] == 0
    assert (classes[counts > 0] >= 1).all()
    assert classes.is_monotonic_increasing is False  # zeros at the end
    assert classes.iloc[1:11].is_monotonic_increasing
    assert np.isnan(classes.iloc[12])


def test_splits():
    split = split_of(pd.to_datetime(["2006-01-15", "2010-07-01", "2021-08-16", "2023-01-01", "2024-07-01"]))
    assert split.isna().tolist() == [True, False, False, False, True]
    assert split.dropna().tolist() == ["train", "val", "test"]
