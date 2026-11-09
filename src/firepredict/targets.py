"""The 0-4 risk scale and the train / validation / test years.

Risk classes follow the benchmark (Caron et al. 2025): class 0 = no fire,
classes 1-4 = K-Means with 4 clusters on the days with fire, ordered from
smallest to largest. Thresholds are fitted on training years only.
For burned area we cluster log(1 + ha) instead of raw ha (our variant):
on raw ha, a few huge fires take 3 of the 4 clusters.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

SPLITS = {
    "train": [*range(2006, 2021), 2022],
    "val": [2021],  # 2024 has no usable labels
    "test": [2023],
    "test_2025": [2025],  # extra test, outside the benchmark design
}
TRAIN_START = "2006-03-01"  # skip the FWI spin-up at the start of 2006


def split_of(dates: pd.DatetimeIndex) -> pd.Series:
    """Split name for each date, or NaN for dates in no split."""
    year_to_split = {year: name for name, years in SPLITS.items() for year in years}
    split = pd.Series(dates.year.map(year_to_split), index=dates)
    split[dates < TRAIN_START] = np.nan
    return split


class RiskClasses:
    """0 = no fire; 1-4 = K-Means clusters of the positive days."""

    def __init__(self, log: bool = False):
        self.log = log
        self.thresholds = None  # 3 boundaries between classes 1-2, 2-3, 3-4

    def _scale(self, values: np.ndarray) -> np.ndarray:
        return np.log1p(values) if self.log else values

    def fit(self, values: pd.Series) -> "RiskClasses":
        positive = self._scale(values[values > 0].to_numpy()).reshape(-1, 1)
        km = KMeans(n_clusters=4, random_state=42, n_init=10).fit(positive)
        centres = np.sort(km.cluster_centers_.ravel())
        # In 1-D, "nearest centre" is the same as cutting halfway between centres.
        self.thresholds = (centres[1:] + centres[:-1]) / 2
        return self

    def thresholds_in_units(self) -> np.ndarray:
        return np.expm1(self.thresholds) if self.log else self.thresholds

    def predict(self, values: pd.Series) -> pd.Series:
        classes = 1 + np.searchsorted(self.thresholds, self._scale(values.to_numpy()))
        classes = np.where(values > 0, classes, 0).astype(float)
        classes[values.isna()] = np.nan
        return pd.Series(classes, index=values.index)
