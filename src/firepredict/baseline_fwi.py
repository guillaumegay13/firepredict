"""FWI baseline: the official fire danger index, used on its own.

Two versions, both fitted on training years only:
- classes: cut FWI into 0-4 so each level is as frequent as the matching
  fire class in training (quantile matching).
- binary: the single FWI threshold with the best F1 in training.

Operational uses FWI of the day before the issue date; oracle uses FWI of
the target day.

Usage:
    uv run python -m firepredict.baseline_fwi
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from firepredict.dataset import PROCESSED_DIR, ROOT
from firepredict.evaluate import scores
from firepredict.features import SETTINGS, make_xy
from firepredict.targets import RiskClasses, split_of

OUT = ROOT / "results" / "baseline_fwi.csv"
HORIZONS = [0, 1]


def fit_class_cuts(fwi: pd.Series, classes: pd.Series) -> np.ndarray:
    """FWI cut points so that predicted class shares match the true ones."""
    shares = classes.value_counts(normalize=True).sort_index().reindex(range(5), fill_value=0)
    return np.quantile(fwi, shares.cumsum().to_numpy()[:-1])


def fit_binary_cut(fwi: pd.Series, fire: pd.Series) -> float:
    candidates = np.quantile(fwi, np.linspace(0.5, 0.99, 100))
    return max(candidates, key=lambda cut: f1_score(fire, fwi >= cut))


def run() -> pd.DataFrame:
    daily = pd.read_parquet(PROCESSED_DIR / "daily.parquet")
    state = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    rows = []
    for setting in SETTINGS:
        for horizon in HORIZONS:
            X, y = make_xy(state, daily, horizon, setting)
            fwi = X["target_fwi"] if setting == "oracle" else X["fwi"]
            # A row belongs to the split of its target day.
            split = pd.Series(split_of(y.index + pd.Timedelta(days=horizon)).to_numpy(), index=y.index)
            train = split == "train"

            risk = RiskClasses().fit(y.fire_count[train])
            true = risk.predict(y.fire_count)
            class_cuts = fit_class_cuts(fwi[train], true[train])
            binary_cut = fit_binary_cut(fwi[train], true[train] > 0)

            predictions = {
                "classes": pd.Series(np.searchsorted(class_cuts, fwi), index=fwi.index),
                "binary": (fwi >= binary_cut).astype(int),
            }
            for version, pred in predictions.items():
                for name in ["val", "test", "test_2025"]:
                    mask = split == name
                    rows.append(
                        {
                            "setting": setting,
                            "horizon": f"J+{horizon}" if horizon else "J0",
                            "version": version,
                            "split": name,
                            **scores(true[mask], pred[mask]),
                            "fwi_cuts": np.round(class_cuts if version == "classes" else [binary_cut], 1).tolist(),
                        }
                    )
    results = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT, index=False)
    return results


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(run().round(2).to_string(index=False))
