"""CatBoost, 0-4 version: predicts the risk class of the target day.

- learns the 5 classes directly (multiclass), on training years 2006+
- ~90% of days are class 0, so during learning we keep only a share of the
  no-fire days (all, 50%, 25% or 10%), as in the benchmark; validation
  (2021) picks the share with the best IoU, and the number of trees
- the predicted class is the most likely one
- scored on 2023 and 2025 with IoU (0-4 curves) and the binary scores
- FWI cut into 0-4 (cuts fitted on the same training days) is scored on the
  same rows for comparison

Usage:
    uv run python -m firepredict.model_catboost_classes              # operational
    uv run python -m firepredict.model_catboost_classes oracle       # perfect weather forecast
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from firepredict.baseline_fwi import fit_class_cuts
from firepredict.dataset import PROCESSED_DIR, ROOT
from firepredict.evaluate import confusion, scores
from firepredict.features import make_xy
from firepredict.targets import RiskClasses, split_of

OUT_DIR = ROOT / "results"
SEEDS = [0, 1, 2, 3, 4]
ZERO_SHARES = [1.0, 0.5, 0.25, 0.1]
TEST_SPLITS = ["val", "test", "test_2025"]


def fit_predict(X, y, train, val, zero_share: float, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    keep_zero = rng.random(len(y)) < zero_share
    learn = train & ((y > 0) | keep_zero)
    model = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=1000,
        learning_rate=0.05,
        depth=4,
        early_stopping_rounds=100,
        random_seed=seed,
        verbose=False,
    )
    model.fit(X[learn], y[learn], eval_set=(X[val], y[val]))
    return pd.Series(model.predict(X).ravel().astype(float), index=X.index)


def run(horizon: int = 1, setting: str = "operational") -> tuple[pd.DataFrame, dict]:
    daily = pd.read_parquet(PROCESSED_DIR / "daily.parquet")
    state = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    X, y = make_xy(state, daily, horizon, setting)
    split = pd.Series(split_of(y.index + pd.Timedelta(days=horizon)).to_numpy(), index=y.index)
    keep = split.notna()
    X, y, split = X[keep], y[keep], split[keep]
    train, val = split == "train", split == "val"

    true = RiskClasses().fit(y.fire_count[train]).predict(y.fire_count)
    fwi = X.target_fwi if setting == "oracle" else X.fwi
    fwi_class = pd.Series(np.searchsorted(fit_class_cuts(fwi[train], true[train]), fwi), index=X.index).astype(float)

    rows, predictions = [], {"FWI": fwi_class}
    for seed in SEEDS:
        by_share = {s: fit_predict(X, true, train, val, s, seed) for s in ZERO_SHARES}
        best = max(ZERO_SHARES, key=lambda s: scores(true[val], by_share[s][val])["iou"])
        predictions[f"CatBoost seed {seed}"] = by_share[best]
        for name in TEST_SPLITS:
            m = split == name
            rows.append({"model": "CatBoost", "seed": seed, "zero_share": best, "split": name,
                         **scores(true[m], by_share[best][m])})  # fmt: skip
    for name in TEST_SPLITS:
        m = split == name
        rows.append({"model": "FWI", "seed": 0, "split": name, **scores(true[m], fwi_class[m])})

    results = pd.DataFrame(rows).assign(horizon=horizon, setting=setting)
    test = split == "test"
    confusions = {
        "FWI": confusion(true[test], fwi_class[test]),
        "CatBoost (seed 0)": confusion(true[test], predictions["CatBoost seed 0"][test]),
    }
    return results, confusions


if __name__ == "__main__":
    import sys

    pd.set_option("display.width", 200)
    setting = sys.argv[1] if len(sys.argv) > 1 else "operational"
    results, confusions = run(setting=setting)
    results.to_csv(OUT_DIR / f"catboost_classes_{setting}.csv", index=False)
    cols = ["iou", "under", "over", "f1", "precision", "recall"]
    summary = results.groupby(["split", "model"])[cols].agg(["mean", "std"]).round(2)
    print(summary.reindex(TEST_SPLITS, level=0).to_string())
    print("\nchosen share of no-fire days kept:", results[results.model == "CatBoost"].groupby("seed").zero_share.first().tolist())
    for name, table in confusions.items():
        print(f"\n2023 confusion, {name} (rows = true class):\n{table.to_string()}")
