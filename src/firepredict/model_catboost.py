"""CatBoost, binary version: "at least one fire >= 0.1 ha on the target day?"

- learns on a window of training years (all of them, or recent ones only)
- validation (2021) picks the number of trees (early stopping) and the
  warning threshold on the predicted probability (best F1)
- scored on 2023 and 2025, never seen during learning, for the whole year
  and separately for summer (Jun-Sep) and the rest of the year
- 5 seeds; we report mean and standard deviation
- the FWI rule (warn if FWI >= cut, cut fitted on the same window) is scored
  on the same rows for comparison

Usage:
    uv run python -m firepredict.model_catboost             # operational
    uv run python -m firepredict.model_catboost oracle      # perfect weather forecast
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score

from firepredict.baseline_fwi import fit_binary_cut
from firepredict.dataset import PROCESSED_DIR, ROOT
from firepredict.evaluate import scores
from firepredict.features import make_xy
from firepredict.targets import SPLITS, split_of

OUT = ROOT / "results" / "catboost_binary.csv"
SEEDS = [0, 1, 2, 3, 4]
TEST_SPLITS = ["val", "test", "test_2025"]
TRAIN_WINDOWS = {
    "2006+": SPLITS["train"],
    "recent": [2017, 2018, 2019, 2020, 2022],  # benchmark's training years
}


def best_threshold(fire: pd.Series, proba: np.ndarray) -> float:
    candidates = np.linspace(0.02, 0.8, 79)
    return max(candidates, key=lambda t: f1_score(fire, proba >= t))


def score_rows(fire, warn, split, season, **info) -> list[dict]:
    rows = []
    for name in TEST_SPLITS:
        for season_name in ["all year", "Jun-Sep", "Oct-May"]:
            mask = (split == name) & ((season == season_name) | (season_name == "all year"))
            rows.append({"split": name, "season": season_name, **info, **scores(fire[mask], warn[mask])})
    return rows


def run(horizon: int = 1, setting: str = "operational") -> pd.DataFrame:
    daily = pd.read_parquet(PROCESSED_DIR / "daily.parquet")
    state = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    X, y = make_xy(state, daily, horizon, setting)
    target_day = y.index + pd.Timedelta(days=horizon)
    split = pd.Series(split_of(target_day).to_numpy(), index=y.index)
    season = pd.Series(np.where(target_day.month.isin([6, 7, 8, 9]), "Jun-Sep", "Oct-May"), index=y.index)
    keep = split.notna()
    X, fire, split, season = X[keep], (y.fire_count[keep] > 0).astype(int), split[keep], season[keep]
    val = split == "val"

    rows = []
    for window, years in TRAIN_WINDOWS.items():
        train = (split == "train") & X.index.year.isin(years)
        info = {"window": window, "train_fire_days": int(fire[train].sum())}

        fwi = X.target_fwi if setting == "oracle" else X.fwi
        cut = fit_binary_cut(fwi[train], fire[train])
        rows += score_rows(fire, (fwi >= cut).astype(int), split, season, model="FWI", seed=0, **info)

        for seed in SEEDS:
            model = CatBoostClassifier(
                iterations=2000,
                learning_rate=0.03,
                depth=4,
                early_stopping_rounds=200,
                random_seed=seed,
                verbose=False,
            )
            model.fit(X[train], fire[train], eval_set=(X[val], fire[val]))
            proba = model.predict_proba(X)[:, 1]
            warn = pd.Series((proba >= best_threshold(fire[val], proba[val])).astype(int), index=X.index)
            rows += score_rows(fire, warn, split, season, model="CatBoost", seed=seed, **info)
    return pd.DataFrame(rows).assign(horizon=horizon, setting=setting)


if __name__ == "__main__":
    import sys

    pd.set_option("display.width", 200)
    setting = sys.argv[1] if len(sys.argv) > 1 else "operational"
    results = run(setting=setting)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT.with_name(f"catboost_binary_{setting}.csv"), index=False)
    print(results.groupby("window").train_fire_days.first().to_string(), "\n")
    f1 = results.groupby(["split", "season", "window", "model"]).f1.agg(["mean", "std"])
    table = f1["mean"].round(2).astype(str) + " ± " + f1["std"].fillna(0).round(2).astype(str)
    print(table.unstack(["window", "model"]).reindex(TEST_SPLITS, level=0).to_string())
