"""Model B, spread: "if a fire starts, will it grow to >= 1 ha (or >= 10 ha)?"

One row per fire (fires >= 0.1 ha, the same "start" as model A), with the
weather and FWI of its day: the forecast a fire service would have for that
day, assumed correct.

Evaluation: leave-one-year-out. Each year (2006-2025 except 2024) is
predicted by a model learned on all the other years. Ranking quality is
measured with ROC AUC (0.5 = random, 1 = perfect) on all held-out
predictions together, and compared with simple references on the same fires.

Usage:
    uv run python -m firepredict.model_spread
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from firepredict.dataset import MIN_FIRE_HA, PROCESSED_DIR, RESULTS_DIR, UNRELIABLE_YEARS, load_bdiff
from firepredict.features import weather_state

OUT = RESULTS_DIR / "spread.csv"
SIZES_HA = [1, 10]
SEEDS = [0, 1, 2, 3, 4]


def fire_table() -> pd.DataFrame:
    """Each fire >= MIN_FIRE_HA with the weather state of its day."""
    daily = pd.read_parquet(PROCESSED_DIR / "daily.parquet")
    state = pd.read_parquet(PROCESSED_DIR / "features.parquet")[weather_state(daily).columns]
    fires = load_bdiff()
    fires = fires[(fires.area_ha >= MIN_FIRE_HA) & ~fires.alert_time.dt.year.isin(UNRELIABLE_YEARS)]
    day = fires.alert_time.dt.normalize()
    table = state.reindex(day).set_index(fires.index)
    table["month"] = day.dt.month.to_numpy()
    table["day_of_year"] = day.dt.day_of_year.to_numpy()
    table["year"] = day.dt.year.to_numpy()
    table["area_ha"] = fires.area_ha
    return table[table.index.notna() & (day >= "2006-03-01").to_numpy()]


def leave_one_year_out(X: pd.DataFrame, big: pd.Series, years: pd.Series, seed: int) -> pd.Series:
    proba = pd.Series(np.nan, index=X.index)
    for year in sorted(years.unique()):
        test = years == year
        model = CatBoostClassifier(iterations=300, learning_rate=0.05, depth=4, random_seed=seed, verbose=False)
        model.fit(X[~test], big[~test])
        proba[test] = model.predict_proba(X[test])[:, 1]
    return proba


def month_rate(big: pd.Series, month: pd.Series, years: pd.Series) -> pd.Series:
    """Calendar reference: share of big fires in that month, from the other years."""
    out = pd.Series(np.nan, index=big.index)
    for year in years.unique():
        test = years == year
        out[test] = month[test].map(big[~test].groupby(month[~test]).mean())
    return out


def run() -> pd.DataFrame:
    table = fire_table()
    X = table.drop(columns=["area_ha", "year"])
    rows = []
    for size in SIZES_HA:
        big = (table.area_ha >= size).astype(int)
        references = {
            "calendar (month)": month_rate(big, table.month, table.year),
            "FWI": table.fwi,
            "ISI (wind + dry litter)": table.isi,
            "mean wind": table.ffm_mean,
        }
        catboost = [leave_one_year_out(X, big, table.year, seed) for seed in SEEDS]
        for name, score in references.items():
            ok = score.notna()
            rows.append({"size": f">= {size} ha", "model": name,
                         "roc_auc": roc_auc_score(big[ok], score[ok]), "pr_auc": average_precision_score(big[ok], score[ok])})  # fmt: skip
        aucs = [roc_auc_score(big, p) for p in catboost]
        prs = [average_precision_score(big, p) for p in catboost]
        rows.append({"size": f">= {size} ha", "model": "CatBoost", "roc_auc": np.mean(aucs),
                     "roc_auc_std": np.std(aucs), "pr_auc": np.mean(prs)})  # fmt: skip
        rows.append({"size": f">= {size} ha", "model": "(share of fires that grow)", "pr_auc": big.mean(),
                     "fires": len(big), "big_fires": int(big.sum())})  # fmt: skip
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    results = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT, index=False)
    print(results.round(3).to_string(index=False))
