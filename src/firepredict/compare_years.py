"""What adds to the FWI? Small models vs CatBoost, judged on every year.

Question: "at least one fire >= 0.1 ha tomorrow?" (J+1, operational: data up
to the day before the forecast is issued).

Leave-one-year-out: each year 2006-2025 (except 2024) is hidden in turn, the
model learns on the other years, and predicts the hidden year. Scores:
  - roc_auc:    ranking quality over all hidden-year predictions together
  - year_auc:   mean of the per-year ROC AUCs
  - beats_fwi:  number of years where the model ranks days better than FWI alone

Small models (logistic regression), one change at a time:
  FWI alone | + 30-day dryness | + human calendar | + both | + burning period
CatBoost (all ~45 inputs, 5 seeds), with and without the dryness features,
and with the burning period.

Burning period (Var fire-use orders, see human_check): orange = Feb-Mar
(target day), red = June-Sept (target day), windy = mean wind > 40 km/h at
the windiest core station outside June-Sept. Wind is only known up to D-1, so
"windy" uses the last known day.

Usage:
    uv run python -m firepredict.compare_years
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from firepredict.dataset import PROCESSED_DIR, RESULTS_DIR
from firepredict.department import burning_periods
from firepredict.features import drought_state, make_xy
from firepredict.human_check import WINDY_MS
from firepredict.targets import SPLITS

OUT = RESULTS_DIR / "compare_years.csv"
FIRST_DAY = "2006-04-01"  # 90-day dryness windows need 3 months of history
SEEDS = [0, 1, 2, 3, 4]
# Raw values only: anomalies need a "normal" computed without the hidden year.
DRYNESS = ["rr_sum_30d", "rr_sum_60d", "rr_sum_90d", "water_balance_30d", "water_balance_60d",
           "water_balance_90d", "days_since_rain_10mm"]  # fmt: skip
HUMAN = ["weekend", "public_holiday", "school_holiday"]
BURNING = ["orange_period", "red_period", "windy_off_season"]
SMALL = {
    "FWI alone": ["log_fwi"],
    "FWI + 30-day dryness": ["log_fwi", "water_balance_30d"],
    "FWI + human calendar": ["log_fwi", *HUMAN],
    "FWI + both": ["log_fwi", "water_balance_30d", *HUMAN],
    "FWI + burning period": ["log_fwi", *BURNING],
    "FWI + burning period + dryness": ["log_fwi", "water_balance_30d", *BURNING],
}


def data() -> tuple[pd.DataFrame, pd.Series]:
    daily = pd.read_parquet(PROCESSED_DIR / "daily.parquet")
    state = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    state = state.join(drought_state(daily, SPLITS["train"])[DRYNESS])
    X, y = make_xy(state, daily, horizon=1, setting="operational")
    X = X.assign(log_fwi=np.log1p(X.fwi), weekend=X.day_of_week.ge(5).astype(int))
    rules = burning_periods(X.index + pd.Timedelta(days=1)).set_axis(X.index)  # of the target day
    X = X.join(rules).assign(windy_off_season=(X.ffm_max.gt(WINDY_MS) & rules.red_period.eq(0)).astype(int))
    keep = X.index >= FIRST_DAY
    return X[keep], (y.fire_count[keep] > 0).astype(int)


def leave_one_year_out(X: pd.DataFrame, y: pd.Series, make_model) -> pd.Series:
    years = (X.index + pd.Timedelta(days=1)).year  # year of the target day
    proba = pd.Series(np.nan, index=X.index)
    for year in np.unique(years):
        test = years == year
        model = make_model().fit(X[~test], y[~test])
        proba[test] = model.predict_proba(X[test])[:, 1]
    return proba


def scores(y: pd.Series, proba: pd.Series, fwi_year_auc: pd.Series | None) -> dict:
    years = (y.index + pd.Timedelta(days=1)).year
    by_year = pd.Series({yr: roc_auc_score(y[years == yr], proba[years == yr]) for yr in np.unique(years)})
    return {
        "roc_auc": roc_auc_score(y, proba),
        "year_auc": by_year.mean(),
        "beats_fwi": None if fwi_year_auc is None else int((by_year > fwi_year_auc + 1e-9).sum()),
        "by_year": by_year,
    }


def run() -> pd.DataFrame:
    X, y = data()
    print(f"{len(X)} days, {y.sum()} fire days, {(X.index + pd.Timedelta(days=1)).year.nunique()} years")
    rows, fwi_by_year = [], None
    for name, cols in SMALL.items():
        # Missing values are filled inside each fold, from training rows only.
        proba = leave_one_year_out(X[cols], y, lambda: make_pipeline(SimpleImputer(strategy="median"),
                                                                        StandardScaler(), LogisticRegression()))  # fmt: skip
        s = scores(y, proba, fwi_by_year)
        fwi_by_year = s["by_year"] if fwi_by_year is None else fwi_by_year
        rows.append({"model": name, **s})
        print(f"  {name}: {s['roc_auc']:.3f}")

    base = X.columns.drop(["log_fwi", "weekend", *DRYNESS, *BURNING])
    catboost = {
        "CatBoost (current inputs)": base,
        "CatBoost + dryness": base.append(pd.Index(DRYNESS)),
        "CatBoost + burning period": base.append(pd.Index(BURNING)),
    }
    for name, cols in catboost.items():
        for seed in SEEDS:
            model = lambda: CatBoostClassifier(iterations=500, learning_rate=0.03, depth=4,  # noqa: E731
                                               random_seed=seed, verbose=False, allow_writing_files=False)  # fmt: skip
            s = scores(y, leave_one_year_out(X[cols], y, model), fwi_by_year)
            rows.append({"model": name, "seed": seed, **s})
        print(f"  {name}: {np.mean([r['roc_auc'] for r in rows if r['model'] == name]):.3f}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    results = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    results.drop(columns="by_year").to_csv(OUT, index=False)
    summary = results.groupby("model", sort=False)[["roc_auc", "year_auc", "beats_fwi"]].agg(["mean", "std"])
    print(summary.round(3).to_string())
    per_year = pd.DataFrame({m: g.by_year.apply(pd.Series).mean() for m, g in results.groupby("model", sort=False)})
    print("\nROC AUC per hidden year:\n" + per_year.round(2).to_string())
