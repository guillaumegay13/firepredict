"""Commune models vs FWI references, judged on every year.

Question: "will a fire >= 0.1 ha start in this commune tomorrow?" (J+1,
operational). Leave-one-year-out over 2006-2025 (without unusable years, such
as 2024 for the Var): each year is hidden in turn, models learn on the other
years, and rank the hidden year's commune-days.

The commune's usual fire rate is recomputed for every hidden year, from the
other years only, and for each row also without the row's own year, so no
row ever sees the hidden year's fires or its own label.

Scores per hidden year:
  - roc_auc:  ranking of all commune-days of the year
  - top15:    share of fire starts falling in that day's 15 highest-ranked communes
Pooled: top-15 catch over all hidden-year fire starts together, and the
paired per-year difference to the reference (Wilcoxon test).

With data/processed/spatial_rates.parquet (model_spatial), the smoothed
spatial rate is also compared, alone, x FWI, and inside the small model.

Usage:
    uv run python -m firepredict.compare_communes            # references + small model
    uv run python -m firepredict.compare_communes catboost   # also CatBoost (slow)
"""

import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

from firepredict.dataset import PROCESSED_DIR, RESULTS_DIR
from firepredict.model_communes import SMALL, SMALL_LANDSCAPE, HORIZON, TOP_K, build_panel, current_features, fit_small, small_inputs

OUT = RESULTS_DIR / "compare_communes.csv"
SPATIAL = PROCESSED_DIR / "spatial_rates.parquet"  # from model_spatial
REFERENCE = "history x FWI"
SEEDS = [0, 1, 2, 3, 4]


def rates_without(counts: pd.DataFrame, hidden: int) -> pd.DataFrame:
    """Usual yearly rate per (year, commune), never using `hidden` nor the row's own year.

    counts: fire starts per year (rows) and commune (columns).
    """
    others = counts.drop(index=hidden)
    total, n = others.sum(), len(others)
    rate = (total - others) / (n - 1)  # training years: without their own year
    rate.loc[hidden] = total / n  # hidden year: all other years
    return rate


def leave_one_year_out(panel: pd.DataFrame, models: dict) -> dict[str, pd.Series]:
    year = pd.Series((panel.index.get_level_values("date") + pd.Timedelta(days=HORIZON)).year, index=panel.index)
    insee = panel.index.get_level_values("insee")
    counts = panel.target.groupby([year, insee]).sum().unstack(fill_value=0)
    scores = {name: pd.Series(np.nan, index=panel.index) for name in models}
    spatial = pd.read_parquet(SPATIAL) if SPATIAL.exists() else None
    for hidden in sorted(year.unique()):
        rate = rates_without(counts, hidden).stack()
        fold = panel.assign(usual_rate=rate.reindex(pd.MultiIndex.from_arrays([year, insee])).to_numpy())
        if spatial is not None:
            fold["spatial_rate"] = spatial[str(hidden)].reindex(insee).to_numpy()
        test = (year == hidden).to_numpy()
        for name, model in models.items():
            scores[name][test] = model(fold, test)
        print(f"  {hidden} done", flush=True)
    return scores


def history_x(column: str):
    return lambda fold, test: (fold.usual_rate * fold[column])[test].to_numpy()


def small_spatial(fold: pd.DataFrame, test: np.ndarray) -> np.ndarray:
    """The small model with the smoothed spatial rate in place of raw history."""
    return small(fold.assign(usual_rate=fold.spatial_rate), test)


def logistic(columns: list[str]):
    """A logistic model on a subset of the small model's inputs."""

    def model(fold: pd.DataFrame, test: np.ndarray) -> np.ndarray:
        X = small_inputs(fold)[columns]
        X = X.fillna(X[~test].median())
        return fit_small(X[~test], fold.target[~test]).predict_proba(X[test])[:, 1]

    return model


small = logistic(SMALL)
# Nested models: each adds one group of inputs to the one before.
NESTED = {
    "1. history + local FWI": ["log_usual_rate", "log_local_fwi"],
    "2. + burning calendar": ["log_usual_rate", "log_local_fwi", "orange_period", "red_period"],
    "3. + 30-day dryness (= small model)": SMALL,
    "4. + landscape": [*SMALL, *SMALL_LANDSCAPE],
}


def catboost(seed: int):
    def model(fold: pd.DataFrame, test: np.ndarray) -> np.ndarray:
        features = current_features(fold)
        m = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, random_seed=seed,
                               verbose=False, allow_writing_files=False)  # fmt: skip
        m.fit(fold.loc[~test, features], fold.target[~test])
        return m.predict_proba(fold.loc[test, features])[:, 1]

    return model


def per_year(panel: pd.DataFrame, score: pd.Series) -> pd.DataFrame:
    date = panel.index.get_level_values("date")
    year = (date + pd.Timedelta(days=HORIZON)).year
    rank = score.groupby(level="date").rank(ascending=False, method="first")
    caught = (rank <= TOP_K) & (panel.target == 1)
    rows = {}
    for y in np.unique(year):
        m = year == y
        rows[y] = {"roc_auc": roc_auc_score(panel.target[m], score[m]),
                   "top15": caught[m].sum() / panel.target[m].sum(),
                   "fires": int(panel.target[m].sum()), "caught": int(caught[m].sum())}  # fmt: skip
    return pd.DataFrame(rows).T


def run(with_catboost: bool) -> pd.DataFrame:
    panel = build_panel()
    models = {
        "history only": lambda fold, test: fold.usual_rate[test].to_numpy(),
        "history x FWI": history_x("fwi"),
        "history x local FWI": history_x("local_fwi"),
        "small model": small,
        **{name: logistic(columns) for name, columns in NESTED.items() if columns != SMALL},
    }
    if SPATIAL.exists():
        models |= {
            "spatial only": lambda fold, test: fold.spatial_rate[test].to_numpy(),
            "spatial x FWI": lambda fold, test: (fold.spatial_rate * fold.fwi)[test].to_numpy(),
            "small model (spatial)": small_spatial,
        }
    if with_catboost:
        models |= {f"CatBoost seed {s}": catboost(s) for s in SEEDS}
    scores = leave_one_year_out(panel, models)
    return pd.concat({name: per_year(panel, s) for name, s in scores.items()}, names=["model", "year"])


def summary(results: pd.DataFrame, reference: str = REFERENCE) -> pd.DataFrame:
    ref = results.loc[reference]
    rows = {}
    for name, r in results.groupby(level="model", sort=False):
        r = r.droplevel("model")
        d_auc, d_top = r.roc_auc - ref.roc_auc, r.top15 - ref.top15
        rows[name] = {
            "roc_auc": r.roc_auc.mean(),
            "top15 (pooled)": r.caught.sum() / r.fires.sum(),
            "fires caught": int(r.caught.sum()),
            "auc vs ref": d_auc.mean(),
            "years better (auc)": f"{(d_auc > 0).sum()}/{len(r)}",
            "p (auc)": wilcoxon(d_auc).pvalue if (d_auc != 0).any() else np.nan,
            "years better (top15)": f"{(d_top > 0).sum()}/{(d_top != 0).sum()}",
            "p (top15)": wilcoxon(d_top).pvalue if (d_top != 0).any() else np.nan,
        }  # fmt: skip
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    results = run(with_catboost="catboost" in sys.argv[1:])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT)
    print(f"\nreference: {REFERENCE}, {int(results.loc[REFERENCE].fires.sum())} fire starts in hidden years\n")
    print(summary(results).to_string(float_format=lambda v: f"{v:.3f}"))
