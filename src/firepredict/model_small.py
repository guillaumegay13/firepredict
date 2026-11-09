"""Small day model: FWI + burning period + 30-day water balance (logistic regression).

Same protocol as model_catboost, so the scores are comparable:
- question: "at least one fire >= 0.1 ha tomorrow?" (J+1, operational)
- learns on the training years (2006-2020 + 2022)
- the warning threshold on the predicted probability is the best F1 on 2021
- scored on 2023 and 2025, never seen, for the whole year and by season
References on the same days: FWI >= cut (cut fitted on the training years),
and "warn every day from June to September".

Deterministic (no random seed), so one run.

Usage:
    uv run python -m firepredict.model_small
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from firepredict.baseline_fwi import fit_binary_cut
from firepredict.compare_years import BURNING, data
from firepredict.dataset import RESULTS_DIR
from firepredict.model_catboost import TEST_SPLITS, best_threshold, score_rows
from firepredict.targets import split_of

OUT = RESULTS_DIR / "model_small.csv"
FEATURES = ["log_fwi", "water_balance_30d", *BURNING]


def run() -> pd.DataFrame:
    X, fire = data()
    target_day = X.index + pd.Timedelta(days=1)
    split = pd.Series(split_of(target_day).to_numpy(), index=X.index)
    season = pd.Series(np.where(target_day.month.isin([6, 7, 8, 9]), "Jun-Sep", "Oct-May"), index=X.index)
    keep = split.notna()
    X, fire, split, season = X[keep], fire[keep], split[keep], season[keep]
    train, val = split == "train", split == "val"

    model = make_pipeline(StandardScaler(), LogisticRegression())
    model.fit(X.loc[train, FEATURES], fire[train])
    proba = pd.Series(model.predict_proba(X[FEATURES])[:, 1], index=X.index)
    warn_small = (proba >= best_threshold(fire[val], proba[val])).astype(int)

    cut = fit_binary_cut(X.fwi[train], fire[train])
    references = {
        "warn Jun-Sep": (season == "Jun-Sep").astype(int),
        f"FWI >= {cut:.1f}": (X.fwi >= cut).astype(int),
        "small model": warn_small,
    }
    rows = []
    for name, warn in references.items():
        rows += score_rows(fire, warn, split, season, model=name)
    results = pd.DataFrame(rows)

    ranking = {s: {"FWI": roc_auc_score(fire[split == s], X.fwi[split == s]),
                   "small model": roc_auc_score(fire[split == s], proba[split == s])} for s in TEST_SPLITS}  # fmt: skip
    weights = pd.Series(model[-1].coef_[0], index=FEATURES)
    print("weights (inputs standardised; + = more fires):\n" + weights.round(2).to_string())
    print("\nROC AUC:\n" + pd.DataFrame(ranking).round(2).to_string())
    return results


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    results = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT, index=False)
    table = results.pivot_table(index=["split", "season"], columns="model", values="f1", sort=False).round(2)
    print("\nF1:\n" + table.to_string())
    warned = results[results.season == "all year"].pivot_table(index="split", columns="model", values="warned_days")
    print("\nwarned days (fire days: " + results[results.season == "all year"].groupby("split").fire_days.first().to_string().replace("\n", ", ") + "):")
    print(warned.to_string())
