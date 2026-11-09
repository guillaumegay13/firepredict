"""How much better is the small commune model, in fires caught, with uncertainty?

Same every-year protocol as compare_communes (each of 19 years hidden in
turn). References:
  - FWI alone (Var):  the official index, the same value for every commune;
                      it can only tell days apart, not communes
  - local FWI alone:  FWI from each commune's nearest stations, closer to the
                      zone-level danger forecasts fire services use
  - history x FWI:    the strongest FWI-based reference (adds commune history)

Two ways to spend warnings, counting positive commune-days caught (days on
which at least one fire >= 0.1 ha started in the commune):
  - pooled budget: the highest-scored share of all commune-days over the 19
    years. A retrospective ranking measure: some days get many warnings,
    others none, and no live rule could pick exactly these.
  - daily capacity: the K highest-scored communes every day (K = 5, 15, 30),
    a rule fire services could actually apply.
Ties (FWI alone gives every commune the same score) are broken at random.
95% ranges come from resampling the 19 years with replacement (1000 times);
they are conditional on the chosen model (inputs were selected on these years).

Usage:
    uv run python -m firepredict.value_check
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from firepredict.compare_communes import history_x, leave_one_year_out, small
from firepredict.dataset import ROOT
from firepredict.model_communes import HORIZON, build_panel

OUT = ROOT / "results" / "value_check.csv"
BUDGETS = [0.01, 0.05, 0.10]
DAILY_K = [5, 15, 30]
MODEL = "small model"
REFERENCES = ["FWI alone (Var)", "local FWI alone", "history x FWI"]
N_BOOT = 1000


def caught(y: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Fires caught at each budget in BUDGETS."""
    hits = np.cumsum(y[np.argsort(-score, kind="stable")])
    return np.array([hits[int(len(y) * share) - 1] for share in BUDGETS])


def daily_caught(panel: pd.DataFrame, score: np.ndarray, year: np.ndarray) -> pd.DataFrame:
    """Positive commune-days caught per year by warning the K top communes each day."""
    rank = pd.Series(score, index=panel.index).groupby(level="date").rank(ascending=False, method="first")
    hit = panel.target.to_numpy() == 1
    return pd.DataFrame({k: pd.Series(hit & (rank.to_numpy() <= k)).groupby(year).sum() for k in DAILY_K})


def run() -> pd.DataFrame:
    panel = build_panel()
    models = {
        "FWI alone (Var)": lambda fold, test: fold.fwi[test].to_numpy(),
        "local FWI alone": lambda fold, test: fold.local_fwi[test].to_numpy(),
        "history x FWI": history_x("fwi"),
        MODEL: small,
    }
    scores = leave_one_year_out(panel, models)
    rng = np.random.default_rng(0)
    jitter = rng.uniform(0, 1e-9, len(panel))  # random tie-breaking, tiny next to real differences
    year = (panel.index.get_level_values("date") + pd.Timedelta(days=HORIZON)).year.to_numpy()
    y = panel.target.to_numpy()
    s = {name: v.to_numpy() * (1 + jitter) + jitter for name, v in scores.items()}
    years = np.unique(year)
    blocks = {yr: np.flatnonzero(year == yr) for yr in years}
    auc_by_year = {name: np.array([roc_auc_score(y[blocks[yr]], v[blocks[yr]]) for yr in years]) for name, v in s.items()}

    names = [*REFERENCES, MODEL]
    full = {name: caught(y, s[name]) for name in names}
    boot = {name: [] for name in names}
    for _ in range(N_BOOT):
        idx = np.concatenate([blocks[yr] for yr in rng.choice(years, len(years))])
        for name in names:
            boot[name].append(caught(y[idx], s[name][idx]))
    boot = {name: np.array(v) for name, v in boot.items()}
    daily = {name: daily_caught(panel, s[name], year) for name in names}
    draws = [rng.choice(years, len(years)) for _ in range(N_BOOT)]

    rows = []
    for ref in REFERENCES:
        d_auc = auc_by_year[MODEL] - auc_by_year[ref]
        boot_auc = [d_auc[rng.integers(0, len(years), len(years))].mean() for _ in range(N_BOOT)]
        row = {"reference": ref, "ref AUC": round(auc_by_year[ref].mean(), 3),
               "model AUC": round(auc_by_year[MODEL].mean(), 3), "AUC gain": round(d_auc.mean(), 3),
               "AUC gain 95%": np.percentile(boot_auc, [2.5, 97.5]).round(3).tolist(),
               "years better": f"{(d_auc > 0).sum()}/{len(years)}"}  # fmt: skip
        gains = boot[MODEL] / boot[ref] - 1
        for j, share in enumerate(BUDGETS):
            low, high = np.nanpercentile(gains[:, j], [2.5, 97.5])
            base, model = full[ref][j], full[MODEL][j]
            row[f"{share:.0%}: fires ref -> model"] = f"{base} -> {model}"
            row[f"{share:.0%}: gain (95%)"] = f"{model / base - 1:+.0%} ({low:+.0%} to {high:+.0%})"
        for k in DAILY_K:
            base, model = daily[ref][k], daily[MODEL][k]
            ratios = [model.loc[d].sum() / base.loc[d].sum() - 1 for d in draws]
            low, high = np.nanpercentile(ratios, [2.5, 97.5])
            row[f"top {k}/day: ref -> model"] = f"{base.sum()} -> {model.sum()}"
            row[f"top {k}/day: gain (95%)"] = f"{model.sum() / base.sum() - 1:+.0%} ({low:+.0%} to {high:+.0%})"
        rows.append(row)
    return pd.DataFrame(rows).set_index("reference")


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 40)
    results = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT)
    print(results.T.to_string())
