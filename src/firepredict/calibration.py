"""Are the small commune model's probabilities right? Plus PR AUC and one confusion table.

Same every-year protocol as compare_communes (19 years hidden in turn).

1. Probabilities. "Climatology" is the honest probabilistic reference: each
   commune's usual daily rate (its yearly rate / 365.25), the same every day.
     - Brier score (mean squared error of the probability; lower is better)
     - Brier skill score vs climatology (> 0 = better than climatology)
     - log loss
     - reliability table: commune-days grouped by predicted chance, using the
       map's colour bands; predicted vs observed rate in each band
2. PR AUC (average precision). For rare events it says more than ROC AUC;
   random = the fire rate (0.086%). Reported pooled over all hidden years and
   as the median of years: with 11-124 fires a year, a mean of yearly values
   is swayed by a few lucky top-ranked commune-days.
3. Confusion table at one warning budget (the 10% highest-scored commune-days,
   about 15 communes a day on average).

Outputs: results/calibration.csv, results/calibration.png

Usage:
    uv run python -m firepredict.calibration
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

from firepredict.compare_communes import history_x, leave_one_year_out, small
from firepredict.dataset import RESULTS_DIR
from firepredict.model_communes import HORIZON, build_panel
from firepredict.value_check import FWI_ALONE

OUT = RESULTS_DIR / "calibration.csv"
FIGURE = RESULTS_DIR / "calibration.png"
BANDS = [0, 0.0003, 0.001, 0.003, 0.01, 1]  # the map's colour bands
BAND_LABELS = ["< 0.03%", "0.03-0.1%", "0.1-0.3%", "0.3-1%", ">= 1%"]
BUDGET = 0.10
DAYS_PER_YEAR = 365.25


def reliability(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    band = pd.cut(p, BANDS, labels=BAND_LABELS, right=False)
    t = pd.DataFrame({"band": band, "y": y, "p": p}).groupby("band", observed=False)
    return pd.DataFrame({"commune-days": t.size(), "fires": t.y.sum(), "predicted": t.p.mean(), "observed": t.y.mean()})


def confusion_at_budget(y: np.ndarray, score: np.ndarray) -> dict:
    warn = np.zeros(len(y), bool)
    warn[np.argsort(-score, kind="stable")[: int(len(y) * BUDGET)]] = True
    tp, fp = int((warn & (y == 1)).sum()), int((warn & (y == 0)).sum())
    fn, tn = int((~warn & (y == 1)).sum()), int((~warn & (y == 0)).sum())
    return {"fires caught (TP)": tp, "false warnings (FP)": fp, "fires missed (FN)": fn,
            "quiet, not warned (TN)": tn, "precision": tp / (tp + fp), "recall": tp / (tp + fn)}  # fmt: skip


def run() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = build_panel()
    models = {
        "climatology": lambda fold, test: (fold.usual_rate / DAYS_PER_YEAR)[test].to_numpy(),
        FWI_ALONE: lambda fold, test: fold.fwi[test].to_numpy(),
        "history x FWI": history_x("fwi"),
        "small model": small,
    }
    scores = {k: v.to_numpy() for k, v in leave_one_year_out(panel, models).items()}
    y = panel.target.to_numpy().astype(int)
    year = (panel.index.get_level_values("date") + pd.Timedelta(days=HORIZON)).year.to_numpy()
    years = np.unique(year)

    rng = np.random.default_rng(0)
    jitter = rng.uniform(0, 1e-9, len(y))  # random tie-breaking for FWI alone
    # Average precision handles tied scores itself, so it is computed without jitter.
    pr = {name: np.array([average_precision_score(y[year == yr], s[year == yr]) for yr in years])
          for name, s in scores.items()}  # fmt: skip
    ranking = {}
    for name, s in scores.items():
        ranking[name] = {"PR AUC (pooled)": average_precision_score(y, s),
                         "PR AUC (median of years)": np.median(pr[name]),
                         "years PR AUC above small model": int((pr[name] > pr["small model"]).sum()),
                         **confusion_at_budget(y, s + jitter)}  # fmt: skip
    ranking = pd.DataFrame(ranking).T

    probs = {name: np.clip(scores[name], 1e-6, 1 - 1e-6) for name in ["climatology", "small model"]}
    clim = brier_score_loss(y, probs["climatology"])
    proba = pd.DataFrame({name: {"Brier": brier_score_loss(y, p), "Brier skill vs climatology": 1 - brier_score_loss(y, p) / clim,
                                 "log loss": log_loss(y, p), "mean predicted": p.mean(), "observed rate": y.mean()}
                          for name, p in probs.items()}).T  # fmt: skip
    table = reliability(y, probs["small model"])
    plot(table, len(years))
    return ranking, proba, table


def plot(table: pd.DataFrame, n_years: int) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    t = table[table["commune-days"] > 0]
    ax.loglog(t.predicted, t.observed, "o-", color="#c9452a", label="small model")
    for _, r in t.iterrows():
        half = 1.96 * np.sqrt(max(r.fires, 1)) / r["commune-days"]  # Poisson 95% range
        ax.plot([r.predicted] * 2, [max(r.observed - half, 1e-6), r.observed + half], color="#c9452a", alpha=0.4)
    lims = [1e-4, 3e-2]
    ax.plot(lims, lims, "--", color="grey", label="perfect calibration")
    ax.set(xlim=lims, ylim=lims, xlabel="predicted chance of a fire start", ylabel="observed share of commune-days",
           title=f"Calibration, {n_years} hidden years")  # fmt: skip
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=150)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    ranking, proba, table = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.concat({"ranking": ranking, "probabilities": proba}).to_csv(OUT)
    print("Ranking and confusion at the 10% budget\n" + ranking.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nProbabilities\n" + proba.to_string(float_format=lambda v: f"{v:.5f}"))
    print("\nReliability (small model)\n" + table.to_string(float_format=lambda v: f"{v:.4%}"))
