"""Does the small commune model travel? Learn on one department, forecast another.

Departments: the Var (83) and the Bouches-du-Rhône (13). Same every-year
protocol as compare_communes: each year Y is hidden in turn, in both
departments at once, so no model ever learns from year Y's fires.

For a hidden year Y and a target department T, models learn on:
  - in-department:  T, other years (as compare_communes)
  - transfer S->T:  the other department S only, other years
  - pooled:         both departments, other years; with or without a
                    department indicator (1 for the second department)
Each department's usual fire rate comes from its own records, years other
than Y (and for a training row, also without the row's own year). Inputs:
  - small:                   the small model (SMALL)
  - small + landscape:       + buildings near wildland, roads in wildland (BD TOPO)
  - no history:              SMALL without the usual rate, so the transferred
                             model uses nothing from T's fire records
  - no history + landscape:  the same, + landscape
Missing inputs are filled with the training rows' medians.

Scores, on T's commune-days of year Y, against the references of value_check
(FWI alone, local FWI alone, history x FWI with T's own history):
  - AUC by year, years better, Wilcoxon test (summary.csv)
  - positive commune-days caught at pooled budgets and daily capacities, with
    95% ranges from resampling years (gains.csv)
  - calibration bands of the small model: in-department, transferred, pooled
    (calibration.csv); scores per year (by_year.csv)

Each department's panel is built once, with that department's settings, and
cached as data/processed/<department>/commune_panel.parquet (delete it to rebuild).

Usage:
    uv run python -m firepredict.transfer          # builds missing panels, then compares
Output: results/transfer/
"""

import os
import subprocess
import sys

import numpy as np
import pandas as pd

from firepredict.calibration import reliability
from firepredict.compare_communes import per_year, rates_without, summary
from firepredict.dataset import PROCESSED_DIR, ROOT
from firepredict.model_communes import HORIZON, SMALL, SMALL_LANDSCAPE, build_panel, fit_small, small_inputs
from firepredict.value_check import gains

DEPARTMENTS = ["83", "13"]
OUT_DIR = ROOT / "results" / "transfer"
COLUMNS = ["target", "fwi", "local_fwi", "local_wb_30d", "orange_period", "red_period",
           "buildings_near_wildland_50m", "roads_in_wildland_km"]  # fmt: skip
NO_HISTORY = [c for c in SMALL if c != "log_usual_rate"]
INPUTS = {
    "small": SMALL,
    "small + landscape": [*SMALL, *SMALL_LANDSCAPE],
    "no history": NO_HISTORY,
    "no history + landscape": [*NO_HISTORY, *SMALL_LANDSCAPE],
}
INDICATOR = "second_department"
REFERENCES = ["FWI alone", "local FWI alone", "history x FWI"]
SHOWN = ["in-department: small", "transfer: small", "transfer: no history + landscape", "pooled: small"]


def panel_path(code: str):
    return ROOT / "data" / "processed" / code / "commune_panel.parquet"


def load_panels() -> dict[str, pd.DataFrame]:
    for code in DEPARTMENTS:
        if not panel_path(code).exists():
            env = {**os.environ, "FIREPREDICT_DEPARTMENT": code}
            subprocess.run([sys.executable, "-m", "firepredict.transfer", "panel"], env=env, check=True)
    return {code: pd.read_parquet(panel_path(code)) for code in DEPARTMENTS}


def target_year(panel: pd.DataFrame) -> np.ndarray:
    return (panel.index.get_level_values("date") + pd.Timedelta(days=HORIZON)).year.to_numpy()


def usual_rates(panel: pd.DataFrame, hidden: int) -> np.ndarray:
    """Usual rate per row from the department's own years, never using `hidden`
    nor the row's own year. A department without labels for `hidden` (the Var
    in 2024) uses all its years, each row without its own."""
    year, insee = target_year(panel), panel.index.get_level_values("insee")
    counts = panel.target.groupby([year, insee]).sum().unstack(fill_value=0)
    if hidden in counts.index:
        rate = rates_without(counts, hidden)
    else:
        rate = (counts.sum() - counts) / (len(counts) - 1)
    return rate.stack().reindex(pd.MultiIndex.from_arrays([year, insee])).to_numpy()


def fit_predict(train: list[tuple[pd.DataFrame, pd.Series]], test: pd.DataFrame, columns: list[str]) -> np.ndarray:
    X = pd.concat([x[columns] for x, _ in train])
    fill = X.median()
    model = fit_small(X.fillna(fill), pd.concat([y for _, y in train]))
    return model.predict_proba(test[columns].fillna(fill))[:, 1]


def run() -> dict[str, dict[str, pd.Series]]:
    """Scores per target department and model, every row scored in its hidden year."""
    panels = load_panels()
    scores = {code: {} for code in DEPARTMENTS}
    years = sorted(set().union(*(target_year(p) for p in panels.values())))
    for hidden in years:
        X, y, test = {}, {}, {}
        for code, panel in panels.items():
            fold = panel.assign(usual_rate=usual_rates(panel, hidden))
            X[code] = small_inputs(fold).assign(**{INDICATOR: float(code != DEPARTMENTS[0])})
            X[code][["fwi", "local_fwi", "usual_rate"]] = fold[["fwi", "local_fwi", "usual_rate"]]
            y[code], test[code] = fold.target, target_year(panel) == hidden
        for target in DEPARTMENTS:
            t = test[target]
            if not t.any():
                continue
            (source,) = [c for c in DEPARTMENTS if c != target]
            own, other = (X[target][~t], y[target][~t]), (X[source][~test[source]], y[source][~test[source]])
            here = X[target][t]
            out = {
                "FWI alone": here.fwi,
                "local FWI alone": here.local_fwi,
                "history x FWI": here.usual_rate * here.fwi,
            }
            for name, columns in INPUTS.items():
                out[f"in-department: {name}"] = fit_predict([own], here, columns)
                out[f"transfer: {name}"] = fit_predict([other], here, columns)
            out["pooled: small"] = fit_predict([own, other], here, SMALL)
            out["pooled: small + department"] = fit_predict([own, other], here, [*SMALL, INDICATOR])
            out["pooled: small + landscape"] = fit_predict([own, other], here, [*SMALL, *SMALL_LANDSCAPE])
            for name, values in out.items():
                scores[target].setdefault(name, []).append(pd.Series(np.asarray(values), index=here.index))
        print(f"  {hidden} done", flush=True)
    return {code: {name: pd.concat(parts).reindex(panels[code].index) for name, parts in s.items()}
            for code, s in scores.items()}  # fmt: skip


def report(code: str, panel: pd.DataFrame, scores: dict[str, pd.Series]) -> list[pd.DataFrame]:
    """Summary, gains, calibration bands and per-year scores for one target department."""
    results = pd.concat({name: per_year(panel, s) for name, s in scores.items()}, names=["model", "year"])
    table = summary(results, "history x FWI")[["roc_auc", "fires caught", "years better (auc)", "p (auc)"]]
    for ref in ["FWI alone", "in-department: small"]:
        table = table.join(summary(results, ref)[["years better (auc)", "p (auc)"]].add_suffix(f" vs {ref}"))
    table = table.rename(columns={"fires caught": "top 15/day caught"})
    table["AUC sd over years"] = results.roc_auc.groupby(level="model").std()
    gain = pd.concat({name: gains(panel, scores, name, REFERENCES) for name in SHOWN}, names=["model", "reference"])
    y = panel.target.to_numpy().astype(int)
    calib = pd.concat({name: reliability(y, np.clip(scores[name].to_numpy(), 1e-6, 1 - 1e-6))
                       for name in ["in-department: small", "transfer: small", "pooled: small"]}, names=["model"])  # fmt: skip
    return [t.assign(department=code) for t in (table, gain, calib, results)]


def main() -> None:
    scores = run()
    panels = load_panels()
    reports = [report(code, panels[code], scores[code]) for code in DEPARTMENTS]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(["summary", "gains", "calibration", "by_year"]):
        pd.concat([r[i] for r in reports]).to_csv(OUT_DIR / f"{name}.csv")
    for code, (table, gain, calib, _) in zip(DEPARTMENTS, reports):
        n = int(panels[code].target.sum())
        print(f"\n=== target {code}: {n} positive commune-days, {len(np.unique(target_year(panels[code])))} hidden years")
        print(table.drop(columns="department").to_string(float_format=lambda v: f"{v:.3f}"))
        print(gain.drop(columns="department").T.to_string())
        print("calibration bands\n" + calib.drop(columns="department").to_string(float_format=lambda v: f"{v:.4%}"))


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 40)
    if sys.argv[1:] == ["panel"]:
        build_panel()[COLUMNS].to_parquet(PROCESSED_DIR / "commune_panel.parquet")
    else:
        main()
