"""Version 3: one small commune model shared by several departments.

Inputs: the five of the small model (SMALL), plus
  - rate_x_summer: log usual rate x June-September. Busy communes concentrate
    their fires in summer more than quiet ones; without this term the model
    gives every commune the same seasonal shape (it underestimated the Var's
    busiest communes in summer by half, and overestimated Aix-en-Provence in
    winter).
  - one indicator per department after the first (dept_13, ...): it sets each
    department's level, so the chances are calibrated per department. The
    transfer test showed rankings travel between departments but probabilities
    do not.
The model is a single logistic regression trained on every department's fires.

Commands:
  evaluate  every-year test: each year is hidden in all departments at once;
            the model learns on the other years of every department. Writes
            results/v3/summary.csv and the hidden-year predictions used by the
            map (results/v3/<department>_loyo.parquet).
  freeze    train on every labelled year of every department and save
            models/small_commune_v3, with each department's 2026 forecasts and
            their SHA-256 (an outcome-blinded retrospective test). Never
            overwrites an existing version.
  forecast  archive today's forecast for every department in
            forecasts/v3/<department>/<target day>.csv (never overwritten).

Each department's commune table is built in a separate process with that
department's settings (see department.py) and cached in
data/processed/<department>/v3_panel.parquet; `evaluate` and `freeze` rebuild
it, `forecast` too (the weather changes every day).

Usage:
    uv run python -m firepredict.model_v3 evaluate
    uv run python -m firepredict.model_v3 freeze
    uv run python -m firepredict.model_v3 forecast
"""

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from firepredict.calibration import reliability
from firepredict.compare_communes import per_year
from firepredict.dataset import ROOT
from firepredict.model_communes import HORIZON, SMALL, build_panel, fit_small, small_inputs
from firepredict.transfer import usual_rates

DEPARTMENTS = ["83", "13"]
VERSION = "small_commune_v3"
MODEL_DIR = ROOT / "models" / VERSION
OUT_DIR = ROOT / "results" / "v3"
ARCHIVE = ROOT / "forecasts" / "v3"
TEST_YEAR = 2026
COLUMNS = ["target", "split", "fwi", "local_fwi", "local_wb_30d", "orange_period", "red_period",
           "buildings_near_wildland_50m", "roads_in_wildland_km"]  # small_inputs reads the landscape too
INDICATORS = [f"dept_{code}" for code in DEPARTMENTS[1:]]
INPUTS = [*SMALL, "rate_x_summer", *INDICATORS]


def panel_path(code: str):
    return ROOT / "data" / "processed" / code / "v3_panel.parquet"


def build_panels() -> dict[str, pd.DataFrame]:
    """Each department's commune table, with forecast rows, built in its own process."""
    for code in DEPARTMENTS:
        env = {**os.environ, "FIREPREDICT_DEPARTMENT": code}
        subprocess.run([sys.executable, "-m", "firepredict.model_v3", "panel"], env=env, check=True)
    return {code: pd.read_parquet(panel_path(code)) for code in DEPARTMENTS}


def target_year(panel: pd.DataFrame) -> np.ndarray:
    return (panel.index.get_level_values("date") + pd.Timedelta(days=HORIZON)).year.to_numpy()


def inputs(panel: pd.DataFrame, code: str, usual_rate) -> pd.DataFrame:
    X = small_inputs(panel.assign(usual_rate=usual_rate))[SMALL]
    X["rate_x_summer"] = X.log_usual_rate * X.red_period
    for name in INDICATORS:
        X[name] = float(name == f"dept_{code}")
    return X[INPUTS]


def rates_all_years(panel: pd.DataFrame) -> tuple[np.ndarray, pd.Series]:
    """Usual rate per row (labelled rows without their own year; later rows all
    years), and per commune for forecasts."""
    labelled = panel.target.notna().to_numpy()
    year, insee = target_year(panel), panel.index.get_level_values("insee")
    counts = panel.target[labelled].groupby([year[labelled], insee[labelled]]).sum().unstack(fill_value=0)
    total, n = counts.sum(), len(counts)
    own = ((total - counts) / (n - 1)).stack().reindex(pd.MultiIndex.from_arrays([year, insee])).to_numpy()
    forecast = (total / n).rename("usual_rate")
    return np.where(labelled, own, forecast.reindex(insee).to_numpy()), forecast


def evaluate() -> pd.DataFrame:
    panels = {code: p[p.target.notna()] for code, p in build_panels().items()}
    years = sorted(set().union(*(target_year(p) for p in panels.values())))
    scores = {code: [] for code in DEPARTMENTS}
    for hidden in years:
        X, y, test = {}, {}, {}
        for code, panel in panels.items():
            X[code] = inputs(panel, code, usual_rates(panel, hidden))
            y[code], test[code] = panel.target, target_year(panel) == hidden
        train = pd.concat([X[c][~test[c]] for c in DEPARTMENTS])
        fill = train.median()
        model = fit_small(train.fillna(fill), pd.concat([y[c][~test[c]] for c in DEPARTMENTS]))
        for code in DEPARTMENTS:
            if test[code].any():
                here = X[code][test[code]]
                scores[code].append(pd.Series(model.predict_proba(here.fillna(fill))[:, 1], index=here.index))
        print(f"  {hidden} done", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for code, panel in panels.items():
        v3 = pd.concat(scores[code]).reindex(panel.index)
        v3.rename("p").to_frame().to_parquet(OUT_DIR / f"{code}_loyo.parquet")
        rates = usual_rates(panel, -1)  # reference only: each row without its own year
        reference = pd.Series(rates * panel.fwi.to_numpy(), index=panel.index)
        r_v3, r_ref = per_year(panel, v3), per_year(panel, reference)
        d = r_v3.roc_auc - r_ref.roc_auc
        band = reliability(panel.target.to_numpy(), v3.to_numpy())
        rows.append({"department": code, "AUC v3": r_v3.roc_auc.mean(), "AUC history x FWI": r_ref.roc_auc.mean(),
                     "years better": f"{(d > 0).sum()}/{len(d)}", "p": wilcoxon(d).pvalue,
                     "top15/day v3": int(r_v3.caught.sum()), "top15/day history x FWI": int(r_ref.caught.sum()),
                     "positive commune-days": int(panel.target.sum()),
                     **{f"band {b}: predicted / observed": f"{row.predicted:.3%} / {row.observed:.3%}"
                        for b, row in band.iterrows()}})  # fmt: skip
    summary = pd.DataFrame(rows).set_index("department")
    summary.to_csv(OUT_DIR / "summary.csv")
    return summary


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    if MODEL_DIR.exists():
        raise SystemExit(f"{MODEL_DIR} already exists: frozen versions are never overwritten. Change VERSION.")
    panels = build_panels()
    X, y, rates = {}, {}, []
    for code, panel in panels.items():
        row_rates, forecast_rates = rates_all_years(panel)
        X[code], y[code] = inputs(panel, code, row_rates), panel.target
        rates.append(forecast_rates.rename_axis("insee").reset_index().assign(department=code))
    labelled = {code: y[code].notna() for code in DEPARTMENTS}
    train = pd.concat([X[c][labelled[c]] for c in DEPARTMENTS])
    fill = train.median()
    model = fit_small(train.fillna(fill), pd.concat([y[c][labelled[c]] for c in DEPARTMENTS]))

    MODEL_DIR.mkdir(parents=True)
    joblib.dump(model, MODEL_DIR / "model.joblib")
    pd.concat(rates)[["department", "insee", "usual_rate"]].to_csv(MODEL_DIR / "usual_rates.csv", index=False)
    metadata = {
        "version": VERSION,
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
        "departments": DEPARTMENTS,
        "question": "chance that at least one fire >= 0.1 ha starts in the commune on the target day; "
                    "issued the morning before, from data up to two days before the target day",
        "inputs": INPUTS,
        "weights_standardised": dict(zip(INPUTS, model[-1].coef_[0].round(4).tolist())),
        "fill_values": fill.round(6).to_dict(),
        "training_positive_commune_days": {c: int(y[c][labelled[c]].sum()) for c in DEPARTMENTS},
        "training_years": {c: sorted(set(target_year(panels[c])[labelled[c].to_numpy()].tolist())) for c in DEPARTMENTS},
    }  # fmt: skip
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")

    future = []
    for code, panel in panels.items():
        rows = panel[(target_year(panel) == TEST_YEAR) & panel.target.isna().to_numpy()]
        chance = predict(rows, code)
        future.append(pd.DataFrame({"department": code, "target_day": target_day(rows), "insee": rows.index.get_level_values("insee"),
                                    "chance": chance.round(8).to_numpy()}))  # fmt: skip
    forecasts = pd.concat(future)
    forecasts.to_csv(MODEL_DIR / "forecasts_2026.csv", index=False)
    metadata["model_sha256"] = sha256(MODEL_DIR / "model.joblib")
    metadata["forecasts_2026"] = {"kind": "outcome-blinded retrospective test", "rows": len(forecasts),
                                  "first_day": forecasts.target_day.min(), "last_day": forecasts.target_day.max(),
                                  "sha256": sha256(MODEL_DIR / "forecasts_2026.csv")}  # fmt: skip
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    return metadata


def target_day(rows: pd.DataFrame) -> pd.Index:
    return (rows.index.get_level_values("date") + pd.Timedelta(days=HORIZON)).strftime("%Y-%m-%d")


def predict(rows: pd.DataFrame, code: str) -> pd.Series:
    """Chance of a fire start for forecast rows of one department, with the frozen v3 model."""
    model = joblib.load(MODEL_DIR / "model.joblib")
    metadata = json.loads((MODEL_DIR / "metadata.json").read_text())
    rates = pd.read_csv(MODEL_DIR / "usual_rates.csv", dtype={"department": str, "insee": str})
    rates = rates[rates.department == code].set_index("insee").usual_rate
    X = inputs(rows, code, rates.reindex(rows.index.get_level_values("insee")).to_numpy())
    X = X[metadata["inputs"]].fillna(pd.Series(metadata["fill_values"]))
    return pd.Series(model.predict_proba(X)[:, 1], index=rows.index)


def forecast() -> None:
    for code, panel in build_panels().items():
        issue = panel.index.get_level_values("date").max()
        rows = panel.xs(issue, level="date", drop_level=False)
        chance = predict(rows, code)
        table = pd.DataFrame({"target_day": target_day(rows), "issued_on": issue.strftime("%Y-%m-%d"),
                              "weather_up_to": (issue - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                              "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                              "model": VERSION, "department": code, "insee": rows.index.get_level_values("insee"),
                              "chance": chance.round(8).to_numpy()}).sort_values("chance", ascending=False)  # fmt: skip
        path = ARCHIVE / code / f"{table.target_day.iloc[0]}.csv"
        if path.exists():
            print(f"{path.relative_to(ROOT)} already archived, left unchanged")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)
        print(f"wrote {path.relative_to(ROOT)}: expected starts {table.chance.sum():.2f}")


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    command = sys.argv[1] if len(sys.argv) > 1 else "evaluate"
    if command == "panel":
        from firepredict.department import CODE

        build_panel(forecast=True)[COLUMNS].to_parquet(panel_path(CODE))
    elif command == "evaluate":
        print(evaluate().T.to_string(float_format=lambda v: f"{v:.4f}"))
    elif command == "freeze":
        m = freeze()
        keys = ["version", "frozen_at", "weights_standardised", "training_positive_commune_days", "forecasts_2026"]
        print(json.dumps({k: m[k] for k in keys}, indent=2, ensure_ascii=False))
    elif command == "forecast":
        forecast()
    else:
        raise SystemExit(f"unknown command {command}: evaluate, freeze or forecast")
