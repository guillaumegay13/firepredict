"""Freeze the small commune model and record its 2026 forecasts.

The model's inputs were chosen after looking at results across all years, so
the every-year test is exploratory and somewhat optimistic. A cleaner test is
a year whose fire records the model has never seen: 2026. BDIFF publishes a
year's fires months later, so we freeze the model, save every 2026 forecast
it makes, and record a SHA-256 fingerprint of that file. When the 2026 records
come out, the frozen forecasts are scored as they are, with no change.

This is an outcome-blinded retrospective test, not a live trial: the 2026
forecasts were all computed on the freeze date, from weather downloaded that
day. The fingerprint proves the file has not changed; the date it was made
public (git history) proves when it existed. Live forecasts are archived day
by day by firepredict.forecast.

Training: every labelled year (2006-2025 without 2024). A commune's usual rate
uses all labelled years without the row's own year; days after the last
labelled year use all of them (saved in usual_rates.csv for later use).

Output: models/<VERSION>/
  model.joblib        scikit-learn pipeline (standardise + logistic regression)
  usual_rates.csv     each commune's usual yearly rate, for forecasts
  metadata.json       inputs, weights, fill values, training data, versions, fingerprints
  forecasts_2026.csv  chance of a fire start per commune and day of 2026 so far

An existing version is never overwritten: change VERSION to freeze a new one.

Usage:
    uv run python -m firepredict.freeze
"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone

import joblib
import pandas as pd

from firepredict.dataset import RAW_DIR, ROOT
from firepredict.download import BDTOPO_EDITION
from firepredict.model_communes import HORIZON, SMALL, build_panel, fit_small, small_inputs

VERSION = "small_commune_v2"
MODELS = ROOT / "models"
TEST_YEAR = 2026


def target_year(panel: pd.DataFrame) -> pd.Index:
    return (panel.index.get_level_values("date") + pd.Timedelta(days=HORIZON)).year


def usual_rate_all_years(panel: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Usual rate per row (without the row's own labelled year), and per commune for forecasts."""
    year, insee = target_year(panel), panel.index.get_level_values("insee")
    labelled = panel.target.notna().to_numpy()
    counts = panel.target[labelled].groupby([year[labelled], insee[labelled]]).sum().unstack(fill_value=0)
    total, n = counts.sum(), len(counts)
    own = ((total - counts) / (n - 1)).stack().reindex(pd.MultiIndex.from_arrays([year, insee])).to_numpy()
    for_forecasts = total / n
    rate = pd.Series(own, index=panel.index).fillna(pd.Series(for_forecasts.reindex(insee).to_numpy(), index=panel.index))
    return rate, for_forecasts.rename("usual_rate")


def predict(panel: pd.DataFrame, version: str = VERSION) -> pd.Series:
    """Chance of a fire start for forecast rows, with a frozen model and its saved preprocessing."""
    folder = MODELS / version
    model = joblib.load(folder / "model.joblib")
    metadata = json.loads((folder / "metadata.json").read_text())
    rates = pd.read_csv(folder / "usual_rates.csv", dtype={"insee": str}).set_index("insee").usual_rate
    panel = panel.assign(usual_rate=rates.reindex(panel.index.get_level_values("insee")).to_numpy())
    X = small_inputs(panel)[metadata["inputs"]].fillna(pd.Series(metadata["fill_values"]))
    return pd.Series(model.predict_proba(X)[:, 1], index=panel.index)


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    out = MODELS / VERSION
    if out.exists():
        raise SystemExit(f"{out} already exists: frozen versions are never overwritten. Change VERSION.")
    panel = build_panel(forecast=True)
    panel["usual_rate"], rates = usual_rate_all_years(panel)
    labelled = panel.target.notna()
    X = small_inputs(panel)[SMALL]
    fill = X[labelled].median()
    model = fit_small(X[labelled].fillna(fill), panel.target[labelled])

    out.mkdir(parents=True)
    joblib.dump(model, out / "model.joblib")
    rates.rename_axis("insee").reset_index().to_csv(out / "usual_rates.csv", index=False)
    metadata = {
        "version": VERSION,
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
        "question": "chance that at least one fire >= 0.1 ha starts in the commune on the target day; "
                    "issued the morning before, from data up to two days before the target day",
        "inputs": SMALL,
        "weights_standardised": dict(zip(SMALL, model[-1].coef_[0].round(4).tolist())),
        "fill_values": fill.round(6).to_dict(),
        "training_years": sorted(set(target_year(panel)[labelled.to_numpy()])),
        "training_rows": int(labelled.sum()),
        "training_positive_commune_days": int(panel.target[labelled].sum()),
        "bdtopo_edition": BDTOPO_EDITION,
        "raw_data": {k: {"sha256": v["sha256"], "downloaded_at": v["downloaded_at"]}
                     for k, v in json.loads((RAW_DIR / "manifest.json").read_text()).items()},
    }  # fmt: skip
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")

    # The 2026 forecasts go through predict(), the same path live forecasts use.
    future = (target_year(panel) == TEST_YEAR) & ~labelled.to_numpy()
    chance = predict(panel[future], VERSION)
    day = (chance.index.get_level_values("date") + pd.Timedelta(days=HORIZON)).strftime("%Y-%m-%d")
    forecasts = pd.DataFrame({"target_day": day, "insee": chance.index.get_level_values("insee"), "chance": chance.round(8).to_numpy()})
    forecasts.to_csv(out / "forecasts_2026.csv", index=False)
    metadata["model_sha256"] = sha256(out / "model.joblib")
    metadata["forecasts_2026"] = {"kind": "outcome-blinded retrospective test", "first_day": forecasts.target_day.min(),
                                  "last_day": forecasts.target_day.max(), "rows": len(forecasts),
                                  "sha256": sha256(out / "forecasts_2026.csv")}  # fmt: skip
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    return metadata


if __name__ == "__main__":
    m = freeze()
    keys = ["version", "frozen_at", "git_commit", "weights_standardised", "training_positive_commune_days", "forecasts_2026"]
    print(json.dumps({k: m[k] for k in keys}, indent=2, ensure_ascii=False))
