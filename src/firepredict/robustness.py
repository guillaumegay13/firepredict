"""Does the small commune model's gain hold when the labels change?

Same every-year protocol as compare_communes, for three label choices:
  - main:          fires >= 0.1 ha, all labelled years
  - without 2023:  2023 removed (BDIFF changed its form that year and every
                   2023 record is flagged "partial or invalid")
  - fires >= 1 ha: a higher cut-off, less sensitive to how small fires are recorded
Reference: commune history x FWI.

Usage:
    uv run python -m firepredict.robustness
"""

import numpy as np
import pandas as pd

from firepredict.compare_communes import REFERENCE, history_x, leave_one_year_out, per_year, small, summary
from firepredict.dataset import ROOT
from firepredict.model_communes import HORIZON, build_panel

OUT = ROOT / "results" / "robustness.csv"


def variants() -> dict[str, pd.DataFrame]:
    main = build_panel()
    year = (main.index.get_level_values("date") + pd.Timedelta(days=HORIZON)).year
    return {"main": main, "without 2023": main[np.asarray(year != 2023)], "fires >= 1 ha": build_panel(min_ha=1.0)}


def run() -> pd.DataFrame:
    models = {REFERENCE: history_x("fwi"), "small model": small}
    rows = {}
    for name, panel in variants().items():
        scores = leave_one_year_out(panel, models)
        results = pd.concat({m: per_year(panel, s) for m, s in scores.items()}, names=["model", "year"])
        table = summary(results).loc["small model"]
        rows[name] = {"positive commune-days": int(results.loc[REFERENCE].fires.sum()),
                      "ref AUC": results.loc[REFERENCE].roc_auc.mean(), **table.to_dict()}  # fmt: skip
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    results = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT)
    cols = ["positive commune-days", "ref AUC", "roc_auc", "auc vs ref", "years better (auc)", "p (auc)", "top15 (pooled)"]
    print(results[cols].to_string(float_format=lambda v: f"{v:.3f}"))
