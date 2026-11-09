"""Do long-term dryness features rank fire days better than the FWI's drought codes?

Each feature is scored with ROC AUC (0.5 = random, 1 = perfect) on the fire
day's own value, for fire days of increasing size, 2006-2025 without 2024.
Two views:
  - all year:  includes the season effect (summer is dry and has fires)
  - June-Sept: within the fire season only, where the season can't help

Usage:
    uv run python -m firepredict.drought_check
"""

import pandas as pd
from sklearn.metrics import roc_auc_score

from firepredict.dataset import MIN_FIRE_HA, PROCESSED_DIR, RESULTS_DIR, load_bdiff
from firepredict.features import drought_state
from firepredict.targets import SPLITS

OUT = RESULTS_DIR / "drought_check.csv"
FIRST_DAY = "2006-04-01"  # 90-day windows need 3 months of history
SIZES_HA = [0, 1, 10, 100]
SUMMER = [6, 7, 8, 9]
# Score sign: higher score = drier = more risk.
CANDIDATES = {
    "FWI": ("fwi", 1),
    "Drought Code (FWI, ~50 days)": ("dc", 1),
    "Build-up Index (FWI)": ("bui", 1),
    "days since rain >= 2 mm": ("days_since_rain", 1),
    "rain, last 9 days": ("rr_sum_9d", -1),
    "rain, last 30 / 60 / 90 days": ("rr_sum_{w}d", -1),
    "water balance, last 30 / 60 / 90 days": ("water_balance_{w}d", -1),
    "  vs normal for the season": ("water_balance_{w}d_anomaly", -1),
    "rain vs normal, last 30 / 60 / 90 days": ("rr_sum_{w}d_anomaly", -1),
    "days since rain >= 10 mm": ("days_since_rain_10mm", 1),
    "  vs normal for the season ": ("days_since_rain_10mm_anomaly", 1),
}


def run() -> pd.DataFrame:
    daily = pd.read_parquet(PROCESSED_DIR / "daily.parquet")
    state = pd.read_parquet(PROCESSED_DIR / "features.parquet").join(drought_state(daily, SPLITS["train"]))
    keep = daily.labels_ok.eq(True) & (daily.index >= FIRST_DAY)
    state, daily = state[keep], daily[keep]
    largest = largest_fire(daily.index)

    rows = []
    for view, months in [("all year", range(1, 13)), ("June-Sept", SUMMER)]:
        m = state.index.month.isin(months)
        for size in SIZES_HA:
            fire_day = (largest[m] >= max(size, MIN_FIRE_HA)).astype(int)
            label = f">= {size} ha" if size else "any fire"
            for name, (col, sign) in CANDIDATES.items():
                cols = [col.format(w=w) for w in (30, 60, 90)] if "{w}" in col else [col]
                aucs = []
                for c in cols:
                    ok = state.loc[m, c].notna()
                    aucs.append(roc_auc_score(fire_day[ok], sign * state.loc[m, c][ok]))
                rows.append({"view": view, "size": label, "feature": name,
                             "auc": " / ".join(f"{a:.2f}" for a in aucs), "best": max(aucs),
                             "fire_days": int(fire_day.sum())})  # fmt: skip
    return pd.DataFrame(rows)


def largest_fire(dates: pd.DatetimeIndex) -> pd.Series:
    """Size of the day's largest fire (ha), 0 on days without a fire."""
    fires = load_bdiff()
    return fires.groupby(fires.alert_time.dt.normalize()).area_ha.max().reindex(dates).fillna(0)


if __name__ == "__main__":
    results = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT, index=False)
    for view in ["all year", "June-Sept"]:
        r = results[results.view == view]
        wide = r.pivot(index="feature", columns="size", values="auc").reindex(CANDIDATES)
        wide = wide[["any fire", ">= 1 ha", ">= 10 ha", ">= 100 ha"]]
        days = r.groupby("size").fire_days.first()
        print(f"\n{view}  (fire days: " + ", ".join(f"{k} {v}" for k, v in days.items()) + ")")
        print(wide.to_string())
