"""Two checks on the human side of fire starts (fires >= 0.1 ha, 2006-2025 without 2024).

1. More people, more fires? Holidays, the tourist peak and weekends, compared
   at the same weather danger. For each group of days:
       observed fire starts / expected fire starts
   where "expected" gives each day the average number of starts of all days
   in its FWI band. A ratio of 1.3 means 30% more fires than the weather
   alone would suggest.

2. Fire causes (BDIFF "Nature"): how each cause relates to the FWI, the
   season, the week and the time of day.

3. Burning rules (Var prefectoral orders of 16 May 2013 and 2 November 2025):
   owners may burn brush in the green period (wind < 40 km/h) and orange
   period (no wind), only between 5h and 13h; never in the red period
   (1 June - 30 Sept, and any day with wind > 40 km/h). Fire starts by
   period and cause vs the FWI expectation, and alert hours by period.
   The prefect's extra red days (drought) are not in this calendar.

Usage:
    uv run python -m firepredict.human_check
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from firepredict.dataset import MIN_FIRE_HA, PROCESSED_DIR, UNRELIABLE_YEARS, load_bdiff
from firepredict.features import calendar

FIRST_DAY = "2006-03-01"  # after the FWI spin-up
FWI_BANDS = [0, 5, 15, 30, 50, np.inf]
WINDY_MS = 40 / 3.6  # 40 km/h mean wind, at the windiest core station
PERIODS = ["green", "orange", "red (season)", "red (wind)"]
CAUSE_GROUPS = {
    "burning-type (individuals + works)": ["Involontaire (particulier)", "Involontaire (travaux)"],
    "arson": ["Malveillance"],
    "other": ["Accidentelle", "Naturelle"],
}
HOURS = {"0-5h": (0, 4), "5-13h (burning allowed)": (5, 12), "13-18h": (13, 17), "18-24h": (18, 23)}
CAUSES = {
    "Involontaire (particulier)": "accident, individuals",
    "Malveillance": "arson",
    "Involontaire (travaux)": "works",
    "Accidentelle": "accident, infrastructure",
    "Naturelle": "lightning",
}


def days() -> pd.DataFrame:
    daily = pd.read_parquet(PROCESSED_DIR / "daily.parquet")
    state = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    keep = daily.labels_ok.eq(True) & (daily.index >= FIRST_DAY)
    d = daily.loc[keep, ["fire_count"]].join(state[["fwi", "ffm_max"]]).join(calendar(daily.index[keep]))
    d["band"] = pd.cut(d.fwi, FWI_BANDS, right=False)
    d["expected"] = d.groupby("band", observed=True).fire_count.transform("mean")
    return d


def ratio(d: pd.DataFrame, mask: pd.Series) -> dict:
    obs, exp = d.fire_count[mask].sum(), d.expected[mask].sum()
    half = 1.96 * np.sqrt(obs)  # Poisson approximation
    return {"days": int(mask.sum()), "fires": int(obs), "expected": round(exp, 1),
            "ratio": round(obs / exp, 2), "95% range": f"{(obs - half) / exp:.2f}-{(obs + half) / exp:.2f}"}  # fmt: skip


def people_check() -> pd.DataFrame:
    d = days()
    peak = d.month.isin([7, 8])
    groups = {
        "school holidays, outside July-Aug": d.school_holiday.eq(1) & ~peak,
        "school days, outside July-Aug": d.school_holiday.eq(0) & ~peak,
        "July-Aug (tourist peak)": peak,
        "June + Sept (fire season, fewer tourists)": d.month.isin([6, 9]),
        "weekends": d.day_of_week.ge(5),
        "weekdays": d.day_of_week.lt(5),
        "public holidays": d.public_holiday.eq(1),
    }
    return pd.DataFrame({name: ratio(d, m) for name, m in groups.items()}).T


def cause_check() -> pd.DataFrame:
    fires = load_bdiff()
    fires = fires[(fires.area_ha >= MIN_FIRE_HA) & ~fires.alert_time.dt.year.isin(UNRELIABLE_YEARS)]
    fires = fires[fires.alert_time >= FIRST_DAY]
    day = fires.alert_time.dt.normalize()
    d = days()
    fires = fires[day.isin(d.index)]
    day = fires.alert_time.dt.normalize()
    fwi = d.fwi.reindex(day).to_numpy()
    cal = calendar(pd.DatetimeIndex(day))

    rows = {}
    for code, name in {**CAUSES, None: "all fires"}.items():
        m = (fires.cause == code).to_numpy() if code else np.ones(len(fires), bool)
        fire_day = d.index.isin(day[m])
        hours = fires.alert_time[m].dt.hour
        rows[name] = {
            "fires": int(m.sum()),
            "FWI AUC": round(roc_auc_score(fire_day, d.fwi), 2),
            "median FWI": round(float(np.median(fwi[m])), 1),
            "June-Sept": f"{cal.month[m].isin([6, 7, 8, 9]).mean():.0%}",
            "weekend": f"{(cal.day_of_week[m] >= 5).mean():.0%}",
            "alert 8h-18h": f"{hours.between(8, 17).mean():.0%}",
            "alert 18h-24h": f"{hours.between(18, 23).mean():.0%}",
            ">= 10 ha": f"{(fires.area_ha[m] >= 10).mean():.0%}",
            "burned ha": int(fires.area_ha[m].sum()),
        }  # fmt: skip
    return pd.DataFrame(rows).T


def burning_period(d: pd.DataFrame) -> pd.Series:
    period = pd.Series("green", index=d.index)
    period[d.month.isin([2, 3])] = "orange"
    period[d.ffm_max > WINDY_MS] = "red (wind)"
    period[d.month.isin([6, 7, 8, 9])] = "red (season)"
    return period


def fires_by_cause() -> pd.DataFrame:
    fires = load_bdiff()
    fires = fires[(fires.area_ha >= MIN_FIRE_HA) & (fires.alert_time >= FIRST_DAY)]
    return fires.assign(day=fires.alert_time.dt.normalize(), hour=fires.alert_time.dt.hour)


def rules_check() -> tuple[pd.DataFrame, pd.DataFrame]:
    d = days()
    d["period"] = burning_period(d)
    fires = fires_by_cause()
    fires = fires[fires.day.isin(d.index)]
    fires["period"] = d.period.reindex(fires.day).to_numpy()

    rows = {}
    for group, codes in CAUSE_GROUPS.items():
        count = fires[fires.cause.isin(codes)].groupby("day").size().reindex(d.index, fill_value=0)
        expected = count.groupby(d.band, observed=True).transform("mean")
        for period in PERIODS:
            m = d.period == period
            rows[(group, period)] = {"days": int(m.sum()), "fires": int(count[m].sum()),
                                     "expected": round(expected[m].sum(), 1),
                                     "ratio": round(count[m].sum() / expected[m].sum(), 2)}  # fmt: skip
    ratios = pd.DataFrame(rows).T

    burning = fires[fires.cause.isin(CAUSE_GROUPS["burning-type (individuals + works)"])]
    hours = {}
    for period in PERIODS:
        h = burning.hour[burning.period == period]
        hours[period] = {"fires": len(h), **{k: f"{h.between(a, b).mean():.0%}" for k, (a, b) in HOURS.items()}}
    return ratios, pd.DataFrame(hours).T


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print("1. Fire starts vs what the FWI alone would expect (ratio 1 = as expected)\n")
    print(people_check().to_string())
    print("\n2. Fire causes\n")
    print(cause_check().to_string())
    ratios, hours = rules_check()
    print("\n3a. Fire starts by burning period and cause vs the FWI expectation\n")
    print(ratios.to_string())
    print("\n3b. Alert hour of burning-type fires (individuals + works), by period\n")
    print(hours.to_string())
