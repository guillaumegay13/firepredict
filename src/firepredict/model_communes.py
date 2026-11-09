"""Model A by commune: "will a fire >= 0.1 ha start in this commune tomorrow?"

One row per commune and issue day D (forecast J+1, operational: data up to D-1).
Inputs:
  - commune:      size, population, land cover, wildland near houses
  - local weather of D-1 from the nearest core station (temperature, rain,
    wind, humidity) and local rain over the last 7 days
  - department state of D-1: FWI system, days since rain
  - fire history: the commune's usual yearly rate (training years, without
    the row's own year). No input uses recent fire records: BDIFF publishes
    a year's fires months later, so a live forecast cannot know them.
  - calendar of the target day

Evaluation on the test years:
  - ROC / PR AUC over all commune-days
  - top-15 capture: each day, rank the 153 communes; share of fire starts
    that fall in the 15 highest-ranked communes (~10%)
References: commune history alone, and commune history x FWI.

Phase 2 adds the burning period (the prefecture's fire-use orders, dated in
department.py; Var: orange = Feb-Mar, red = June-Sept; windy day outside
summer) and each commune's 30-day water balance (rain - evaporation at its
nearest stations), compared as:
  CatBoost (current inputs) | CatBoost + these inputs | a small logistic model

Usage:
    uv run python -m firepredict.model_communes
"""

import numpy as np
import pandas as pd
import xarray as xr
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xclim.indicators.atmos import cffwis_indices

from firepredict.dataset import MIN_FIRE_HA, PROCESSED_DIR, RESULTS_DIR, START, load_bdiff, load_meteo
from firepredict.department import burning_periods
from firepredict.download import BDIFF_LAST_YEAR
from firepredict.features import LATITUDE, calendar
from firepredict.targets import SPLITS, split_of

OUT = RESULTS_DIR / "communes_start.csv"
HORIZON = 1
TOP_K = 15
SEEDS = [0, 1, 2, 3, 4]
STATIC = ["area_km2", "population", "density", "built_up_share", "farmland_share",
          "forest_share", "scrub_share", "wildland_near_houses_km2"]  # fmt: skip
DEPARTMENT = ["fwi", "ffmc", "dmc", "dc", "isi", "bui", "days_since_rain"]
LOCAL = {"tx": "TX", "rr": "RR", "ffm": "FFM", "un": "UN", "etp": "ETPGRILLE"}  # local column -> Météo-France variable
# Inputs added in phase 2 (burning rules, local dryness); "current" models leave them out.
WEATHER_NEW = ["local_etp", "local_wb_30d", "local_fwi", "orange_period", "red_period", "windy_off_season"]
LANDSCAPE = ["wildland_km2", "garrigue_km2", "buildings", "buildings_in_wildland", "buildings_near_wildland_50m",
             "roads_in_wildland_km", "tracks_in_wildland_km"]  # from landscape.py (BD TOPO)
NEW = [*WEATHER_NEW, *LANDSCAPE]
WINDY_MS = 40 / 3.6  # same rule as human_check
SMALL = ["log_usual_rate", "log_local_fwi", "local_wb_30d", "orange_period", "red_period"]
# Landscape inputs, tested on top of SMALL: once commune history is known they
# add nothing (history already reflects where houses and roads meet wildland).
SMALL_LANDSCAPE = ["log_buildings_near_wildland", "log_roads_in_wildland"]


def daily_starts(dates: pd.DatetimeIndex, communes: pd.Index, min_ha: float = MIN_FIRE_HA) -> pd.DataFrame:
    """Fire starts per day (rows) and commune (columns); NaN where labels are unusable."""
    daily = pd.read_parquet(PROCESSED_DIR / "daily.parquet")
    fires = load_bdiff()
    fires = fires[(fires.area_ha >= min_ha) & fires.insee.isin(communes)]
    starts = pd.crosstab(fires.alert_time.dt.normalize(), fires.insee)
    starts = starts.reindex(index=dates, columns=communes, fill_value=0).astype(float)
    starts[~daily.labels_ok.reindex(dates, fill_value=False).to_numpy()] = np.nan
    return starts


def local_weather(dates: pd.DatetimeIndex, communes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Each variable as a (day x commune) table from the commune's nearest station.

    Gaps at a station are filled with the department mean of that day.
    """
    meteo = load_meteo()
    out = {}
    for col, var in LOCAL.items():
        by_station = meteo.pivot_table(index="date", columns="NUM_POSTE", values=var).reindex(dates)
        station = communes[f"station_{var.lower()}"]
        table = by_station[station.to_numpy()].set_axis(communes.index, axis=1)
        out[col] = table.apply(lambda c: c.fillna(by_station.mean(axis=1)))
    out["rr_7d"] = out["rr"].rolling(7).sum()
    out["wb_30d"] = (out["rr"] - out["etp"]).rolling(30).sum()  # rain - evaporation
    out["fwi"] = local_fwi(out)
    return out


def local_fwi(weather: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """FWI for each commune from its nearest stations, same approximation as
    features.fwi (daily max temperature, min humidity, mean wind, daily rain)."""

    def grid(table: pd.DataFrame, units: str) -> xr.DataArray:
        return xr.DataArray(table.to_numpy(), coords={"time": table.index.to_numpy(), "insee": table.columns},
                            dims=("time", "insee"), attrs={"units": units})  # fmt: skip

    *_, fwi = cffwis_indices(
        tas=grid(weather["tx"], "degC"),
        pr=grid(weather["rr"], "mm/d"),
        sfcWind=grid(weather["ffm"], "m/s"),
        hurs=grid(weather["un"], "%"),
        lat=xr.DataArray(LATITUDE, attrs={"units": "degrees_north"}),
        season_method=None,
    )
    return pd.DataFrame(fwi.transpose("time", "insee").values, index=weather["tx"].index, columns=weather["tx"].columns)


def usual_rate(starts: pd.DataFrame) -> pd.DataFrame:
    """Commune's yearly fire rate over training years; for a training year,
    computed without that year so a row never sees its own label."""
    train_years = SPLITS["train"]
    per_year = starts.groupby(starts.index.year).sum().reindex(train_years)
    total = per_year.sum()
    rate = pd.DataFrame(index=starts.index, columns=starts.columns, dtype=float)
    for year in starts.index.year.unique():
        in_year = starts.index.year == year
        if year in train_years:
            rate[in_year] = ((total - per_year.loc[year]) / (len(train_years) - 1)).to_numpy()
        else:
            rate[in_year] = (total / len(train_years)).to_numpy()
    return rate


def build_panel(forecast: bool = False, min_ha: float = MIN_FIRE_HA) -> pd.DataFrame:
    """One row per commune and issue day. With forecast=True, also keep the
    days after the last fire records (target NaN), up to the latest day the
    weather allows."""
    communes = pd.read_parquet(PROCESSED_DIR / "communes.parquet")
    state = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    dates = pd.date_range(START, state.index[-1] + pd.Timedelta(days=1), name="date")
    starts = daily_starts(dates, communes.index, min_ha)
    weather = local_weather(dates, communes)

    # Day x commune tables; "known" ones are shifted so row D holds D-1.
    tables = {f"local_{k}": v.shift(1) for k, v in weather.items()}
    tables["usual_rate"] = usual_rate(starts)
    tables["target"] = starts.shift(-HORIZON)
    panel = pd.concat({k: v.stack(future_stack=True) for k, v in tables.items()}, axis=1)
    panel.index.names = ["date", "insee"]

    days = panel.index.get_level_values("date")
    dept = state[DEPARTMENT].shift(1, freq="D").reindex(days)
    cal = calendar(dates + pd.Timedelta(days=HORIZON)).set_axis(dates).reindex(days)
    landscape = pd.read_parquet(PROCESSED_DIR / "landscape.parquet")[LANDSCAPE]
    static = communes[STATIC].join(landscape).reindex(panel.index.get_level_values("insee"))
    panel = pd.concat(
        [panel, dept.set_axis(panel.index), cal.set_axis(panel.index), static.set_axis(panel.index)], axis=1
    )
    # Burning period of the target day; wind is only known up to D-1.
    windy = state.ffm_max.shift(1, freq="D").reindex(days).to_numpy() > WINDY_MS
    target_day = days + pd.Timedelta(days=HORIZON)
    periods = burning_periods(target_day)
    panel["orange_period"] = periods.orange_period.to_numpy()
    panel["red_period"] = periods.red_period.to_numpy()
    panel["windy_off_season"] = (windy & (panel.red_period == 0)).astype(int)
    panel["split"] = split_of(target_day).to_numpy()
    keep = panel.target.notna() & panel.split.notna()
    if forecast:
        keep |= target_day.year > BDIFF_LAST_YEAR
    panel = panel[keep]
    panel["target"] = panel.target.gt(0).astype(int).where(panel.target.notna())
    return panel.astype({c: "float32" for c in panel.columns if c not in ("target", "split")})


def top_k_capture(panel: pd.DataFrame, score: pd.Series) -> float:
    """Share of fire starts that fall in the TOP_K communes ranked by score that day."""
    rank = score.groupby(level="date").rank(ascending=False, method="first")
    fires = panel.target == 1
    return float((rank[fires] <= TOP_K).mean())


def evaluate(panel: pd.DataFrame, score: pd.Series, **info) -> list[dict]:
    rows = []
    for name in ["val", "test", "test_2025"]:
        m = panel.split == name
        y, s = panel.target[m], score[m]
        rows.append({"split": name, **info, "roc_auc": roc_auc_score(y, s),
                     "pr_auc": average_precision_score(y, s), "top15_capture": top_k_capture(panel[m], s),
                     "fire_commune_days": int(y.sum())})  # fmt: skip
    return rows


def current_features(panel: pd.DataFrame) -> pd.Index:
    return panel.columns.drop(["target", "split", *NEW])


def fit(panel: pd.DataFrame, seed: int, features: pd.Index | None = None) -> CatBoostClassifier:
    """CatBoost on the training years, early-stopped on validation."""
    features = current_features(panel) if features is None else features
    train, val = panel.split == "train", panel.split == "val"
    model = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.05,
        depth=6,
        early_stopping_rounds=150,
        eval_metric="AUC",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(panel.loc[train, features], panel.target[train], eval_set=(panel.loc[val, features], panel.target[val]))
    return model


def small_inputs(panel: pd.DataFrame) -> pd.DataFrame:
    """The few inputs of the small model, with NaN left in place."""
    return pd.DataFrame({
        "log_usual_rate": np.log(panel.usual_rate + 0.01),
        "log_local_fwi": np.log1p(panel.local_fwi),
        "local_wb_30d": panel.local_wb_30d,
        "orange_period": panel.orange_period,
        "red_period": panel.red_period,
        "log_buildings_near_wildland": np.log1p(panel.buildings_near_wildland_50m),
        "log_roads_in_wildland": np.log1p(panel.roads_in_wildland_km),
    })[[*SMALL, *SMALL_LANDSCAPE]]  # fmt: skip


def fit_small(X: pd.DataFrame, y: pd.Series):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X, y)


def small_model(panel: pd.DataFrame) -> pd.Series:
    """Logistic regression on a few inputs, trained on the training years."""
    train = panel.split == "train"
    X = small_inputs(panel)[SMALL]
    X = X.fillna(X[train].median())
    model = fit_small(X[train], panel.target[train])
    print("small model weights (standardised):\n" + pd.Series(model[-1].coef_[0], index=SMALL).round(2).to_string())
    return pd.Series(model.predict_proba(X)[:, 1], index=panel.index)


def run() -> pd.DataFrame:
    panel = build_panel()
    current = current_features(panel)
    variants = {
        "CatBoost": current,
        "CatBoost + landscape": current.append(pd.Index(LANDSCAPE)),
        "CatBoost + landscape + weather/burning": current.append(pd.Index(NEW)),
    }
    print(f"panel: {len(panel):,} commune-days, {int(panel.target.sum())} with a fire start, {len(current)} features")

    rows = []
    rows += evaluate(panel, panel.usual_rate, model="history only")
    rows += evaluate(panel, panel.usual_rate * panel.fwi, model="history x FWI")
    rows += evaluate(panel, panel.usual_rate * panel.local_fwi, model="history x local FWI")
    rows += evaluate(panel, panel.buildings_near_wildland_50m, model="buildings near wildland")
    rows += evaluate(panel, panel.buildings_near_wildland_50m * panel.local_fwi, model="buildings near wildland x local FWI")
    rows += evaluate(panel, small_model(panel), model="small model")
    for name, features in variants.items():
        for seed in SEEDS:
            model = fit(panel, seed, features)
            score = pd.Series(model.predict_proba(panel[features])[:, 1], index=panel.index)
            rows += evaluate(panel, score, model=name, seed=seed)
            if seed == SEEDS[0]:
                importance = pd.Series(model.get_feature_importance(), index=features).nlargest(12)
                print(f"{name}, top features (seed 0):\n" + importance.round(1).to_string())
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    results = run()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT, index=False)
    cols = ["roc_auc", "pr_auc", "top15_capture"]
    summary = results.groupby(["split", "model"])[cols].agg(["mean", "std"]).round(3)
    print(summary.reindex(["val", "test", "test_2025"], level=0).to_string())
    print(results.groupby("split").fire_commune_days.first().to_string())
