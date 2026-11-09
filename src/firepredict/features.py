"""Features, built in two layers so that leakage rules live in one place.

1. build_state(daily): one row per day t, holding only what is observed on
   day t or before (weather, rain history, FWI).
   No input is built from recent fire records: BDIFF publishes a year's fires
   months later, so a live forecast could not know them.
   Saved to data/processed/<department>/features.parquet.

2. make_xy(state, daily, horizon, setting): model inputs for a forecast
   issued on the morning of day D, once Météo-France's day D-1 is complete.
     - known state     -> state of D-1                (both settings)
     - calendar        -> calendar of the target day  (known in advance)
     - target weather  -> weather + FWI of the target day (oracle only:
                          a stand-in for a perfect weather forecast)
     - target          -> fire label of the target day D + horizon

Usage:
    uv run python -m firepredict.features
"""

import holidays
import numpy as np
import pandas as pd
import xarray as xr
from xclim.indicators.atmos import cffwis_indices

from firepredict import department
from firepredict.dataset import PROCESSED_DIR, RAW_DIR

LATITUDE = department.SETTINGS.latitude  # centre of the department; sets day length in DMC/DC

WEATHER_COLS = [
    "tx_mean", "tx_max", "tn_mean", "tn_min", "rr_mean", "rr_max",
    "ffm_mean", "ffm_max", "fxi_mean", "fxi_max", "un_mean", "un_min",
]  # fmt: skip
RAIN_WINDOWS = [3, 5, 7, 9]
SIGNIFICANT_RAIN_MM = 2.0  # department-mean rain that counts as "rain"
DROUGHT_WINDOWS = [30, 60, 90]
SETTINGS = ("operational", "oracle")


# --- layer 1: daily state ----------------------------------------------------


def fwi(daily: pd.DataFrame) -> pd.DataFrame:
    """FWI system from daily weather, run continuously from the first day.

    The FWI is defined from noon observations. We only have daily values, so
    we use the usual approximation:
      noon temperature  -> daily max temperature (tx_mean)
      noon humidity     -> daily min relative humidity (un_mean)
      noon wind speed   -> daily mean wind speed (ffm_mean)
      24 h rain         -> daily rain (rr_mean)
    Inputs are averaged over the core stations first, then FWI is computed
    once for the department.

    No fire-season start/stop: in Provence fires happen all year, so the codes
    run every day (xclim season_method=None). The first weeks of 2006 are a
    spin-up from the default start values and should not be trusted.

    Each value on day t uses weather up to day t only, plus the codes of
    day t-1.
    """

    def series(column: str, units: str) -> xr.DataArray:
        return xr.DataArray(
            daily[column].to_numpy(),
            coords={"time": daily.index.to_numpy()},
            dims="time",
            attrs={"units": units},
        )

    dc, dmc, ffmc, isi, bui, fwi_ = cffwis_indices(
        tas=series("tx_mean", "degC"),
        pr=series("rr_mean", "mm/d"),
        sfcWind=series("ffm_mean", "m/s"),
        hurs=series("un_mean", "%"),
        lat=xr.DataArray(LATITUDE, attrs={"units": "degrees_north"}),
        season_method=None,
    )
    out = pd.DataFrame(
        {
            "ffmc": ffmc.values,
            "dmc": dmc.values,
            "dc": dc.values,
            "isi": isi.values,
            "bui": bui.values,
            "fwi": fwi_.values,
        },
        index=daily.index,
    )
    # Daily Severity Rating (Van Wagner 1970): a nonlinear rescaling of FWI.
    out["dsr"] = 0.0272 * out.fwi**1.77
    return out


def days_since(event: pd.Series) -> pd.Series:
    """0 on an event day, then 1, 2, ... until the next event. NaN before the first."""
    groups = event.cumsum()
    out = event.groupby(groups).cumcount().astype(float)
    out[groups == 0] = np.nan
    return out


def weather_state(daily: pd.DataFrame) -> pd.DataFrame:
    state = daily[WEATHER_COLS].copy()
    # Direction is an angle: 350° and 10° are close, so encode it as sin/cos.
    # NaN on the few days no core station reported a gust direction.
    rad = np.deg2rad(daily.dxi_mean_deg)
    state["gust_dir_sin"] = np.sin(rad)
    state["gust_dir_cos"] = np.cos(rad)
    for w in RAIN_WINDOWS:
        state[f"rr_sum_{w}d"] = daily.rr_mean.rolling(w).sum()
    state["days_since_rain"] = days_since(daily.rr_mean >= SIGNIFICANT_RAIN_MM)
    return state.join(fwi(daily))


def seasonal_normal(values: pd.Series, years: list[int], half_window: int = 15) -> pd.Series:
    """Average for each day of the year over `years`, smoothed over +/- half_window days."""
    train = values[values.index.year.isin(years)]
    by_day = train.groupby(train.index.day_of_year).mean().reindex(range(1, 367)).interpolate()
    wrapped = pd.concat([by_day.iloc[-half_window:], by_day, by_day.iloc[:half_window]])
    smooth = wrapped.rolling(2 * half_window + 1, center=True).mean().iloc[half_window:-half_window]
    return pd.Series(smooth.reindex(values.index.day_of_year).to_numpy(), index=values.index)


def drought_state(daily: pd.DataFrame, normal_years: list[int]) -> pd.DataFrame:
    """Long-term dryness from rain and evaporation, over windows ending on day t.

    Water balance = rain - potential evapotranspiration (ETP, which includes
    sun, heat, wind and humidity). Negative = the ground loses water.
    Anomalies compare each value with its normal for that time of year,
    computed on `normal_years` only (training years, no test-year leakage).
    """
    balance = daily.rr_mean - daily.etpgrille_mean
    state = {}
    for w in DROUGHT_WINDOWS:
        state[f"rr_sum_{w}d"] = daily.rr_mean.rolling(w).sum()
        state[f"water_balance_{w}d"] = balance.rolling(w).sum()
    state["days_since_rain_10mm"] = days_since(daily.rr_mean >= 10)
    state = pd.DataFrame(state)
    for col in list(state.columns):
        state[f"{col}_anomaly"] = state[col] - seasonal_normal(state[col], normal_years)
    return state


def build_state(daily: pd.DataFrame) -> pd.DataFrame:
    return weather_state(daily)


# --- layer 2: model inputs ---------------------------------------------------


def calendar(dates: pd.DatetimeIndex) -> pd.DataFrame:
    public = holidays.France(years=range(dates.year.min(), dates.year.max() + 1))
    school = pd.read_csv(
        RAW_DIR / "calendar" / "vacances_scolaires.csv", parse_dates=["date"], index_col="date"
    )[department.SETTINGS.school_zone]
    return pd.DataFrame(
        {
            "day_of_week": dates.day_of_week,
            "month": dates.month,
            "day_of_year": dates.day_of_year,
            "public_holiday": dates.isin(pd.to_datetime(list(public))).astype(int),
            "school_holiday": school.reindex(dates).astype(int).to_numpy(),
            "high_season": dates.month.isin([7, 8]).astype(int),
        },
        index=dates,
    )


def make_xy(
    state: pd.DataFrame, daily: pd.DataFrame, horizon: int, setting: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inputs X and labels y, one row per issue date D (the index).

    y holds fire_count and burned_ha of day D + horizon. Rows whose target
    day has no usable label are dropped.
    """
    if setting not in SETTINGS:
        raise ValueError(f"setting must be one of {SETTINGS}")
    issue = state.index
    target = issue + pd.Timedelta(days=horizon)

    def on_target_day(df: pd.DataFrame) -> pd.DataFrame:
        return df.reindex(target).set_axis(issue)

    X = state.shift(1).join(calendar(target).set_axis(issue))
    if setting == "oracle":
        target_weather = weather_state(daily).columns
        X = X.join(on_target_day(state[target_weather]).add_prefix("target_"))

    y = on_target_day(daily[["fire_count", "burned_ha", "labels_ok"]])
    keep = y.labels_ok.eq(True)
    return X[keep], y.loc[keep, ["fire_count", "burned_ha"]]


def build() -> pd.DataFrame:
    daily = pd.read_parquet(PROCESSED_DIR / "daily.parquet")
    state = build_state(daily)
    state.to_parquet(PROCESSED_DIR / "features.parquet")
    print(f"daily state: {state.shape[1]} columns, {len(state)} days")
    return state


if __name__ == "__main__":
    build()
