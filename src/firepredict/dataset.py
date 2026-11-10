"""Build the daily department table: one row per date, fire labels + department weather.

Output: data/processed/<department>/daily.parquet

Label rules (see results/<department>/audit/data_audit.md for the evidence):
- The main target counts fires of at least MIN_FIRE_HA. BDIFF changed its
  fire form in 2023 and started recording many more tiny fires; this cut-off
  removes most of that artifact. Columns for all fires are kept for
  comparison with the benchmark, which uses every fire.
- Days with no recorded fire are explicit zeros, except in years where the
  labels are unusable (UNRELIABLE_YEARS) or not yet published: there the
  labels are NaN and `labels_ok` is False.

Weather rules:
- Météo-France values flagged doubtful (quality code 2) are dropped. Code 9
  means "filtered" (passed first-level checks, not yet validated) and is kept.
- Only "core" stations are used: stations with at least CORE_COVERAGE of days
  between 2006 and the last date. A fixed set avoids drift in the aggregates
  when stations open or close. Each day uses the core stations that reported.
- Météo-France's daily files are published two days late (on day D they stop
  at D-2), but its hourly file already holds D-1. Days missing from the daily
  files are rebuilt from the hourly file with Météo-France's own daily windows
  (see hourly_to_daily); official daily values replace them once published.

Usage:
    uv run python -m firepredict.dataset
"""

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from firepredict.department import CODE, SETTINGS

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / CODE
PROCESSED_DIR = ROOT / "data" / "processed" / CODE
RESULTS_DIR = ROOT / "results" / CODE

START = "2006-01-01"
MIN_FIRE_HA = 0.1
UNRELIABLE_YEARS = list(SETTINGS.unreliable_years)
CORE_COVERAGE = 0.95
DOUBTFUL = 2  # Météo-France quality code: "donnée douteuse en cours de vérification"

# Météo-France daily variables we use: column -> family file.
WEATHER_VARS = {
    "RR": "RR-T-Vent",  # precipitation (mm)
    "TN": "RR-T-Vent",  # min temperature (°C)
    "TX": "RR-T-Vent",  # max temperature (°C)
    "FFM": "RR-T-Vent",  # mean wind speed at 10 m (m/s)
    "FXI": "RR-T-Vent",  # max instantaneous gust (m/s)
    "DXI": "RR-T-Vent",  # direction of max gust (degrees)
    "UN": "autres-parametres",  # min relative humidity (%)
    "ETPGRILLE": "autres-parametres",  # potential evapotranspiration (mm), Penman-Monteith on Météo-France's grid
}


# --- fires -------------------------------------------------------------------


def load_bdiff() -> pd.DataFrame:
    """One row per fire: date, commune, burned area (ha), cause, data quality."""
    (path,) = (RAW_DIR / "bdiff").glob("bdiff_*.zip")
    raw = zipfile.ZipFile(path).read("Incendies.csv")
    # The first 3 lines are a header block describing the export filter.
    df = pd.read_csv(io.BytesIO(raw), sep=";", skiprows=3, dtype=str)
    return pd.DataFrame(
        {
            "alert_time": pd.to_datetime(df["Date de première alerte"]),
            "insee": df["Code INSEE"],
            "commune": df["Nom de la commune"],
            "area_ha": pd.to_numeric(df["Surface parcourue (m2)"]) / 1e4,
            "cause": df["Nature"],
            "quality_flag": df["Précision de la donnée"],
            "area_precision": df["Précision des surfaces"],
        }
    )


def daily_labels(fires: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    fires = fires.assign(date=fires.alert_time.dt.normalize())
    main = fires[fires.area_ha >= MIN_FIRE_HA].groupby("date")
    every = fires.groupby("date")
    labels = pd.DataFrame(
        {
            "fire_count": main.size(),
            "burned_ha": main.area_ha.sum(),
            "fire_count_all": every.size(),
            "burned_ha_all": every.area_ha.sum(),
        }
    )
    labels = labels.reindex(dates).fillna(0)

    last_label_year = fires.alert_time.dt.year.max()
    labels["labels_ok"] = ~dates.year.isin(UNRELIABLE_YEARS) & (dates.year <= last_label_year)
    labels.loc[~labels.labels_ok, labels.columns != "labels_ok"] = np.nan
    return labels


# --- weather -----------------------------------------------------------------


def load_meteo() -> pd.DataFrame:
    """Long table, one row per station-day, from 2006, doubtful values removed.

    Days after the last day of the daily files are added from the hourly file."""
    daily = load_daily_files()
    last = daily.loc[daily.TX.notna(), "date"].max()
    recent = hourly_to_daily(daily, after=last)
    if recent.empty:
        return daily
    return pd.concat([daily[daily.date <= last], recent], ignore_index=True)


# Hourly variable -> (daily variable, aggregation, hours to shift so that a
# window falls on one calendar day). Météo-France's daily windows, in UTC:
# rain and max temperature from 06h on D to 06h on D+1; min temperature from
# 18h on D-1 to 18h on D; wind, gusts and min humidity from 00h to 24h. An
# hourly value stamped HH covers the hour ending at HH. Checked against the
# official daily values for 2025-2026 in the Var: >= 99.9% of station-days
# agree within 0.05 (mean wind: 98%).
HOURLY_WINDOWS = {
    "RR1": ("RR", "sum", 7),
    "TX": ("TX", "max", 7),
    "TN": ("TN", "min", -5),
    "FF": ("FFM", "mean", 1),
    "FXI": ("FXI", "max", 1),
    "UN": ("UN", "min", 1),
}
MIN_HOURS = 20  # a day needs at least 20 of its 24 hourly values
ETP_DAYS = 7  # potential evaporation is not hourly: use the station's mean of the last 7 days


def hourly_to_daily(daily: pd.DataFrame, after: pd.Timestamp) -> pd.DataFrame:
    """Station-days after `after`, rebuilt from the hourly "latest" file."""
    paths = sorted((RAW_DIR / "meteofrance").glob(f"H_{CODE}_latest*.csv.gz"))
    if not paths:
        return daily.iloc[:0]
    cols = [*HOURLY_WINDOWS, "DXI"]
    h = pd.read_csv(paths[-1], sep=";", dtype={"NUM_POSTE": str},
                    usecols=["NUM_POSTE", "NOM_USUEL", "AAAAMMJJHH", *cols, *("Q" + c for c in cols)])  # fmt: skip
    for c in cols:
        h.loc[h["Q" + c] == DOUBTFUL, c] = np.nan
    h["time"] = pd.to_datetime(h.AAAAMMJJHH.astype(str), format="%Y%m%d%H")
    columns = {}
    for hourly, (name, how, shift) in HOURLY_WINDOWS.items():
        day = (h.time - pd.Timedelta(hours=shift)).dt.normalize()
        grouped = h.assign(date=day).groupby(["NUM_POSTE", "NOM_USUEL", "date"])[hourly]
        columns[name] = grouped.agg(how).where(grouped.count() >= MIN_HOURS)
    # Direction of the day's strongest gust.
    gusts = h.assign(date=(h.time - pd.Timedelta(hours=1)).dt.normalize()).dropna(subset=["FXI"])
    strongest = gusts.loc[gusts.groupby(["NUM_POSTE", "NOM_USUEL", "date"]).FXI.idxmax()]
    columns["DXI"] = strongest.set_index(["NUM_POSTE", "NOM_USUEL", "date"]).DXI
    out = pd.DataFrame(columns).reset_index()
    names = [name for name, _, _ in HOURLY_WINDOWS.values()]
    out = out[(out.date > after) & out[names].notna().any(axis=1)]
    recent_etp = daily[daily.date > after - pd.Timedelta(days=ETP_DAYS)].groupby("NUM_POSTE").ETPGRILLE.mean()
    out["ETPGRILLE"] = out.NUM_POSTE.map(recent_etp)
    return out[daily.columns]


def load_daily_files() -> pd.DataFrame:
    """Long table from the daily files only."""
    frames = []
    for family in sorted(set(WEATHER_VARS.values())):
        cols = [v for v, f in WEATHER_VARS.items() if f == family]
        parts = []
        for path in sorted((RAW_DIR / "meteofrance").glob(f"Q_{CODE}_*_{family}.csv.gz")):
            df = pd.read_csv(
                path,
                sep=";",
                usecols=["NUM_POSTE", "NOM_USUEL", "AAAAMMJJ", *cols, *("Q" + c for c in cols)],
                dtype={"NUM_POSTE": str},
            )
            df["date"] = pd.to_datetime(df.AAAAMMJJ.astype(str))
            parts.append(df[df.date >= START])
        df = pd.concat(parts)
        for c in cols:
            df.loc[df["Q" + c] == DOUBTFUL, c] = np.nan
        frames.append(df.set_index(["NUM_POSTE", "NOM_USUEL", "date"])[cols])
    return pd.concat(frames, axis=1).reset_index()


def core_stations(meteo: pd.DataFrame, var: str) -> pd.Series:
    """Stations reporting `var` on at least CORE_COVERAGE of all days."""
    n_days = meteo.date.nunique()
    coverage = meteo[meteo[var].notna()].groupby(["NUM_POSTE", "NOM_USUEL"]).size() / n_days
    return coverage[coverage >= CORE_COVERAGE]


def circular_mean_deg(deg: pd.Series) -> float:
    rad = np.deg2rad(deg.dropna())
    if rad.empty:
        return np.nan
    return float(np.rad2deg(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360)


def daily_weather(meteo: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Department-level daily weather from core stations.

    Returns the table and, for documentation, the core station names per variable.
    """
    aggregations = {
        "TX": ["mean", "max"],
        "TN": ["mean", "min"],
        "RR": ["mean", "max"],
        "FFM": ["mean", "max"],
        "FXI": ["mean", "max"],
        "UN": ["mean", "min"],
        "ETPGRILLE": ["mean"],
    }
    columns, stations = {}, {}
    for var, funcs in aggregations.items():
        core = core_stations(meteo, var)
        stations[var] = sorted(core.index.get_level_values("NOM_USUEL"))
        values = meteo[meteo.NUM_POSTE.isin(core.index.get_level_values("NUM_POSTE"))]
        grouped = values.groupby("date")[var]
        for f in funcs:
            columns[f"{var.lower()}_{f}"] = grouped.agg(f)
        columns[f"{var.lower()}_n_stations"] = grouped.count()

    # Gust direction is missing on calm days, so no station passes the coverage
    # test; use the gust (FXI) core stations instead.
    core = core_stations(meteo, "FXI")
    stations["DXI"] = stations["FXI"]
    gust_dir = meteo[meteo.NUM_POSTE.isin(core.index.get_level_values("NUM_POSTE"))]
    columns["dxi_mean_deg"] = gust_dir.groupby("date").DXI.agg(circular_mean_deg)
    return pd.DataFrame(columns), stations


# --- build -------------------------------------------------------------------


def build() -> pd.DataFrame:
    meteo = load_meteo()
    weather, stations = daily_weather(meteo)
    # The two Météo-France file families are not updated on the same day; stop
    # at the last day that has temperatures.
    dates = pd.date_range(START, weather.tx_mean.last_valid_index(), freq="D", name="date")
    daily = daily_labels(load_bdiff(), dates).join(weather.reindex(dates))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(PROCESSED_DIR / "daily.parquet")
    lines = [f"{var}: {len(names)} stations: {', '.join(names)}" for var, names in stations.items()]
    (PROCESSED_DIR / "core_stations.txt").write_text("\n".join(lines) + "\n")
    print(f"daily table: {len(daily)} days, {daily.index.min().date()} -> {daily.index.max().date()}")
    print("\n".join(lines))
    return daily


if __name__ == "__main__":
    build()
