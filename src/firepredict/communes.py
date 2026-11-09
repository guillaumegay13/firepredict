"""One row per commune of the department: land cover, people, nearest weather stations, fire history.

Output: data/processed/<department>/communes.parquet (+ communes.geojson for maps)

Land cover comes from Corine Land Cover 2018 (25 ha minimum mapping unit):
  forest        311-313   broad-leaved, coniferous, mixed forest
  scrub         321-324   grassland, moors, garrigue/maquis, transitional woodland
  farmland      2xx
  built-up      1xx
"Wildland near houses" is forest or scrub within WUI_BUFFER_M of built-up land,
a rough proxy for the wildland-urban interface where most fires start.

Usage:
    uv run python -m firepredict.communes
"""

import geopandas as gpd
import numpy as np
import pandas as pd

from firepredict.dataset import MIN_FIRE_HA, PROCESSED_DIR, RAW_DIR, core_stations, load_bdiff, load_meteo
from firepredict.department import CODE
from firepredict.targets import SPLITS

LAMBERT93 = 2154
WUI_BUFFER_M = 200
COVER = {
    "forest": ("311", "312", "313"),
    "scrub": ("321", "322", "323", "324"),
    "farmland": ("2",),
    "built_up": ("1",),
}
STATION_VARS = ["TX", "RR", "FFM", "UN", "ETPGRILLE"]


def load_communes() -> gpd.GeoDataFrame:
    communes = gpd.read_file(RAW_DIR / "geography" / f"communes_{CODE}.geojson").to_crs(LAMBERT93)
    communes = communes.rename(columns={"code": "insee", "nom": "name"}).set_index("insee")
    communes["area_km2"] = communes.area / 1e6
    communes["density"] = communes.population / communes.area_km2
    return communes[["name", "population", "area_km2", "density", "geometry"]]


def land_cover(communes: gpd.GeoDataFrame) -> pd.DataFrame:
    corine = gpd.read_file(RAW_DIR / "geography" / f"corine_2018_{CODE}.geojson").to_crs(LAMBERT93)
    corine["cover"] = None
    for name, prefixes in COVER.items():
        corine.loc[corine.code_18.str.startswith(prefixes), "cover"] = name
    pieces = gpd.overlay(communes.reset_index()[["insee", "geometry"]], corine[["cover", "geometry"]])
    pieces["km2"] = pieces.area / 1e6
    shares = pieces.pivot_table(index="insee", columns="cover", values="km2", aggfunc="sum").fillna(0)
    shares = shares.div(communes.area_km2, axis=0).add_suffix("_share")

    wild = corine[corine.cover.isin(["forest", "scrub"])].dissolve()
    near_houses = corine[corine.cover == "built_up"].buffer(WUI_BUFFER_M).union_all()
    wui = gpd.GeoDataFrame(geometry=wild.intersection(near_houses), crs=LAMBERT93)
    wui = gpd.overlay(communes.reset_index()[["insee", "geometry"]], wui)
    shares["wildland_near_houses_km2"] = (wui.set_index("insee").area / 1e6).groupby(level=0).sum()
    return shares.fillna(0)


def nearest_stations(communes: gpd.GeoDataFrame) -> pd.DataFrame:
    """For each variable, the nearest core station to the commune centroid."""
    meteo = load_meteo()
    coords = pd.concat(
        [pd.read_csv(p, sep=";", usecols=["NUM_POSTE", "LAT", "LON"], dtype={"NUM_POSTE": str})
         for p in (RAW_DIR / "meteofrance").glob(f"Q_{CODE}_*_RR-T-Vent.csv.gz")]
    ).drop_duplicates("NUM_POSTE").set_index("NUM_POSTE")  # fmt: skip
    stations = gpd.GeoDataFrame(
        coords, geometry=gpd.points_from_xy(coords.LON, coords.LAT), crs=4326
    ).to_crs(LAMBERT93)
    centroids = communes.centroid
    out = {}
    for var in STATION_VARS:
        core = stations.loc[core_stations(meteo, var).index.get_level_values("NUM_POSTE")]
        dist = np.array([[c.distance(s) for s in core.geometry] for c in centroids])
        out[f"station_{var.lower()}"] = core.index[dist.argmin(axis=1)]
        out[f"station_{var.lower()}_km"] = dist.min(axis=1) / 1000
    return pd.DataFrame(out, index=communes.index)


def fire_history(communes: gpd.GeoDataFrame) -> pd.DataFrame:
    """Fires >= MIN_FIRE_HA per commune, training years only (no test-year leakage)."""
    fires = load_bdiff()
    train_years = SPLITS["train"]
    fires = fires[(fires.area_ha >= MIN_FIRE_HA) & fires.alert_time.dt.year.isin(train_years)]
    unknown = ~fires.insee.isin(communes.index)
    if unknown.any():
        print(f"  {unknown.sum()} training fires with a commune code not in today's list: {sorted(fires.insee[unknown].unique())}")
    per_commune = fires.groupby("insee").agg(fires=("area_ha", "size"), burned_ha=("area_ha", "sum"))
    per_commune = per_commune.reindex(communes.index).fillna(0)
    return pd.DataFrame(
        {
            "train_fires_per_year": per_commune.fires / len(train_years),
            "train_burned_ha_per_year": per_commune.burned_ha / len(train_years),
        }
    )


def build() -> gpd.GeoDataFrame:
    communes = load_communes()
    table = communes.join([land_cover(communes), nearest_stations(communes), fire_history(communes)])
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    table.drop(columns="geometry").to_parquet(PROCESSED_DIR / "communes.parquet")
    table.to_crs(4326).to_file(PROCESSED_DIR / "communes.geojson", driver="GeoJSON")
    print(f"communes: {len(table)} rows, {table.shape[1] - 1} columns")
    return table


if __name__ == "__main__":
    build()
