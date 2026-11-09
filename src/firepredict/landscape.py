"""Where people meet vegetation, per commune, from IGN BD TOPO (metre precision).

Wildland = BD TOPO vegetation zones of type wood, forest (closed or open,
broadleaf, conifer, mixed) or "lande ligneuse" (garrigue, maquis). Hedges,
vineyards and orchards are left out.
Buildings = residential and undifferentiated buildings (annexes, commercial,
industrial etc. left out), taken at their centroid.

Columns, per commune:
  wildland_km2, garrigue_km2          wildland area, of which lande ligneuse
  buildings                           number of buildings
  buildings_in_wildland               inside a wildland zone
  buildings_near_wildland_50m/200m    within 50 m / 200 m of wildland
  roads_in_wildland_km                paved roads crossing wildland
  tracks_in_wildland_km               tracks, stone roads and paths in wildland

Caveat: one BD TOPO edition (2026) is used for all years, so houses built
after a fire year are counted for that year too.

Output: data/processed/landscape.parquet

Usage:
    uv run python -m firepredict.landscape
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from firepredict.communes import LAMBERT93, load_communes
from firepredict.dataset import PROCESSED_DIR, RAW_DIR

WILDLAND = ["Bois", "Forêt fermée mixte", "Forêt ouverte", "Forêt fermée de conifères",
            "Forêt fermée de feuillus", "Lande ligneuse"]  # fmt: skip
GARRIGUE = ["Lande ligneuse"]
BUILDING_USES = ["Résidentiel", "Indifférencié"]
ROADS = ["Route à 1 chaussée", "Route à 2 chaussées", "Rond-point", "Bretelle", "Type autoroutier"]
TRACKS = ["Chemin", "Route empierrée", "Sentier"]
NEAR_M = [50, 200]


def read(layer: str, columns: list[str]) -> gpd.GeoDataFrame:
    (path,) = (RAW_DIR / "bdtopo").glob("*.gpkg")
    return gpd.read_file(path, layer=layer, columns=columns, engine="pyogrio").to_crs(LAMBERT93)


def in_commune(points: gpd.GeoSeries, communes: gpd.GeoDataFrame) -> pd.Series:
    """INSEE code of the commune containing each point (NaN outside the Var)."""
    joined = gpd.sjoin(gpd.GeoDataFrame(geometry=points), communes.reset_index()[["insee", "geometry"]], predicate="within")
    return joined.insee.groupby(level=0).first().reindex(points.index)


def build() -> pd.DataFrame:
    communes = load_communes()
    out = pd.DataFrame(index=communes.index)

    veg = read("zone_de_vegetation", ["nature"])
    wild = veg[veg.nature.isin(WILDLAND)].reset_index(drop=True)
    pieces = gpd.overlay(communes.reset_index()[["insee", "geometry"]], wild[["nature", "geometry"]])
    pieces["km2"] = pieces.area / 1e6
    out["wildland_km2"] = pieces.groupby("insee").km2.sum()
    out["garrigue_km2"] = pieces[pieces.nature.isin(GARRIGUE)].groupby("insee").km2.sum()
    print(f"wildland: {len(wild):,} zones, {pieces.km2.sum():,.0f} km2")

    buildings = read("batiment", ["usage_1"])
    buildings = buildings[buildings.usage_1.isin(BUILDING_USES)]
    points = buildings.geometry.centroid.reset_index(drop=True)
    commune = in_commune(points, communes)
    out["buildings"] = commune.value_counts()
    tree = wild.sindex
    for name, dist in [("buildings_in_wildland", 0), *((f"buildings_near_wildland_{d}m", d) for d in NEAR_M)]:
        if dist == 0:
            hit = np.unique(tree.query(points, predicate="intersects")[0])
        else:
            hit = np.unique(tree.query(points, predicate="dwithin", distance=dist)[0])
        out[name] = commune.iloc[hit].value_counts()
    print(f"buildings: {len(points):,}")

    roads = read("troncon_de_route", ["nature"])
    for name, kinds in [("roads_in_wildland_km", ROADS), ("tracks_in_wildland_km", TRACKS)]:
        lines = roads[roads.nature.isin(kinds)].geometry.reset_index(drop=True)
        i_line, i_wild = tree.query(lines, predicate="intersects")
        clipped = shapely.intersection(lines.values[i_line], wild.geometry.values[i_wild])
        # Keep line parts only (a road touching a zone edge gives a point), then
        # split the pieces at commune borders, so each commune gets its own length.
        clipped = gpd.GeoSeries(clipped, crs=LAMBERT93).explode(index_parts=False)
        clipped = clipped[clipped.geom_type == "LineString"]
        pieces = gpd.overlay(gpd.GeoDataFrame(geometry=clipped.reset_index(drop=True), crs=LAMBERT93),
                             communes.reset_index()[["insee", "geometry"]], keep_geom_type=True)  # fmt: skip
        out[name] = (pieces.length / 1000).groupby(pieces.insee).sum()
    return out.fillna(0)


if __name__ == "__main__":
    table = build()
    table.to_parquet(PROCESSED_DIR / "landscape.parquet")
    pd.set_option("display.width", 200)
    print(table.describe().round(1).T.to_string())
    print("\ncorrelation with fires per year (training years):")
    fires = pd.read_parquet(PROCESSED_DIR / "communes.parquet").train_fires_per_year
    print(table.corrwith(fires).round(2).sort_values(ascending=False).to_string())
