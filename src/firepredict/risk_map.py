"""Daily fire-start risk map by commune, as one self-contained HTML page.

For every day of the test years (2023, 2025), each commune is coloured by the
commune model's chance of a fire start that day (forecast issued the morning
before, operational). Communes where a fire really started are marked.
2026 has no fire records yet: it shows forecasts only, up to the latest day
the weather data allows (two days after the last complete Météo-France day).
Run download (meteo) + dataset + features first to get the newest forecast.

Model: the small commune model (local FWI, 30-day dryness, burning period,
buildings and roads in wildland, commune history).
  - 2023 and 2025: every-year test predictions, each year hidden from the
    model (as in compare_communes), so the map shows days it has never seen
  - 2026 forecasts: the frozen model (firepredict.freeze), the same one the
    daily archive uses; its version is shown on the map

Output: results/risk_map.html

Usage:
    uv run python -m firepredict.risk_map
"""

import json

import pandas as pd
import shapely

from firepredict.communes import load_communes
from firepredict.dataset import PROCESSED_DIR, RESULTS_DIR, ROOT, load_meteo
from firepredict.compare_communes import leave_one_year_out, small
from firepredict import model_v3
from firepredict.department import CODE, DEPARTMENTS
from firepredict.freeze import FROZEN_DEPARTMENT, VERSION, predict
from firepredict.model_communes import HORIZON, build_panel

TEMPLATE = ROOT / "src" / "firepredict" / "risk_map_template.html"
OUT = RESULTS_DIR / "risk_map.html"
SPLITS_SHOWN = ["test", "test_2025"]
WIDTH = 1000  # SVG units
SIMPLIFY_M = 120
LOCAL_VARS = ["TX", "RR", "FFM", "UN"]  # weather shown in the pop-ups
PAGES = ["83", "13"]  # departments published on the site, the first at its root


def svg_shapes(communes) -> tuple[list[str], list[tuple[float, float]], float]:
    """Commune outlines as SVG paths, plus a label point each, in a WIDTH-wide box."""
    minx, miny, maxx, maxy = communes.total_bounds
    scale = WIDTH / (maxx - minx)

    def xy(x, y):
        return f"{(x - minx) * scale:.1f} {(maxy - y) * scale:.1f}"

    paths, points = [], []
    for geom in communes.geometry.simplify(SIMPLIFY_M):
        rings = [r for poly in getattr(geom, "geoms", [geom]) for r in [poly.exterior, *poly.interiors]]
        paths.append("".join("M" + "L".join(xy(x, y) for x, y in ring.coords) + "Z" for ring in rings))
        p = shapely.point_on_surface(geom)
        points.append((round((p.x - minx) * scale, 1), round((maxy - p.y) * scale, 1)))
    return paths, points, round((maxy - miny) * scale, 1)


def series(values: pd.Series, digits: int = 1) -> list:
    return [None if pd.isna(v) else round(float(v), digits) for v in values]


def weather(dates: pd.DatetimeIndex, communes: pd.DataFrame) -> dict:
    """Observed weather for the pop-ups: each commune's nearest stations, by day,
    and a few department values. Days not observed yet are null."""
    meteo = load_meteo()
    used = pd.unique(communes[[f"station_{v.lower()}" for v in LOCAL_VARS]].to_numpy().ravel())
    meteo = meteo[meteo.NUM_POSTE.isin(used)]
    names = meteo.groupby("NUM_POSTE").NOM_USUEL.first()
    stations = {}
    for station, rows in meteo.groupby("NUM_POSTE"):
        rows = rows.set_index("date").reindex(dates)
        stations[station] = {"name": names[station].replace("_", " ").title(),
                             **{v: series(rows[v]) for v in LOCAL_VARS}}  # fmt: skip
    state = pd.read_parquet(PROCESSED_DIR / "features.parquet").reindex(dates)
    return {
        "dates": list(dates.strftime("%Y-%m-%d")),
        "stations": stations,
        "fwi": series(state.fwi),
        "gust": series(state.fxi_max),
        "dry_days": series(state.days_since_rain, 0),
    }


def model_for_map() -> str | None:
    """The frozen model whose forecasts the map shows: v3 when frozen and it
    covers the department, else v2 for the Var, else none (past years only)."""
    if model_v3.MODEL_DIR.exists() and CODE in model_v3.DEPARTMENTS:
        return model_v3.VERSION
    return VERSION if CODE == FROZEN_DEPARTMENT else None


def build() -> dict:
    model = model_for_map()
    panel = build_panel(forecast=model is not None)
    labelled = panel[panel.target.notna()]
    loyo = model_v3.OUT_DIR / f"{CODE}_loyo.parquet"
    if model == model_v3.VERSION and loyo.exists():
        past = pd.read_parquet(loyo).p  # v3 every-year test: each year hidden in every department
    else:
        past = leave_one_year_out(labelled, {"small": small})["small"]
    future = panel[panel.target.isna()]
    if model == model_v3.VERSION:
        parts = [past, model_v3.predict(future, CODE)]
    else:
        parts = [past, predict(future)] if model else [past]
    panel = panel.assign(p=pd.concat(parts).reindex(panel.index))
    shown = panel[panel.split.isin(SPLITS_SHOWN) | panel.target.isna()]

    communes = load_communes()
    table = pd.read_parquet(PROCESSED_DIR / "communes.parquet")
    paths, points, height = svg_shapes(communes)
    order = {insee: i for i, insee in enumerate(communes.index)}

    days = []
    for issue, day in shown.groupby(level="date"):
        day = day.droplevel("date").reindex(communes.index)
        days.append({
            "date": (issue + pd.Timedelta(days=HORIZON)).strftime("%Y-%m-%d"),
            "p": [round(v * 10_000) for v in day.p.fillna(0)],  # chance in 1/10,000
            # None = no fire records yet (a forecast)
            "fires": None if day.target.isna().all() else [order[i] for i in day.index[day.target == 1]],
        })  # fmt: skip
    probs = shown.p
    print(f"{len(days)} days; chance per commune-day: median {probs.median():.4f}, "
          f"90% {probs.quantile(0.9):.4f}, 99% {probs.quantile(0.99):.4f}, max {probs.max():.4f}")  # fmt: skip
    first, last = pd.Timestamp(days[0]["date"]), pd.Timestamp(days[-1]["date"])
    return {
        "width": WIDTH,
        "height": height,
        "communes": [
            {"name": row["name"], "d": d, "x": x, "y": y, "pop": int(row.population), "km2": round(row.area_km2, 1),
             "fires_yr": round(row.train_fires_per_year, 2),
             "st": {v: row[f"station_{v.lower()}"] for v in LOCAL_VARS}}
            for (_, row), d, (x, y) in zip(table.loc[communes.index].iterrows(), paths, points)
        ],  # fmt: skip
        "days": days,
        "model": model or "none (past years only)",
        "department": CODE,
        # Department picker: the first department's page is the site root,
        # the others are in a sub-folder named after their code.
        "departments": [{"code": c, "name": s.name,
                         "href": (".." if CODE != PAGES[0] else ".") + ("" if c == PAGES[0] else f"/{c}") + "/"}
                        for c, s in DEPARTMENTS.items() if c in PAGES],  # fmt: skip
        "weather": weather(pd.date_range(first, last, name="date"), table),
    }


if __name__ == "__main__":
    data = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(TEMPLATE.read_text().replace("__DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":"))))
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.1f} MB)")
