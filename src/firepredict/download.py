"""Download raw data for the department (FIREPREDICT_DEPARTMENT, default 83) into data/raw/<department>/.

Each source gets its own folder. Every downloaded file is recorded in
data/raw/<department>/manifest.json (URL, download time, size, SHA-256) so
that we know exactly which data version a result was built from.

Usage:
    uv run python -m firepredict.download            # all sources
    uv run python -m firepredict.download bdiff      # one source
    uv run python -m firepredict.download --force    # re-download existing files
    FIREPREDICT_DEPARTMENT=13 uv run python -m firepredict.download   # another department
"""

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import certifi
import geopandas as gpd
import pyogrio
import requests

from firepredict.dataset import RAW_DIR
from firepredict.department import CODE

MANIFEST = RAW_DIR / "manifest.json"

DEPARTMENT = CODE
BDIFF_FIRST_YEAR = 2006
BDIFF_LAST_YEAR = 2025  # last complete year

DATA_GOUV_API = "https://www.data.gouv.fr/api/1/datasets"
METEO_DAILY_DATASET = "6569b51ae64326786e4e8e1a"  # Météo-France, données quotidiennes


# --- helpers -----------------------------------------------------------------


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def record(path: Path, url: str, note: str = "") -> None:
    """Add or update one file's entry in the manifest."""
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    manifest[str(path.relative_to(RAW_DIR))] = {
        "url": url,
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "note": note,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def fetch(url: str, dest: Path, force: bool, session=requests, note: str = "", **kwargs) -> None:
    if dest.exists() and not force:
        print(f"  skip (exists) {dest.relative_to(RAW_DIR)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  get {url}")
    with session.get(url, stream=True, timeout=300, **kwargs) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with tmp.open("wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.rename(dest)
    record(dest, url, note)


def data_gouv_resources(dataset_id: str) -> list[dict]:
    r = requests.get(f"{DATA_GOUV_API}/{dataset_id}/", timeout=60)
    r.raise_for_status()
    return r.json()["resources"]


# --- sources -----------------------------------------------------------------


def bdiff_ca_bundle() -> str:
    """The BDIFF server sends the wrong intermediate certificate, so normal
    verification fails. We add the correct, public intermediate (GEANT TLS
    RSA 1, issued by the HARICA root that certifi trusts) to certifi's
    bundle. The full chain is still checked; nothing is disabled."""
    intermediate = Path(__file__).parent / "certs" / "harica-geant-tls-r1.pem"
    bundle = Path(tempfile.gettempdir()) / "firepredict-ca-bundle.pem"
    bundle.write_text(Path(certifi.where()).read_text() + intermediate.read_text())
    return str(bundle)


def download_bdiff(force: bool) -> None:
    """BDIFF fire records for the department.

    The site has no API. Its search form stores the filter in a session
    cookie, and /incendies/zip then exports the current selection as a zip
    (Incendies.csv + two PDFs). We keep the site's default fire type
    ("F" = forest fires), as the benchmark does.
    """
    base = "https://bdiff.agriculture.gouv.fr"
    dest = RAW_DIR / "bdiff" / f"bdiff_{DEPARTMENT}_{BDIFF_FIRST_YEAR}-{BDIFF_LAST_YEAR}.zip"
    session = requests.Session()
    session.verify = bdiff_ca_bundle()
    params = {
        "if[deprts][value]": DEPARTMENT,
        "if[periodeAnnees][anneeDeb]": BDIFF_FIRST_YEAR,
        "if[periodeAnnees][anneeFin]": BDIFF_LAST_YEAR,
        "if[submit]": "",
    }
    if not dest.exists() or force:
        session.get(f"{base}/incendies", params=params, timeout=120).raise_for_status()
    note = f"filter: dept {DEPARTMENT}, years {BDIFF_FIRST_YEAR}-{BDIFF_LAST_YEAR}, site defaults otherwise"
    fetch(f"{base}/incendies/zip", dest, force, session=session, note=note)


def download_meteo(force: bool) -> None:
    """Météo-France daily station data for the department, from 1950 onward.

    Two file families per period: RR-T-Vent (rain, temperature, wind) and
    autres-parametres (humidity, sunshine, etc.). Resource URLs are read from
    the data.gouv.fr API, because the name of the "latest" file changes each
    year.
    """
    for res in data_gouv_resources(METEO_DAILY_DATASET):
        title = res["title"]
        if f"departement_{DEPARTMENT}_" not in title or "avant-1949" in title:
            continue
        url = res["url"]
        fetch(url, RAW_DIR / "meteofrance" / url.rsplit("/", 1)[-1], force, note=title)


def download_meteo_latest(force: bool) -> None:
    """Re-download only the current-period files ("latest" in their URL), which
    Météo-France updates every day. Use before rebuilding a fresh forecast."""
    for res in data_gouv_resources(METEO_DAILY_DATASET):
        url = res["url"]
        if f"Q_{DEPARTMENT}_latest" in url:
            fetch(url, RAW_DIR / "meteofrance" / url.rsplit("/", 1)[-1], True, note=res["title"])


def download_ddtm83_perimeters(force: bool) -> None:
    """DDTM 83 burned-area perimeters (shapefiles, zipped). Var only.

    One dataset covers 1958-2022, then there is one dataset per year.
    Optional in phase 1; used to cross-check BDIFF burned areas.
    """
    if DEPARTMENT != "83":
        print("  Var only, skipped")
        return
    r = requests.get(
        f"{DATA_GOUV_API}/",
        params={"q": "contours feux de forêt Var", "page_size": 100},
        timeout=60,
    )
    r.raise_for_status()
    for ds in r.json()["data"]:
        title = ds["title"]
        if "feux de forêt dans le Var" not in title:
            continue
        atom = [
            res["url"]
            for res in ds["resources"]
            if res["url"].startswith("https://atom.geo-ide.developpement-durable.gouv.fr/")
        ]
        if not atom:
            print(f"  no internet download link for: {title}")
            continue
        years = title.rsplit(" ", 1)[-1] if " en " in title else "1958-2022"
        fetch(atom[0], RAW_DIR / "ddtm83" / f"contours_feux_var_{years}.zip", force, note=title)


def download_school_holidays(force: bool) -> None:
    """French school holidays per zone, one row per day, 1990-2028.

    Listed on data.gouv.fr ("Vacances scolaires par zones"). The official
    Education ministry API only starts in 2017. Each department's zone is set
    in department.py.
    """
    url = "https://raw.githubusercontent.com/AntoineAugusti/vacances-scolaires/master/data.csv"
    fetch(url, RAW_DIR / "calendar" / "vacances_scolaires.csv", force, note="Vacances scolaires par zones")


def download_communes(force: bool) -> None:
    """Commune boundaries + population: geo.api.gouv.fr (IGN boundaries, INSEE population)."""
    url = (
        f"https://geo.api.gouv.fr/departements/{DEPARTMENT}/communes"
        "?fields=nom,code,population,surface&format=geojson&geometry=contour"
    )
    fetch(url, RAW_DIR / "geography" / f"communes_{DEPARTMENT}.geojson", force, note="geo.api.gouv.fr")


def download_geography(force: bool) -> None:
    """Commune boundaries, and Corine Land Cover 2018 for the department.

    Corine: IGN Géoplateforme WFS, polygons in the bounding box of the
    department's communes (Lambert-93, widened by 1 km and rounded out to the
    km), fetched in pages and saved as one GeoJSON.
    """
    download_communes(force)
    communes = RAW_DIR / "geography" / f"communes_{DEPARTMENT}.geojson"

    dest = RAW_DIR / "geography" / f"corine_2018_{DEPARTMENT}.geojson"
    if dest.exists() and not force:
        print(f"  skip (exists) {dest.relative_to(RAW_DIR)}")
        return
    minx, miny, maxx, maxy = gpd.read_file(communes).to_crs(2154).total_bounds
    km = 1000
    bbox = f"{(minx - km) // km * km:.0f},{(miny - km) // km * km:.0f},{-(-(maxx + km) // km) * km:.0f},{-(-(maxy + km) // km) * km:.0f}"
    wfs = (
        "https://data.geopf.fr/wfs/ows?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature"
        "&TYPENAMES=LANDCOVER.CLC18_FR:clc18_fr&OUTPUTFORMAT=application/json"
        f"&BBOX={bbox},urn:ogc:def:crs:EPSG::2154"  # department, Lambert-93
    )
    features, page = [], 1000
    while True:
        print(f"  get Corine features {len(features)}+")
        r = requests.get(f"{wfs}&COUNT={page}&STARTINDEX={len(features)}", timeout=300)
        r.raise_for_status()
        batch = r.json()
        features += batch["features"]
        if len(batch["features"]) < page:
            break
    batch["features"] = features
    dest.write_text(json.dumps(batch))
    record(dest, wfs, note=f"Corine Land Cover 2018, {len(features)} polygons")


BDTOPO_FEED = "https://data.geopf.fr/telechargement/resource/BDTOPO"
BDTOPO_DATE = "2026-09-15"  # same edition for every department when available
BDTOPO_LAYERS = ["batiment", "troncon_de_route", "zone_de_vegetation"]  # the layers landscape.py reads


def bdtopo_edition() -> str:
    """Name of the BD TOPO edition for the department: BDTOPO_DATE if IGN has
    it, else the latest one. Read from IGN's download feed (all themes,
    GeoPackage, Lambert-93)."""
    r = requests.get(BDTOPO_FEED, params={"zone": f"D{int(DEPARTMENT):03d}", "format": "GPKG"}, timeout=60)
    r.raise_for_status()
    editions = sorted(set(re.findall(rf"BDTOPO_[\d-]+_TOUSTHEMES_GPKG_LAMB93_D{int(DEPARTMENT):03d}_[\d-]+", r.text)))
    dated = [e for e in editions if e.endswith(BDTOPO_DATE)]
    return dated[0] if dated else max(editions, key=lambda e: e.rsplit("_", 1)[-1])


def download_bdtopo(force: bool) -> None:
    """IGN BD TOPO for the department (buildings, roads, vegetation), one GeoPackage.

    The edition is read from IGN's feed (see bdtopo_edition). The 7z archive
    is unpacked with bsdtar (libarchive, shipped with macOS); to save disk,
    only the layers in BDTOPO_LAYERS are copied into a smaller GeoPackage,
    and the archive and the full GeoPackage are deleted. The manifest keeps
    the archive's URL, size and SHA-256.
    """
    folder = RAW_DIR / "bdtopo"
    existing = sorted(folder.glob("*.gpkg"))
    if existing and not force:
        print(f"  skip (exists) {existing[0].relative_to(RAW_DIR)}")
        return
    edition = bdtopo_edition()
    url = f"https://data.geopf.fr/telechargement/download/BDTOPO/{edition}/{edition}.7z"
    archive = folder / f"{edition}.7z"
    fetch(url, archive, force, note=f"IGN BD TOPO, department {DEPARTMENT}, Lambert-93, Licence Ouverte Etalab 2.0")
    names = subprocess.run(["bsdtar", "-tf", archive], capture_output=True, text=True, check=True).stdout.split()
    (inner,) = [n for n in names if n.endswith(".gpkg")]
    full = folder / "tmp" / "full.gpkg"  # unpacked archive, deleted below
    full.parent.mkdir(exist_ok=True)
    with full.open("wb") as out:
        subprocess.run(["bsdtar", "-xOf", archive, inner], stdout=out, check=True)
    gpkg = archive.with_suffix(".gpkg")
    keep_layers(full, gpkg)
    full.unlink()
    full.parent.rmdir()
    archive.unlink()
    record(gpkg, url, note=f"layers {', '.join(BDTOPO_LAYERS)} of {inner}, extracted from {archive.name}")
    manifest = json.loads(MANIFEST.read_text())
    manifest[str(archive.relative_to(RAW_DIR))]["note"] += "; deleted after extraction"
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def keep_layers(source: Path, dest: Path) -> None:
    """Copy BDTOPO_LAYERS, with all their columns, from one GeoPackage to a new one."""
    tmp = dest.parent / "tmp" / dest.name  # out of landscape.py's sight until complete
    tmp.parent.mkdir(exist_ok=True)
    tmp.unlink(missing_ok=True)
    for layer in BDTOPO_LAYERS:
        print(f"  copy layer {layer}")
        table = pyogrio.read_dataframe(source, layer=layer, use_arrow=True)
        pyogrio.write_dataframe(table, tmp, layer=layer, driver="GPKG", append=tmp.exists())
    tmp.rename(dest)


SOURCES = {
    "bdiff": download_bdiff,
    "meteo": download_meteo,
    "meteo-latest": download_meteo_latest,
    "ddtm83": download_ddtm83_perimeters,
    "calendar": download_school_holidays,
    "geography": download_geography,
    "communes": download_communes,
    "bdtopo": download_bdtopo,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", nargs="*", choices=list(SOURCES), help="default: all")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()
    for name in args.sources or SOURCES:
        print(f"[{name}]")
        SOURCES[name](args.force)


if __name__ == "__main__":
    main()
