"""Download raw data for the Var (83) into data/raw/.

Each source gets its own folder. Every downloaded file is recorded in
data/raw/manifest.json (URL, download time, size, SHA-256) so that we know
exactly which data version a result was built from.

Usage:
    uv run python -m firepredict.download            # all sources
    uv run python -m firepredict.download bdiff      # one source
    uv run python -m firepredict.download --force    # re-download existing files
"""

import argparse
import hashlib
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import certifi
import requests

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
MANIFEST = RAW_DIR / "manifest.json"

DEPARTMENT = "83"
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
    """BDIFF fire records for the Var.

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
    """Météo-France daily station data, department 83, from 1950 onward.

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
    """DDTM 83 burned-area perimeters (shapefiles, zipped).

    One dataset covers 1958-2022, then there is one dataset per year.
    Optional in phase 1; used to cross-check BDIFF burned areas.
    """
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
    Education ministry API only starts in 2017. The Var (académie de Nice) is
    in zone B over the whole period.
    """
    url = "https://raw.githubusercontent.com/AntoineAugusti/vacances-scolaires/master/data.csv"
    fetch(url, RAW_DIR / "calendar" / "vacances_scolaires.csv", force, note="Vacances scolaires par zones")


def download_geography(force: bool) -> None:
    """Commune boundaries + population, and Corine Land Cover 2018, for the Var.

    Communes: geo.api.gouv.fr (IGN boundaries, INSEE population).
    Corine: IGN Géoplateforme WFS, polygons in the Var bounding box (Lambert-93),
    fetched in pages and saved as one GeoJSON.
    """
    url = (
        f"https://geo.api.gouv.fr/departements/{DEPARTMENT}/communes"
        "?fields=nom,code,population,surface&format=geojson&geometry=contour"
    )
    fetch(url, RAW_DIR / "geography" / f"communes_{DEPARTMENT}.geojson", force, note="geo.api.gouv.fr")

    dest = RAW_DIR / "geography" / f"corine_2018_{DEPARTMENT}.geojson"
    if dest.exists() and not force:
        print(f"  skip (exists) {dest.relative_to(RAW_DIR)}")
        return
    wfs = (
        "https://data.geopf.fr/wfs/ows?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature"
        "&TYPENAMES=LANDCOVER.CLC18_FR:clc18_fr&OUTPUTFORMAT=application/json"
        "&BBOX=915000,6210000,1045000,6315000,urn:ogc:def:crs:EPSG::2154"  # Var, Lambert-93
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


BDTOPO_EDITION = "BDTOPO_3-5_TOUSTHEMES_GPKG_LAMB93_D083_2026-09-15"


def download_bdtopo(force: bool) -> None:
    """IGN BD TOPO for the Var (buildings, roads, vegetation), one GeoPackage.

    Editions are listed at https://data.geopf.fr/telechargement/resource/BDTOPO
    (filter zone=D083, format=GPKG). The 7z archive is unpacked with bsdtar
    (libarchive, shipped with macOS) and only the .gpkg is kept.
    """
    url = f"https://data.geopf.fr/telechargement/download/BDTOPO/{BDTOPO_EDITION}/{BDTOPO_EDITION}.7z"
    archive = RAW_DIR / "bdtopo" / f"{BDTOPO_EDITION}.7z"
    fetch(url, archive, force, note="IGN BD TOPO, Var, Lambert-93, Licence Ouverte Etalab 2.0")
    gpkg = archive.with_suffix(".gpkg")
    if gpkg.exists() and not force:
        return
    names = subprocess.run(["bsdtar", "-tf", archive], capture_output=True, text=True, check=True).stdout.split()
    (inner,) = [n for n in names if n.endswith(".gpkg")]
    with gpkg.open("wb") as out:
        subprocess.run(["bsdtar", "-xOf", archive, inner], stdout=out, check=True)
    record(gpkg, url, note=f"extracted from {archive.name}: {inner}")


SOURCES = {
    "bdiff": download_bdiff,
    "meteo": download_meteo,
    "meteo-latest": download_meteo_latest,
    "ddtm83": download_ddtm83_perimeters,
    "calendar": download_school_holidays,
    "geography": download_geography,
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
