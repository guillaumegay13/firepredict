"""Department settings: folders, burning calendar, FWI latitude."""

import geopandas as gpd
import pandas as pd
import pytest

from firepredict.dataset import PROCESSED_DIR, RAW_DIR, RESULTS_DIR
from firepredict.department import CODE, DEPARTMENTS, burning_periods


def test_folders_are_per_department():
    for folder in (RAW_DIR, PROCESSED_DIR, RESULTS_DIR):
        assert folder.name == CODE


def test_var_calendar_is_the_month_rule():
    days = pd.date_range("2006-01-01", "2026-12-31")
    periods = burning_periods(days, "83")
    assert (periods.orange_period == days.month.isin([2, 3])).all()
    assert (periods.red_period == days.month.isin([6, 7, 8, 9])).all()


def test_bouches_du_rhone_orange_period_ends_with_the_2013_order():
    days = pd.to_datetime(["2013-02-15", "2013-03-31", "2013-04-01", "2014-02-15", "2014-07-01"])
    periods = burning_periods(days, "13")
    assert periods.orange_period.tolist() == [1, 1, 0, 0, 0]
    assert periods.red_period.tolist() == [0, 0, 0, 0, 1]


@pytest.mark.parametrize("code", list(DEPARTMENTS))
def test_latitude_is_the_centroid(code):
    path = RAW_DIR.parent / code / "geography" / f"communes_{code}.geojson"
    if not path.exists():
        pytest.skip(f"needs {path.name}: run python -m firepredict.download geography")
    communes = gpd.read_file(path).to_crs(2154)
    centroid = gpd.GeoSeries([communes.union_all().centroid], crs=2154).to_crs(4326)
    assert abs(centroid.y[0] - DEPARTMENTS[code].latitude) < 0.05
