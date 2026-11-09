"""Leakage tests: a forecast issued on day D must not see data from D or later."""

import numpy as np
import pandas as pd
import pytest

from firepredict.dataset import RAW_DIR
from firepredict.features import build_state, days_since, make_xy

pytestmark = pytest.mark.skipif(
    not (RAW_DIR / "calendar" / "vacances_scolaires.csv").exists(),
    reason="needs school holidays: run python -m firepredict.download calendar",
)


def synthetic_daily(n_days: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-06-01", periods=n_days, freq="D", name="date")
    tx = rng.uniform(20, 38, n_days)
    return pd.DataFrame(
        {
            "tx_mean": tx,
            "tx_max": tx + 2,
            "tn_mean": tx - 12,
            "tn_min": tx - 15,
            "rr_mean": rng.exponential(1, n_days),
            "rr_max": rng.exponential(5, n_days),
            "ffm_mean": rng.uniform(1, 8, n_days),
            "ffm_max": rng.uniform(5, 15, n_days),
            "fxi_mean": rng.uniform(5, 20, n_days),
            "fxi_max": rng.uniform(10, 30, n_days),
            "un_mean": rng.uniform(15, 60, n_days),
            "un_min": rng.uniform(5, 40, n_days),
            "dxi_mean_deg": rng.uniform(0, 360, n_days),
            "fire_count": rng.poisson(0.3, n_days).astype(float),
            "burned_ha": rng.exponential(1, n_days),
            "labels_ok": True,
        },
        index=dates,
    )


def perturb_from(daily: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    """Change every value on `day` and after."""
    changed = daily.copy()
    numeric = changed.columns.drop("labels_ok")
    changed.loc[day:, numeric] = changed.loc[day:, numeric] * 1.7 + 3
    return changed


@pytest.mark.parametrize("horizon", [0, 1, 7])
def test_operational_features_ignore_issue_day_and_later(horizon):
    daily = synthetic_daily()
    issue_day = daily.index[60]
    X, _ = make_xy(build_state(daily), daily, horizon, "operational")
    changed = perturb_from(daily, issue_day)
    X2, _ = make_xy(build_state(changed), changed, horizon, "operational")

    pd.testing.assert_series_equal(X.loc[issue_day], X2.loc[issue_day])
    # Sanity check: the perturbation is visible one day later.
    assert not X.loc[issue_day + pd.Timedelta(days=1)].equals(X2.loc[issue_day + pd.Timedelta(days=1)])


def test_oracle_adds_only_target_day_weather():
    daily = synthetic_daily()
    issue_day, horizon = daily.index[60], 1
    target_day = issue_day + pd.Timedelta(days=horizon)
    X, _ = make_xy(build_state(daily), daily, horizon, "oracle")
    changed = perturb_from(daily, target_day + pd.Timedelta(days=1))
    X2, _ = make_xy(build_state(changed), changed, horizon, "oracle")

    pd.testing.assert_series_equal(X.loc[issue_day], X2.loc[issue_day])


def test_target_is_label_of_target_day():
    daily = synthetic_daily()
    _, y = make_xy(build_state(daily), daily, 1, "operational")
    day = daily.index[30]
    assert y.loc[day, "fire_count"] == daily.loc[day + pd.Timedelta(days=1), "fire_count"]


def test_rows_with_unusable_labels_are_dropped():
    daily = synthetic_daily()
    daily.loc[daily.index[40:50], ["fire_count", "burned_ha"]] = np.nan
    daily.loc[daily.index[40:50], "labels_ok"] = False
    X, y = make_xy(build_state(daily), daily, 0, "operational")
    assert not y.index.isin(daily.index[40:50]).any()


def test_no_input_uses_fire_records():
    """Live forecasts cannot know recent fires, so no input may depend on labels."""
    daily = synthetic_daily()
    changed = daily.assign(fire_count=daily.fire_count * 3 + 1, burned_ha=daily.burned_ha * 5)
    X, _ = make_xy(build_state(daily), daily, 1, "operational")
    X2, _ = make_xy(build_state(changed), changed, 1, "operational")
    pd.testing.assert_frame_equal(X, X2)


def test_days_since():
    event = pd.Series([False, True, False, False, True, False])
    assert days_since(event).tolist()[1:] == [0, 1, 2, 0, 1]
    assert np.isnan(days_since(event).iloc[0])
