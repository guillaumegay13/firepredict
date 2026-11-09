"""Smoothed commune fire rates: a hierarchical spatial Poisson model (BYM).

With about 6 fires per commune in 20 years, a commune's own history is mostly
luck. This model estimates each commune's daily rate of fire starts from
three sources at once:
  - its landscape (BD TOPO: buildings and roads in or near wildland, ...),
  - its neighbours (communes that share a border),
  - its own history,
and weighs them by how much evidence each commune has (partial pooling).

    fire-start days[c, y] ~ Poisson(days[y] * rate[c] * year[y])
    log rate[c] = a + X[c] . b + s_spatial * phi[c] + s_own * theta[c]
    phi ~ ICAR(neighbours)   theta ~ Normal(0, 1)   log year ~ Normal(0, s_year)

For the every-year test, the model is fitted once per hidden year without that
year (the year effect of the hidden year is then its average, 1). Rates of
the training rows use all other years, own year included: a small optimistic
bias on training rows only, which can only make the downstream models trust
this input more than they should.

Output: data/processed/spatial_rates.parquet, one column per hidden year
(and "all" for a fit on every labelled year), one row per commune; values are
expected fire-start days per day.

Usage:
    uv run python -m firepredict.model_spatial
"""

import arviz as az
import geopandas as gpd
import numpy as np
import pandas as pd
import pymc as pm

from firepredict.communes import load_communes
from firepredict.dataset import PROCESSED_DIR, START
from firepredict.model_communes import HORIZON, daily_starts

OUT = PROCESSED_DIR / "spatial_rates.parquet"
COVARIATES = ["buildings_near_wildland_50m", "buildings_in_wildland", "roads_in_wildland_km",
              "tracks_in_wildland_km", "wildland_km2", "garrigue_km2", "buildings"]  # fmt: skip


def neighbours(communes: gpd.GeoDataFrame) -> np.ndarray:
    """Adjacency matrix: 1 where two communes share a border (10 m tolerance)."""
    pairs = gpd.sjoin(communes[["geometry"]].reset_index(drop=True).assign(geometry=lambda g: g.buffer(10)),
                      communes[["geometry"]].reset_index(drop=True), predicate="intersects")  # fmt: skip
    W = np.zeros((len(communes), len(communes)))
    W[pairs.index, pairs.index_right] = 1
    np.fill_diagonal(W, 0)
    return W


def yearly_counts(communes: pd.Index) -> tuple[pd.DataFrame, pd.Series]:
    """Days with a fire start per (year, commune), and labelled days per year."""
    features = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    dates = pd.date_range(START, features.index[-1], name="date")
    starts = daily_starts(dates, communes)
    starts = starts[starts.notna().all(axis=1) & (starts.index >= pd.Timestamp(START) + pd.Timedelta(days=HORIZON))]
    fire_days = (starts > 0).groupby(starts.index.year).sum()
    return fire_days, starts.groupby(starts.index.year).size()


def covariates(communes: pd.Index) -> np.ndarray:
    land = pd.read_parquet(PROCESSED_DIR / "landscape.parquet")[COVARIATES]
    table = pd.read_parquet(PROCESSED_DIR / "communes.parquet")[["area_km2", "population"]]
    X = np.log1p(land.join(table).reindex(communes))
    return ((X - X.mean()) / X.std()).to_numpy()


def fit(counts: pd.DataFrame, days: pd.Series, X: np.ndarray, W: np.ndarray, seed: int = 0) -> pd.Series:
    """Posterior mean daily rate per commune (year effect at its average)."""
    n_years, n_communes = counts.shape
    with pm.Model():
        a = pm.Normal("a", -8, 3)
        b = pm.Normal("b", 0, 0.5, shape=X.shape[1])
        s_spatial = pm.HalfNormal("s_spatial", 1)
        s_own = pm.HalfNormal("s_own", 1)
        s_year = pm.HalfNormal("s_year", 1)
        phi = pm.ICAR("phi", W=W)
        theta = pm.Normal("theta", 0, 1, shape=n_communes)
        year = pm.Normal("year", 0, 1, shape=n_years)
        log_rate = pm.Deterministic("log_rate", a + pm.math.dot(X, b) + s_spatial * phi + s_own * theta)
        mu = np.log(days.to_numpy())[:, None] + log_rate[None, :] + s_year * year[:, None]
        pm.Poisson("fires", mu=pm.math.exp(mu), observed=counts.to_numpy())
        trace = pm.sample(1000, tune=2000, chains=4, target_accept=0.95, random_seed=seed, progressbar=False)
    rhat = float(az.rhat(trace, var_names=["log_rate"])["log_rate"].max())
    if rhat > 1.01:
        print(f"  warning: max R-hat of the commune rates is {rhat:.3f}")
    log_rate = trace.posterior["log_rate"].stack(sample=("chain", "draw")).values
    # Mean of the rate (not of the log), with the year effect averaged over its prior.
    s_year = trace.posterior["s_year"].stack(sample=("chain", "draw")).values
    return pd.Series(np.exp(log_rate + s_year**2 / 2).mean(axis=1), index=counts.columns)


def build() -> pd.DataFrame:
    communes = load_communes()
    W = neighbours(communes)
    assert (W.sum(axis=1) > 0).all(), "every commune needs a neighbour for the ICAR term"
    counts, days = yearly_counts(communes.index)
    X = covariates(communes.index)
    rates = {}
    for hidden in [*counts.index, "all"]:
        keep = counts.index != hidden
        rates[hidden] = fit(counts[keep], days[keep], X, W)
        print(f"  {hidden}: mean {rates[hidden].mean() * 365:.2f} fire days per commune-year", flush=True)
    return pd.DataFrame(rates).set_axis([str(c) for c in rates], axis=1)


if __name__ == "__main__":
    rates = build()
    rates.to_parquet(OUT)
    print(f"wrote {OUT.name}: {rates.shape}")
