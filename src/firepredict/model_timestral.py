"""Timestral zero-shot: Mistral AI's time-series model, used without training.

For each issue day D (rolling origin), the model sees the daily fire count
(fires >= 0.1 ha) up to D-1 and forecasts D (J0) and D+1 (J+1) as 21
quantiles. Inputs:
  - target:            daily fire count (NaN where labels are unusable)
  - past covariates:   weather and FWI, known up to D-1
  - known covariates:  calendar flags (0/1), plus, in the oracle setting,
                       weather and FWI of the forecast days
  - categorical known: month, day of week

Warning rule: warn if the forecast at quantile level q is >= 0.5 (at least
"one fire" after rounding). Validation (2021) picks q.

Usage:
    uv run python -m firepredict.model_timestral          # both sizes
    uv run python -m firepredict.model_timestral 28M      # one size
"""

import numpy as np
import pandas as pd
from timestral import TimestralForecaster

from firepredict.dataset import PROCESSED_DIR, RESULTS_DIR
from firepredict.evaluate import scores
from firepredict.features import calendar
from firepredict.targets import SPLITS, RiskClasses, split_of

OUT_DIR = RESULTS_DIR / "timestral"
SIZES = ["mistralai/Timestral-28M", "mistralai/Timestral-209M"]
HORIZON = 2  # forecast steps: D (J0) and D+1 (J+1)
# ~5.6 years of history per forecast. The full 17 years fits the model but is
# 2.5x slower on CPU; FWI's longest memory (DC) is a few months.
CONTEXT_DAYS = 2048
BATCH = 32
PAST = ["tx_mean", "un_min", "ffm_mean", "fxi_max", "rr_mean", "fwi", "ffmc", "dc"]
FLAGS = ["school_holiday", "public_holiday", "high_season"]
CATEGORIES = ["month", "day_of_week"]
TEST_SPLITS = ["val", "test", "test_2025"]


def inputs(setting: str) -> dict[str, pd.DataFrame]:
    """Full daily series, extended by HORIZON days so the last origins have a horizon."""
    daily = pd.read_parquet(PROCESSED_DIR / "daily.parquet")
    state = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    dates = pd.date_range(daily.index[0], daily.index[-1] + pd.Timedelta(days=HORIZON), name="date")
    cal = calendar(dates)
    weather = state[PAST].reindex(dates)
    return {
        "target": daily.fire_count.reindex(dates).to_frame(),
        "past": weather if setting == "operational" else weather.iloc[:, :0],
        "known": cal[FLAGS].join(weather) if setting == "oracle" else cal[FLAGS],
        "categorical": cal[CATEGORIES],
    }


def forecast(forecaster: TimestralForecaster, data: dict, origins: pd.DatetimeIndex) -> np.ndarray:
    """Quantiles (origin, step, quantile) for forecasts issued on each origin day."""
    dates = data["target"].index
    out = []
    for start in range(0, len(origins), BATCH):
        ends = [dates.get_loc(d) for d in origins[start : start + BATCH]]  # first forecast day

        def window(df: pd.DataFrame, extra: int) -> np.ndarray:
            # (batch, rows, time): the last CONTEXT_DAYS before each origin, plus `extra`
            values = df.to_numpy().T
            return np.stack([values[:, end - CONTEXT_DAYS : end + extra] for end in ends])

        kwargs = {"prediction_length": HORIZON}
        if data["past"].shape[1]:
            kwargs["past_covariates"] = window(data["past"], 0)
        fc = forecaster.forecast(
            window(data["target"], 0),
            known_covariates=window(data["known"], HORIZON),
            categorical_known_covariates=window(data["categorical"], HORIZON),
            **kwargs,
        )
        out.append(fc.quantiles[:, 0].numpy())
    return np.concatenate(out), fc.quantile_levels


def pinball(y: np.ndarray, q: np.ndarray, levels) -> float:
    """Mean quantile (pinball) loss over all levels; lower is better."""
    diff = y[:, None] - q
    return float(np.mean(np.maximum(np.asarray(levels) * diff, (np.asarray(levels) - 1) * diff)))


def run(sizes: list[str]) -> pd.DataFrame:
    daily = pd.read_parquet(PROCESSED_DIR / "daily.parquet")
    risk = RiskClasses().fit(daily.fire_count[split_of(daily.index) == "train"])
    years = [y for name in TEST_SPLITS for y in SPLITS[name]]
    origins = daily.index[daily.index.year.isin(years)]

    rows = []
    for size in sizes:
        forecaster = TimestralForecaster.from_pretrained(size)
        for setting in ["operational", "oracle"]:
            q, levels = forecast(forecaster, inputs(setting), origins)
            for step, horizon in enumerate(["J0", "J+1"]):
                target = origins + pd.Timedelta(days=step)
                truth = daily.fire_count.reindex(target).set_axis(origins)
                split = pd.Series(split_of(target).to_numpy(), index=origins)
                ok = truth.notna() & split.isin(TEST_SPLITS)
                qs = pd.DataFrame(q[:, step], index=origins, columns=levels)[ok]
                truth, split = truth[ok], split[ok]

                val = split == "val"
                best = max(levels, key=lambda lv: scores(truth[val] > 0, (qs[lv][val] >= 0.5))["f1"])
                warn_class = risk.predict(qs[best].round())
                true_class = risk.predict(truth)
                for name in TEST_SPLITS:
                    m = split == name
                    rows.append(
                        {
                            "model": size.split("/")[-1],
                            "setting": setting,
                            "horizon": horizon,
                            "split": name,
                            "quantile": best,
                            "pinball": pinball(truth[m].to_numpy(), qs[m].to_numpy(), levels),
                            **scores(true_class[m], warn_class[m]),
                        }
                    )
                OUT_DIR.mkdir(parents=True, exist_ok=True)
                qs.assign(truth=truth, split=split).to_parquet(
                    OUT_DIR / f"{size.split('/')[-1]}_{setting}_{horizon}.parquet"
                )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys

    pd.set_option("display.width", 200)
    sizes = [s for s in SIZES if any(arg in s for arg in sys.argv[1:])] or SIZES
    results = run(sizes)
    for model, part in results.groupby("model"):
        part.to_csv(OUT_DIR / f"scores_{model}.csv", index=False)
    cols = ["model", "setting", "horizon", "split", "quantile", "f1", "precision", "recall", "iou", "warned_days", "fire_days", "pinball"]
    print(results[cols].round(3).to_string(index=False))
