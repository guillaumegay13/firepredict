"""Archive today's forecast from the frozen model: the live test.

Each run writes forecasts/<target day>.csv with every commune's chance of a
fire start on the target day, as computed now, from the weather available
now. Committed and pushed the same day, the archive is a dated public record
that cannot be edited afterwards; once BDIFF publishes the year, it is scored
as it stands. An existing file is never overwritten.

Run after refreshing the weather (pipeline stage "forecast" does both).

Usage:
    uv run python -m firepredict.forecast
"""

from datetime import datetime, timezone

import pandas as pd

from firepredict.dataset import PROCESSED_DIR, ROOT
from firepredict.freeze import VERSION, predict
from firepredict.model_communes import HORIZON, build_panel

ARCHIVE = ROOT / "forecasts"


def latest_forecast() -> pd.DataFrame:
    panel = build_panel(forecast=True)
    issue = panel.index.get_level_values("date").max()
    rows = panel.xs(issue, level="date", drop_level=False)
    chance = predict(rows)
    names = pd.read_parquet(PROCESSED_DIR / "communes.parquet").name
    insee = chance.index.get_level_values("insee")
    return pd.DataFrame({
        "target_day": (issue + pd.Timedelta(days=HORIZON)).strftime("%Y-%m-%d"),
        "issued_on": issue.strftime("%Y-%m-%d"),
        "weather_up_to": (issue - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": VERSION,
        "insee": insee,
        "commune": names.reindex(insee).to_numpy(),
        "chance": chance.round(8).to_numpy(),
    }).sort_values("chance", ascending=False)  # fmt: skip


if __name__ == "__main__":
    table = latest_forecast()
    path = ARCHIVE / f"{table.target_day.iloc[0]}.csv"
    if path.exists():
        print(f"{path.relative_to(ROOT)} already archived, left unchanged")
    else:
        ARCHIVE.mkdir(exist_ok=True)
        table.to_csv(path, index=False)
        top = ", ".join(f"{c} {p:.2%}" for c, p in zip(table.commune[:5], table.chance[:5]))
        print(f"wrote {path.relative_to(ROOT)}: expected starts {table.chance.sum():.2f}; highest: {top}")
