"""Daily values rebuilt from Météo-France's hourly file use its daily windows."""

import numpy as np
import pandas as pd

from firepredict import dataset
from firepredict.department import CODE


def hourly_file(folder, start: str, end: str) -> None:
    hours = pd.date_range(start, end, freq="h")
    h = pd.DataFrame({"NUM_POSTE": "83001001", "NOM_USUEL": "TEST", "AAAAMMJJHH": hours.strftime("%Y%m%d%H").astype(int),
                      "RR1": 1.0, "TX": 10.0, "TN": 10.0, "FF": 2.0, "FXI": 5.0, "DXI": 90, "UN": 80.0})  # fmt: skip
    for c in ["RR1", "TX", "TN", "FF", "FXI", "DXI", "UN"]:
        h["Q" + c] = 1
    at = pd.Series(hours.strftime("%Y-%m-%d %H")).to_numpy()
    # Last hour of 9 Nov's max-temperature window (06h on 10 Nov) counts for 9 Nov.
    h.loc[at == "2026-11-10 06", "TX"] = 30.0
    # 06h on 9 Nov closes 8 Nov's window: not part of 9 Nov.
    h.loc[at == "2026-11-09 06", "TX"] = 40.0
    # Min temperature: 18h on 9 Nov closes 9 Nov's window, 19h opens 10 Nov's.
    h.loc[at == "2026-11-09 18", "TN"] = -3.0
    h.loc[at == "2026-11-09 19", "TN"] = -9.0
    h.loc[at == "2026-11-09 15", ["FXI", "DXI"]] = [20.0, 270]
    # A doubtful value (quality code 2) is dropped.
    h.loc[at == "2026-11-09 06", "QRR1"] = 2
    folder.mkdir(parents=True, exist_ok=True)
    h.to_csv(folder / f"H_{CODE}_latest-2025-2026.csv.gz", sep=";", index=False)


def test_hourly_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset, "RAW_DIR", tmp_path)
    hourly_file(tmp_path / "meteofrance", "2026-11-08 01:00", "2026-11-10 06:00")
    daily = pd.DataFrame({"NUM_POSTE": "83001001", "NOM_USUEL": "TEST", "date": pd.date_range("2026-11-02", "2026-11-08"),
                          "RR": 0.0, "TN": 5.0, "TX": 15.0, "FFM": 3.0, "FXI": 9.0, "DXI": 0, "UN": 50.0,
                          "ETPGRILLE": np.arange(1.0, 8.0)})  # fmt: skip

    out = dataset.hourly_to_daily(daily, after=pd.Timestamp("2026-11-08")).set_index("date")

    # 8 Nov is in the daily files; 10 Nov has too few hours.
    assert list(out.index) == [pd.Timestamp("2026-11-09")]
    day = out.loc["2026-11-09"]
    assert day.RR == 24.0
    assert day.TX == 30.0
    assert day.TN == -3.0
    assert day.FXI == 20.0 and day.DXI == 270
    assert day.FFM == 2.0 and day.UN == 80.0
    # Evaporation is not hourly: the mean of the last 7 days of the daily files.
    assert day.ETPGRILLE == 4.0
    assert sorted(out.reset_index().columns) == sorted(daily.columns)
