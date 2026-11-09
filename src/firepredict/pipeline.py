"""Run the project end to end, or one part of it, with one command.

Stages, in order:
  data       download raw data, check it, build the daily and commune tables
  evaluate   the main results: FWI baseline, small models, every-year tests
  extra      slower or secondary analyses (CatBoost, Timestral, spatial model)
  forecast   refresh this year's weather, archive today's forecast, rebuild the map

Each step is a module of this package, run as `python -m firepredict.<module>`.
A failing step stops the run.

The department is the Var (83) unless --department or the environment
variable FIREPREDICT_DEPARTMENT says otherwise (see department.py). The
"forecast" stage uses the frozen model, which exists for the Var only.

Usage:
    uv run python -m firepredict.pipeline                 # data + evaluate + forecast
    uv run python -m firepredict.pipeline forecast        # today's map only
    uv run python -m firepredict.pipeline all             # everything, including "extra" (hours)
    uv run python -m firepredict.pipeline --list          # show the steps
    uv run python -m firepredict.pipeline --department 13 # Bouches-du-Rhône: data + evaluate
"""

import argparse
import os
import subprocess
import sys
import time

STAGES = {
    "data": [
        ["download"],
        ["audit"],
        ["dataset"],
        ["features"],
        ["communes"],
        ["landscape"],
    ],
    "evaluate": [
        ["baseline_fwi"],
        ["drought_check"],
        ["human_check"],
        ["compare_years"],
        ["model_small"],
        ["model_spread"],
        ["compare_communes"],
        ["value_check"],
        ["calibration"],
        ["robustness"],
    ],
    "extra": [
        ["model_catboost"],
        ["model_catboost", "oracle"],
        ["model_catboost_classes"],
        ["model_communes"],
        ["model_spatial"],
        ["compare_communes", "catboost"],
        ["model_timestral", "28M"],  # needs: uv sync --extra timestral
    ],
    "forecast": [
        ["download", "meteo-latest"],
        ["dataset"],
        ["features"],
        ["forecast"],
        ["risk_map"],
    ],
}
DEFAULT = ["data", "evaluate", "forecast"]


def run_step(step: list[str]) -> None:
    command = [sys.executable, "-m", f"firepredict.{step[0]}", *step[1:]]
    print(f"\n=== {' '.join(step)}", flush=True)
    start = time.time()
    subprocess.run(command, check=True)
    print(f"=== {' '.join(step)}: done in {time.time() - start:.0f} s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stages", nargs="*", choices=[*STAGES, "all"], help=f"default: {' '.join(DEFAULT)}")
    parser.add_argument("--list", action="store_true", help="print the steps and exit")
    parser.add_argument("--department", help="department code, e.g. 13 (default: $FIREPREDICT_DEPARTMENT or 83)")
    args = parser.parse_args()
    if args.department:
        os.environ["FIREPREDICT_DEPARTMENT"] = args.department  # read by every step
    names = list(STAGES) if "all" in args.stages else (args.stages or DEFAULT)
    if os.environ.get("FIREPREDICT_DEPARTMENT", "83") != "83" and "forecast" not in args.stages:
        names = [n for n in names if n != "forecast"]  # no frozen model outside the Var
    for name in names:
        for step in STAGES[name]:
            if args.list:
                print(f"{name:9} {' '.join(step)}")
            else:
                run_step(step)


if __name__ == "__main__":
    main()
