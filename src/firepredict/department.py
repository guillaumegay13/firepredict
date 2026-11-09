"""The department under study, and what changes from one department to another.

Choose it with the environment variable FIREPREDICT_DEPARTMENT (default 83,
the Var), for example:
    FIREPREDICT_DEPARTMENT=13 uv run python -m firepredict.pipeline data
Every module reads it from here. Raw data, processed tables and results go
to data/raw/<code>/, data/processed/<code>/ and results/<code>/, so
departments never overwrite each other.

Burning calendar. Each prefecture sets by order ("arrêté préfectoral relatif
à l'emploi du feu") when owners may burn vegetation. We encode it as dated
rules, each in force from its `start` day until the next rule starts:
  - orange: periods where burning is restricted but possible
    (Var: allowed without wind; Bouches-du-Rhône 2004-2013: only with the
    mayor's authorisation)
  - red:    periods where burning is banned
Periods are (first day, last day) as "MM-DD", both included.

Sources (checked November 2026):
  Var (83): orders of 16 May 2013 and 2 November 2025: orange 1 Feb - 31 Mar,
    red 1 June - 30 Sept. Applied to all years: the 2006-2012 rules were not found.
  Bouches-du-Rhône (13):
    - order n°1272 of 19 May 2004 (abrogates n°1793 of 30 April 1992):
      "périodes sensibles" 1 Feb - 31 Mar and 1 June - 30 Sept; in and within
      200 m of woodland, fire use is banned in these periods except with the
      mayor's authorisation, asked 5 days ahead.
      https://www.aixenprovence.fr/IMG/pdf_emploidufeu.pdf
      Companion order n°2004140-14 (controlled burning): banned 1 June - 30 Sept.
      http://www.ofme.org/documents/Loisreglements/Arretes_prefectoraux/13_brulage_dirige_incineration.pdf
    - order n°2013354-0004 of 20 December 2013, modified 10 June 2014 and
      31 May 2016: burning green waste is banned in general; the allowed
      exceptions (brush-clearing duty, forestry, farming) are banned near
      woodland from 1 June to 30 Sept, with wind under 30 km/h otherwise.
      No February-March period any more.
      https://www.bouches-du-rhone.gouv.fr/content/download/21516/131519/file/160302_Memo_emploi_du_feu_Actu.pdf
      https://www.ventabren.fr/wp-content/uploads/2015/11/Enmploi-du-feu-10_2016.pdf
      https://www.ofme.org/documents/Loisreglements/Arretes_prefectoraux/13_20160531_AP_modificatif_emploi_du_feu_signe.pdf
    - Not confirmed: the Chambre d'agriculture PACA reports orders of
      14 March 2017 and 24 September 2025 (the latter repealing the former),
      still with a June-September ban. We did not find their text, so the
      2013 rule is kept from 2014 onward.
      https://paca.chambres-agriculture.fr/sinformer/agriculteur-lessentiel-a-savoir/arretes-brulage
  Neither department's extra red days (prefect's orders in drought, e.g. the
  Bouches-du-Rhône ban extended to 15 November 2017) are included.

School holidays: the zone column of the vacances-scolaires file. Nice and
Aix-Marseille are in zone B over 2006-2026 (Ministry calendar data from
2017; Bulletin officiel n°17 of 23 April 2015 for 2015-2017; zone lists
before the 2015 reform).
"""

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

CODE = os.environ.get("FIREPREDICT_DEPARTMENT", "83")


@dataclass(frozen=True)
class BurningRule:
    start: str  # first day the rule is in force, "YYYY-MM-DD"
    orange: tuple[tuple[str, str], ...]
    red: tuple[tuple[str, str], ...]
    source: str


@dataclass(frozen=True)
class Department:
    name: str
    latitude: float  # centroid of the commune boundaries, rounded to 0.1°; used by the FWI
    school_zone: str  # column of the vacances-scolaires file
    unreliable_years: tuple[int, ...]  # years whose fire labels are not usable
    burning: tuple[BurningRule, ...]


SUMMER = (("06-01", "09-30"),)
DEPARTMENTS = {
    "83": Department(
        name="Var",
        latitude=43.4,
        school_zone="vacances_zone_b",  # académie de Nice
        unreliable_years=(2024,),  # almost empty in BDIFF (3 fires)
        burning=(BurningRule("2006-01-01", (("02-01", "03-31"),), SUMMER, "orders of 16 May 2013 and 2 November 2025"),),
    ),
    "13": Department(
        name="Bouches-du-Rhône",
        latitude=43.5,
        school_zone="vacances_zone_b",  # académie d'Aix-Marseille
        unreliable_years=(),  # 2024 is low but plausible (22 fire days >= 0.1 ha; 25 in 2014)
        burning=(
            BurningRule("2006-01-01", (("02-01", "03-31"),), SUMMER, "order n°1272 of 19 May 2004"),
            BurningRule("2013-12-20", (), SUMMER, "order n°2013354-0004 of 20 December 2013, modified"),
        ),
    ),
}
if CODE not in DEPARTMENTS:
    raise SystemExit(f"FIREPREDICT_DEPARTMENT={CODE}: no settings yet; add it to DEPARTMENTS in department.py")
SETTINGS = DEPARTMENTS[CODE]


def burning_periods(dates: pd.DatetimeIndex, code: str = CODE) -> pd.DataFrame:
    """orange_period and red_period (0/1) of each date, from the rule in force that day."""
    rules = DEPARTMENTS[code].burning
    starts = pd.to_datetime([r.start for r in rules])
    which = np.searchsorted(starts, dates, side="right") - 1
    day = dates.strftime("%m-%d")
    out = {"orange_period": np.zeros(len(dates), int), "red_period": np.zeros(len(dates), int)}
    for i, rule in enumerate(rules):
        in_force = which == i
        for column, periods in [("orange_period", rule.orange), ("red_period", rule.red)]:
            for first, last in periods:
                out[column][in_force & (day >= first) & (day <= last)] = 1
    return pd.DataFrame(out, index=dates)
