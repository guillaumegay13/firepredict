# firepredict

**Daily wildfire-start risk for each commune of the Var (83), France, built only from open data.**

Every morning, the model gives each of the Var's 153 communes a chance that a fire of 0.1 ha or more starts there the next day. It combines a Fire Weather Index (FWI) computed for each commune from Météo-France stations with three things the FWI leaves out: each commune's usual fire rate, the burning calendar set by the prefecture, and one month of dryness (rain minus evaporation).

**Live map, updated every morning:** https://guillaumegay13.github.io/firepredict/ (Var and Bouches-du-Rhône, with a department picker)

The frozen model and its model card are also on Hugging Face: [GuillaumeGay/firepredict](https://huggingface.co/GuillaumeGay/firepredict).

> **Status: research preview.** This is not an official warning tool. Fire danger warnings in France are issued by Météo-France and the prefectures. The results below are exploratory (see [Limits](#limits)); the clean test is the 2026 one.

## Results

Each year from 2006 to 2025 (without 2024) is hidden in turn: the model learns on the other years, then forecasts the hidden year. The unit is a **positive commune-day**: a day on which at least one fire of 0.1 ha or more started in the commune (902 over the 19 years, from 910 fires).

**Against the FWI alone**, the model is clearly better:

| Warning rule | FWI alone (Var) | Model | Gain (95% range) |
|---|---|---|---|
| The 15 highest-risk communes every day | 85 | **213** | **+151%** (+104% to +221%) |
| The 30 highest-risk communes every day | 190 | 344 | +81% (+58% to +107%) |
| The 10% highest-risk commune-days over all years | 230 | 354 | +54% (+33% to +74%) |
| Ranking of commune-days (ROC AUC) | 0.718 | **0.782** | better in 19 / 19 years |

The FWI computed here is an approximation from daily station data (daily maximum temperature, minimum humidity, mean wind, daily rain), not the official noon-based product.

**Against a stronger reference**, the FWI multiplied by each commune's usual fire rate, the gain is smaller:

- The model **ranks commune-days better**: AUC 0.782 against 0.757, better in 16 of 19 years. This holds without 2023 (0.787 against 0.761) and for fires of 1 ha or more (0.780 against 0.720).
- But it **does not catch more fires at practical warning levels**: 213 against 222 with 15 communes a day (−4%, −10% to +2%). On its very top warnings it does worse (−22% at 1% of commune-days).

**What each input adds** (ranking of commune-days):

| Inputs | ROC AUC |
|---|---|
| Commune history + local FWI | 0.767 |
| + burning calendar | 0.775 |
| + 30-day dryness (**the model**) | **0.782** |
| + buildings and roads in wildland (IGN BD TOPO) | 0.781 |

Buildings and roads near vegetation add nothing once each commune's fire history is known: that history already reflects where houses and roads meet vegetation.

**Other findings:**

- **The predicted chances are calibrated** in the bands used by the map: commune-days predicted at 0.1–0.3% had fire starts on 0.17% of them, those at 0.3–1% on 0.45%. Above 1% there are too few cases to check.
- **Accidental and work fires outside summer are what the FWI misses most.** In February–March, when brush burning is allowed, these fires are 2.2 times what the FWI band would suggest. Arson, by contrast, follows the FWI closely. These are associations, not proven causes.
- **Location within a day is hard to improve.** None of the fixed rankings we tested (fire history, a spatial Bayesian model, the landscape) puts more than about 25% of fire starts in the day's 15 highest-ranked communes.

The full analysis, with every table, is in [`reports/phase1_results.md`](reports/phase1_results.md).

### 2026 test and live archive

The inputs were chosen after looking at results across all years, so the scores above are somewhat optimistic. Two cleaner tests are running:

- **Retrospective 2026 test, blind to the outcomes.** On 9 November 2026, before any 2026 fire record was published, the model was frozen ([`models/small_commune_v2/`](models/small_commune_v2/)) and its forecasts for 1 January – 9 November 2026 saved (`forecasts_2026.csv`, SHA-256 `2be6091ab25539dc47bb751822d2a9b2b48f21218f8238bf0be1e1653dc8c6d6`). They were all computed on that day, from the weather available then. When BDIFF publishes the 2026 fires, they will be scored as they are.
- **Live archive.** From 9 November 2026, each day's forecast from the frozen model is saved in [`forecasts/`](forecasts/) and committed the same day, so the git history dates it.

## The model

A logistic regression with five inputs:

| Input | Effect | Source |
|---|---|---|
| FWI from the commune's nearest stations | + (strongest) | Météo-France |
| Commune's usual fire rate (other years) | + | BDIFF |
| Rain minus evaporation over the last 30 days | drier → + | Météo-France |
| February–March (brush burning allowed) | + | Var prefectoral orders on fire use |
| June–September (burning banned) | − | Var prefectoral orders on fire use |

The forecast for day D+1 is issued on the morning of day D and uses data up to D-1 only. No input uses recent fire records: BDIFF publishes a year's fires months later, so a live forecast cannot know them. Automated tests check both rules.

Larger models (CatBoost, the Timestral time-series model) were tested too. They do no better and are less consistent from year to year.

### Version 4: the current model (research candidate)

`small_commune_v4` ([`models/small_commune_v4/`](models/small_commune_v4/), frozen on 9 November 2026) is the model behind the map and the archive [`forecasts/v4/`](forecasts/v4/). It is one logistic regression for the Var and the Bouches-du-Rhône, chosen from a ladder of variants tested the same way (`python -m firepredict.model_v4 evaluate`):

- **each commune's usual rate**, as positive commune-days per usable day, smoothed toward its department's rate (empirical Bayes), so a commune with little history is not judged on a handful of fires;
- **local FWI, 30-day rain minus evaporation, the burning periods**, as before;
- **the date as a smooth yearly curve**, which can differ by department, instead of the v3 "usual rate × summer" term;
- **each commune's own seasonal pattern**, its share of fire days in June–September smoothed toward its department's share, so Aix-en-Provence keeps its summer peak and Draguignan its flatter pattern.

Compared with v3, it has a lower log loss in every test, and corrects v3's clear calibration failures: the Var's summer overestimated by 1.6× in the forward test, and the Bouches-du-Rhône winter overestimated by 3.6×. Variants adding recency weights or a sharper seasonal curve were not reliably better, so the simplest was kept.

> **Status.** Frozen as a research candidate for an outcome-blinded evaluation on 2026 labels. Historically it has a lower log loss than v2/v3, but it did not meet the original calibration acceptance criteria. An exploratory protocol, defined after inspecting results, found no clearly failing calibration groups; many groups remain inconclusive. Reliable subgroup calibration, superiority to the other variants, and operational usefulness have not been established.

The 2026 scoring rule is written in `models/small_commune_v4/metadata.json` (SHA-256 of `forecasts_2026.csv`: `1770b0b5e51aea65d39383043e3a7480d063574a1160d3eb52bfae00237b959e`). Versions 2 and 3 stay frozen, and their archives keep running, so all three can be compared on 2026.

### Version 3: one model for several departments (superseded by v4)

`small_commune_v3` ([`models/small_commune_v3/`](models/small_commune_v3/), frozen on 9 November 2026) is a single logistic regression trained on the Var and the Bouches-du-Rhône together (2,140 positive commune-days). It adds two inputs to the five above:

- **usual rate × June–September:** busy communes concentrate their fires in summer more than quiet ones. Without it, the model underestimated the Var's busiest communes in summer by half and overestimated Aix-en-Provence in winter.
- **one level per department**, so the chances are calibrated in each department. Rankings transfer between departments; probabilities do not.

Every-year test, each year hidden in both departments at once:

| | Var (83) | Bouches-du-Rhône (13) |
|---|---|---|
| ROC AUC, v3 | 0.779 (v2: 0.782) | 0.864 (single-department model: 0.865) |
| FWI × commune history | 0.757; v3 better in 17 / 19 years | 0.858; v3 better in 13 / 20 years |
| 15 communes a day: positive commune-days caught | 219 (reference 220) | 630 (reference 605) |

The archive [`forecasts/v3/`](forecasts/v3/) keeps running for comparison; the map now uses v4. The v2 archive for the Var (`forecasts/*.csv`) keeps running unchanged, so the original live test continues. Version 3 has its own 2026 test (SHA-256 of `forecasts_2026.csv`: `783debd1cd6848fb0cd667ef2abc418a2221379c9698366e1e6ead8eae0a856c`).

## Run it

Requirements: [uv](https://docs.astral.sh/uv/), Python 3.12, about 4 GB of free disk (the IGN BD TOPO for the Var is 455 MB compressed and 2.7 GB unpacked; only the three layers used, 1.4 GB, are kept).

```bash
git clone https://github.com/guillaumegay13/firepredict.git
cd firepredict
uv sync
uv run python -m firepredict.pipeline            # download, build, evaluate, make the map
```

The first run downloads all the data and takes about an hour. Then:

```bash
uv run python -m firepredict.pipeline forecast   # refresh the weather, archive today's forecast, rebuild the map
uv run python -m firepredict.pipeline --list     # every step
uv run python -m firepredict.pipeline all        # also the slow secondary analyses (hours)
uv run pytest                                    # tests
```

### Other departments

The department is chosen with `--department` (or the environment variable `FIREPREDICT_DEPARTMENT`); the default is the Var (83). Each department has its own folders (`data/raw/13/`, `data/processed/13/`, `results/13/`), and its settings in [`department.py`](src/firepredict/department.py): school-holiday zone, years with unusable fire records, FWI latitude, and the burning calendar of its prefectoral orders, as dated rules. The Bouches-du-Rhône (13) is set up:

```bash
uv run python -m firepredict.pipeline --department 13   # download, build, evaluate (no frozen model: no forecast stage)
uv run python -m firepredict.transfer                   # learn on one department, forecast the other; results/transfer/
```

To add a department, add its settings to `department.py` first. Only the three BD TOPO layers used are kept (about 1.5 GB per department).

Data downloaded before departments had their own folders sit directly in `data/raw/`; move them to `data/raw/83/` (and `data/processed/` to `data/processed/83/`) to reuse them.

The map is written to `results/83/risk_map.html`, a single page to open in a browser: one colour per commune, real fire starts as dots, and a pop-up with local weather for each commune. Past days (2023, 2025) show predictions made with that year hidden; 2026 shows the frozen model's forecasts. Each step can also be run on its own, for example `uv run python -m firepredict.value_check`.

## Data and credits

Raw data is downloaded by the code into `data/raw/<department>/` and is not redistributed in this repository. Every file is recorded with its URL, download date and SHA-256 in `data/raw/<department>/manifest.json`.

| Data | Provider | Licence |
|---|---|---|
| Fire records 2006–2025 (BDIFF) | Ministère de l'Agriculture, [bdiff.agriculture.gouv.fr](https://bdiff.agriculture.gouv.fr) | see below |
| Daily weather, all stations of the department | Météo-France, via [data.gouv.fr](https://www.data.gouv.fr/datasets/donnees-climatologiques-de-base-quotidiennes) | Licence Ouverte / Etalab 2.0 |
| Buildings, roads, vegetation (BD TOPO, September 2026) | IGN, [geoservices.ign.fr](https://geoservices.ign.fr/bdtopo) | Licence Ouverte / Etalab 2.0 |
| Burned-area outlines (Var only) | DDTM du Var, via data.gouv.fr | Licence Ouverte / Etalab 2.0 |
| Land cover (Corine Land Cover 2018) | Copernicus / IGN Géoplateforme | Copernicus open data |
| Commune boundaries and population | geo.api.gouv.fr (IGN, INSEE) | Licence Ouverte / Etalab 2.0 |
| School holidays | [vacances-scolaires](https://github.com/AntoineAugusti/vacances-scolaires), from Ministère de l'Éducation nationale open data | see the repository |
| Burning calendar | Var prefectoral orders of 16 May 2013 and 2 November 2025; Bouches-du-Rhône orders of 19 May 2004 and 20 December 2013 (sources in `department.py`) | public acts |

**BDIFF.** Source : Base de Données sur les Incendies de Forêts en France, [bdiff.agriculture.gouv.fr](https://bdiff.agriculture.gouv.fr). Droits de reproduction réservés et strictement limités. BDIFF data may be reused for personal, associative or professional purposes, unchanged, with this source cited, and may not be sold or rented. This repository does not contain BDIFF data; the code downloads it from the official site.

## Limits

- **2024 is excluded for the Var:** BDIFF holds only 3 Var fires for 2024, against 50–380 in other years. The Bouches-du-Rhône's 2024 looks normal and is kept.
- **2023 form change:** BDIFF started recording many more tiny fires in 2023, so only fires of 0.1 ha or more are counted.
- **Locations are communes:** BDIFF gives the commune of each fire, not its position.
- **One BD TOPO edition (2026) is used for every year**, so recent buildings are counted for earlier years too.
- **Burning calendar:** the 2013 and 2025 rules are applied to all years. The 2006–2012 rules and the prefect's extra red days are not included.
- **Exploratory results:** the inputs and the 30-day window were chosen after looking at these same years, so p-values and ranges are optimistic. The ranges are conditional on the chosen model.
- **Approximate FWI:** computed from daily station values, not the official noon-based FWI or Météo-France's forecasts.
- **Past features use later knowledge:** weather stations were chosen by their coverage over 2006–2026, and the leave-one-year-out test trains on years after the one it scores.
- **Two departments:** the Bouches-du-Rhône was added as a second department, with a transfer test between the two ([report](reports/bouches_du_rhone_and_transfer.md)). Other departments need their settings, mainly their burning orders, added to `department.py`.

## Repository layout

```
src/firepredict/    the code: one module per step (see pipeline.py)
tests/              leakage and scoring tests
models/             the frozen model and its 2026 test forecasts
forecasts/          the live archive: one file per forecast day
reports/            the full results report
data/, results/     created by the code, not tracked
```

## Licence and contact

Code under the [MIT licence](LICENSE). Data under the licences above.

Guillaume Gay, guillaume.gay@protonmail.com. Feedback from fire services, researchers and local authorities is very welcome.
