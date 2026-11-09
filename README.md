# firepredict

**Daily wildfire-start risk for each commune of the Var (83), France, built only from open data.**

Every morning, the model gives each of the Var's 153 communes a chance that a fire of 0.1 ha or more starts there the next day. It combines a Fire Weather Index (FWI) computed for each commune from Météo-France stations with three things the FWI leaves out: each commune's usual fire rate, the burning calendar set by the prefecture, and one month of dryness (rain minus evaporation).

The frozen model and its model card are also on Hugging Face: [GuillaumeGay/firepredict-var-small-commune](https://huggingface.co/GuillaumeGay/firepredict-var-small-commune).

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

## Run it

Requirements: [uv](https://docs.astral.sh/uv/), Python 3.12, about 4 GB of disk (the IGN BD TOPO for the Var is 455 MB compressed, 2.7 GB unpacked).

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

The map is written to `results/risk_map.html`, a single page to open in a browser: one colour per commune, real fire starts as dots, and a pop-up with local weather for each commune. Past days (2023, 2025) show predictions made with that year hidden; 2026 shows the frozen model's forecasts. Each step can also be run on its own, for example `uv run python -m firepredict.value_check`.

## Data and credits

Raw data is downloaded by the code into `data/raw/` and is not redistributed in this repository. Every file is recorded with its URL, download date and SHA-256 in `data/raw/manifest.json`.

| Data | Provider | Licence |
|---|---|---|
| Fire records 2006–2025 (BDIFF) | Ministère de l'Agriculture, [bdiff.agriculture.gouv.fr](https://bdiff.agriculture.gouv.fr) | see below |
| Daily weather, all Var stations | Météo-France, via [data.gouv.fr](https://www.data.gouv.fr/datasets/donnees-climatologiques-de-base-quotidiennes) | Licence Ouverte / Etalab 2.0 |
| Buildings, roads, vegetation (BD TOPO, September 2026) | IGN, [geoservices.ign.fr](https://geoservices.ign.fr/bdtopo) | Licence Ouverte / Etalab 2.0 |
| Burned-area outlines | DDTM du Var, via data.gouv.fr | Licence Ouverte / Etalab 2.0 |
| Land cover (Corine Land Cover 2018) | Copernicus / IGN Géoplateforme | Copernicus open data |
| Commune boundaries and population | geo.api.gouv.fr (IGN, INSEE) | Licence Ouverte / Etalab 2.0 |
| School holidays | [vacances-scolaires](https://github.com/AntoineAugusti/vacances-scolaires), from Ministère de l'Éducation nationale open data | see the repository |
| Burning calendar | Var prefectoral orders of 16 May 2013 and 2 November 2025 | public acts |

**BDIFF.** Source : Base de Données sur les Incendies de Forêts en France, [bdiff.agriculture.gouv.fr](https://bdiff.agriculture.gouv.fr). Droits de reproduction réservés et strictement limités. BDIFF data may be reused for personal, associative or professional purposes, unchanged, with this source cited, and may not be sold or rented. This repository does not contain BDIFF data; the code downloads it from the official site.

## Limits

- **2024 is excluded:** BDIFF holds only 3 Var fires for 2024, against 50–380 in other years.
- **2023 form change:** BDIFF started recording many more tiny fires in 2023, so only fires of 0.1 ha or more are counted.
- **Locations are communes:** BDIFF gives the commune of each fire, not its position.
- **One BD TOPO edition (2026) is used for every year**, so recent buildings are counted for earlier years too.
- **Burning calendar:** the 2013 and 2025 rules are applied to all years. The 2006–2012 rules and the prefect's extra red days are not included.
- **Exploratory results:** the inputs and the 30-day window were chosen after looking at these same years, so p-values and ranges are optimistic. The ranges are conditional on the chosen model.
- **Approximate FWI:** computed from daily station values, not the official noon-based FWI or Météo-France's forecasts.
- **Past features use later knowledge:** weather stations were chosen by their coverage over 2006–2026, and the leave-one-year-out test trains on years after the one it scores.
- **One department:** the results have not yet been checked on other departments.

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
