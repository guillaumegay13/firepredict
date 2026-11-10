# Wildfire-start risk by commune, Var (83) and Bouches-du-Rhône (13): results

*27 October 2026, updated 10 November 2026. Data: BDIFF fires 2006–2025, Météo-France daily stations, DDTM 83 perimeters, Corine Land Cover 2018, IGN BD TOPO (September 2026), prefectoral orders on fire use (Var, Bouches-du-Rhône).*

The work started with the Var alone (sections 1–8). The current model, version 4, covers the Var and the Bouches-du-Rhône and is described first. Older sections are kept as the record of how we got there; where their numbers are superseded, they say so.

## Summary

- **Current model: version 4** (`models/small_commune_v4`, frozen 9 November 2026), one logistic regression for the Var and the Bouches-du-Rhône. Warning the 15 highest-risk communes each day, it catches **237 positive commune-days against 85 for the FWI alone in the Var, and 632 against 169 in the Bouches-du-Rhône** (every year hidden in turn). Against the FWI multiplied by each commune's fire history, the gain is modest: +7% and +5% at 15 communes a day. It is a research candidate: it did not meet the calibration rules written before the results, and the clean test is 2026. See "Current model" below.
- **For "which days are dangerous", the official Fire Weather Index (FWI) is the best tool we found.** CatBoost and Mistral's Timestral do not beat it. For a yes/no "fire tomorrow?" forecast, a calendar rule ("warn every day from June to September") scores about as well as any of them.
- **Weather predicts big fires much better than fire starts.** Most fires are started by people. The weather decides whether a fire can grow, so FWI ranks days with a fire of 100 ha or more almost perfectly (0.94), but days with any fire much less well (0.79).
- **On dangerous days, more fires start; each fire is only a little more likely to grow.** Once a fire has started, the weather only weakly predicts whether it will grow (0.59–0.67). This is partly because firefighters attack fires early on dangerous days.
- **Where fires start follows people, not forest.** Fires start where houses meet vegetation. A commune model puts 26–34% of fire starts into each day's top 15 communes (random: 10%), but simple rankings by size, history or vegetation near houses already get 20–33%.
- **The fire records have two problems.** 2024 is almost empty for the Var, and in 2023 BDIFF started recording many more tiny fires. Both affect the benchmark's test year (2023).
- **The FWI misses accidental and work fires outside summer, and the burning rules explain part of them.** In February–March, when owners may burn brush, these fires are 2.2 times what the FWI predicts. Arson, by contrast, matches the FWI exactly. A small model (FWI + burning period + 30-day rain minus evaporation) ranks fire days better than the FWI in 13 of 19 years (+0.015, p = 0.03). On the two test years it ties the FWI's F1 with half the warnings.
- **At commune level, a small model (local FWI, commune fire history, burning calendar, 30-day dryness) is far better than the FWI alone**: warning the 15 highest-risk communes each day, it catches 213 positive commune-days against 85 (19 years, leave-one-year-out). Against the FWI multiplied by each commune's fire history, it ranks commune-days better (AUC 0.782 against 0.757, 16 of 19 years) but catches about as many at practical warning levels. See the update below: these are the corrected results for version 2, now superseded by version 4.
- **Location within a day is hard to improve.** None of the fixed rankings tested puts more than about 25% of fire starts in the day's top 15 communes; buildings and roads near vegetation (IGN BD TOPO) add nothing once fire history is known.

## Current model: version 4 (9 November 2026)

**What changed since version 2.** Version 2 was trained on the Var alone. Adding the Bouches-du-Rhône (1,238 positive commune-days over 2006–2025, 2024 usable there; see [`bouches_du_rhone_and_transfer.md`](bouches_du_rhone_and_transfer.md)) showed that the ranking transfers between departments but the probability levels do not. Version 3 (`small_commune_v3`) trained one model on both departments, with a department level and a "usual rate × June–September" term. It overestimated some communes badly: Draguignan in July 2026 at 1.52% a day, against a historical July rate of 0.85%. Version 4 replaces that term with inputs learned from the data rather than set by hand.

**Inputs of version 4** (standardised weights):

| Input | Weight | What it carries |
|---|---|---|
| Commune's usual rate, smoothed toward its department's rate (empirical Bayes, gamma–Poisson) | +0.86 | Where fires usually start; a commune with few fires is not judged on a handful of them |
| Local FWI (nearest stations) | +0.65 | Today's fire weather |
| Rain minus evaporation, last 30 days | −0.51 | Dryness |
| Commune's share of fire days in June–September, smoothed toward its department's share (beta–binomial) | +0.25 | Each commune's own season: Aix-en-Provence peaks in summer, Draguignan is flatter |
| Orange period (brush burning allowed with conditions) | +0.22 | Human ignitions outside summer |
| Red period (burning banned) | −0.10 | |
| Day of year (sine, cosine), and the same by department; department indicator | small | Seasonal curve, calibrated per department |

Usual rates, summer shares and smoothing strengths are computed without the hidden year and without the row's own year. Draguignan in July 2026 is now 0.97%.

**How it was chosen.** Nine variants (a ladder from version 2 pooled, through version 3, to version 4 and three extensions) were compared on the same folds, in two tests: every year hidden in turn (leave-one-year-out), and a forward test learning on earlier years only. Acceptance rules were written before the results:

| Variant | Var: AUC / top 15 caught (every year) | Bouches-du-Rhône: AUC / top 15 caught | Lower log loss than v2 and v3, every test |
|---|---|---|---|
| v2 pooled | 0.776 / 219 | 0.866 / 633 | — |
| v3 (rate × summer) | 0.779 / 219 | 0.864 / 630 | — |
| **v4 (+ commune season)** | **0.784 / 237** | **0.869 / 631** | **yes** |
| v4 + recency weights | 0.783 / 238 | 0.870 / 631 | yes |
| v4 + second curve | 0.786 / 238 | 0.870 / 629 | yes |
| v4 + season levels | 0.783 / 238 | 0.869 / 629 | yes |

The three extensions were not reliably better than version 4 in paired bootstrap comparisons of log loss, so the simplest was kept.

**Calibration** (expected ÷ observed fire days, 1 = right):

| Group | v3 | v4 |
|---|---|---|
| Var, June–September, forward test | 1.63 | 1.37 |
| Var, June–September, every year | 1.16 | 0.98 |
| Bouches-du-Rhône, November–January, every year | 3.6 | 2.0 |
| Bouches-du-Rhône, November–January, forward test | 3.1 | 1.9 |

Version 4 removes version 3's clear failures but does not meet the original rule (0.8–1.25 in every season with 30 events or more): no variant did. Under an exploratory protocol defined after the results, with 95% ranges from resampling years, no group clearly fails for version 4; many are inconclusive, and Bouches-du-Rhône winters have too few fires to judge (22 over 20 years).

**Against the FWI references** (every year hidden in turn; 95% ranges from resampling years):

| Warning rule | Var: FWI alone → v4 | Var: history × FWI → v4 | Bouches-du-Rhône: FWI alone → v4 | Bouches-du-Rhône: history × FWI → v4 |
|---|---|---|---|---|
| 15 communes a day | 85 → 237, +179% | 222 → 237, +7% (−3% to +18%) | 169 → 632, +274% | 602 → 632, +5% (+2% to +9%) |
| 30 communes a day | 190 → 377, +98% | 328 → 377, +15% (+6% to +25%) | 323 → 834, +158% | 818 → 834, +2% (−1% to +5%) |
| 10% of commune-days | 230 → 355, +54% | 307 → 355, +16% (+3% to +30%) | 454 → 755, +66% | 737 → 755, +2% (−1% to +6%) |
| 5 communes a day | 30 → 102 | 96 → 102, +6% (−5% to +26%) | 66 → 365 | 382 → 365, −4% (−9% to 0%) |
| ROC AUC | 0.718 → 0.784, 19 / 19 years | 0.757 → 0.784, 16 / 19 | 0.781 → 0.869, 20 / 20 | 0.858 → 0.869, 15 / 20 |

**What can be claimed.** Version 4 is far better than the FWI alone at choosing communes, in both departments and every year. Against the FWI combined with fire history, it ranks commune-days better in most years, with a modest gain at practical warning levels and none on the very top warnings. Its probabilities are closer to what happened than version 3's, but reliable calibration by season is not established.

**Status and the 2026 test.** Frozen on 9 November 2026 as a research candidate. Its forecasts for 1 January – 9 November 2026 are saved in `models/small_commune_v4/forecasts_2026.csv` (85,136 rows, SHA-256 `1770b0b5…`), computed before any 2026 fire record was published. The scoring rule is written in its `metadata.json`. Versions 2 and 3 stay frozen, and the daily archives of all three keep running (`forecasts/`, `forecasts/v3/`, `forecasts/v4/`), so they can be compared on 2026. The map uses version 4.

## Version 2: update after an independent review (9 November 2026)

An independent review of the code and claims (OpenAI Codex, GPT-6.1, read-only) confirmed the arithmetic and the time alignment, and found one critical problem and several overstatements. Changes:

1. **Inputs built from recent fire records were removed** (the commune's fires over the last 30 days, and the Var's fires over the last 1–30 days). BDIFF publishes a year's fires months later, so a live forecast cannot know them; the earlier backtests used information the live model would not have.
2. **Two small leaks were closed:** at department level, missing values are now filled inside each fold; at commune level, no input depends on labels any more, so no hidden-year fire can reach training rows through a rolling window. An automated test now checks that no input changes when fire records change.
3. **Buildings and roads in wildland were dropped from the model:** with recent-fire inputs gone, they add nothing once commune history is known (AUC 0.781 with them, 0.782 without).
4. **Wording corrected:** the FWI here is an approximation from daily station data, not the official product; counts are positive commune-days (902, from 910 fires), not fires; the 2026 test is a retrospective test blind to the outcomes, not a live trial; the ~25% figure is what the fixed rankings tested reach, not a proven ceiling; human-cause findings are associations, not causes; all every-year results are exploratory, because the inputs were chosen on the same years.

**Corrected commune results** (small model: local FWI, usual rate, burning calendar, 30-day dryness; 19 years, leave-one-year-out):

| Reference | AUC (reference → model) | Years better | 15 communes/day: caught | 10% of commune-days: caught |
|---|---|---|---|---|
| FWI alone (Var) | 0.718 → 0.782 | 19 / 19 | 85 → 213, **+151%** (+104% to +221%) | 230 → 354, +54% (+33% to +74%) |
| Local FWI alone | 0.728 → 0.782 | 18 / 19 | 123 → 213, +73% (+46% to +111%) | 256 → 354, +38% (+20% to +58%) |
| FWI × commune history | 0.757 → 0.782 | 16 / 19 (p < 0.001) | 222 → 213, −4% (−10% to +2%) | 307 → 354, +15% (+2% to +30%) |

| Inputs (nested) | AUC |
|---|---|
| History + local FWI | 0.767 |
| + burning calendar | 0.775 |
| + 30-day dryness (the model) | 0.782 |
| + buildings and roads in wildland | 0.781 |

- **Robustness:** without 2023, AUC 0.787 against 0.761 (16 of 18 years); for fires of 1 ha or more, 0.780 against 0.720 (16 of 19).
- **Calibration** holds in the map's bands (0.1–0.3%: predicted 0.167%, observed 0.169%; 0.3–1%: 0.38% against 0.45%); above 1% there are only 17 commune-days.
- **On its very top warnings the model does worse than FWI × history** (59 against 76 at the 1% most alarming commune-days), and its pooled PR AUC is lower (0.0030 against 0.0034). Recent-fire inputs were what sharpened the top of the old ranking.
- **Department level** is unchanged for the small day model; CatBoost drops (within-year AUC 0.746) without the recent-fire inputs.
- **Frozen model:** `models/small_commune_v2` (five inputs, the Var only), frozen on 9 November 2026, with its 2026 forecasts (SHA-256 `2be6091a…`). Daily forecasts are archived in `forecasts/` from 9 November 2026. Superseded on the map by version 4 (above).

Sections 7 and 8 below describe the first version; their commune numbers are superseded by this update.

## 1. What we set out to do

Predict, every morning, the risk of wildfire in the Var for the next day (J+1), and compare four approaches on the same data: the FWI (the reference), CatBoost, Timestral (Mistral AI, zero-shot) and an ensemble. The design follows the French benchmark by Caron et al. (2025, arXiv 2506.04254): daily department-level predictions, a 0–4 risk scale, train 2006–2020 + 2022, validation 2021, test 2023, with 2025 as an extra test.

**Forecast timing.** A forecast is issued on the morning of day D, once Météo-France's data for D-1 is complete. *Operational* means the model sees data up to D-1 only. *Oracle* adds the actual weather of the target day, standing in for a perfect weather forecast. Automated tests check that no operational feature uses data from day D or later.

## 2. Data: what we found

| Issue | Evidence | What we did |
|---|---|---|
| **2024 almost empty** | 3 Var fires in BDIFF, against 50–380 in other years; neighbouring departments look normal | 2024 labels removed; validation uses 2021 only |
| **2023 small-fire artifact** | New BDIFF form; every 2023 record flagged "partial or invalid"; fires under 100 m² jump from 10–31 a year to 216 | Main target counts fires ≥ 0.1 ha only |
| **Pre-1992 records** | All fires recorded are ≥ 0.1 ha, and there were 4–5 times more fires in the 1980s | Kept 2006+ for models; the 0.1 ha cut-off stays consistent with 1973+ |
| Large fires reliable | All 47 DDTM perimeters ≥ 10 ha (2006–2022) are found in BDIFF | — |
| Weather clean | 14–30 stations with more than 95% coverage, 2006–2026 | Fixed set of "core" stations |
| Quality codes misread at first | Météo-France code 9 means "filtered" (passed first checks), not suspect; the doubtful code is 2 | Fixed: code 2 dropped, code 9 kept. FWI moved by 0.006 on average; it restored evaporation (ETP, 30 stations) |

Note: Prométhée's public data link leads to BDIFF; a Prométhée extract (2×2 km cells, consistent history) needs a formal request.

## 3. Results: "fire tomorrow?" (department level)

F1 (0 = useless, 1 = perfect) for "at least one fire ≥ 0.1 ha tomorrow", operational J+1. CatBoost is the mean of 5 runs.

| | 2023 (test) | 2025 (2nd test) |
|---|---|---|
| *Naive: warn every day* | *0.25* | *0.18* |
| *Naive: warn every day from June to September* | *0.25* | *0.39* |
| FWI ≥ 24.6 | 0.26 | **0.42** |
| CatBoost | 0.25 ± 0.01 | 0.38 ± 0.01 |
| Timestral-28M (zero-shot) | 0.27 | 0.25 |

Ranking quality (ROC AUC, 0.5 = random): FWI 0.65 / 0.87, CatBoost 0.64 / 0.81, Timestral 0.56 / 0.81 (2023 / 2025).

- **A perfect weather forecast adds almost nothing.** Oracle CatBoost 0.26 / 0.39, oracle FWI 0.27 / 0.41.
- **Training on recent years only (2017–2022) makes CatBoost worse:** 0.20 / 0.28. More data matters more than recent data.
- **Timestral** reaches its 2023 score by warning on 274 days out of 365. It ranks days worse than FWI, and the oracle weather makes it worse. By the brief's rule, no ensemble was built.
- **The 0–4 scale.** CatBoost has a higher IoU than FWI (0.15 against 0.05 on 2023), but it never predicts classes 2–4. In effect it is a yes/no model that warns often. Days with 2 or more fires are too rare to learn severity from: 7 in the 2023 test.

**Why the models stall.** Month by month, both FWI and CatBoost catch most summer fire days, but warn on almost every summer day. Outside summer, fires start when FWI is low: 57% of 2023 fire days fell between October and May, mostly garden burning and fires during works.

## 4. Fire size: weather decides growth, people decide starts

Ranking of days by the fire day's own weather, 2006–2025:

| | Any fire ≥ 0.1 ha | ≥ 1 ha | ≥ 10 ha | ≥ 100 ha |
|---|---|---|---|---|
| Calendar (month's usual rate) | 0.69 | 0.69 | 0.73 | 0.83 |
| **FWI** | 0.79 | 0.82 | 0.86 | **0.94** |
| Mean wind | 0.58 | 0.63 | 0.71 | 0.79 |
| Fire days | 705 | 249 | 64 | 18 |

**Fire level: "if a fire starts, will it grow?"** 920 fires; each year predicted from the other years.

| | ≥ 1 ha | ≥ 10 ha |
|---|---|---|
| FWI | 0.59 | 0.67 |
| CatBoost | 0.52 | 0.61 |

Big-fire *days* are predictable mainly because **more fires start** on dangerous days, not because each fire is much more likely to grow. The records reflect outcomes after firefighting: on high-danger days the Var pre-positions crews and aircraft. The weather used is also the department average, which blurs local wind.

## 5. Where fires start (commune level)

153 communes, 1.05 million commune-days, 902 fire starts. J+1, operational. Each day, communes are ranked by risk; the table shows the share of fire starts that fall in that day's top 15.

| Ranking | 2023 | 2025 |
|---|---|---|
| Random | 10% | 10% |
| Commune area | 18% | 23% |
| Population | 11% | 26% |
| Vegetation within 200 m of houses | 11% | 33% |
| Commune fire history | 20% | 28% |
| **CatBoost by commune** | **26%** | **34%** |

Across communes, fire counts follow commune size (correlation +0.44), vegetation near houses (+0.42) and population (+0.35). They go *down* with forest share (−0.23). The model mostly rediscovers the obvious communes; its gain beyond simple rankings is small and within noise for 61 / 39 test starts.

## 6. Beyond the FWI: people, burning rules and dryness

**Causes.** BDIFF gives a cause for 97% of fires ≥ 0.1 ha. How well the FWI ranks the days of each cause:

| Cause | Fires | FWI AUC | In June–Sept |
|---|---|---|---|
| Accident, individuals (burning, barbecues…) | 375 (41%) | 0.74 | 40% |
| Arson | 240 (26%) | **0.86** | 74% |
| Works (farming, forestry, building) | 142 (15%) | 0.73 | 32% |
| Accident, infrastructure | 92 (10%) | 0.82 | 68% |
| Lightning | 40 (4%) | 0.71 | 78% |

Arsonists pick dangerous days, so the FWI already covers them. Accidental and work fires are the ones it misses.

**People and burning rules.** Fire starts compared with what the FWI alone predicts (1.00 = as predicted). The periods come from the Var's fire-use orders (16 May 2013, 2 November 2025): owners may burn brush in the green period and, without wind, in the orange period (1 Feb – 31 Mar); burning is banned in the red period (1 June – 30 Sept, and any day with wind above 40 km/h).

| | Accidents + works | Arson |
|---|---|---|
| Green period | 1.18 | 0.94 |
| Orange period (Feb–Mar) | **2.21** | 1.04 |
| Red period (summer) | 0.67 | 1.01 |
| Windy days outside summer | 1.43 | 1.02 |

At the same FWI, school holidays outside summer have about 20% more fires, July–August about 30% more than June and September, and weekends fewer than weekdays. Most accidental fires are reported between 13h and 18h, although burning is only allowed from 5h to 13h. The data cannot tell whether morning burns reignite later or people burn outside the allowed hours. The 2006–2012 rules and the prefect's extra red days are not included.

**Dryness.** Rain minus potential evaporation (ETP) over the last 30 days ranks summer fire days better than the FWI's own Drought Code (0.69 against 0.63; 0.77 against 0.67 for fires ≥ 100 ha). One month matters more than two or three.

**Does it add to the FWI?** Each of 19 years (2006–2025 without 2024) hidden in turn; ranking of days within the hidden year:

| Model | Within-year AUC | Years better than FWI |
|---|---|---|
| FWI alone | 0.734 | — |
| FWI + 30-day dryness | 0.737 | 12 |
| FWI + burning period | 0.739 | 9 |
| **FWI + burning period + dryness** | **0.749** | **13** (p = 0.03) |
| CatBoost, with or without these inputs | 0.736–0.739 | about 10 |

Pooled over all years CatBoost scores 0.760, but only by guessing which years are busier; within a year it ranks days like the FWI. With about 700 fire days, a small model with five inputs does better than CatBoost with 45.

**On the benchmark's test years** the small model ties the FWI: F1 0.25 / 0.42 against 0.27 / 0.42 (2023 / 2025). It warns on half as many days (43 against 93 in 2025), with more of its warnings right (40% against 29%) but fewer fire days caught (46% against 73%).

## 7. The commune model, judged on every year (first version, superseded)

*The numbers in this section come from the first version, which used recent fire records and BD TOPO inputs. See the update at the top for corrected results.*

**New inputs.** From IGN BD TOPO (metre precision, September 2026 edition): buildings inside or within 50 m of wildland (woods, forest, garrigue/maquis), and paved roads and tracks crossing wildland. "Buildings within 50 m of wildland" is the input most linked to fire counts by commune (correlation 0.54, against 0.44 for commune area). Each commune also gets its own FWI and 30-day rain minus evaporation from its nearest stations; in August 2025 the local FWI ranged from 24 to 50 across communes.

**Protocol.** Each of 19 years (2006–2025 without 2024) is hidden in turn. A commune's usual fire rate is recomputed for every hidden year without it, and for each row without the row's own year.

**Small commune model** (logistic regression, 8 inputs): commune history, local FWI, local 30-day water balance, orange and red burning periods, buildings near wildland, roads in wildland, the commune's fires over the last 30 days.

| Model | Ranking of commune-days (AUC) | Years better than history × FWI | Top-15 catch |
|---|---|---|---|
| Commune history only | 0.654 | 0 / 19 | 24.4% |
| **History × FWI (reference)** | 0.757 | — | 24.4% |
| History × local FWI | 0.763 | 12 / 19 (p = 0.07) | 23.7% |
| Spatial Bayesian rate × FWI | 0.760 | 12 / 19 (p = 0.40) | 24.3% |
| **Small model** | **0.778** | **18 / 19 (p < 0.001)** | 25.1% |
| CatBoost (all inputs, 5 seeds) | 0.778 ± 0.002 | 13–14 / 19 (p = 0.006–0.03) | 26.2% (25.5–27.2) |

**Same number of warnings** (all 19 years; a fire start happens on 0.086% of commune-days):

| Warned commune-days | History × FWI: right / fires caught | Small model: right / fires caught |
|---|---|---|
| 0.5% | 0.89% / 47 | **1.10% / 58** |
| 1% | 0.72% / 76 | **0.83% / 87** |
| 2% | 0.60% / 127 | 0.61% / 128 |
| 5% | 0.40% / 209 | **0.44% / 233** |
| 10% | 0.29% / 307 | **0.34% / 353** |

**Against the FWI as fire services use it** (one value per day, no commune history), with 95% ranges from resampling the 19 years:

| Reference | AUC gain | 5% budget: fires caught | 10% budget: fires caught |
|---|---|---|---|
| FWI alone (Var) | +0.060 (0.045 to 0.074) | 142 → 233, **+64%** (+43% to +108%) | 230 → 353, **+53%** (+31% to +74%) |
| Local FWI alone (nearest stations) | +0.050 (0.035 to 0.066) | 125 → 233, +86% (+47% to +142%) | 256 → 353, +38% (+20% to +58%) |
| History × FWI | +0.021 (0.012 to 0.030) | 209 → 233, +11% (+2% to +27%) | 307 → 353, +15% (+4% to +29%) |

At a 1% budget the gain over history × FWI (+14%) is within noise (−12% to +46%).

**Are the probabilities right?** Commune-days grouped by the model's predicted chance (the map's colour bands), all 19 hidden years:

| Predicted chance | Commune-days | Fire starts | Mean predicted | Observed |
|---|---|---|---|---|
| < 0.03% | 353,474 | 39 | 0.014% | 0.011% |
| 0.03–0.1% | 386,741 | 231 | 0.060% | 0.060% |
| 0.1–0.3% | 280,158 | 456 | 0.162% | 0.163% |
| 0.3–1% | 30,890 | 156 | 0.43% | 0.51% |
| ≥ 1% | 1,377 | 20 | 1.59% | 1.45% |

The predicted chances match what happened in every band (chart: `results/calibration.png`). Against climatology (each commune's usual daily rate), log loss improves by 7% (0.00642 against 0.00688). The Brier skill score is small (+0.1%), as always for events this rare.

**Precision–recall.** PR AUC is better pooled over all years (0.0042 against 0.0034 for history × FWI; 0.0020 for FWI alone; random 0.0009). Year by year it is a tie: the reference is higher in 10 of 19 years, and two years with a few lucky top-ranked commune-days (2010, 2014) dominate a mean of yearly values.

**Confusion table** at the 10% warning budget (about 15 communes a day on average), all 19 years:

| | Fire start | No fire start |
|---|---|---|
| **Warned** | 353 (caught) | 104,911 (false warnings) |
| **Not warned** | 549 (missed) | 946,827 |

Precision 0.34%, recall 39% (FWI alone: 0.22% and 26%). With fire starts on 0.086% of commune-days, precision stays low for any model; the useful comparison is fires caught for a given number of warnings.

**Location within a day.** A hierarchical spatial Poisson model (BYM: landscape, neighbouring communes and own history, fitted with PyMC) gives smoothed rates that agree with raw history (correlation 0.96) and does not improve the daily ranking. Even with each commune's true rate, the 15 highest-rate communes would hold only 25% of fire starts: fires are spread over many communes with low individual risk. A fixed ranking cannot do better, and all models reach it. CatBoost goes about one point above it using information that changes daily (recent fires, local weather), short of significance (p = 0.06–0.16).

**What can be claimed.** Across 19 years, the small model ranks commune-days better than the FWI combined with each commune's fire history (18/19 years, p < 0.001), and is right more often for the same number of warnings. It does not improve the daily choice of communes beyond their usual risk. The risk map used this model until version 4.

## 8. Method and limits

- **No peeking at the future.** Forecasts for day D+1 use data up to D-1. Automated tests check this at department level; at commune level, a commune's usual rate never includes the hidden year or the row's own year.
- **Selection bias.** The burning period, the 30-day window and the small model's inputs were chosen after looking at results across all years. The every-year scores are therefore somewhat optimistic. The clean test is 2026 (below).
- **Frozen models and retrospective 2026 test.** On 9 November 2026 versions 2, 3 and 4 were frozen (trained on every labelled year) and their forecasts for 1 January – 9 November 2026 saved with SHA-256 fingerprints. They were all computed on that day, so this is a retrospective test blind to the outcomes, not a live trial. Live forecasts are archived daily in `forecasts/` (v2), `forecasts/v3/` and `forecasts/v4/`.
- **Labels.** The Var's 2024 is excluded (almost empty); the Bouches-du-Rhône's 2024 is usable. The Var recorded fewer fires in 2013–2022 than the model expects, probably a change in recording (not proven). The 2023 form change is handled by counting fires ≥ 0.1 ha only. BDIFF gives the commune of each fire, not its position.
- **Landscape.** One BD TOPO edition (September 2026) is used for every year, so buildings built later are counted for earlier years.
- **Burning calendar.** Var: the rules of the 2013 and 2025 orders are applied to all years; the 2006–2012 rules and the prefect's extra red days are not included. Bouches-du-Rhône: restricted periods February–March and June–September until 19 December 2013, then June–September only.
- **Sample size.** Var: 902 positive commune-days over 19 years, 11 to 124 per year. Bouches-du-Rhône: 1,238 over 20 years. Gains are reported with 95% ranges from resampling years.

## 9. What this means

1. **For each commune, the small model is better than the FWI.** Version 4 ranks commune-days better than the FWI alone in every year of both departments, and catches about three times more fire starts in the day's top 15 communes. At department level, the gain is smaller (13 of 19 years) and does not show on the two test years alone.
2. **The missing information is human ignition**: who starts fires, when and where. It is not missing weather. The burning calendar is the first human signal we found that the FWI misses.
3. **Fire services need two numbers:** the chance of a fire starting, and how bad a fire would get. FWI answers the second only.
4. **Judge any model against naive references** (calendar, commune size or history). Without them, our results would look much better than they are.

## 10. Not done yet

- Timestral-209M (about an hour of CPU; not expected to change the conclusion), the J0 and J+7 horizons, burned area as the target, and SHAP explanations.
- Score the 2026 forecasts of versions 2, 3 and 4 when BDIFF publishes 2026.
- Forecasts further ahead than tomorrow (D+2 to D+90).
- One BD TOPO edition (2026) is used for all years, so buildings built after a given year are counted for it.

## 11. Recommendations for phase 2

1. **Predict unusual local risk**, not raw risk: "commune X is 4 times above its normal today". This is what fire services can act on, and it is a fairer test than ranking big communes first. Use the same ideas at commune level: few inputs, the burning period, and each commune's local 30-day dryness.
2. **More fires to learn from:** done for the Bouches-du-Rhône (version 4). Next: Alpes-Maritimes, Vaucluse, Corsica and the other Mediterranean departments in BDIFF.
3. **Information that changes daily and points to a commune**, the most likely way to improve on the ~25% reached by fixed rankings: burning declarations, works, fire series, finer local weather (ERA5-Land or AROME, free accounts).
4. **Fix the labels at the source:** ask BDIFF / DDTM 83 about the Var's 2024 records.
5. **Keep evaluating on every year (leave-one-year-out)**, not one test year. Single years hold only 20–60 fire days.

## Reproduce

```
uv run python -m firepredict.download              # raw data → data/raw/
uv run python -m firepredict.audit                 # data checks → results/audit/
uv run python -m firepredict.dataset               # daily table
uv run python -m firepredict.features              # daily features
uv run python -m firepredict.baseline_fwi          # FWI baseline
uv run python -m firepredict.model_catboost        # CatBoost yes/no (add "oracle" for perfect weather)
uv run python -m firepredict.model_catboost_classes
uv run python -m firepredict.model_timestral 28M
uv run python -m firepredict.model_spread
uv run python -m firepredict.communes
uv run python -m firepredict.model_communes
uv run python -m firepredict.drought_check         # dryness features vs FWI codes
uv run python -m firepredict.human_check           # holidays, causes, burning periods
uv run python -m firepredict.compare_years         # every year in turn: what adds to the FWI
uv run python -m firepredict.model_small           # small day model, benchmark protocol
uv run python -m firepredict.download bdtopo       # IGN BD TOPO, Var (455 MB)
uv run python -m firepredict.landscape             # buildings and roads in wildland per commune
uv run python -m firepredict.model_spatial         # spatial Bayesian commune rates (about 15 min)
uv run python -m firepredict.compare_communes      # every year in turn, commune level (add "catboost": slow)
uv run python -m firepredict.download meteo-latest # refresh this year's weather before a new forecast
uv run python -m firepredict.value_check           # fires caught vs FWI references, 95% ranges
uv run python -m firepredict.calibration           # probabilities, PR AUC, confusion table
uv run python -m firepredict.freeze                # freeze the model, save 2026 forecasts
uv run python -m firepredict.risk_map              # commune map → results/<department>/risk_map.html
uv run python -m firepredict.pipeline --department 13 data   # Bouches-du-Rhône tables
uv run python -m firepredict.model_v4 evaluate     # version 4 ladder, both departments → results/v4/
uv run python -m firepredict.model_v4 value        # version 4 against the FWI references
uv run python -m firepredict.model_v4 forecast     # today's forecast, both departments → forecasts/v4/
```

The benchmark repository has no licence. Its scoring and class methods were reimplemented from their description, not copied.
