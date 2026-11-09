# A second department: Bouches-du-Rhône (13), and transfer between departments

*9 November 2026. Same data sources, code and protocol as the Var report ([`phase1_results.md`](phase1_results.md)), with the corrected protocol: no input uses recent fire records, and each year is hidden in turn. Units are positive commune-days: days on which at least one fire of 0.1 ha or more started in the commune.*

## Summary

- **The code now runs for any department** set up in `department.py`; the Bouches-du-Rhône (13) is the second. The Var's results are unchanged, to the last digit.
- **Bouches-du-Rhône data are usable for all 20 years, 2024 included.** The Var's near-empty 2024 does not recur, and the 2023 form change leaves no visible artifact above 0.1 ha.
- **The burning calendar changed in 2013**: the 2004 order restricted burning from February to March and banned it from June to September; the 2013 order keeps only the summer ban. The orders after 2016 could not be read.
- **In the Bouches-du-Rhône the small model beats the FWI by far** (AUC 0.865 against 0.781, 20 of 20 years; +68% positive commune-days at a 10% budget), **but history × FWI only slightly** (+0.007, 14 of 20 years, p = 0.03; +4%). Fires there are concentrated in a few communes, so the commune's history carries most of the signal.
- **The small model travels.** Learned on one department and applied to the other (with the target's own history), it loses little ranking: Var 0.770 against 0.782 at home, Bouches-du-Rhône 0.863 against 0.865. It still beats history × FWI in the Var (15/19 years, p = 0.02) and is level with it in the Bouches-du-Rhône (13/20, p = 0.17).
- **Without the target's fire history, landscape does most of history's job.** Weather and calendar alone fall far behind (Var 0.735, 13: 0.793); adding buildings and roads in wildland brings them to 0.754 and 0.855, on a par with history × FWI on AUC (−0.003 in both), though below it for the 15 riskiest communes each day in the Bouches-du-Rhône (−10%).
- **Pooling both departments neither helps nor hurts**, and a department indicator changes nothing.
- **Probabilities do not travel**: a transferred model's chances are off by up to three times in the top bands, so they need recalibration in the new department.

## 1. What changed in the code

- **One setting chooses the department**: `FIREPREDICT_DEPARTMENT` (or `pipeline --department`), default 83. Every module reads it from [`department.py`](../src/firepredict/department.py). Raw data, processed tables and results have one folder per department (`data/raw/13/`, `data/processed/13/`, `results/13/`).
- **Per-department settings**, in `department.py`: school-holiday zone, years with unusable fire records, FWI latitude, and the burning calendar as dated rules (each prefectoral order in force from its date).
- **Downloads**: the Corine box comes from the department's commune boundaries; the BD TOPO edition is read from IGN's feed (the 15 September 2026 edition for both departments); only the three layers used are kept (1.4–1.5 GB instead of 2.7–3 GB), and the archive is deleted. The manifest keeps the archive's URL, size and SHA-256.
- **Var unchanged**: rebuilt from the same raw files, the Var's daily, feature, commune and landscape tables are identical, and so are the outputs of `compare_communes` and `value_check` (small model AUC 0.782 against 0.757 for history × FWI, 16/19 years, 902 positive commune-days).
- **The FWI latitude** is the centroid of the commune boundaries (Var 43.4°, Bouches-du-Rhône 43.5°). In xclim, any latitude between 30° and 90° N gives the same day-length factors, so this choice does not change any FWI value.

## 2. The Bouches-du-Rhône burning calendar

| Period in force | Order | Orange (restricted) | Red (banned) |
|---|---|---|---|
| 2006 – 19 Dec 2013 | n°1272 of 19 May 2004 | 1 Feb – 31 Mar | 1 June – 30 Sept |
| from 20 Dec 2013 | n°2013354-0004 of 20 Dec 2013, modified 10 June 2014 and 31 May 2016 | none | 1 June – 30 Sept |

- **2004 order** ([text](https://www.aixenprovence.fr/IMG/pdf_emploidufeu.pdf)): two "périodes sensibles", 1 February – 31 March and 1 June – 30 September. In these periods, fire use in and within 200 m of woodland is banned except with the mayor's authorisation, asked 5 days ahead. The companion order on controlled burning bans it from 1 June to 30 September ([text](http://www.ofme.org/documents/Loisreglements/Arretes_prefectoraux/13_brulage_dirige_incineration.pdf)). Both sensitive periods have the same legal rule; we code February–March as "orange" (restricted, like the Var's orange period) and June–September as "red".
- **2013 order**: burning green waste is banned in general. The exceptions (brush-clearing duty, forestry, farming) may burn from 1 October to 31 May, with wind under 30 km/h, and never from 1 June to 30 September near woodland. There is no February–March period any more. Sources: the prefecture's memo ([PDF](https://www.bouches-du-rhone.gouv.fr/content/download/21516/131519/file/160302_Memo_emploi_du_feu_Actu.pdf)), Ventabren's municipal order quoting it ([PDF](https://www.ventabren.fr/wp-content/uploads/2015/11/Enmploi-du-feu-10_2016.pdf)), and the 2016 amendment ([PDF](https://www.ofme.org/documents/Loisreglements/Arretes_prefectoraux/13_20160531_AP_modificatif_emploi_du_feu_signe.pdf)). We read the memo and these secondary texts, not the full 2013 order (published in the prefecture's RAA of December 2013).
- **Not confirmed**: the Chambre d'agriculture PACA mentions orders of 14 March 2017 and 24 September 2025, still with a June–September ban ([page](https://paca.chambres-agriculture.fr/sinformer/agriculteur-lessentiel-a-savoir/arretes-brulage)). We did not find their text, so the 2013 rule is kept to 2025. A municipal burning form of March 2025 still cites the 2013 order ([Eyguières](https://www.eyguieres.org/wp-content/uploads/2025/03/Declaration-de-brulage.pdf)).
- **Not included**: the prefect's extra bans in drought (for example, the ban extended to 15 November 2017, [order](https://www.lancon-provence.fr/fileadmin/user_upload/11_Actualites/Vent_violent/ARRETE_PREF13_INTERDICTION_BRULAGE_15_NOV_2017.pdf)).

**What the fire records say.** Burning-type fires (individuals and works) compared with what the FWI alone predicts:

| February–March | Bouches-du-Rhône | Var |
|---|---|---|
| 2006–2013 | 3.12 (28 fires) | 3.57 (72 fires) |
| 2014–2025 | 1.30 (18 fires) | 1.36 (44 fires) |

The February–March excess drops after 2013 in the Bouches-du-Rhône, as the order change would predict, but it drops just as much in the Var, whose coded calendar did not change. So the drop cannot be attributed to the 2013 order; the counts are also small.

## 3. Data quality

| Check | Bouches-du-Rhône (13) | Var (83) |
|---|---|---|
| BDIFF fires 2006–2025, all sizes / ≥ 0.1 ha | 4,302 / 1,316 | 2,685 / 929 |
| Positive commune-days per labelled year | 28–118 (1,238 in 20 years) | 11–124 (902 in 19 years) |
| 2024 | 225 fires, 30 of ≥ 0.1 ha, 22 fire days (2014: 25) | 3 fires: excluded |
| 2023 form change | new-form fields filled from 2023; no record flagged "partial or invalid" | every 2023 record flagged |
| Fires < 100 m² | 17–73 a year in 2006–2015, 106–182 a year since 2020: a rise before the form change, no 2023 jump | jump from 10–31 to 216 in 2023 |
| Core weather stations (≥ 95% coverage) | 11–15 per variable (TX 12, RR 15, wind 11, humidity 12) | 14–30 |
| Fires in the 15 most affected communes | 52% | 29% |

- **2024 is kept** for the Bouches-du-Rhône: it is the quietest year but within the range of the others. The Var's empty 2024 is a Var problem, not a BDIFF-wide one.
- **The 0.1 ha cut-off is kept** for comparability. The rise in tiny fires started before the 2023 form change; we cannot tell whether it is more small fires or better recording. Fires of 0.1 ha or more show no clear trend.
- **Marseille is one commune** (13055) in both BDIFF and geo.api.gouv.fr; it has the most fires (133 of 0.1 ha or more).
- **Fires are concentrated**: the 15 most affected communes hold half the fire starts, against 29% in the Var. This makes "where" much easier to predict than in the Var.

## 4. Bouches-du-Rhône alone

Same protocol as the Var: each of the 20 years (2006–2025) hidden in turn; each commune's usual rate never includes the hidden year or the row's own year. 119 communes, 862,274 commune-days, 1,238 positive commune-days. The daily capacity "top 15" is 13% of the department's communes (Var: 10%).

| Model | AUC | Years better than history × FWI | Wilcoxon p | Top 15/day caught |
|---|---|---|---|---|
| Commune history only | 0.785 | 0/20 | | 605 |
| **History × FWI (reference)** | 0.858 | — | | 605 |
| History × local FWI | 0.854 | 5/20 | 0.009 (worse) | 631 |
| 1. history + local FWI | 0.858 | 11/20 | 0.93 | 627 |
| 2. + burning calendar | 0.861 | 11/20 | 0.35 | 629 |
| **3. + 30-day dryness = small model** | **0.865** | **14/20** | **0.03** | **634** |
| 4. + landscape | 0.867 | 13/20 | 0.03 | 605 |
| *Var, small model (for comparison)* | *0.782* | *16/19* | *< 0.001* | *213* |
| *Var, history × FWI* | *0.757* | — | | *220* |

**Against the references** (95% ranges from resampling the 20 years):

| Reference | Ref AUC | AUC gain (95%) | Years better | 1% budget | 5% budget | 10% budget | Top 15/day |
|---|---|---|---|---|---|---|---|
| FWI alone (department) | 0.781 | +0.084 (0.073 to 0.098) | 20/20 | 54 → 273 | 237 → 570 | 454 → 764, +68% (+51% to +85%) | 169 → 634 |
| Local FWI alone | 0.782 | +0.084 (0.075 to 0.092) | 20/20 | 68 → 273 | 251 → 570 | 458 → 764, +67% (+59% to +81%) | 247 → 634 |
| History × FWI | 0.858 | +0.007 (0.001 to 0.012) | 14/20 | 263 → 273, +4% (−2% to +10%) | 556 → 570, +3% (−3% to +8%) | 737 → 764, +4% (+1% to +7%) | 602 → 634, +5% (+2% to +9%) |

The same table for the Var (corrected protocol, unchanged by this work):

| Reference | Ref AUC | AUC gain (95%) | Years better | 1% budget | 5% budget | 10% budget | Top 15/day |
|---|---|---|---|---|---|---|---|
| FWI alone (Var) | 0.718 | +0.064 (0.051 to 0.077) | 19/19 | 42 → 59 | 142 → 204 | 230 → 354, +54% (+33% to +74%) | 85 → 213 |
| Local FWI alone | 0.728 | +0.054 (0.040 to 0.069) | 18/19 | 37 → 59 | 125 → 204 | 256 → 354, +38% (+20% to +58%) | 123 → 213 |
| History × FWI | 0.757 | +0.025 (0.015 to 0.034) | 16/19 | 76 → 59, −22% (−36% to −10%) | 209 → 204, −2% (−16% to +14%) | 307 → 354, +15% (+2% to +30%) | 222 → 213, −4% (−10% to +2%) |

- **Against the FWI as fire services use it, the gain is larger than in the Var**: the Bouches-du-Rhône's fires are concentrated in a few communes, so knowing where they usually start is worth a lot.
- **Against history × FWI, the gain is small**: +0.007 AUC, 14 of 20 years, +3–5% positive commune-days for the same warnings. Weather, dryness and the burning calendar add little once the commune's history is known.
- **Local FWI does not help here**: history × local FWI is worse than history × the department FWI (5/20 years). With 11–15 core stations, the nearest station is often far from the commune or on the coast.
- **The burning calendar adds little** (step 2: +0.002, p = 0.35). Its orange period exists only up to 2013.
- **Robustness**: without 2023, 0.866 against 0.858 (13/19 years, p = 0.02); for fires of 1 ha or more (406 positive commune-days), 0.898 against 0.872 (19/20 years, p < 0.001).

**Calibration** (small model, 20 hidden years):

| Predicted chance | Commune-days | Positive | Mean predicted | Observed | *Var: predicted / observed* |
|---|---|---|---|---|---|
| < 0.03% | 367,365 | 43 | 0.011% | 0.012% | *0.014% / 0.011%* |
| 0.03–0.1% | 218,588 | 126 | 0.058% | 0.058% | *0.061% / 0.056%* |
| 0.1–0.3% | 165,329 | 243 | 0.175% | 0.147% | *0.167% / 0.169%* |
| 0.3–1% | 89,780 | 400 | 0.51% | 0.45% | *0.38% / 0.45%* |
| ≥ 1% | 21,212 | 426 | 1.84% | 2.01% | *1.09% / 0% (17 commune-days)* |

Predicted and observed rates agree within about 20% in every band; the model is slightly too high at 0.1–1%. Log loss improves on climatology by 9% (0.00913 against 0.01003; Var 6%). PR AUC: 0.019 pooled against 0.018 for history × FWI and 0.0045 for FWI alone; the reference is higher in 11 of 20 years, as in the Var.

## 5. Transfer between departments

Each year Y is hidden in both departments at once. For a target department T:
- **in-department**: learned on T, other years (the results above);
- **transfer**: learned only on the other department, other years;
- **pooled**: learned on both departments, other years.

Every model uses T's own usual rate (T's years other than Y), except the "no history" variants, which use nothing from T's fire records. Landscape = buildings within 50 m of wildland and roads in wildland (BD TOPO). References are computed on T as before.

AUC over the hidden years (Var: 19 years, 902 positive commune-days; Bouches-du-Rhône: 20 years, 1,238), years better than history × FWI (Wilcoxon p), and positive commune-days caught by warning the 15 top communes each day:

| Model | Var: AUC | Var: vs history × FWI | Var: top 15/day | 13: AUC | 13: vs history × FWI | 13: top 15/day |
|---|---|---|---|---|---|---|
| FWI alone | 0.718 | 2/19 | 82 | 0.781 | 0/20 | 175 |
| Local FWI alone | 0.728 | 3/19 | 120 | 0.782 | 0/20 | 255 |
| History × FWI (reference) | 0.757 | — | 220 | 0.858 | — | 605 |
| in-department: small | **0.782** | 16/19 (p < 0.001) | 213 | 0.865 | 14/20 (p = 0.03) | **634** |
| in-department: small + landscape | 0.781 | 18/19 (p < 0.001) | 222 | 0.867 | 13/20 (p = 0.03) | 605 |
| in-department: no history | 0.746 | 7/19 | 123 | 0.798 | 0/20 | 289 |
| in-department: no history + landscape | 0.767 | 11/19 (p = 0.20) | 197 | 0.857 | 7/20 (p = 0.67) | 545 |
| **transfer: small** | 0.770 | 15/19 (p = 0.02) | 221 | 0.863 | 13/20 (p = 0.17) | 613 |
| transfer: small + landscape | 0.768 | 15/19 (p = 0.02) | 218 | 0.866 | 15/20 (p = 0.06) | 611 |
| transfer: no history | 0.735 | 4/19 | 127 | 0.793 | 0/20 | 286 |
| **transfer: no history + landscape** | 0.754 | 7/19 (p = 0.40) | 197 | 0.855 | 9/20 (p = 0.65) | 544 |
| **pooled: small** | 0.777 | 16/19 (p = 0.001) | 220 | 0.866 | 13/20 (p = 0.008) | 633 |
| pooled: small + department indicator | 0.776 | 16/19 (p = 0.001) | 219 | 0.866 | 13/20 (p = 0.008) | 633 |
| pooled: small + landscape | 0.775 | 18/19 (p < 0.001) | 224 | **0.868** | 14/20 (p = 0.009) | 611 |

- **Transfer with the target's own history** loses 0.012 AUC in the Var and 0.002 in the Bouches-du-Rhône compared with learning at home. The weights learned in one department mostly hold in the other.
- **Without the target's history**, weather, dryness and calendar alone are far behind. Adding landscape recovers most of the gap (Var 0.735 → 0.754, 13: 0.793 → 0.855), to the level of history × FWI on AUC, but not for the top of the daily ranking in the Bouches-du-Rhône (544 against 605). The same holds when learning at home, so this is what landscape can do, not a transfer effect.
- **Pooled**: no gain over learning at home. The department indicator changes nothing because each department's usual rate already sets its base level; we keep the pooled model without it, which can also be applied to a new department.

**Against the references, for the main variants** (positive commune-days caught; 95% ranges from resampling years):

| Target | Model | AUC gain vs history × FWI (95%) | 10% budget: vs FWI alone | 10% budget: vs history × FWI | Top 15/day: vs history × FWI |
|---|---|---|---|---|---|
| Var | in-department: small | +0.025 (0.015 to 0.034) | 230 → 354, +54% | 307 → 354, +15% (+2% to +30%) | 222 → 213, −4% (−10% to +2%) |
| Var | transfer: small | +0.013 (0.004 to 0.022) | 230 → 330, +43% | 307 → 330, +7% (+1% to +17%) | 222 → 220, −1% (−7% to +6%) |
| Var | transfer: no history + landscape | −0.003 (−0.013 to 0.008) | 230 → 320, +39% | 307 → 320, +4% (−7% to +16%) | 222 → 197, −11% (−24% to +2%) |
| Var | pooled: small | +0.019 (0.011 to 0.028) | 230 → 352, +53% | 307 → 352, +15% (+6% to +24%) | 222 → 219, −1% (−7% to +6%) |
| 13 | in-department: small | +0.007 (0.001 to 0.012) | 454 → 764, +68% | 737 → 764, +4% (+1% to +7%) | 602 → 634, +5% (+2% to +9%) |
| 13 | transfer: small | +0.004 (−0.001 to 0.009) | 454 → 732, +61% | 737 → 732, −1% (−4% to +3%) | 602 → 613, +2% (−1% to +5%) |
| 13 | transfer: no history + landscape | −0.003 (−0.014 to 0.007) | 454 → 703, +55% | 737 → 703, −5% (−10% to 0%) | 602 → 544, −10% (−14% to −5%) |
| 13 | pooled: small | +0.007 (0.002 to 0.012) | 454 → 758, +67% | 737 → 758, +3% (−1% to +6%) | 602 → 633, +5% (+2% to +8%) |

All full tables, including the 1% and 5% budgets and the top 5 and top 30 per day: `results/transfer/gains.csv`.

**Calibration does not transfer.** Commune-days grouped by the transferred small model's predicted chance:

| Predicted chance | Var, learned on 13: predicted / observed | 13, learned on the Var: predicted / observed |
|---|---|---|
| < 0.03% | 0.011% / 0.023% | 0.014% / 0.006% |
| 0.03–0.1% | 0.058% / 0.071% | 0.060% / 0.043% |
| 0.1–0.3% | 0.173% / 0.151% | 0.172% / 0.170% |
| 0.3–1% | 0.45% / 0.36% | 0.46% / 0.89% |
| ≥ 1% | 1.26% / 0.98% | 1.21% / 3.80% |

The ranking travels, the probabilities do not: learned on the Var, the model gives the Bouches-du-Rhône's riskiest commune-days a third of their real rate. A model moved to a new department would need its probabilities recalibrated there (for example on one year of local records).

## 6. What can be claimed, and limits

1. **The method generalises; its gain over history × FWI depends on the department.** In both departments the small model beats the FWI alone in every year. Its edge over history × FWI is clear in the Var (16/19) and small in the Bouches-du-Rhône (14/20, +0.007), where fires cluster in a few communes and history says most of what can be said.
2. **The ranking rules learned in one department hold in the other.** This is the main new result: a model fitted on the Var ranks the Bouches-du-Rhône's commune-days almost as well as a model fitted there, and the reverse.
3. **For a department with no usable fire history, landscape is a good stand-in**, roughly as good as the FWI combined with ten years of local history on AUC, though not for the very top of the daily ranking. This supports the hypothesis that landscape helps when the target's own history is not used.
4. **Probabilities must be recalibrated locally**: base rates differ between departments (0.144% against 0.086% of commune-days with a fire start).

Limits:
- **Exploratory, as in the Var.** The model's inputs were chosen on the Var's years; the Bouches-du-Rhône is a fresh department for the in-department test, but the same years, and its burning calendar was coded after the Var's results were known. p-values and ranges are optimistic.
- **Shared years.** Each hidden year is hidden in both departments, so no model learns from the target year's fires anywhere. Neighbouring departments share weather, so the transfer test is not a test on an independent climate.
- **Burning calendar uncertain after 2016**, and the 2013 order was read from the prefecture's memo and municipal texts, not from its full text. The coded calendar gives the Bouches-du-Rhône no orange period after 2013; the burning-period input therefore means something slightly different in the two departments, which also affects the transfer.
- **Few core weather stations** in the Bouches-du-Rhône (11–15), which may explain why local FWI does not help there.
- **The top-15 capacity is not the same share** in the two departments (13% of communes against 10%).
- **Not done**: CatBoost, the spatial Bayesian model and the department-level analyses (compare_years, human_check) were not rerun for the Bouches-du-Rhône beyond a check of the burning periods, and no model was frozen for it.

## Reproduce

```bash
uv run python -m firepredict.pipeline --department 13 data             # download and build (BD TOPO about 0.5 GB)
FIREPREDICT_DEPARTMENT=13 uv run python -m firepredict.compare_communes
FIREPREDICT_DEPARTMENT=13 uv run python -m firepredict.value_check
FIREPREDICT_DEPARTMENT=13 uv run python -m firepredict.calibration
FIREPREDICT_DEPARTMENT=13 uv run python -m firepredict.robustness
uv run python -m firepredict.transfer                                  # needs both departments' data stage
```
