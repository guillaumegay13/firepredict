"""Version 4: how should the shared commune model handle seasons?

A ladder of shared models (all departments together, logistic regression),
each step adding one thing, compared on the same folds:
  A  v2 inputs + department level                       (pooled reference)
  B  A + log usual rate x June-September                (= v3)
  C  A with a clean usual rate: positive commune-days per usable day,
     smoothed toward the department rate (empirical Bayes, gamma-Poisson)
  D  C + the date as a smooth yearly curve (one sine/cosine pair of the
     target day's day of year), shared by all communes
  E  D + the same curve allowed to differ by department
  F  E + each commune's seasonal adjustment: its share of fire days in
     June-September, smoothed toward the department's share (beta-binomial),
     as log(commune share / department share) in summer and the same for the
     rest of the year. Centred on 0 = the department's pattern; a commune with
     few fires stays near 0.
  G  F fitted with more weight on years close to the year predicted:
     weight 0.5 ** (|year - predicted year| / half-life). The half-life (3, 5,
     10 years or none) is chosen in each fold inside the training years only,
     by predicting the last 3 training years from the earlier ones.
  H1 G + a second sine/cosine pair, shared and by department
  H2 G + a level for each season (Nov-Jan, Feb-Mar, Apr-May, Oct; Jun-Sep is
     the reference), shared and by department
Smoothing strengths are estimated in each fold from training years only.

Two tests:
  loyo     each year hidden in every department at once; models learn on the
           other years (as model_v3)
  forward  each year 2016-2025 predicted by models that learn on earlier
           years only
In both, a training row's history inputs never use its own year, and no input
ever uses the hidden (test) year or, in the forward test, any later year. A
training row may use other training years after its own.

Checks, per department: log loss, ROC AUC by year, positive commune-days
caught by the daily top 15, and predicted / observed (E/O) by season
(Nov-Jan, Feb-Mar, Apr-May, Jun-Sep, Oct) and by history level (communes
split in three by their fold rate). Plus Draguignan (83050) in July and
Aix-en-Provence (13001) in Nov-Jan.

Acceptance rules, fixed before looking at results (vs A and vs B):
  1. lower log loss in both departments (loyo and forward)
  2. daily top-15 capture not more than 2 points below A in either department
  3. E/O between 0.8 and 1.25 in every season x department group with at
     least 30 observed positive commune-days
  4. no group with at least 10 observed positive commune-days off by a factor
     of 2 or more
Sparse groups are reported but count as inconclusive.

Second protocol, defined on 9 November 2026 AFTER seeing the results of the
first (so exploratory): rules 3 and 4 with three outcomes. For each group, a
95% range of E/O comes from resampling the test years (2000 times), which
keeps fires that cluster in a year together:
  pass          the whole range is inside the allowed band
  fail          the whole range is outside it
  inconclusive  otherwise
Rule 3's band is 0.8-1.25 (groups >= 30 events); rule 4's is 0.5-2 (groups
>= 10 events), both bounds included, whereas the first protocol fails a group
at exactly 2 or 0.5. The first protocol's results are kept (gates.csv).
"Inconclusive" is unresolved evidence, not acceptance. The ranges come from
predictions already fitted, so they leave out the uncertainty of fitting and
of choosing the ladder.

The frozen v4 is step F (simplest of the steps without a clear failure; G,
H1, H2 were not reliably better in paired year-resampled comparisons).

Usage:
    uv run python -m firepredict.model_v4 evaluate     # both tests, writes results/v4/
    uv run python -m firepredict.model_v4 rejudge      # second protocol from saved predictions
    uv run python -m firepredict.model_v4 freeze       # models/small_commune_v4 (F)
    uv run python -m firepredict.model_v4 forecast     # archive forecasts/v4/<department>/<day>.csv
    uv run python -m firepredict.model_v4 value        # v4 vs FWI references, fires caught
"""

import json
import subprocess
import sys
from datetime import datetime, timezone

import joblib

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import betaln, gammaln
from sklearn.metrics import log_loss, roc_auc_score

from firepredict.compare_communes import per_year
from firepredict.dataset import ROOT
from firepredict.model_communes import HORIZON, fit_small
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from firepredict.model_v3 import DEPARTMENTS, INDICATORS, build_panels, panel_path, sha256

OUT_DIR = ROOT / "results" / "v4"
SUMMER = [6, 7, 8, 9]
SEASONS = {"Nov-Jan": [11, 12, 1], "Feb-Mar": [2, 3], "Apr-May": [4, 5], "Jun-Sep": SUMMER, "Oct": [10]}
WEATHER = ["log_local_fwi", "local_wb_30d", "orange_period", "red_period"]
LADDER = {
    "A pooled v2": ["log_usual_rate", *WEATHER, *INDICATORS],
    "B v3 (rate x summer)": ["log_usual_rate", *WEATHER, "rate_x_summer", *INDICATORS],
    "C clean rate": ["log_rate", *WEATHER, *INDICATORS],
    "D + date curve": ["log_rate", *WEATHER, *INDICATORS, "doy_sin", "doy_cos"],
    "E + date curve by department": ["log_rate", *WEATHER, *INDICATORS, "doy_sin", "doy_cos",
                                     *[f"{d}_doy_{t}" for d in INDICATORS for t in ("sin", "cos")]],
    "F + commune season": ["log_rate", *WEATHER, *INDICATORS, "doy_sin", "doy_cos",
                           *[f"{d}_doy_{t}" for d in INDICATORS for t in ("sin", "cos")], "commune_season"],
}  # fmt: skip
_F = LADDER["F + commune season"]
_DEPT = lambda cols: [f"{d}_{c}" for d in INDICATORS for c in cols]  # noqa: E731
SEASON_LEVELS = ["season_nov_jan", "season_feb_mar", "season_apr_may", "season_oct"]
LADDER |= {
    "G + recency weights": _F,
    "H1 + 2nd curve": [*_F, "doy2_sin", "doy2_cos", *_DEPT(["doy2_sin", "doy2_cos"])],
    "H2 + season levels": [*_F, *SEASON_LEVELS, *_DEPT(SEASON_LEVELS)],
}
WEIGHTED = {"G + recency weights", "H1 + 2nd curve", "H2 + season levels"}
HALF_LIVES = [3, 5, 10, None]
INNER_YEARS = 3
CASES = {"Draguignan, July": ("83", "83050", [7]), "Aix-en-Provence, Nov-Jan": ("13", "13001", [11, 12, 1])}
FORWARD_YEARS = range(2016, 2026)


# --- smoothing ---------------------------------------------------------------


def gamma_poisson_strength(k: np.ndarray, exposure: np.ndarray) -> tuple[float, float]:
    """Department rate per day and prior strength (in days) of a gamma-Poisson model."""
    rate = k.sum() / exposure.sum()

    def nll(log_kappa):
        a, kappa = np.exp(log_kappa) * rate, np.exp(log_kappa)
        return -(gammaln(k + a) - gammaln(a) + a * np.log(kappa / (kappa + exposure)) + k * np.log(exposure / (kappa + exposure))).sum()

    return rate, float(np.exp(minimize_scalar(nll, bounds=(0, 15), method="bounded").x))


def beta_binomial_strength(k_summer: np.ndarray, k: np.ndarray) -> tuple[float, float]:
    """Department summer share and prior strength (in fire days) of a beta-binomial model."""
    share = k_summer.sum() / k.sum()
    used = k > 0

    def nll(log_kappa):
        a, b = np.exp(log_kappa) * share, np.exp(log_kappa) * (1 - share)
        return -(betaln(k_summer[used] + a, k[used] - k_summer[used] + b) - betaln(a, b)).sum()

    return share, float(np.exp(minimize_scalar(nll, bounds=(-3, 10), method="bounded").x))


# --- inputs ------------------------------------------------------------------


def target_day(panel: pd.DataFrame) -> pd.DatetimeIndex:
    return panel.index.get_level_values("date") + pd.Timedelta(days=HORIZON)


def history_inputs(panel: pd.DataFrame, train_years: list[int]) -> pd.DataFrame:
    """Clean smoothed rate and commune seasonal adjustment for every row, from
    `train_years` only. For a row in a training year, everything is estimated
    without that year: the commune's counts, and the department rate, summer
    share and smoothing strengths."""
    day = target_day(panel)
    year, insee = day.year.to_numpy(), panel.index.get_level_values("insee")
    summer = day.month.isin(SUMMER)
    frame = pd.DataFrame({"year": year, "insee": insee, "y": panel.target.to_numpy(), "summer": summer})
    frame["y_summer"] = frame.y * frame.summer
    per = frame.groupby(["year", "insee"]).agg(k=("y", "sum"), days=("y", "size"), k_summer=("y_summer", "sum"))
    per = per[per.index.get_level_values("year").isin(train_years)]
    n_years = len(per.index.get_level_values("year").unique())
    total = per.groupby(level="insee").sum()

    out = pd.DataFrame(index=panel.index, columns=["log_rate", "commune_season", "old_rate"], dtype=float)
    for y in np.unique(year):
        rows = year == y
        own_year = y in per.index.get_level_values("year")
        tot = total.sub(per.xs(y, level="year"), fill_value=0) if own_year else total
        rate, kappa = gamma_poisson_strength(tot.k.to_numpy(), tot.days.to_numpy())
        share, kappa_s = beta_binomial_strength(tot.k_summer.to_numpy(), tot.k.to_numpy())
        t = tot.reindex(insee[rows])
        smoothed = (t.k.to_numpy() + kappa * rate) / (t.days.to_numpy() + kappa)
        q = (t.k_summer.to_numpy() + kappa_s * share) / (t.k.to_numpy() + kappa_s)
        out.loc[rows, "log_rate"] = np.log(smoothed)
        out.loc[rows, "commune_season"] = np.where(summer[rows], np.log(q / share), np.log((1 - q) / (1 - share)))
        # The v2/v3 rate, for models A and B: fire days per year over the same years.
        out.loc[rows, "old_rate"] = t.k.to_numpy() / (n_years - own_year)
    return out


def inputs(panel: pd.DataFrame, code: str, train_years: list[int]) -> pd.DataFrame:
    return other_inputs(history_inputs(panel, train_years), panel, code)


def other_inputs(X: pd.DataFrame, panel: pd.DataFrame, code: str) -> pd.DataFrame:
    """Add the weather, calendar and department inputs to the history inputs X."""
    X = X.copy()
    X["log_usual_rate"] = np.log(X.old_rate + 0.01)  # as in v2/v3
    X["log_local_fwi"] = np.log1p(panel.local_fwi)
    X["local_wb_30d"] = panel.local_wb_30d
    X["orange_period"], X["red_period"] = panel.orange_period, panel.red_period
    X["rate_x_summer"] = X.log_usual_rate * X.red_period
    doy = 2 * np.pi * target_day(panel).day_of_year.to_numpy() / 365.25
    X["doy_sin"], X["doy_cos"] = np.sin(doy), np.cos(doy)
    X["doy2_sin"], X["doy2_cos"] = np.sin(2 * doy), np.cos(2 * doy)
    month = target_day(panel).month
    for level, (_, months) in zip(SEASON_LEVELS, [(k, v) for k, v in SEASONS.items() if k != "Jun-Sep"]):
        X[level] = month.isin(months).astype(float)
    for name in INDICATORS:
        X[name] = float(name == f"dept_{code}")
        for col in ["doy_sin", "doy_cos", "doy2_sin", "doy2_cos", *SEASON_LEVELS]:
            X[f"{name}_{col}"] = X[name] * X[col]
    return X


# --- folds -------------------------------------------------------------------


def run(panels: dict[str, pd.DataFrame], folds: list[tuple[list[int], int]], ladder: dict | None = None):
    """Hidden-year predictions per department and model, and each test row's
    rate from its fold's training years. Each fold: (training years, test year)."""
    ladder = LADDER if ladder is None else ladder
    out = {code: {name: [] for name in ladder} for code in DEPARTMENTS}
    rates = {code: [] for code in DEPARTMENTS}
    chosen = {}
    for train_years, test_year in folds:
        X, y, train, test = {}, {}, {}, {}
        for code, panel in panels.items():
            year = target_day(panel).year
            X[code] = inputs(panel, code, train_years)
            y[code] = panel.target
            train[code], test[code] = np.isin(year, train_years), year == test_year
            rates[code].append(X[code].log_rate[test[code]])
        for name, columns in ladder.items():
            data = pd.concat([X[c][columns][train[c]] for c in DEPARTMENTS])
            fill = data.median()
            weight = None
            if name in WEIGHTED:
                half_life = choose_half_life(panels, train_years, columns)
                weight = recency(np.concatenate([target_day(panels[c])[train[c]].year for c in DEPARTMENTS]), test_year, half_life)
                chosen.setdefault(name, []).append((test_year, half_life))
            model = fit(data.fillna(fill), pd.concat([y[c][train[c]] for c in DEPARTMENTS]), weight)
            for code in DEPARTMENTS:
                if test[code].any():
                    here = X[code][columns][test[code]].fillna(fill)
                    out[code][name].append(pd.Series(model.predict_proba(here)[:, 1], index=here.index))
        print(f"  {test_year} done", flush=True)
    scores = {code: {name: pd.concat(parts) for name, parts in models.items()} for code, models in out.items()}
    for name, picks in chosen.items():
        print(f"  {name}: half-life chosen per fold {picks}", flush=True)
    return scores, {code: pd.concat(parts) for code, parts in rates.items()}


def fit(X: pd.DataFrame, y: pd.Series, weight: np.ndarray | None):
    if weight is None:
        return fit_small(X, y)
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X, y, logisticregression__sample_weight=weight)


def recency(years: np.ndarray, predicted_year: int, half_life: int | None) -> np.ndarray:
    if half_life is None:
        return np.ones(len(years))
    return 0.5 ** (np.abs(years - predicted_year) / half_life)


_INNER: dict = {}  # inner-validation inputs, per evaluation (cleared by evaluate)


def choose_half_life(panels, train_years: list[int], columns: list[str]) -> int | None:
    """Half-life with the lowest log loss when the last INNER_YEARS training years
    are each predicted from the training years before them (training data only)."""
    losses = {h: 0.0 for h in HALF_LIVES}
    for inner_test in sorted(train_years)[-INNER_YEARS:]:
        inner_train = [y for y in train_years if y < inner_test]
        X, yy, tr, te = [], [], [], []
        for code, panel in panels.items():
            year = target_day(panel).year
            keep = np.isin(year, [*inner_train, inner_test])
            key = (code, tuple(inner_train), inner_test)
            if key not in _INNER:  # same inputs for every weighted model of a fold
                _INNER[key] = inputs(panel[keep], code, inner_train)
            Xc = _INNER[key][columns]
            X.append(Xc), yy.append(panel.target[keep])
            tr.append(np.isin(year[keep], inner_train)), te.append(year[keep] == inner_test)
        X, yy, tr, te = pd.concat(X), pd.concat(yy), np.concatenate(tr), np.concatenate(te)
        fill = X[tr].median()  # a validation year without fires still has a valid log loss
        years = np.concatenate([target_day(p).year[np.isin(target_day(p).year, [*inner_train, inner_test])] for p in panels.values()])
        for h in HALF_LIVES:
            model = fit(X[tr].fillna(fill), yy[tr], recency(years[tr], inner_test, h))
            losses[h] += log_loss(yy[te], model.predict_proba(X[te].fillna(fill))[:, 1], labels=[0, 1]) * te.sum()
    return min(losses, key=losses.get)


# --- the frozen model: F -----------------------------------------------------

VERSION = "small_commune_v4"
MODEL_DIR = ROOT / "models" / VERSION
ARCHIVE = ROOT / "forecasts" / "v4"
FROZEN = "F + commune season"  # chosen for parsimony: G, H1, H2 not reliably better
TEST_YEAR = 2026
STATUS = (
    "Frozen as a research candidate for an outcome-blinded evaluation on 2026 labels. Historically it has a "
    "lower log loss than v2/v3, but it did not meet the original calibration acceptance criteria. An exploratory "
    "protocol, defined after inspecting results, found no clearly failing calibration groups; many groups remain "
    "inconclusive. Reliable subgroup calibration, superiority to the G/H1/H2 variants, and operational usefulness "
    "have not been established."
)
SCORING_2026 = (
    "Labels: BDIFF fires >= 0.1 ha, first publication covering all of 2026, commune of first alert, positive "
    "commune-day = at least one such fire. Scored as frozen, both departments separately, on 1 Jan - 9 Nov 2026 "
    "(retrospective, outcome-blinded) and on later days archived in forecasts/v4/ (live): log loss, ROC AUC, "
    "positive commune-days caught by the daily top 15, E/O by season (Nov-Jan, Feb-Mar, Apr-May, Jun-Sep, Oct) "
    "with Poisson ranges (a single year cannot be resampled). Compared with v2 (Var), v3, FWI alone and FWI x commune history on the same "
    "commune-days. If BDIFF's 2026 records look incomplete (as the Var's 2024), report and do not score that department."
)


def freeze() -> dict:
    """Train F on every labelled year of every department; save the model, the
    commune history used for forecasts, and the 2026 forecasts with fingerprints."""
    if MODEL_DIR.exists():
        raise SystemExit(f"{MODEL_DIR} already exists: frozen versions are never overwritten.")
    panels = build_panels()
    columns = LADDER[FROZEN]
    X, y, history = {}, {}, []
    for code, panel in panels.items():
        day = target_day(panel)
        labelled = panel.target.notna().to_numpy()
        future = (day.year == TEST_YEAR) & ~labelled
        rows = panel[labelled | future]
        train_years = sorted(set(target_day(rows).year[rows.target.notna().to_numpy()]))
        X[code], y[code] = inputs(rows, code, train_years), rows.target
        # Commune history for forecasts: the 2026 rows use every training year.
        f = X[code][~rows.target.notna().to_numpy()]
        summer = target_day(f).month.isin(SUMMER)
        h = pd.DataFrame({"log_rate": f.log_rate.groupby(level="insee").first(),
                          "season_summer": f.commune_season[summer].groupby(level="insee").first(),
                          "season_rest": f.commune_season[~summer].groupby(level="insee").first()})  # fmt: skip
        history.append(h.reset_index().assign(department=code))
    train = {c: y[c].notna() for c in DEPARTMENTS}
    data = pd.concat([X[c][columns][train[c]] for c in DEPARTMENTS])
    fill = data.median()
    model = fit_small(data.fillna(fill), pd.concat([y[c][train[c]] for c in DEPARTMENTS]))

    MODEL_DIR.mkdir(parents=True)
    joblib.dump(model, MODEL_DIR / "model.joblib")
    pd.concat(history)[["department", "insee", "log_rate", "season_summer", "season_rest"]].to_csv(
        MODEL_DIR / "commune_history.csv", index=False)  # fmt: skip
    metadata = {
        "version": VERSION, "ladder_step": FROZEN, "status": STATUS, "scoring_2026": SCORING_2026,
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
        "departments": DEPARTMENTS, "inputs": columns,
        "weights_standardised": dict(zip(columns, model[-1].coef_[0].round(4).tolist())),
        "fill_values": fill.round(6).to_dict(),
        "training_positive_commune_days": {c: int(y[c][train[c]].sum()) for c in DEPARTMENTS},
    }  # fmt: skip
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")

    parts = []
    for code, panel in panels.items():
        day = target_day(panel)
        rows = panel[(day.year == TEST_YEAR) & panel.target.isna().to_numpy()]
        parts.append(pd.DataFrame({"department": code, "target_day": target_day(rows).strftime("%Y-%m-%d"),
                                   "insee": rows.index.get_level_values("insee"),
                                   "chance": predict(rows, code).round(8).to_numpy()}))  # fmt: skip
    forecasts = pd.concat(parts)
    forecasts.to_csv(MODEL_DIR / "forecasts_2026.csv", index=False)
    metadata["model_sha256"] = sha256(MODEL_DIR / "model.joblib")
    metadata["forecasts_2026"] = {"kind": "outcome-blinded retrospective forecasts", "rows": len(forecasts),
                                  "first_day": forecasts.target_day.min(), "last_day": forecasts.target_day.max(),
                                  "sha256": sha256(MODEL_DIR / "forecasts_2026.csv")}  # fmt: skip
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    return metadata


def predict(rows: pd.DataFrame, code: str) -> pd.Series:
    """Chance of a fire start for forecast rows of one department, with the frozen v4 model."""
    model = joblib.load(MODEL_DIR / "model.joblib")
    metadata = json.loads((MODEL_DIR / "metadata.json").read_text())
    h = pd.read_csv(MODEL_DIR / "commune_history.csv", dtype={"department": str, "insee": str})
    h = h[h.department == code].set_index("insee").reindex(rows.index.get_level_values("insee"))
    summer = target_day(rows).month.isin(SUMMER)
    history = pd.DataFrame({"log_rate": h.log_rate.to_numpy(),
                            "commune_season": np.where(summer, h.season_summer, h.season_rest),
                            "old_rate": np.nan}, index=rows.index)  # fmt: skip
    X = other_inputs(history, rows, code)[metadata["inputs"]].fillna(pd.Series(metadata["fill_values"]))
    return pd.Series(model.predict_proba(X)[:, 1], index=rows.index)


def past_predictions(code: str) -> pd.Series:
    """F's every-year-test predictions for one department: saved by `evaluate`,
    else recomputed (both departments are needed, the model is shared)."""
    saved = OUT_DIR / "predictions_loyo.parquet"
    if saved.exists():
        return pd.read_parquet(saved).p.xs((code, FROZEN), level=["department", "model"])
    panels = labelled_panels()
    scores, _ = run(panels, all_folds(panels)["loyo"], {FROZEN: LADDER[FROZEN]})
    return scores[code][FROZEN]


def value() -> pd.DataFrame:
    """v4 (F, every-year test) against the FWI references on the same commune-days:
    ranking by year, positive commune-days caught at pooled budgets and at daily
    capacities, with 95% ranges from resampling years (value_check.gains)."""
    from firepredict.transfer import usual_rates
    from firepredict.value_check import gains

    rows = []
    for code, panel in labelled_panels().items():
        year = target_day(panel).year
        history = pd.Series(np.nan, index=panel.index)
        for hidden in np.unique(year):  # usual yearly rate without the hidden year (as v2's reference)
            rows_h = year == hidden
            history[rows_h] = usual_rates(panel, hidden)[rows_h]
        scores = {"FWI alone (department)": panel.fwi, "local FWI alone": panel.local_fwi,
                  "history x FWI": history * panel.fwi, "v4": past_predictions(code).reindex(panel.index)}  # fmt: skip
        references = [k for k in scores if k != "v4"]
        rows.append(gains(panel, scores, model="v4", references=references).assign(department=code))
    table = pd.concat(rows)
    table.to_csv(OUT_DIR / "value.csv")
    return table


def forecast() -> None:
    """Archive today's v4 forecast for every department (never overwritten)."""
    for code, panel in build_panels().items():
        issue = panel.index.get_level_values("date").max()
        rows = panel.xs(issue, level="date", drop_level=False)
        table = pd.DataFrame({"target_day": target_day(rows).strftime("%Y-%m-%d"), "issued_on": issue.strftime("%Y-%m-%d"),
                              "weather_up_to": (issue - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                              "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                              "model": VERSION, "department": code, "insee": rows.index.get_level_values("insee"),
                              "chance": predict(rows, code).round(8).to_numpy()}).sort_values("chance", ascending=False)  # fmt: skip
        path = ARCHIVE / code / f"{table.target_day.iloc[0]}.csv"
        if path.exists():
            print(f"{path.relative_to(ROOT)} already archived, left unchanged")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)
        print(f"wrote {path.relative_to(ROOT)}: expected starts {table.chance.sum():.2f}")


# --- checks ------------------------------------------------------------------


def checks(panels, scores, rates, test: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, groups = [], []
    for code, panel in panels.items():
        for name, p in scores[code].items():
            sub = panel.loc[p.index]
            y = sub.target.to_numpy()
            r = per_year(sub, p)
            rows.append({"test": test, "department": code, "model": name, "log loss": log_loss(y, p.to_numpy(), labels=[0, 1]),
                         "AUC (mean of years)": r.roc_auc.mean(), "top15/day caught": int(r.caught.sum()),
                         "top15/day capture": r.caught.sum() / r.fires.sum(), "positive commune-days": int(y.sum())})  # fmt: skip
            month = target_day(sub).month
            years_all = target_day(sub).year.to_numpy()
            # History level: communes ranked within each fold (test year) by their
            # rate from that fold's training years, never the test year's fires.
            r = rates[code].loc[p.index]
            fold = target_day(sub).year
            per_commune = r.groupby([fold, r.index.get_level_values("insee")]).first()
            ranks = per_commune.groupby(level=0).rank(pct=True, method="average")
            level = ranks.reindex(pd.MultiIndex.from_arrays([fold, r.index.get_level_values("insee")])).to_numpy()
            for season, months in SEASONS.items():
                m = month.isin(months)
                groups.append(group(test, code, name, f"season {season}", y[m], p.to_numpy()[m], years_all[m]))
            for lo, hi, label in [(0, 1 / 3, "quiet third"), (1 / 3, 2 / 3, "middle third"), (2 / 3, 1.01, "busy third")]:
                m = (level >= lo) & (level < hi)
                groups.append(group(test, code, name, f"communes: {label}", y[m], p.to_numpy()[m], years_all[m]))
            years = target_day(sub).year
            for year in np.unique(years):
                m = years == year
                groups.append(group(test, code, name, f"year {year}", y[m], p.to_numpy()[m]))
            for season, months in SEASONS.items():
                m = month.isin(months) & (years <= 2022)
                groups.append(group(test, code, name, f"before 2023, season {season}", y[m], p.to_numpy()[m]))
            for case, (dept, insee, months) in CASES.items():
                if dept == code:
                    m = (sub.index.get_level_values("insee") == insee) & month.isin(months)
                    groups.append(group(test, code, name, case, y[m], p.to_numpy()[m], years_all[m]))
    return pd.DataFrame(rows), pd.DataFrame(groups)


def group(test, code, model, name, y, p, years=None) -> dict:
    o, e = float(y.sum()), float(p.sum())
    if years is not None:  # observed and expected per test year, for the year resampling
        for yr in np.unique(years):
            m = years == yr
            BY_YEAR.append({"test": test, "department": code, "model": model, "group": name, "year": int(yr),
                            "observed": float(y[m].sum()), "expected": float(p[m].sum())})  # fmt: skip
    return {"test": test, "department": code, "model": model, "group": name, "observed": o, "expected": round(e, 1),
            "E/O": e / o if o else np.nan}  # fmt: skip


BY_YEAR: list[dict] = []


def judge(by_year: pd.DataFrame, n_boot: int = 2000) -> pd.DataFrame:
    """Second protocol: rules 3 and 4 with pass / fail / inconclusive, from 95%
    ranges of E/O obtained by resampling test years."""
    rng = np.random.default_rng(0)
    rows = []
    for (test, code, model, name), g in by_year.groupby(["test", "department", "model", "group"]):
        o_total = g.observed.sum()
        draws = rng.integers(0, len(g), size=(n_boot, len(g)))
        o, e = g.observed.to_numpy()[draws].sum(1), g.expected.to_numpy()[draws].sum(1)
        ratio = np.where(o > 0, e / np.maximum(o, 1e-9), np.inf)  # no fire drawn: infinitely too high
        low, high = np.quantile(ratio, [0.025, 0.975], method="inverted_cdf")  # keeps infinities
        row = {"test": test, "department": code, "model": model, "group": name, "observed": o_total,
               "E/O": g.expected.sum() / o_total if o_total else np.nan, "low": low, "high": high}  # fmt: skip
        for rule, (lo, hi, minimum) in {"rule 3": (0.8, 1.25, 30), "rule 4": (0.5, 2.0, 10)}.items():
            if rule == "rule 3" and not name.startswith("season"):
                row[rule] = ""
            elif o_total < minimum:
                row[rule] = "too few events"
            elif lo <= low and high <= hi:
                row[rule] = "pass"
            elif high < lo or low > hi:
                row[rule] = "fail"
            else:
                row[rule] = "inconclusive"
        rows.append(row)
    return pd.DataFrame(rows)


def gates(summary: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """The acceptance rules of the module docstring, for every model vs A and vs B."""
    rows = []
    for name in LADDER:
        row = {"model": name}
        for ref in ["A pooled v2", "B v3 (rate x summer)"]:
            better = []
            for (test, code), s in summary.groupby(["test", "department"]):
                ll = s.set_index("model")["log loss"]
                better.append(ll[name] < ll[ref])
            row[f"1. lower log loss than {ref[0]} (all tests, both departments)"] = all(better)
        cap = summary.set_index(["test", "department", "model"])["top15/day capture"]
        pairs = summary[["test", "department"]].drop_duplicates().itertuples(index=False)
        row["2. top-15 capture within 2 points of A"] = all(cap[(t, d, name)] >= cap[(t, d, "A pooled v2")] - 0.02 for t, d in pairs)
        g = groups[(groups.model == name) & groups.group.str.startswith("season")]
        big = g[g.observed >= 30]
        row["3. E/O in 0.8-1.25 for seasons with >= 30 events"] = bool(big["E/O"].between(0.8, 1.25).all())
        # Rule 4 uses the groups it was written for (seasons, history levels, the two
        # cases); the year-by-year and before-2023 groups are reported, not gated.
        gated = groups[~groups.group.str.startswith(("year ", "before 2023"))]
        g10 = gated[(gated.model == name) & (gated.observed >= 10)]
        row["4. no factor-2 error (groups >= 10 events)"] = bool(((g10["E/O"] < 2) & (g10["E/O"] > 0.5)).all())
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def labelled_panels() -> dict[str, pd.DataFrame]:
    panels = {code: pd.read_parquet(panel_path(code)) for code in DEPARTMENTS}
    return {code: p[p.target.notna()] for code, p in panels.items()}


def all_folds(panels) -> dict[str, list[tuple[list[int], int]]]:
    years = sorted(set().union(*(set(target_day(p).year) for p in panels.values())))
    return {"loyo": [([y for y in years if y != h], h) for h in years],
            "forward": [([y for y in years if y < t], t) for t in FORWARD_YEARS]}  # fmt: skip


def rejudge() -> pd.DataFrame:
    """Second protocol again from the saved predictions, without refitting."""
    BY_YEAR.clear()
    panels = labelled_panels()
    for test, folds in all_folds(panels).items():
        saved = pd.read_parquet(OUT_DIR / f"predictions_{test}.parquet").p
        scores = {c: {m: saved.xs((c, m), level=["department", "model"]) for m in LADDER} for c in DEPARTMENTS}
        rates = {c: [] for c in DEPARTMENTS}
        for train_years, test_year in folds:
            for code, panel in panels.items():
                rows = target_day(panel).year == test_year
                rates[code].append(history_inputs(panel[np.isin(target_day(panel).year, [*train_years, test_year])], train_years)
                                   .log_rate[rows[np.isin(target_day(panel).year, [*train_years, test_year])]])  # fmt: skip
        checks(panels, scores, {c: pd.concat(r) for c, r in rates.items()}, test)
    judged = judge(pd.DataFrame(BY_YEAR))
    judged.to_csv(OUT_DIR / "judged.csv", index=False)
    return judged


def evaluate() -> None:
    BY_YEAR.clear()
    _INNER.clear()
    panels = labelled_panels()
    folds_by_test = all_folds(panels)
    summary, groups = [], []
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for test, folds in folds_by_test.items():
        print(f"{test}:", flush=True)
        scores, rates = run(panels, folds)
        pd.concat({(c, m): s.rename("p") for c, ms in scores.items() for m, s in ms.items()},
                  names=["department", "model"]).to_frame().to_parquet(OUT_DIR / f"predictions_{test}.parquet")  # fmt: skip
        s, g = checks(panels, scores, rates, test)
        summary.append(s)
        groups.append(g)
    summary, groups = pd.concat(summary), pd.concat(groups)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    groups.to_csv(OUT_DIR / "groups.csv", index=False)
    result = gates(summary, groups)
    result.to_csv(OUT_DIR / "gates.csv")
    judged = judge(pd.DataFrame(BY_YEAR))
    judged.to_csv(OUT_DIR / "judged.csv", index=False)
    outcomes = {}
    for rule in ["rule 3", "rule 4"]:
        j = judged[judged[rule] != ""]
        outcomes[rule] = j.groupby("model")[rule].value_counts().unstack(fill_value=0)
    print(pd.concat(outcomes, axis=1).to_string())
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(summary.round(4).to_string(index=False))
    print(groups[groups.group.str.startswith("season") | groups.group.isin(CASES)].round(2).to_string(index=False))
    print(result.T.to_string())


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "evaluate"
    if command == "evaluate":
        evaluate()
    elif command == "freeze":
        m = freeze()
        print(json.dumps({k: m[k] for k in ["version", "frozen_at", "status", "weights_standardised", "forecasts_2026"]}, indent=2, ensure_ascii=False))
    elif command == "forecast":
        forecast()
    elif command == "value":
        pd.set_option("display.width", 250)
        pd.set_option("display.max_colwidth", 40)
        print(value().T.to_string())
    elif command == "rejudge":
        j = rejudge()
        for rule in ["rule 3", "rule 4"]:
            print(j[j[rule] != ""].groupby("model")[rule].value_counts().unstack(fill_value=0).to_string())
    else:
        raise SystemExit(f"unknown command {command}: evaluate")
