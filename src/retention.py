"""Retention planning on top of an already-scored batch.

Everything here is a business rule or campaign assumption layered on the model
output. Nothing refits or re-scores the model or changes the official decision
threshold: the inputs are the columns produced by ``inference.score_prepared``.
Suggested actions are heuristics based on observed features, not treatments
with a measured effect, and no function estimates customers retained or revenue.

Ranking policies (rank 1 = contact first; ties always fall back to file order):

* ``score``:    model churn score (uncalibrated), highest first.
* ``priority``: planning risk band (High, Medium, Low), then the number of
  observed rule triggers, then model churn score, then file order.

The calibrated probability is a monotonic remap of the score, so it never
changes either order and is not used for ranking.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd

import dashboard as db
from inference import CALIBRATED_COL, PREDICTION_COL, PROBABILITY_COL
from utils import ID_COL

ROW_ID_COL = "Row ID"
ID_TYPE_COL = "Identifier Type"
BAND_COL = "Planning Risk Band"
RISK_BANDS = db.RISK_SEGMENTS
ACTION_COL = "Suggested Action"
OTHER_ACTIONS_COL = "Other Applicable Actions"
REASON_COL = "Suggestion Reason"
CONDITION_COL = "Eligibility Condition"
N_TRIGGERS_COL = "Trigger Count"
BASIS_COL = "Suggestion Basis"
SCORE_RANK_COL = "Score Rank"
PRIORITY_RANK_COL = "Priority Rank"
ELIGIBLE_COL = "Eligible"
SELECTED_COL = "Selected"
ORDER_COL = "Selection Order"
COST_COL = "Estimated Cost"

HEURISTIC_BASIS = "Heuristic rule; not a validated treatment"
PROFILE_FEATURES = ("Tenure", "Complain", "SatisfactionScore", "OrderCount", "DaySinceLastOrder")
STRATEGIES = {"priority": "Priority ranking (business rules)", "score": "Highest model score first"}
RANK_COLUMNS = {"priority": PRIORITY_RANK_COL, "score": SCORE_RANK_COL}

COMPLAINT, ONBOARDING, SERVICE, REENGAGE = (
    "Complaint resolution follow-up", "Onboarding assistance", "Service-quality follow-up",
    "Re-engagement message")
MANUAL, GENERAL = "Manual review", "General check-in"
ACTIONS = (COMPLAINT, ONBOARDING, SERVICE, REENGAGE, MANUAL, GENERAL)

EXPORT_COLUMNS = [
    ROW_ID_COL, ID_COL, ID_TYPE_COL, PRIORITY_RANK_COL, SCORE_RANK_COL, PROBABILITY_COL,
    CALIBRATED_COL, PREDICTION_COL, BAND_COL, *PROFILE_FEATURES, ACTION_COL, OTHER_ACTIONS_COL,
    REASON_COL, CONDITION_COL, N_TRIGGERS_COL, BASIS_COL, ELIGIBLE_COL, SELECTED_COL, ORDER_COL,
    COST_COL,
]


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
def validate_amount(value, name: str) -> float:
    """A finite, non-negative number (budgets, costs, rule thresholds)."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a number, got {value!r}.")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite number of at least 0, got {value}.")
    return value


def _validate_max_customers(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or float(value) != int(value) or int(value) < 0:
        raise ValueError(f"Maximum customers must be a whole number of at least 0, got {value!r}.")
    return int(value)


def _validate_strategy(strategy: str) -> str:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown selection strategy {strategy!r}; use one of {', '.join(STRATEGIES)}.")
    return strategy


# --------------------------------------------------------------------------- #
# Identifiers and risk bands
# --------------------------------------------------------------------------- #
def identifier_types(results: pd.DataFrame) -> pd.Series:
    """How each row is identified: a unique CustomerID, or only its position in this upload."""
    kind = pd.Series("Row position in this upload only", index=results.index, name=ID_TYPE_COL)
    if ID_COL in results.columns:
        ids = results[ID_COL]
        duplicated = ids.notna() & ids.duplicated(keep=False)
        kind[ids.notna() & ~duplicated] = ID_COL
        kind[duplicated] = f"{ID_COL} (duplicated in this upload)"
    return kind


def identifier_note(results: pd.DataFrame) -> str:
    if ID_COL not in results.columns:
        return (f"No `{ID_COL}` column: customers are identified by `{ROW_ID_COL}`, their position in this "
                "upload. Row IDs are not stable across files and cannot be matched to other systems.")
    kinds = identifier_types(results)
    unusable = int((kinds != ID_COL).sum())
    if not unusable:
        return f"Customers are identified by `{ID_COL}`; `{ROW_ID_COL}` is their position in this upload."
    return (f"{unusable:,} row(s) have a missing or duplicated `{ID_COL}` and can only be identified by "
            f"`{ROW_ID_COL}`, their position in this upload (not stable across files).")


def assign_bands(probability_pct, low, high) -> pd.Series:
    """Planning bands on the model score. They never change the official prediction."""
    return db.assign_risk_segments(probability_pct, low, high).rename(BAND_COL)


# --------------------------------------------------------------------------- #
# Rule-based suggestions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RuleConfig:
    max_short_tenure: float = 2        # Tenure (months) at or below this value
    max_low_satisfaction: float = 2    # SatisfactionScore (1-5) at or below this value
    min_inactive_days: float = 10      # DaySinceLastOrder at or above this value

    def validated(self) -> "RuleConfig":
        validate_amount(self.max_short_tenure, "Short-tenure limit")
        validate_amount(self.max_low_satisfaction, "Low-satisfaction limit")
        validate_amount(self.min_inactive_days, "Inactivity limit")
        return self


def _rules(cfg: RuleConfig) -> list[tuple]:
    """(feature, action, condition, test, reason) in priority order."""
    return [
        ("Complain", COMPLAINT, "Complain = 1", lambda v: v == 1,
         lambda v: "Complaint recorded (Complain = 1)"),
        ("Tenure", ONBOARDING, f"Tenure ≤ {cfg.max_short_tenure:g} months",
         lambda v: v <= cfg.max_short_tenure,
         lambda v: f"Tenure of {v:g} month(s) (≤ {cfg.max_short_tenure:g})"),
        ("SatisfactionScore", SERVICE, f"SatisfactionScore ≤ {cfg.max_low_satisfaction:g}",
         lambda v: v <= cfg.max_low_satisfaction,
         lambda v: f"Satisfaction score {v:g} (≤ {cfg.max_low_satisfaction:g})"),
        ("DaySinceLastOrder", REENGAGE, f"DaySinceLastOrder ≥ {cfg.min_inactive_days:g}",
         lambda v: v >= cfg.min_inactive_days,
         lambda v: f"{v:g} days since last order (≥ {cfg.min_inactive_days:g})"),
    ]


def rules_table(cfg: RuleConfig = RuleConfig()) -> pd.DataFrame:
    rows = [{"Order": i, "Suggested action": action, "Applies when": condition, "Feature used": feature}
            for i, (feature, action, condition, _, _) in enumerate(_rules(cfg.validated()), 1)]
    n = len(rows)
    rows.append({"Order": n + 1, "Suggested action": MANUAL, "Feature used": "—",
                 "Applies when": "No rule applies and at least one rule feature is missing"})
    rows.append({"Order": n + 2, "Suggested action": GENERAL, "Feature used": "—",
                 "Applies when": "No rule applies and every rule feature is observed"})
    return pd.DataFrame(rows)


def suggest_actions(df: pd.DataFrame, cfg: RuleConfig = RuleConfig()) -> pd.DataFrame:
    """One suggestion per row; the first applicable rule wins, the others are listed.

    Rules only use observed values: a missing value never triggers a rule, and a
    rule whose feature column is absent is skipped.
    """
    rules = _rules(cfg.validated())
    per_rule = []
    for feature, action, condition, test, reason in rules:
        values = (pd.to_numeric(df[feature], errors="coerce") if feature in df.columns
                  else pd.Series(np.nan, index=df.index))
        fired = (values.notna() & test(values)).to_numpy()
        texts = [reason(v) if hit else None for v, hit in zip(values.to_numpy(), fired)]
        per_rule.append((feature, action, condition, fired, texts, values.isna().to_numpy()))

    rows = []
    for i in range(len(df)):
        hits = [(action, condition, texts[i]) for _, action, condition, fired, texts, _ in per_rule if fired[i]]
        missing = [feature for feature, *_, isna in per_rule if isna[i]]
        if hits:
            rows.append((hits[0][0], "; ".join(h[0] for h in hits[1:]), "; ".join(h[2] for h in hits),
                         hits[0][1], len(hits)))
        elif missing:
            rows.append((MANUAL, "", "No rule applies to the observed values and "
                         f"{', '.join(missing)} is missing; review before contacting.",
                         "No rule applies and at least one rule feature is missing", 0))
        else:
            rows.append((GENERAL, "", "No rule condition was observed; prioritised by model score only.",
                         "No rule applies and every rule feature is observed", 0))
    out = pd.DataFrame(rows, index=df.index,
                       columns=[ACTION_COL, OTHER_ACTIONS_COL, REASON_COL, CONDITION_COL, N_TRIGGERS_COL])
    out[N_TRIGGERS_COL] = out[N_TRIGGERS_COL].astype(int)
    out[BASIS_COL] = HEURISTIC_BASIS
    return out


# --------------------------------------------------------------------------- #
# Action plan and ranking
# --------------------------------------------------------------------------- #
def rank_customers(plan: pd.DataFrame, strategy: str = "priority") -> pd.Series:
    """Deterministic 1-based ranks; ties are broken by original row order."""
    _validate_strategy(strategy)
    score = pd.to_numeric(plan[PROBABILITY_COL], errors="coerce").fillna(-1).to_numpy(dtype=float)
    keys = [np.arange(len(plan)), -score]
    if strategy == "priority":
        keys += [-plan[N_TRIGGERS_COL].to_numpy(dtype=int),
                 -plan[BAND_COL].cat.codes.to_numpy(dtype=int)]
    order = np.lexsort(keys)
    ranks = np.empty(len(plan), dtype=int)
    ranks[order] = np.arange(1, len(plan) + 1)
    return pd.Series(ranks, index=plan.index, name=RANK_COLUMNS[strategy])


def build_action_plan(results: pd.DataFrame, low, high, cfg: RuleConfig = RuleConfig()) -> pd.DataFrame:
    """Per-customer planning table in the original row order of ``results``."""
    if results is None or len(results) == 0:
        raise ValueError("There are no scored customers to plan for.")
    if PROBABILITY_COL not in results.columns:
        raise ValueError(f"The scored batch has no '{PROBABILITY_COL}' column.")
    plan = pd.DataFrame({ROW_ID_COL: np.arange(1, len(results) + 1)}, index=results.index)
    if ID_COL in results.columns:
        plan[ID_COL] = results[ID_COL]
    plan[ID_TYPE_COL] = identifier_types(results)
    for column in (PROBABILITY_COL, CALIBRATED_COL, PREDICTION_COL):
        if column in results.columns:
            plan[column] = results[column]
    plan[BAND_COL] = assign_bands(results[PROBABILITY_COL], low, high)
    for column in PROFILE_FEATURES:
        if column in results.columns:
            plan[column] = results[column]
    plan = plan.join(suggest_actions(results, cfg))
    for strategy in STRATEGIES:
        plan[RANK_COLUMNS[strategy]] = rank_customers(plan, strategy)
    return plan


def filter_plan(plan: pd.DataFrame, bands=None, actions=None, predicted_only: bool = False,
                sort: str = "priority") -> pd.DataFrame:
    """Filter the plan; ``sort`` is 'priority', 'score' or 'original'."""
    mask = pd.Series(True, index=plan.index)
    if bands is not None:
        mask &= plan[BAND_COL].isin(list(bands))
    if actions is not None:
        mask &= plan[ACTION_COL].isin(list(actions))
    if predicted_only and PREDICTION_COL in plan.columns:
        mask &= plan[PREDICTION_COL] == 1
    view = plan[mask]
    if sort in RANK_COLUMNS:
        view = view.sort_values(RANK_COLUMNS[sort], kind="mergesort")
    elif sort != "original":
        raise ValueError(f"Unknown sort order {sort!r}.")
    return view


def action_counts(plan: pd.DataFrame) -> pd.DataFrame:
    counts = plan[ACTION_COL].value_counts().reindex(list(ACTIONS), fill_value=0)
    return pd.DataFrame({"Action": counts.index, "Customers": counts.to_numpy(dtype=int)})


# --------------------------------------------------------------------------- #
# Eligibility, costs and budget-constrained selection
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Scenario:
    """Campaign assumptions. ``cost`` is one amount per contact or a {action: amount} mapping."""
    name: str = "Scenario"
    budget: float = 0.0
    cost: Union[float, dict] = 0.0
    max_customers: Optional[int] = None
    min_score: float = 0.0
    min_band: str = "Low"
    strategy: str = "priority"
    predicted_only: bool = False


def eligible_mask(plan: pd.DataFrame, min_score: float = 0.0, min_band: str = "Low",
                  predicted_only: bool = False) -> pd.Series:
    """Score ≥ ``min_score`` (%) and band at or above ``min_band``; optionally predicted churners only."""
    min_score = validate_amount(min_score, "Minimum score")
    if min_score > 100:
        raise ValueError(f"Minimum score must be between 0 and 100, got {min_score}.")
    if min_band not in RISK_BANDS:
        raise ValueError(f"Unknown risk band {min_band!r}; use one of {', '.join(RISK_BANDS)}.")
    mask = (plan[PROBABILITY_COL] >= min_score) & (plan[BAND_COL].cat.codes >= RISK_BANDS.index(min_band))
    if predicted_only:
        if PREDICTION_COL not in plan.columns:
            raise ValueError("Official predictions are unavailable, so 'predicted churners only' "
                             "cannot be applied.")
        mask &= plan[PREDICTION_COL] == 1
    return mask.rename(ELIGIBLE_COL)


def customer_costs(plan: pd.DataFrame, cost) -> pd.Series:
    if isinstance(cost, dict):
        prices = {action: validate_amount(value, f"Cost for '{action}'") for action, value in cost.items()}
        unpriced = sorted(set(plan[ACTION_COL]) - set(prices))
        if unpriced:
            raise ValueError(f"No cost was set for: {', '.join(unpriced)}.")
        return plan[ACTION_COL].map(prices).astype(float)
    return pd.Series(validate_amount(cost, "Cost per customer"), index=plan.index, dtype=float)


def select_customers(plan: pd.DataFrame, scenario: Scenario) -> pd.DataFrame:
    """Mark eligible and selected customers, keeping the original row order.

    Eligible customers are taken strictly in rank order while the running cost
    stays within budget and the contact limit is not reached. Selection stops at
    the first customer that does not fit, so a lower-ranked customer is never
    chosen in place of a higher-ranked one.
    """
    budget = validate_amount(scenario.budget, "Budget")
    limit = _validate_max_customers(scenario.max_customers)
    rank_col = RANK_COLUMNS[_validate_strategy(scenario.strategy)]
    costs = customer_costs(plan, scenario.cost)
    eligible = eligible_mask(plan, scenario.min_score, scenario.min_band, scenario.predicted_only)

    order = plan.index[eligible.to_numpy()][np.argsort(plan.loc[eligible, rank_col].to_numpy(), kind="stable")]
    fits = costs.loc[order].cumsum().to_numpy() <= budget + 1e-9
    if limit is not None:
        fits &= np.arange(len(order)) < limit
    chosen = order[fits]

    out = plan.copy()
    out[ELIGIBLE_COL] = eligible
    out[SELECTED_COL] = out.index.isin(chosen)
    out[ORDER_COL] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out.loc[chosen, ORDER_COL] = np.arange(1, len(chosen) + 1)
    out[COST_COL] = costs.where(out[SELECTED_COL], 0.0)
    return out


def _profile(frame: pd.DataFrame) -> dict:
    row = {"Customers": len(frame),
           "Mean model score (%)": frame[PROBABILITY_COL].mean() if len(frame) else np.nan}
    if CALIBRATED_COL in frame.columns:
        row["Mean calibrated probability (%)"] = frame[CALIBRATED_COL].mean() if len(frame) else np.nan
    if PREDICTION_COL in frame.columns:
        row["Official predicted churners"] = int((frame[PREDICTION_COL] == 1).sum())
    if "Complain" in frame.columns:
        row["Complaint share"] = (frame["Complain"] == 1).mean() if len(frame) else np.nan
    for column in ("Tenure", "SatisfactionScore", "OrderCount", "DaySinceLastOrder"):
        if column in frame.columns:
            row[f"Median {column}"] = pd.to_numeric(frame[column], errors="coerce").median()
    return row


def summarize_selection(selection: pd.DataFrame, scenario: Scenario) -> dict:
    selected = selection[selection[SELECTED_COL]]
    cost = float(selected[COST_COL].sum())
    budget = validate_amount(scenario.budget, "Budget")
    summary = {"Eligible": int(selection[ELIGIBLE_COL].sum()), "Selected": len(selected),
               "Estimated cost": round(cost, 2), "Unspent budget": round(budget - cost, 2)}
    bands = selected[BAND_COL].value_counts()
    summary.update({f"{band} band": int(bands.get(band, 0)) for band in RISK_BANDS})
    summary.update({k: v for k, v in _profile(selected).items() if k != "Customers"})
    return summary


def plan_scenario(plan: pd.DataFrame, scenario: Scenario) -> tuple[pd.DataFrame, dict]:
    selection = select_customers(plan, scenario)
    return selection, summarize_selection(selection, scenario)


def selection_profile(selection: pd.DataFrame) -> pd.DataFrame:
    """Observed characteristics of selected vs eligible-but-not-selected vs all scored customers."""
    groups = {"Selected": selection[selection[SELECTED_COL]],
              "Eligible, not selected": selection[selection[ELIGIBLE_COL] & ~selection[SELECTED_COL]],
              "All scored": selection}
    return pd.DataFrame({name: _profile(frame) for name, frame in groups.items()}).rename_axis("Measure")


# --------------------------------------------------------------------------- #
# What-if scenarios
# --------------------------------------------------------------------------- #
SCENARIO_FIELDS = {"Scenario": "name", "Budget": "budget", "Cost per customer": "cost",
                   "Max customers (0 = no limit)": "max_customers", "Min model score (%)": "min_score",
                   "Lowest band": "min_band", "Strategy": "strategy"}


def preset_scenarios(budget: float, cost: float, strategy: str = "priority") -> list[Scenario]:
    """Conservative / moderate / broad outreach that differ only in which bands are eligible."""
    return [Scenario("Conservative", budget, cost, None, 0.0, "High", strategy),
            Scenario("Moderate", budget, cost, None, 0.0, "Medium", strategy),
            Scenario("Broad", budget, cost, None, 0.0, "Low", strategy)]


def scenario_table(scenarios: list[Scenario]) -> pd.DataFrame:
    rows = [{label: getattr(s, attr) for label, attr in SCENARIO_FIELDS.items()} for s in scenarios]
    for row in rows:
        row["Max customers (0 = no limit)"] = row["Max customers (0 = no limit)"] or 0
    return pd.DataFrame(rows, columns=list(SCENARIO_FIELDS))


def scenarios_from_table(table: pd.DataFrame) -> tuple[list[Scenario], list[str]]:
    """Parse an edited scenario table; invalid rows are reported and skipped."""
    scenarios, errors = [], []
    for i, row in enumerate(table.to_dict("records"), 1):
        name = str(row.get("Scenario") or f"Scenario {i}").strip() or f"Scenario {i}"
        try:
            limit = _cell(row, "Max customers (0 = no limit)", 0)
            min_score = validate_amount(_cell(row, "Min model score (%)", 0.0), "Minimum score")
            band = _cell(row, "Lowest band", "Low")
            if min_score > 100 or band not in RISK_BANDS:
                raise ValueError("Minimum score must be 0–100 and the band one of Low, Medium, High.")
            scenarios.append(Scenario(
                name, validate_amount(_cell(row, "Budget"), "Budget"),
                validate_amount(_cell(row, "Cost per customer"), "Cost per customer"),
                _validate_max_customers(limit) or None, min_score, band,
                _validate_strategy(_cell(row, "Strategy", "priority"))))
        except (TypeError, ValueError) as exc:
            errors.append(f"Row {i} ({name}) was skipped: {exc}")
    return scenarios, errors


def _cell(row: dict, column: str, default=None):
    value = row.get(column)
    if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)) or value == "":
        if default is None:
            raise ValueError(f"'{column}' is missing.")
        return default
    return value


def compare_scenarios(plan: pd.DataFrame, scenarios: list[Scenario]) -> pd.DataFrame:
    """One row per scenario: its assumptions followed by the resulting selection summary.

    ``Cost per customer`` is empty for per-action costs; ``Max customers`` is empty when unlimited.
    """
    rows = []
    for s in scenarios:
        _, summary = plan_scenario(plan, s)
        rows.append({"Scenario": s.name, "Strategy": STRATEGIES[s.strategy], "Budget": s.budget,
                     "Cost per customer": np.nan if isinstance(s.cost, dict) else float(s.cost),
                     "Max customers": s.max_customers, "Min model score (%)": s.min_score,
                     "Lowest band": s.min_band, **summary})
    table = pd.DataFrame(rows)
    table["Max customers"] = table["Max customers"].astype("Int64")
    return table


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def export_action_plan(plan: pd.DataFrame, selected_only: bool = False) -> pd.DataFrame:
    """Fixed-schema export. Columns that do not apply (e.g. no calibrator) are left empty.

    The full export keeps the original row order; the selected-only export is in
    selection order.
    """
    out = plan.reindex(columns=EXPORT_COLUMNS)
    if selected_only:
        if SELECTED_COL not in plan.columns:
            raise ValueError("No customers have been selected yet.")
        out = out[plan[SELECTED_COL].fillna(False).astype(bool)].sort_values(ORDER_COL, kind="mergesort")
    return out.reset_index(drop=True)
