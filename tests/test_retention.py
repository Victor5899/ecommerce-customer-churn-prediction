"""Retention planning rules, selection and the Retention Intelligence page (no model training)."""
import numpy as np
import pandas as pd
import pytest

import retention as rt
from inference import CALIBRATED_COL, PREDICTION_COL, PROBABILITY_COL
from utils import ID_COL


def scored(calibrated: bool = True, predictions: bool = True) -> pd.DataFrame:
    """Six scored customers; rows 2 and 3 tie on score."""
    df = pd.DataFrame({
        ID_COL: [501, 502, 503, 504, 505, 506],
        "Tenure": [1.0, 20.0, 20.0, np.nan, 30.0, 5.0],
        "Complain": [1, 0, 0, 0, 0, 1],
        "SatisfactionScore": [4, 1, 5, 3, 4, 2],
        "OrderCount": [2.0, 5.0, 1.0, 3.0, 8.0, 2.0],
        "DaySinceLastOrder": [3.0, 2.0, 2.0, 1.0, 12.0, 0.0],
        PROBABILITY_COL: [80.0, 70.0, 70.0, 40.0, 10.0, 65.0],
    })
    if predictions:
        df[PREDICTION_COL] = (df[PROBABILITY_COL] >= 25).astype(int)
    if calibrated:
        df[CALIBRATED_COL] = df[PROBABILITY_COL] * 0.5
    return df


@pytest.fixture
def plan():
    return rt.build_action_plan(scored(), 30, 60)


# --------------------------------------------------------------------------- #
# Risk bands and their separation from official predictions
# --------------------------------------------------------------------------- #
def test_band_boundaries_are_inclusive_at_the_lower_edge():
    bands = rt.assign_bands([29.99, 30, 59.99, 60, 0, 100], 30, 60)
    assert list(bands) == ["Low", "Medium", "Medium", "High", "Low", "High"]
    assert bands.name == rt.BAND_COL


@pytest.mark.parametrize("low,high", [(60, 30), (30, 30), (0, 50), (50, 100)])
def test_invalid_band_boundaries_are_rejected(low, high):
    with pytest.raises(ValueError):
        rt.build_action_plan(scored(), low, high)


def test_planning_bands_never_change_official_predictions():
    results = scored()
    results.loc[4, PREDICTION_COL] = 1           # flagged by the official threshold, but low score
    original = results.copy()
    for low, high in [(5, 10), (30, 60), (90, 95)]:
        plan = rt.build_action_plan(results, low, high)
        pd.testing.assert_series_equal(plan[PREDICTION_COL], original[PREDICTION_COL])
    assert rt.build_action_plan(results, 30, 60).loc[4, rt.BAND_COL] == "Low"
    pd.testing.assert_frame_equal(results, original)


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #
def test_score_ranking_breaks_ties_by_file_order(plan):
    assert list(plan[rt.SCORE_RANK_COL]) == [1, 2, 3, 5, 6, 4]


def test_priority_ranking_uses_band_then_triggers_then_score(plan):
    # Row 6 (65%, High, complaint + low satisfaction = 2 triggers) outranks rows 2 and 3 (70%, High).
    # Row 2 (1 trigger) outranks row 3 (0 triggers) despite the equal score.
    assert list(plan.sort_values(rt.PRIORITY_RANK_COL)[rt.ROW_ID_COL]) == [1, 6, 2, 3, 4, 5]


def test_ranking_is_deterministic_and_a_permutation(plan):
    again = rt.build_action_plan(scored(), 30, 60)
    for col in (rt.SCORE_RANK_COL, rt.PRIORITY_RANK_COL):
        assert sorted(plan[col]) == list(range(1, len(plan) + 1))
        pd.testing.assert_series_equal(plan[col], again[col])
    with pytest.raises(ValueError):
        rt.rank_customers(plan, "random")


def test_filter_plan_sorts_and_filters(plan):
    view = rt.filter_plan(plan, bands=["High"], actions=None, sort="score")
    assert list(view[rt.ROW_ID_COL]) == [1, 2, 3, 6]
    assert list(rt.filter_plan(plan, sort="original").index) == list(plan.index)
    assert rt.filter_plan(plan, actions=[rt.COMPLAINT])[rt.ROW_ID_COL].tolist() == [1, 6]


# --------------------------------------------------------------------------- #
# Suggestion rules
# --------------------------------------------------------------------------- #
def test_first_rule_wins_and_other_actions_are_listed(plan):
    first = plan.loc[0]
    assert first[rt.ACTION_COL] == rt.COMPLAINT
    assert first[rt.OTHER_ACTIONS_COL] == rt.ONBOARDING
    assert "Complaint recorded" in first[rt.REASON_COL] and "Tenure of 1 month" in first[rt.REASON_COL]
    assert first[rt.CONDITION_COL] == "Complain = 1" and first[rt.N_TRIGGERS_COL] == 2
    assert plan.loc[1, rt.ACTION_COL] == rt.SERVICE
    assert plan.loc[4, rt.ACTION_COL] == rt.REENGAGE
    assert (plan[rt.BASIS_COL] == rt.HEURISTIC_BASIS).all()


def test_missing_values_lead_to_manual_review_not_a_rule(plan):
    row = plan.loc[3]                             # Tenure missing, nothing else applies
    assert row[rt.ACTION_COL] == rt.MANUAL and "Tenure is missing" in row[rt.REASON_COL]
    assert row[rt.N_TRIGGERS_COL] == 0
    assert plan.loc[2, rt.ACTION_COL] == rt.GENERAL   # every rule feature observed, none applies


def test_absent_feature_columns_skip_their_rule():
    plan = rt.build_action_plan(scored().drop(columns=["Complain"]), 30, 60)
    assert rt.COMPLAINT not in set(plan[rt.ACTION_COL])
    assert plan.loc[0, rt.ACTION_COL] == rt.ONBOARDING
    assert plan.loc[2, rt.ACTION_COL] == rt.MANUAL and "Complain" in plan.loc[2, rt.REASON_COL]
    assert "Complain" not in plan.columns


def test_rule_thresholds_are_configurable_and_validated():
    plan = rt.build_action_plan(scored(), 30, 60, rt.RuleConfig(max_short_tenure=0, min_inactive_days=1))
    assert plan.loc[0, rt.OTHER_ACTIONS_COL] == rt.REENGAGE   # tenure 1 no longer counts as short
    assert plan.loc[2, rt.ACTION_COL] == rt.REENGAGE
    with pytest.raises(ValueError):
        rt.suggest_actions(scored(), rt.RuleConfig(max_short_tenure=-1))
    assert list(rt.rules_table()["Suggested action"]) == list(rt.ACTIONS)


# --------------------------------------------------------------------------- #
# Eligibility, budgets and selection
# --------------------------------------------------------------------------- #
def test_eligibility_by_score_band_and_prediction(plan):
    assert rt.eligible_mask(plan, min_band="High").tolist() == [True, True, True, False, False, True]
    assert rt.eligible_mask(plan, min_score=66).sum() == 3
    assert rt.eligible_mask(plan, predicted_only=True).sum() == 5
    with pytest.raises(ValueError):
        rt.eligible_mask(plan, min_band="Extreme")
    with pytest.raises(ValueError):
        rt.eligible_mask(plan, min_score=101)
    no_pred = rt.build_action_plan(scored(predictions=False), 30, 60)
    with pytest.raises(ValueError, match="predictions are unavailable"):
        rt.eligible_mask(no_pred, predicted_only=True)


def test_budget_limits_selection_and_reports_unspent_budget(plan):
    selection, summary = rt.plan_scenario(plan, rt.Scenario(budget=25, cost=10, strategy="score"))
    assert selection[rt.SELECTED_COL].tolist() == [True, True, False, False, False, False]
    assert summary["Eligible"] == 6 and summary["Selected"] == 2
    assert summary["Estimated cost"] == 20 and summary["Unspent budget"] == 5
    assert summary["High band"] == 2 and summary["Low band"] == 0
    assert selection[rt.COST_COL].sum() == 20


def test_zero_budget_zero_cost_and_contact_limit(plan):
    _, summary = rt.plan_scenario(plan, rt.Scenario(budget=0, cost=10))
    assert summary["Selected"] == 0 and summary["Unspent budget"] == 0
    _, summary = rt.plan_scenario(plan, rt.Scenario(budget=0, cost=0))
    assert summary["Selected"] == 6
    selection, summary = rt.plan_scenario(plan, rt.Scenario(budget=1000, cost=1, max_customers=3))
    assert summary["Selected"] == 3
    assert selection.loc[selection[rt.SELECTED_COL], rt.PRIORITY_RANK_COL].tolist() == [1, 3, 2]


@pytest.mark.parametrize("cost", [-1, float("nan"), float("inf"), True, "10", None])
def test_invalid_costs_are_rejected(plan, cost):
    with pytest.raises(ValueError):
        rt.select_customers(plan, rt.Scenario(budget=100, cost=cost))


@pytest.mark.parametrize("field,value", [("budget", -5), ("budget", float("nan")),
                                         ("max_customers", -1), ("max_customers", 1.5),
                                         ("strategy", "cheapest")])
def test_invalid_scenario_settings_are_rejected(plan, field, value):
    kwargs = {"budget": 100, "cost": 1, field: value}
    with pytest.raises(ValueError):
        rt.select_customers(plan, rt.Scenario(**kwargs))


def test_per_action_costs_and_strict_rank_order(plan):
    costs = {a: 5.0 for a in rt.ACTIONS}
    costs[rt.SERVICE] = 50.0
    # Score order: rows 1 (complaint, 5), 2 (service, 50), 3 (general, 5). Budget 20 stops at row 2.
    selection = rt.select_customers(plan, rt.Scenario(budget=20, cost=costs, strategy="score"))
    assert selection[rt.SELECTED_COL].tolist() == [True, False, False, False, False, False]
    with pytest.raises(ValueError, match="No cost was set"):
        rt.select_customers(plan, rt.Scenario(budget=20, cost={rt.COMPLAINT: 1.0}))


def test_selection_profile_describes_groups(plan):
    selection = rt.select_customers(plan, rt.Scenario(budget=20, cost=10, strategy="score"))
    profile = rt.selection_profile(selection)
    assert profile.loc["Customers"].tolist() == [2, 4, 6]
    assert profile.loc["Complaint share", "Selected"] == 0.5
    assert profile.loc["Mean model score (%)", "Selected"] == 75


# --------------------------------------------------------------------------- #
# What-if scenarios
# --------------------------------------------------------------------------- #
def test_preset_scenarios_differ_only_in_eligibility(plan):
    comparison = rt.compare_scenarios(plan, rt.preset_scenarios(1000, 10))
    assert comparison["Scenario"].tolist() == ["Conservative", "Moderate", "Broad"]
    assert comparison["Eligible"].tolist() == [4, 5, 6]
    assert comparison["Selected"].tolist() == [4, 5, 6]
    assert comparison["Estimated cost"].tolist() == [40, 50, 60]
    assert comparison["Max customers"].isna().all()


def test_scenarios_do_not_mutate_the_plan(plan):
    before = plan.copy()
    rt.compare_scenarios(plan, rt.preset_scenarios(30, 10) + [rt.Scenario("x", 0, 0, 2)])
    pd.testing.assert_frame_equal(plan, before)


def test_scenario_table_round_trip_and_invalid_rows(plan):
    table = rt.scenario_table(rt.preset_scenarios(100, 10))
    scenarios, errors = rt.scenarios_from_table(table)
    assert not errors and [s.min_band for s in scenarios] == ["High", "Medium", "Low"]
    assert all(s.max_customers is None for s in scenarios)
    bad = pd.concat([table, pd.DataFrame([
        {"Scenario": "Neg", "Budget": -1, "Cost per customer": 1},
        {"Scenario": "Band", "Budget": 1, "Cost per customer": 1, "Lowest band": "Top"},
        {"Scenario": "Empty", "Budget": None, "Cost per customer": 1},
    ])], ignore_index=True)
    scenarios, errors = rt.scenarios_from_table(bad)
    assert len(scenarios) == 3 and len(errors) == 3
    assert all("was skipped" in e for e in errors)


# --------------------------------------------------------------------------- #
# Identifiers, row order and export
# --------------------------------------------------------------------------- #
def test_identifiers_and_row_order_are_preserved(plan):
    results = scored()
    assert plan.index.equals(results.index)
    assert plan[ID_COL].tolist() == results[ID_COL].tolist()
    assert plan[rt.ROW_ID_COL].tolist() == [1, 2, 3, 4, 5, 6]
    assert (plan[rt.ID_TYPE_COL] == ID_COL).all()


def test_missing_and_duplicated_ids_fall_back_to_row_position():
    results = scored()
    results.loc[1, ID_COL] = np.nan
    results.loc[3, ID_COL] = results.loc[2, ID_COL]
    kinds = rt.identifier_types(results)
    assert kinds[0] == ID_COL and kinds[1].startswith("Row position")
    assert kinds[2] == kinds[3] == f"{ID_COL} (duplicated in this upload)"
    assert "3 row(s)" in rt.identifier_note(results)
    no_ids = rt.build_action_plan(scored().drop(columns=[ID_COL]), 30, 60)
    assert ID_COL not in no_ids.columns and no_ids[rt.ID_TYPE_COL].str.startswith("Row position").all()
    assert "not stable across files" in rt.identifier_note(scored().drop(columns=[ID_COL]))


@pytest.mark.parametrize("calibrated", [True, False])
def test_export_schema_is_fixed(calibrated):
    plan = rt.build_action_plan(scored(calibrated=calibrated), 30, 60)
    full = rt.export_action_plan(plan)
    assert list(full.columns) == rt.EXPORT_COLUMNS
    assert full[rt.ROW_ID_COL].tolist() == [1, 2, 3, 4, 5, 6]
    assert full[CALIBRATED_COL].notna().all() == calibrated
    assert full[rt.SELECTED_COL].isna().all()

    selection = rt.select_customers(plan, rt.Scenario(budget=30, cost=10))
    chosen = rt.export_action_plan(selection, selected_only=True)
    assert list(chosen.columns) == rt.EXPORT_COLUMNS
    assert chosen[rt.ORDER_COL].tolist() == [1, 2, 3]
    assert chosen[rt.ROW_ID_COL].tolist() == [1, 6, 2]
    assert rt.export_action_plan(selection)[rt.ROW_ID_COL].tolist() == [1, 2, 3, 4, 5, 6]
    with pytest.raises(ValueError):
        rt.export_action_plan(plan, selected_only=True)


def test_empty_or_unscored_batches_are_rejected():
    with pytest.raises(ValueError):
        rt.build_action_plan(scored().iloc[0:0], 30, 60)
    with pytest.raises(ValueError):
        rt.build_action_plan(scored().drop(columns=[PROBABILITY_COL]), 30, 60)


# --------------------------------------------------------------------------- #
# Dashboard page
# --------------------------------------------------------------------------- #
def _render_retention_page(threshold):
    from pathlib import Path

    import dashboard_pages as pages

    ctx = pages.DashboardContext(
        model=None, model_sha="", model_name="m", schema=None, metadata=None, threshold=threshold,
        metadata_error=None, reference=None, reference_error="not loaded", output_dir=Path("."),
        model_path=Path("model.joblib"), train_command="train", excel_sheets=None, parse_upload=None)
    pages.retention_intelligence(ctx)


@pytest.mark.parametrize("calibrated,predictions", [(False, True), (True, True), (False, False)])
def test_retention_page_renders_without_calibration_or_predictions(calibrated, predictions):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_render_retention_page, args=(0.25 if predictions else None,),
                               default_timeout=60)
    at.session_state["batch"] = {"filename": "batch.csv",
                                 "results": scored(calibrated=calibrated, predictions=predictions)}
    at.run()
    assert not at.exception and not at.error
    labels = {m.label: m.value for m in at.metric}
    assert labels["Eligible customers"] == "5" and labels["Unspent budget"] == "950.00"
    infos = " ".join(i.value for i in at.info)
    assert ("Calibrated probabilities are unavailable" in infos) == (not calibrated)
    warnings = " ".join(w.value for w in at.warning)
    assert ("Official predictions are unavailable" in warnings) == (not predictions)


def test_retention_page_without_a_scored_batch():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_render_retention_page, args=(0.25,)).run()
    assert not at.exception and not at.error
    assert "Score a file on **Churn Prediction** first" in " ".join(i.value for i in at.info)
