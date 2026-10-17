"""Dashboard calculations: actual vs predicted churn, risk segments, filters, artifacts."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import charts
import dashboard as db
from inference import PREDICTION_COL, PROBABILITY_COL, score_prepared, prepare_features
from utils import ID_COL, TARGET_COL

ROOT = Path(__file__).resolve().parents[1]


def test_actual_churn_uses_labels_only():
    df = pd.DataFrame({TARGET_COL: [1, 0, 0, 1, 0], PREDICTION_COL: [1, 1, 1, 1, 1]})
    summary = db.actual_churn(df)
    assert (summary.n_customers, summary.n_churned) == (5, 2)
    assert summary.rate == pytest.approx(0.4)


@pytest.mark.parametrize("labels", [[1, None, 0], [1, 2, 0], ["yes", "no", "no"]])
def test_incomplete_or_invalid_labels_are_not_treated_as_actual_churn(labels):
    df = pd.DataFrame({TARGET_COL: labels})
    assert db.labelled_target(df) is None
    with pytest.raises(ValueError):
        db.actual_churn(df)


def test_batch_summary_separates_predictions_from_actual_labels():
    results = pd.DataFrame({PREDICTION_COL: [1, 1, 0, 0], PROBABILITY_COL: [90.0, 30.0, 10.0, 20.0]})
    summary = db.batch_summary(results)
    assert summary.n_predicted_churn == 2 and summary.predicted_rate == 0.5
    assert summary.mean_probability == pytest.approx(37.5)
    assert summary.actual is None

    labelled = db.batch_summary(results.assign(**{TARGET_COL: [0, 0, 0, 1]}))
    assert labelled.actual.rate == 0.25 and labelled.predicted_rate == 0.5


def test_batch_summary_without_threshold_has_no_predicted_count():
    summary = db.batch_summary(pd.DataFrame({PROBABILITY_COL: [10.0, 80.0]}))
    assert summary.n_predicted_churn is None and summary.predicted_rate is None


def test_risk_segments_boundaries_are_inclusive_on_the_lower_edge():
    seg = db.assign_risk_segments(pd.Series([0, 24.99, 25, 59.99, 60, 100]), 25, 60)
    assert list(seg) == ["Low", "Low", "Medium", "Medium", "High", "High"]
    counts = db.risk_segment_counts(seg)
    assert counts["Customers"].tolist() == [2, 2, 2] and counts["Share"].sum() == pytest.approx(1)


@pytest.mark.parametrize("low,high", [(60, 25), (30, 30), (0, 50), (50, 100)])
def test_invalid_risk_bounds_are_rejected(low, high):
    with pytest.raises(ValueError):
        db.assign_risk_segments([10.0], low, high)


def test_default_risk_bounds_start_at_threshold():
    assert db.default_risk_bounds(0.25) == (25, 60)
    assert db.default_risk_bounds(0.7) == (70, 80)
    assert db.default_risk_bounds(None) == (30, 60)


def test_risk_segments_do_not_change_threshold_predictions(trained_model, schema, customers):
    X, _ = prepare_features(customers, schema)
    results, proba = score_prepared(trained_model, customers, X, 0.4)
    before = results.copy()
    results[db.RISK_COL] = db.assign_risk_segments(results[PROBABILITY_COL], 10, 90)
    assert results[PREDICTION_COL].tolist() == (proba >= 0.4).astype(int).tolist()
    pd.testing.assert_frame_equal(results.drop(columns=db.RISK_COL), before)
    assert results[ID_COL].tolist() == customers[ID_COL].tolist()


def test_filter_by_risk_sorts_and_filters_without_mutating():
    results = pd.DataFrame({ID_COL: [1, 2, 3, 4], PROBABILITY_COL: [40.0, 90.0, 10.0, 90.0],
                            PREDICTION_COL: [1, 1, 0, 1]})
    results[db.RISK_COL] = db.assign_risk_segments(results[PROBABILITY_COL], 25, 60)
    view = db.filter_by_risk(results, ["High", "Medium"], sort="desc")
    assert view[ID_COL].tolist() == [2, 4, 1]
    assert db.filter_by_risk(results, sort="original")[ID_COL].tolist() == [1, 2, 3, 4]
    assert db.filter_by_risk(results, min_probability=50, sort="asc")[ID_COL].tolist() == [2, 4]
    assert db.filter_by_risk(results, predicted_only=True)[ID_COL].tolist() == [2, 4, 1]
    assert results[ID_COL].tolist() == [1, 2, 3, 4]


def test_churn_rate_by_segments_and_missing_values():
    df = pd.DataFrame({"Tenure": [0, 1, 3, 30, np.nan, 30], TARGET_COL: [1, 1, 0, 0, 1, 1]})
    table = db.churn_rate_by(df, "Tenure").set_index("Segment")
    assert table.loc["0–1 months", "Churn Rate"] == 1.0
    assert table.loc["24+ months", "Churn Rate"] == 0.5
    assert table.loc[db.MISSING_LABEL, "Customers"] == 1
    assert table["Customers"].sum() == len(df)
    assert list(table.index) == ["0–1 months", "2–5 months", "24+ months", db.MISSING_LABEL]


def test_segment_column_handles_discrete_continuous_and_text():
    discrete = db.segment_column(pd.Series([1, 5, 3, None], name="SatisfactionScore"))
    assert list(discrete.cat.categories) == ["1", "3", "5", db.MISSING_LABEL]
    continuous = db.segment_column(pd.Series(np.linspace(0, 300, 50), name="CashbackAmount"))
    assert continuous.notna().all() and len(continuous.cat.categories) == db.NUMERIC_BINS
    text = db.segment_column(pd.Series([" Male", "Female", None], name="Gender"))
    assert list(text) == ["Male", "Female", db.MISSING_LABEL]
    complaint = db.segment_column(pd.Series([0, 1], name="Complain"))
    assert list(complaint) == ["No complaint", "Complained"]


def test_key_observations_are_data_driven(customers):
    notes = db.key_observations(customers)
    summary = db.actual_churn(customers)
    assert notes[0].startswith(f"{summary.n_churned:,} of {summary.n_customers:,}")
    assert any("complaint status" in n for n in notes)


def test_feature_distribution_splits_by_status_only_with_labels(customers):
    labelled = db.feature_distribution(customers, "Complain")
    assert set(labelled["Status"]) == {"Retained", "Churned"}
    assert labelled["Customers"].sum() == len(customers)
    unlabelled = db.feature_distribution(customers.drop(columns=TARGET_COL), "Complain")
    assert "Status" not in unlabelled.columns


def test_filter_customers_ranges_selections_and_missing():
    df = pd.DataFrame({"Tenure": [1, 5, np.nan, 20], "CityTier": [1, 2, 3, 1]})
    kept = db.filter_customers(df, {"Tenure": (0, 10)}, {"CityTier": [1, 3]})
    assert kept.index.tolist() == [0, 2]
    assert db.filter_customers(df, {"Tenure": (0, 10)}, include_missing=False).index.tolist() == [0, 1]


def test_profile_by_churn_reports_medians(customers):
    profile = db.profile_by_churn(customers, ["Tenure", "Complain", "Gender"])
    assert list(profile.columns) == ["Retained", "Churned", "Missing %"]
    assert "Gender" not in profile.index
    churned = customers[customers[TARGET_COL] == 1]
    assert profile.loc["Tenure", "Churned"] == pytest.approx(churned["Tenure"].median())


def test_upload_quality_reports_duplicates_and_ids(customers):
    df = pd.concat([customers, customers.iloc[:3]], ignore_index=True)
    q = db.upload_quality(df, [c for c in customers.columns if c not in (ID_COL, TARGET_COL)])
    assert q["n_rows"] == len(df) and q["exact_duplicate_rows"] == 3
    assert q["duplicate_feature_rows"] >= 3 and q["duplicate_ids"] == 6
    assert q["has_labels"] and q["missing"]["Missing"].sum() == df.isna().sum().sum()


def test_category_aliases_only_reports_present_pairs():
    found = db.category_aliases({"PreferredPaymentMode": ["CC", "Credit Card", "UPI"],
                                 "PreferredLoginDevice": ["Mobile Phone"]})
    assert found == [("PreferredPaymentMode", "CC", "Credit Card")]


def test_artifact_issues_flags_missing_and_stale_files(tmp_path):
    model = tmp_path / "model.joblib"
    model.write_bytes(b"x")
    (tmp_path / "roc_curve.png").write_bytes(b"x")
    stale = tmp_path / "pr_curve.png"
    stale.write_bytes(b"x")
    old = os.path.getmtime(model) - 7200
    os.utime(stale, (old, old))
    metadata = {"artifacts": {"plots": ["roc_curve.png", "pr_curve.png", "confusion_matrix.png"]}}
    issues = db.artifact_issues(str(tmp_path), metadata, str(model))
    assert len(issues) == 2
    assert any("pr_curve.png" in i and "different training run" in i for i in issues)
    assert any("confusion_matrix.png" in i and "missing" in i for i in issues)


@pytest.mark.skipif(not (ROOT / "outputs" / "metrics.json").exists(), reason="no saved metrics")
def test_saved_artifacts_are_consistent():
    metadata = json.loads((ROOT / "outputs" / "metrics.json").read_text())
    assert db.artifact_issues(str(ROOT / "outputs"), metadata, str(ROOT / "outputs" / "best_model.joblib")) == []


def test_charts_build_from_aggregates(customers):
    table = db.churn_rate_by(customers, "Tenure")
    results = pd.DataFrame({PROBABILITY_COL: [0.0, 25.0, 99.0, 100.0]})
    segments = db.assign_risk_segments(results[PROBABILITY_COL], 25, 60)
    built = [
        charts.churn_rate_bars(table, "t", 0.2), charts.churn_donut(10, 2),
        charts.distribution_bars(db.feature_distribution(customers, "Tenure"), "t"),
        charts.risk_histogram(results[PROBABILITY_COL], segments, 25.0),
        charts.confusion_matrix_chart({"tn": 5, "fp": 1, "fn": 2, "tp": 3}),
        charts.missing_values_chart(db.missing_value_table(customers)),
        charts.cv_comparison_chart(pd.DataFrame({"model_name": ["a", "b"], "cv_pr_auc_mean": [.8, .7],
                                                 "cv_pr_auc_std": [.01, .02], "selected": [True, False]})),
    ]
    for chart in built:
        assert chart.to_dict()


def _render_model_performance(output_dir, metrics_path):
    import json
    from pathlib import Path

    import dashboard_pages as pages

    ctx = pages.DashboardContext(
        model=None, model_sha="", model_name="m", schema=None,
        metadata=json.loads(Path(metrics_path).read_text()), threshold=0.25, metadata_error=None,
        reference=None, reference_error="not loaded", output_dir=Path(output_dir),
        model_path=Path(output_dir) / "best_model.joblib", train_command="train",
        excel_sheets=None, parse_upload=None)
    pages.model_performance(ctx)


@pytest.mark.skipif(not (ROOT / "outputs" / "metrics.json").exists(), reason="no saved metrics")
def test_missing_plot_artifacts_do_not_crash_model_page(tmp_path):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_render_model_performance,
                               args=(str(tmp_path), str(ROOT / "outputs" / "metrics.json"))).run()
    assert not at.exception and not at.error
    assert any("is missing" in w.value for w in at.warning)
    assert any("not found" in c.value for c in at.caption)
