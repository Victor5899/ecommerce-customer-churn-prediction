"""Explainability (permutation importance, Shapley values) and probability calibration."""
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline

import calibration as cal
import explainability as xai
from inference import (
    CALIBRATED_COL, PREDICTION_COL, PROBABILITY_COL, file_sha256, positive_class_proba,
    prepare_features, score_prepared,
)
from train_models import build_preprocessor
from utils import separate_features_target

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
DATASET = ROOT / "data" / "E Commerce Dataset.xlsx"
REAL = (OUTPUTS / "best_model.joblib").exists() and (OUTPUTS / "metrics.json").exists() and DATASET.exists()
real_only = pytest.mark.skipif(not REAL, reason="trained artifacts or dataset not found")


@pytest.fixture(scope="session")
def rf_model(make_df):
    X, y = separate_features_target(make_df(n=400, seed=3))
    pre, _, _ = build_preprocessor(X)
    model = Pipeline([("pre", pre), ("clf", RandomForestClassifier(n_estimators=40, max_depth=6,
                                                                    random_state=0))])
    return model.fit(X, y)


@pytest.fixture
def prepared(make_df, schema):
    X, y = separate_features_target(make_df(n=200, seed=7))
    X, _ = prepare_features(X, schema)
    return X, y


class ReversedClasses:
    """A model whose classes_ are ordered [1, 0]."""

    def __init__(self, model):
        self.model, self.classes_ = model, np.array([1, 0])

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, ::-1]


# --------------------------------------------------------------------------- #
# Global importance
# --------------------------------------------------------------------------- #
def test_transformed_feature_sources_map_one_hot_columns_to_raw_names(rf_model, schema):
    sources = xai.transformed_feature_sources(rf_model)
    assert len(sources) == rf_model.named_steps["clf"].n_features_in_
    assert set(sources) == set(schema.features)
    for col in schema.categorical:
        assert sources.count(col) == len(schema.categories[col])


def test_impurity_importance_is_aggregated_per_raw_feature(rf_model, schema):
    imp = xai.impurity_importance(rf_model)
    assert set(imp.index) == set(schema.features)
    assert imp.sum() == pytest.approx(1.0)
    assert xai.impurity_importance(Pipeline([("pre", rf_model.named_steps["pre"]),
                                             ("clf", object())])) is None


def test_permutation_importance_preserves_names_and_orders_by_mean(trained_model, prepared):
    X, y = prepared
    table, baseline = xai.permutation_importance_table(trained_model, X, y, n_repeats=3, seed=0)
    assert sorted(table["feature"]) == sorted(X.columns)
    assert table["importance_mean"].is_monotonic_decreasing
    assert table["rank"].tolist() == list(range(1, len(X.columns) + 1))
    assert (table["importance_std"] >= 0).all() and 0 < baseline <= 1
    assert set(table.head(3)["feature"]) & {"Tenure", "Complain", "CashbackAmount"}


@pytest.mark.parametrize("case", ["one_class", "few_repeats", "empty"])
def test_permutation_importance_rejects_invalid_inputs(trained_model, prepared, case):
    X, y = prepared
    kwargs = {"n_repeats": 3}
    if case == "one_class":
        y = np.zeros(len(y), dtype=int)
    elif case == "few_repeats":
        kwargs["n_repeats"] = 1
    else:
        X, y = X.iloc[:0], y.iloc[:0]
    with pytest.raises(xai.ExplanationError):
        xai.permutation_importance_table(trained_model, X, y, **kwargs)


def test_compare_importances_requires_matching_names():
    perm = pd.DataFrame({"feature": ["a", "b", "c"], "importance_mean": [.3, .2, .1], "rank": [1, 2, 3]})
    comparison = xai.compare_importances(perm, pd.Series({"a": .5, "b": .3, "c": .2}))
    assert comparison["impurity_rank"].tolist() == [1, 2, 3]
    assert xai.rank_correlation(comparison) == pytest.approx(1.0)
    assert xai.compare_importances(perm, pd.Series({"a": .5, "x": .5})) is None
    assert xai.compare_importances(perm, None) is None


# --------------------------------------------------------------------------- #
# Shapley values
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model_name", ["trained_model", "rf_model"])
def test_shapley_values_add_up_to_the_model_prediction(request, model_name, prepared):
    model = request.getfixturevalue(model_name)
    X, _ = prepared
    exp = xai.shapley_values(model, X.iloc[[0]], X.iloc[10:40], n_permutations=4)
    assert exp.contributions.shape == (X.shape[1],) and exp.std_errors.shape == (X.shape[1],)
    assert exp.prediction == pytest.approx(positive_class_proba(model, X.iloc[[0]])[0], abs=1e-12)
    assert exp.base_value == pytest.approx(positive_class_proba(model, X.iloc[10:40]).mean(), abs=1e-12)
    assert exp.is_consistent(1e-9)
    frame = exp.to_frame()
    assert list(frame.columns) == ["feature", "value", "contribution", "std_error"]
    assert frame["contribution"].abs().is_monotonic_decreasing
    assert sorted(frame["feature"]) == sorted(X.columns)


def test_shapley_values_explain_the_positive_class_regardless_of_class_order(trained_model, prepared):
    X, _ = prepared
    reversed_model = ReversedClasses(trained_model)
    a = xai.shapley_values(trained_model, X.iloc[[1]], X.iloc[20:40], n_permutations=2, seed=5)
    b = xai.shapley_values(reversed_model, X.iloc[[1]], X.iloc[20:40], n_permutations=2, seed=5)
    np.testing.assert_allclose(a.contributions, b.contributions, atol=1e-12)
    assert b.prediction == pytest.approx(trained_model.predict_proba(X.iloc[[1]])[0, 1])


def test_feature_that_matches_background_gets_zero_contribution(trained_model, prepared):
    X, _ = prepared
    row = X.iloc[[0]].copy()
    background = X.iloc[10:30].copy()
    background["Tenure"] = row["Tenure"].iloc[0] if pd.notna(row["Tenure"].iloc[0]) else 5.0
    row["Tenure"] = background["Tenure"].iloc[0]
    exp = xai.shapley_values(trained_model, row, background, n_permutations=2)
    assert exp.contributions[exp.features.index("Tenure")] == pytest.approx(0.0, abs=1e-12)


def test_shapley_values_handle_missing_values_and_are_deterministic(rf_model, prepared):
    X, _ = prepared
    row = X.iloc[[2]].copy()
    row["Tenure"] = np.nan
    row["Gender"] = np.nan
    a = xai.shapley_values(rf_model, row, X.iloc[30:60], n_permutations=2, seed=1)
    b = xai.shapley_values(rf_model, row, X.iloc[30:60], n_permutations=2, seed=1)
    assert a.is_consistent(1e-9)
    np.testing.assert_array_equal(a.contributions, b.contributions)
    assert a.feature_values[a.features.index("Tenure")] is None


@pytest.mark.parametrize("case", ["two_rows", "empty_background", "odd_permutations", "missing_column",
                                  "not_frame"])
def test_shapley_values_reject_invalid_inputs(trained_model, prepared, case):
    X, _ = prepared
    row, background, kwargs = X.iloc[[0]], X.iloc[5:15], {}
    if case == "two_rows":
        row = X.iloc[:2]
    elif case == "empty_background":
        background = X.iloc[:0]
    elif case == "odd_permutations":
        kwargs["n_permutations"] = 3
    elif case == "missing_column":
        background = background.drop(columns="Tenure")
    else:
        row = X.iloc[0]
    with pytest.raises(xai.ExplanationError):
        xai.shapley_values(trained_model, row, background, **kwargs)


class AdditiveModel:
    """P(churn) = 0.2 + 0.005 * Tenure + 0.1 * Complain, so Shapley values are known exactly."""

    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        p = 0.2 + 0.005 * X["Tenure"].to_numpy(float) + 0.1 * X["Complain"].to_numpy(float)
        return np.column_stack([1 - p, p])


def test_shapley_values_are_exact_with_zero_error_for_an_additive_model(prepared):
    X, _ = prepared
    X = X.assign(Tenure=X["Tenure"].fillna(10.0))
    row, background = X.iloc[[0]], X.iloc[10:40]
    exp = xai.shapley_values(AdditiveModel(), row, background, n_permutations=4)
    expected = {"Tenure": 0.005 * (row["Tenure"].iloc[0] - background["Tenure"].mean()),
                "Complain": 0.1 * (row["Complain"].iloc[0] - background["Complain"].mean())}
    for feature, value in zip(exp.features, exp.contributions):
        assert value == pytest.approx(expected.get(feature, 0.0), abs=1e-12)
    np.testing.assert_allclose(exp.std_errors, 0.0, atol=1e-12)


def test_shapley_std_errors_need_two_pairs_per_background_row(trained_model, prepared):
    X, _ = prepared
    assert np.isnan(xai.shapley_values(trained_model, X.iloc[[0]], X.iloc[5:15],
                                       n_permutations=2).std_errors).all()
    se = xai.shapley_values(trained_model, X.iloc[[0]], X.iloc[5:15], n_permutations=4).std_errors
    assert np.all(np.isfinite(se)) and np.all(se >= 0)


def test_explain_customers_limits_the_batch_size(trained_model, prepared):
    X, _ = prepared
    assert len(xai.explain_customers(trained_model, X.iloc[:2], X.iloc[5:10], n_permutations=2)) == 2
    with pytest.raises(xai.ExplanationError):
        xai.explain_customers(trained_model, X.iloc[:xai.MAX_EXPLAINED_CUSTOMERS + 1], X.iloc[5:10])


def test_explainability_artifact_validation(tmp_path, schema, prepared):
    X, _ = prepared
    meta = {"schema_version": xai.EXPLAIN_SCHEMA_VERSION, "model_sha256": "abc",
            "permutation_importance": {"features": []},
            "background": {"records": X.iloc[:3].astype(object).where(X.iloc[:3].notna(), None).to_dict("records")}}
    path = tmp_path / "explainability.json"
    path.write_text(json.dumps(meta))
    loaded = xai.load_explainability(str(path), "abc")
    bg = xai.background_frame(loaded, schema)
    assert list(bg.columns) == schema.features and len(bg) == 3
    with pytest.raises(xai.ExplanationError, match="stale"):
        xai.load_explainability(str(path), "other")
    path.write_text(json.dumps({**meta, "schema_version": 99}))
    with pytest.raises(xai.ExplanationError):
        xai.load_explainability(str(path), "abc")
    with pytest.raises(xai.ExplanationError):
        xai.load_explainability(str(tmp_path / "missing.json"), "abc")


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def _miscalibrated(n, seed):
    rng = np.random.default_rng(seed)
    true_p = rng.uniform(0.01, 0.99, n)
    y = (rng.uniform(size=n) < true_p).astype(int)
    score = 1 / (1 + np.exp(-0.4 * np.log(true_p / (1 - true_p))))  # under-confident scores
    return y, score


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibrators_improve_brier_on_fresh_data_and_preserve_order(method):
    y_fit, p_fit = _miscalibrated(2000, 0)
    y_new, p_new = _miscalibrated(2000, 1)
    calibrator = cal.fit_calibrator(method, y_fit, p_fit)
    assert cal.check_calibrator(calibrator) == []
    calibrated = cal.apply_calibrator(calibrator, p_new)
    assert brier_score_loss(y_new, calibrated) < brier_score_loss(y_new, p_new)
    order = np.argsort(p_new)
    assert np.all(np.diff(calibrated[order]) >= -1e-12)


def test_check_calibrator_rejects_decreasing_mapping():
    y, p = _miscalibrated(500, 2)
    reversed_fit = cal.fit_calibrator("sigmoid", 1 - y, p)
    assert cal.check_calibrator(reversed_fit)


def test_select_and_fit_uses_only_validation_data():
    y_val, p_val = _miscalibrated(800, 3)
    calibrator, cv_table = cal.select_and_fit(y_val, p_val, seed=0)
    assert set(cv_table["method"]) == set(cal.CALIBRATION_METHODS)
    assert calibrator["method"] == cal.select_method(cv_table) != "none"
    refit = cal.fit_calibrator(calibrator["method"], y_val, p_val)
    grid = np.linspace(0, 1, 11)
    np.testing.assert_allclose(cal.apply_calibrator(calibrator, grid), cal.apply_calibrator(refit, grid))


def test_select_method_prefers_simpler_method_on_ties():
    table = pd.DataFrame({"method": ["isotonic", "sigmoid", "none"], "brier": [0.1, 0.1, 0.1]})
    assert cal.select_method(table) == "none"


def test_reliability_table_and_ece():
    y = np.array([0, 0, 1, 1, 1])
    p = np.array([0.05, 0.15, 0.15, 0.95, 1.0])
    table = cal.reliability_table(y, p, n_bins=10)
    assert table["n"].sum() == 5 and table.loc[9, "n"] == 2
    assert table.loc[1, "observed_rate"] == 0.5 and table.loc[1, "mean_predicted"] == pytest.approx(0.15)
    assert np.isnan(table.loc[5, "observed_rate"])
    expected = (1 * 0.05 + 2 * 0.35 + 2 * 0.025) / 5
    assert cal.expected_calibration_error(table) == pytest.approx(expected)
    metrics = cal.probability_metrics(y, p)
    assert metrics["brier"] == pytest.approx(brier_score_loss(y, p))
    assert metrics["roc_auc"] is not None and metrics["n_positive"] == 3


@pytest.mark.parametrize("p", [[0.2, 1.2], [np.nan, 0.5], []])
def test_calibration_rejects_invalid_probabilities(p):
    with pytest.raises(cal.CalibrationError):
        cal.reliability_table([0, 1][:len(p)], p)


def _write_calibration(tmp_path, method="sigmoid", model_sha="m"):
    y, p = _miscalibrated(400, 4)
    calibrator = cal.fit_calibrator(method, y, p)
    meta = {"schema_version": cal.CALIBRATION_SCHEMA_VERSION, "model_sha256": model_sha, "method": method,
            "calibrator_file": None, "calibrator_sha256": None}
    if method != "none":
        joblib.dump(calibrator, tmp_path / "calibrator.joblib")
        meta.update(calibrator_file="calibrator.joblib",
                    calibrator_sha256=file_sha256(str(tmp_path / "calibrator.joblib")))
    (tmp_path / "calibration.json").write_text(json.dumps(meta))
    return tmp_path / "calibration.json"


def test_load_calibration_validates_model_hash_and_calibrator_file(tmp_path):
    path = _write_calibration(tmp_path)
    meta, calibrator = cal.load_calibration(str(path), str(tmp_path), "m")
    assert calibrator["method"] == "sigmoid"
    with pytest.raises(cal.CalibrationError, match="different model"):
        cal.load_calibration(str(path), str(tmp_path), "other")
    joblib.dump({"method": "sigmoid", "estimator": None}, tmp_path / "calibrator.joblib")
    with pytest.raises(cal.CalibrationError, match="SHA-256"):
        cal.load_calibration(str(path), str(tmp_path), "m")
    os.remove(tmp_path / "calibrator.joblib")
    with pytest.raises(cal.CalibrationError, match="not found"):
        cal.load_calibration(str(path), str(tmp_path), "m")


def test_load_calibration_without_calibrator_and_bad_schema(tmp_path):
    path = _write_calibration(tmp_path, method="none")
    assert cal.load_calibration(str(path), str(tmp_path), "m")[1] is None
    path.write_text(json.dumps({"schema_version": 0}))
    with pytest.raises(cal.CalibrationError):
        cal.load_calibration(str(path), str(tmp_path), "m")


def test_calibrated_scoring_keeps_predictions_and_replaces_stale_columns(trained_model, schema, customers):
    y, p = _miscalibrated(400, 5)
    calibrator = cal.fit_calibrator("sigmoid", y, p)
    df = customers.assign(**{CALIBRATED_COL: 1.0})
    X, report = prepare_features(df, schema)
    assert CALIBRATED_COL not in report.extra_columns
    plain, proba = score_prepared(trained_model, df, X, 0.4)
    calibrated, _ = score_prepared(trained_model, df, X, 0.4, calibrator)
    pd.testing.assert_series_equal(plain[PREDICTION_COL], calibrated[PREDICTION_COL])
    pd.testing.assert_series_equal(plain[PROBABILITY_COL], calibrated[PROBABILITY_COL])
    assert CALIBRATED_COL not in plain.columns
    np.testing.assert_allclose(calibrated[CALIBRATED_COL],
                               np.round(cal.apply_calibrator(calibrator, proba) * 100, 2))


# --------------------------------------------------------------------------- #
# Real artifacts (skipped without them)
# --------------------------------------------------------------------------- #
@real_only
def test_build_artifacts_reproduces_partitions_without_test_leakage(tmp_path):
    import explain_calibrate as ec
    from utils import load_raw_dataset

    model_path = OUTPUTS / "best_model.joblib"
    metadata = json.loads((OUTPUTS / "metrics.json").read_text())
    model, sha = joblib.load(model_path), file_sha256(str(model_path))
    raw = load_raw_dataset(str(DATASET))
    result = ec.build_artifacts(model, sha, metadata, raw, str(tmp_path), source="x",
                                source_sha="y", n_repeats=2, seed=metadata["random_seed"])

    parts = ec.reconstruct_partitions(raw, metadata)
    X_val, y_val = parts["validation"]
    calib = result["calibration"]
    assert calib["fit"]["n_rows"] == metadata["data"]["split"]["partitions"]["validation"]["n_rows"]
    assert calib["evaluation"]["partition"] == "test"
    meta, calibrator = cal.load_calibration(str(tmp_path / "calibration.json"), str(tmp_path), sha)
    if calibrator is not None:
        refit = cal.fit_calibrator(calibrator["method"], y_val, positive_class_proba(model, X_val))
        grid = np.linspace(0, 1, 21)
        np.testing.assert_allclose(cal.apply_calibrator(calibrator, grid), cal.apply_calibrator(refit, grid))
    explain = xai.load_explainability(str(tmp_path / "explainability.json"), sha)
    assert explain["permutation_importance"]["partition"] == "validation"
    assert sorted(r["feature"] for r in explain["permutation_importance"]["features"]) == sorted(X_val.columns)

    tampered = json.loads(json.dumps(metadata))
    tampered["data"]["split"]["partitions"]["validation"]["n_rows"] += 1
    with pytest.raises(ec.ReproductionError):
        ec.reconstruct_partitions(raw, tampered)


def _render_performance_without_xai(output_dir, metrics_path):
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


def _render_explanations(model_path, output_dir, with_artifacts):
    import json
    from pathlib import Path

    import joblib
    import pandas as pd

    import dashboard_pages as pages
    import explainability as xai
    from inference import file_sha256, prepare_features, schema_from_model, score_prepared

    model = joblib.load(model_path)
    sha, schema = file_sha256(model_path), schema_from_model(model)
    explain = xai.load_explainability(str(Path(output_dir) / "explainability.json"), sha) if with_artifacts else None
    background = xai.background_frame(explain, schema) if explain else None
    rows = pd.DataFrame.from_records(json.loads((Path(output_dir) / "explainability.json").read_text())
                                     ["background"]["records"][:3])
    X, _ = prepare_features(rows, schema)
    scored, _ = score_prepared(model, rows, X, 0.25)
    ctx = pages.DashboardContext(
        model=model, model_sha=sha, model_name="m", schema=schema, metadata=None, threshold=0.25,
        metadata_error=None, reference=None, reference_error=None, output_dir=Path(output_dir),
        model_path=Path(model_path), train_command="train", excel_sheets=None, parse_upload=None,
        explainability=explain, background=background)
    pages._render_explanations(ctx, scored, scored)


@real_only
def test_model_page_renders_when_xai_artifacts_are_missing(tmp_path):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_render_performance_without_xai,
                               args=(str(tmp_path), str(OUTPUTS / "metrics.json"))).run()
    assert not at.exception and not at.error
    warnings = " ".join(w.value for w in at.warning)
    assert "Permutation importance is unavailable" in warnings
    assert "Calibration results are unavailable" in warnings


@real_only
@pytest.mark.skipif(not (OUTPUTS / "explainability.json").exists(), reason="no explainability artifact")
@pytest.mark.parametrize("with_artifacts", [True, False])
def test_customer_explanations_render_or_degrade_gracefully(with_artifacts):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_render_explanations, default_timeout=60, args=(
        str(OUTPUTS / "best_model.joblib"), str(OUTPUTS), with_artifacts)).run()
    assert not at.exception and not at.error
    if with_artifacts:
        assert not at.warning
        assert {"Model churn score", "Reference value"} <= {m.label for m in at.metric}
    else:
        assert "Customer explanations are unavailable" in " ".join(w.value for w in at.warning)
