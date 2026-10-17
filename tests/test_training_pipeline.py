import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from inference import (
    PREDICTION_COL, file_sha256, load_metadata, positive_class_proba, predict_churn,
    read_uploaded_table, resolve_threshold, schema_from_model,
)
from train_models import run_training
from utils import ID_COL, TARGET_COL, to_jsonable, write_json_atomic

TINY_SPACES = {
    "logreg": (LogisticRegression(max_iter=2000, class_weight="balanced"),
               {"clf__C": [0.1, 1.0]}),
    "rf": (RandomForestClassifier(n_estimators=30, random_state=0, class_weight="balanced_subsample"),
           {"clf__max_depth": [3, 6]}),
}


@pytest.fixture(scope="module")
def training_run(tmp_path_factory, make_df):
    outdir = tmp_path_factory.mktemp("outputs")
    raw = make_df(n=500, seed=5, n_duplicates=40)
    metrics = run_training(raw, str(outdir), seed=0, n_iter=2, cv_folds=3, n_jobs=1,
                           spaces=TINY_SPACES, source="synthetic")
    return raw, outdir, metrics


def test_artifacts_are_written(training_run):
    _, outdir, _ = training_run
    for name in ["best_model.joblib", "metrics.json", "classification_report.txt",
                 "model_comparison.csv", "data_quality_report.json", "removed_duplicates.csv",
                 "roc_curve.png", "pr_curve.png", "confusion_matrix.png", "feature_importance.png"]:
        assert (outdir / name).stat().st_size > 0, name


def test_metadata_is_consistent(training_run):
    raw, outdir, metrics = training_run
    saved = load_metadata(str(outdir / "metrics.json"))
    assert saved == json.loads(json.dumps(to_jsonable(metrics)))

    model_sha = file_sha256(str(outdir / "best_model.joblib"))
    threshold = resolve_threshold(saved, model_sha)
    assert saved["test_metrics"]["threshold"] == threshold
    assert saved["validation_metrics"]["threshold"] == threshold
    assert saved["random_seed"] == 0
    assert saved["model_selection"]["criterion"] == "mean_cv_average_precision"
    assert "val_pr_auc" not in json.dumps(saved)

    dedup = saved["data"]["deduplication"]
    assert dedup["n_removed_identical_records"] == 40
    assert saved["data"]["n_rows_after_deduplication"] == len(raw) - 40
    assert set(saved["data"]["split"]["shared_feature_records_between_partitions"].values()) == {0}
    sizes = [p["n_rows"] for p in saved["data"]["split"]["partitions"].values()]
    assert sum(sizes) == len(raw) - 40


def test_selected_model_has_best_cv_score(training_run):
    _, outdir, metrics = training_run
    candidates = metrics["candidates"]
    assert {c["model"] for c in candidates} == set(TINY_SPACES)
    best = max(candidates, key=lambda c: c["cv_pr_auc_mean"])
    assert best["selected"] and metrics["model_selection"]["selected_model"] == best["model"]
    assert metrics["threshold"]["value"] == best["validation_threshold"]

    table = pd.read_csv(outdir / "model_comparison.csv")
    assert len(table) == len(TINY_SPACES)
    for col in ["cv_pr_auc_mean", "validation_roc_auc", "validation_pr_auc",
                "validation_precision", "validation_recall", "validation_f1"]:
        assert table[col].notna().all(), col


def test_reported_test_metrics_match_saved_model(training_run):
    raw, outdir, metrics = training_run
    test = metrics["test_metrics"]
    cm = test["confusion_matrix"]
    assert cm["tp"] + cm["fn"] == test["n_positive"]
    assert sum(cm.values()) == test["n_samples"]
    assert test["recall"] == pytest.approx(cm["tp"] / (cm["tp"] + cm["fn"]))
    report = (outdir / "classification_report.txt").read_text()
    assert f"threshold {test['threshold']:.2f}" in report


def test_end_to_end_inference_from_saved_artifacts(training_run, make_df):
    _, outdir, metrics = training_run
    model = joblib.load(outdir / "best_model.joblib")
    threshold = resolve_threshold(load_metadata(str(outdir / "metrics.json")),
                                  file_sha256(str(outdir / "best_model.joblib")))

    new = make_df(n=25, seed=99).drop(columns=[TARGET_COL])
    df = read_uploaded_table(new.to_csv(index=False).encode(), "new.csv")
    results, proba, _ = predict_churn(model, df, schema_from_model(model), threshold)

    assert results[ID_COL].tolist() == new[ID_COL].tolist()
    assert results[PREDICTION_COL].tolist() == (proba >= threshold).astype(int).tolist()
    assert schema_from_model(model).features == metrics["features"]["features"]


def test_to_jsonable_handles_numpy_types(tmp_path):
    data = {"a": np.float64(0.5), "b": np.int64(3), "c": np.array([1, 2]), "d": np.bool_(True),
            "e": (np.float32(1.5), None), "f": float("nan"), 7: "key"}
    path = tmp_path / "x.json"
    write_json_atomic(data, str(path))
    loaded = json.loads(path.read_text())
    assert loaded == {"a": 0.5, "b": 3, "c": [1, 2], "d": True, "e": [1.5, None], "f": None, "7": "key"}
    assert list(tmp_path.iterdir()) == [path]
