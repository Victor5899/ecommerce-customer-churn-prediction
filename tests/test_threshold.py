import json

import numpy as np
import pytest

from inference import (
    METADATA_SCHEMA_VERSION, MetadataError, apply_threshold, load_metadata, resolve_threshold,
)
from train_models import THRESHOLD_GRID, pick_threshold


def test_apply_threshold_below_equal_above():
    proba = np.array([0.26, 0.27, 0.28, 0.0, 1.0])
    assert apply_threshold(proba, 0.27).tolist() == [0, 1, 1, 0, 1]


def test_apply_threshold_exact_grid_value_is_positive():
    proba = np.array([135 / 500])
    assert apply_threshold(proba, 0.27).tolist() == [1]


@pytest.mark.parametrize("bad", [0, 1, -0.1, 1.5, float("nan"), float("inf"), None, "0.3", True])
def test_apply_threshold_rejects_invalid_thresholds(bad):
    with pytest.raises(ValueError):
        apply_threshold(np.array([0.5]), bad)


@pytest.mark.parametrize("proba", [[0.2, np.nan], [1.2], [-0.1], [[0.1, 0.2]]])
def test_apply_threshold_rejects_invalid_probabilities(proba):
    with pytest.raises(ValueError):
        apply_threshold(np.array(proba, dtype=float), 0.5)


def test_threshold_grid():
    assert THRESHOLD_GRID[0] == 0.05 and THRESHOLD_GRID[-1] == 0.9 and len(THRESHOLD_GRID) == 86


def test_pick_threshold_maximises_fbeta_under_precision_constraint():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.35, 0.3, 0.32, 0.1, 0.1, 0.1, 0.1, 0.1])
    best = pick_threshold(y, p, beta=2.0, min_precision=0.55)
    # 0.11-0.30 give the same F2 (4 TP, 1 FP); ties go to the lowest threshold.
    assert best["value"] == pytest.approx(0.11)
    assert best["recall"] == 1.0 and best["precision"] == pytest.approx(0.8)
    assert best["fbeta"] == pytest.approx(4 / 4.2)
    assert best["constraint_satisfied"] is True


def test_pick_threshold_respects_min_precision():
    y = np.array([1, 0, 0, 0, 1, 0])
    p = np.array([0.9, 0.6, 0.6, 0.6, 0.2, 0.1])
    best = pick_threshold(y, p, beta=2.0, min_precision=0.9)
    assert best["precision"] >= 0.9
    assert best["value"] > 0.6


def test_pick_threshold_flags_unmet_constraint():
    y = np.array([1, 0, 0, 0])
    p = np.array([0.5, 0.5, 0.5, 0.5])
    best = pick_threshold(y, p, min_precision=0.9)
    assert best["constraint_satisfied"] is False
    assert best["recall"] == 1.0


def valid_metadata(threshold=0.3, sha="abc"):
    return {
        "schema_version": METADATA_SCHEMA_VERSION,
        "threshold": {"value": threshold},
        "test_metrics": {"threshold": threshold},
        "artifacts": {"model_sha256": sha},
    }


def test_resolve_threshold_valid():
    assert resolve_threshold(valid_metadata(), "abc") == 0.3


@pytest.mark.parametrize("mutate, message", [
    (lambda m: m.update(schema_version=1), "schema version"),
    (lambda m: m.pop("threshold"), "threshold.value"),
    (lambda m: m["threshold"].update(value=1.5), "Invalid decision threshold"),
    (lambda m: m["threshold"].update(value="high"), "Invalid decision threshold"),
    (lambda m: m["test_metrics"].update(threshold=0.5), "different|computed at"),
    (lambda m: m.pop("test_metrics"), "computed at"),
])
def test_resolve_threshold_rejects_bad_metadata(mutate, message):
    metadata = valid_metadata()
    mutate(metadata)
    with pytest.raises(MetadataError, match=message):
        resolve_threshold(metadata, "abc")


def test_resolve_threshold_rejects_other_model():
    with pytest.raises(MetadataError, match="SHA-256"):
        resolve_threshold(valid_metadata(sha="abc"), "def")


def test_legacy_metrics_file_is_rejected(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"selection": {"threshold": {"thr": 0.27}}}))
    with pytest.raises(MetadataError, match="schema version"):
        resolve_threshold(load_metadata(str(path)))


def test_load_metadata_errors(tmp_path):
    with pytest.raises(MetadataError, match="not found"):
        load_metadata(str(tmp_path / "missing.json"))
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(MetadataError, match="could not be read"):
        load_metadata(str(bad))
