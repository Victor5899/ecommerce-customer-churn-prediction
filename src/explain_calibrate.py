"""Build explainability and calibration artifacts for the saved model without retraining it.

The training/validation/test partitions are rebuilt from the settings in
``metrics.json`` and verified against the recorded partition sizes and PR-AUC
values before anything is written. The model file and ``metrics.json`` are not
modified. Outputs (in ``--outdir``):

* ``explainability.json``: permutation importance on the validation partition,
  impurity-importance comparison and a training-partition background sample
  for per-customer Shapley values.
* ``calibration.json``: calibration method selected by cross-validation on the
  validation partition, plus a single comparison on the held-out test partition.
* ``calibrator.joblib``: only when a calibrator is selected and passes its checks.
"""
from __future__ import annotations

import argparse
import math
import os
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.metrics import average_precision_score

import calibration as cal
import explainability as xai
from inference import file_sha256, load_metadata, positive_class_proba, resolve_threshold
from train_models import _dump_atomic
from utils import ID_COL, deduplicate, load_raw_dataset, split_xy, write_json_atomic

EXPLAIN_FILE = "explainability.json"
CALIBRATION_FILE = "calibration.json"
CALIBRATOR_FILE = "calibrator.joblib"
BACKGROUND_SIZE = 100


class ReproductionError(RuntimeError):
    """Raised when the saved partitions or scores cannot be reproduced exactly."""


def reconstruct_partitions(raw_df, metadata: dict) -> dict:
    data, split = metadata["data"], metadata["data"]["split"]
    clean, _, _ = deduplicate(raw_df, conflict_policy=data["deduplication"]["conflict_policy"])
    df = clean.drop(columns=[ID_COL], errors="ignore")
    X_train, X_val, X_test, y_train, y_val, y_test = split_xy(
        df, split["test_size"], split["validation_size_of_remaining"], metadata["random_seed"])
    parts = {"train": (X_train, y_train), "validation": (X_val, y_val), "test": (X_test, y_test)}
    for name, (_, y) in parts.items():
        expected = split["partitions"][name]
        if len(y) != expected["n_rows"] or int(y.sum()) != expected["n_positive"]:
            raise ReproductionError(
                f"Rebuilt {name} partition has {len(y)} rows / {int(y.sum())} churners, but the "
                f"metadata records {expected['n_rows']} / {expected['n_positive']}. Re-run training.")
    return parts


def verify_scores(model, parts: dict, metadata: dict) -> dict:
    """Recompute PR-AUC on the rebuilt partitions and require an exact match with the metadata."""
    scores = {}
    for name, key in (("validation", "validation_metrics"), ("test", "test_metrics")):
        X, y = parts[name]
        scores[name] = positive_class_proba(model, X)
        recorded = metadata[key]["pr_auc"]
        if not math.isclose(average_precision_score(y, scores[name]), recorded, abs_tol=1e-9):
            raise ReproductionError(f"Recomputed {name} PR-AUC does not match the metadata ({recorded}).")
    return scores


def build_artifacts(model, model_sha: str, metadata: dict, raw_df, outdir: str, *, source: str,
                    source_sha: str, n_repeats: int = 10, seed: int = 42, n_jobs=None) -> dict:
    threshold = resolve_threshold(metadata, model_sha)
    parts = reconstruct_partitions(raw_df, metadata)
    scores = verify_scores(model, parts, metadata)
    X_train, _ = parts["train"]
    X_val, y_val = parts["validation"]
    _, y_test = parts["test"]
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    data_info = {"source": source, "source_sha256": source_sha}

    table, baseline = xai.permutation_importance_table(
        model, X_val, y_val, n_repeats=n_repeats, seed=seed, n_jobs=n_jobs)
    comparison = xai.compare_importances(table, xai.impurity_importance(model))
    background = X_train.sample(min(BACKGROUND_SIZE, len(X_train)), random_state=seed)
    explain_meta = {
        "schema_version": xai.EXPLAIN_SCHEMA_VERSION, "created_at_utc": created,
        "model_sha256": model_sha, "data": data_info,
        "permutation_importance": {
            "partition": "validation", "n_rows": len(y_val), "n_positive": int(y_val.sum()),
            "scoring": "average_precision", "baseline_score": baseline,
            "n_repeats": n_repeats, "seed": seed, "features": table.to_dict("records"),
        },
        "impurity_comparison": None if comparison is None else {
            "spearman_rank_correlation": xai.rank_correlation(comparison),
            "features": comparison.to_dict("records"),
        },
        "background": {"partition": "train", "n_rows": len(background), "seed": seed,
                       "records": background.to_dict("records")},
    }
    write_json_atomic(explain_meta, os.path.join(outdir, EXPLAIN_FILE))

    calibrator, cv_table = cal.select_and_fit(y_val, scores["validation"], seed=seed)
    rejected = cal.check_calibrator(calibrator) if calibrator["method"] != "none" else []
    selected = calibrator["method"]
    if rejected:
        calibrator = cal.fit_calibrator("none", y_val, scores["validation"])

    p_test = scores["test"]
    evaluation = {
        "partition": "test", "uncalibrated": cal.probability_metrics(y_test, p_test),
        "reliability": {"uncalibrated": cal.reliability_table(y_test, p_test).to_dict("records")},
    }
    calibrator_path = os.path.join(outdir, CALIBRATOR_FILE)
    calibrator_sha, threshold_info = None, {"value": threshold, "labels_use": "uncalibrated model score"}
    if calibrator["method"] != "none":
        p_cal = cal.apply_calibrator(calibrator, p_test)
        evaluation["calibrated"] = cal.probability_metrics(y_test, p_cal)
        evaluation["reliability"]["calibrated"] = cal.reliability_table(y_test, p_cal).to_dict("records")
        thr_cal = float(cal.apply_calibrator(calibrator, [threshold])[0])
        threshold_info["calibrated_value_at_threshold"] = thr_cal
        threshold_info["test_label_agreement_at_mapped_threshold"] = float(
            np.mean((p_cal >= thr_cal) == (p_test >= threshold)))
        _dump_atomic(calibrator, calibrator_path)
        calibrator_sha = file_sha256(calibrator_path)
    elif os.path.exists(calibrator_path):
        os.remove(calibrator_path)

    calibration_meta = {
        "schema_version": cal.CALIBRATION_SCHEMA_VERSION, "created_at_utc": created,
        "model_sha256": model_sha, "data": data_info,
        "method": calibrator["method"], "selected_by_cv": selected,
        "rejected_reasons": rejected,
        "calibrator_file": CALIBRATOR_FILE if calibrator_sha else None,
        "calibrator_sha256": calibrator_sha,
        "fit": {"partition": "validation", "n_rows": len(y_val), "n_positive": int(y_val.sum())},
        "selection": {"criterion": cal.SELECTION_CRITERION, "cv": "StratifiedKFold(n_splits=5, shuffle=True)",
                      "seed": seed, "results": cv_table.to_dict("records")},
        "threshold": threshold_info,
        "evaluation": evaluation,
    }
    write_json_atomic(calibration_meta, os.path.join(outdir, CALIBRATION_FILE))
    return {"explainability": explain_meta, "calibration": calibration_meta}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--n-repeats", type=int, default=10)
    ap.add_argument("--n-jobs", type=int, default=None)
    args = ap.parse_args()

    model_path = os.path.join(args.outdir, "best_model.joblib")
    metadata = load_metadata(os.path.join(args.outdir, "metrics.json"))
    model, model_sha = joblib.load(model_path), file_sha256(model_path)
    result = build_artifacts(
        model, model_sha, metadata, load_raw_dataset(args.input), args.outdir, source=args.input,
        source_sha=file_sha256(args.input), n_repeats=args.n_repeats,
        seed=metadata["random_seed"], n_jobs=args.n_jobs)

    pi = result["explainability"]["permutation_importance"]["features"][:5]
    calib = result["calibration"]
    ev = calib["evaluation"]
    print("[OK] Partitions and recorded PR-AUC reproduced exactly.")
    print("Top permutation importance (validation, drop in PR-AUC): "
          + ", ".join(f"{r['feature']}={r['importance_mean']:.3f}" for r in pi))
    print(f"Calibration method: {calib['method']} (selected by validation CV: {calib['selected_by_cv']})")
    for name in ("uncalibrated", "calibrated"):
        if name in ev:
            m = ev[name]
            print(f"Test {name}: Brier={m['brier']:.4f} log loss={m['log_loss']:.4f} ECE={m['ece']:.4f} "
                  f"ROC-AUC={m['roc_auc']:.4f} PR-AUC={m['pr_auc']:.4f}")


if __name__ == "__main__":
    main()
