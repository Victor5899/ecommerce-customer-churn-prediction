"""Post-hoc probability calibration and probability-quality metrics.

Calibrators map the fitted model's churn score to a calibrated probability.
They are fitted on the validation partition only; churn labels keep using the
uncalibrated score and the tuned decision threshold.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from inference import file_sha256

CALIBRATION_SCHEMA_VERSION = 1
CALIBRATION_METHODS = ("none", "sigmoid", "isotonic")
SELECTION_CRITERION = "lowest mean out-of-fold Brier score on the validation partition"
EPS = 1e-6


class CalibrationError(ValueError):
    """Raised when a calibrator cannot be fitted, applied or trusted."""


def _check(y, p) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p, dtype=float).ravel()
    if p.size == 0 or not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise CalibrationError("Probabilities must be a non-empty array of finite values in [0, 1].")
    if y is None:
        return None, p
    y = np.asarray(y).astype(int).ravel()
    if len(y) != len(p) or not np.isin(y, [0, 1]).all():
        raise CalibrationError("Labels must be 0/1 and aligned with the probabilities.")
    return y, p


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_calibrator(method: str, y, p) -> dict:
    """Fit ``method`` on (label, score) pairs. 'sigmoid' is logistic regression on the log-odds."""
    if method not in CALIBRATION_METHODS:
        raise CalibrationError(f"Unknown calibration method {method!r}.")
    y, p = _check(y, p)
    if len(np.unique(y)) < 2:
        raise CalibrationError("Calibration data must contain both classes.")
    if method == "sigmoid":
        estimator = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p), y)
    elif method == "isotonic":
        estimator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y)
    else:
        estimator = None
    return {"method": method, "estimator": estimator}


def apply_calibrator(calibrator: Optional[dict], p) -> np.ndarray:
    _, p = _check(None, p)
    if calibrator is None or calibrator["method"] == "none":
        return p
    estimator = calibrator["estimator"]
    if calibrator["method"] == "sigmoid":
        out = estimator.predict_proba(_logit(p))[:, list(estimator.classes_).index(1)]
    else:
        out = estimator.predict(p)
    return np.clip(np.asarray(out, dtype=float), 0.0, 1.0)


def check_calibrator(calibrator: dict) -> list[str]:
    """Sanity checks a calibrator must pass before it is saved or used."""
    grid = np.linspace(0, 1, 201)
    out = apply_calibrator(calibrator, grid)
    problems = []
    if not np.all(np.isfinite(out)) or np.any((out < 0) | (out > 1)):
        problems.append("outputs outside [0, 1]")
    if np.any(np.diff(out) < -1e-12):
        problems.append("not monotonic non-decreasing, so it would reorder customers")
    if calibrator["method"] == "sigmoid" and calibrator["estimator"].coef_.ravel()[0] <= 0:
        problems.append("non-positive slope")
    return problems


# --------------------------------------------------------------------------- #
# Probability quality
# --------------------------------------------------------------------------- #
def reliability_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Equal-width bins on [0, 1]; the last bin includes 1.0. Empty bins have NaN rates."""
    y, p = _check(y, p)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        rows.append({"bin": b, "lower": edges[b], "upper": edges[b + 1], "n": n,
                     "mean_predicted": float(p[mask].mean()) if n else np.nan,
                     "observed_rate": float(y[mask].mean()) if n else np.nan})
    return pd.DataFrame(rows)


def expected_calibration_error(table: pd.DataFrame) -> float:
    used = table[table["n"] > 0]
    total = used["n"].sum()
    return float((used["n"] / total * (used["observed_rate"] - used["mean_predicted"]).abs()).sum())


def probability_metrics(y, p, n_bins: int = 10) -> dict:
    y, p = _check(y, p)
    both = len(np.unique(y)) == 2
    return {
        "n_samples": int(len(y)), "n_positive": int(y.sum()),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, EPS, 1 - EPS), labels=[0, 1])),
        "ece": expected_calibration_error(reliability_table(y, p, n_bins)),
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "pr_auc": float(average_precision_score(y, p)) if both else None,
    }


def cross_validate_methods(y_val, p_val, *, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    """Out-of-fold Brier score and log loss of each method, using validation data only."""
    y, p = _check(y_val, p_val)
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = {m: np.empty_like(p) for m in CALIBRATION_METHODS}
    for fit_idx, eval_idx in folds.split(p.reshape(-1, 1), y):
        for method in CALIBRATION_METHODS:
            calibrator = fit_calibrator(method, y[fit_idx], p[fit_idx])
            oof[method][eval_idx] = apply_calibrator(calibrator, p[eval_idx])
    return pd.DataFrame([
        {"method": m, "brier": float(brier_score_loss(y, oof[m])),
         "log_loss": float(log_loss(y, np.clip(oof[m], EPS, 1 - EPS), labels=[0, 1])),
         "ece": expected_calibration_error(reliability_table(y, oof[m]))}
        for m in CALIBRATION_METHODS
    ])


def select_method(cv_table: pd.DataFrame) -> str:
    """Lowest out-of-fold Brier; ties favour the simpler method (order of CALIBRATION_METHODS)."""
    order = {m: i for i, m in enumerate(CALIBRATION_METHODS)}
    ranked = cv_table.assign(order=cv_table["method"].map(order)).sort_values(["brier", "order"])
    return str(ranked.iloc[0]["method"])


def select_and_fit(y_val, p_val, *, n_splits: int = 5, seed: int = 42):
    """Choose a method by cross-validation on validation data, then refit it on all of it."""
    cv_table = cross_validate_methods(y_val, p_val, n_splits=n_splits, seed=seed)
    method = select_method(cv_table)
    return fit_calibrator(method, y_val, p_val), cv_table


# --------------------------------------------------------------------------- #
# Artifact
# --------------------------------------------------------------------------- #
def load_calibration(meta_path: str, output_dir: str, model_sha256: str):
    """Return ``(metadata, calibrator_or_None)`` after integrity checks.

    The calibrator is ``None`` when the saved selection kept the uncalibrated scores.
    """
    import joblib

    if not os.path.exists(meta_path):
        raise CalibrationError(f"Calibration metadata not found: {meta_path}.")
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (OSError, ValueError) as exc:
        raise CalibrationError(f"Calibration metadata could not be read ({exc}).") from exc
    if not isinstance(meta, dict) or meta.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
        raise CalibrationError("Calibration metadata has an unsupported schema version.")
    if meta.get("model_sha256") != model_sha256:
        raise CalibrationError("Calibration was fitted for a different model file (stale).")
    method = meta.get("method")
    if method not in CALIBRATION_METHODS:
        raise CalibrationError(f"Unknown calibration method {method!r} in metadata.")
    if method == "none":
        return meta, None

    path = os.path.join(output_dir, meta.get("calibrator_file") or "")
    if not os.path.isfile(path):
        raise CalibrationError(f"Calibrator file not found: {path}.")
    if file_sha256(path) != meta.get("calibrator_sha256"):
        raise CalibrationError("Calibrator file does not match its metadata (SHA-256 mismatch).")
    calibrator = joblib.load(path)
    if not isinstance(calibrator, dict) or calibrator.get("method") != method:
        raise CalibrationError("Calibrator file content does not match the recorded method.")
    problems = check_calibrator(calibrator)
    if problems:
        raise CalibrationError(f"Calibrator failed its checks: {', '.join(problems)}.")
    return meta, calibrator
