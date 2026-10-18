"""Explanations for the fitted churn pipeline.

* Permutation importance on raw input features (the pipeline handles encoding).
* Impurity importance of the tree model, with one-hot columns summed back to
  their source feature.
* Per-customer interventional Shapley values estimated by antithetic permutation
  sampling over a background sample. Contributions are on the model's
  (uncalibrated) churn-score scale and sum exactly to
  ``prediction - reference value``.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import get_scorer

from inference import FeatureSchema, positive_class_proba, prepare_features

EXPLAIN_SCHEMA_VERSION = 1
MAX_EXPLAINED_CUSTOMERS = 5
DEFAULT_PERMUTATIONS = 8
MAX_HYBRID_ROWS = 250_000


class ExplanationError(ValueError):
    """Raised when an explanation cannot be computed or an artifact is unusable."""


# --------------------------------------------------------------------------- #
# Global importance
# --------------------------------------------------------------------------- #
def transformed_feature_sources(model) -> list[str]:
    """Raw feature name for each column the final estimator receives."""
    pre = model.named_steps["pre"]
    sources = []
    for name, transformer, cols in pre.transformers_:
        if transformer == "drop" or len(cols) == 0:
            continue
        if transformer == "passthrough" or not hasattr(transformer, "named_steps") \
                or "encoder" not in transformer.named_steps:
            sources += [str(c) for c in cols]
            continue
        encoder = transformer.named_steps["encoder"]
        drop_idx = getattr(encoder, "drop_idx_", None)
        for i, (col, cats) in enumerate(zip(cols, encoder.categories_)):
            dropped = drop_idx is not None and drop_idx[i] is not None
            sources += [str(col)] * (len(cats) - int(dropped))
    n_in = getattr(model.named_steps["clf"], "n_features_in_", len(sources))
    if len(sources) != n_in:
        raise ExplanationError(f"Mapped {len(sources)} model inputs to raw features, but the estimator "
                               f"expects {n_in}; the preprocessing layout is not supported.")
    return sources


def impurity_importance(model) -> Optional[pd.Series]:
    """Impurity importance per raw feature (one-hot columns summed), or ``None`` if unavailable."""
    clf = model.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return None
    sources = transformed_feature_sources(model)
    return pd.Series(clf.feature_importances_, index=sources).groupby(level=0, sort=False).sum()


def permutation_importance_table(model, X: pd.DataFrame, y, *, scoring: str = "average_precision",
                                 n_repeats: int = 10, seed: int = 42, n_jobs=None):
    """Return ``(table, baseline_score)``; importance = drop in ``scoring`` when a feature is shuffled."""
    y = pd.Series(np.asarray(y))
    if len(X) == 0 or len(X) != len(y):
        raise ExplanationError("Permutation importance needs a non-empty, aligned dataset.")
    if y.nunique() < 2:
        raise ExplanationError("Permutation importance needs both classes in the evaluation data.")
    if n_repeats < 2:
        raise ExplanationError("Use at least two permutation repeats to estimate variability.")
    baseline = float(get_scorer(scoring)(model, X, y))
    result = permutation_importance(model, X, y, scoring=scoring, n_repeats=n_repeats,
                                    random_state=seed, n_jobs=n_jobs)
    table = pd.DataFrame({
        "feature": [str(c) for c in X.columns],
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
        "importance_min": result.importances.min(axis=1),
        "importance_max": result.importances.max(axis=1),
    }).sort_values("importance_mean", ascending=False, kind="mergesort").reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1)
    return table, baseline


def compare_importances(permutation: pd.DataFrame, impurity: Optional[pd.Series]) -> Optional[pd.DataFrame]:
    """Side-by-side ranks; ``None`` if impurity importance is unavailable or names do not match."""
    if impurity is None or set(impurity.index) != set(permutation["feature"]):
        return None
    table = permutation[["feature", "importance_mean", "rank"]].rename(
        columns={"importance_mean": "permutation", "rank": "permutation_rank"})
    table["impurity"] = table["feature"].map(impurity)
    table["impurity_rank"] = table["impurity"].rank(ascending=False, method="min").astype(int)
    return table


def rank_correlation(comparison: Optional[pd.DataFrame]) -> Optional[float]:
    if comparison is None or len(comparison) < 3:
        return None
    return float(comparison["permutation"].rank().corr(comparison["impurity"].rank()))


# --------------------------------------------------------------------------- #
# Per-customer Shapley values
# --------------------------------------------------------------------------- #
@dataclass
class CustomerExplanation:
    features: list[str]
    feature_values: list
    contributions: np.ndarray
    std_errors: np.ndarray
    base_value: float
    prediction: float
    model_prediction: float
    n_samples: int

    @property
    def additivity_error(self) -> float:
        return abs(self.base_value + float(self.contributions.sum()) - self.prediction)

    def is_consistent(self, tolerance: float = 1e-6) -> bool:
        return (self.additivity_error <= tolerance
                and abs(self.prediction - self.model_prediction) <= tolerance)

    def to_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame({
            "feature": self.features,
            "value": self.feature_values,
            "contribution": self.contributions,
            "std_error": self.std_errors,
        })
        order = np.argsort(-np.abs(frame["contribution"].to_numpy()), kind="mergesort")
        return frame.iloc[order].reset_index(drop=True)


def shapley_values(model, x_row: pd.DataFrame, background: pd.DataFrame, *,
                   n_permutations: int = DEFAULT_PERMUTATIONS, seed: int = 0) -> CustomerExplanation:
    """Interventional Shapley values of the positive-class probability for one prepared row.

    For each background row, ``n_permutations`` feature orderings (antithetic
    pairs) are sampled; features are switched from the background value to the
    customer's value one at a time and the change in churn score is credited to
    the switched feature. Each sample's credits sum to ``f(x) - f(background)``,
    so the averages add up exactly to ``prediction - base_value``.
    """
    if not isinstance(x_row, pd.DataFrame) or len(x_row) != 1:
        raise ExplanationError("Exactly one customer row is required.")
    if background is None or len(background) == 0:
        raise ExplanationError("The background sample is empty.")
    if n_permutations < 2 or n_permutations % 2:
        raise ExplanationError("n_permutations must be an even number >= 2.")
    features = list(x_row.columns)
    missing = [c for c in features if c not in background.columns]
    if missing:
        raise ExplanationError(f"Background sample lacks feature(s): {', '.join(missing)}.")
    background = background[features]

    d, n_bg = len(features), len(background)
    n_samples = n_bg * n_permutations
    if n_samples * (d + 1) > MAX_HYBRID_ROWS:
        raise ExplanationError("Explanation request is too large; reduce the background or permutations.")

    rng = np.random.default_rng(seed)
    perms = np.empty((n_samples, d), dtype=int)
    for s in range(0, n_samples, 2):
        perms[s] = rng.permutation(d)
        perms[s + 1] = perms[s][::-1]
    bg_index = np.repeat(np.arange(n_bg), n_permutations)

    x = x_row.to_numpy(dtype=object)[0]
    Z = background.to_numpy(dtype=object)[bg_index]
    position = np.argsort(perms, axis=1)
    steps = np.arange(d + 1)[None, :, None]
    use_x = position[:, None, :] < steps
    hybrid = np.where(use_x, x[None, None, :], Z[:, None, :]).reshape(-1, d)
    numeric = pd.api.types.is_numeric_dtype
    dtypes = {c: float if numeric(x_row[c]) and numeric(background[c]) else object for c in features}
    frame = pd.DataFrame(hybrid, columns=features).astype(dtypes)

    try:
        scores = positive_class_proba(model, frame).reshape(n_samples, d + 1)
        model_prediction = float(positive_class_proba(model, x_row)[0])
    except Exception as exc:
        raise ExplanationError(f"The model could not score the explanation inputs ({exc}).") from exc

    contributions = np.empty((n_samples, d))
    np.put_along_axis(contributions, perms, np.diff(scores, axis=1), axis=1)
    # The background rows are fixed (they define the reference value), so the only
    # Monte Carlo noise is the choice of orderings within each background row.
    n_pairs = n_permutations // 2
    pair_means = contributions.reshape(n_bg, n_pairs, 2, d).mean(axis=2)
    if n_pairs > 1:
        within_var = pair_means.var(axis=1, ddof=1).sum(axis=0)
        std_errors = np.sqrt(within_var / n_pairs) / n_bg
    else:
        std_errors = np.full(d, np.nan)
    values = [None if pd.isna(v) else v for v in x]
    return CustomerExplanation(
        features=features, feature_values=values, contributions=contributions.mean(axis=0),
        std_errors=std_errors, base_value=float(scores[:, 0].mean()), prediction=float(scores[0, -1]),
        model_prediction=model_prediction, n_samples=n_samples,
    )


def explain_customers(model, rows: pd.DataFrame, background: pd.DataFrame, *,
                      limit: int = MAX_EXPLAINED_CUSTOMERS, **kwargs) -> list[CustomerExplanation]:
    if len(rows) > limit:
        raise ExplanationError(f"At most {limit} customers can be explained at once.")
    return [shapley_values(model, rows.iloc[[i]], background, **kwargs) for i in range(len(rows))]


# --------------------------------------------------------------------------- #
# Artifact
# --------------------------------------------------------------------------- #
def load_explainability(path: str, model_sha256: str) -> dict:
    """Load ``explainability.json`` and check it belongs to the loaded model."""
    if not os.path.exists(path):
        raise ExplanationError(f"Explainability artifact not found: {path}.")
    try:
        with open(path) as f:
            meta = json.load(f)
    except (OSError, ValueError) as exc:
        raise ExplanationError(f"Explainability artifact could not be read ({exc}).") from exc
    if not isinstance(meta, dict) or meta.get("schema_version") != EXPLAIN_SCHEMA_VERSION:
        raise ExplanationError("Explainability artifact has an unsupported schema version.")
    if meta.get("model_sha256") != model_sha256:
        raise ExplanationError("Explainability artifact was computed for a different model file (stale).")
    for key in ("permutation_importance", "background"):
        if key not in meta:
            raise ExplanationError(f"Explainability artifact has no '{key}' section.")
    return meta


def background_frame(meta: dict, schema: FeatureSchema) -> pd.DataFrame:
    records = meta["background"].get("records") or []
    if not records:
        raise ExplanationError("The saved background sample is empty.")
    X, _ = prepare_features(pd.DataFrame.from_records(records), schema)
    return X
