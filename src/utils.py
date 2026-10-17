"""Data loading, duplicate handling, leakage-resistant splitting and JSON helpers."""
from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COL = "Churn"
ID_COL = "CustomerID"
DATA_SHEET = "E Comm"
NON_FEATURE_COLS = (ID_COL, TARGET_COL)

CONFLICT_POLICIES = ("keep_grouped", "drop", "error")


# --------------------------------------------------------------------------- #
# Loading and feature/target separation
# --------------------------------------------------------------------------- #
def load_raw_dataset(path: str, sheet_name: str = DATA_SHEET) -> pd.DataFrame:
    """Read the customer table, keeping CustomerID so duplicates can be traced."""
    df = pd.read_excel(path, sheet_name=sheet_name)
    df.columns = [str(c).strip() for c in df.columns]
    validate_target(df)
    return df


def load_dataset(path: str, sheet_name: str = DATA_SHEET) -> pd.DataFrame:
    """Read the customer table without the CustomerID column."""
    df = load_raw_dataset(path, sheet_name)
    return df.drop(columns=[ID_COL], errors="ignore")


def validate_target(df: pd.DataFrame) -> None:
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found. Columns: {list(df.columns)}")
    target = df[TARGET_COL]
    if target.isna().any():
        raise ValueError(f"Target column '{TARGET_COL}' has {int(target.isna().sum())} missing values.")
    values = set(pd.unique(target))
    if not values <= {0, 1}:
        raise ValueError(f"Target column '{TARGET_COL}' must contain only 0/1, found {sorted(values)}.")


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def separate_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    validate_target(df)
    X = df[feature_columns(df)]
    y = df[TARGET_COL].astype(int)
    return X, y


# --------------------------------------------------------------------------- #
# Duplicate handling
# --------------------------------------------------------------------------- #
def feature_row_hash(X: pd.DataFrame, columns: Optional[Iterable[str]] = None) -> np.ndarray:
    """Stable per-row hash of feature values. Missing values compare equal."""
    cols = list(columns) if columns is not None else list(X.columns)
    return pd.util.hash_pandas_object(X[cols], index=False).to_numpy()


def audit_duplicates(df: pd.DataFrame) -> dict:
    """Describe duplicate feature records (CustomerID and Churn excluded from identity)."""
    feats = feature_columns(df)
    groups = df.groupby(feats, dropna=False, sort=False)
    sizes = groups.size()
    label_counts = groups[TARGET_COL].nunique()
    conflicting = label_counts[label_counts > 1]

    report = {
        "feature_identity_columns": feats,
        "excluded_from_identity": [c for c in NON_FEATURE_COLS if c in df.columns],
        "missing_values_compare_equal": True,
        "n_rows": int(len(df)),
        "n_unique_feature_records": int(len(sizes)),
        "n_rows_in_duplicate_groups": int(sizes[sizes > 1].sum()),
        "n_duplicate_groups": int((sizes > 1).sum()),
        "group_size_distribution": {int(k): int(v) for k, v in sizes.value_counts().sort_index().items()},
        "n_redundant_identical_records": int(df.duplicated(feats + [TARGET_COL]).sum()),
        "n_conflicting_label_groups": int(len(conflicting)),
        "n_rows_in_conflicting_groups": int(sizes.loc[conflicting.index].sum()) if len(conflicting) else 0,
    }
    if ID_COL in df.columns:
        report["customer_id_unique"] = bool(df[ID_COL].is_unique)
        report["id_gap_within_duplicate_groups"] = _id_gap_summary(df, feats)
    return report


def _id_gap_summary(df: pd.DataFrame, feats: list[str]) -> dict:
    dup = df[df.duplicated(feats, keep=False)]
    if dup.empty:
        return {"most_common_gap": None, "groups_with_most_common_gap": 0}
    gid = dup.groupby(feats, dropna=False, sort=False).ngroup()
    gaps = Counter()
    for _, ids in dup[ID_COL].groupby(gid):
        gaps.update(np.diff(np.sort(ids.to_numpy())).tolist())
    gap, count = gaps.most_common(1)[0]
    return {"most_common_gap": int(gap), "groups_with_most_common_gap": int(count),
            "n_distinct_gaps": int(len(gaps))}


def deduplicate(df: pd.DataFrame, conflict_policy: str = "keep_grouped"):
    """Collapse identical (features + label) records and apply a conflict policy.

    Records with identical features *and* identical labels are collapsed to the
    record with the lowest CustomerID (original order if no ID column).

    Groups with identical features but conflicting labels are handled by
    ``conflict_policy``:
      * ``keep_grouped``: keep one record per distinct label; the split keeps the
        group inside a single partition.
      * ``drop``: remove every record of the group (listed in the removal log).
      * ``error``: raise ``ValueError``.

    Returns ``(clean_df, report, removed_df)``. ``removed_df`` lists every removed
    record with the reason and the CustomerID that was kept in its place.
    """
    if conflict_policy not in CONFLICT_POLICIES:
        raise ValueError(f"conflict_policy must be one of {CONFLICT_POLICIES}, got {conflict_policy!r}")

    validate_target(df)
    feats = feature_columns(df)
    report = audit_duplicates(df)
    report["conflict_policy"] = conflict_policy

    has_id = ID_COL in df.columns
    ordered = df.sort_values(ID_COL, kind="mergesort") if has_id else df
    identical_mask = ordered.duplicated(feats + [TARGET_COL], keep="first")
    removed_identical = ordered.index[identical_mask.to_numpy()]

    removed = df.loc[removed_identical].copy()
    removed["removal_reason"] = "identical_features_and_label"
    if has_id:
        record_group = df.groupby(feats + [TARGET_COL], dropna=False, sort=False).ngroup()
        kept_id_by_group = df[ID_COL].groupby(record_group).min()
        removed["kept_customer_id"] = record_group.loc[removed_identical].map(kept_id_by_group).to_numpy()
    clean = df.drop(index=removed_identical)

    conflict_mask = clean.duplicated(feats, keep=False)
    if conflict_mask.any():
        if conflict_policy == "error":
            raise ValueError(
                f"{int(conflict_mask.sum())} records share identical features but have conflicting "
                f"'{TARGET_COL}' labels. Choose conflict_policy='keep_grouped' or 'drop'."
            )
        if conflict_policy == "drop":
            dropped = clean[conflict_mask].copy()
            dropped["removal_reason"] = "conflicting_label_group"
            removed = pd.concat([removed, dropped])
            clean = clean[~conflict_mask]

    report["n_removed_identical_records"] = int(len(removed_identical))
    report["n_removed_conflicting_records"] = int((removed["removal_reason"] == "conflicting_label_group").sum())
    report["n_rows_after_deduplication"] = int(len(clean))
    report["churn_rate_before"] = float(df[TARGET_COL].mean())
    report["churn_rate_after"] = float(clean[TARGET_COL].mean()) if len(clean) else None
    return clean, report, removed


# --------------------------------------------------------------------------- #
# Leakage-resistant splitting
# --------------------------------------------------------------------------- #
def grouped_stratified_split(y: pd.Series, groups: np.ndarray, test_size: float, seed: int):
    """Split rows so every group lands in one partition, stratifying on group label.

    A group's label is positive if any of its records is positive. Returns a
    boolean mask selecting the test rows and whether stratification was applied.
    """
    group_label = pd.Series(np.asarray(y), index=groups).groupby(level=0).max()
    counts = group_label.value_counts()
    stratify = group_label.to_numpy() if len(counts) > 1 and counts.min() >= 2 else None
    _, test_groups = train_test_split(
        group_label.index.to_numpy(), test_size=test_size, stratify=stratify, random_state=seed
    )
    return np.isin(groups, test_groups), stratify is not None


def split_xy(df: pd.DataFrame, test_size: float, val_size: float, seed: int):
    """Train/validation/test split that keeps identical feature records together.

    ``val_size`` is a fraction of the data remaining after the test split.
    Returns ``X_train, X_val, X_test, y_train, y_val, y_test``.
    """
    X, y = separate_features_target(df)
    groups = feature_row_hash(X)

    test_mask, _ = grouped_stratified_split(y, groups, test_size, seed)
    X_tmp, y_tmp, g_tmp = X[~test_mask], y[~test_mask], groups[~test_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    val_mask, _ = grouped_stratified_split(y_tmp, g_tmp, val_size, seed)
    X_train, y_train = X_tmp[~val_mask], y_tmp[~val_mask]
    X_val, y_val = X_tmp[val_mask], y_tmp[val_mask]
    return X_train, X_val, X_test, y_train, y_val, y_test


def count_shared_feature_records(X_a: pd.DataFrame, X_b: pd.DataFrame) -> int:
    """Number of rows in ``X_b`` whose feature values also appear in ``X_a``."""
    cols = list(X_a.columns)
    return int(np.isin(feature_row_hash(X_b, cols), feature_row_hash(X_a, cols)).sum())


# --------------------------------------------------------------------------- #
# JSON helpers
# --------------------------------------------------------------------------- #
def to_jsonable(obj):
    """Recursively convert numpy/pandas scalars and containers to JSON types."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if obj is None or isinstance(obj, (str, int)):
        return obj
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return str(obj)


def default_file_mode() -> int:
    umask = os.umask(0)
    os.umask(umask)
    return 0o666 & ~umask


def write_json_atomic(data: dict, path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(to_jsonable(data), f, indent=2)
        os.chmod(tmp, default_file_mode())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
