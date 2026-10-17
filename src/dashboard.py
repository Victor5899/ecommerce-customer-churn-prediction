"""Streamlit-free calculations behind the analytics dashboard.

Historical (actual) churn is only ever computed from a complete 0/1 ``Churn``
column; model predictions are summarised separately and never used as outcomes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from inference import PREDICTION_COL, PROBABILITY_COL
from utils import ID_COL, TARGET_COL

MISSING_LABEL = "Missing"
CHURN_LABELS = {0: "Retained", 1: "Churned"}
RISK_COL = "Risk Segment"
RISK_SEGMENTS = ("Low", "Medium", "High")
MIN_SEGMENT_SIZE = 30
DISCRETE_MAX_VALUES = 15
NUMERIC_BINS = 10

SEGMENT_BINS = {
    "Tenure": ([-np.inf, 2, 6, 12, 24, np.inf],
               ["0–1 months", "2–5 months", "6–11 months", "12–23 months", "24+ months"]),
    "OrderCount": ([-np.inf, 2, 3, 6, np.inf], ["≤1 order", "2 orders", "3–5 orders", "6+ orders"]),
    "DaySinceLastOrder": ([-np.inf, 2, 5, 10, np.inf], ["0–1 days", "2–4 days", "5–9 days", "10+ days"]),
}
VALUE_LABELS = {"Complain": {0: "No complaint", 1: "Complained"}}
OVERVIEW_BREAKDOWNS = {
    "Tenure": "tenure",
    "SatisfactionScore": "satisfaction score",
    "Complain": "complaint status",
    "OrderCount": "order count",
}
# Category pairs that look like alternative spellings of the same value.
CATEGORY_ALIASES = {
    "PreferredLoginDevice": [("Phone", "Mobile Phone")],
    "PreferredPaymentMode": [("CC", "Credit Card"), ("COD", "Cash on Delivery")],
    "PreferedOrderCat": [("Mobile", "Mobile Phone")],
}


# --------------------------------------------------------------------------- #
# Actual and predicted churn
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ChurnSummary:
    n_customers: int
    n_churned: int

    @property
    def rate(self) -> float:
        return self.n_churned / self.n_customers if self.n_customers else float("nan")


def labelled_target(df: pd.DataFrame) -> Optional[pd.Series]:
    """Return ``Churn`` as 0/1 integers if every row has a valid label, else ``None``."""
    if TARGET_COL not in df.columns or df.empty:
        return None
    y = pd.to_numeric(df[TARGET_COL], errors="coerce")
    if y.isna().any() or not y.isin([0, 1]).all():
        return None
    return y.astype(int)


def actual_churn(df: pd.DataFrame) -> ChurnSummary:
    y = labelled_target(df)
    if y is None:
        raise ValueError(f"Actual churn requires a complete 0/1 '{TARGET_COL}' column.")
    return ChurnSummary(len(y), int(y.sum()))


@dataclass(frozen=True)
class BatchSummary:
    n_customers: int
    n_predicted_churn: Optional[int]
    mean_probability: float
    actual: Optional[ChurnSummary]

    @property
    def predicted_rate(self) -> Optional[float]:
        if self.n_predicted_churn is None or not self.n_customers:
            return None
        return self.n_predicted_churn / self.n_customers


def batch_summary(results: pd.DataFrame) -> BatchSummary:
    """Summarise a scored upload. ``actual`` is set only if the file has complete labels."""
    n = len(results)
    predicted = int(results[PREDICTION_COL].sum()) if PREDICTION_COL in results.columns else None
    mean_p = float(results[PROBABILITY_COL].mean()) if n else float("nan")
    y = labelled_target(results)
    actual = ChurnSummary(n, int(y.sum())) if y is not None else None
    return BatchSummary(n, predicted, mean_p, actual)


# --------------------------------------------------------------------------- #
# Risk segments (presentation only; never changes the decision threshold)
# --------------------------------------------------------------------------- #
def validate_risk_bounds(low, high) -> tuple[float, float]:
    low, high = float(low), float(high)
    if not 0 < low < high < 100:
        raise ValueError(f"Risk boundaries must satisfy 0 < low < high < 100, got {low} and {high}.")
    return low, high


def default_risk_bounds(threshold: Optional[float]) -> tuple[int, int]:
    """Start the medium band at the decision threshold so 'Low' never contains predicted churners."""
    if threshold is None:
        return 30, 60
    low = int(min(max(round(threshold * 100), 1), 89))
    return low, max(60, low + 10)


def assign_risk_segments(probability_pct, low, high) -> pd.Series:
    """Low: p < low; Medium: low <= p < high; High: p >= high (p in percent)."""
    low, high = validate_risk_bounds(low, high)
    p = pd.to_numeric(pd.Series(probability_pct), errors="raise").astype(float)
    labels = np.select([p >= high, p >= low], ["High", "Medium"], default="Low")
    return pd.Series(pd.Categorical(labels, categories=RISK_SEGMENTS, ordered=True),
                     index=p.index, name=RISK_COL)


def risk_segment_counts(segments: pd.Series) -> pd.DataFrame:
    counts = segments.value_counts().reindex(list(RISK_SEGMENTS), fill_value=0)
    total = int(counts.sum())
    return pd.DataFrame({
        "Segment": list(RISK_SEGMENTS),
        "Customers": counts.to_numpy(dtype=int),
        "Share": counts.to_numpy(dtype=float) / total if total else 0.0,
    })


def filter_by_risk(results: pd.DataFrame, segments=None, min_probability: float = 0.0,
                   predicted_only: bool = False, sort: str = "desc") -> pd.DataFrame:
    """Filter scored rows; ``sort`` is 'desc', 'asc' or 'original' (stable for ties)."""
    mask = results[PROBABILITY_COL] >= min_probability
    if segments is not None:
        mask &= results[RISK_COL].isin(list(segments))
    if predicted_only and PREDICTION_COL in results.columns:
        mask &= results[PREDICTION_COL] == 1
    view = results[mask]
    if sort in ("desc", "asc"):
        view = view.sort_values(PROBABILITY_COL, ascending=sort == "asc", kind="mergesort")
    return view


# --------------------------------------------------------------------------- #
# Segmentation and churn breakdowns
# --------------------------------------------------------------------------- #
def segment_column(series: pd.Series) -> pd.Series:
    """Map a feature to ordered, readable segments; missing values become 'Missing'."""
    name = series.name
    numeric = pd.to_numeric(series, errors="coerce") if pd.api.types.is_numeric_dtype(series) else None
    if name in SEGMENT_BINS:
        edges, labels = SEGMENT_BINS[name]
        seg = pd.cut(pd.to_numeric(series, errors="coerce"), edges, labels=labels, right=False)
        categories = list(labels)
    elif name in VALUE_LABELS:
        mapping = VALUE_LABELS[name]
        seg = pd.to_numeric(series, errors="coerce").map(mapping)
        categories = list(mapping.values())
    elif numeric is not None and numeric.nunique() > DISCRETE_MAX_VALUES:
        lo, hi = np.floor(numeric.min()), np.ceil(numeric.max())
        edges = np.unique(np.linspace(lo, hi, NUMERIC_BINS + 1))
        categories = [f"{a:,.4g}–{b:,.4g}" for a, b in zip(edges[:-1], edges[1:])]
        seg = pd.cut(numeric, edges, labels=categories, include_lowest=True)
    elif numeric is not None:
        mapping = {v: f"{v:g}" for v in sorted(numeric.dropna().unique())}
        seg = numeric.map(mapping)
        categories = list(dict.fromkeys(mapping.values()))
    else:
        seg = series.where(series.isna(), series.astype(str).str.strip())
        categories = sorted(seg.dropna().unique())
    seg = seg.astype(object)
    seg = seg.where(seg.notna(), MISSING_LABEL)
    if (seg == MISSING_LABEL).any():
        categories.append(MISSING_LABEL)
    return pd.Series(pd.Categorical(seg, categories=categories, ordered=True),
                     index=series.index, name=name)


def churn_rate_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Actual churn rate per segment of ``column``. Requires complete labels."""
    y = labelled_target(df)
    if y is None:
        raise ValueError(f"Churn rates require a complete 0/1 '{TARGET_COL}' column.")
    frame = pd.DataFrame({"Segment": segment_column(df[column]), "Churned": y})
    table = (frame.groupby("Segment", observed=True)["Churned"]
             .agg(Customers="size", Churned="sum").reset_index())
    table["Churn Rate"] = table["Churned"] / table["Customers"]
    table["Share"] = table["Customers"] / len(df)
    return table


def feature_distribution(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Customer counts per segment, split by actual churn status when labels are available."""
    frame = pd.DataFrame({"Segment": segment_column(df[column])})
    keys = ["Segment"]
    y = labelled_target(df)
    if y is not None:
        frame["Status"] = y.map(CHURN_LABELS)
        keys.append("Status")
    return frame.groupby(keys, observed=True).size().reset_index(name="Customers")


def _extremes(table: pd.DataFrame, min_size: int):
    usable = table[(table["Customers"] >= min_size) & (table["Segment"].astype(str) != MISSING_LABEL)]
    if len(usable) < 2:
        return None
    ordered = usable.sort_values("Churn Rate", kind="mergesort")
    return ordered.iloc[-1], ordered.iloc[0]


def key_observations(df: pd.DataFrame, min_size: int = MIN_SEGMENT_SIZE) -> list[str]:
    """Short, data-driven statements about actual churn in a labelled dataset."""
    summary = actual_churn(df)
    notes = [f"{summary.n_churned:,} of {summary.n_customers:,} customer records "
             f"({summary.rate:.1%}) are labelled as churned."]
    for column, description in OVERVIEW_BREAKDOWNS.items():
        if column not in df.columns:
            continue
        extremes = _extremes(churn_rate_by(df, column), min_size)
        if extremes is None:
            continue
        high, low = extremes
        ratio = f" ({high['Churn Rate'] / low['Churn Rate']:.1f}× higher)" if low["Churn Rate"] > 0 else ""
        notes.append(
            f"By {description}, churn is highest for **{high['Segment']}** "
            f"({high['Churn Rate']:.1%} of {int(high['Customers']):,}) and lowest for "
            f"**{low['Segment']}** ({low['Churn Rate']:.1%} of {int(low['Customers']):,}){ratio}."
        )
    return notes


def filter_customers(df: pd.DataFrame, ranges: Optional[dict] = None,
                     selections: Optional[dict] = None, include_missing: bool = True) -> pd.DataFrame:
    """Keep rows inside numeric ``ranges`` {col: (lo, hi)} and category ``selections`` {col: [values]}."""
    mask = pd.Series(True, index=df.index)
    for column, (lo, hi) in (ranges or {}).items():
        values = pd.to_numeric(df[column], errors="coerce")
        mask &= values.between(lo, hi) | (values.isna() & include_missing)
    for column, allowed in (selections or {}).items():
        values = df[column]
        mask &= values.isin(list(allowed)) | (values.isna() & include_missing)
    return df[mask]


def profile_by_churn(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Median of each numeric feature for retained vs churned customers."""
    y = labelled_target(df)
    if y is None:
        raise ValueError(f"Churn profiles require a complete 0/1 '{TARGET_COL}' column.")
    numeric = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
    medians = df[numeric].groupby(y).median().T.rename(columns=CHURN_LABELS)
    medians["Missing %"] = df[numeric].isna().mean() * 100
    return medians.reindex(columns=["Retained", "Churned", "Missing %"]).rename_axis("Feature")


# --------------------------------------------------------------------------- #
# Data quality
# --------------------------------------------------------------------------- #
def missing_value_table(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.isna().sum()
    counts = counts[counts > 0].sort_values(ascending=False, kind="mergesort")
    return pd.DataFrame({"Column": counts.index, "Missing": counts.to_numpy(dtype=int),
                         "Missing %": counts.to_numpy(dtype=float) / max(len(df), 1) * 100})


def upload_quality(df: pd.DataFrame, features: list[str]) -> dict:
    """Data-quality findings for an uploaded file (independent of the training data)."""
    present = [c for c in features if c in df.columns]
    quality = {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "missing": missing_value_table(df[present]),
        "rows_with_missing": int(df[present].isna().any(axis=1).sum()),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "duplicate_feature_rows": int(df[present].duplicated().sum()) if present else 0,
        "has_ids": ID_COL in df.columns,
        "has_labels": labelled_target(df) is not None,
    }
    if quality["has_ids"]:
        ids = df[ID_COL]
        quality["missing_ids"] = int(ids.isna().sum())
        quality["duplicate_ids"] = int(ids[ids.notna()].duplicated(keep=False).sum())
    return quality


def category_aliases(categories: dict) -> list[tuple[str, str, str]]:
    """Return (column, a, b) for known alias pairs where both spellings occur."""
    found = []
    for column, pairs in CATEGORY_ALIASES.items():
        values = set(categories.get(column, []))
        found += [(column, a, b) for a, b in pairs if a in values and b in values]
    return found


def artifact_issues(output_dir, metadata: dict, model_path, tolerance_s: float = 3600) -> list[str]:
    """List saved artifacts that are missing or were written apart from the model file."""
    artifacts = metadata.get("artifacts") or {}
    names = [artifacts.get(k) for k in ("model_comparison", "data_quality_report", "removed_duplicates")]
    names = [n for n in names + list(artifacts.get("plots") or []) if n]
    model_mtime = os.path.getmtime(model_path) if os.path.exists(model_path) else None
    issues = []
    for name in names:
        path = os.path.join(output_dir, name)
        if not os.path.exists(path):
            issues.append(f"`{name}` is missing.")
        elif model_mtime is not None and abs(os.path.getmtime(path) - model_mtime) > tolerance_s:
            issues.append(f"`{name}` was written more than an hour apart from the model file and "
                          "may come from a different training run.")
    return issues
