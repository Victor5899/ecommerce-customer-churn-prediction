"""Dataset compatibility checks for uploaded customer files.

An upload is classified as ready to score, needing confirmed mapping or
preprocessing, or incompatible. Changes are proposed only when they can be
justified without guessing:

* column renames when a name matches a required column apart from
  capitalisation, spacing and punctuation, or is a listed spelling variant of it;
* yes/no or true/false text converted to 1/0 in the binary ``Complain`` and ``Churn`` columns;
* category values that match a known training category apart from capitalisation.

Differently named columns are never matched on similarity, number formats are
never guessed, and missing values are never filled here: partially missing
values are imputed by the fitted pipeline exactly as before. Nothing is applied
without the user's confirmation.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

import dashboard as db
from inference import OUTPUT_COLS, FeatureSchema, coerce_numeric
from utils import ID_COL, TARGET_COL

READY, NEEDS_MAPPING, INCOMPATIBLE = "Ready to score", "Needs mapping/preprocessing", "Incompatible"
OPTIONAL_COLUMNS = (ID_COL, TARGET_COL)
BINARY_COLUMNS = ("Complain", TARGET_COL)
TRUE_VALUES = {"1", "1.0", "yes", "y", "true"}
FALSE_VALUES = {"0", "0.0", "no", "n", "false"}
# Spelling variants of the same feature, in normalised form (lower case, letters and digits only).
NAME_VARIANTS = {
    "PreferedOrderCat": {"preferredordercat", "preferedordercategory", "preferredordercategory"},
    "DaySinceLastOrder": {"dayssincelastorder"},
    "NumberOfDeviceRegistered": {"numberofdevicesregistered"},
    "NumberOfAddress": {"numberofaddresses"},
    "HourSpendOnApp": {"hoursspendonapp", "hourspentonapp", "hoursspentonapp"},
}


def normalise_name(name) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


@dataclass(frozen=True)
class Fix:
    kind: str                       # "rename", "binary" or "category_case"
    column: str                     # target column name (after renaming)
    detail: str
    source: Optional[str] = None    # original column name, for renames
    mapping: tuple = ()             # (original value, new value) pairs, for value fixes


@dataclass
class CompatibilityReport:
    status: str = READY
    n_rows: int = 0
    missing: list = field(default_factory=list)
    extra: list = field(default_factory=list)
    duplicates: list = field(default_factory=list)
    problems: list = field(default_factory=list)   # blocking
    warnings: list = field(default_factory=list)   # data quality, not blocking
    hints: list = field(default_factory=list)      # similar names that were deliberately not mapped
    fixes: list = field(default_factory=list)      # need confirmation before they are applied
    has_target: bool = False
    target_usable: bool = False


def check_compatibility(df: Optional[pd.DataFrame], schema: FeatureSchema,
                        raw_names: Optional[list] = None) -> CompatibilityReport:
    """Check an upload against the model schema.

    ``raw_names`` is the header as written (see ``inference.read_column_names``);
    it is needed to detect repeated names that pandas renames on reading. With
    ``df=None`` only the header is checked.
    """
    report = CompatibilityReport()
    required = list(schema.features)
    known = set(required) | set(OPTIONAL_COLUMNS)
    names = [str(c).strip() for c in (raw_names if raw_names is not None
                                      else ([] if df is None else df.columns))]
    _check_duplicates(report, names, known)
    if df is None:
        return _finish(report)

    report.n_rows = len(df)
    if df.empty:
        report.problems.append("The file contains no customer records.")
        return _finish(report)

    renames = _propose_renames(report, list(df.columns), required, known)
    view = df.rename(columns=renames)
    report.extra = [c for c in view.columns if c not in known and c not in OUTPUT_COLS]
    _check_numeric(report, view, schema)
    _check_categories(report, view, schema)
    _check_target(report, view)
    _check_quality(report, view, [c for c in required if c in view.columns])
    return _finish(report)


def _finish(report: CompatibilityReport) -> CompatibilityReport:
    report.status = INCOMPATIBLE if report.problems else NEEDS_MAPPING if report.fixes else READY
    return report


def _check_duplicates(report, names, known) -> None:
    counts = pd.Series(names, dtype=object).value_counts()
    known_norm = {normalise_name(k) for k in known}
    for name, count in sorted(counts[counts > 1].items()):
        report.duplicates.append(name)
        if name in known or normalise_name(name) in known_norm:
            report.problems.append(
                f"Duplicate column '{name}' appears {count} times. The checker cannot tell which copy "
                "holds the correct values; remove or rename the extra copies.")
        else:
            report.warnings.append(f"Extra column '{name}' appears {count} times; it is not used for prediction.")


def _propose_renames(report, columns, required, known) -> dict:
    renames = {}
    unclaimed = [c for c in columns if c not in known and c not in OUTPUT_COLS]
    for target in required + list(OPTIONAL_COLUMNS):
        if target in columns:
            continue
        keys = {normalise_name(target)} | NAME_VARIANTS.get(target, set())
        candidates = [c for c in unclaimed if normalise_name(c) in keys and c not in renames]
        if len(candidates) == 1:
            source = candidates[0]
            why = ("differs only in capitalisation, spacing or punctuation"
                   if normalise_name(source) == normalise_name(target) else "is a known spelling variant")
            renames[source] = target
            report.fixes.append(Fix("rename", target, f"Rename '{source}' to '{target}' (the name {why}).",
                                    source=source))
        elif len(candidates) > 1:
            message = (f"Several columns could be '{target}' ({', '.join(candidates)}); none is mapped "
                       "because the right one cannot be determined from the file.")
            (report.problems if target in required else report.warnings).append(message)
        elif target in required:
            report.missing.append(target)
            close = difflib.get_close_matches(normalise_name(target),
                                              [normalise_name(c) for c in unclaimed], n=2, cutoff=0.75)
            similar = [c for c in unclaimed if normalise_name(c) in close]
            if similar:
                report.hints.append(
                    f"'{target}' is missing. Similar column name(s) {', '.join(repr(c) for c in similar)} were "
                    "not mapped: a similar name does not show that a column holds the same data. If it "
                    "does, rename it in your file.")
    if report.missing:
        report.problems.insert(0, f"Missing required column(s): {', '.join(report.missing)}. No column with "
                                  "an equivalent name was found, and missing features are never filled in.")
    return renames


def _binary_fix(values: pd.Series, col: str) -> Optional[Fix]:
    observed = values.dropna()
    text = observed.astype(str).str.strip()
    if text.empty or not text.str.lower().isin(TRUE_VALUES | FALSE_VALUES).all():
        return None
    mapping = tuple((v, int(v.lower() in TRUE_VALUES)) for v in sorted(set(text)))
    shown = ", ".join(f"'{v}' → {n}" for v, n in mapping)
    return Fix("binary", col, f"Convert '{col}' to 0/1: {shown}. Missing values stay missing.", mapping=mapping)


def _check_numeric(report, view, schema) -> None:
    for col in schema.numeric:
        if col not in view.columns:
            continue
        coerced, problem = coerce_numeric(view[col], col)
        if problem is None:
            if (coerced < 0).any():
                report.warnings.append(f"'{col}' has {int((coerced < 0).sum())} negative value(s); the training "
                                       "data has none, so these predictions may be less reliable.")
            continue
        fix = _binary_fix(view[col], col) if col in BINARY_COLUMNS else None
        if fix:
            report.fixes.append(fix)
        else:
            report.problems.append(f"Invalid type: {problem} A number is required, and number formats "
                                   "(such as thousands separators or units) are not guessed.")


def _check_categories(report, view, schema) -> None:
    for col in schema.categorical:
        if col not in view.columns:
            continue
        known = schema.categories.get(col, [])
        by_lower = {}
        for category in known:
            by_lower.setdefault(category.lower(), []).append(category)
        text = view[col].dropna().astype(str).str.strip()
        unseen = sorted(set(text) - set(known) - {""})
        mapping = {v: by_lower[v.lower()][0] for v in unseen if len(by_lower.get(v.lower(), [])) == 1}
        if mapping:
            shown = ", ".join(f"'{a}' → '{b}'" for a, b in mapping.items())
            report.fixes.append(Fix("category_case", col, f"Match '{col}' values that differ from a known "
                                    f"category only in capitalisation: {shown}.", mapping=tuple(mapping.items())))
        rest = [v for v in unseen if v not in mapping]
        if rest:
            report.warnings.append(
                f"'{col}' has value(s) not seen in training ({', '.join(rest[:5])}); they are scored as none "
                "of the known categories, so these predictions may be less reliable.")


def _check_target(report, view) -> None:
    if TARGET_COL not in view.columns:
        return
    report.has_target = True
    report.target_usable = db.labelled_target(view) is not None
    if report.target_usable:
        return
    fix = None
    if coerce_numeric(view[TARGET_COL], TARGET_COL)[1] is not None:
        fix = _binary_fix(view[TARGET_COL], TARGET_COL)
    if fix and view[TARGET_COL].notna().all():
        report.fixes.append(fix)
    else:
        report.warnings.append(f"'{TARGET_COL}' is present but is not a complete 0/1 column, so it will not be "
                               "used to report actual churn. It is never used for prediction.")


def _check_quality(report, view, present) -> None:
    quality = db.upload_quality(view, present)
    empty = [c for c in present if view[c].isna().all()]
    if empty:
        report.warnings.append(f"Column(s) with no values: {', '.join(empty)}. Every value would be filled "
                               "from the training data, so these features say nothing about these customers.")
    if quality["rows_with_missing"]:
        report.warnings.append(f"{quality['rows_with_missing']:,} row(s) have missing feature values; the model "
                               "pipeline fills them from the training data (median or most frequent value).")
    all_missing = int(view[present].isna().all(axis=1).sum()) if present else 0
    if all_missing:
        report.warnings.append(f"{all_missing:,} row(s) have no feature values at all; their scores would be "
                               "based only on filled-in values.")
    if quality["exact_duplicate_rows"]:
        report.warnings.append(f"{quality['exact_duplicate_rows']:,} row(s) are exact copies of an earlier row.")
    if quality["has_ids"]:
        if quality["missing_ids"]:
            report.warnings.append(f"{quality['missing_ids']:,} row(s) have no `{ID_COL}`.")
        if quality["duplicate_ids"]:
            report.warnings.append(f"{quality['duplicate_ids']:,} row(s) share a `{ID_COL}` with another row.")


def apply_fixes(df: pd.DataFrame, fixes: list) -> pd.DataFrame:
    """Apply confirmed fixes to a copy. Missing values are left missing."""
    out = df.rename(columns={f.source: f.column for f in fixes if f.kind == "rename"})
    for f in fixes:
        lookup = dict(f.mapping)
        if f.kind == "binary":
            out[f.column] = out[f.column].map(
                lambda v: lookup[str(v).strip()] if pd.notna(v) else np.nan).astype(float)
        elif f.kind == "category_case":
            out[f.column] = out[f.column].map(lambda v: lookup.get(str(v).strip(), v) if pd.notna(v) else v)
    return out


def preview_fixes(df: pd.DataFrame, fixes: list, n: int = 5) -> pd.DataFrame:
    """Before/after values of the affected columns, preferring rows whose values change."""
    after = apply_fixes(df, fixes)
    sources = {f.column: f.source for f in fixes if f.kind == "rename"}
    changed = pd.Series(False, index=df.index)
    for f in fixes:
        if f.kind != "rename":
            before = df[sources.get(f.column, f.column)]
            changed |= before.notna() & (before.astype(str) != after[f.column].astype(str))
    rows = (df.index[changed] if changed.any() else df.index)[:n]
    frame = {}
    for f in fixes:
        source = sources.get(f.column, f.column)
        frame[f"{source} (before)"] = df.loc[rows, source]
        frame[f"{f.column} (after)"] = after.loc[rows, f.column]
    return pd.DataFrame(frame)
