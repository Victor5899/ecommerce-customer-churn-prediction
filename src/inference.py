"""File ingestion, input validation, threshold policy and batch prediction."""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from utils import DATA_SHEET, NON_FEATURE_COLS

METADATA_SCHEMA_VERSION = 2
SUPPORTED_EXTENSIONS = (".csv", ".xlsx")
PREDICTION_COL = "Churn Prediction"
PROBABILITY_COL = "Churn Probability"
CALIBRATED_COL = "Calibrated Churn Probability"
OUTPUT_COLS = (PREDICTION_COL, PROBABILITY_COL, CALIBRATED_COL)


class DataValidationError(ValueError):
    """Raised for user-correctable problems with an uploaded file."""


class MetadataError(ValueError):
    """Raised when model metadata is missing, malformed or inconsistent with the model."""


# --------------------------------------------------------------------------- #
# Threshold policy
# --------------------------------------------------------------------------- #
def validate_threshold(threshold) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float, np.integer, np.floating)):
        raise ValueError(f"Threshold must be a number, got {threshold!r}.")
    value = float(threshold)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(f"Threshold must be strictly between 0 and 1, got {value}.")
    return value


def apply_threshold(proba, threshold) -> np.ndarray:
    """Return 1 where ``proba >= threshold``, else 0."""
    thr = validate_threshold(threshold)
    p = np.asarray(proba, dtype=float)
    if p.ndim != 1:
        raise ValueError(f"Probabilities must be one-dimensional, got shape {p.shape}.")
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must be finite and within [0, 1].")
    return (p >= thr).astype(int)


def positive_class_proba(model, X: pd.DataFrame) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"{type(model).__name__} does not provide predict_proba().")
    classes = list(getattr(model, "classes_", [0, 1]))
    if 1 not in classes:
        raise ValueError(f"Model classes {classes} do not include the positive class 1.")
    return model.predict_proba(X)[:, classes.index(1)]


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_metadata(path: str) -> dict:
    if not os.path.exists(path):
        raise MetadataError(f"Model metadata file not found: {path}. Re-run training.")
    try:
        with open(path) as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise MetadataError(f"Model metadata file {path} could not be read: {exc}") from exc
    if not isinstance(metadata, dict):
        raise MetadataError(f"Model metadata file {path} does not contain a JSON object.")
    return metadata


def resolve_threshold(metadata: dict, model_sha256: Optional[str] = None) -> float:
    """Return the validated decision threshold recorded for the trained model.

    Raises ``MetadataError`` if the metadata uses an unknown schema, the threshold
    is missing or invalid, the reported test metrics were computed at a different
    threshold, or the metadata belongs to a different model file.
    """
    version = metadata.get("schema_version")
    if version != METADATA_SCHEMA_VERSION:
        raise MetadataError(
            f"Model metadata schema version is {version!r}, expected {METADATA_SCHEMA_VERSION}. "
            "Re-run training to regenerate outputs/metrics.json."
        )
    try:
        raw = metadata["threshold"]["value"]
    except (KeyError, TypeError):
        raise MetadataError("Model metadata has no 'threshold.value' entry.") from None
    try:
        threshold = validate_threshold(raw)
    except ValueError as exc:
        raise MetadataError(f"Invalid decision threshold in metadata: {exc}") from None

    test_thr = (metadata.get("test_metrics") or {}).get("threshold")
    if test_thr is None or not math.isclose(float(test_thr), threshold, abs_tol=1e-12):
        raise MetadataError(
            f"Reported test metrics were computed at threshold {test_thr!r}, "
            f"but the configured threshold is {threshold}."
        )

    expected_sha = (metadata.get("artifacts") or {}).get("model_sha256")
    if model_sha256 is not None and expected_sha != model_sha256:
        raise MetadataError(
            "Model metadata does not belong to the loaded model file (SHA-256 mismatch). "
            "Re-run training so the model and metrics are regenerated together."
        )
    return threshold


# --------------------------------------------------------------------------- #
# File ingestion
# --------------------------------------------------------------------------- #
def file_extension(filename: str) -> str:
    return os.path.splitext(str(filename))[1].lower()


def _as_buffer(source):
    if isinstance(source, (bytes, bytearray)):
        return io.BytesIO(source)
    if hasattr(source, "seek"):
        source.seek(0)
    return source


def list_excel_sheets(source) -> list[str]:
    try:
        with pd.ExcelFile(_as_buffer(source)) as xl:
            return [str(s) for s in xl.sheet_names]
    except Exception as exc:
        raise DataValidationError(f"The Excel file could not be opened ({exc}).") from exc


def resolve_sheet(sheet_names: list[str], preferred: str = DATA_SHEET) -> Optional[str]:
    """Pick the data sheet: the preferred sheet if present, the only sheet, else ``None``."""
    if preferred in sheet_names:
        return preferred
    if len(sheet_names) == 1:
        return sheet_names[0]
    return None


def read_uploaded_table(source, filename: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Read a CSV or XLSX upload into a DataFrame with validated, unique column names."""
    ext = file_extension(filename)
    if ext not in SUPPORTED_EXTENSIONS:
        raise DataValidationError(
            f"Unsupported file type '{ext or filename}'. Upload a .csv or .xlsx file."
        )

    if ext == ".csv":
        df = _read_csv(source)
    else:
        sheets = list_excel_sheets(source)
        if sheet_name is None:
            sheet_name = resolve_sheet(sheets)
            if sheet_name is None:
                raise DataValidationError(
                    f"The workbook has several sheets ({', '.join(sheets)}) and none is named "
                    f"'{DATA_SHEET}'. Select the sheet that contains the customer records."
                )
        elif sheet_name not in sheets:
            raise DataValidationError(f"Sheet '{sheet_name}' not found. Available: {', '.join(sheets)}.")
        try:
            df = pd.read_excel(_as_buffer(source), sheet_name=sheet_name)
        except Exception as exc:
            raise DataValidationError(f"Sheet '{sheet_name}' could not be read ({exc}).") from exc

    df.columns = [str(c).strip() for c in df.columns]
    duplicated = sorted({c for c in df.columns if list(df.columns).count(c) > 1})
    if duplicated:
        raise DataValidationError(f"Duplicate column names: {', '.join(duplicated)}.")
    df = df.dropna(how="all")
    if df.empty:
        raise DataValidationError("The uploaded file contains no customer records.")
    return df.reset_index(drop=True)


def _read_csv(source) -> pd.DataFrame:
    last_error = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(_as_buffer(source), encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
        except pd.errors.EmptyDataError:
            raise DataValidationError("The uploaded CSV file is empty.") from None
        except pd.errors.ParserError as exc:
            raise DataValidationError(f"The CSV file could not be parsed ({exc}).") from exc
    raise DataValidationError(f"The CSV file encoding is not supported ({last_error}).")


# --------------------------------------------------------------------------- #
# Feature schema and validation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FeatureSchema:
    features: list[str]
    numeric: list[str]
    categorical: list[str]
    categories: dict[str, list[str]]

    def to_dict(self) -> dict:
        return {"features": self.features, "numeric": self.numeric,
                "categorical": self.categorical, "categories": self.categories}


def schema_from_model(model) -> FeatureSchema:
    """Read the expected input columns and known categories from the fitted pipeline."""
    pre = model.named_steps["pre"]
    numeric, categorical, categories = [], [], {}
    for name, transformer, cols in pre.transformers_:
        if name == "num":
            numeric = list(cols)
        elif name == "cat":
            categorical = list(cols)
            encoder = transformer.named_steps["encoder"]
            categories = {c: [str(v) for v in cats] for c, cats in zip(cols, encoder.categories_)}
    features = [str(c) for c in getattr(model, "feature_names_in_", numeric + categorical)]
    return FeatureSchema(features, numeric, categorical, categories)


@dataclass
class ValidationReport:
    n_rows: int
    excluded_columns: list[str] = field(default_factory=list)
    extra_columns: list[str] = field(default_factory=list)
    missing_values: dict[str, int] = field(default_factory=dict)
    unseen_categories: dict[str, list[str]] = field(default_factory=dict)


def _row_list(mask: pd.Series, limit: int = 5) -> str:
    rows = (np.flatnonzero(mask.to_numpy()) + 1)[:limit]
    return ", ".join(str(r) for r in rows)


def prepare_features(df: pd.DataFrame, schema: FeatureSchema) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate an uploaded table and return model-ready features in training column order.

    Row order and index are preserved. Missing values are left for the fitted
    imputers; unseen categories are left for the encoder (encoded as all zeros)
    and reported.
    """
    if df is None or len(df) == 0:
        raise DataValidationError("The uploaded file contains no customer records.")

    missing = [c for c in schema.features if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"Missing required column(s): {', '.join(missing)}. "
            f"The file must contain these {len(schema.features)} feature columns: "
            f"{', '.join(schema.features)}. Column names are case-sensitive."
        )

    report = ValidationReport(n_rows=len(df))
    report.excluded_columns = [c for c in NON_FEATURE_COLS if c in df.columns]
    report.extra_columns = [
        c for c in df.columns
        if c not in schema.features and c not in NON_FEATURE_COLS and c not in OUTPUT_COLS
    ]

    X = df[schema.features].copy()
    problems = []
    for col in schema.numeric:
        original = X[col]
        if original.dtype == object:
            original = original.replace(r"^\s*$", np.nan, regex=True)
        if pd.api.types.is_bool_dtype(original):
            problems.append(f"'{col}' contains true/false values; a number is required.")
            continue
        coerced = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & coerced.isna()
        if invalid.any():
            examples = ", ".join(repr(v) for v in pd.unique(original[invalid])[:3])
            problems.append(
                f"'{col}' has {int(invalid.sum())} non-numeric value(s) (e.g. {examples}) "
                f"in row(s) {_row_list(invalid)}."
            )
            continue
        infinite = np.isinf(coerced.astype(float))
        if infinite.any():
            problems.append(f"'{col}' has infinite value(s) in row(s) {_row_list(infinite)}.")
            continue
        X[col] = coerced.astype(float)

    if problems:
        raise DataValidationError("Invalid numeric data. " + " ".join(problems))

    for col in schema.categorical:
        values = X[col]
        as_text = values.where(values.isna(), values.astype(str).str.strip())
        as_text = as_text.replace("", np.nan).astype(object)
        X[col] = as_text.where(as_text.notna(), np.nan)
        known = set(schema.categories.get(col, []))
        unseen = sorted({v for v in X[col].dropna().unique() if v not in known})
        if unseen:
            report.unseen_categories[col] = unseen

    counts = X.isna().sum()
    report.missing_values = {c: int(n) for c, n in counts.items() if n > 0}
    return X, report


def score_prepared(model, df: pd.DataFrame, X: pd.DataFrame, threshold: Optional[float],
                   calibrator: Optional[dict] = None):
    """Score features from ``prepare_features`` and attach results to the uploaded rows.

    ``results`` keeps every uploaded column and row in the original order and
    appends the 0/1 churn prediction (only when a valid threshold is available)
    and the churn probability in percent. With a validated ``calibrator`` the
    calibrated probability is added too; the prediction always uses the
    uncalibrated score. Any existing prediction columns are replaced.
    """
    if len(X) != len(df) or not X.index.equals(df.index):
        raise ValueError("Prepared features are not aligned with the uploaded rows.")
    proba = positive_class_proba(model, X)
    results = df.drop(columns=[c for c in OUTPUT_COLS if c in df.columns])
    if threshold is not None:
        results[PREDICTION_COL] = apply_threshold(proba, threshold)
    results[PROBABILITY_COL] = np.round(proba * 100, 2)
    if calibrator is not None:
        from calibration import apply_calibrator

        results[CALIBRATED_COL] = np.round(apply_calibrator(calibrator, proba) * 100, 2)
    return results, proba


def predict_churn(model, df: pd.DataFrame, schema: FeatureSchema, threshold: Optional[float]):
    """Validate and score a batch of customers. Returns ``(results, probabilities, report)``."""
    X, report = prepare_features(df, schema)
    results, proba = score_prepared(model, df, X, threshold)
    return results, proba, report
