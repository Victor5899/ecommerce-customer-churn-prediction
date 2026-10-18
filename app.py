import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import dashboard_pages as pages  # noqa: E402
from calibration import load_calibration  # noqa: E402
from explainability import background_frame, load_explainability  # noqa: E402
from inference import (  # noqa: E402
    MetadataError, file_sha256, list_excel_sheets, load_metadata, read_uploaded_table,
    resolve_threshold, schema_from_model,
)
from utils import DATA_SHEET  # noqa: E402

OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = OUTPUT_DIR / "best_model.joblib"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
EXPLAIN_PATH = OUTPUT_DIR / "explainability.json"
CALIBRATION_PATH = OUTPUT_DIR / "calibration.json"
CALIBRATOR_PATH = OUTPUT_DIR / "calibrator.joblib"
DATA_PATH = ROOT / "data" / "E Commerce Dataset.xlsx"
TRAIN_COMMAND = 'python src/train_models.py --input "data/E Commerce Dataset.xlsx"'
XAI_COMMAND = 'python src/explain_calibrate.py --input "data/E Commerce Dataset.xlsx"'

PAGES = {
    "Executive Overview": pages.executive_overview,
    "Customer Analytics": pages.customer_analytics,
    "Churn Prediction": pages.churn_prediction,
    "Model Performance": pages.model_performance,
    "Data Quality": pages.data_quality,
}

st.set_page_config(
    page_title="E-Commerce Customer Churn Analytics",
    page_icon="📊",
    layout="wide"
)


def _mtime(path: Path):
    return path.stat().st_mtime if path.exists() else None


@st.cache_resource(show_spinner="Loading model...")
def load_model(path: str, mtime):
    return joblib.load(path), file_sha256(path)


@st.cache_data(show_spinner=False)
def load_metadata_cached(path: str, mtime):
    return load_metadata(path)


@st.cache_data(show_spinner="Loading historical dataset...")
def load_reference(path: str, mtime) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=DATA_SHEET)


@st.cache_data(show_spinner=False)
def file_sha256_cached(path: str, mtime):
    return file_sha256(path)


@st.cache_data(show_spinner=False)
def load_explainability_cached(path: str, mtime, model_sha: str):
    return load_explainability(path, model_sha)


@st.cache_resource(show_spinner=False)
def load_calibration_cached(path: str, mtime, calibrator_mtime, model_sha: str):
    return load_calibration(path, str(OUTPUT_DIR), model_sha)


@st.cache_data(show_spinner=False, max_entries=8)
def excel_sheets(raw: bytes):
    return list_excel_sheets(raw)


@st.cache_data(show_spinner="Reading file...", max_entries=8)
def parse_upload(raw: bytes, filename: str, sheet):
    return read_uploaded_table(raw, filename, sheet)


try:
    model, model_sha = load_model(str(MODEL_PATH), _mtime(MODEL_PATH))
    schema = schema_from_model(model)
except Exception as exc:
    st.error(f"The trained model could not be loaded from `{MODEL_PATH}` ({exc}). "
             f"Train it first with: `{TRAIN_COMMAND}`")
    st.stop()

metadata, threshold, metadata_error = None, None, None
try:
    metadata = load_metadata_cached(str(METRICS_PATH), _mtime(METRICS_PATH))
    threshold = resolve_threshold(metadata, model_sha)
except MetadataError as exc:
    metadata, metadata_error = None, str(exc)

reference, reference_error = None, None
if DATA_PATH.exists():
    try:
        reference = load_reference(str(DATA_PATH), _mtime(DATA_PATH))
    except Exception as exc:
        reference_error = f"could not read `{DATA_PATH.name}`: {exc}"
else:
    reference_error = f"`{DATA_PATH.relative_to(ROOT)}` not found"

data_sha = file_sha256_cached(str(DATA_PATH), _mtime(DATA_PATH)) if DATA_PATH.exists() else None
xai_warnings = []

explain_meta, background, explain_error = None, None, None
try:
    explain_meta = load_explainability_cached(str(EXPLAIN_PATH), _mtime(EXPLAIN_PATH), model_sha)
    background = background_frame(explain_meta, schema)
except Exception as exc:
    explain_meta, explain_error = None, str(exc)

calibration_meta, calibrator, calibration_error = None, None, None
try:
    calibration_meta, calibrator = load_calibration_cached(
        str(CALIBRATION_PATH), _mtime(CALIBRATION_PATH), _mtime(CALIBRATOR_PATH), model_sha)
except Exception as exc:
    calibration_error = str(exc)

for name, meta in (("Explainability", explain_meta), ("Calibration", calibration_meta)):
    recorded = ((meta or {}).get("data") or {}).get("source_sha256")
    if meta is not None and data_sha is not None and recorded != data_sha:
        xai_warnings.append(f"{name} results were computed from a different version of the dataset "
                            f"file. Re-run `{XAI_COMMAND}`.")

if metadata is not None:
    model_name = metadata["model_selection"]["selected_model_name"]
else:
    model_name = type(model.named_steps["clf"]).__name__

ctx = pages.DashboardContext(
    model=model, model_sha=model_sha, model_name=model_name, schema=schema, metadata=metadata,
    threshold=threshold, metadata_error=metadata_error, reference=reference,
    reference_error=reference_error, output_dir=OUTPUT_DIR, model_path=MODEL_PATH,
    train_command=TRAIN_COMMAND, excel_sheets=excel_sheets, parse_upload=parse_upload,
    explainability=explain_meta, background=background, explainability_error=explain_error,
    calibration=calibration_meta, calibrator=calibrator, calibration_error=calibration_error,
    xai_command=XAI_COMMAND, xai_warnings=xai_warnings,
)

with st.sidebar:
    st.markdown("## 📊 Churn Analytics")
    st.caption("E-commerce customer churn: history, drivers and batch risk scoring.")
    page = st.radio("Navigate", list(PAGES), key="page", label_visibility="collapsed")
    st.divider()
    st.markdown("**Model status**")
    if threshold is not None:
        st.info(f"**{model_name}** · Decision threshold: {threshold:.2f}. "
                "Metadata verified against the model file.")
    else:
        st.error("Model metadata is missing or does not match the model. Predictions show "
                 "probabilities only.")
    st.caption(f"Explanations: {'available' if explain_meta else 'unavailable'} · Calibration: "
               + (calibration_meta['method'] if calibration_meta else 'unavailable'))
    batch = st.session_state.get(pages.BATCH_KEY)
    st.caption(f"Scored batch: `{batch['filename']}` ({len(batch['results']):,} rows)"
               if batch else "Scored batch: none yet")

PAGES[page](ctx)
