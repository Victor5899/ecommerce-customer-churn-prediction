import hashlib
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from inference import (  # noqa: E402
    PREDICTION_COL, DataValidationError, MetadataError, file_extension, file_sha256,
    list_excel_sheets, load_metadata, prepare_features, read_uploaded_table, resolve_sheet,
    resolve_threshold, schema_from_model, score_prepared,
)
from utils import DATA_SHEET, TARGET_COL  # noqa: E402

OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = OUTPUT_DIR / "best_model.joblib"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
DATA_PATH = ROOT / "data" / "E Commerce Dataset.xlsx"
TRAIN_COMMAND = 'python src/train_models.py --input "data/E Commerce Dataset.xlsx"'

st.set_page_config(
    page_title="E-Commerce Customer Churn Prediction",
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


@st.cache_data(show_spinner=False)
def reference_summary(path: str, mtime) -> dict:
    data = pd.read_excel(path, sheet_name=DATA_SHEET)
    return {"total": len(data), "churn_rate": float(data[TARGET_COL].mean())}


@st.cache_data(show_spinner=False, max_entries=8)
def excel_sheets(raw: bytes):
    return list_excel_sheets(raw)


@st.cache_data(show_spinner="Reading file...", max_entries=8)
def parse_upload(raw: bytes, filename: str, sheet):
    return read_uploaded_table(raw, filename, sheet)


st.title("📊 E-Commerce Customer Churn Prediction Dashboard")

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

if metadata is not None:
    model_name = metadata["model_selection"]["selected_model_name"]
else:
    model_name = type(model.named_steps["clf"]).__name__

st.markdown(f"""
Predict whether a customer is likely to churn using the trained **{model_name}** model.
""")

if threshold is not None:
    st.info(
        f"**Decision threshold: {threshold:.2f}.** A customer is predicted to churn when the churn "
        f"probability is at least {threshold:.0%}. The threshold was chosen on the validation set "
        f"({metadata['threshold']['objective']}); the test metrics below use the same threshold."
    )
else:
    st.error(
        f"**Decision threshold unavailable.** {metadata_error} Churn probabilities are still "
        "calculated, but customers are not classified as high or low risk, and saved evaluation "
        f"results are hidden because they may not describe this model. Fix: `{TRAIN_COMMAND}`"
    )

st.header("Dataset Overview")

if DATA_PATH.exists():
    try:
        summary = reference_summary(str(DATA_PATH), _mtime(DATA_PATH))
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Customers", summary["total"])
        with col2:
            st.metric("Churn Rate", f"{summary['churn_rate']*100:.2f}%")
        with col3:
            st.metric("Features", len(schema.features))
        if metadata is not None:
            data_info = metadata["data"]
            st.caption(
                f"{data_info['deduplication']['n_removed_identical_records']} records duplicate another "
                "customer's features and churn label exactly (only CustomerID differs) and were removed "
                f"before training; {data_info['n_rows_after_deduplication']} unique records were used."
            )
    except Exception as exc:
        st.warning(f"The reference dataset could not be summarised ({exc}).")
else:
    st.caption(f"Reference dataset not found at `{DATA_PATH}`; overview skipped.")


def render_predictions():
    uploaded_file = st.file_uploader(
        "Upload customer data",
        type=["csv", "xlsx"]
    )
    with st.expander("Expected file format"):
        st.markdown(
            f"CSV or XLSX with one row per customer and these **{len(schema.features)}** columns "
            "(names are case-sensitive). `CustomerID` is optional and kept in the results; "
            f"`{TARGET_COL}` is optional and ignored. For workbooks, the `{DATA_SHEET}` sheet is "
            "used when present. Missing values are allowed."
        )
        st.code(", ".join(schema.features))

    if uploaded_file is None:
        return

    raw = uploaded_file.getvalue()
    sheet = None
    try:
        if file_extension(uploaded_file.name) == ".xlsx":
            sheets = excel_sheets(raw)
            sheet = resolve_sheet(sheets)
            if sheet is None:
                sheet = st.selectbox(
                    f"This workbook has several sheets and none is named '{DATA_SHEET}'. "
                    "Choose the sheet that contains the customer records:",
                    sheets,
                )
            elif len(sheets) > 1:
                st.caption(f"Using sheet '{sheet}' (workbook sheets: {', '.join(sheets)}).")
        df = parse_upload(raw, uploaded_file.name, sheet)
        X, report = prepare_features(df, schema)
    except DataValidationError as exc:
        st.error(f"**This file cannot be scored.** {exc}")
        return
    except Exception as exc:
        st.error(f"**Unexpected error while reading the file:** {exc}")
        return

    st.subheader("Uploaded Data")
    st.caption(f"{len(df)} rows, {df.shape[1]} columns")
    st.dataframe(df.head())

    if report.extra_columns:
        st.info(f"Extra column(s) not used for prediction (kept in the results): "
                f"{', '.join(report.extra_columns)}.")
    if report.missing_values:
        details = ", ".join(f"{c} ({n})" for c, n in report.missing_values.items())
        st.info(f"Missing values will be filled from the training data (median for numbers, most "
                f"frequent value for categories): {details}.")
    if report.unseen_categories:
        details = "; ".join(f"{c}: {', '.join(v)}" for c, v in report.unseen_categories.items())
        st.warning(f"Categories not seen during training (treated as none of the known categories, "
                   f"so these predictions may be less reliable): {details}.")

    key = (hashlib.sha256(raw).hexdigest(), sheet, threshold, model_sha)
    if st.button("Predict Churn"):
        try:
            results, _ = score_prepared(model, df, X, threshold)
        except Exception as exc:
            st.error(f"**Prediction failed:** {exc}")
            return
        st.session_state["prediction"] = {"key": key, "results": results}

    stored = st.session_state.get("prediction")
    if not stored or stored["key"] != key:
        return
    results = stored["results"]

    st.subheader("Prediction Results")
    st.dataframe(results)

    if threshold is not None:
        st.subheader("Prediction Summary")
        high_risk = int((results[PREDICTION_COL] == 1).sum())
        low_risk = int((results[PREDICTION_COL] == 0).sum())

        c1, c2 = st.columns(2)

        with c1:
            st.metric("High Risk Customers", high_risk)

        with c2:
            st.metric("Low Risk Customers", low_risk)
        st.caption(f"High risk = churn probability ≥ {threshold:.0%}.")

    csv = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Results",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv",
        on_click="ignore",
    )


render_predictions()

st.header("Model Performance")

if metadata is None:
    st.warning("Saved evaluation results are hidden because the model metadata is unavailable "
               "or does not match the loaded model.")
else:
    test = metadata["test_metrics"]
    st.caption(
        f"Held-out test set: {test['n_samples']} customers ({test['n_positive']} churners), not used "
        f"for model selection or threshold tuning. Classification metrics at threshold "
        f"{test['threshold']:.2f}."
    )
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("ROC-AUC", f"{test['roc_auc']:.3f}")
    m2.metric("PR-AUC", f"{test['pr_auc']:.3f}")
    m3.metric("Precision", f"{test['precision']:.1%}")
    m4.metric("Recall", f"{test['recall']:.1%}")
    m5.metric("F1 Score", f"{test['f1']:.1%}")

    with st.expander("Candidate model comparison"):
        comparison = pd.DataFrame(metadata["candidates"])
        st.dataframe(comparison[[
            "model_name", "selected", "cv_pr_auc_mean", "cv_pr_auc_std", "validation_pr_auc",
            "validation_roc_auc", "validation_threshold", "validation_precision",
            "validation_recall", "validation_f1",
        ]])
        st.caption(
            "Selection uses the mean cross-validated PR-AUC on the training data. Validation "
            "precision/recall/F1 use each model's own validation-tuned threshold, so they are "
            "optimistic and only for comparison. Test results are reported for the selected model only."
        )

    plots = [
        ("confusion_matrix.png", "Confusion Matrix"),
        ("feature_importance.png", "Feature Importance"),
        ("roc_curve.png", "ROC Curve"),
        ("pr_curve.png", "Precision Recall Curve"),
    ]
    for row in range(0, len(plots), 2):
        for column, (filename, caption) in zip(st.columns(2), plots[row:row + 2]):
            with column:
                path = OUTPUT_DIR / filename
                if path.exists():
                    st.image(str(path), caption=caption)
                else:
                    st.caption(f"{caption} not found ({filename}).")
