"""Dashboard smoke tests against the saved artifacts in outputs/ (skipped if absent)."""
import io
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"
MODEL = ROOT / "outputs" / "best_model.joblib"
METRICS = ROOT / "outputs" / "metrics.json"
DATASET = ROOT / "data" / "E Commerce Dataset.xlsx"

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
pytestmark = pytest.mark.skipif(
    not (MODEL.exists() and METRICS.exists()), reason="trained artifacts not found in outputs/"
)


def app_with_upload(app_path, upload_path, upload_name):
    import runpy

    import streamlit as st

    class FakeUpload:
        name = upload_name

        def getvalue(self):
            with open(upload_path, "rb") as f:
                return f.read()

    original = st.file_uploader
    st.file_uploader = lambda *args, **kwargs: FakeUpload()
    try:
        runpy.run_path(app_path, run_name="__main__")
    finally:
        st.file_uploader = original


PAGES = ["Executive Overview", "Customer Analytics", "Churn Prediction", "Model Performance",
         "Data Quality", "Retention Intelligence"]


def open_page(at, page: str):
    at.sidebar.radio(key="page").set_value(page)
    return at.run()


def run_with_upload(path: Path, name: str):
    at = streamlit_testing.AppTest.from_function(
        app_with_upload, args=(str(APP), str(path), name), default_timeout=120)
    return open_page(at.run(), "Churn Prediction")


def texts(elements):
    return " ".join(e.value for e in elements)


def test_dashboard_loads_with_configured_threshold():
    at = streamlit_testing.AppTest.from_file(str(APP), default_timeout=120).run()
    threshold = json.loads(METRICS.read_text())["threshold"]["value"]
    assert not at.exception
    assert not at.error
    assert f"Decision threshold: {threshold:.2f}" in texts(at.info)


@pytest.mark.skipif(not DATASET.exists(), reason="original workbook not found")
def test_original_workbook_upload_and_prediction():
    at = run_with_upload(DATASET, DATASET.name)
    assert not at.exception and not at.error
    assert "Using sheet 'E Comm'" in texts(at.caption)

    at.button[0].click().run()
    assert not at.exception and not at.error
    metrics = {m.label: int(m.value.replace(",", "")) for m in at.metric
               if m.label in ("Customers scored", "Predicted to churn", "Predicted to stay")}
    n_rows = len(pd.read_excel(DATASET, sheet_name="E Comm"))
    assert metrics["Predicted to churn"] + metrics["Predicted to stay"] == n_rows
    assert metrics["Customers scored"] == n_rows

    # Predictions persist to other pages and never replace historical (actual) churn.
    at = open_page(at, "Executive Overview")
    assert not at.exception and not at.error
    values = {m.label: m.value for m in at.metric}
    assert values["Customers scored"] == f"{n_rows:,}"
    assert values["Actual churn rate"] == f"{pd.read_excel(DATASET, sheet_name='E Comm')['Churn'].mean():.1%}"
    at = open_page(at, "Data Quality")
    assert not at.exception and not at.error
    at = open_page(at, "Retention Intelligence")
    assert not at.exception and not at.error
    assert {"Eligible customers", "Selected customers", "Unspent budget"} <= {m.label for m in at.metric}


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_errors(page):
    at = open_page(streamlit_testing.AppTest.from_file(str(APP), default_timeout=120).run(), page)
    assert not at.exception
    assert not at.error


def test_model_performance_shows_saved_test_metrics_and_threshold():
    test = json.loads(METRICS.read_text())["test_metrics"]
    at = open_page(streamlit_testing.AppTest.from_file(str(APP), default_timeout=120).run(),
                   "Model Performance")
    values = {}
    for m in at.metric:  # the test-set tab renders before the validation tab
        values.setdefault(m.label, m.value)
    assert values["Decision threshold"] == f"{test['threshold']:.2f}"
    assert values["Recall"] == f"{test['recall']:.1%}"
    assert values["PR-AUC"] == f"{test['pr_auc']:.3f}"


def test_missing_columns_show_streamlit_error(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"CustomerID": [1], "Tenure": [3]}).to_csv(path, index=False)
    at = run_with_upload(path, path.name)
    assert not at.exception
    assert "Missing required column" in texts(at.error)
    assert len(at.button) == 0


def test_unreadable_file_shows_streamlit_error(tmp_path):
    path = tmp_path / "broken.xlsx"
    path.write_bytes(b"not an excel file")
    at = run_with_upload(path, path.name)
    assert not at.exception
    assert "could not be opened" in texts(at.error)


def test_workbook_without_e_comm_sheet_offers_sheet_choice(tmp_path):
    path = tmp_path / "multi.xlsx"
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        pd.DataFrame({"note": ["x"]}).to_excel(writer, sheet_name="Notes", index=False)
        pd.DataFrame({"note": ["y"]}).to_excel(writer, sheet_name="Other", index=False)
    path.write_bytes(buffer.getvalue())
    at = run_with_upload(path, path.name)
    assert not at.exception
    assert len(at.selectbox) == 1 and at.selectbox[0].options == ["Notes", "Other"]
