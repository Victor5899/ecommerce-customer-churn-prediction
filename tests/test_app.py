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


def run_with_upload(path: Path, name: str):
    at = streamlit_testing.AppTest.from_function(
        app_with_upload, args=(str(APP), str(path), name), default_timeout=120)
    return at.run()


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
    metrics = {m.label: m.value for m in at.metric}
    total = int(metrics["High Risk Customers"]) + int(metrics["Low Risk Customers"])
    assert total == len(pd.read_excel(DATASET, sheet_name="E Comm"))


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
