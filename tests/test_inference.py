import io

import numpy as np
import pandas as pd
import pytest

from inference import (
    PREDICTION_COL, PROBABILITY_COL, DataValidationError, positive_class_proba, predict_churn,
    prepare_features, read_uploaded_table, resolve_sheet,
)
from utils import DATA_SHEET, ID_COL, TARGET_COL


def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def to_xlsx_bytes(sheets: dict):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
    return buffer.getvalue()


DATA_DICT = pd.DataFrame({"Variable": ["Churn", "Tenure"], "Discerption": ["Churn Flag", "Tenure"]})


# --------------------------------------------------------------------------- #
# File ingestion
# --------------------------------------------------------------------------- #
def test_csv_upload(customers):
    df = read_uploaded_table(to_csv_bytes(customers), "customers.csv")
    assert df.shape == customers.shape
    assert df[ID_COL].tolist() == customers[ID_COL].tolist()


def test_csv_with_bom_and_padded_headers(customers):
    padded = customers.rename(columns={"Tenure": " Tenure "})
    raw = padded.to_csv(index=False).encode("utf-8-sig")
    df = read_uploaded_table(raw, "customers.CSV")
    assert "Tenure" in df.columns and df.columns[0] == ID_COL


def test_original_workbook_uses_e_comm_sheet(customers):
    raw = to_xlsx_bytes({"Data Dict": DATA_DICT, DATA_SHEET: customers})
    df = read_uploaded_table(raw, "E Commerce Dataset.xlsx")
    assert df.shape == customers.shape


def test_single_sheet_workbook(customers):
    raw = to_xlsx_bytes({"Sheet1": customers})
    assert read_uploaded_table(raw, "upload.xlsx").shape == customers.shape


def test_ambiguous_workbook_requires_sheet_choice(customers):
    raw = to_xlsx_bytes({"Notes": DATA_DICT, "Customers": customers})
    with pytest.raises(DataValidationError, match="several sheets"):
        read_uploaded_table(raw, "upload.xlsx")
    assert read_uploaded_table(raw, "upload.xlsx", sheet_name="Customers").shape == customers.shape
    with pytest.raises(DataValidationError, match="not found"):
        read_uploaded_table(raw, "upload.xlsx", sheet_name="Nope")


def test_resolve_sheet():
    assert resolve_sheet(["Data Dict", DATA_SHEET]) == DATA_SHEET
    assert resolve_sheet(["Only"]) == "Only"
    assert resolve_sheet(["A", "B"]) is None


@pytest.mark.parametrize("raw, name, message", [
    (b"", "empty.csv", "empty"),
    (b"Tenure,Gender\n", "headers.csv", "no customer records"),
    (b"a,b\n1,2\n", "data.txt", "Unsupported file type"),
    (b"not a workbook", "broken.xlsx", "could not be opened"),
])
def test_bad_files_raise_validation_errors(raw, name, message):
    with pytest.raises(DataValidationError, match=message):
        read_uploaded_table(raw, name)


def test_duplicate_column_names_rejected():
    raw = b"Tenure,Tenure \n1,2\n"
    with pytest.raises(DataValidationError, match="Duplicate column"):
        read_uploaded_table(raw, "dup.csv")


# --------------------------------------------------------------------------- #
# Feature validation
# --------------------------------------------------------------------------- #
def test_missing_required_columns(customers, schema):
    with pytest.raises(DataValidationError) as exc:
        prepare_features(customers.drop(columns=["Tenure", "Gender"]), schema)
    message = str(exc.value)
    assert "Tenure" in message and "Gender" in message and "Missing required" in message


def test_id_target_and_extra_columns_are_excluded(customers, schema):
    df = customers.assign(Region="North", Notes="x")
    X, report = prepare_features(df, schema)
    assert list(X.columns) == schema.features
    assert report.excluded_columns == [ID_COL, TARGET_COL]
    assert report.extra_columns == ["Region", "Notes"]


def test_columns_are_reordered_to_training_order(customers, schema):
    shuffled_cols = customers[customers.columns[::-1]]
    X, _ = prepare_features(shuffled_cols, schema)
    assert list(X.columns) == schema.features


def test_non_numeric_values_rejected(customers, schema):
    df = customers.astype({"Tenure": object})
    df.loc[3, "Tenure"] = "ten"
    with pytest.raises(DataValidationError, match=r"'Tenure'.*'ten'.*row\(s\) 4"):
        prepare_features(df, schema)


def test_infinite_values_rejected(customers, schema):
    df = customers.copy()
    df.loc[0, "CashbackAmount"] = np.inf
    with pytest.raises(DataValidationError, match="infinite"):
        prepare_features(df, schema)


def test_numeric_strings_and_blanks_are_accepted(customers, schema):
    df = customers.astype({"Tenure": object})
    df.loc[0, "Tenure"] = "12"
    df.loc[1, "Tenure"] = "  "
    X, report = prepare_features(df, schema)
    assert X.loc[0, "Tenure"] == 12.0 and np.isnan(X.loc[1, "Tenure"])
    assert report.missing_values["Tenure"] >= 1


def test_missing_values_reported_and_imputed(customers, schema, trained_model):
    df = customers.copy()
    df.loc[:9, ["Tenure", "Gender"]] = np.nan
    results, proba, report = predict_churn(trained_model, df, schema, 0.5)
    assert report.missing_values["Gender"] == 10
    assert np.isfinite(proba).all()


def test_unseen_categories_reported_and_scored(customers, schema, trained_model):
    df = customers.copy()
    df.loc[0, "PreferredPaymentMode"] = "Bitcoin"
    df.loc[1, "Gender"] = 1
    results, proba, report = predict_churn(trained_model, df, schema, 0.5)
    assert report.unseen_categories == {"PreferredPaymentMode": ["Bitcoin"], "Gender": ["1"]}
    assert np.isfinite(proba).all()


def test_empty_dataframe_rejected(customers, schema):
    with pytest.raises(DataValidationError, match="no customer records"):
        prepare_features(customers.iloc[0:0], schema)


# --------------------------------------------------------------------------- #
# Prediction output
# --------------------------------------------------------------------------- #
def test_prediction_dimensions_and_threshold(customers, schema, trained_model):
    results, proba, _ = predict_churn(trained_model, customers, schema, 0.4)
    assert len(results) == len(customers) == len(proba)
    assert list(results.columns) == list(customers.columns) + [PREDICTION_COL, PROBABILITY_COL]
    assert results[PREDICTION_COL].tolist() == (proba >= 0.4).astype(int).tolist()
    assert np.allclose(results[PROBABILITY_COL], np.round(proba * 100, 2))


def test_rows_stay_aligned_with_identifiers(customers, schema, trained_model):
    shuffled = customers.sample(frac=1, random_state=3).reset_index(drop=True)
    results, proba, _ = predict_churn(trained_model, shuffled, schema, 0.5)
    assert results[ID_COL].tolist() == shuffled[ID_COL].tolist()

    X_reference, _ = prepare_features(customers, schema)
    reference = pd.Series(positive_class_proba(trained_model, X_reference), index=customers[ID_COL])
    assert np.allclose(proba, reference.loc[shuffled[ID_COL]].to_numpy())


def test_probabilities_only_without_threshold(customers, schema, trained_model):
    results, _, _ = predict_churn(trained_model, customers, schema, None)
    assert PREDICTION_COL not in results.columns and PROBABILITY_COL in results.columns


def test_existing_prediction_columns_are_replaced(customers, schema, trained_model):
    previous = customers.assign(**{PREDICTION_COL: 9, PROBABILITY_COL: -1.0})
    results, _, report = predict_churn(trained_model, previous, schema, 0.5)
    assert set(results[PREDICTION_COL]) <= {0, 1}
    assert results[PROBABILITY_COL].between(0, 100).all()
    assert report.extra_columns == []
