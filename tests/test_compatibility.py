"""Dataset compatibility checker: statuses, safe/unsafe mappings, types, duplicates, target."""
import io

import numpy as np
import pandas as pd
import pytest

import compatibility as compat
from inference import (
    predict_churn, prepare_features, read_column_names, read_uploaded_table,
)
from utils import ID_COL, TARGET_COL


def check(df, schema, names=None):
    return compat.check_compatibility(df, schema, list(df.columns) if names is None else names)


def confirm(df, schema):
    """Apply proposed fixes and re-check, as the dashboard does after confirmation."""
    report = check(df, schema)
    assert report.status == compat.NEEDS_MAPPING, report.problems
    fixed = compat.apply_fixes(df, report.fixes)
    return report, fixed, compat.check_compatibility(fixed, schema)


# --------------------------------------------------------------------------- #
# Ready to score
# --------------------------------------------------------------------------- #
def test_compatible_file_is_ready_and_recognises_the_target(customers, schema):
    report = check(customers, schema)
    assert report.status == compat.READY
    assert not report.fixes and not report.problems and not report.missing and not report.extra
    assert report.has_target and report.target_usable
    assert any("missing feature values" in w for w in report.warnings)   # fixture has ~3% missing


def test_target_is_optional(customers, schema):
    report = check(customers.drop(columns=[TARGET_COL, ID_COL]), schema)
    assert report.status == compat.READY and not report.has_target


def test_extra_columns_are_reported_but_do_not_block(customers, schema):
    report = check(customers.assign(Region="North"), schema)
    assert report.status == compat.READY and report.extra == ["Region"]


# --------------------------------------------------------------------------- #
# Missing features and mappings
# --------------------------------------------------------------------------- #
def test_missing_features_are_incompatible_and_never_filled(customers, schema):
    report = check(customers.drop(columns=["Tenure", "Gender"]), schema)
    assert report.status == compat.INCOMPATIBLE
    assert set(report.missing) == {"Tenure", "Gender"}
    assert report.problems[0].startswith("Missing required column(s)")
    assert not report.fixes


def test_safe_renames_need_confirmation_then_match_the_clean_file(customers, schema):
    renamed = customers.rename(columns={
        "Tenure": "tenure", "DaySinceLastOrder": "Days Since Last Order",
        "PreferedOrderCat": "PreferredOrderCat", "OrderAmountHikeFromlastYear": "order_amount_hike_from_last_year",
        ID_COL: "customer_id"})
    report, fixed, recheck = confirm(renamed, schema)
    assert {f.kind for f in report.fixes} == {"rename"}
    assert {f.source: f.column for f in report.fixes}["PreferredOrderCat"] == "PreferedOrderCat"
    assert recheck.status == compat.READY
    X_fixed, _ = prepare_features(fixed, schema)
    X_clean, _ = prepare_features(customers, schema)
    pd.testing.assert_frame_equal(X_fixed, X_clean)
    assert fixed[ID_COL].tolist() == customers[ID_COL].tolist()


@pytest.mark.parametrize("new_name", ["MonthsAsCustomer", "Tenures", "Age"])
def test_differently_named_columns_are_not_mapped(customers, schema, new_name):
    report = check(customers.rename(columns={"Tenure": new_name}), schema)
    assert report.status == compat.INCOMPATIBLE and report.missing == ["Tenure"]
    assert not report.fixes and new_name in report.extra
    if new_name == "Tenures":
        assert "were not mapped" in report.hints[0]


def test_complaint_is_not_treated_as_complain(customers, schema):
    report = check(customers.rename(columns={"Complain": "Complaint"}), schema)
    assert report.status == compat.INCOMPATIBLE and "Complain" in report.missing


def test_ambiguous_candidates_are_not_mapped(customers, schema):
    df = customers.rename(columns={"Tenure": "tenure"}).assign(TENURE=1.0)
    report = check(df, schema)
    assert report.status == compat.INCOMPATIBLE
    assert any("Several columns could be 'Tenure'" in p for p in report.problems)


# --------------------------------------------------------------------------- #
# Types and value transformations
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("column,value", [("Tenure", "ten"), ("CashbackAmount", "1,234.5"),
                                          ("CityTier", "Tier 1")])
def test_invalid_numeric_values_are_incompatible(customers, schema, column, value):
    df = customers.astype({column: object})
    df.loc[2, column] = value
    report = check(df, schema)
    assert report.status == compat.INCOMPATIBLE
    assert any(p.startswith("Invalid type") and f"'{column}'" in p for p in report.problems)


def test_yes_no_complaint_flag_converts_to_binary_without_filling_missing(customers, schema):
    df = customers.copy()
    df["Complain"] = customers["Complain"].map({1: "Yes", 0: "no"}).astype(object)
    df.loc[0, "Complain"] = np.nan
    report, fixed, recheck = confirm(df, schema)
    assert [f.kind for f in report.fixes] == ["binary"]
    assert recheck.status == compat.READY
    assert np.isnan(fixed.loc[0, "Complain"])
    assert fixed["Complain"].iloc[1:].tolist() == customers["Complain"].iloc[1:].astype(float).tolist()
    assert df["Complain"].iloc[1] in ("Yes", "no")             # input not mutated


def test_boolean_complaint_flag_converts(customers, schema):
    _, fixed, recheck = confirm(customers.assign(Complain=customers["Complain"].astype(bool)), schema)
    assert recheck.status == compat.READY and set(fixed["Complain"]) <= {0.0, 1.0}


def test_unrecognised_flag_text_is_incompatible(customers, schema):
    df = customers.assign(Complain=customers["Complain"].map({1: "Yes", 0: "maybe"}))
    assert check(df, schema).status == compat.INCOMPATIBLE


def test_category_case_is_matched_but_unknown_categories_only_warn(customers, schema):
    df = customers.copy()
    df.loc[0, "PreferredLoginDevice"] = "mobile phone"
    report, fixed, recheck = confirm(df, schema)
    assert fixed.loc[0, "PreferredLoginDevice"] == "Mobile Phone"
    assert recheck.status == compat.READY

    df.loc[0, "PreferredLoginDevice"] = "Tablet"
    report = check(df, schema)
    assert report.status == compat.READY
    assert any("not seen in training" in w and "Tablet" in w for w in report.warnings)


def test_preview_shows_before_and_after_for_changed_rows(customers, schema):
    df = customers.rename(columns={"Tenure": "tenure"})
    df["Complain"] = customers["Complain"].map({1: "yes", 0: "no"})
    report = check(df, schema)
    preview = compat.preview_fixes(df, report.fixes, n=3)
    assert list(preview.columns) == ["tenure (before)", "Tenure (after)",
                                     "Complain (before)", "Complain (after)"]
    assert len(preview) == 3 and set(preview["Complain (after)"]) <= {0.0, 1.0}


# --------------------------------------------------------------------------- #
# Optional target
# --------------------------------------------------------------------------- #
def test_text_churn_labels_are_offered_for_evaluation(customers, schema):
    df = customers.assign(**{TARGET_COL: customers[TARGET_COL].map({1: "Yes", 0: "No"})})
    _, fixed, recheck = confirm(df, schema)
    assert recheck.target_usable and fixed[TARGET_COL].tolist() == customers[TARGET_COL].astype(float).tolist()


@pytest.mark.parametrize("labels", [[2] * 300, [np.nan] + [0] * 299])
def test_unusable_churn_column_warns_but_does_not_block(customers, schema, labels):
    report = check(customers.assign(**{TARGET_COL: labels}), schema)
    assert report.status == compat.READY and report.has_target and not report.target_usable
    assert any("not be used to report actual churn" in w for w in report.warnings)


# --------------------------------------------------------------------------- #
# Duplicates, headers and data quality
# --------------------------------------------------------------------------- #
def test_duplicate_feature_columns_are_detected_from_the_raw_header(customers):
    raw = pd.concat([customers, customers[["Tenure"]]], axis=1).to_csv(index=False).encode()
    names = read_column_names(raw, "dup.csv")
    df = read_uploaded_table(raw, "dup.csv")
    assert "Tenure.1" in df.columns                                      # pandas renames silently
    assert names.count("Tenure") == 2


def test_duplicate_columns_block_only_when_they_are_model_columns(customers, schema):
    header = list(customers.columns) + ["Tenure", "Notes", "Notes"]
    report = compat.check_compatibility(None, schema, header)
    assert report.status == compat.INCOMPATIBLE and report.duplicates == ["Notes", "Tenure"]
    assert any("Duplicate column 'Tenure'" in p for p in report.problems)
    assert any("Extra column 'Notes'" in w for w in report.warnings)
    report = compat.check_compatibility(None, schema, list(customers.columns) + ["Notes", "Notes"])
    assert report.status == compat.READY


def test_read_column_names_handles_xlsx_bom_and_bad_files(customers):
    buffer = io.BytesIO()
    customers.to_excel(buffer, sheet_name="E Comm", index=False)
    assert read_column_names(buffer.getvalue(), "x.xlsx", "E Comm") == list(customers.columns)
    bom = customers.rename(columns={ID_COL: f" {ID_COL}"}).to_csv(index=False).encode("utf-8-sig")
    assert read_column_names(bom, "x.csv")[0] == ID_COL
    assert read_column_names(b"not excel", "x.xlsx") == [] and read_column_names(b"", "x.csv") == []


def test_quality_warnings_and_empty_files(customers, schema):
    df = customers.copy()
    df.loc[0, "Tenure"] = -3
    df["WarehouseToHome"] = np.nan
    df.loc[5, ID_COL] = df.loc[4, ID_COL]
    report = check(df, schema)
    assert report.status == compat.READY
    text = " ".join(report.warnings)
    assert "negative value" in text and "no values: WarehouseToHome" in text and "share a `CustomerID`" in text
    assert check(customers.iloc[0:0], schema).status == compat.INCOMPATIBLE


def test_ready_files_score_exactly_as_before(customers, schema, trained_model):
    assert check(customers, schema).status == compat.READY
    renamed = customers.rename(columns={"Tenure": "TENURE"})
    _, fixed, _ = confirm(renamed, schema)
    before, p_before, _ = predict_churn(trained_model, customers, schema, 0.3)
    after, p_after, _ = predict_churn(trained_model, fixed, schema, 0.3)
    np.testing.assert_array_equal(p_before, p_after)
