import numpy as np
import pandas as pd
import pytest

from utils import (
    DATA_SHEET, ID_COL, TARGET_COL, count_shared_feature_records, deduplicate, feature_columns,
    load_dataset, load_raw_dataset, separate_features_target, split_xy,
)


def write_workbook(path, df):
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"Variable": ["Churn"], "Description": ["Churn flag"]}).to_excel(
            writer, sheet_name="Data Dict", index=False)
        df.to_excel(writer, sheet_name=DATA_SHEET, index=False)


def test_load_dataset_reads_data_sheet_and_drops_id(tmp_path, customers):
    path = tmp_path / "data.xlsx"
    write_workbook(path, customers)

    raw = load_raw_dataset(str(path))
    df = load_dataset(str(path))

    assert len(raw) == len(customers) and ID_COL in raw.columns
    assert ID_COL not in df.columns
    assert list(df.columns) == [c for c in customers.columns if c != ID_COL]


def test_separate_features_target(customers):
    X, y = separate_features_target(customers)
    assert TARGET_COL not in X.columns and ID_COL not in X.columns
    assert X.shape == (len(customers), 18)
    assert y.dtype == int and set(y.unique()) <= {0, 1}
    assert feature_columns(customers) == list(X.columns)


@pytest.mark.parametrize("bad", [[0, 2], [0, np.nan], ["yes", "no"]])
def test_invalid_target_is_rejected(customers, bad):
    df = customers.head(2).copy()
    df[TARGET_COL] = pd.Series(bad, dtype=object)
    with pytest.raises(ValueError, match=TARGET_COL):
        separate_features_target(df)


def test_missing_target_is_rejected(customers):
    with pytest.raises(ValueError, match="not found"):
        separate_features_target(customers.drop(columns=[TARGET_COL]))


def _split(df, seed=42):
    return split_xy(df.drop(columns=[ID_COL]), 0.2, 0.2, seed)


def test_split_is_reproducible_and_sized(make_df):
    df = make_df(n=500)
    a, b = _split(df), _split(df)
    for part_a, part_b in zip(a, b):
        assert part_a.index.equals(part_b.index)
    X_train, X_val, X_test, *_ = a
    assert len(X_train) + len(X_val) + len(X_test) == len(df)
    assert len(X_test) == 100 and len(X_val) == 80
    assert not _split(df, seed=7)[2].index.equals(X_test.index)


def test_split_is_stratified(make_df):
    df = make_df(n=1000)
    _, _, _, y_train, y_val, y_test = _split(df)
    rate = df[TARGET_COL].mean()
    for y in (y_train, y_val, y_test):
        assert abs(y.mean() - rate) < 0.02


def test_identical_records_never_cross_partitions(make_df):
    df = make_df(n=400, n_duplicates=150).drop(columns=[ID_COL])
    X_train, X_val, X_test, *_ = split_xy(df, 0.2, 0.2, 42)
    assert count_shared_feature_records(X_train, X_val) == 0
    assert count_shared_feature_records(X_train, X_test) == 0
    assert count_shared_feature_records(X_val, X_test) == 0


def test_conflicting_label_groups_stay_in_one_partition(make_df):
    df = make_df(n=400)
    conflicts = df.iloc[:60].copy()
    conflicts[ID_COL] += 5000
    conflicts[TARGET_COL] = 1 - conflicts[TARGET_COL]
    clean, report, _ = deduplicate(pd.concat([df, conflicts], ignore_index=True), "keep_grouped")
    assert report["n_conflicting_label_groups"] == 60

    X_train, X_val, X_test, *_ = split_xy(clean.drop(columns=[ID_COL]), 0.2, 0.2, 3)
    assert count_shared_feature_records(X_train, X_test) == 0
    assert count_shared_feature_records(X_train, X_val) == 0
    assert count_shared_feature_records(X_val, X_test) == 0
