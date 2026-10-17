import numpy as np
import pandas as pd
import pytest

from utils import ID_COL, TARGET_COL, audit_duplicates, deduplicate


def with_conflicts(df, n):
    conflicts = df.iloc[:n].copy()
    conflicts[ID_COL] += 9000
    conflicts[TARGET_COL] = 1 - conflicts[TARGET_COL]
    return pd.concat([df, conflicts], ignore_index=True)


def test_audit_excludes_id_from_identity_and_reports_counts(make_df):
    df = make_df(n=200, n_duplicates=30, id_offset=1470)
    report = audit_duplicates(df)

    assert ID_COL not in report["feature_identity_columns"]
    assert TARGET_COL not in report["feature_identity_columns"]
    assert report["customer_id_unique"] is True
    assert report["n_redundant_identical_records"] == 30
    assert report["n_duplicate_groups"] == 30
    assert report["n_rows_in_duplicate_groups"] == 60
    assert report["group_size_distribution"] == {1: 170, 2: 30}
    assert report["n_conflicting_label_groups"] == 0
    assert report["id_gap_within_duplicate_groups"]["most_common_gap"] == 1470


def test_missing_values_count_as_equal_for_identity(make_df):
    df = make_df(n=50)
    df.loc[0, "Tenure"] = np.nan
    copy = df.iloc[[0]].copy()
    copy[ID_COL] = 99999
    report = audit_duplicates(pd.concat([df, copy], ignore_index=True))
    assert report["n_redundant_identical_records"] == 1


def test_deduplicate_keeps_lowest_id_and_logs_removals(make_df):
    df = make_df(n=100, n_duplicates=10)
    shuffled = df.sample(frac=1, random_state=0)
    clean, report, removed = deduplicate(shuffled)

    assert len(clean) == 100
    assert report["n_removed_identical_records"] == 10
    assert set(removed[ID_COL]) == set(range(51001, 51011))
    assert (removed["kept_customer_id"] == removed[ID_COL] - 1000).all()
    assert (removed["removal_reason"] == "identical_features_and_label").all()
    assert list(clean.index) == [i for i in shuffled.index if i in clean.index]


def test_conflicting_labels_detected(make_df):
    report = audit_duplicates(with_conflicts(make_df(n=100), 5))
    assert report["n_conflicting_label_groups"] == 5
    assert report["n_rows_in_conflicting_groups"] == 10


def test_keep_grouped_retains_both_labels(make_df):
    df = with_conflicts(make_df(n=100), 5)
    clean, report, removed = deduplicate(df, "keep_grouped")
    assert len(clean) == 105 and removed.empty
    assert report["n_removed_conflicting_records"] == 0


def test_drop_policy_removes_and_logs_conflicts(make_df):
    df = with_conflicts(make_df(n=100), 5)
    clean, report, removed = deduplicate(df, "drop")
    assert len(clean) == 95
    assert report["n_removed_conflicting_records"] == 10
    assert (removed["removal_reason"] == "conflicting_label_group").sum() == 10


def test_error_policy_raises(make_df):
    with pytest.raises(ValueError, match="conflicting"):
        deduplicate(with_conflicts(make_df(n=100), 2), "error")


def test_identical_and_conflicting_records_together(make_df):
    df = make_df(n=100)
    exact = df.iloc[[0]].assign(**{ID_COL: 70000})
    flipped = df.iloc[[0]].assign(**{ID_COL: 70001, TARGET_COL: 1 - df.loc[0, TARGET_COL]})
    clean, report, removed = deduplicate(pd.concat([df, exact, flipped], ignore_index=True))
    assert report["n_removed_identical_records"] == 1
    assert report["n_conflicting_label_groups"] == 1
    assert len(clean) == 101


def test_invalid_policy_rejected(customers):
    with pytest.raises(ValueError, match="conflict_policy"):
        deduplicate(customers, "first")
