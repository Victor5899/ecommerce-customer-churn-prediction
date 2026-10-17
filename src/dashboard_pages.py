"""Streamlit page renderers for the churn analytics dashboard."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import streamlit as st

import charts
import dashboard as db
from inference import (
    PREDICTION_COL, PROBABILITY_COL, DataValidationError, FeatureSchema, file_extension,
    prepare_features, resolve_sheet, score_prepared,
)
from utils import DATA_SHEET, ID_COL, TARGET_COL

BATCH_KEY = "batch"
CAUSATION_NOTE = ("These are observational patterns in historical data. They describe which customers "
                  "churned, not why, and do not establish that changing a feature would change churn.")


@dataclass
class DashboardContext:
    model: object
    model_sha: str
    model_name: str
    schema: FeatureSchema
    metadata: Optional[dict]
    threshold: Optional[float]
    metadata_error: Optional[str]
    reference: Optional[pd.DataFrame]
    reference_error: Optional[str]
    output_dir: Path
    model_path: Path
    train_command: str
    excel_sheets: Callable
    parse_upload: Callable


def _batch() -> Optional[dict]:
    return st.session_state.get(BATCH_KEY)


def _pct(value: Optional[float], digits: int = 1) -> str:
    return "—" if value is None else f"{value:.{digits}%}"


def _page_header(title: str, description: str) -> None:
    st.title(title)
    st.caption(description)


def _reference_or_warn(ctx: DashboardContext) -> Optional[pd.DataFrame]:
    if ctx.reference is None:
        st.warning(f"The historical dataset is unavailable ({ctx.reference_error}). "
                   "Historical analytics cannot be shown.")
    return ctx.reference


# --------------------------------------------------------------------------- #
# A. Executive overview
# --------------------------------------------------------------------------- #
def executive_overview(ctx: DashboardContext) -> None:
    _page_header("Executive Overview",
                 "Actual churn in the historical dataset alongside model predictions for the "
                 "most recently scored batch. The two are calculated separately.")
    df = _reference_or_warn(ctx)
    batch = _batch()
    summary = db.actual_churn(df) if df is not None else None

    hist, pred = st.columns(2)
    with hist.container(border=True):
        st.markdown("**Historical outcomes** · actual churn labels")
        c1, c2 = st.columns(2)
        c1.metric("Customer records", f"{summary.n_customers:,}" if summary else "—")
        c2.metric("Actual churn rate", _pct(summary.rate if summary else None),
                  help="Share of historical records with Churn = 1. Never derived from predictions.")
        if ctx.metadata is not None:
            data_info = ctx.metadata["data"]
            st.caption(f"{data_info['n_rows_after_deduplication']:,} unique records were used for "
                       f"training after removing {data_info['deduplication']['n_removed_identical_records']} "
                       f"exact duplicates (churn rate {data_info['deduplication']['churn_rate_after']:.1%}).")
    with pred.container(border=True):
        st.markdown("**Current scoring batch** · model predictions")
        c1, c2 = st.columns(2)
        if batch is None:
            c1.metric("Customers scored", "—")
            c2.metric("Predicted to churn", "—")
            st.caption("No file has been scored in this session. Use **Churn Prediction** to score one.")
        else:
            bs = db.batch_summary(batch["results"])
            c1.metric("Customers scored", f"{bs.n_customers:,}")
            c2.metric("Predicted to churn",
                      "—" if bs.n_predicted_churn is None else f"{bs.n_predicted_churn:,}",
                      None if bs.predicted_rate is None else f"{bs.predicted_rate:.1%} of batch",
                      delta_color="off")
            st.caption(f"File: `{batch['filename']}`. Predictions are model estimates, not outcomes.")

    if df is None:
        return

    left, right = st.columns([1, 2])
    with left:
        st.subheader("Churn distribution")
        st.altair_chart(charts.churn_donut(summary.n_customers - summary.n_churned, summary.n_churned))
    with right:
        st.subheader("Key observations")
        for note in db.key_observations(df):
            st.markdown(f"- {note}")
        st.caption(f"Segments with fewer than {db.MIN_SEGMENT_SIZE} customers are ignored. {CAUSATION_NOTE}")

    st.subheader("Actual churn by segment")
    columns = [c for c in db.OVERVIEW_BREAKDOWNS if c in df.columns]
    for start in range(0, len(columns), 2):
        for slot, column in zip(st.columns(2), columns[start:start + 2]):
            with slot.container(border=True):
                st.altair_chart(charts.churn_rate_bars(
                    db.churn_rate_by(df, column), f"Churn rate by {db.OVERVIEW_BREAKDOWNS[column]}",
                    summary.rate))
    st.caption("Dashed line: overall historical churn rate. Hover over bars for segment sizes.")


# --------------------------------------------------------------------------- #
# B. Customer analytics
# --------------------------------------------------------------------------- #
def _analytics_filters(df: pd.DataFrame) -> pd.DataFrame:
    with st.expander("Segment filters", expanded=True):
        cols = st.columns(4)
        ranges, selections = {}, {}
        if "Tenure" in df.columns and df["Tenure"].notna().any():
            lo, hi = int(df["Tenure"].min()), int(df["Tenure"].max())
            if lo < hi:
                ranges["Tenure"] = cols[0].slider("Tenure (months)", lo, hi, (lo, hi))
        for slot, column, label in zip(cols[1:], ["CityTier", "PreferedOrderCat", "Complain"],
                                       ["City tier", "Preferred order category", "Complaint filed"]):
            if column in df.columns:
                options = sorted(df[column].dropna().unique().tolist())
                chosen = slot.multiselect(label, options, default=options,
                                          format_func=lambda v, c=column: db.VALUE_LABELS.get(c, {}).get(v, str(v)))
                if len(chosen) < len(options):
                    selections[column] = chosen
        include_missing = st.checkbox("Include customers with missing values in filtered columns", True)
    return db.filter_customers(df, ranges, selections, include_missing)


def customer_analytics(ctx: DashboardContext) -> None:
    _page_header("Customer Analytics",
                 "Explore customer features and, where actual outcomes are available, how churn "
                 "varies across segments.")
    sources = {}
    if ctx.reference is not None:
        sources["Historical dataset (actual churn labels)"] = ctx.reference
    batch = _batch()
    if batch is not None:
        sources[f"Current scoring batch ({batch['filename']})"] = batch["results"]
    if not sources:
        st.warning(f"No data to analyse: the historical dataset is unavailable ({ctx.reference_error}) "
                   "and no file has been scored in this session.")
        return
    source = st.radio("Data source", list(sources), horizontal=True)
    data = sources[source]
    filtered = _analytics_filters(data)
    labelled = db.labelled_target(data) is not None

    if filtered.empty:
        st.info("No customers match the selected filters. Widen the filters to see results.")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Customers in selection", f"{len(filtered):,}", f"{len(filtered) / len(data):.0%} of source",
              delta_color="off")
    if labelled:
        m2.metric("Actual churn rate (selection)", _pct(db.actual_churn(filtered).rate))
        m3.metric("Actual churn rate (all)", _pct(db.actual_churn(data).rate))
    else:
        st.info(f"This source has no complete `{TARGET_COL}` column, so actual churn patterns are not "
                "shown. Distributions describe the uploaded customers only.")

    features = [c for c in ctx.schema.features if c in filtered.columns]
    default = features.index("Tenure") if "Tenure" in features else 0
    feature = st.selectbox("Feature", features, index=default,
                           help="Numeric features with many values are grouped into equal-width bins.")
    missing = int(filtered[feature].isna().sum())

    left, right = st.columns(2)
    with left.container(border=True):
        st.altair_chart(charts.distribution_bars(db.feature_distribution(filtered, feature),
                                                 f"Customers by {feature}"))
    if labelled:
        with right.container(border=True):
            st.altair_chart(charts.churn_rate_bars(db.churn_rate_by(filtered, feature),
                                                   f"Actual churn rate by {feature}",
                                                   db.actual_churn(filtered).rate))
    st.caption(f"{missing:,} customer(s) in the selection have no value for {feature}; they are shown "
               f"as '{db.MISSING_LABEL}'." if missing else f"No missing values for {feature} in the selection.")

    if labelled:
        with st.expander("Median feature values: retained vs churned"):
            st.dataframe(db.profile_by_churn(filtered, features).style.format("{:.2f}"))
    st.caption(CAUSATION_NOTE)


# --------------------------------------------------------------------------- #
# C. Churn prediction
# --------------------------------------------------------------------------- #
def _threshold_banner(ctx: DashboardContext) -> None:
    if ctx.threshold is not None:
        st.info(
            f"**Decision threshold: {ctx.threshold:.2f}.** A customer is predicted to churn when the "
            f"churn probability is at least {ctx.threshold:.0%}. The threshold was chosen on the "
            f"validation set ({ctx.metadata['threshold']['objective']}); the saved test metrics use "
            "the same threshold."
        )
    else:
        st.error(
            f"**Decision threshold unavailable.** {ctx.metadata_error} Churn probabilities are still "
            "calculated, but customers are not classified as churn or no churn, and saved evaluation "
            f"results are hidden because they may not describe this model. Fix: `{ctx.train_command}`"
        )


def churn_prediction(ctx: DashboardContext) -> None:
    _page_header("Churn Prediction",
                 f"Score a batch of customers with the trained {ctx.model_name} model.")
    _threshold_banner(ctx)

    uploaded_file = st.file_uploader("Upload customer data", type=["csv", "xlsx"])
    with st.expander("Expected file format"):
        st.markdown(
            f"CSV or XLSX with one row per customer and these **{len(ctx.schema.features)}** columns "
            f"(names are case-sensitive). `{ID_COL}` is optional and kept in the results; "
            f"`{TARGET_COL}` is optional, ignored for prediction, and used only to report actual churn "
            f"in the file. For workbooks, the `{DATA_SHEET}` sheet is used when present. Missing "
            "values are allowed."
        )
        st.code(", ".join(ctx.schema.features))

    if uploaded_file is None:
        batch = _batch()
        if batch is None:
            st.info("Upload a CSV or XLSX file to score customers. Results stay available on the other "
                    "pages for the rest of this session.")
            return
        st.caption(f"Showing results for the last scored file, `{batch['filename']}`. "
                   "Upload a file to replace them.")
        if st.button("Clear results"):
            st.session_state.pop(BATCH_KEY, None)
            st.rerun()
        _render_batch(ctx, batch)
        return

    raw = uploaded_file.getvalue()
    sheet = None
    try:
        if file_extension(uploaded_file.name) == ".xlsx":
            sheets = ctx.excel_sheets(raw)
            sheet = resolve_sheet(sheets)
            if sheet is None:
                sheet = st.selectbox(
                    f"This workbook has several sheets and none is named '{DATA_SHEET}'. "
                    "Choose the sheet that contains the customer records:",
                    sheets,
                )
            elif len(sheets) > 1:
                st.caption(f"Using sheet '{sheet}' (workbook sheets: {', '.join(sheets)}).")
        df = ctx.parse_upload(raw, uploaded_file.name, sheet)
        X, report = prepare_features(df, ctx.schema)
    except DataValidationError as exc:
        st.error(f"**This file cannot be scored.** {exc}")
        return
    except Exception as exc:
        st.error(f"**Unexpected error while reading the file:** {exc}")
        return

    with st.expander(f"Preview uploaded data ({len(df):,} rows, {df.shape[1]} columns)"):
        st.dataframe(df.head(20))
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

    key = (hashlib.sha256(raw).hexdigest(), sheet, ctx.threshold, ctx.model_sha)
    if st.button("Predict Churn", type="primary"):
        try:
            with st.spinner(f"Scoring {len(df):,} customers..."):
                results, _ = score_prepared(ctx.model, df, X, ctx.threshold)
        except Exception as exc:
            st.error(f"**Prediction failed:** {exc}")
            return
        st.session_state[BATCH_KEY] = {
            "key": key, "filename": uploaded_file.name, "results": results, "report": report,
            "quality": db.upload_quality(df, ctx.schema.features),
        }

    batch = _batch()
    if not batch or batch["key"] != key:
        st.caption("Click **Predict Churn** to score this file.")
        return
    _render_batch(ctx, batch)


def _render_batch(ctx: DashboardContext, batch: dict) -> None:
    results = batch["results"]
    summary = db.batch_summary(results)

    st.subheader("Prediction summary")
    m = st.columns(4)
    m[0].metric("Customers scored", f"{summary.n_customers:,}")
    if summary.n_predicted_churn is not None:
        m[1].metric("Predicted to churn", f"{summary.n_predicted_churn:,}",
                    f"{summary.predicted_rate:.1%} of batch", delta_color="off")
        m[2].metric("Predicted to stay", f"{summary.n_customers - summary.n_predicted_churn:,}")
    m[3].metric("Mean churn probability", f"{summary.mean_probability:.1f}%")
    if summary.actual is not None:
        st.caption(f"This file also contains actual `{TARGET_COL}` labels: {summary.actual.n_churned:,} "
                   f"customers ({summary.actual.rate:.1%}) actually churned. Labels were not used for scoring.")

    low_default, high_default = db.default_risk_bounds(ctx.threshold)
    low, high = st.slider("Risk band boundaries (churn probability, %)", 1, 99,
                          (low_default, high_default), key="risk_bounds",
                          help="Presentation only: Low < first value ≤ Medium < second value ≤ High. "
                               "The churn prediction always uses the model's decision threshold.")
    if low >= high:
        st.warning("The two risk boundaries must differ. Move the slider handles apart.")
        return
    scored = results.copy()
    scored[db.RISK_COL] = db.assign_risk_segments(scored[PROBABILITY_COL], low, high)
    threshold_note = (f"the decision threshold is {ctx.threshold:.0%}" if ctx.threshold is not None
                      else "no decision threshold is available")
    st.caption(f"Low < {low}% ≤ Medium < {high}% ≤ High. Bands do not change predictions; "
               f"{threshold_note}.")

    counts = db.risk_segment_counts(scored[db.RISK_COL])
    seg_cols = st.columns(3)
    for slot, row in zip(seg_cols, counts.itertuples()):
        slot.metric(f"{row.Segment} risk", f"{row.Customers:,}", f"{row.Share:.1%}", delta_color="off")
    with st.container(border=True):
        st.altair_chart(charts.risk_histogram(
            scored[PROBABILITY_COL], scored[db.RISK_COL],
            None if ctx.threshold is None else ctx.threshold * 100))

    st.subheader("Customer risk list")
    f1, f2, f3, f4 = st.columns([2, 2, 1.5, 1.5])
    chosen = f1.multiselect("Risk segments", list(db.RISK_SEGMENTS), default=list(db.RISK_SEGMENTS))
    min_p = f2.slider("Minimum churn probability (%)", 0, 100, 0)
    sort = f3.selectbox("Sort by risk", ["desc", "asc", "original"],
                        format_func={"desc": "Highest first", "asc": "Lowest first",
                                     "original": "Original file order"}.get)
    predicted_only = f4.checkbox("Predicted churners only", disabled=ctx.threshold is None)
    view = db.filter_by_risk(scored, chosen, min_p, predicted_only, sort)

    lead = [c for c in (ID_COL, PROBABILITY_COL, db.RISK_COL, PREDICTION_COL) if c in view.columns]
    ordered = view[lead + [c for c in view.columns if c not in lead]]
    st.caption(f"Showing {len(view):,} of {len(scored):,} customers.")
    if view.empty:
        st.info("No customers match these filters.")
    else:
        st.dataframe(ordered, hide_index=True, column_config={
            PROBABILITY_COL: st.column_config.ProgressColumn(
                PROBABILITY_COL, format="%.1f%%", min_value=0, max_value=100),
            PREDICTION_COL: st.column_config.NumberColumn(
                PREDICTION_COL, help="1 = predicted to churn at the decision threshold"),
        })

    d1, d2 = st.columns(2)
    d1.download_button("Download all results (original order)",
                       scored.to_csv(index=False).encode("utf-8"),
                       file_name="predictions.csv", mime="text/csv", on_click="ignore")
    d2.download_button("Download filtered view", ordered.to_csv(index=False).encode("utf-8"),
                       file_name="predictions_filtered.csv", mime="text/csv",
                       on_click="ignore", disabled=view.empty)
    st.caption("Downloads contain every uploaded column (including customer IDs) plus the churn "
               f"probability, prediction and the risk segment for the current band boundaries.")


# --------------------------------------------------------------------------- #
# D. Model performance
# --------------------------------------------------------------------------- #
def _metric_row(metrics: dict) -> None:
    cols = st.columns(6)
    for slot, (label, key, fmt) in zip(cols, [
        ("Precision", "precision", "{:.1%}"), ("Recall", "recall", "{:.1%}"), ("F1-score", "f1", "{:.1%}"),
        ("ROC-AUC", "roc_auc", "{:.3f}"), ("PR-AUC", "pr_auc", "{:.3f}"), ("Accuracy", "accuracy", "{:.1%}"),
    ]):
        value = metrics.get(key)
        slot.metric(label, "—" if value is None else fmt.format(value))


def model_performance(ctx: DashboardContext) -> None:
    _page_header("Model Performance",
                 "Saved evaluation results from the last training run. Nothing on this page is "
                 "recomputed or retrained.")
    if ctx.metadata is None:
        st.warning(f"Saved evaluation results are hidden because the model metadata is unavailable "
                   f"or does not match the loaded model. {ctx.metadata_error or ''} "
                   f"Regenerate them with `{ctx.train_command}`.")
        return
    meta = ctx.metadata
    for issue in db.artifact_issues(ctx.output_dir, meta, ctx.model_path):
        st.warning(issue)

    selection, thr = meta["model_selection"], meta["threshold"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Selected model", selection["selected_model_name"])
    c2.metric("Decision threshold", f"{thr['value']:.2f}")
    c3.metric("Trained (UTC)", meta.get("created_at_utc", "—")[:16].replace("T", " "))
    st.caption(f"Threshold objective: {thr['objective']}. Rule: {thr['rule']}.")

    test_tab, cv_tab, val_tab, plot_tab = st.tabs(
        ["Held-out test set", "Cross-validation & comparison", "Validation set", "Saved plots"])

    with test_tab:
        test = meta.get("test_metrics")
        if not test:
            st.warning("Test metrics are missing from the metadata.")
        else:
            st.caption(f"{test['n_samples']:,} customers ({test['n_positive']} churners), not used for "
                       f"model selection or threshold tuning. Classification metrics at threshold "
                       f"{test['threshold']:.2f}.")
            _metric_row(test)
            left, right = st.columns([1, 1])
            with left.container(border=True):
                st.markdown("**Confusion matrix (test set)**")
                if "confusion_matrix" in test:
                    st.altair_chart(charts.confusion_matrix_chart(test["confusion_matrix"]))
                else:
                    st.caption("Confusion matrix not recorded in the metadata.")
            with right:
                cm = test.get("confusion_matrix")
                if cm:
                    st.markdown(
                        f"- **{cm['tp']}** of {cm['tp'] + cm['fn']} actual churners were flagged "
                        f"(recall {test['recall']:.1%}).\n"
                        f"- **{cm['fp']}** of {cm['tn'] + cm['fp']} retained customers were flagged "
                        f"incorrectly (false positives).\n"
                        f"- **{cm['tp']}** of {cm['tp'] + cm['fp']} flagged customers actually "
                        f"churned (precision {test['precision']:.1%})."
                    )
                ref = meta.get("test_metrics_reference_threshold_0_5")
                if ref:
                    with st.expander("Reference: test metrics at the default 0.5 threshold"):
                        st.dataframe(pd.DataFrame(
                            {"Tuned threshold": test, "0.5 threshold": ref}
                        ).loc[["threshold", "precision", "recall", "f1", "accuracy"]].T,
                            column_config={c: st.column_config.NumberColumn(format="%.3f")
                                           for c in ["threshold", "precision", "recall", "f1", "accuracy"]})
                        st.caption("Shown for context only. The dashboard uses the tuned threshold.")

    with cv_tab:
        candidates = pd.DataFrame(meta.get("candidates") or [])
        if candidates.empty:
            st.warning("No candidate model comparison is recorded in the metadata.")
        else:
            st.caption(f"Selection criterion: {selection['description']} "
                       f"CV scheme: {selection['cv']}, {selection['n_iter']} random-search iterations per model.")
            with st.container(border=True):
                st.altair_chart(charts.cv_comparison_chart(candidates))
            table = candidates[[
                "model_name", "selected", "cv_pr_auc_mean", "cv_pr_auc_std", "validation_threshold",
                "validation_pr_auc", "validation_roc_auc", "validation_precision", "validation_recall",
                "validation_f1",
            ]].rename(columns={"model_name": "Model", "selected": "Selected"})
            st.dataframe(table, hide_index=True, column_config={
                c: st.column_config.NumberColumn(format="%.3f") for c in table.columns[2:]})
            st.caption("CV PR-AUC is computed on the training partition only. Validation metrics use "
                       "each model's own validation-tuned threshold, so they are optimistic and only "
                       "for comparison. Test results are reported for the selected model only.")

    with val_tab:
        val = meta.get("validation_metrics")
        if not val:
            st.warning("Validation metrics are missing from the metadata.")
        else:
            st.caption(f"{val['n_samples']:,} customers ({val['n_positive']} churners). The threshold was "
                       "tuned on this partition, so these numbers are optimistic; use the test tab for "
                       "an unbiased estimate.")
            _metric_row(val)

    with plot_tab:
        plots = [("confusion_matrix.png", "Confusion matrix"), ("feature_importance.png", "Feature importance"),
                 ("roc_curve.png", "ROC curve (test)"), ("pr_curve.png", "Precision-recall curve (test)")]
        for start in range(0, len(plots), 2):
            for slot, (filename, caption) in zip(st.columns(2), plots[start:start + 2]):
                path = ctx.output_dir / filename
                with slot:
                    if not path.exists():
                        st.caption(f"{caption} not found ({filename}).")
                        continue
                    try:
                        st.image(str(path), caption=caption)
                    except Exception as exc:
                        st.caption(f"{caption} could not be displayed ({exc}).")


# --------------------------------------------------------------------------- #
# E. Data quality
# --------------------------------------------------------------------------- #
def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def data_quality(ctx: DashboardContext) -> None:
    _page_header("Data Quality",
                 "Findings recorded when the model was trained, kept separate from checks on the "
                 "most recently scored upload. The source workbook is never modified.")
    training_tab, upload_tab = st.tabs(["Training data (source workbook)", "Current upload"])
    with training_tab:
        _training_quality(ctx)
    with upload_tab:
        _upload_quality(ctx)


def _training_quality(ctx: DashboardContext) -> None:
    meta = ctx.metadata
    report = _read_json(ctx.output_dir / "data_quality_report.json")
    if report is None and meta is not None:
        report = meta["data"].get("deduplication")
    df = ctx.reference

    c = st.columns(4)
    c[0].metric("Raw records", f"{len(df):,}" if df is not None else "—")
    c[1].metric("Columns", f"{df.shape[1]}" if df is not None else "—")
    c[2].metric("Exact duplicates removed", f"{report['n_removed_identical_records']:,}" if report else "—")
    c[3].metric("Records used for training", f"{report['n_rows_after_deduplication']:,}" if report else "—")
    if df is None:
        st.warning(f"The source workbook could not be loaded ({ctx.reference_error}).")

    left, right = st.columns(2)
    with left.container(border=True):
        st.markdown("**Missing values in the source workbook**")
        if df is not None:
            missing = db.missing_value_table(df)
            if missing.empty:
                st.caption("No missing values.")
            else:
                st.altair_chart(charts.missing_values_chart(missing))
                st.caption(f"{int(df.isna().any(axis=1).sum()):,} records have at least one missing value. "
                           "They are kept; the model pipeline imputes them (median / most frequent value) "
                           "using training data only.")
    with right.container(border=True):
        st.markdown("**Duplicate handling**")
        if report is None:
            st.warning("No duplicate-removal report was found (`outputs/data_quality_report.json`).")
        else:
            gap = report.get("id_gap_within_duplicate_groups") or {}
            st.markdown(
                f"- {report['n_duplicate_groups']:,} groups of records share every feature value; "
                f"{report['n_removed_identical_records']:,} redundant copies were removed.\n"
                f"- Duplicates differ only in `{ID_COL}`; {report['n_conflicting_label_groups']} groups "
                f"have conflicting churn labels.\n"
                f"- Churn rate {report['churn_rate_before']:.2%} before and "
                f"{report['churn_rate_after']:.2%} after de-duplication."
                + (f"\n- In {gap['groups_with_most_common_gap']} groups the two IDs are exactly "
                   f"{gap['most_common_gap']:,} apart, which suggests copied records rather than "
                   "distinct customers." if gap.get("most_common_gap") else "")
            )
            st.caption("Records are compared on all features with missing values treated as equal. "
                       "Remaining identical records are kept in the same split partition.")

    removed_path = ctx.output_dir / "removed_duplicates.csv"
    with st.expander("Removed duplicate records"):
        if removed_path.exists():
            removed = pd.read_csv(removed_path)
            st.dataframe(removed, hide_index=True)
            st.download_button("Download removed_duplicates.csv", removed.to_csv(index=False).encode("utf-8"),
                               file_name="removed_duplicates.csv", mime="text/csv", on_click="ignore")
        else:
            st.caption("`removed_duplicates.csv` not found.")

    if meta is not None:
        split = meta["data"]["split"]
        with st.expander("Train / validation / test split"):
            st.dataframe(pd.DataFrame(split["partitions"]).T.rename_axis("Partition"),
                         column_config={"churn_rate": st.column_config.NumberColumn(format="%.3f")})
            st.caption(f"Method: {split['method']}.")

    st.subheader("Known limitations")
    notes = [
        "The dataset is a single snapshot with no observation dates, so churn cannot be related to time "
        "and the model cannot account for changes in customer behaviour.",
        "Classes are imbalanced (roughly 1 churner in 6), which is why PR-AUC and recall are reported "
        "alongside accuracy.",
        "Relationships in the data are observational and do not establish causes of churn.",
    ]
    aliases = db.category_aliases(ctx.schema.categories)
    if aliases:
        pairs = ", ".join(f"{col}: '{a}' / '{b}'" for col, a, b in aliases)
        notes.append(f"Some categories look like alternative spellings of the same value ({pairs}). "
                     "They are kept as recorded, so the model treats them as separate categories.")
    st.markdown("\n".join(f"- {n}" for n in notes))


def _upload_quality(ctx: DashboardContext) -> None:
    batch = _batch()
    if batch is None:
        st.info("No file has been scored in this session. Score a file on **Churn Prediction** to see "
                "its data-quality checks here.")
        return
    q, report = batch["quality"], batch["report"]
    st.caption(f"File: `{batch['filename']}`. These checks describe the uploaded file only.")
    c = st.columns(4)
    c[0].metric("Rows", f"{q['n_rows']:,}")
    c[1].metric("Columns", f"{q['n_columns']}")
    c[2].metric("Rows with missing features", f"{q['rows_with_missing']:,}")
    c[3].metric("Identical feature rows", f"{q['duplicate_feature_rows']:,}",
                help="Rows whose feature values repeat an earlier row. They are scored, not removed.")

    findings = []
    if q["exact_duplicate_rows"]:
        findings.append(f"{q['exact_duplicate_rows']:,} rows are exact copies of an earlier row.")
    if q["has_ids"]:
        if q["duplicate_ids"]:
            findings.append(f"{q['duplicate_ids']:,} rows share a `{ID_COL}` with another row.")
        if q["missing_ids"]:
            findings.append(f"{q['missing_ids']:,} rows have no `{ID_COL}`.")
    else:
        findings.append(f"No `{ID_COL}` column; rows are identified by their position in the file.")
    if report.unseen_categories:
        findings.append("Unseen categories: " + "; ".join(
            f"{c}: {', '.join(v)}" for c, v in report.unseen_categories.items()) + ".")
    if report.extra_columns:
        findings.append(f"Extra columns kept but not used: {', '.join(report.extra_columns)}.")
    findings.append(f"Actual `{TARGET_COL}` labels: {'present' if q['has_labels'] else 'not present or incomplete'}.")
    st.markdown("\n".join(f"- {f}" for f in findings))

    if not q["missing"].empty:
        with st.container(border=True):
            st.markdown("**Missing values in model features**")
            st.altair_chart(charts.missing_values_chart(q["missing"]))
