"""Streamlit page renderers for the churn analytics dashboard."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import streamlit as st

import charts
import compatibility as compat
import dashboard as db
import explainability as xai
import retention as rt
from inference import (
    CALIBRATED_COL, PREDICTION_COL, PROBABILITY_COL, DataValidationError, FeatureSchema,
    file_extension, prepare_features, read_column_names, resolve_sheet, score_prepared,
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
    explainability: Optional[dict] = None
    background: Optional[pd.DataFrame] = None
    explainability_error: Optional[str] = "Explainability artifacts were not loaded."
    calibration: Optional[dict] = None
    calibrator: Optional[dict] = None
    calibration_error: Optional[str] = "Calibration artifacts were not loaded."
    xai_command: str = "python src/explain_calibrate.py --input <dataset>"
    xai_warnings: list = field(default_factory=list)


@st.cache_data(show_spinner=False, max_entries=64)
def _explain_cached(_model, model_sha: str, row: pd.DataFrame, background: pd.DataFrame):
    return xai.shapley_values(_model, row, background)


@st.cache_data(show_spinner=False, max_entries=8)
def _column_names(raw: bytes, filename: str, sheet) -> list:
    return read_column_names(raw, filename, sheet)


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
            f"model churn score (uncalibrated) is at least {ctx.threshold:.0%}. The threshold was chosen on the "
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
    digest = hashlib.sha256(raw).hexdigest()
    sheet, df = None, None
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
        header = _column_names(raw, uploaded_file.name, sheet)
        check = compat.check_compatibility(None, ctx.schema, header)
        if check.status != compat.INCOMPATIBLE:
            df = ctx.parse_upload(raw, uploaded_file.name, sheet)
            check = compat.check_compatibility(df, ctx.schema, header)
    except DataValidationError as exc:
        st.error(f"**This file cannot be scored.** {exc}")
        return
    except Exception as exc:
        st.error(f"**Unexpected error while reading the file:** {exc}")
        return
    df, fixed = _resolve_compatibility(ctx, check, df, digest)
    if df is None:
        return
    try:
        X, report = prepare_features(df, ctx.schema)
    except DataValidationError as exc:
        st.error(f"**This file cannot be scored.** {exc}")
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

    calibration_id = (ctx.calibration or {}).get("calibrator_sha256") if ctx.calibrator else None
    key = (digest, sheet, ctx.threshold, ctx.model_sha, calibration_id, fixed)
    if st.button("Predict Churn", type="primary"):
        try:
            with st.spinner(f"Scoring {len(df):,} customers..."):
                results, _ = score_prepared(ctx.model, df, X, ctx.threshold, ctx.calibrator)
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


def _render_compatibility(ctx: DashboardContext, check: compat.CompatibilityReport) -> None:
    n = len(ctx.schema.features)
    if check.status == compat.INCOMPATIBLE:
        st.error(f"**Compatibility: {check.status}.** This file cannot be scored. " + " ".join(check.problems))
    elif check.status == compat.NEEDS_MAPPING:
        st.warning(f"**Compatibility: {check.status}.** All {n} required features can be provided after "
                   "the changes below are reviewed and confirmed.")
    else:
        target = (f"A complete 0/1 `{TARGET_COL}` column was found; it is used only to report actual churn."
                  if check.target_usable else f"No usable `{TARGET_COL}` column; predictions only.")
        st.success(f"**Compatibility: {check.status}.** All {n} required feature columns are present with "
                   f"usable values. {target}")
    notes = check.hints + check.warnings
    if notes:
        with st.expander(f"Compatibility notes ({len(notes)})"):
            st.markdown("\n".join(f"- {note}" for note in notes))


def _resolve_compatibility(ctx: DashboardContext, check: compat.CompatibilityReport,
                           df: Optional[pd.DataFrame], digest: str):
    """Return ``(dataframe to score, fixes applied)``, or ``(None, False)`` to stop here."""
    _render_compatibility(ctx, check)
    if check.status == compat.INCOMPATIBLE:
        return None, False
    if check.status == compat.READY:
        return df, False
    st.markdown("**Proposed changes**")
    st.markdown("\n".join(f"- {fix.detail}" for fix in check.fixes))
    st.dataframe(compat.preview_fixes(df, check.fixes), hide_index=True)
    st.caption("Preview of affected columns. The changes apply only to this session's copy of the upload; "
               "your file is not modified.")
    if not st.checkbox("Apply these changes and continue", key=f"compat_confirm_{digest}"):
        st.info("Nothing has been changed. Confirm the proposed changes above to score this file.")
        return None, False
    fixed = compat.apply_fixes(df, check.fixes)
    recheck = compat.check_compatibility(fixed, ctx.schema)
    if recheck.status != compat.READY:
        st.error("**The file is still not ready after these changes.** " + " ".join(recheck.problems))
        return None, False
    return fixed, True


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
    m[3].metric("Mean model churn score", f"{summary.mean_probability:.1f}%",
                help="Average uncalibrated model score, not an expected churn rate.")
    if summary.actual is not None:
        st.caption(f"This file also contains actual `{TARGET_COL}` labels: {summary.actual.n_churned:,} "
                   f"customers ({summary.actual.rate:.1%}) actually churned. Labels were not used for scoring.")

    low_default, high_default = db.default_risk_bounds(ctx.threshold)
    low, high = st.slider("Risk band boundaries (model churn score, %)", 1, 99,
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
    min_p = f2.slider("Minimum model churn score (%)", 0, 100, 0)
    sort = f3.selectbox("Sort by risk", ["desc", "asc", "original"],
                        format_func={"desc": "Highest first", "asc": "Lowest first",
                                     "original": "Original file order"}.get)
    predicted_only = f4.checkbox("Predicted churners only", disabled=ctx.threshold is None)
    view = db.filter_by_risk(scored, chosen, min_p, predicted_only, sort)

    lead = [c for c in (ID_COL, PROBABILITY_COL, CALIBRATED_COL, db.RISK_COL, PREDICTION_COL)
            if c in view.columns]
    ordered = view[lead + [c for c in view.columns if c not in lead]]
    st.caption(f"Showing {len(view):,} of {len(scored):,} customers.")
    if view.empty:
        st.info("No customers match these filters.")
    else:
        st.dataframe(ordered, hide_index=True, column_config={
            PROBABILITY_COL: st.column_config.ProgressColumn(
                PROBABILITY_COL, format="%.1f%%", min_value=0, max_value=100,
                help="Model churn score (uncalibrated). Predictions and risk bands use this score."),
            CALIBRATED_COL: st.column_config.NumberColumn(
                CALIBRATED_COL, format="%.1f%%",
                help="Validation-calibrated probability estimate. Shown for interpretation only."),
            PREDICTION_COL: st.column_config.NumberColumn(
                PREDICTION_COL, help="1 = predicted to churn at the decision threshold"),
        })
    if CALIBRATED_COL in scored.columns:
        st.caption(f"`{PROBABILITY_COL}` is the model's raw score; `{CALIBRATED_COL}` rescales it so "
                   "values better match observed churn frequencies. Customer ranking and predictions "
                   "are unchanged.")

    d1, d2 = st.columns(2)
    d1.download_button("Download all results (original order)",
                       scored.to_csv(index=False).encode("utf-8"),
                       file_name="predictions.csv", mime="text/csv", on_click="ignore")
    d2.download_button("Download filtered view", ordered.to_csv(index=False).encode("utf-8"),
                       file_name="predictions_filtered.csv", mime="text/csv",
                       on_click="ignore", disabled=view.empty)
    st.caption("Downloads contain every uploaded column (including customer IDs) plus the churn "
               f"probability, prediction and the risk segment for the current band boundaries.")
    _render_explanations(ctx, scored, view)


def _customer_label(results: pd.DataFrame, index) -> str:
    if ID_COL in results.columns and pd.notna(results.at[index, ID_COL]):
        return f"{ID_COL} {results.at[index, ID_COL]} (row {index + 1})"
    return f"Row {index + 1}"


def _render_explanations(ctx: DashboardContext, scored: pd.DataFrame, view: pd.DataFrame) -> None:
    st.subheader("Explain individual customers")
    if ctx.explainability is None or ctx.background is None:
        st.warning(f"Customer explanations are unavailable: {ctx.explainability_error} "
                   f"Generate them with `{ctx.xai_command}`.")
        return
    if view.empty:
        st.info("No customers in the current risk list to explain.")
        return
    options = [int(i) for i in view.index[:200]]
    chosen = st.multiselect(
        f"Customers to explain (up to {xai.MAX_EXPLAINED_CUSTOMERS}, from the risk list above)", options,
        default=options[:1], max_selections=xai.MAX_EXPLAINED_CUSTOMERS,
        format_func=lambda i: _customer_label(scored, i))
    if not chosen:
        st.caption("Select a customer to see which features moved their churn score.")
        return
    st.caption(
        "Each bar is a feature's Shapley contribution, in percentage points of the model's "
        "**uncalibrated churn score**. Contributions start from the reference value, which is the average "
        f"score of {ctx.explainability['background']['n_rows']} training customers, and add up to this "
        "customer's score. They describe how the model arrived at its score. They are not causes of "
        "churn, and changing a feature would not necessarily change the outcome.")

    try:
        X, _ = prepare_features(scored.loc[chosen], ctx.schema)
    except DataValidationError as exc:
        st.error(f"These customers cannot be explained: {exc}")
        return
    for index, tab in zip(chosen, st.tabs([_customer_label(scored, i) for i in chosen])):
        with tab:
            try:
                with st.spinner("Computing explanation..."):
                    exp = _explain_cached(ctx.model, ctx.model_sha, X.loc[[index]], ctx.background)
            except Exception as exc:
                st.error(f"The explanation could not be computed: {exc}")
                continue
            c = st.columns(4)
            c[0].metric("Model churn score", f"{exp.prediction:.1%}")
            c[1].metric("Reference value", f"{exp.base_value:.1%}", help="Average score of the background sample.")
            if CALIBRATED_COL in scored.columns:
                value = scored.at[index, CALIBRATED_COL]
                shown = ">99.9%" if value > 99.9 else "<0.1%" if value < 0.1 else f"{value:.1f}%"
                c[2].metric("Calibrated probability", shown,
                            help="Validation-calibrated estimate; not explained by the contributions below.")
            if PREDICTION_COL in scored.columns:
                c[3].metric("Prediction", "Churn" if scored.at[index, PREDICTION_COL] == 1 else "No churn")
            if not exp.is_consistent(1e-6):
                st.warning(f"The explanation does not reconcile with the model's prediction "
                           f"(difference {max(exp.additivity_error, abs(exp.prediction - exp.model_prediction)):.2e}); "
                           "treat it with caution.")
            frame = exp.to_frame()
            st.altair_chart(charts.contribution_waterfall(frame, exp.base_value, exp.prediction))
            up = frame[frame["contribution"] > 0].head(3)["feature"].tolist()
            down = frame[frame["contribution"] < 0].head(3)["feature"].tolist()
            st.markdown(f"- Raised the score most: {', '.join(up) or 'none'}\n"
                        f"- Lowered the score most: {', '.join(down) or 'none'}")
            missing = [f for f, v in zip(exp.features, exp.feature_values) if v is None]
            if missing:
                st.caption(f"Missing values ({', '.join(missing)}) were imputed by the model pipeline; their "
                           "contribution reflects the imputed value.")
            with st.expander("Contribution table"):
                st.dataframe(frame.assign(value=frame["value"].map(lambda v: "missing" if v is None else str(v)),
                                          contribution=frame["contribution"] * 100,
                                          std_error=frame["std_error"] * 100)
                             .rename(columns={"contribution": "Contribution (pp)",
                                              "std_error": "Sampling std error (pp)"}),
                             hide_index=True, column_config={
                                 c: st.column_config.NumberColumn(format="%+.2f")
                                 for c in ["Contribution (pp)"]})
                st.caption(f"Estimated from {exp.n_samples:,} sampled feature orderings. Standard errors "
                           "show the sampling uncertainty for this fixed background sample; they do not "
                           "include uncertainty from which background customers were chosen.")


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

    test_tab, cv_tab, val_tab, imp_tab, cal_tab, plot_tab = st.tabs(
        ["Held-out test set", "Cross-validation & comparison", "Validation set", "Feature importance",
         "Calibration", "Saved plots"])
    for warning in ctx.xai_warnings:
        st.warning(warning)
    with imp_tab:
        _importance_tab(ctx)
    with cal_tab:
        _calibration_tab(ctx)

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


def _importance_tab(ctx: DashboardContext) -> None:
    if ctx.explainability is None:
        st.warning(f"Permutation importance is unavailable: {ctx.explainability_error} "
                   f"Generate it with `{ctx.xai_command}`.")
        return
    pi = ctx.explainability["permutation_importance"]
    table = pd.DataFrame(pi["features"])
    st.caption(
        f"Permutation importance on the **validation partition** ({pi['n_rows']:,} customers, "
        f"{pi['n_positive']} churners), which was not used to fit the model. Each feature was shuffled "
        f"{pi['n_repeats']} times and the drop in PR-AUC from the unshuffled "
        f"{pi['baseline_score']:.3f} was recorded. Error bars show ±1 standard deviation across repeats.")
    with st.container(border=True):
        st.altair_chart(charts.importance_chart(table))
    st.caption("A large drop means the model relies on that feature to rank churners, not that the "
               "feature causes churn. Correlated features can share or hide importance, and values near "
               "zero or below it mean shuffling made no measurable difference.")

    comparison = ctx.explainability.get("impurity_comparison")
    with st.expander("Comparison with impurity-based Random Forest importance"):
        if not comparison:
            st.caption("Impurity importance is unavailable for this model or its features did not match.")
        else:
            st.dataframe(pd.DataFrame(comparison["features"]).rename(columns={
                "feature": "Feature", "permutation": "Permutation (PR-AUC drop)",
                "permutation_rank": "Permutation rank", "impurity": "Impurity importance",
                "impurity_rank": "Impurity rank"}), hide_index=True, column_config={
                c: st.column_config.NumberColumn(format="%.4f")
                for c in ["Permutation (PR-AUC drop)", "Impurity importance"]})
            rho = comparison.get("spearman_rank_correlation")
            st.caption(
                (f"Spearman rank correlation: {rho:.2f}. " if rho is not None else "")
                + "Impurity importance is computed on training data, with one-hot columns summed back to "
                  "their original feature. It tends to favour features with many distinct values, so "
                  "the two rankings can differ.")


def _calibration_tab(ctx: DashboardContext) -> None:
    if ctx.calibration is None:
        st.warning(f"Calibration results are unavailable: {ctx.calibration_error} "
                   f"Generate them with `{ctx.xai_command}`.")
        return
    meta = ctx.calibration
    ev, fit, thr = meta["evaluation"], meta["fit"], meta["threshold"]
    adopted = meta["method"] != "none"
    st.caption(
        f"Calibration remaps the model's churn score so that, for example, customers scored around 30% "
        f"churn about 30% of the time. Candidate methods were compared by 5-fold cross-validation on the "
        f"**validation partition** ({fit['n_rows']:,} customers) and the winner was refitted on all of it. "
        f"The **test partition** ({ev['uncalibrated']['n_samples']:,} customers) was used once, for the "
        "comparison below.")
    if adopted:
        st.success(f"Selected method: **{meta['method']}** (logistic regression on the score's log-odds)."
                   if meta["method"] == "sigmoid" else f"Selected method: **{meta['method']}**.")
    else:
        reason = (f" The selected calibrator failed its checks ({', '.join(meta['rejected_reasons'])})."
                  if meta.get("rejected_reasons") else
                  " Cross-validation did not show that calibration lowers the Brier score.")
        st.info("No calibrator is applied; the dashboard uses the uncalibrated scores." + reason)

    names = {"brier": "Brier score (lower is better)", "log_loss": "Log loss (lower is better)",
             "ece": "Expected calibration error (lower is better)", "roc_auc": "ROC-AUC (ranking)",
             "pr_auc": "PR-AUC (ranking)"}
    columns = {"Uncalibrated": ev["uncalibrated"]}
    if "calibrated" in ev:
        columns["Calibrated"] = ev["calibrated"]
    table = pd.DataFrame({k: {names[m]: v[m] for m in names} for k, v in columns.items()})
    if "Calibrated" in table:
        table["Change"] = table["Calibrated"] - table["Uncalibrated"]
    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Probability quality on the test partition**")
        st.dataframe(table, column_config={c: st.column_config.NumberColumn(format="%.4f")
                                           for c in table.columns})
        st.caption("Brier score and log loss measure how close the probabilities are to the outcomes. "
                   "ROC-AUC and PR-AUC measure ranking only. A monotonic calibrator keeps the customer "
                   "order, so these stay essentially unchanged; calibration is not expected to improve them.")
        if adopted:
            st.caption(
                f"Decision threshold unchanged at {thr['value']:.2f} on the **uncalibrated** score. That "
                f"corresponds to a calibrated probability of {thr['calibrated_value_at_threshold']:.1%}: "
                "thresholding the calibrated probability there gives the same churn / no-churn labels as "
                f"the raw-score rule for {thr['test_label_agreement_at_mapped_threshold']:.1%} of test "
                "customers. This compares two decision rules with each other; it is not accuracy. Test-set "
                "accuracy, precision and recall at the threshold are on the held-out test tab.")
    with right.container(border=True):
        st.markdown("**Reliability diagram (test partition)**")
        tables = {"Uncalibrated": pd.DataFrame(ev["reliability"]["uncalibrated"])}
        if "calibrated" in ev["reliability"]:
            tables["Calibrated"] = pd.DataFrame(ev["reliability"]["calibrated"])
        st.altair_chart(charts.reliability_chart(tables))
        st.caption("Points on the dashed diagonal are perfectly calibrated. Points below it mean the "
                   "probabilities are too high; points above mean they are too low. Hover over a point "
                   "to see how many customers are in its bin; bins with few customers are noisy.")

    with st.expander("Method selection (validation cross-validation)"):
        st.dataframe(pd.DataFrame(meta["selection"]["results"]), hide_index=True,
                     column_config={c: st.column_config.NumberColumn(format="%.4f")
                                    for c in ["brier", "log_loss", "ece"]})
        st.caption(f"Criterion: {meta['selection']['criterion']}. 'none' keeps the uncalibrated scores. "
                   "The decision threshold was also tuned on the validation partition, so the "
                   "test-set comparison above is the unbiased check.")


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


# --------------------------------------------------------------------------- #
# F. Retention intelligence
# --------------------------------------------------------------------------- #
RETENTION_NOTE = (
    "Suggested actions are heuristic rules based on observed customer features. They have not been "
    "tested as treatments, and nothing on this page estimates customers retained or revenue saved. A "
    "high rank means a high model score or more observed triggers; it does not mean the customer will churn.")
PLAN_COLUMNS = [rt.PRIORITY_RANK_COL, rt.ROW_ID_COL, ID_COL, PROBABILITY_COL, CALIBRATED_COL,
                PREDICTION_COL, rt.BAND_COL, *rt.PROFILE_FEATURES, rt.ACTION_COL, rt.REASON_COL,
                rt.OTHER_ACTIONS_COL]


def _csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def _plan_columns(frame: pd.DataFrame, extra=()) -> list[str]:
    return [c for c in PLAN_COLUMNS + list(extra) if c in frame.columns]


def _plan_column_config() -> dict:
    return {
        PROBABILITY_COL: st.column_config.ProgressColumn(
            "Model churn score", format="%.1f%%", min_value=0, max_value=100,
            help="Uncalibrated model score. Planning bands and rankings use this score."),
        CALIBRATED_COL: st.column_config.NumberColumn(
            "Calibrated probability", format="%.1f%%",
            help="Validation-calibrated estimate; shown for interpretation only."),
        PREDICTION_COL: st.column_config.NumberColumn(
            "Official prediction", help="1 = predicted to churn at the official decision threshold"),
        rt.BAND_COL: st.column_config.TextColumn(
            rt.BAND_COL, help="Planning only; independent of the official decision threshold."),
    }


def retention_intelligence(ctx: DashboardContext) -> None:
    _page_header("Retention Intelligence",
                 "Plan retention outreach for the current scored batch with transparent, configurable "
                 "business rules. The model, its decision threshold and saved artifacts are not changed.")
    st.markdown(
        "1. **Planning risk bands** group customers by model churn score.\n"
        "2. **Rules** suggest a next action from observed features (complaints, tenure, satisfaction, "
        "days since last order).\n"
        "3. **Priority ranking** orders customers by a documented policy.\n"
        "4. **Budget planner** selects customers in rank order until the budget or contact limit is reached.\n"
        "5. **What-if comparison** sets several planning scenarios side by side.")
    st.caption(RETENTION_NOTE)

    batch = _batch()
    if batch is None or batch["results"].empty:
        st.info("No file has been scored in this session. Score a file on **Churn Prediction** first; "
                "this page reuses those scores and never re-runs the model.")
        return
    results = batch["results"]
    st.caption(f"Batch: `{batch['filename']}` ({len(results):,} customers). {rt.identifier_note(results)}")
    if CALIBRATED_COL not in results.columns:
        st.info("Calibrated probabilities are unavailable for this batch, so only the model churn score is "
                "shown. Rankings never use calibrated values, so they are unaffected.")
    has_predictions = PREDICTION_COL in results.columns
    if not has_predictions:
        st.warning("Official predictions are unavailable because no verified decision threshold was "
                   "loaded. Planning still works on the model score; 'predicted churners only' is disabled.")
    missing_features = [c for c in rt.PROFILE_FEATURES if c not in results.columns]
    if missing_features:
        st.info(f"Not in this file: {', '.join(missing_features)}. Rules that need them are skipped.")

    with st.expander("Planning settings: risk bands and rule thresholds"):
        low_default, high_default = db.default_risk_bounds(ctx.threshold)
        low, high = st.slider("Planning risk bands (model churn score, %)", 1, 99,
                              (low_default, high_default), key="retention_bounds",
                              help="Low < first value ≤ Medium < second value ≤ High. Planning only.")
        c = st.columns(3)
        cfg = rt.RuleConfig(
            c[0].number_input("Short tenure: at most (months)", 0.0, 120.0, 2.0, 1.0),
            c[1].number_input("Low satisfaction: score at most", 0.0, 5.0, 2.0, 1.0),
            c[2].number_input("Inactivity: days since last order at least", 0.0, 365.0, 10.0, 1.0))
    if low >= high:
        st.warning("The two risk-band boundaries must differ. Move the slider handles apart.")
        return
    threshold_note = (f"Official predictions use the decision threshold of {ctx.threshold:.0%} and are "
                      "not affected." if ctx.threshold is not None else "No official threshold is loaded.")
    st.caption(f"Planning bands: Low < {low}% ≤ Medium < {high}% ≤ High. {threshold_note}")
    plan = rt.build_action_plan(results, low, high, cfg)

    tabs = st.tabs(["Prioritisation", "Suggested actions", "Budget planner", "What-if scenarios"])
    with tabs[0]:
        _retention_priorities(plan, has_predictions)
    with tabs[1]:
        _retention_actions(plan, cfg)
    with tabs[2]:
        selection, budget, cost = _retention_budget(plan, has_predictions)
    with tabs[3]:
        _retention_scenarios(plan, budget, cost)

    st.subheader("Download the retention action plan")
    export_source = selection if selection is not None else plan
    d1, d2 = st.columns(2)
    d1.download_button("Download full action plan (original file order)",
                       _csv(rt.export_action_plan(export_source)), file_name="retention_action_plan.csv",
                       mime="text/csv", on_click="ignore")
    d2.download_button("Download selected customers (selection order)",
                       _csv(rt.export_action_plan(selection, selected_only=True)) if selection is not None
                       else b"", file_name="retention_selected_customers.csv", mime="text/csv",
                       on_click="ignore", disabled=selection is None or not selection[rt.SELECTED_COL].any())
    st.caption(f"Both files share one fixed column layout. `{rt.ROW_ID_COL}` is the row's position in the "
               f"uploaded file and `{rt.ID_TYPE_COL}` states whether a unique `{ID_COL}` is available. "
               "Eligibility, selection and cost columns reflect the budget planner settings; columns that do "
               "not apply (for example, no calibrator) are left empty.")


def _retention_priorities(plan: pd.DataFrame, has_predictions: bool) -> None:
    st.caption(
        "**Priority ranking policy:** planning risk band (High first), then the number of observed rule "
        "triggers, then model churn score, then original file order. **Score ranking:** model churn score "
        "only, ties in file order. Both are planning orders, not churn guarantees.")
    f = st.columns([2, 3, 1.5, 1.5])
    bands = f[0].multiselect("Planning risk bands", list(rt.RISK_BANDS), default=list(rt.RISK_BANDS))
    present = [a for a in rt.ACTIONS if a in set(plan[rt.ACTION_COL])]
    actions = f[1].multiselect("Suggested actions", present, default=present)
    sort = f[2].selectbox("Order", ["priority", "score", "original"],
                          format_func={"priority": "Priority rank", "score": "Model score rank",
                                       "original": "Original file order"}.get)
    predicted_only = f[3].checkbox("Official predicted churners only", disabled=not has_predictions,
                                   key="retention_predicted_only")
    view = rt.filter_plan(plan, bands, actions, predicted_only, sort)
    st.caption(f"Showing {len(view):,} of {len(plan):,} customers.")
    if view.empty:
        st.info("No customers match these filters.")
        return
    st.dataframe(view[_plan_columns(view, [rt.SCORE_RANK_COL])], hide_index=True,
                 column_config=_plan_column_config())
    st.download_button("Download this view", _csv(view[_plan_columns(view, [rt.SCORE_RANK_COL])]),
                       file_name="retention_priorities_filtered.csv", mime="text/csv", on_click="ignore")


def _retention_actions(plan: pd.DataFrame, cfg: rt.RuleConfig) -> None:
    left, right = st.columns([1, 1])
    with left.container(border=True):
        st.altair_chart(charts.action_mix_chart(rt.action_counts(plan), "Suggested next action (all customers)"))
    with right:
        st.markdown("**Rules, applied in this order**")
        st.dataframe(rt.rules_table(cfg), hide_index=True)
        st.caption("The first rule that applies sets the suggested action; other applicable actions are "
                   "listed alongside it. Missing values never trigger a rule. No rule offers discounts: "
                   "incentives need evidence of effect that this dataset does not provide.")
    st.caption(f"Every suggestion is labelled “{rt.HEURISTIC_BASIS}”. {CAUSATION_NOTE}")


def _retention_budget(plan: pd.DataFrame, has_predictions: bool):
    st.caption("Enter your own campaign assumptions. The defaults are placeholders, not estimates; amounts "
               "are in your own currency units.")
    c = st.columns(3)
    budget = c[0].number_input("Total campaign budget", min_value=0.0, value=1000.0, step=100.0)
    limit = c[1].number_input("Maximum customers to contact (0 = no limit)", min_value=0, value=0, step=10)
    strategy = c[2].selectbox("Selection strategy", list(rt.STRATEGIES), format_func=rt.STRATEGIES.get)
    c = st.columns(3)
    min_score = c[0].slider("Minimum model churn score (%)", 0, 100, 0, key="retention_min_score")
    min_band = c[1].selectbox("Lowest eligible planning band", list(rt.RISK_BANDS), index=1)
    predicted_only = c[2].checkbox("Only official predicted churners", disabled=not has_predictions)
    per_action = st.toggle("Set a different cost for each suggested action")
    if per_action:
        present = [a for a in rt.ACTIONS if a in set(plan[rt.ACTION_COL])]
        cols = st.columns(min(len(present), 3))
        cost = {a: cols[i % len(cols)].number_input(f"Cost: {a}", min_value=0.0, value=10.0, step=1.0)
                for i, a in enumerate(present)}
    else:
        cost = st.number_input("Cost per contacted customer", min_value=0.0, value=10.0, step=1.0)

    scenario = rt.Scenario("Planner", budget, cost, int(limit) or None, min_score, min_band, strategy,
                           predicted_only and has_predictions)
    try:
        selection, summary = rt.plan_scenario(plan, scenario)
    except ValueError as exc:
        st.warning(f"These settings cannot be planned: {exc}")
        return None, budget, 10.0
    costs = list(cost.values()) if isinstance(cost, dict) else [cost]
    if budget == 0 and any(costs):
        st.info("The budget is zero, so no customers with a non-zero cost can be selected.")
    if not any(costs):
        st.info("Every cost is zero, so the budget does not limit selection; only the contact limit does.")

    m = st.columns(4)
    m[0].metric("Eligible customers", f"{summary['Eligible']:,}")
    m[1].metric("Selected customers", f"{summary['Selected']:,}")
    m[2].metric("Estimated campaign cost", f"{summary['Estimated cost']:,.2f}")
    m[3].metric("Unspent budget", f"{summary['Unspent budget']:,.2f}")
    not_selected = summary["Eligible"] - summary["Selected"]
    if not_selected:
        st.caption(f"{not_selected:,} eligible customer(s) were not selected because the budget or contact "
                   "limit was reached. Customers are taken strictly in rank order: selection stops at the "
                   "first customer who does not fit, so nobody is skipped in favour of a lower rank.")
    b = st.columns(3)
    for slot, band in zip(b, rt.RISK_BANDS):
        slot.metric(f"Selected: {band} band", f"{summary[f'{band} band']:,}")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Observed characteristics**")
        st.dataframe(rt.selection_profile(selection), column_config={
            c: st.column_config.NumberColumn(format="%.2f") for c in
            ("Selected", "Eligible, not selected", "All scored")})
        st.caption("Descriptive only: medians and shares of observed values; the model score and calibrated "
                   "probability are model estimates.")
    chosen = selection[selection[rt.SELECTED_COL]]
    with right.container(border=True):
        if chosen.empty:
            st.caption("No customers are selected with these settings.")
        else:
            st.altair_chart(charts.action_mix_chart(rt.action_counts(chosen), "Suggested action (selected)"))
    if not chosen.empty:
        with st.expander(f"Selected customers ({len(chosen):,})"):
            ordered = chosen.sort_values(rt.ORDER_COL, kind="mergesort")
            st.dataframe(ordered[[rt.ORDER_COL, rt.COST_COL] + _plan_columns(ordered)], hide_index=True,
                         column_config=_plan_column_config())
    return selection, budget, float(pd.Series(costs).mean())


def _retention_scenarios(plan: pd.DataFrame, budget: float, cost: float) -> None:
    st.caption(
        "Each row is one set of campaign assumptions. The presets share the planner's budget and cost "
        "(the average cost when costs differ by action) and differ only in the lowest eligible band. Edit "
        "any cell or add rows; scenarios use one cost per customer. Changing scenarios only re-runs these "
        "planning rules; the model and threshold are untouched.")
    table = rt.scenario_table(rt.preset_scenarios(budget, cost))
    edited = st.data_editor(table, num_rows="dynamic", hide_index=True, column_config={
        "Lowest band": st.column_config.SelectboxColumn(options=list(rt.RISK_BANDS), required=True),
        "Strategy": st.column_config.SelectboxColumn(options=list(rt.STRATEGIES), required=True),
        "Budget": st.column_config.NumberColumn(min_value=0.0),
        "Cost per customer": st.column_config.NumberColumn(min_value=0.0),
        "Max customers (0 = no limit)": st.column_config.NumberColumn(min_value=0, step=1),
        "Min model score (%)": st.column_config.NumberColumn(min_value=0.0, max_value=100.0),
    })
    scenarios, errors = rt.scenarios_from_table(edited)
    for error in errors:
        st.warning(error)
    if not scenarios:
        st.info("Add at least one valid scenario to compare.")
        return
    comparison = rt.compare_scenarios(plan, scenarios)
    st.dataframe(comparison, hide_index=True, column_config={
        c: st.column_config.NumberColumn(format="%.2f") for c in comparison.columns
        if c.startswith(("Mean", "Median", "Complaint", "Estimated", "Unspent"))})
    left, right = st.columns(2)
    with left.container(border=True):
        st.altair_chart(charts.scenario_band_chart(comparison))
    with right.container(border=True):
        st.altair_chart(charts.scenario_cost_chart(comparison))
    st.caption(
        "**Trade-off:** wider eligibility and larger budgets contact more customers at a higher estimated "
        "cost, and usually add customers with lower model scores and fewer observed triggers. Compare the "
        "mean score, complaint share and band mix of each scenario. A scenario that contacts more customers "
        "is not inherently more effective: this dataset contains no evidence of how customers respond to "
        "outreach, so no retention or revenue outcomes are estimated.")
