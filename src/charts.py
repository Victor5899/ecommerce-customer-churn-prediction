"""Altair chart builders. Inputs are small, pre-aggregated frames from ``dashboard``."""
from __future__ import annotations

from typing import Optional

import altair as alt
import pandas as pd

PRIMARY = "#1F4E79"
NEUTRAL = "#8A99A8"
STATUS_COLORS = {"Retained": "#2E86AB", "Churned": "#D1495B"}
RISK_COLORS = {"Low": "#2E8B57", "Medium": "#E0A030", "High": "#C0392B"}
HEIGHT = 280


def _order(series: pd.Series) -> list[str]:
    if isinstance(series.dtype, pd.CategoricalDtype):
        present = set(series.astype(str))
        return [str(c) for c in series.cat.categories if str(c) in present]
    return list(dict.fromkeys(series.astype(str)))


def _scale(mapping: dict) -> alt.Scale:
    return alt.Scale(domain=list(mapping), range=list(mapping.values()))


def churn_donut(n_retained: int, n_churned: int) -> alt.Chart:
    data = pd.DataFrame({"Status": ["Retained", "Churned"], "Customers": [n_retained, n_churned]})
    data["Share"] = data["Customers"] / max(data["Customers"].sum(), 1)
    return alt.Chart(data).mark_arc(innerRadius=70).encode(
        theta="Customers:Q",
        color=alt.Color("Status:N", scale=_scale(STATUS_COLORS), legend=alt.Legend(orient="bottom")),
        tooltip=["Status", alt.Tooltip("Customers:Q", format=","), alt.Tooltip("Share:Q", format=".1%")],
    ).properties(height=HEIGHT)


def churn_rate_bars(table: pd.DataFrame, title: str, overall_rate: Optional[float] = None) -> alt.Chart:
    order = _order(table["Segment"])
    data = table.assign(Segment=table["Segment"].astype(str))
    bars = alt.Chart(data).mark_bar(color=STATUS_COLORS["Churned"], cornerRadiusEnd=3).encode(
        x=alt.X("Segment:N", sort=order, title=None, axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("Churn Rate:Q", axis=alt.Axis(format="%"), title="Actual churn rate"),
        tooltip=["Segment", alt.Tooltip("Churn Rate:Q", format=".1%"),
                 alt.Tooltip("Customers:Q", format=","), alt.Tooltip("Churned:Q", format=",")],
    )
    layers = [bars]
    if overall_rate is not None:
        rule = alt.Chart(pd.DataFrame({"Overall": [overall_rate]})).mark_rule(
            color=PRIMARY, strokeDash=[5, 4]).encode(
            y="Overall:Q", tooltip=[alt.Tooltip("Overall:Q", format=".1%", title="Overall churn rate")])
        layers.append(rule)
    return alt.layer(*layers).properties(title=title, height=HEIGHT)


def distribution_bars(table: pd.DataFrame, title: str) -> alt.Chart:
    order = _order(table["Segment"])
    data = table.assign(Segment=table["Segment"].astype(str))
    encoding = dict(
        x=alt.X("Segment:N", sort=order, title=None, axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("Customers:Q", title="Customers"),
        tooltip=["Segment", alt.Tooltip("Customers:Q", format=",")],
    )
    if "Status" in data.columns:
        encoding["color"] = alt.Color("Status:N", scale=_scale(STATUS_COLORS),
                                      legend=alt.Legend(orient="bottom", title="Actual outcome"))
        encoding["tooltip"] = ["Segment", "Status", alt.Tooltip("Customers:Q", format=",")]
        return alt.Chart(data).mark_bar().encode(**encoding).properties(title=title, height=HEIGHT)
    return alt.Chart(data).mark_bar(color=NEUTRAL).encode(**encoding).properties(title=title, height=HEIGHT)


def risk_histogram(probability_pct: pd.Series, segments: pd.Series,
                   threshold_pct: Optional[float]) -> alt.Chart:
    left = (probability_pct.astype(float).clip(0, 99.999) // 5) * 5
    data = (pd.DataFrame({"Bin": left, "Segment": segments.astype(str)})
            .groupby(["Bin", "Segment"]).size().reset_index(name="Customers"))
    data["Range"] = data["Bin"].map(lambda b: f"{b:.0f}–{b + 5:.0f}%")
    bars = alt.Chart(data).mark_bar(binSpacing=1).encode(
        x=alt.X("Bin:Q", bin=alt.Bin(step=5, extent=[0, 100]), title="Churn probability (%)"),
        y=alt.Y("Customers:Q", stack=True),
        color=alt.Color("Segment:N", scale=_scale(RISK_COLORS), legend=alt.Legend(orient="bottom")),
        tooltip=["Range", "Segment", alt.Tooltip("Customers:Q", format=",")],
    )
    layers = [bars]
    if threshold_pct is not None:
        layers.append(alt.Chart(pd.DataFrame({"Threshold": [threshold_pct]})).mark_rule(
            color=PRIMARY, strokeDash=[5, 4], size=2).encode(
            x="Threshold:Q", tooltip=[alt.Tooltip("Threshold:Q", title="Decision threshold (%)")]))
    return alt.layer(*layers).properties(title="Churn probability distribution", height=HEIGHT)


def confusion_matrix_chart(cm: dict) -> alt.Chart:
    data = pd.DataFrame([
        {"Actual": "Retained (0)", "Predicted": "Retained (0)", "Count": cm["tn"], "Cell": "True negative"},
        {"Actual": "Retained (0)", "Predicted": "Churned (1)", "Count": cm["fp"], "Cell": "False positive"},
        {"Actual": "Churned (1)", "Predicted": "Retained (0)", "Count": cm["fn"], "Cell": "False negative"},
        {"Actual": "Churned (1)", "Predicted": "Churned (1)", "Count": cm["tp"], "Cell": "True positive"},
    ])
    order = ["Retained (0)", "Churned (1)"]
    base = alt.Chart(data).encode(
        x=alt.X("Predicted:N", sort=order, axis=alt.Axis(labelAngle=0, orient="top")),
        y=alt.Y("Actual:N", sort=order),
    )
    heat = base.mark_rect().encode(
        color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues"), legend=None),
        tooltip=["Cell", "Actual", "Predicted", alt.Tooltip("Count:Q", format=",")],
    )
    max_count = max(data["Count"].max(), 1)
    text = base.mark_text(fontSize=18, fontWeight="bold").encode(
        text=alt.Text("Count:Q", format=","),
        color=alt.condition(alt.datum.Count > max_count / 2, alt.value("white"), alt.value("#1B2631")),
    )
    return (heat + text).properties(height=HEIGHT)


def cv_comparison_chart(candidates: pd.DataFrame) -> alt.Chart:
    data = candidates.assign(
        low=candidates["cv_pr_auc_mean"] - candidates["cv_pr_auc_std"],
        high=candidates["cv_pr_auc_mean"] + candidates["cv_pr_auc_std"],
        Status=candidates["selected"].map({True: "Selected", False: "Candidate"}),
    )
    order = list(data.sort_values("cv_pr_auc_mean", ascending=False)["model_name"])
    y = alt.Y("model_name:N", sort=order, title=None)
    bars = alt.Chart(data).mark_bar().encode(
        y=y, x=alt.X("cv_pr_auc_mean:Q", title="Mean CV PR-AUC (± 1 std)", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Status:N", scale=_scale({"Selected": PRIMARY, "Candidate": NEUTRAL}),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=[alt.Tooltip("model_name:N", title="Model"),
                 alt.Tooltip("cv_pr_auc_mean:Q", format=".3f", title="Mean CV PR-AUC"),
                 alt.Tooltip("cv_pr_auc_std:Q", format=".3f", title="Std")],
    )
    errors = alt.Chart(data).mark_errorbar(color="#1B2631").encode(
        y=y, x=alt.X("low:Q", title=""), x2="high:Q")
    return (bars + errors).properties(height=40 * len(data) + 40)


def missing_values_chart(table: pd.DataFrame) -> alt.Chart:
    return alt.Chart(table).mark_bar(color=PRIMARY, cornerRadiusEnd=3).encode(
        y=alt.Y("Column:N", sort="-x", title=None),
        x=alt.X("Missing %:Q", title="Missing values (% of rows)"),
        tooltip=["Column", alt.Tooltip("Missing:Q", format=","), alt.Tooltip("Missing %:Q", format=".2f")],
    ).properties(height=max(26 * len(table), 80))
