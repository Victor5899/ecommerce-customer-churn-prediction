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
        x=alt.X("Bin:Q", bin=alt.Bin(step=5, extent=[0, 100]), title="Model churn score, uncalibrated (%)"),
        y=alt.Y("Customers:Q", stack=True),
        color=alt.Color("Segment:N", scale=_scale(RISK_COLORS), legend=alt.Legend(orient="bottom")),
        tooltip=["Range", "Segment", alt.Tooltip("Customers:Q", format=",")],
    )
    layers = [bars]
    if threshold_pct is not None:
        layers.append(alt.Chart(pd.DataFrame({"Threshold": [threshold_pct]})).mark_rule(
            color=PRIMARY, strokeDash=[5, 4], size=2).encode(
            x="Threshold:Q", tooltip=[alt.Tooltip("Threshold:Q", title="Decision threshold (%)")]))
    return alt.layer(*layers).properties(title="Model churn score distribution", height=HEIGHT)


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


def importance_chart(table: pd.DataFrame, top_n: int = 15) -> alt.Chart:
    """Horizontal bars of mean permutation importance with ±1 std error bars."""
    data = table.head(top_n).assign(
        low=lambda t: t["importance_mean"] - t["importance_std"],
        high=lambda t: t["importance_mean"] + t["importance_std"],
    )
    y = alt.Y("feature:N", sort=list(data["feature"]), title=None)
    bars = alt.Chart(data).mark_bar(color=PRIMARY, cornerRadiusEnd=3).encode(
        y=y, x=alt.X("importance_mean:Q", title="Mean drop in PR-AUC when shuffled (± 1 std)"),
        tooltip=[alt.Tooltip("feature:N", title="Feature"),
                 alt.Tooltip("importance_mean:Q", format=".4f", title="Mean drop"),
                 alt.Tooltip("importance_std:Q", format=".4f", title="Std across repeats"),
                 alt.Tooltip("importance_min:Q", format=".4f", title="Min"),
                 alt.Tooltip("importance_max:Q", format=".4f", title="Max")],
    )
    errors = alt.Chart(data).mark_errorbar(color="#1B2631").encode(y=y, x=alt.X("low:Q", title=""), x2="high:Q")
    return (bars + errors).properties(height=24 * len(data) + 40)


def reliability_chart(tables: dict) -> alt.Chart:
    """Observed churn rate vs mean predicted probability per bin, one line per probability source."""
    frames = [t.assign(Source=name) for name, t in tables.items()]
    data = pd.concat(frames, ignore_index=True)
    data = data[data["n"] > 0]
    colors = {"Uncalibrated": NEUTRAL, "Calibrated": PRIMARY}
    diagonal = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
        color="#B0B7C0", strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
    lines = alt.Chart(data).mark_line(point=True).encode(
        x=alt.X("mean_predicted:Q", title="Mean predicted probability", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("observed_rate:Q", title="Observed churn rate", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Source:N", scale=_scale({k: colors.get(k, PRIMARY) for k in tables}),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Source", alt.Tooltip("lower:Q", format=".1f", title="Bin from"),
                 alt.Tooltip("upper:Q", format=".1f", title="Bin to"), alt.Tooltip("n:Q", title="Customers"),
                 alt.Tooltip("mean_predicted:Q", format=".3f"), alt.Tooltip("observed_rate:Q", format=".3f")],
    )
    return (diagonal + lines).properties(height=320)


def contribution_waterfall(frame: pd.DataFrame, base_value: float, prediction: float,
                           top_n: int = 10) -> alt.Chart:
    """Waterfall from the reference value to the prediction, in percentage points."""
    head, tail = frame.head(top_n), frame.iloc[top_n:]
    rows = [{"label": f"{r.feature} = {'missing' if r.value is None else r.value}",
             "delta": r.contribution * 100, "se": r.std_error * 100} for r in head.itertuples()]
    if len(tail):
        rows.append({"label": f"{len(tail)} other features", "delta": tail["contribution"].sum() * 100,
                     "se": float("nan")})
    data = pd.DataFrame(rows)
    data["end"] = base_value * 100 + data["delta"].cumsum()
    data["start"] = data["end"] - data["delta"]
    data["Direction"] = data["delta"].map(lambda d: "Increases score" if d >= 0 else "Decreases score")
    order = list(data["label"])
    bars = alt.Chart(data).mark_bar().encode(
        y=alt.Y("label:N", sort=order, title=None),
        x=alt.X("start:Q", title="Model churn score (%)"), x2="end:Q",
        color=alt.Color("Direction:N", scale=_scale({"Increases score": STATUS_COLORS["Churned"],
                                                     "Decreases score": STATUS_COLORS["Retained"]}),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=[alt.Tooltip("label:N", title="Feature"),
                 alt.Tooltip("delta:Q", format="+.2f", title="Contribution (pp)"),
                 alt.Tooltip("se:Q", format=".2f", title="Sampling std error (pp)"),
                 alt.Tooltip("end:Q", format=".1f", title="Running score (%)")],
    )
    marks = pd.DataFrame({"x": [base_value * 100, prediction * 100],
                          "Marker": ["Reference value", "Prediction"]})
    rules = alt.Chart(marks).mark_rule(strokeDash=[4, 3], color="#1B2631").encode(
        x="x:Q", tooltip=["Marker", alt.Tooltip("x:Q", format=".1f", title="Score (%)")])
    return (bars + rules).properties(height=26 * len(data) + 50)


def missing_values_chart(table: pd.DataFrame) -> alt.Chart:
    return alt.Chart(table).mark_bar(color=PRIMARY, cornerRadiusEnd=3).encode(
        y=alt.Y("Column:N", sort="-x", title=None),
        x=alt.X("Missing %:Q", title="Missing values (% of rows)"),
        tooltip=["Column", alt.Tooltip("Missing:Q", format=","), alt.Tooltip("Missing %:Q", format=".2f")],
    ).properties(height=max(26 * len(table), 80))
