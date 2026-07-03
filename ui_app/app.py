from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from data_loader import DEFAULT_RESULTS_PATH, load_results, validate_entries
from metrics import (
    CATEGORIES,
    DIMENSIONS,
    SPECIAL_METRICS,
    SYSTEM_COLORS,
    SYSTEM_LABELS,
    SYSTEMS,
    available_categories,
    build_dashboard_data,
    display_label,
    validate_summary,
)


st.set_page_config(
    page_title="Local Multi-Domain Assistant Evaluation Dashboard",
    layout="wide",
)


def format_value(value, places: int = 2, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if places == 0:
        return f"{int(round(number))}{suffix}"
    return f"{number:.{places}f}{suffix}"


def metric_label(metric: str) -> str:
    return metric.replace("_", " ").title()


def value_for(metrics: dict, key: str):
    return metrics.get(key)


def special_value_for(metrics: dict, key: str):
    return metrics.get("special_metrics", {}).get(key)


def dimension_value_for(metrics: dict, key: str):
    return metrics.get("dimension_score_averages", {}).get(key)


def category_value_for(metrics: dict, key: str):
    return metrics.get("category_accuracy", {}).get(key)


def comparison_table(system_metrics: dict[str, dict]) -> pd.DataFrame:
    rows = [
        ("Total questions evaluated", "total_questions_evaluated", 0, value_for),
        ("Average accuracy", "average_accuracy", 2, value_for),
        ("Median accuracy", "median_accuracy", 2, value_for),
        ("Average latency (s)", "average_latency", 2, value_for),
        ("Hallucination count", "hallucination_count", 0, value_for),
        ("Leakage count", "leakage_count", 0, value_for),
        ("Contamination count", "contamination_count", 0, value_for),
        ("False rejection count", "false_rejection_count", 0, value_for),
    ]

    for metric in SPECIAL_METRICS:
        rows.append((metric_label(metric), metric, 3, special_value_for))

    for dimension in DIMENSIONS:
        rows.append((f"Dimension: {metric_label(dimension)}", dimension, 2, dimension_value_for))

    table_rows = []
    for label, key, places, getter in rows:
        row = {"Metric": label}
        for system in SYSTEMS:
            row[SYSTEM_LABELS[system]] = format_value(
                getter(system_metrics[system], key),
                places=places,
            )
        table_rows.append(row)
    return pd.DataFrame(table_rows)


def make_bar_chart(
    rows: list[dict],
    x_field: str,
    y_field: str,
    y_title: str,
    height: int = 280,
):
    if not rows:
        st.info("N/A")
        return

    encodings = {
        "x": alt.X(f"{x_field}:N", title=None),
        "y": alt.Y(f"{y_field}:Q", title=y_title),
        "color": alt.Color(
            "System:N",
            scale=alt.Scale(
                domain=[SYSTEM_LABELS[system] for system in SYSTEMS],
                range=[SYSTEM_COLORS[system] for system in SYSTEMS],
            ),
            legend=alt.Legend(title=None),
        ),
        "tooltip": [x_field, "System", alt.Tooltip(f"{y_field}:Q", format=".3f")],
    }
    if x_field != "System":
        encodings["xOffset"] = alt.XOffset("System:N")

    chart = (
        alt.Chart(pd.DataFrame(rows))
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(**encodings)
        .properties(height=height)
    )
    st.altair_chart(chart, width="stretch")


def system_bar_chart(system_metrics: dict[str, dict], key: str, title: str, places: int = 3):
    rows = []
    for system in SYSTEMS:
        value = system_metrics[system].get(key)
        if value is not None:
            rows.append({"System": SYSTEM_LABELS[system], title: value})
    make_bar_chart(rows, "System", title, title)


def special_bar_chart(system_metrics: dict[str, dict], key: str, title: str):
    rows = []
    for system in SYSTEMS:
        value = system_metrics[system].get("special_metrics", {}).get(key)
        if value is not None:
            rows.append({"System": SYSTEM_LABELS[system], title: value})
    make_bar_chart(rows, "System", title, title)


def reliability_chart(system_metrics: dict[str, dict]):
    rows = []
    for system in SYSTEMS:
        metrics = system_metrics[system]
        for key, label in (
            ("hallucination_count", "Hallucination"),
            ("leakage_count", "Leakage"),
            ("contamination_count", "Contamination"),
        ):
            rows.append(
                {
                    "System": SYSTEM_LABELS[system],
                    "Metric": label,
                    "Count": metrics.get(key),
                }
            )

    chart = (
        alt.Chart(pd.DataFrame(rows))
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("Metric:N", title=None),
            xOffset=alt.XOffset("System:N"),
            y=alt.Y("Count:Q", title="Count"),
            color=alt.Color(
                "System:N",
                scale=alt.Scale(
                    domain=[SYSTEM_LABELS[system] for system in SYSTEMS],
                    range=[SYSTEM_COLORS[system] for system in SYSTEMS],
                ),
                legend=alt.Legend(title=None),
            ),
            tooltip=["Metric", "System", "Count"],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, width="stretch")


def category_chart(system_metrics: dict[str, dict]):
    rows = []
    for system in SYSTEMS:
        for category in CATEGORIES:
            value = category_value_for(system_metrics[system], category)
            if value is not None:
                rows.append(
                    {
                        "System": SYSTEM_LABELS[system],
                        "Category": display_label(category),
                        "Accuracy": value,
                    }
                )
    make_bar_chart(rows, "Category", "Accuracy", "Average accuracy", height=340)


def dimension_chart(system_metrics: dict[str, dict]):
    rows = []
    for system in SYSTEMS:
        for dimension in DIMENSIONS:
            value = dimension_value_for(system_metrics[system], dimension)
            if value is not None:
                rows.append(
                    {
                        "System": SYSTEM_LABELS[system],
                        "Dimension": metric_label(dimension),
                        "Score": value,
                    }
                )
    make_bar_chart(rows, "Dimension", "Score", "Average score", height=340)


def render_summary_cards(summary_cards: dict):
    card_data = [
        (
            "Total questions",
            str(summary_cards.get("total_questions", 0)),
            "Current view",
        ),
        (
            "Best accuracy system",
            summary_cards["best_accuracy_system"]["label"],
            format_value(summary_cards["best_accuracy_system"]["value"], 2),
        ),
        (
            "Fastest system",
            summary_cards["fastest_system"]["label"],
            "N/A"
            if summary_cards["fastest_system"]["value"] is None
            else f"{format_value(summary_cards['fastest_system']['value'], 2)} s",
        ),
        (
            "Lowest contamination system",
            summary_cards["lowest_contamination_system"]["label"],
            format_value(summary_cards["lowest_contamination_system"]["value"], 0),
        ),
        (
            "Best cross-domain robustness system",
            summary_cards["best_cross_domain_robustness_system"]["label"],
            format_value(summary_cards["best_cross_domain_robustness_system"]["value"], 3),
        ),
    ]

    columns = st.columns(len(card_data))
    for column, (label, value, detail) in zip(columns, card_data):
        with column:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-detail">{detail}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_question_comparison(questions: list[dict]):
    with st.expander("Question-by-question comparison", expanded=False):
        if not questions:
            st.info("N/A")
            return

        for question in questions:
            st.markdown(
                f"**Q{question['index']}. {question['question']}**  \n"
                f"`{display_label(question.get('question_type'))}`"
            )
            columns = st.columns(3)
            for column, system in zip(columns, SYSTEMS):
                system_result = question["systems"][system]
                metrics = system_result.get("metrics", {})
                with column:
                    st.markdown(
                        f"<div class='system-heading system-{system.lower()}'>"
                        f"{SYSTEM_LABELS[system]}</div>",
                        unsafe_allow_html=True,
                    )
                    st.write(system_result.get("response", "N/A"))
                    st.caption(
                        " | ".join(
                            [
                                f"Accuracy: {format_value(metrics.get('accuracy'), 2)}",
                                f"Latency: {format_value(metrics.get('latency'), 2)} s",
                                f"Hallucination: {format_value(metrics.get('hallucination'), 0)}",
                                f"Leakage: {format_value(metrics.get('leakage'), 0)}",
                                f"Contamination: {format_value(metrics.get('contamination'), 0)}",
                                f"False rejection: {format_value(metrics.get('false_rejection'), 0)}",
                            ]
                        )
                    )
            st.divider()


st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
    }
    .metric-card {
        border: 1px solid #d9dee8;
        border-radius: 8px;
        padding: 0.9rem 1rem;
        background: #ffffff;
        min-height: 112px;
    }
    .metric-label {
        color: #4b5563;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .metric-value {
        color: #111827;
        font-size: 1.25rem;
        font-weight: 700;
        line-height: 1.25;
        margin-top: 0.35rem;
    }
    .metric-detail {
        color: #6b7280;
        font-size: 0.84rem;
        margin-top: 0.4rem;
    }
    .system-heading {
        border-radius: 6px;
        color: #ffffff;
        font-weight: 700;
        margin-bottom: 0.45rem;
        padding: 0.35rem 0.55rem;
    }
    .system-a { background: #2563eb; }
    .system-b { background: #f97316; }
    .system-c { background: #16a34a; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Local Multi-Domain Assistant Evaluation Dashboard")

load_result = load_results(DEFAULT_RESULTS_PATH)
all_errors = list(load_result.errors)
if not all_errors:
    all_errors.extend(validate_entries(load_result.entries))

if load_result.warnings:
    for warning in load_result.warnings:
        st.warning(warning)

if all_errors:
    for error in all_errors:
        st.error(error)
    st.stop()

header_left, header_right = st.columns([3, 1])
with header_left:
    st.caption(
        f"Source: `{load_result.path}` | "
        f"Last loaded: {load_result.loaded_at} | "
        f"File modified: {load_result.source_modified_at or 'N/A'}"
    )
with header_right:
    if st.button("Refresh latest results", width="stretch"):
        st.rerun()

category_options = available_categories(load_result.entries)
selected_label = st.selectbox(
    "Question type",
    options=["All"] + category_options,
    format_func=lambda value: "All" if value == "All" else display_label(value),
)
selected_category = None if selected_label == "All" else selected_label

dashboard_data = build_dashboard_data(load_result.entries, selected_category)
summary_errors = validate_summary(dashboard_data)
if summary_errors:
    for error in summary_errors:
        st.error(error)
    st.stop()

render_summary_cards(dashboard_data["summary_cards"])

st.subheader("System-wise Comparison")
st.dataframe(
    comparison_table(dashboard_data["system_metrics"]),
    hide_index=True,
    width="stretch",
)

chart_row_one = st.columns(2)
with chart_row_one[0]:
    st.subheader("Average Accuracy Comparison")
    system_bar_chart(dashboard_data["system_metrics"], "average_accuracy", "Average accuracy")
with chart_row_one[1]:
    st.subheader("Average Latency Comparison")
    system_bar_chart(dashboard_data["system_metrics"], "average_latency", "Average latency (s)")

chart_row_two = st.columns(2)
with chart_row_two[0]:
    st.subheader("Reliability Counts")
    reliability_chart(dashboard_data["system_metrics"])
with chart_row_two[1]:
    st.subheader("Cross-domain Robustness Comparison")
    special_bar_chart(
        dashboard_data["system_metrics"],
        "cross_domain_robustness",
        "Cross-domain robustness",
    )

chart_row_three = st.columns(2)
with chart_row_three[0]:
    st.subheader("Category-wise Accuracy")
    category_chart(dashboard_data["system_metrics"])
with chart_row_three[1]:
    st.subheader("Dimension Score Comparison")
    dimension_chart(dashboard_data["system_metrics"])

render_question_comparison(dashboard_data["questions"])
