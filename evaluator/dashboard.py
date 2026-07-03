import json
import os

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Multi-Domain Assistant Evaluation",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "results")
FRESH_JSON = os.path.join(RESULT_DIR, "fresh_run_results.json")
CATEGORY_CSV = os.path.join(RESULT_DIR, "category_summary.csv")
ABLATION_CSV = os.path.join(RESULT_DIR, "ablation_summary.csv")
CONFUSION_CSV = os.path.join(RESULT_DIR, "intent_confusion_matrix.csv")
STATS_JSON = os.path.join(RESULT_DIR, "statistical_tests.json")
METRICS_SUMMARY_JSON = os.path.join(RESULT_DIR, "metrics_summary.json")
LEGACY_RESULTS = os.path.join(RESULT_DIR, "results.json")

METRIC_COLUMNS = [
    "context_contamination_rate",
    "memory_recall",
    "knowledge_growth",
    "cross_domain_robustness",
    "intent_classification_accuracy",
    "intent_macro_f1",
    "false_rejection",
    "false_rejection_rate",
    "latency_ms",
]

SUMMARY_NUMERIC_COLUMNS = METRIC_COLUMNS + [
    "n",
    "mean",
    "median",
    "std",
    "ci95_low",
    "ci95_high",
    "p95",
    "macro_f1",
    "macro_precision",
    "macro_recall",
]

TEXT_COLUMNS = [
    "system",
    "category",
    "metric",
    "suite",
    "case_id",
    "subtype",
    "query",
    "response",
    "predicted_intent",
    "gold_intent",
    "evaluation_method",
]


METRIC_SPECS = [
    (
        "context_contamination",
        "context_contamination_rate",
        "Context Contamination Rate",
        "rate",
    ),
    (
        "context_contamination",
        "false_rejection_rate",
        "False Rejection Rate",
        "rate",
    ),
    (
        "memory_recall",
        "memory_recall",
        "Memory Recall",
        "rate",
    ),
    (
        "knowledge_growth",
        "knowledge_growth",
        "Knowledge Growth",
        "rate",
    ),
    (
        "cross_domain_robustness",
        "cross_domain_robustness",
        "Cross-Domain Robustness",
        "rate",
    ),
    (
        "intent_classification",
        "intent_classification_accuracy",
        "Intent Accuracy",
        "rate",
    ),
    (
        "intent_classification",
        "intent_macro_f1",
        "Intent Macro F1",
        "macro_f1",
    ),
    (
        "latency",
        "latency_ms",
        "Latency Median",
        "latency_median",
    ),
    (
        "latency",
        "latency_ms",
        "Latency P95",
        "latency_p95",
    ),
]

SYSTEM_LABELS = {
    "A": "System A",
    "B": "System B",
    "C": "System C",
}

SYSTEM_ORDER = ["System A", "System B", "System C"]

SUMMARY_TABLE_COLUMNS = [
    ("Avg Accuracy", "avg_accuracy", 2),
    ("Avg Latency", "avg_latency", 2),
    ("Hallucinations", "hallucinations", 0),
    ("Contaminations", "contaminations", 0),
    ("Intent Accuracy", "intent_accuracy", 3),
    ("Context Switching Score", "context_switching_score", 2),
    ("Overall Composite Score", "overall_composite_score", 2),
]


def read_json(path, default):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return default


def read_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return normalize_dataframe(pd.read_csv(path))
    except (pd.errors.EmptyDataError, OSError, UnicodeDecodeError):
        return pd.DataFrame()


def stringify_cell(value):
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False)
    if value is None or pd.isna(value):
        return "N/A"
    return str(value)


def normalize_dataframe(df):
    df = df.copy()
    for column in SUMMARY_NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in TEXT_COLUMNS:
        if column in df.columns:
            df[column] = df[column].map(stringify_cell)
    return df


def dataframe_for_display(df):
    df = normalize_dataframe(df)
    display = df.copy()
    for column in display.columns:
        if column not in SUMMARY_NUMERIC_COLUMNS:
            display[column] = display[column].map(stringify_cell)
    return display.replace({pd.NA: "N/A"}).fillna("N/A")


def clean_value(value):
    if pd.isna(value):
        return None
    return float(value)


def format_percent(value):
    value = clean_value(value)
    return "N/A" if value is None else f"{value * 100:.1f}%"


def format_number(value, suffix=""):
    value = clean_value(value)
    return "N/A" if value is None else f"{value:.1f}{suffix}"


def display_system_name(system):
    return SYSTEM_LABELS.get(str(system), str(system))


def summary_metric_value(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {
        "",
        "n/a",
        "na",
        "nan",
        "none",
        "null",
    }:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def format_summary_metric(value, places):
    number = summary_metric_value(value)
    if number is None:
        return "N/A"
    if places == 0:
        return str(int(round(number)))
    return f"{number:.{places}f}"


def ordered_summary_systems(systems):
    if not isinstance(systems, dict):
        return []
    ordered = [system for system in SYSTEM_ORDER if system in systems]
    extras = sorted(system for system in systems if system not in ordered)
    return ordered + extras


def final_ranks(systems):
    scored = []
    for system, metrics in systems.items():
        if isinstance(metrics, dict):
            value = summary_metric_value(metrics.get("overall_composite_score"))
            if value is not None:
                scored.append((system, value))

    ranks = {}
    previous_value = None
    previous_rank = None
    for index, (system, value) in enumerate(
        sorted(scored, key=lambda item: item[1], reverse=True),
        start=1,
    ):
        if previous_value is not None and value == previous_value:
            rank = previous_rank
        else:
            rank = index
        ranks[system] = rank
        previous_value = value
        previous_rank = rank
    return ranks


def build_system_comparison_summary(metrics_summary):
    systems = metrics_summary.get("systems", {})
    if not isinstance(systems, dict):
        return pd.DataFrame(), pd.DataFrame()

    ranks = final_ranks(systems)
    table_rows = []
    chart_rows = []

    for system in ordered_summary_systems(systems):
        metrics = systems.get(system, {})
        if not isinstance(metrics, dict):
            continue

        table_row = {"System": system}
        for label, key, places in SUMMARY_TABLE_COLUMNS:
            table_row[label] = format_summary_metric(metrics.get(key), places)
            value = summary_metric_value(metrics.get(key))
            if value is not None:
                chart_rows.append({
                    "System": system,
                    "Metric": label,
                    "Value": value,
                })
        table_row["Final Rank"] = ranks.get(system, "N/A")
        table_rows.append(table_row)

    return pd.DataFrame(table_rows), pd.DataFrame(chart_rows)


def combine_chart_rows(category_chart_df, summary_chart_df):
    frames = []
    summary_metrics = set()

    if not summary_chart_df.empty:
        summary_metrics = set(summary_chart_df["Metric"].dropna().unique())
        frames.append(summary_chart_df)

    if not category_chart_df.empty:
        category_chart_df = category_chart_df.copy()
        category_chart_df["System"] = category_chart_df["System"].map(
            display_system_name
        )
        if summary_metrics:
            category_chart_df = category_chart_df[
                ~category_chart_df["Metric"].isin(summary_metrics)
            ]
        if not category_chart_df.empty:
            frames.append(category_chart_df)

    if not frames:
        return pd.DataFrame(columns=["System", "Metric", "Value"])
    return pd.concat(frames, ignore_index=True)


def category_row(category_df, system, category, metric):
    metric_candidates = [metric]
    if metric == "intent_classification_accuracy":
        metric_candidates.append("intent_accuracy")
    if metric == "false_rejection_rate":
        metric_candidates.append("false_rejection")
    rows = category_df[
        (category_df["system"] == system)
        & (category_df["category"] == category)
        & (category_df["metric"].isin(metric_candidates))
    ]
    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()


def build_display_rows(category_df):
    systems = sorted(category_df["system"].dropna().unique())
    display_rows = []
    chart_rows = []
    for system in systems:
        for category, metric, label, kind in METRIC_SPECS:
            row = category_row(category_df, system, category, metric)
            if not row:
                continue
            if kind == "macro_f1":
                raw_value = row.get("mean")
                if raw_value is None or pd.isna(raw_value):
                    raw_value = row.get("macro_f1")
                raw_median = None
                raw_ci_low = None
                raw_ci_high = None
                raw_p95 = None
            elif kind == "latency_median":
                raw_value = row.get("median")
                raw_median = row.get("median")
                raw_ci_low = row.get("ci95_low")
                raw_ci_high = row.get("ci95_high")
                raw_p95 = row.get("p95")
            elif kind == "latency_p95":
                raw_value = row.get("p95")
                raw_median = row.get("median")
                raw_ci_low = row.get("ci95_low")
                raw_ci_high = row.get("ci95_high")
                raw_p95 = row.get("p95")
            else:
                raw_value = row.get("mean")
                raw_median = row.get("median")
                raw_ci_low = row.get("ci95_low")
                raw_ci_high = row.get("ci95_high")
                raw_p95 = row.get("p95")

            value = clean_value(raw_value)
            display_rows.append({
                "System": system,
                "Metric": label,
                "N": int(row.get("n", 0)) if not pd.isna(row.get("n")) else 0,
                "Mean": (
                    format_percent(value)
                    if kind in {"rate", "macro_f1"}
                    else format_number(value, " ms")
                ),
                "Median": (
                    format_percent(raw_median)
                    if kind == "rate"
                    else format_number(raw_median, " ms")
                    if kind.startswith("latency")
                    else "N/A"
                ),
                "95% CI": (
                    f"{format_percent(raw_ci_low)} to {format_percent(raw_ci_high)}"
                    if kind == "rate"
                    else f"{format_number(raw_ci_low, ' ms')} to {format_number(raw_ci_high, ' ms')}"
                    if kind.startswith("latency")
                    else "N/A"
                ),
                "P95": (
                    format_number(raw_p95, " ms")
                    if kind.startswith("latency")
                    else "N/A"
                ),
            })
            if value is not None:
                chart_rows.append({
                    "System": system,
                    "Metric": label,
                    "Value": value * 100 if kind in {"rate", "macro_f1"} else value,
                })
    return pd.DataFrame(display_rows), pd.DataFrame(chart_rows)


def legacy_notice():
    st.warning(
        "Clean benchmark outputs were not found. The legacy results file is "
        "available, but it is not sufficient for the final research claim."
    )
    if os.path.exists(LEGACY_RESULTS):
        legacy = read_json(LEGACY_RESULTS, [])
        st.write(f"Legacy questions: {len(legacy)}")
        st.dataframe(
            dataframe_for_display(pd.DataFrame(legacy)),
            width="stretch",
        )
    else:
        st.error("No result files were found. Run the clean experiment first.")


st.title("Multi-Domain Assistant Evaluation")
st.caption(
    "Composite score is calculated from accuracy, contamination resistance, "
    "hallucination resistance, intent accuracy, and latency."
)

category_df = read_csv(CATEGORY_CSV)
fresh_data = read_json(FRESH_JSON, {})
metrics_summary = (
    read_json(METRICS_SUMMARY_JSON, {})
    if os.path.exists(METRICS_SUMMARY_JSON)
    else {}
)

summary_display_df, summary_chart_df = build_system_comparison_summary(
    metrics_summary
)

st.header("System Comparison Summary")
if not os.path.exists(METRICS_SUMMARY_JSON):
    st.info("Run analysis first to generate metrics_summary.json")
elif summary_display_df.empty:
    st.warning("metrics_summary.json was found, but no system metrics could be read.")
else:
    st.dataframe(summary_display_df, width="stretch", hide_index=True)

if category_df.empty:
    legacy_notice()
    st.stop()

metadata = fresh_data.get("metadata", {}) if isinstance(fresh_data, dict) else {}
fresh_rows = fresh_data.get("rows", []) if isinstance(fresh_data, dict) else []

st.header("Clean Run")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Sample Count", len(fresh_rows))
col2.metric("Systems", category_df["system"].nunique())
col3.metric("Historical Results", str(metadata.get("historical_results_included", False)))
col4.metric("Timeout", f"{metadata.get('timeout_seconds', 'N/A')} s")

model = metadata.get("model", {}) if isinstance(metadata.get("model"), dict) else {}
st.caption(
    "Model: "
    + str(model.get("name", "mistral:latest"))
    + " | Digest: "
    + str(model.get("digest", "unknown"))
    + " | Historical results included: "
    + str(metadata.get("historical_results_included", False))
)
st.caption(
    "Suite: "
    + str(metadata.get("suite", "all"))
    + " | Limit: "
    + str(metadata.get("limit", "N/A"))
    + " | Fast research mode: "
    + str(metadata.get("fast_research_mode", "N/A"))
)

display_df, category_chart_df = build_display_rows(category_df)
chart_df = combine_chart_rows(category_chart_df, summary_chart_df)

st.header("Metric Summary")
st.dataframe(dataframe_for_display(display_df), width="stretch", hide_index=True)

st.header("Metric Charts")
if chart_df.empty:
    st.info("No chartable metrics were found.")
else:
    selected_metric = st.selectbox(
        "Metric",
        sorted(chart_df["Metric"].unique()),
    )
    metric_chart = chart_df[chart_df["Metric"] == selected_metric].pivot(
        index="System",
        columns="Metric",
        values="Value",
    )
    st.bar_chart(metric_chart)

st.header("Intent Confusion Matrix")
confusion_df = read_csv(CONFUSION_CSV)
if confusion_df.empty:
    st.info("No confusion matrix was found.")
else:
    st.dataframe(dataframe_for_display(confusion_df), width="stretch", hide_index=True)

st.header("Ablation Chart")
ablation_df = read_csv(ABLATION_CSV)
if ablation_df.empty:
    st.info("No ablation summary was found.")
else:
    ablation_metric = st.selectbox(
        "Ablation metric",
        sorted(ablation_df["metric"].unique()),
    )
    subset = ablation_df[ablation_df["metric"] == ablation_metric].copy()
    subset["stage"] = pd.to_numeric(
        subset["ablation"].astype(str).str.extract(r"C(\d+)")[0],
        errors="coerce",
    )
    subset = subset.sort_values("stage")
    value_column = "median" if ablation_metric == "latency_ms" else "mean"
    st.line_chart(
        subset.set_index("ablation")[[value_column]].rename(
            columns={value_column: ablation_metric}
        )
    )
    st.dataframe(dataframe_for_display(subset), width="stretch", hide_index=True)

st.header("Paired Statistical Tests")
stats = read_json(STATS_JSON, {})
if stats:
    st.json(stats, expanded=False)
else:
    st.info("No statistical test output was found.")

st.header("Response-Level Fresh Results")
if fresh_rows:
    raw_df = normalize_dataframe(pd.DataFrame(fresh_rows))
    systems = st.multiselect(
        "Systems",
        sorted(raw_df["system"].dropna().unique()),
        default=sorted(raw_df["system"].dropna().unique()),
    )
    suites = st.multiselect(
        "Suites",
        sorted(raw_df["suite"].dropna().unique()),
        default=sorted(raw_df["suite"].dropna().unique()),
    )
    filtered = raw_df[
        raw_df["system"].isin(systems)
        & raw_df["suite"].isin(suites)
    ]
    st.dataframe(dataframe_for_display(filtered), width="stretch", hide_index=True)
else:
    st.info("Fresh response rows were not found in the JSON output.")
