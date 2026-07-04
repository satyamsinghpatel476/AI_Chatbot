from __future__ import annotations

import csv
import html
import io
import math
import statistics
from datetime import datetime
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .live_results_summarizer import compute_live_summary


SYSTEM_KEYS = ("A", "B", "C")
SYSTEM_LABELS = {
    "A": "System A",
    "B": "System B",
    "C": "System C",
}
SYSTEM_COLORS = {
    "A": "#65f4ff",
    "B": "#ff7ad9",
    "C": "#87ffc0",
}
DIMENSIONS = (
    "correctness",
    "task_fulfillment",
    "relevance",
    "completeness",
    "clarity",
    "calibration",
)
CATEGORIES = (
    "robotics",
    "daily",
    "general",
    "unknown",
    "mixed",
    "ambiguous",
    "unverifiable",
    "personal_save",
    "personal_recall",
    "learning_save",
    "learning_recall",
)
SPECIAL_METRICS = (
    "memory_recall",
    "knowledge_growth",
    "cross_domain_robustness",
    "intent_classification_accuracy",
    "domain_resolution_accuracy",
)
COUNT_METRICS = (
    "hallucination",
    "leakage",
    "contamination",
    "false_rejection",
)
NULL_STRINGS = {"", "none", "null", "nan", "n/a", "na", "missing"}


def _metadata(answer: dict[str, Any]) -> dict[str, Any]:
    metadata = answer.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _value(source: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in source:
            return source[key]
    return None


def _nested_value(answer: dict[str, Any], *keys: str) -> Any:
    metadata = _metadata(answer)
    for source in (answer, metadata):
        value = _value(source, *keys)
        if value is not None:
            return value
    return None


def as_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None

    text = str(value).strip()
    if text.lower() in NULL_STRINGS:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    label = str(value).strip().lower()
    if label in NULL_STRINGS:
        return None
    label = label.replace("-", "_").replace(" ", "_")
    aliases = {
        "robot": "robotics",
        "robotic": "robotics",
        "daily_life": "daily",
        "daily_digital_assistance": "daily",
        "general_daily": "daily",
        "consumer": "daily",
        "cross_domain": "mixed",
        "mixed_domain": "mixed",
        "robot": "robotics",
        "robotics_support": "robotics",
        "uncertain": "ambiguous",
        "fake": "unverifiable",
        "unverified": "unverifiable",
        "personal_memory_save": "personal_save",
        "personal_memory_recall": "personal_recall",
        "knowledge_save": "learning_save",
        "knowledge_recall": "learning_recall",
    }
    return aliases.get(label, label)


def display_label(value: str | None) -> str:
    if value is None:
        return "N/A"
    return value.replace("_", " ").title()


def rounded(value: float | None, places: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), places)


def mean(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return sum(valid) / len(valid) if valid else None


def median(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return float(statistics.median(valid)) if valid else None


def count_positive(values: list[float | None]) -> int | None:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return sum(1 for value in valid if value > 0)


def get_latency(answer: dict[str, Any]) -> float | None:
    latency = as_number(_nested_value(answer, "total_latency", "latency", "latency_seconds"))
    if latency is not None:
        return latency
    latency_ms = as_number(_nested_value(answer, "latency_ms"))
    return latency_ms / 1000.0 if latency_ms is not None else None


def get_question_type(entry: dict[str, Any], answer: dict[str, Any] | None = None) -> str | None:
    if answer:
        label = normalize_label(_nested_value(answer, "question_type", "category", "type"))
        if label:
            return label
        domain = normalize_label(_nested_value(answer, "resolved_domain", "domain"))
        if domain in CATEGORIES:
            return domain

    for key in ("question_type", "category", "type"):
        label = normalize_label(entry.get(key))
        if label:
            return label
    return None


def _dimension_score(answer: dict[str, Any], dimension: str) -> float | None:
    metadata = _metadata(answer)
    for source in (answer, metadata):
        scores = source.get("dimension_scores")
        if isinstance(scores, dict):
            value = as_number(scores.get(dimension))
            if value is not None:
                return value
        value = as_number(source.get(dimension))
        if value is not None:
            return value
    return None


def _special_metric(answer: dict[str, Any], metric: str) -> float | None:
    if metric == "intent_classification_accuracy":
        value = as_number(_nested_value(answer, metric, "intent_accuracy"))
        if value is not None:
            return value
        predicted = _nested_value(answer, "predicted_intent", "classified_intent", "intent")
        expected = _nested_value(answer, "expected_intent", "gold_intent")
        if predicted and expected:
            return 1.0 if str(predicted).lower() == str(expected).lower() else 0.0
        intent_correct = as_number(_nested_value(answer, "intent_correct"))
        if intent_correct is not None:
            return 1.0 if intent_correct > 0 else 0.0

    if metric == "domain_resolution_accuracy":
        value = as_number(_nested_value(answer, metric))
        if value is not None:
            return value
        resolved = _nested_value(answer, "resolved_domain", "domain")
        expected = _nested_value(answer, "expected_domain", "gold_domain")
        if resolved and expected:
            return 1.0 if str(resolved).lower() == str(expected).lower() else 0.0

    return as_number(_nested_value(answer, metric))


def _row_from_answer(entry: dict[str, Any], system: str, answer: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp": entry.get("timestamp", ""),
        "source": entry.get("source") or "manual_chat",
        "question": entry.get("question", ""),
        "question_type": get_question_type(entry, answer),
        "expected_intent": _nested_value(answer, "expected_intent", "gold_intent"),
        "system": system,
        "response": answer.get("response", ""),
        "main_rag_used": bool(_nested_value(answer, "main_rag_used")),
        "temporary_rag_used": bool(_nested_value(answer, "temporary_rag_used")),
        "combined_context_chars": as_number(_nested_value(answer, "combined_context_chars")),
        "latency": get_latency(answer),
        "accuracy": as_number(_nested_value(answer, "accuracy", "score")),
        "hallucination": as_number(_nested_value(answer, "hallucination", "hallucinated")),
        "leakage": as_number(_nested_value(answer, "leakage", "context_leakage", "privacy_leakage")),
        "contamination": as_number(_nested_value(answer, "contamination", "context_contamination", "context_contamination_rate")),
        "context_contamination_rate": as_number(_nested_value(answer, "context_contamination_rate")),
        "false_rejection": as_number(_nested_value(answer, "false_rejection")),
        "memory_recall": _special_metric(answer, "memory_recall"),
        "knowledge_growth": _special_metric(answer, "knowledge_growth"),
        "cross_domain_robustness": _special_metric(answer, "cross_domain_robustness"),
        "intent_classification_accuracy": _special_metric(answer, "intent_classification_accuracy"),
        "domain_resolution_accuracy": _special_metric(answer, "domain_resolution_accuracy"),
        "domain_resolution_accuracy_strict": _special_metric(answer, "domain_resolution_accuracy_strict"),
        "domain_resolution_accuracy_relaxed": _special_metric(answer, "domain_resolution_accuracy_relaxed"),
        "predicted_intent": _nested_value(answer, "predicted_intent", "classified_intent", "intent"),
        "resolved_domain": _nested_value(answer, "resolved_domain", "domain"),
        "judge_rationale": _nested_value(answer, "judge_rationale", "rationale"),
        "evaluation_method": _nested_value(answer, "evaluation_method"),
        "evaluation_mode": _nested_value(answer, "evaluation_mode"),
        "requires_human_review": _nested_value(answer, "requires_human_review"),
        "benchmark_compatible_live_run": bool(_nested_value(answer, "benchmark_compatible_live_run")),
        "state_reset_applied": bool(_nested_value(answer, "state_reset_applied")),
        "exact_evaluator_imported": bool(_nested_value(answer, "exact_evaluator_imported")),
        "fallback_evaluator_used": bool(_nested_value(answer, "fallback_evaluator_used")),
        "dimension_scores": {
            dimension: _dimension_score(answer, dimension)
            for dimension in DIMENSIONS
        },
    }


def collect_rows(history: list[dict[str, Any]], system: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in history:
        for system_key, answer in (entry.get("answers") or {}).items():
            if system and system_key != system:
                continue
            if isinstance(answer, dict):
                rows.append(_row_from_answer(entry, system_key, answer))
    return rows


def compute_system_metrics(history: list[dict[str, Any]], system: str) -> dict[str, Any]:
    rows = collect_rows(history, system)
    dimension_scores = {
        dimension: rounded(mean([row["dimension_scores"].get(dimension) for row in rows]))
        for dimension in DIMENSIONS
    }
    special_metrics = {
        metric: rounded(mean([row.get(metric) for row in rows]))
        for metric in SPECIAL_METRICS
    }
    category_accuracy = {
        category: rounded(mean([
            row["accuracy"]
            for row in rows
            if normalize_label(row.get("question_type")) == category
        ]))
        for category in CATEGORIES
    }

    return {
        "system": system,
        "label": SYSTEM_LABELS[system],
        "color": SYSTEM_COLORS[system],
        "count": len(rows),
        "total_questions_evaluated": len(rows),
        "avg_accuracy": rounded(mean([row["accuracy"] for row in rows])),
        "average_accuracy": rounded(mean([row["accuracy"] for row in rows])),
        "median_accuracy": rounded(median([row["accuracy"] for row in rows])),
        "avg_latency": rounded(mean([row["latency"] for row in rows])),
        "average_latency": rounded(mean([row["latency"] for row in rows])),
        "hallucination_count": count_positive([row["hallucination"] for row in rows]),
        "leakage_count": count_positive([row["leakage"] for row in rows]),
        "contamination_count": count_positive([row["contamination"] for row in rows]),
        "false_rejection_count": count_positive([row["false_rejection"] for row in rows]),
        "dimension_score_averages": dimension_scores,
        "special_metrics": special_metrics,
        "category_accuracy": category_accuracy,
    }


def _winner(
    systems: dict[str, dict[str, Any]],
    metric_path: tuple[str, ...],
    mode: str,
) -> dict[str, Any]:
    values: dict[str, float] = {}
    for system, metrics in systems.items():
        value: Any = metrics
        for key in metric_path:
            value = value.get(key) if isinstance(value, dict) else None
        number = as_number(value)
        if number is not None:
            values[system] = number

    if not values:
        return {"label": "N/A", "system": None, "value": None}

    best_value = max(values.values()) if mode == "max" else min(values.values())
    winners = [
        SYSTEM_LABELS[system]
        for system, value in values.items()
        if math.isclose(value, best_value, rel_tol=1e-9, abs_tol=1e-9)
    ]
    return {
        "label": ", ".join(winners),
        "system": winners[0] if len(winners) == 1 else None,
        "value": rounded(best_value),
    }


def available_categories(history: list[dict[str, Any]]) -> list[dict[str, str]]:
    found = {normalize_label(row.get("question_type")) for row in collect_rows(history)}
    found.discard(None)
    ordered = [category for category in CATEGORIES if category in found]
    extras = sorted(category for category in found if category not in CATEGORIES)
    return [
        {"key": category, "label": display_label(category)}
        for category in ordered + extras
    ]


def _comparison_rows(system_metrics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        ("Count", "count"),
        ("Avg Accuracy", "avg_accuracy"),
        ("Median Accuracy", "median_accuracy"),
        ("Avg Latency", "avg_latency"),
        ("Hallucination", "hallucination_count"),
        ("Leakage", "leakage_count"),
        ("Contamination", "contamination_count"),
        ("False Rejection", "false_rejection_count"),
    ]
    rows = []
    for label, key in fields:
        row = {"metric": label}
        for system in SYSTEM_KEYS:
            row[system] = system_metrics[system].get(key)
        rows.append(row)
    return rows


def question_summaries(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    questions = []
    for index, entry in enumerate(history, start=1):
        systems = {}
        for system in SYSTEM_KEYS:
            answer = (entry.get("answers") or {}).get(system)
            systems[system] = _row_from_answer(entry, system, answer) if isinstance(answer, dict) else None
        questions.append({
            "index": index,
            "timestamp": entry.get("timestamp", ""),
            "question": entry.get("question", ""),
            "question_type": get_question_type(entry),
            "question_type_label": display_label(get_question_type(entry)),
            "systems": systems,
        })
    return questions


def _evaluation_mode_summary(history: list[dict[str, Any]]) -> dict[str, Any]:
    methods = {
        str(row.get("evaluation_method") or "").strip()
        for row in collect_rows(history)
        if str(row.get("evaluation_method") or "").strip()
    }
    if not methods:
        return {
            "evaluation_mode": "none",
            "evaluation_mode_label": "No evaluated responses",
            "evaluation_method": "N/A",
            "evaluation_message": "No session results yet.",
            "evaluation_warning": None,
        }

    has_research = "blind_mistral_judge_with_gold_constraints" in methods
    has_demo = any(method != "blind_mistral_judge_with_gold_constraints" for method in methods)
    if has_research and has_demo:
        return {
            "evaluation_mode": "mixed",
            "evaluation_mode_label": "Mixed evaluation modes",
            "evaluation_method": "mixed",
            "evaluation_message": "Session contains both research evaluator and demo metrics.",
            "evaluation_warning": "Avoid mixing modes for paper-facing comparisons.",
        }
    if has_research:
        return {
            "evaluation_mode": "research",
            "evaluation_mode_label": "Research evaluator",
            "evaluation_method": "blind_mistral_judge_with_gold_constraints",
            "evaluation_message": "Metrics generated with benchmark-compatible evaluator.",
            "evaluation_warning": None,
        }
    return {
        "evaluation_mode": "demo",
        "evaluation_mode_label": "Demo metrics",
        "evaluation_method": ", ".join(sorted(methods)),
        "evaluation_message": "Demo metrics only; not for paper results.",
        "evaluation_warning": "These metrics are approximate and should not be used in paper.",
    }


def compute_session_metrics(history: list[dict[str, Any]]) -> dict[str, Any]:
    return compute_live_summary(history)


def _iter_csv_rows(history: list[dict[str, Any]], metrics: dict[str, Any] | None = None):
    export_metadata = (metrics or {}).get("export_metadata", {})
    for row in collect_rows(history):
        dimension_scores = row["dimension_scores"]
        yield {
            "export_evaluation_mode": export_metadata.get("evaluation_mode"),
            "export_question_source": export_metadata.get("question_source"),
            "export_question_count": export_metadata.get("question_count"),
            "export_evaluator_method": export_metadata.get("evaluator_method"),
            "export_exact_evaluator_imported": export_metadata.get("exact_evaluator_imported"),
            "export_fallback_evaluator_used": export_metadata.get("fallback_evaluator_used"),
            "export_main_rag_used": export_metadata.get("main_rag_used"),
            "export_state_reset_applied": export_metadata.get("state_reset_applied"),
            "timestamp": row["timestamp"],
            "source": row["source"],
            "question": row["question"],
            "question_type": row["question_type"],
            "expected_intent": row["expected_intent"],
            "system": row["system"],
            "response": row["response"],
            "latency": row["latency"],
            "accuracy": row["accuracy"],
            "hallucination": row["hallucination"],
            "leakage": row["leakage"],
            "contamination": row["contamination"],
            "false_rejection": row["false_rejection"],
            "correctness": dimension_scores.get("correctness"),
            "task_fulfillment": dimension_scores.get("task_fulfillment"),
            "relevance": dimension_scores.get("relevance"),
            "completeness": dimension_scores.get("completeness"),
            "clarity": dimension_scores.get("clarity"),
            "calibration": dimension_scores.get("calibration"),
            "context_contamination_rate": row["context_contamination_rate"],
            "judge_rationale": row["judge_rationale"],
            "evaluation_method": row["evaluation_method"],
            "predicted_intent": row["predicted_intent"],
            "resolved_domain": row["resolved_domain"],
            "domain_resolution_accuracy": row["domain_resolution_accuracy"],
            "domain_resolution_accuracy_strict": row["domain_resolution_accuracy_strict"],
            "domain_resolution_accuracy_relaxed": row["domain_resolution_accuracy_relaxed"],
            "main_rag_used": row["main_rag_used"],
            "temporary_rag_used": row["temporary_rag_used"],
            "memory_recall": row["memory_recall"],
            "knowledge_growth": row["knowledge_growth"],
            "cross_domain_robustness": row["cross_domain_robustness"],
            "intent_classification_accuracy": row["intent_classification_accuracy"],
            "requires_human_review": row["requires_human_review"],
            "benchmark_compatible_live_run": row["benchmark_compatible_live_run"],
            "state_reset_applied": row["state_reset_applied"],
            "exact_evaluator_imported": row["exact_evaluator_imported"],
            "fallback_evaluator_used": row["fallback_evaluator_used"],
            "combined_context_chars": row["combined_context_chars"],
        }


def build_csv(history: list[dict[str, Any]], metrics: dict[str, Any] | None = None) -> str:
    metrics = metrics or compute_session_metrics(history)
    output = io.StringIO()
    columns = [
        "export_evaluation_mode",
        "export_question_source",
        "export_question_count",
        "export_evaluator_method",
        "export_exact_evaluator_imported",
        "export_fallback_evaluator_used",
        "export_main_rag_used",
        "export_state_reset_applied",
        "timestamp",
        "source",
        "question",
        "question_type",
        "expected_intent",
        "system",
        "response",
        "latency",
        "accuracy",
        "hallucination",
        "leakage",
        "contamination",
        "false_rejection",
        "correctness",
        "task_fulfillment",
        "relevance",
        "completeness",
        "clarity",
        "calibration",
        "context_contamination_rate",
        "judge_rationale",
        "evaluation_method",
        "predicted_intent",
        "resolved_domain",
        "domain_resolution_accuracy",
        "domain_resolution_accuracy_strict",
        "domain_resolution_accuracy_relaxed",
        "main_rag_used",
        "temporary_rag_used",
        "memory_recall",
        "knowledge_growth",
        "cross_domain_robustness",
        "intent_classification_accuracy",
        "requires_human_review",
        "benchmark_compatible_live_run",
        "state_reset_applied",
        "exact_evaluator_imported",
        "fallback_evaluator_used",
        "combined_context_chars",
    ]
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    for row in _iter_csv_rows(history, metrics):
        writer.writerow(row)
    return output.getvalue()


def _display(value: Any, places: int = 3) -> str:
    number = as_number(value)
    if number is not None:
        return str(round(number, places))
    if value in (None, ""):
        return "N/A"
    return str(value)


def _styled_table(data: list[list[Any]], header_color: str = "#2b145f") -> Table:
    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_color)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b88cff")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3efff")]),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))
    return table


def _system_table(metrics: dict[str, Any]) -> list[list[Any]]:
    data = [[
        "System",
        "Count",
        "Avg Accuracy",
        "Median Accuracy",
        "Avg Latency",
        "Hallucination",
        "Leakage",
        "Contamination",
        "False Rejection",
    ]]
    for system in SYSTEM_KEYS:
        item = metrics["systems"][system]
        data.append([
            SYSTEM_LABELS[system],
            item["count"],
            _display(item["avg_accuracy"]),
            _display(item["median_accuracy"]),
            _display(item["avg_latency"]),
            _display(item["hallucination_count"], 0),
            _display(item["leakage_count"], 0),
            _display(item["contamination_count"], 0),
            _display(item["false_rejection_count"], 0),
        ])
    return data


def _dimension_table(metrics: dict[str, Any]) -> list[list[Any]]:
    data = [["System", *[display_label(dimension) for dimension in DIMENSIONS]]]
    for system in SYSTEM_KEYS:
        scores = metrics["systems"][system]["dimension_score_averages"]
        data.append([SYSTEM_LABELS[system], *[_display(scores.get(dimension)) for dimension in DIMENSIONS]])
    return data


def _special_table(metrics: dict[str, Any]) -> list[list[Any]]:
    data = [[
        "System",
        "Hallucination",
        "Contamination",
        "Leakage",
        "Memory Recall",
        "Knowledge Growth",
        "Cross-domain",
        "Intent Accuracy",
        "Domain Resolution",
    ]]
    for system in SYSTEM_KEYS:
        item = metrics["systems"][system]
        special = metrics["systems"][system]["special_metrics"]
        data.append([
            SYSTEM_LABELS[system],
            _display(item.get("hallucination_count"), 0),
            _display(item.get("contamination_count"), 0),
            _display(item.get("leakage_count"), 0),
            _display(special.get("memory_recall")),
            _display(special.get("knowledge_growth")),
            _display(special.get("cross_domain_robustness")),
            _display(special.get("intent_classification_accuracy")),
            _display(special.get("domain_resolution_accuracy")),
        ])
    return data


def _category_table(metrics: dict[str, Any]) -> list[list[Any]]:
    categories = metrics["categories"] or metrics["all_categories"]
    data = [["Category", "System A", "System B", "System C"]]
    for category in categories:
        key = category["key"]
        data.append([
            category["label"],
            *[
                _display(metrics["systems"][system]["category_accuracy"].get(key))
                for system in SYSTEM_KEYS
            ],
        ])
    return data


def _history_source_counts(history: list[dict[str, Any]]) -> dict[str, int]:
    manual_count = sum(
        1
        for entry in history
        if (entry.get("source") or "manual_chat") == "manual_chat"
    )
    auto_pdf_count = sum(
        1
        for entry in history
        if entry.get("source") == "auto_pdf"
    )
    return {
        "manual_chat": manual_count,
        "auto_pdf": auto_pdf_count,
        "total": len(history),
    }


def build_pdf(history: list[dict[str, Any]], metrics: dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(letter),
        title="Voice Assistant Session Report",
        leftMargin=24,
        rightMargin=24,
        topMargin=24,
        bottomMargin=24,
    )
    styles = getSampleStyleSheet()
    source_counts = _history_source_counts(history)
    story = [
        Paragraph("Local Multi-Domain AI Assistant", styles["Title"]),
        Paragraph(f"Session report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]),
        Paragraph(f"Evaluation mode: {html.escape(str(metrics.get('evaluation_mode_label', 'N/A')))}", styles["Normal"]),
        Paragraph(f"Question source: {html.escape(str(metrics.get('question_source', 'N/A')))}", styles["Normal"]),
        Paragraph(f"Evaluator method: {html.escape(str(metrics.get('evaluation_method', 'N/A')))}", styles["Normal"]),
        Paragraph(f"Exact evaluator imported: {html.escape(str(metrics.get('export_metadata', {}).get('exact_evaluator_imported', False)))}", styles["Normal"]),
        Paragraph(f"Fallback evaluator used: {html.escape(str(metrics.get('export_metadata', {}).get('fallback_evaluator_used', False)))}", styles["Normal"]),
        Paragraph(f"Main RAG used: {html.escape(str(metrics.get('export_metadata', {}).get('main_rag_used', False)))}", styles["Normal"]),
        Paragraph(f"State reset applied: {html.escape(str(metrics.get('export_metadata', {}).get('state_reset_applied', False)))}", styles["Normal"]),
        Paragraph(html.escape(str(metrics.get("evaluation_message", ""))), styles["Normal"]),
        Paragraph(f"Manual questions count: {source_counts['manual_chat']}", styles["Normal"]),
        Paragraph(f"Auto PDF questions count: {source_counts['auto_pdf']}", styles["Normal"]),
        Paragraph(f"Total questions: {source_counts['total']}", styles["Normal"]),
        Paragraph(f"Main project RAG used count: {metrics.get('summary_cards', {}).get('main_rag_used_count', 0)}", styles["Normal"]),
        Paragraph(f"Temporary RAG used count: {metrics.get('summary_cards', {}).get('temporary_rag_used_count', 0)}", styles["Normal"]),
        Spacer(1, 10),
        Paragraph("Main Comparison", styles["Heading2"]),
        _styled_table(_system_table(metrics)),
        Spacer(1, 12),
        Paragraph("Dimension Scores", styles["Heading2"]),
        _styled_table(_dimension_table(metrics)),
        Spacer(1, 12),
        Paragraph("Special Metrics", styles["Heading2"]),
        _styled_table(_special_table(metrics)),
        Spacer(1, 12),
        Paragraph("Category-wise Accuracy", styles["Heading2"]),
        _styled_table(_category_table(metrics)),
        Spacer(1, 12),
    ]
    warning = metrics.get("evaluation_warning")
    if warning:
        story.insert(4, Paragraph(f"Warning: {html.escape(str(warning))}", styles["Normal"]))

    if not history:
        story.append(Paragraph("No session results available yet.", styles["Normal"]))
    else:
        story.append(Paragraph("Recent Question-Answer Summary", styles["Heading2"]))
        for entry in history[-8:]:
            question = html.escape(str(entry.get("question", ""))[:700])
            source = html.escape(str(entry.get("source") or "manual_chat"))
            story.append(Paragraph(f"<b>Source:</b> {source}", styles["BodyText"]))
            story.append(Paragraph(f"<b>Q:</b> {question}", styles["BodyText"]))
            for system_key, answer in (entry.get("answers") or {}).items():
                metadata = _metadata(answer) if isinstance(answer, dict) else {}
                rag_note = (
                    f"main_rag_used={metadata.get('main_rag_used', False)}, "
                    f"temporary_rag_used={metadata.get('temporary_rag_used', False)}, "
                    f"combined_context_chars={metadata.get('combined_context_chars', 0)}"
                )
                response = html.escape(str(answer.get("response", ""))[:900])
                method = html.escape(str(_nested_value(answer, "evaluation_method") or "N/A"))
                rationale = html.escape(str(_nested_value(answer, "judge_rationale", "rationale") or ""))
                story.append(Paragraph(f"<b>System {system_key}:</b> {response}", styles["BodyText"]))
                story.append(Paragraph(f"<b>Evaluation method:</b> {method}", styles["BodyText"]))
                if rationale:
                    story.append(Paragraph(f"<b>Judge rationale:</b> {rationale[:700]}", styles["BodyText"]))
                story.append(Paragraph(f"<b>RAG:</b> {html.escape(rag_note)}", styles["BodyText"]))
            story.append(Spacer(1, 8))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()
