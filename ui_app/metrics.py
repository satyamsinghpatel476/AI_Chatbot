from __future__ import annotations

import math
import statistics
from typing import Any


SYSTEMS = ("A", "B", "C")
SYSTEM_LABELS = {
    "A": "System A",
    "B": "System B",
    "C": "System C",
}
SYSTEM_COLORS = {
    "A": "#2563eb",
    "B": "#f97316",
    "C": "#16a34a",
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

NULL_STRINGS = {"", "none", "null", "nan", "n/a", "na", "missing"}


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
        "consumer": "daily",
        "cross_domain": "mixed",
        "mixed_domain": "mixed",
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


def mean(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def median(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return float(statistics.median(valid))


def rounded(value: float | None, places: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), places)


def get_system_result(entry: dict[str, Any], system: str) -> dict[str, Any] | None:
    results = entry.get("results")
    if not isinstance(results, dict):
        return None

    candidates = (system, SYSTEM_LABELS[system], SYSTEM_LABELS[system].lower())
    for key in candidates:
        result = results.get(key)
        if isinstance(result, dict):
            return result
    return None


def get_question_type(entry: dict[str, Any], result: dict[str, Any] | None = None) -> str | None:
    if result:
        label = normalize_label(result.get("question_type"))
        if label:
            return label

    for key in ("question_type", "category", "type"):
        label = normalize_label(entry.get(key))
        if label:
            return label

    for system in SYSTEMS:
        system_result = get_system_result(entry, system)
        if system_result:
            label = normalize_label(system_result.get("question_type"))
            if label:
                return label
    return None


def get_latency(result: dict[str, Any]) -> float | None:
    for key in ("total_latency", "latency", "latency_seconds"):
        latency = as_number(result.get(key))
        if latency is not None:
            return latency

    latency_ms = as_number(result.get("latency_ms"))
    if latency_ms is not None:
        return latency_ms / 1000.0
    return None


def get_flag(result: dict[str, Any], key: str, fallback_key: str | None = None) -> float | None:
    value = as_number(result.get(key))
    if value is None and fallback_key:
        value = as_number(result.get(fallback_key))
    return value


def count_positive(values: list[float | None]) -> int:
    return sum(1 for value in values if value is not None and value > 0)


def collect_system_rows(entries: list[dict[str, Any]], system: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, entry in enumerate(entries, start=1):
        result = get_system_result(entry, system)
        if result is None:
            continue

        rows.append(
            {
                "entry_index": index,
                "entry": entry,
                "result": result,
                "question_type": get_question_type(entry, result),
                "accuracy": as_number(result.get("accuracy")),
                "latency": get_latency(result),
                "hallucination": get_flag(result, "hallucination"),
                "leakage": get_flag(result, "leakage"),
                "contamination": get_flag(
                    result,
                    "contamination",
                    "context_contamination_rate",
                ),
                "false_rejection": get_flag(result, "false_rejection"),
            }
        )
    return rows


def compute_system_metrics(entries: list[dict[str, Any]], system: str) -> dict[str, Any]:
    rows = collect_system_rows(entries, system)

    dimension_scores: dict[str, float | None] = {}
    for dimension in DIMENSIONS:
        values: list[float | None] = []
        for row in rows:
            scores = row["result"].get("dimension_scores")
            if isinstance(scores, dict):
                values.append(as_number(scores.get(dimension)))
        dimension_scores[dimension] = rounded(mean(values))

    category_accuracy: dict[str, float | None] = {}
    for category in CATEGORIES:
        values = [
            row["accuracy"]
            for row in rows
            if normalize_label(row["question_type"]) == category
        ]
        category_accuracy[category] = rounded(mean(values))

    special_metrics: dict[str, float | None] = {}
    for metric in SPECIAL_METRICS:
        values = [as_number(row["result"].get(metric)) for row in rows]
        special_metrics[metric] = rounded(mean(values))

    return {
        "system": system,
        "label": SYSTEM_LABELS[system],
        "total_questions_evaluated": len(rows),
        "average_accuracy": rounded(mean([row["accuracy"] for row in rows])),
        "median_accuracy": rounded(median([row["accuracy"] for row in rows])),
        "average_latency": rounded(mean([row["latency"] for row in rows])),
        "hallucination_count": count_positive([row["hallucination"] for row in rows]),
        "leakage_count": count_positive([row["leakage"] for row in rows]),
        "contamination_count": count_positive([row["contamination"] for row in rows]),
        "false_rejection_count": count_positive([row["false_rejection"] for row in rows]),
        "dimension_score_averages": dimension_scores,
        "category_accuracy": category_accuracy,
        "special_metrics": special_metrics,
    }


def entry_matches_category(entry: dict[str, Any], selected_category: str | None) -> bool:
    if not selected_category:
        return True
    target = normalize_label(selected_category)
    labels = {get_question_type(entry)}
    for system in SYSTEMS:
        result = get_system_result(entry, system)
        if result:
            labels.add(get_question_type(entry, result))
    return target in labels


def filter_entries(
    entries: list[dict[str, Any]],
    selected_category: str | None = None,
) -> list[dict[str, Any]]:
    if not selected_category:
        return entries
    return [entry for entry in entries if entry_matches_category(entry, selected_category)]


def available_categories(entries: list[dict[str, Any]]) -> list[str]:
    found = set()
    for entry in entries:
        label = get_question_type(entry)
        if label:
            found.add(label)
    ordered = [category for category in CATEGORIES if category in found]
    extras = sorted(found - set(CATEGORIES))
    return ordered + extras


def _winner(
    systems: dict[str, dict[str, Any]],
    metric_path: tuple[str, ...],
    mode: str,
) -> dict[str, Any]:
    values: dict[str, float] = {}
    for system, metrics in systems.items():
        value: Any = metrics
        for key in metric_path:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        number = as_number(value)
        if number is not None:
            values[system] = number

    if not values:
        return {"label": "N/A", "systems": [], "value": None}

    best_value = max(values.values()) if mode == "max" else min(values.values())
    winners = [
        SYSTEM_LABELS[system]
        for system, value in values.items()
        if math.isclose(value, best_value, rel_tol=1e-9, abs_tol=1e-9)
    ]
    return {
        "label": ", ".join(winners),
        "systems": winners,
        "value": rounded(best_value),
    }


def summarize_cards(
    entries: list[dict[str, Any]],
    systems: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "total_questions": len(entries),
        "best_accuracy_system": _winner(systems, ("average_accuracy",), "max"),
        "fastest_system": _winner(systems, ("average_latency",), "min"),
        "lowest_contamination_system": _winner(systems, ("contamination_count",), "min"),
        "best_cross_domain_robustness_system": _winner(
            systems,
            ("special_metrics", "cross_domain_robustness"),
            "max",
        ),
    }


def summarize_result_for_question(result: dict[str, Any] | None) -> dict[str, Any]:
    if result is None:
        return {
            "response": "N/A",
            "metrics": {},
            "dimension_scores": {},
        }

    return {
        "response": result.get("response") or "N/A",
        "metrics": {
            "accuracy": rounded(as_number(result.get("accuracy"))),
            "latency": rounded(get_latency(result)),
            "hallucination": get_flag(result, "hallucination"),
            "leakage": get_flag(result, "leakage"),
            "contamination": get_flag(
                result,
                "contamination",
                "context_contamination_rate",
            ),
            "false_rejection": get_flag(result, "false_rejection"),
            "memory_recall": rounded(as_number(result.get("memory_recall"))),
            "knowledge_growth": rounded(as_number(result.get("knowledge_growth"))),
            "cross_domain_robustness": rounded(
                as_number(result.get("cross_domain_robustness"))
            ),
            "intent_classification_accuracy": rounded(
                as_number(result.get("intent_classification_accuracy"))
            ),
            "domain_resolution_accuracy": rounded(
                as_number(result.get("domain_resolution_accuracy"))
            ),
        },
        "dimension_scores": result.get("dimension_scores")
        if isinstance(result.get("dimension_scores"), dict)
        else {},
    }


def question_summaries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    for index, entry in enumerate(entries, start=1):
        questions.append(
            {
                "index": index,
                "question": entry.get("question") or "N/A",
                "question_type": get_question_type(entry),
                "systems": {
                    system: summarize_result_for_question(
                        get_system_result(entry, system)
                    )
                    for system in SYSTEMS
                },
            }
        )
    return questions


def build_dashboard_data(
    entries: list[dict[str, Any]],
    selected_category: str | None = None,
) -> dict[str, Any]:
    filtered_entries = filter_entries(entries, selected_category)
    system_metrics = {
        system: compute_system_metrics(filtered_entries, system)
        for system in SYSTEMS
    }

    return {
        "selected_category": selected_category,
        "available_categories": available_categories(entries),
        "summary_cards": summarize_cards(filtered_entries, system_metrics),
        "system_metrics": system_metrics,
        "questions": question_summaries(filtered_entries),
    }


def validate_summary(dashboard_data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    systems = dashboard_data.get("system_metrics")
    if not isinstance(systems, dict):
        return ["Summary metrics were not computed."]

    for system in SYSTEMS:
        metrics = systems.get(system)
        if not isinstance(metrics, dict):
            errors.append(f"Summary metrics missing for System {system}.")
            continue
        if "average_accuracy" not in metrics or "average_latency" not in metrics:
            errors.append(f"Incomplete summary metrics for System {system}.")
    return errors
