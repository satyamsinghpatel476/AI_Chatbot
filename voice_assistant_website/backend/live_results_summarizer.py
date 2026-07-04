from __future__ import annotations

import json
import math
import statistics
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_RESULTS_PATH = PROJECT_ROOT / "evaluator" / "results" / "results.json"
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
COUNT_METRICS = (
    "hallucination",
    "leakage",
    "contamination",
    "false_rejection",
)
SPECIAL_METRICS = (
    "memory_recall",
    "knowledge_growth",
    "cross_domain_robustness",
    "intent_classification_accuracy",
    "domain_resolution_accuracy",
    "domain_resolution_accuracy_strict",
    "domain_resolution_accuracy_relaxed",
)
NULL_STRINGS = {"", "none", "null", "nan", "n/a", "na", "missing"}


def configure_project_runtime() -> None:
    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


configure_project_runtime()

try:
    import summarize_results as official_summarizer  # type: ignore
except Exception:  # pragma: no cover - degraded local environment only
    official_summarizer = None  # type: ignore[assignment]


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


def mean(values: Any) -> float | None:
    valid = [number for value in values if (number := as_number(value)) is not None]
    return sum(valid) / len(valid) if valid else None


def median(values: Any) -> float | None:
    valid = [number for value in values if (number := as_number(value)) is not None]
    return float(statistics.median(valid)) if valid else None


def rounded(value: float | None, places: int = 4) -> float | None:
    return None if value is None else round(float(value), places)


def count_positive(values: Any) -> int | None:
    valid = [number for value in values if (number := as_number(value)) is not None]
    if not valid:
        return None
    return sum(1 for number in valid if number > 0)


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
        "hallucination": "unverifiable",
        "personal": "personal_recall",
        "memory": "personal_recall",
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


def _latency(answer: dict[str, Any]) -> float | None:
    latency = as_number(_nested_value(answer, "total_latency", "latency", "latency_seconds"))
    if latency is not None:
        return latency
    latency_ms = as_number(_nested_value(answer, "latency_ms"))
    return latency_ms / 1000.0 if latency_ms is not None else None


def _question_type(entry: dict[str, Any], answer: dict[str, Any] | None = None) -> str | None:
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


def _special_metric(answer: dict[str, Any], metric: str) -> float | None:
    value = as_number(_nested_value(answer, metric))
    if value is not None:
        return value

    if metric == "intent_classification_accuracy":
        value = as_number(_nested_value(answer, "intent_accuracy", "intent_correct"))
        if value is not None:
            return 1.0 if value > 0 else 0.0
        predicted = normalize_label(_nested_value(answer, "predicted_intent", "corrected_predicted_intent", "classified_intent", "intent"))
        expected = normalize_label(_nested_value(answer, "expected_intent", "gold_intent"))
        if predicted and expected:
            return 1.0 if predicted == expected else 0.0

    if metric.startswith("domain_resolution_accuracy"):
        resolved = normalize_label(_nested_value(answer, "resolved_domain", "domain"))
        expected = normalize_label(_nested_value(answer, "expected_domain", "gold_domain", "expected_intent"))
        if resolved and expected:
            if metric.endswith("_relaxed") and {resolved, expected} <= {"general", "unknown"}:
                return 1.0
            return 1.0 if resolved == expected else 0.0

    return None


def row_from_answer(entry: dict[str, Any], system: str, answer: dict[str, Any]) -> dict[str, Any]:
    metadata = _metadata(answer)
    question_source = (
        entry.get("question_source")
        or (entry.get("metadata") or {}).get("question_source")
        or metadata.get("question_source")
        or entry.get("source")
        or "manual_chat"
    )
    return {
        "timestamp": entry.get("timestamp", ""),
        "source": entry.get("source") or question_source or "manual_chat",
        "question_source": question_source,
        "question": entry.get("question", ""),
        "question_type": _question_type(entry, answer),
        "expected_intent": _nested_value(answer, "expected_intent", "gold_intent"),
        "system": system,
        "response": answer.get("response", ""),
        "main_rag_used": bool(_nested_value(answer, "main_rag_used")),
        "temporary_rag_used": bool(_nested_value(answer, "temporary_rag_used")),
        "combined_context_chars": as_number(_nested_value(answer, "combined_context_chars")),
        "latency": _latency(answer),
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
        "predicted_intent": _nested_value(answer, "predicted_intent", "corrected_predicted_intent", "classified_intent", "intent"),
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
        answers = entry.get("answers") or entry.get("results") or {}
        for system_key, answer in answers.items():
            if system and system_key != system:
                continue
            if isinstance(answer, dict):
                rows.append(row_from_answer(entry, system_key, answer))
    return rows


def official_entries_to_history(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for index, entry in enumerate(entries, start=1):
        results = entry.get("results") if isinstance(entry, dict) else None
        if not isinstance(results, dict):
            continue
        first_result = next((value for value in results.values() if isinstance(value, dict)), {})
        question_type = first_result.get("question_type") if isinstance(first_result, dict) else None
        history.append({
            "timestamp": entry.get("timestamp", ""),
            "question": entry.get("question", ""),
            "answers": deepcopy(results),
            "source": "official_results",
            "question_source": "evaluator/results/results.json",
            "question_type": question_type,
            "metadata": {
                "official_result_index": index,
                "question_source": "evaluator/results/results.json",
                "evaluation_mode": "official_results",
            },
        })
    return history


def load_official_results(path: Path | None = None) -> list[dict[str, Any]]:
    target = path or OFFICIAL_RESULTS_PATH
    with target.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Expected evaluator results file to contain a list.")
    return data


def official_results_available() -> bool:
    return OFFICIAL_RESULTS_PATH.exists()


def official_results_count() -> int:
    if not official_results_available():
        return 0
    try:
        return len(load_official_results())
    except Exception:
        return 0


def system_rows_from_official(entries: list[dict[str, Any]], system: str) -> list[dict[str, Any]]:
    if official_summarizer is not None:
        return official_summarizer.system_rows(entries, system)
    rows = []
    for entry in entries:
        result = (entry.get("results") or {}).get(system)
        if isinstance(result, dict):
            rows.append({"question": entry.get("question"), **result})
    return rows


def summarize_results_like(entries: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "benchmark_entries": len(entries),
        "systems": {},
    }
    for system in SYSTEM_KEYS:
        rows = system_rows_from_official(entries, system)
        dimension_averages = (
            official_summarizer.dimension_averages(rows)
            if official_summarizer is not None
            else {
                dimension: mean(
                    row.get("dimension_scores", {}).get(dimension)
                    for row in rows
                    if isinstance(row.get("dimension_scores"), dict)
                )
                for dimension in DIMENSIONS
            }
        )
        category_accuracy = (
            official_summarizer.category_accuracy(rows)
            if official_summarizer is not None
            else {
                category: mean(row.get("accuracy") for row in rows if row.get("question_type") == category)
                for category in CATEGORIES
            }
        )
        memory = (
            official_summarizer.subset_metric(rows, {"personal_save", "personal_recall"}, "memory_recall")
            if official_summarizer is not None
            else mean(row.get("memory_recall") for row in rows if row.get("question_type") in {"personal_save", "personal_recall"})
        )
        learning = (
            official_summarizer.subset_metric(rows, {"learning_save", "learning_recall"}, "knowledge_growth")
            if official_summarizer is not None
            else mean(row.get("knowledge_growth") for row in rows if row.get("question_type") in {"learning_save", "learning_recall"})
        )
        mixed = (
            official_summarizer.subset_metric(rows, {"mixed"}, "cross_domain_robustness")
            if official_summarizer is not None
            else mean(row.get("cross_domain_robustness") for row in rows if row.get("question_type") == "mixed")
        )
        intent = mean(
            row.get("intent_classification_accuracy")
            for row in rows
            if row.get("predicted_intent") is not None
            or row.get("corrected_predicted_intent") is not None
        )
        output["systems"][system] = {
            "count": len(rows),
            "avg_accuracy": mean(row.get("accuracy") for row in rows),
            "median_accuracy": median(row.get("accuracy") for row in rows),
            "avg_latency": mean(row.get("latency") for row in rows),
            "hallucination_count": sum(1 for row in rows if row.get("hallucination")),
            "leakage_count": sum(1 for row in rows if row.get("leakage")),
            "contamination_count": sum(1 for row in rows if row.get("contamination")),
            "false_rejection_count": sum(1 for row in rows if row.get("false_rejection")),
            "dimension_score_averages": dimension_averages,
            "category_accuracy": category_accuracy,
            "special_metrics": {
                "memory_recall": memory,
                "knowledge_growth": learning,
                "cross_domain_robustness": mixed,
                "intent_classification_accuracy": intent,
                "domain_resolution_accuracy": mean(row.get("domain_resolution_accuracy") for row in rows) if system == "C" else None,
                "domain_resolution_accuracy_strict": mean(row.get("domain_resolution_accuracy_strict") for row in rows) if system == "C" else None,
                "domain_resolution_accuracy_relaxed": mean(row.get("domain_resolution_accuracy_relaxed") for row in rows) if system == "C" else None,
            },
        }
    return output


def _winner(systems: dict[str, dict[str, Any]], metric_path: tuple[str, ...], mode: str) -> dict[str, Any]:
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


def _system_metrics(rows: list[dict[str, Any]], system: str) -> dict[str, Any]:
    dimension_scores = {
        dimension: rounded(mean(row["dimension_scores"].get(dimension) for row in rows))
        for dimension in DIMENSIONS
    }
    special_metrics = {
        "memory_recall": rounded(mean(row.get("memory_recall") for row in rows if row.get("question_type") in {"personal_save", "personal_recall"})),
        "knowledge_growth": rounded(mean(row.get("knowledge_growth") for row in rows if row.get("question_type") in {"learning_save", "learning_recall"})),
        "cross_domain_robustness": rounded(mean(row.get("cross_domain_robustness") for row in rows if row.get("question_type") == "mixed")),
        "intent_classification_accuracy": rounded(mean(
            row.get("intent_classification_accuracy")
            for row in rows
            if row.get("predicted_intent") is not None
            or row.get("intent_classification_accuracy") is not None
        )),
        "domain_resolution_accuracy": rounded(mean(row.get("domain_resolution_accuracy") for row in rows if system == "C")),
        "domain_resolution_accuracy_strict": rounded(mean(row.get("domain_resolution_accuracy_strict") for row in rows if system == "C")),
        "domain_resolution_accuracy_relaxed": rounded(mean(row.get("domain_resolution_accuracy_relaxed") for row in rows if system == "C")),
    }
    category_accuracy = {
        category: rounded(mean(row.get("accuracy") for row in rows if normalize_label(row.get("question_type")) == category))
        for category in CATEGORIES
    }
    return {
        "system": system,
        "label": SYSTEM_LABELS[system],
        "color": SYSTEM_COLORS[system],
        "count": len(rows),
        "total_questions_evaluated": len(rows),
        "avg_accuracy": rounded(mean(row.get("accuracy") for row in rows)),
        "average_accuracy": rounded(mean(row.get("accuracy") for row in rows)),
        "median_accuracy": rounded(median(row.get("accuracy") for row in rows)),
        "avg_latency": rounded(mean(row.get("latency") for row in rows)),
        "average_latency": rounded(mean(row.get("latency") for row in rows)),
        "hallucination_count": count_positive(row.get("hallucination") for row in rows),
        "leakage_count": count_positive(row.get("leakage") for row in rows),
        "contamination_count": count_positive(row.get("contamination") for row in rows),
        "false_rejection_count": count_positive(row.get("false_rejection") for row in rows),
        "dimension_score_averages": dimension_scores,
        "special_metrics": special_metrics,
        "category_accuracy": category_accuracy,
    }


def available_categories(history: list[dict[str, Any]]) -> list[dict[str, str]]:
    found = {normalize_label(row.get("question_type")) for row in collect_rows(history)}
    found.discard(None)
    ordered = [category for category in CATEGORIES if category in found]
    extras = sorted(category for category in found if category not in CATEGORIES)
    return [{"key": category, "label": display_label(category)} for category in ordered + extras]


def question_summaries(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    questions = []
    for index, entry in enumerate(history, start=1):
        systems = {}
        answers = entry.get("answers") or entry.get("results") or {}
        for system in SYSTEM_KEYS:
            answer = answers.get(system)
            systems[system] = row_from_answer(entry, system, answer) if isinstance(answer, dict) else None
        questions.append({
            "index": index,
            "timestamp": entry.get("timestamp", ""),
            "question": entry.get("question", ""),
            "question_type": _question_type(entry),
            "question_type_label": display_label(_question_type(entry)),
            "systems": systems,
        })
    return questions


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
    return [
        {"metric": label, **{system: system_metrics[system].get(key) for system in SYSTEM_KEYS}}
        for label, key in fields
    ]


def _mode_summary(history: list[dict[str, Any]], rows: list[dict[str, Any]], *, mode: str | None = None) -> dict[str, Any]:
    methods = {
        str(row.get("evaluation_method") or "").strip()
        for row in rows
        if str(row.get("evaluation_method") or "").strip()
    }
    sources = {
        str(row.get("question_source") or row.get("source") or "").strip()
        for row in rows
        if str(row.get("question_source") or row.get("source") or "").strip()
    }
    method_text = ", ".join(sorted(methods)) if methods else "N/A"
    source_text = ", ".join(sorted(sources)) if sources else "live session"
    has_benchmark = any(
        row.get("benchmark_compatible_live_run")
        or str(row.get("evaluation_method") or "").startswith("benchmark_compatible")
        or row.get("evaluation_method") == "blind_mistral_judge_with_gold_constraints"
        for row in rows
    )
    has_demo = any(row.get("evaluation_method") == "demo_metrics" for row in rows)

    if mode == "official":
        return {
            "evaluation_mode": "official",
            "evaluation_mode_label": "Official Results Mode",
            "evaluation_method": method_text,
            "evaluation_source": "evaluator/results/results.json",
            "evaluation_meaning": "exact paper results",
            "evaluation_message": "Mode: Official Results Mode | Source: evaluator/results/results.json | Meaning: exact paper results",
            "evaluation_warning": None,
            "question_source": "evaluator/results/results.json",
        }
    if not rows:
        return {
            "evaluation_mode": "none",
            "evaluation_mode_label": "No evaluated responses",
            "evaluation_method": "N/A",
            "evaluation_source": source_text,
            "evaluation_meaning": "no session results yet",
            "evaluation_message": "No session results yet.",
            "evaluation_warning": None,
            "question_source": source_text,
        }
    if has_benchmark and not has_demo:
        return {
            "evaluation_mode": "benchmark_live",
            "evaluation_mode_label": "Benchmark-Compatible Live Evaluation",
            "evaluation_method": method_text,
            "evaluation_source": "live generated answers",
            "evaluation_meaning": "close to evaluator.py but may differ due to regenerated LLM outputs",
            "evaluation_message": (
                "Mode: Benchmark-Compatible Live Evaluation | Source: live generated answers | "
                "Meaning: close to evaluator.py but may differ due to regenerated LLM outputs"
            ),
            "evaluation_warning": "Website live evaluation uses isolated session reset but does not delete permanent project memory.",
            "question_source": source_text,
        }
    if has_benchmark and has_demo:
        return {
            "evaluation_mode": "mixed",
            "evaluation_mode_label": "Mixed evaluation modes",
            "evaluation_method": method_text,
            "evaluation_source": source_text,
            "evaluation_meaning": "mixed live session modes",
            "evaluation_message": "Session contains both benchmark-compatible and demo metrics.",
            "evaluation_warning": "Avoid mixing modes for paper-facing comparisons.",
            "question_source": source_text,
        }
    return {
        "evaluation_mode": "demo",
        "evaluation_mode_label": "Demo Metrics",
        "evaluation_method": method_text,
        "evaluation_source": "live session",
        "evaluation_meaning": "not for paper",
        "evaluation_message": "Mode: Demo Metrics | Source: live session | Meaning: not for paper",
        "evaluation_warning": "These metrics are approximate and should not be used in paper.",
        "question_source": source_text,
    }


def compute_live_summary(history: list[dict[str, Any]], *, mode: str | None = None) -> dict[str, Any]:
    rows = collect_rows(history)
    systems = {
        system: _system_metrics([row for row in rows if row.get("system") == system], system)
        for system in SYSTEM_KEYS
    }
    summary_cards = {
        "total_questions": len(history),
        "total_responses": sum(metrics["count"] for metrics in systems.values()),
        "main_rag_used_count": sum(1 for entry in history if any(
            bool(_nested_value(answer, "main_rag_used"))
            for answer in (entry.get("answers") or entry.get("results") or {}).values()
            if isinstance(answer, dict)
        )),
        "temporary_rag_used_count": sum(1 for entry in history if any(
            bool(_nested_value(answer, "temporary_rag_used"))
            for answer in (entry.get("answers") or entry.get("results") or {}).values()
            if isinstance(answer, dict)
        )),
        "best_accuracy_system": _winner(systems, ("avg_accuracy",), "max"),
        "fastest_system": _winner(systems, ("avg_latency",), "min"),
        "lowest_contamination_system": _winner(systems, ("contamination_count",), "min"),
        "best_cross_domain_robustness_system": _winner(systems, ("special_metrics", "cross_domain_robustness"), "max"),
    }
    mode_data = _mode_summary(history, rows, mode=mode)
    export_metadata = {
        "evaluation_mode": mode_data["evaluation_mode_label"],
        "question_source": mode_data.get("question_source"),
        "question_count": len(history),
        "evaluator_method": mode_data.get("evaluation_method"),
        "exact_evaluator_imported": any(row.get("exact_evaluator_imported") for row in rows),
        "fallback_evaluator_used": any(row.get("fallback_evaluator_used") for row in rows),
        "main_rag_used": summary_cards["main_rag_used_count"] > 0,
        "state_reset_applied": any(row.get("state_reset_applied") for row in rows),
    }
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **mode_data,
        "total_questions": len(history),
        "total_system_responses": summary_cards["total_responses"],
        "systems": systems,
        "system_order": list(SYSTEM_KEYS),
        "system_labels": SYSTEM_LABELS,
        "system_colors": SYSTEM_COLORS,
        "dimensions": list(DIMENSIONS),
        "special_metrics": list(SPECIAL_METRICS),
        "categories": available_categories(history),
        "all_categories": [{"key": category, "label": display_label(category)} for category in CATEGORIES],
        "summary_cards": summary_cards,
        "comparison_table": _comparison_rows(systems),
        "questions": question_summaries(history),
        "export_metadata": export_metadata,
    }


def compute_official_results_summary(path: Path | None = None) -> dict[str, Any]:
    entries = load_official_results(path)
    history = official_entries_to_history(entries)
    summary = compute_live_summary(history, mode="official")
    summary["official_results_count"] = len(entries)
    summary["summarize_results_like"] = summarize_results_like(entries)
    return summary
