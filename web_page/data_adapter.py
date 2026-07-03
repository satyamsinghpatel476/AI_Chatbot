from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "evaluator" / "results" / "results.json"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "results_summary.json"

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


@dataclass(frozen=True)
class LoadResult:
    path: Path
    entries: list[dict[str, Any]]
    errors: list[str]
    warnings: list[str]
    loaded_at: str
    source_modified_at: str | None
    file_size_bytes: int | None


def _format_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


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


def load_results_file(path: Path | str = DEFAULT_RESULTS_PATH) -> LoadResult:
    result_path = Path(path)
    loaded_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    errors: list[str] = []
    warnings: list[str] = []
    source_modified_at: str | None = None
    file_size_bytes: int | None = None

    if not result_path.exists():
        return LoadResult(
            path=result_path,
            entries=[],
            errors=[f"Missing results file: {result_path}"],
            warnings=warnings,
            loaded_at=loaded_at,
            source_modified_at=source_modified_at,
            file_size_bytes=file_size_bytes,
        )

    try:
        stat = result_path.stat()
        source_modified_at = _format_timestamp(stat.st_mtime)
        file_size_bytes = stat.st_size
        raw_text = result_path.read_text(encoding="utf-8")
    except OSError as exc:
        return LoadResult(
            path=result_path,
            entries=[],
            errors=[f"Could not read {result_path}: {exc}"],
            warnings=warnings,
            loaded_at=loaded_at,
            source_modified_at=source_modified_at,
            file_size_bytes=file_size_bytes,
        )

    if not raw_text.strip():
        return LoadResult(
            path=result_path,
            entries=[],
            errors=[f"Empty JSON file: {result_path}"],
            warnings=warnings,
            loaded_at=loaded_at,
            source_modified_at=source_modified_at,
            file_size_bytes=file_size_bytes,
        )

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return LoadResult(
            path=result_path,
            entries=[],
            errors=[
                "Malformed JSON in "
                f"{result_path} at line {exc.lineno}, column {exc.colno}: "
                f"{exc.msg}"
            ],
            warnings=warnings,
            loaded_at=loaded_at,
            source_modified_at=source_modified_at,
            file_size_bytes=file_size_bytes,
        )

    if not isinstance(payload, list):
        return LoadResult(
            path=result_path,
            entries=[],
            errors=[
                "Unexpected results schema: expected a list of benchmark "
                f"entries in {result_path}"
            ],
            warnings=warnings,
            loaded_at=loaded_at,
            source_modified_at=source_modified_at,
            file_size_bytes=file_size_bytes,
        )

    entries: list[dict[str, Any]] = []
    skipped = 0
    for entry in payload:
        if isinstance(entry, dict):
            entries.append(entry)
        else:
            skipped += 1

    if skipped:
        warnings.append(f"Skipped {skipped} non-object benchmark entries.")

    return LoadResult(
        path=result_path,
        entries=entries,
        errors=errors,
        warnings=warnings,
        loaded_at=loaded_at,
        source_modified_at=source_modified_at,
        file_size_bytes=file_size_bytes,
    )


def validate_entries(entries: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    if not entries:
        errors.append("No benchmark entries found in results.json.")
        return errors

    system_counts = {system: 0 for system in SYSTEMS}
    missing_examples: list[str] = []

    for index, entry in enumerate(entries, start=1):
        results = entry.get("results")
        if not isinstance(results, dict):
            if len(missing_examples) < 5:
                missing_examples.append(f"entry {index}: missing results object")
            continue

        missing_systems: list[str] = []
        for system in SYSTEMS:
            if isinstance(results.get(system), dict):
                system_counts[system] += 1
            else:
                missing_systems.append(system)

        if missing_systems and len(missing_examples) < 5:
            missing_examples.append(
                f"entry {index}: missing System {', '.join(missing_systems)}"
            )

    for system, count in system_counts.items():
        if count == 0:
            errors.append(f"No System {system} results found in results.json.")

    if missing_examples:
        errors.append(
            "Some benchmark entries are missing required A, B, C results: "
            + "; ".join(missing_examples)
        )

    return errors


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
        "color": SYSTEM_COLORS[system],
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
                "question_type_label": display_label(get_question_type(entry)),
                "systems": {
                    system: summarize_result_for_question(
                        get_system_result(entry, system)
                    )
                    for system in SYSTEMS
                },
            }
        )
    return questions


def comparison_rows(system_metrics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[tuple[str, str, int, str | None]] = [
        ("Total questions evaluated", "total_questions_evaluated", 0, None),
        ("Average accuracy", "average_accuracy", 2, None),
        ("Median accuracy", "median_accuracy", 2, None),
        ("Average latency (s)", "average_latency", 2, None),
        ("Hallucination count", "hallucination_count", 0, None),
        ("Leakage count", "leakage_count", 0, None),
        ("Contamination count", "contamination_count", 0, None),
        ("False rejection count", "false_rejection_count", 0, None),
    ]

    for metric in SPECIAL_METRICS:
        rows.append((metric.replace("_", " ").title(), metric, 3, "special_metrics"))

    for dimension in DIMENSIONS:
        rows.append(
            (
                f"Dimension: {dimension.replace('_', ' ').title()}",
                dimension,
                2,
                "dimension_score_averages",
            )
        )

    output: list[dict[str, Any]] = []
    for label, key, places, group in rows:
        row: dict[str, Any] = {"metric": label, "places": places}
        for system in SYSTEMS:
            metrics = system_metrics[system]
            value = metrics.get(key) if group is None else metrics.get(group, {}).get(key)
            row[system] = value
        output.append(row)
    return output


def available_categories(entries: list[dict[str, Any]]) -> list[dict[str, str]]:
    found = set()
    for entry in entries:
        label = get_question_type(entry)
        if label:
            found.add(label)
    ordered = [category for category in CATEGORIES if category in found]
    extras = sorted(found - set(CATEGORIES))
    return [
        {"key": category, "label": display_label(category)}
        for category in ordered + extras
    ]


def build_dashboard_data(load_result: LoadResult) -> dict[str, Any]:
    system_metrics = {
        system: compute_system_metrics(load_result.entries, system)
        for system in SYSTEMS
    }

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": {
            "path": str(load_result.path),
            "modified_at": load_result.source_modified_at,
            "loaded_at": load_result.loaded_at,
            "file_size_bytes": load_result.file_size_bytes,
        },
        "systems": [
            {
                "key": system,
                "label": SYSTEM_LABELS[system],
                "color": SYSTEM_COLORS[system],
            }
            for system in SYSTEMS
        ],
        "categories": available_categories(load_result.entries),
        "dimensions": list(DIMENSIONS),
        "summary_cards": summarize_cards(load_result.entries, system_metrics),
        "system_metrics": system_metrics,
        "comparison_table": comparison_rows(system_metrics),
        "questions": question_summaries(load_result.entries),
        "validation": {
            "warnings": load_result.warnings,
            "errors": [],
        },
    }


def validate_dashboard_data(dashboard_data: dict[str, Any]) -> list[str]:
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


def export_summary(
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    results_path: Path | str = DEFAULT_RESULTS_PATH,
) -> tuple[Path, dict[str, Any]]:
    load_result = load_results_file(results_path)
    validation_errors = list(load_result.errors)
    if not validation_errors:
        validation_errors.extend(validate_entries(load_result.entries))
    if validation_errors:
        raise ValueError("\n".join(validation_errors))

    dashboard_data = build_dashboard_data(load_result)
    summary_errors = validate_dashboard_data(dashboard_data)
    if summary_errors:
        raise ValueError("\n".join(summary_errors))

    output = Path(output_path)
    output.write_text(
        json.dumps(dashboard_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output, dashboard_data
