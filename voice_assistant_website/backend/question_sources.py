from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

from .pdf_auto_questions import extract_questions_from_pdf_result


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_RESULTS_PATH = PROJECT_ROOT / "evaluator" / "results" / "results.json"
BENCHMARK_JSON_CANDIDATES = (
    PROJECT_ROOT / "benchmarks" / "benchmark_500.json",
    PROJECT_ROOT / "benchmark_500.json",
    PROJECT_ROOT / "evaluator" / "benchmark_500.json",
)
BENCHMARK_PDF_CANDIDATES = (
    PROJECT_ROOT / "benchmarks" / "benchmark_500.pdf",
    PROJECT_ROOT / "benchmark_500.pdf",
)


def configure_project_runtime() -> None:
    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def _category_from_results(results: dict[str, Any]) -> str | None:
    for result in results.values():
        if isinstance(result, dict) and result.get("question_type"):
            return str(result["question_type"])
    return None


def _clean_question_entry(entry: Any, index: int, *, source: str) -> dict[str, Any] | None:
    if isinstance(entry, str):
        question = entry.strip()
        category = None
        number = index
    elif isinstance(entry, dict):
        question = str(entry.get("question") or entry.get("query") or entry.get("prompt") or "").strip()
        category = entry.get("category") or entry.get("question_type") or entry.get("type")
        number = entry.get("id") or entry.get("number") or index
    else:
        return None

    if not question:
        return None

    try:
        number_value = int(number)
    except (TypeError, ValueError):
        number_value = index

    category_text = str(category).strip() if category else "unknown"
    return {
        "number": number_value,
        "category": category_text,
        "question": question,
        "raw": f"{number_value}. [{category_text}] {question}",
        "source": source,
    }


def _load_json_questions(path: Path, *, source: str) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected {path} to contain a list.")
    questions = []
    for index, entry in enumerate(data, start=1):
        cleaned = _clean_question_entry(entry, index, source=source)
        if cleaned:
            questions.append(cleaned)
    return questions


def load_official_results_questions() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with OFFICIAL_RESULTS_PATH.open(encoding="utf-8") as handle:
        entries = json.load(handle)
    if not isinstance(entries, list):
        raise ValueError("Expected evaluator/results/results.json to contain a list.")

    questions: list[dict[str, Any]] = []
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            continue
        question = str(entry.get("question") or "").strip()
        if not question:
            continue
        results = entry.get("results") if isinstance(entry.get("results"), dict) else {}
        category = _category_from_results(results) or "unknown"
        questions.append({
            "number": index,
            "category": category,
            "question": question,
            "raw": f"{index}. [{category}] {question}",
            "source": "official_results_json",
            "official_results_index": index,
        })
    return questions, {
        "question_source": "official_results_json",
        "source_path": str(OFFICIAL_RESULTS_PATH),
        "extraction_mode": "official_results_json",
        "loaded_message": f"Loaded {len(questions)} official evaluated questions from results.json",
        "category_counts": _category_counts(questions),
    }


def _load_evaluator_auto_questions() -> list[dict[str, Any]]:
    configure_project_runtime()
    module = importlib.import_module("evaluator.evaluator")
    auto_questions = getattr(module, "AUTO_QUESTIONS")
    if not isinstance(auto_questions, list):
        return []
    questions = []
    for index, entry in enumerate(auto_questions, start=1):
        cleaned = _clean_question_entry(entry, index, source="evaluator_auto_questions")
        if cleaned:
            questions.append(cleaned)
    return questions


def _category_counts(questions: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for question in questions:
        category = str(question.get("category") or "unknown")
        counts[category] = counts.get(category, 0) + 1
    return counts


def load_official_benchmark_questions() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    for path in BENCHMARK_JSON_CANDIDATES:
        if path.exists():
            questions = _load_json_questions(path, source="official_benchmark_json")
            return questions, {
                "question_source": "official_benchmark_file",
                "source_path": str(path),
                "extraction_mode": "official_benchmark_json",
                "loaded_message": f"Loaded {len(questions)} questions from {path.name}",
                "category_counts": _category_counts(questions),
            }

    for path in BENCHMARK_PDF_CANDIDATES:
        if path.exists():
            result = extract_questions_from_pdf_result(path)
            questions = result["questions"]
            return questions, {
                "question_source": "official_benchmark_pdf",
                "source_path": str(path),
                "extraction_mode": "official_benchmark_pdf",
                "loaded_message": f"Loaded {len(questions)} questions from {path.name}",
                "category_counts": _category_counts(questions),
                **{key: value for key, value in result.items() if key != "questions"},
            }

    questions = _load_evaluator_auto_questions()
    return questions, {
        "question_source": "evaluator_auto_questions",
        "source_path": "evaluator.evaluator:AUTO_QUESTIONS",
        "extraction_mode": "evaluator_auto_questions",
        "loaded_message": f"Loaded {len(questions)} questions from evaluator AUTO_QUESTIONS",
        "category_counts": _category_counts(questions),
    }
