from __future__ import annotations

import importlib
import inspect
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

from .rag_temp import get_combined_context
from .response_evaluator import evaluate_response
from .session_store import session_store


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTENT_CLASSIFIER_PATH = PROJECT_ROOT / "models" / "intent_classifier"


def configure_project_runtime() -> None:
    """Run existing chatbot systems from the parent project root."""

    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    if Path.cwd() != PROJECT_ROOT:
        os.chdir(PROJECT_ROOT)


configure_project_runtime()
print(f"[voice_assistant_website] PROJECT_ROOT: {PROJECT_ROOT}")
print(f"[voice_assistant_website] cwd: {Path.cwd()}")
print(
    "[voice_assistant_website] models/intent_classifier exists: "
    f"{INTENT_CLASSIFIER_PATH.exists()}"
)


SYSTEM_MODULES = {
    "A": ("chatbot_system_a", "chatbot_system_a"),
    "B": ("chatbot_system_b", "chatbot_system_b"),
    "C": ("chatbot_system_c", "chatbot_system_c"),
}

_SYSTEM_CACHE: dict[str, Callable[..., Any]] = {}
_IMPORT_ERRORS: dict[str, str] = {}


def _load_system(system_key: str) -> Callable[..., Any] | None:
    configure_project_runtime()
    system_key = system_key.upper()
    if system_key in _SYSTEM_CACHE:
        return _SYSTEM_CACHE[system_key]
    if system_key not in SYSTEM_MODULES:
        _IMPORT_ERRORS[system_key] = f"Unknown system: {system_key}"
        return None

    module_name, function_name = SYSTEM_MODULES[system_key]
    try:
        module = importlib.import_module(module_name)
        function = getattr(module, function_name)
    except Exception as exc:
        _IMPORT_ERRORS[system_key] = (
            f"System {system_key} could not be imported: {type(exc).__name__}: {exc}"
        )
        return None

    _SYSTEM_CACHE[system_key] = function
    _IMPORT_ERRORS.pop(system_key, None)
    return function


def _empty_context_info() -> dict[str, Any]:
    return {
        "main_project_rag_context": "",
        "temporary_upload_context": "",
        "combined_context": "",
        "main_rag_used": False,
        "temporary_rag_used": False,
    }


def _rag_metadata(context_info: dict[str, Any]) -> dict[str, Any]:
    combined_context = str(context_info.get("combined_context") or "")
    return {
        "main_rag_used": bool(context_info.get("main_rag_used")),
        "temporary_rag_used": bool(context_info.get("temporary_rag_used")),
        "temporary_files_count": session_store.temporary_files_count(),
        "combined_context_chars": len(combined_context),
    }


def _build_question(question: str, context_info: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    combined_context = str(context_info.get("combined_context") or "").strip()
    metadata = _rag_metadata(context_info)
    if not combined_context:
        return question, metadata
    return (
        "Relevant context from project knowledge and temporary uploaded files:\n"
        f"{combined_context}\n\n"
        "User question:\n"
        f"{question}",
        metadata,
    )


def _call_system(function: Callable[..., Any], question: str) -> Any:
    configure_project_runtime()
    signature = inspect.signature(function)
    if "return_metadata" in signature.parameters:
        return function(question, return_metadata=True)
    return function(question)


def ask_system(
    system_key: str,
    question: str,
    *,
    temporary_context: str | None = None,
    context_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    configure_project_runtime()
    started = time.perf_counter()
    context_info = context_info or _empty_context_info()
    rag_metadata = _rag_metadata(context_info)
    function = _load_system(system_key)
    if function is None:
        return {
            "response": _IMPORT_ERRORS.get(system_key.upper(), "System unavailable."),
            "latency": 0.0,
            "metadata": {
                "error": True,
                "message": _IMPORT_ERRORS.get(system_key.upper(), "System unavailable."),
                **rag_metadata,
            },
        }

    effective_question, rag_metadata = _build_question(question, context_info)
    try:
        raw = _call_system(function, effective_question)
    except Exception as exc:
        latency = time.perf_counter() - started
        return {
            "response": f"System {system_key} failed while answering: {type(exc).__name__}: {exc}",
            "latency": round(latency, 3),
            "metadata": {
                "error": True,
                "message": str(exc),
                **rag_metadata,
            },
        }

    latency = time.perf_counter() - started
    if isinstance(raw, dict):
        response = str(raw.get("response") or raw.get("answer") or raw)
        metadata = dict(raw)
        returned_latency = metadata.get("latency")
        latency_value = returned_latency if isinstance(returned_latency, (int, float)) else latency
    else:
        response = str(raw)
        metadata = {}
        latency_value = latency

    metadata.update(rag_metadata)
    evaluation = evaluate_response(
        question,
        response,
        metadata=metadata,
    )
    metadata.update(evaluation)

    result = {
        "response": response,
        "latency": round(float(latency_value), 3),
        "metadata": metadata,
    }
    if evaluation.get("metrics_evaluated"):
        for key, value in evaluation.items():
            if key not in {"evaluation_method", "fake_terms_found"}:
                result[key] = value
    return {
        **result,
    }


def answer_question(
    question: str,
    system_choice: str,
    *,
    temporary_context: str | None = None,
) -> dict[str, Any]:
    configure_project_runtime()
    selected = system_choice.upper()
    system_keys = ["A", "B", "C"] if selected == "ALL" else [selected]
    context_info = get_combined_context(question)
    return {
        key: ask_system(key, question, temporary_context=temporary_context, context_info=context_info)
        for key in system_keys
    }


def system_status(system_key: str) -> dict[str, Any]:
    configure_project_runtime()
    function = _load_system(system_key)
    return {
        "available": function is not None,
        "error": _IMPORT_ERRORS.get(system_key.upper()),
    }


def health_status() -> dict[str, Any]:
    configure_project_runtime()
    statuses = {key: system_status(key) for key in ["A", "B", "C"]}
    return {
        "project_root": str(PROJECT_ROOT),
        "cwd": str(Path.cwd()),
        "intent_classifier_exists": INTENT_CLASSIFIER_PATH.exists(),
        "system_a_available": statuses["A"]["available"],
        "system_b_available": statuses["B"]["available"],
        "system_c_available": statuses["C"]["available"],
        "system_errors": {
            key: status["error"]
            for key, status in statuses.items()
            if status["error"]
        },
    }
