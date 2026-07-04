from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

from .domain_router import (
    contamination_terms_for_route,
    needs_soft_repair,
    normalize_domain_label as router_normalize_domain_label,
    route_query,
)
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

AB_ROUTING_NOTE = (
    "Routing note: Answer the current question only. "
    "Avoid unrelated previous-topic context."
)
SYSTEM_C_STRICT_ROUTING_NOTE = """Routing note:
The user is switching topics or explicitly requested no contamination.
Answer only the current target domain: {target_domain}.
Do not mention previous-domain concepts unless the user explicitly asks for comparison.
Do not force relationships between domains.
If the target domain is daily, avoid robotics terms.
If the target domain is robotics, avoid consumer app examples.
Keep the answer direct and beginner-friendly."""
SYSTEM_C_MIXED_ROUTING_NOTE = """Routing note:
This is a mixed-domain question.
Explain the relationship carefully.
Do not claim a direct relationship unless justified.
Separate robotics and daily-life perspectives."""
SYSTEM_C_AMBIGUOUS_ROUTING_NOTE = """Routing note:
This question is ambiguous.
Give likely interpretations briefly and ask one concise clarification question."""
SOFT_REPAIR_INSTRUCTION = """Your previous answer included unrelated context from another domain.
Rewrite it so it answers only the user's current question and target domain.
Do not add unrelated examples.
Keep useful content and remove contamination."""

STRICT_DOMAIN_ISOLATION_INSTRUCTION = """STRICT DOMAIN ISOLATION MODE:
Answer only the user's current requested domain.
Do not mention previous topics.
Do not create Robotics Perspective / Daily-Life Perspective unless the user explicitly asks for comparison.
Do not force a relationship between robotics and daily-life.
If the user asks about a daily-life topic, do not mention SLAM, ROS, LiDAR, robots, sensors, mapping, localization, or navigation algorithms.
If the user asks about a robotics topic, do not mention ride-sharing apps, food delivery apps, shopping apps, social media, phone storage, or consumer services.
Give a direct answer first.
Keep the answer concise and relevant."""

MIXED_DOMAIN_FORMAT_INSTRUCTION = """MIXED DOMAIN ANSWER FORMAT:
Use exactly these sections:
- Direct Answer
- Relationship Type: Direct / Indirect / Analogy Only / Unsupported
- Robotics Perspective
- Daily-Life Perspective
- Important Difference
- Final Conclusion
Do not force a relationship. State unsupported relationships clearly."""

STRICT_ISOLATION_PHRASES = (
    "without contamination",
    "without mixing",
    "do not mix",
    "don't mix",
    "avoid mixing",
    "ignore previous",
    "ignore earlier",
    "now explain",
    "answer only",
    "only daily",
    "only robotics",
    "do not bring",
    "do not mention robotics",
    "do not mention daily",
    "do not mention slam",
    "do not mention lidar",
    "separate from previous topic",
)
NO_MIX_PHRASES = (
    "without contamination",
    "without mixing",
    "do not mix",
    "don't mix",
    "avoid mixing",
    "do not bring",
)
MIXED_REQUEST_PHRASES = (
    "compare robotics and daily",
    "compare robotics with daily",
    "compare daily-life and robotics",
    "relationship between",
    "related to robot",
    "related to robotics",
    "related to slam",
    "replace slam",
    "uber gps replace slam",
    "daily app related to robot navigation",
    "daily apps related to robot navigation",
)
DAILY_DOMAIN_TERMS = (
    "food delivery",
    "delivery app",
    "delivery apps",
    "ride-sharing",
    "ride sharing",
    "uber",
    "ola",
    "lyft",
    "zomato",
    "swiggy",
    "phone storage",
    "cloud storage",
    "shopping",
    "shopping app",
    "apps",
    "app",
    "safety",
    "privacy",
    "online scam",
    "online scams",
    "navigation apps",
    "daily use",
    "social media",
    "consumer service",
    "consumer services",
)
ROBOTICS_DOMAIN_TERMS = (
    "slam",
    "ros",
    "ros2",
    "lidar",
    "odometry",
    "ekf",
    "pid",
    "robot",
    "robots",
    "robotics",
    "sensor fusion",
    "navigation in robot",
    "robot navigation",
    "mapping",
    "localization",
    "localisation",
    "occupancy grid",
    "path planning",
)
DAILY_CONTAMINATION_TERMS = (
    "slam",
    "ros",
    "lidar",
    "odometry",
    "ekf",
    "pid",
    "robot",
    "robots",
    "robotics",
    "robot localization",
    "sensor fusion",
    "occupancy grid",
    "path planning",
    "mapping algorithm",
    "navigation algorithm",
    "navigation algorithms",
)
ROBOTICS_CONTAMINATION_TERMS = (
    "ride-sharing",
    "ride sharing",
    "uber",
    "ola",
    "lyft",
    "food delivery",
    "delivery app",
    "delivery apps",
    "shopping app",
    "shopping apps",
    "phone storage",
    "social media",
    "online scam",
    "online scams",
    "consumer app",
    "consumer apps",
    "consumer service",
    "consumer services",
    "zomato",
    "swiggy",
)


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


def _contains_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in phrases)


def _contains_domain_term(text: str, terms: tuple[str, ...]) -> bool:
    lowered = text.lower()
    for term in terms:
        pattern = rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])"
        if re.search(pattern, lowered):
            return True
    return False


def _matching_terms(text: str, terms: tuple[str, ...]) -> list[str]:
    lowered = text.lower()
    matches = []
    for term in terms:
        pattern = rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])"
        if re.search(pattern, lowered):
            matches.append(term)
    return matches


def _normalize_domain_label(value: Any) -> str | None:
    return router_normalize_domain_label(value)


def _focus_current_request(question: str) -> str:
    text = question.lower()
    focus = text
    for marker in (
        "now explain",
        "now describe",
        "now answer",
        "now tell",
        "answer only",
        "only daily",
        "only robotics",
        "answer a beginner question about",
        "answer beginner question about",
        "explain",
    ):
        index = text.rfind(marker)
        if index >= 0:
            focus = text[index:]
            break

    for boundary in (
        " without contamination",
        " without mixing",
        " do not mix",
        " don't mix",
        " avoid mixing",
        " do not bring",
        " do not mention",
    ):
        boundary_index = focus.find(boundary)
        if boundary_index > 0:
            focus = focus[:boundary_index]
            break
    return focus


def _is_explicit_mixed_request(question: str) -> bool:
    text = question.lower()
    if _contains_phrase(text, NO_MIX_PHRASES):
        return False
    if _contains_phrase(text, MIXED_REQUEST_PHRASES):
        return True
    has_daily = _contains_domain_term(text, DAILY_DOMAIN_TERMS)
    has_robotics = _contains_domain_term(text, ROBOTICS_DOMAIN_TERMS)
    asks_relationship = _contains_phrase(text, ("compare", "relationship", "related", "replace", "analogy"))
    return bool(has_daily and has_robotics and asks_relationship)


def _infer_target_domain(question: str) -> str:
    text = question.lower()
    focus = _focus_current_request(question)

    if _is_explicit_mixed_request(question):
        return "mixed"
    if "do not mention robotics" in text or "do not mention slam" in text or "do not mention lidar" in text:
        return "daily"
    if "only daily" in text:
        return "daily"
    if "do not mention daily" in text or "only robotics" in text:
        return "robotics"

    focus_has_daily = _contains_domain_term(focus, DAILY_DOMAIN_TERMS)
    focus_has_robotics = _contains_domain_term(focus, ROBOTICS_DOMAIN_TERMS)
    if focus_has_daily and not focus_has_robotics:
        return "daily"
    if focus_has_robotics and not focus_has_daily:
        return "robotics"

    full_has_daily = _contains_domain_term(text, DAILY_DOMAIN_TERMS)
    full_has_robotics = _contains_domain_term(text, ROBOTICS_DOMAIN_TERMS)
    if full_has_daily and full_has_robotics:
        return "mixed" if not _contains_phrase(text, NO_MIX_PHRASES) else "unknown"
    if full_has_daily:
        return "daily"
    if full_has_robotics:
        return "robotics"
    return "unknown"


def _strict_domain_isolation_requested(question: str, context_info: dict[str, Any]) -> bool:
    text = question.lower()
    if _contains_phrase(text, STRICT_ISOLATION_PHRASES):
        return True

    target_domain = _infer_target_domain(question)
    rag_category = _normalize_domain_label(context_info.get("main_rag_relevance_category"))
    if target_domain == "daily" and rag_category == "robotics" and context_info.get("main_rag_used"):
        return True
    if target_domain == "robotics" and rag_category == "daily" and context_info.get("main_rag_used"):
        return True
    return False


def _has_explicit_robotics_term(question: str) -> bool:
    return _contains_domain_term(question, ROBOTICS_DOMAIN_TERMS)


def _filter_system_c_context(
    question: str,
    context_info: dict[str, Any],
    *,
    strict_domain_isolation: bool,
    target_domain: str,
) -> tuple[dict[str, Any], bool]:
    filtered = dict(context_info)
    if not strict_domain_isolation:
        return filtered, False

    changed = False
    main_context = str(filtered.get("main_project_rag_context") or "")
    temp_context = str(filtered.get("temporary_upload_context") or "")
    rag_category = _normalize_domain_label(filtered.get("main_rag_relevance_category"))
    rag_reason = str(filtered.get("main_rag_relevance_reason") or "")

    def suppress_main(reason: str) -> None:
        nonlocal changed, main_context
        if filtered.get("main_rag_used") or main_context:
            changed = True
        main_context = ""
        filtered["main_project_rag_context"] = ""
        filtered["main_rag_used"] = False
        filtered["main_rag_max_chars"] = 0
        filtered["main_rag_suppressed_for_system_c"] = True
        filtered["main_rag_suppression_reason"] = reason

    if target_domain == "daily":
        if not (rag_category == "daily" or "daily" in rag_reason):
            suppress_main("strict_daily_domain_isolation")
    elif target_domain == "robotics":
        if rag_category == "daily":
            suppress_main("strict_robotics_domain_isolation")
    elif target_domain == "mixed":
        if main_context and len(main_context) > 500:
            main_context = main_context[:500]
            filtered["main_project_rag_context"] = main_context
            filtered["main_rag_max_chars"] = 500
            changed = True
    elif target_domain in {"ambiguous", "unknown"} and not _has_explicit_robotics_term(question):
        suppress_main("ambiguous_without_explicit_robotics_term")

    combined = "\n\n".join(part for part in (temp_context, main_context) if part)
    if target_domain == "mixed" and len(combined) > 500:
        combined = combined[:500]
        changed = True
    filtered["combined_context"] = combined
    filtered["combined_context_chars"] = len(combined)
    return filtered, changed


def _system_c_guard_plan(question: str, context_info: dict[str, Any]) -> dict[str, Any]:
    route_info = dict(context_info.get("route_info") or route_query(question))
    target_domain = str(route_info.get("target_domain") or "unknown")
    resolved_domain = str(route_info.get("resolved_domain") or "unknown")
    strict_domain_isolation = bool(route_info.get("strict_isolation"))
    mixed_format_required = (
        resolved_domain == "mixed"
        and target_domain == "mixed"
        and not strict_domain_isolation
    )
    filtered_context, rag_filter_applied = _filter_system_c_context(
        question,
        context_info,
        strict_domain_isolation=strict_domain_isolation,
        target_domain=target_domain,
    )
    return {
        "target_domain": target_domain,
        "resolved_domain": resolved_domain,
        "route_info": route_info,
        "strict_domain_isolation": strict_domain_isolation,
        "mixed_format_required": mixed_format_required,
        "rag_filter_applied": rag_filter_applied,
        "context_info": filtered_context,
    }


def _focused_current_question(question: str) -> str:
    focus = _focus_current_request(question).strip()
    replacements = (
        (r"^now\s+", ""),
        (r"^answer a beginner question about\s+", "Explain "),
        (r"^answer beginner question about\s+", "Explain "),
        (r"^answer only\s+", ""),
    )
    for pattern, replacement in replacements:
        focus = re.sub(pattern, replacement, focus, flags=re.IGNORECASE).strip()
    if not focus:
        return question
    if not focus.endswith(("?", ".", "!")):
        focus = f"{focus}."
    return focus[0].upper() + focus[1:]


def _system_c_strict_note(target_domain: str) -> str:
    if target_domain == "daily":
        return """Routing note:
The user switched topics and requested a clean daily-life answer.
Answer only the current daily-life topic.
Do not compare domains.
Do not discuss the earlier topic.
Do not use perspective sections.
Keep the answer practical, direct, and beginner-friendly."""
    if target_domain == "robotics":
        return """Routing note:
The user switched topics and requested a clean robotics answer.
Answer only the current robotics topic.
Do not compare domains.
Do not discuss the earlier topic.
Keep the answer useful, direct, and beginner-friendly."""
    return SYSTEM_C_STRICT_ROUTING_NOTE.format(target_domain=target_domain or "unknown")


def _system_c_question_for_call(question: str, guard: dict[str, Any]) -> str:
    if guard.get("strict_domain_isolation"):
        focused_question = _focused_current_question(question)
        return (
            f"{_system_c_strict_note(str(guard.get('target_domain') or 'unknown'))}\n\n"
            f"Current question to answer:\n{focused_question}"
        )
    if guard.get("mixed_format_required"):
        return (
            f"{SYSTEM_C_MIXED_ROUTING_NOTE}\n\n"
            f"Original question:\n{question}"
        )
    if guard.get("resolved_domain") == "ambiguous":
        return (
            f"{SYSTEM_C_AMBIGUOUS_ROUTING_NOTE}\n\n"
            f"Original question:\n{question}"
        )
    return question


def _system_c_contamination_terms(response: str, target_domain: str) -> list[str]:
    return contamination_terms_for_route(response, {"target_domain": target_domain})


def _system_c_repair_prompt(original_question: str, bad_answer: str, target_domain: str) -> str:
    focused_question = _focused_current_question(original_question)
    return (
        f"{SOFT_REPAIR_INSTRUCTION}\n"
        f"Target domain: {target_domain}\n"
        f"Current question to answer: {focused_question}"
    )


def _system_output(raw: Any) -> tuple[str, dict[str, Any], Any]:
    if isinstance(raw, dict):
        response = str(raw.get("response") or raw.get("answer") or raw)
        metadata = dict(raw)
        return response, metadata, metadata.get("latency")
    return str(raw), {}, None


def _ab_question_for_call(question: str, route_info: dict[str, Any]) -> str:
    if not route_info:
        return question
    return f"{AB_ROUTING_NOTE}\n\nOriginal question:\n{question}"


def _known_domain(value: Any) -> str | None:
    label = router_normalize_domain_label(value)
    if label in {"robotics", "daily", "mixed", "ambiguous", "unverifiable"}:
        return label
    return None


def _merge_route_metadata(metadata: dict[str, Any], route_info: dict[str, Any]) -> None:
    route_resolved = _known_domain(route_info.get("resolved_domain"))
    route_target = _known_domain(route_info.get("target_domain"))
    route_confidence = float(route_info.get("confidence") or 0.0)

    existing_intent = _known_domain(
        metadata.get("corrected_predicted_intent")
        or metadata.get("predicted_intent")
        or metadata.get("classified_intent")
        or metadata.get("intent")
    )
    existing_domain = _known_domain(metadata.get("resolved_domain") or metadata.get("domain"))

    metadata["route_info"] = dict(route_info)
    metadata["route_resolved_domain"] = route_info.get("resolved_domain")
    metadata["route_target_domain"] = route_info.get("target_domain")
    metadata["route_confidence"] = route_confidence
    metadata["strict_isolation"] = bool(route_info.get("strict_isolation"))
    metadata["domain_switch"] = bool(route_info.get("domain_switch"))
    metadata["intent_source"] = "system_metadata+website_router"

    if existing_intent:
        metadata["predicted_intent"] = existing_intent
    elif route_resolved:
        metadata["predicted_intent"] = route_resolved

    if existing_domain:
        metadata["resolved_domain"] = existing_domain
    elif route_target:
        metadata["resolved_domain"] = route_target
    elif route_resolved:
        metadata["resolved_domain"] = route_resolved

    expected_domain = route_target or route_resolved
    if expected_domain:
        metadata.setdefault("expected_intent", expected_domain)
        metadata.setdefault("expected_domain", expected_domain)
        predicted = _known_domain(metadata.get("predicted_intent"))
        resolved = _known_domain(metadata.get("resolved_domain"))
        metadata["intent_classification_accuracy"] = (
            1.0 if predicted == expected_domain else 0.0
        ) if predicted else None
        metadata["domain_resolution_accuracy"] = (
            1.0 if resolved == expected_domain else 0.0
        ) if resolved else None


def _empty_context_info() -> dict[str, Any]:
    return {
        "main_project_rag_context": "",
        "temporary_upload_context": "",
        "combined_context": "",
        "main_rag_used": False,
        "temporary_rag_used": False,
        "main_rag_candidate_count": 0,
        "main_rag_relevance_score": 0.0,
        "main_rag_relevance_reason": "not_checked",
        "main_rag_relevance_gate_enabled": True,
        "main_rag_max_chars": 0,
        "combined_context_chars": 0,
    }


def _rag_metadata(context_info: dict[str, Any]) -> dict[str, Any]:
    combined_context = str(context_info.get("combined_context") or "")
    route_info = dict(context_info.get("route_info") or {})
    return {
        "main_rag_used": bool(context_info.get("main_rag_used")),
        "temporary_rag_used": bool(context_info.get("temporary_rag_used")),
        "temporary_files_count": session_store.temporary_files_count(),
        "combined_context_chars": int(context_info.get("combined_context_chars") or len(combined_context)),
        "main_rag_candidate_count": int(context_info.get("main_rag_candidate_count") or 0),
        "main_rag_relevance_score": float(context_info.get("main_rag_relevance_score") or 0.0),
        "main_rag_relevance_reason": context_info.get("main_rag_relevance_reason"),
        "main_rag_relevance_category": context_info.get("main_rag_relevance_category"),
        "main_rag_relevance_gate_enabled": bool(context_info.get("main_rag_relevance_gate_enabled", True)),
        "main_rag_max_chars": int(context_info.get("main_rag_max_chars") or 0),
        "route_info": route_info,
        "route_resolved_domain": route_info.get("resolved_domain"),
        "route_target_domain": route_info.get("target_domain"),
        "route_confidence": route_info.get("confidence"),
        "strict_isolation": bool(route_info.get("strict_isolation")),
        "domain_switch": bool(route_info.get("domain_switch")),
    }


def _build_question(question: str, context_info: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    combined_context = str(context_info.get("combined_context") or "").strip()
    metadata = _rag_metadata(context_info)
    has_relevant_context = bool(
        combined_context
        and (
            context_info.get("main_rag_used")
            or context_info.get("temporary_rag_used")
        )
    )
    if not has_relevant_context:
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
    system_key = system_key.upper()
    route_info = dict(context_info.get("route_info") or route_query(question))
    system_c_guard: dict[str, Any] | None = None
    if system_key == "C":
        system_c_guard = _system_c_guard_plan(question, context_info)
        context_info = system_c_guard["context_info"]
    rag_metadata = _rag_metadata(context_info)
    function = _load_system(system_key)
    if function is None:
        return {
            "response": _IMPORT_ERRORS.get(system_key, "System unavailable."),
            "latency": 0.0,
            "metadata": {
                "error": True,
                "message": _IMPORT_ERRORS.get(system_key, "System unavailable."),
                **rag_metadata,
            },
        }

    system_question = _system_c_question_for_call(question, system_c_guard) if system_c_guard else _ab_question_for_call(question, route_info)
    effective_question, rag_metadata = _build_question(system_question, context_info)
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

    response, metadata, returned_latency = _system_output(raw)
    _merge_route_metadata(metadata, route_info)

    if system_c_guard:
        target_domain = str(system_c_guard.get("target_domain") or "unknown")
        route_info = dict(system_c_guard.get("route_info") or route_info)
        metadata.update({
            "strict_domain_isolation": bool(system_c_guard.get("strict_domain_isolation")),
            "strict_isolation": bool(system_c_guard.get("strict_domain_isolation")),
            "target_domain": target_domain,
            "system_c_mixed_format_required": bool(system_c_guard.get("mixed_format_required")),
            "system_c_rag_filter_applied": bool(system_c_guard.get("rag_filter_applied")),
            "system_c_repair_applied": False,
            "soft_repair_applied": False,
            "contamination_terms_removed": [],
            "system_c_augmented_prompt_used": bool(
                system_c_guard.get("strict_domain_isolation")
                or system_c_guard.get("mixed_format_required")
            ),
        })

        contamination_terms = contamination_terms_for_route(response, route_info)
        metadata["system_c_postcheck_contamination_terms"] = contamination_terms
        metadata["contamination_self_check_terms"] = contamination_terms
        if needs_soft_repair(response, route_info):
            repair_prompt = _system_c_repair_prompt(question, response, target_domain)
            try:
                repair_raw = _call_system(function, repair_prompt)
                repaired_response, _repair_metadata, _repair_latency = _system_output(repair_raw)
                repaired_terms = contamination_terms_for_route(repaired_response, route_info)
                response = repaired_response
                metadata["system_c_repair_applied"] = True
                metadata["soft_repair_applied"] = True
                metadata["system_c_postrepair_contamination_terms"] = repaired_terms
                metadata["contamination_terms_removed"] = [
                    term for term in contamination_terms if term not in repaired_terms
                ]
            except Exception as exc:
                metadata["system_c_repair_error"] = f"{type(exc).__name__}: {exc}"

    latency = time.perf_counter() - started
    latency_value = returned_latency if isinstance(returned_latency, (int, float)) else latency

    metadata.update(rag_metadata)
    _merge_route_metadata(metadata, route_info)
    evaluation = evaluate_response(
        question,
        response,
        metadata=metadata,
    )
    metadata.update(evaluation)
    _merge_route_metadata(metadata, route_info)

    result = {
        "response": response,
        "latency": round(float(latency_value), 3),
        "metadata": metadata,
    }
    if evaluation.get("metrics_evaluated"):
        for key, value in evaluation.items():
            if key not in {"evaluation_method", "fake_terms_found"}:
                result[key] = value
    if system_c_guard:
        for key in (
            "predicted_intent",
            "resolved_domain",
            "route_resolved_domain",
            "route_target_domain",
            "route_confidence",
            "strict_domain_isolation",
            "strict_isolation",
            "domain_switch",
            "target_domain",
            "expected_intent",
            "expected_domain",
            "intent_source",
            "intent_classification_accuracy",
            "domain_resolution_accuracy",
            "system_c_repair_applied",
            "soft_repair_applied",
            "contamination_terms_removed",
            "system_c_postcheck_contamination_terms",
            "contamination_self_check_terms",
            "system_c_postrepair_contamination_terms",
            "system_c_mixed_format_required",
            "system_c_rag_filter_applied",
        ):
            if key in metadata:
                result[key] = metadata[key]
    else:
        for key in (
            "predicted_intent",
            "resolved_domain",
            "route_resolved_domain",
            "route_target_domain",
            "route_confidence",
            "strict_isolation",
            "domain_switch",
            "intent_source",
            "intent_classification_accuracy",
            "domain_resolution_accuracy",
        ):
            if key in metadata:
                result[key] = metadata[key]
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
    previous_context = "\n".join(
        str(entry.get("question") or "")
        for entry in session_store.get_history()[-3:]
    )
    route_info = route_query(question, previous_context=previous_context)
    rag_question = _focused_current_question(question) if route_info.get("strict_isolation") else question
    context_info = get_combined_context(rag_question, route_info=route_info)
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


def _lightweight_system_status(system_key: str) -> dict[str, Any]:
    configure_project_runtime()
    system_key = system_key.upper()
    if system_key in _SYSTEM_CACHE:
        return {
            "available": True,
            "loaded": True,
            "error": None,
        }
    if system_key not in SYSTEM_MODULES:
        return {
            "available": False,
            "loaded": False,
            "error": f"Unknown system: {system_key}",
        }

    module_name, function_name = SYSTEM_MODULES[system_key]
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        return {
            "available": False,
            "loaded": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    if spec is None:
        return {
            "available": False,
            "loaded": False,
            "error": f"Module {module_name} not found.",
        }

    return {
        "available": True,
        "loaded": False,
        "error": None,
        "module": module_name,
        "function": function_name,
        "note": "Lightweight health check did not import the model-heavy system module.",
    }


def health_status(*, probe_imports: bool = False) -> dict[str, Any]:
    configure_project_runtime()
    statuses = {
        key: system_status(key) if probe_imports else _lightweight_system_status(key)
        for key in ["A", "B", "C"]
    }
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
        "system_health_probe_imports": probe_imports,
        "system_health_note": (
            "System availability is checked without importing model-heavy modules. "
            "A chat request still imports the selected system on demand."
            if not probe_imports
            else "System modules were imported during health check."
        ),
    }
