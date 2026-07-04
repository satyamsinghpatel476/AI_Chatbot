from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from backend.domain_router import contamination_terms_for_route, route_query  # noqa: E402
from backend.runtime_bridge import answer_question  # noqa: E402


def _system_c_answer(question: str) -> tuple[str, dict[str, Any], dict[str, Any]]:
    answers = answer_question(question, "C")
    answer = answers.get("C") if isinstance(answers, dict) else {}
    if not isinstance(answer, dict):
        answer = {}
    metadata = answer.get("metadata") if isinstance(answer.get("metadata"), dict) else {}
    return str(answer.get("response") or ""), metadata, answer


def _route_summary(metadata: dict[str, Any], fallback_route: dict[str, Any]) -> dict[str, Any]:
    route_info = metadata.get("route_info") if isinstance(metadata.get("route_info"), dict) else fallback_route
    return {
        "resolved_domain": route_info.get("resolved_domain"),
        "target_domain": route_info.get("target_domain"),
        "strict_isolation": route_info.get("strict_isolation"),
        "domain_switch": route_info.get("domain_switch"),
        "confidence": route_info.get("confidence"),
        "reason": route_info.get("reason"),
    }


def _contains_useful_terms(response: str, terms: tuple[str, ...]) -> bool:
    lowered = response.lower()
    return any(term in lowered for term in terms)


def _run_case(
    name: str,
    question: str,
    check: Callable[[str, dict[str, Any], dict[str, Any]], tuple[bool, dict[str, Any]]],
) -> bool:
    fallback_route = route_query(question)
    response, metadata, answer = _system_c_answer(question)
    passed, details = check(response, metadata, fallback_route)
    route_info = _route_summary(metadata, fallback_route)
    terms = contamination_terms_for_route(response, route_info)

    print(f"\n{name}: {'PASS' if passed else 'FAIL'}")
    print(json.dumps({
        "question": question,
        "route": route_info,
        "soft_repair_applied": metadata.get("soft_repair_applied"),
        "contamination_terms_found": terms,
        "metadata_terms": metadata.get("contamination_self_check_terms"),
        "main_rag_used": metadata.get("main_rag_used"),
        "main_rag_relevance_reason": metadata.get("main_rag_relevance_reason"),
        "details": details,
        "response_preview": response[:800],
        "latency": answer.get("latency"),
    }, indent=2, sort_keys=True))
    return passed


def check_strict_daily(response: str, metadata: dict[str, Any], route: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    route_info = metadata.get("route_info") if isinstance(metadata.get("route_info"), dict) else route
    terms = contamination_terms_for_route(response, {"target_domain": "daily"})
    return (
        route_info.get("target_domain") == "daily"
        and route_info.get("strict_isolation") is True
        and not terms
    ), {"target_domain": route_info.get("target_domain"), "strict_isolation": route_info.get("strict_isolation"), "terms": terms}


def check_strict_robotics(response: str, metadata: dict[str, Any], route: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    route_info = metadata.get("route_info") if isinstance(metadata.get("route_info"), dict) else route
    terms = contamination_terms_for_route(response, {"target_domain": "robotics"})
    return (
        route_info.get("target_domain") == "robotics"
        and route_info.get("strict_isolation") is True
        and not terms
    ), {"target_domain": route_info.get("target_domain"), "strict_isolation": route_info.get("strict_isolation"), "terms": terms}


def check_mixed(response: str, metadata: dict[str, Any], route: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    route_info = metadata.get("route_info") if isinstance(metadata.get("route_info"), dict) else route
    lowered = response.lower()
    no_replace = "replace" in lowered and any(term in lowered for term in ("no", "cannot", "can't", "not", "unsupported"))
    careful = no_replace or any(term in lowered for term in ("indirect", "different"))
    return (
        route_info.get("resolved_domain") == "mixed"
        and route_info.get("strict_isolation") is False
        and careful
    ), {"resolved_domain": route_info.get("resolved_domain"), "strict_isolation": route_info.get("strict_isolation"), "careful_relationship": careful}


def check_robotics_normal(response: str, metadata: dict[str, Any], route: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    route_info = metadata.get("route_info") if isinstance(metadata.get("route_info"), dict) else route
    useful = _contains_useful_terms(response, ("debug", "check", "log", "sensor", "odometry", "map"))
    return (
        route_info.get("target_domain") == "robotics"
        and useful
    ), {"target_domain": route_info.get("target_domain"), "useful_robotics_terms": useful}


def check_daily_normal(response: str, metadata: dict[str, Any], route: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    route_info = metadata.get("route_info") if isinstance(metadata.get("route_info"), dict) else route
    terms = contamination_terms_for_route(response, {"target_domain": "daily"})
    useful = _contains_useful_terms(response, ("privacy", "password", "network", "data", "safe", "risk"))
    return (
        route_info.get("target_domain") == "daily"
        and useful
        and not terms
    ), {"target_domain": route_info.get("target_domain"), "useful_daily_terms": useful, "terms": terms}


def main() -> int:
    cases = [
        (
            "Strict Daily After SLAM",
            "Earlier we discussed SLAM. Now explain food delivery apps without contamination.",
            check_strict_daily,
        ),
        (
            "Strict Daily Online Privacy",
            "Earlier we discussed robot localization. Now explain online privacy without mixing the two domains.",
            check_strict_daily,
        ),
        (
            "Strict Robotics After Daily Apps",
            "After discussing ride-sharing apps, explain SLAM without mixing daily apps.",
            check_strict_robotics,
        ),
        (
            "Mixed Relationship",
            "Can Uber GPS replace SLAM?",
            check_mixed,
        ),
        (
            "Normal Robotics",
            "How can a beginner debug SLAM?",
            check_robotics_normal,
        ),
        (
            "Normal Daily",
            "What risks should I consider while using public Wi-Fi?",
            check_daily_normal,
        ),
    ]

    results = [_run_case(name, question, check) for name, question, check in cases]
    passed = all(results)
    print(f"\nOVERALL: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
