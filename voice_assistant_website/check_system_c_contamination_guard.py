from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from backend.runtime_bridge import ask_system  # noqa: E402


ROBOTICS_FORBIDDEN_IN_DAILY = (
    "slam",
    "ros",
    "lidar",
    "robot",
    "localization",
)
DAILY_FORBIDDEN_IN_ROBOTICS = (
    "uber",
    "ride-sharing",
    "ride sharing",
    "food delivery",
    "shopping",
)
MIXED_REQUIRED = (
    "direct answer",
    "relationship type",
    "robotics perspective",
    "daily-life perspective",
    "important difference",
    "final conclusion",
)


def _contains_any(text: str, terms: tuple[str, ...]) -> list[str]:
    lowered = text.lower()
    return [term for term in terms if term in lowered]


def _run_case(name: str, question: str, check) -> bool:
    answer = ask_system("C", question)
    response = str(answer.get("response") or "")
    metadata = answer.get("metadata") if isinstance(answer.get("metadata"), dict) else {}
    passed, details = check(response, metadata)
    print(f"\n{name}: {'PASS' if passed else 'FAIL'}")
    print(json.dumps({
        "question": question,
        "metadata": {
            "strict_domain_isolation": metadata.get("strict_domain_isolation"),
            "target_domain": metadata.get("target_domain"),
            "system_c_mixed_format_required": metadata.get("system_c_mixed_format_required"),
            "system_c_repair_applied": metadata.get("system_c_repair_applied"),
            "contamination_terms_removed": metadata.get("contamination_terms_removed"),
            "postcheck_terms": metadata.get("system_c_postcheck_contamination_terms"),
            "postrepair_terms": metadata.get("system_c_postrepair_contamination_terms"),
            "main_rag_used": metadata.get("main_rag_used"),
            "main_rag_suppressed_for_system_c": metadata.get("main_rag_suppressed_for_system_c"),
        },
        "details": details,
        "response_preview": response[:700],
    }, indent=2, sort_keys=True))
    return passed


def check_daily_isolated(response: str, _metadata: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    forbidden = _contains_any(response, ROBOTICS_FORBIDDEN_IN_DAILY)
    return not forbidden, {"forbidden_terms": forbidden}


def check_robotics_isolated(response: str, _metadata: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    forbidden = _contains_any(response, DAILY_FORBIDDEN_IN_ROBOTICS)
    return not forbidden, {"forbidden_terms": forbidden}


def check_mixed_format(response: str, _metadata: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    lowered = response.lower()
    missing = [term for term in MIXED_REQUIRED if term not in lowered]
    says_cannot_replace = "cannot replace" in lowered or "can't replace" in lowered or "no," in lowered
    return not missing and says_cannot_replace, {
        "missing_sections": missing,
        "says_cannot_replace": says_cannot_replace,
    }


def check_robotics_normal(response: str, metadata: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    lowered = response.lower()
    is_robotics = metadata.get("target_domain") == "robotics" or "slam" in lowered
    useful = any(term in lowered for term in ("debug", "check", "log", "map", "sensor", "odometry"))
    return is_robotics and useful, {
        "target_domain": metadata.get("target_domain"),
        "contains_useful_debug_terms": useful,
    }


def check_daily_normal(response: str, metadata: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    forbidden = _contains_any(response, ROBOTICS_FORBIDDEN_IN_DAILY)
    daily_terms = any(term in response.lower() for term in ("privacy", "safety", "delivery", "app", "payment"))
    return not forbidden and daily_terms, {
        "target_domain": metadata.get("target_domain"),
        "forbidden_terms": forbidden,
        "contains_daily_terms": daily_terms,
    }


def main() -> int:
    cases = [
        (
            "Strict Daily After Robotics",
            "Earlier we discussed SLAM. Now explain food delivery apps without contamination.",
            check_daily_isolated,
        ),
        (
            "Strict Robotics After Daily Apps",
            "Earlier we discussed ride-sharing apps. Now explain SLAM without mixing daily apps.",
            check_robotics_isolated,
        ),
        (
            "Mixed Relationship Format",
            "Can Uber GPS replace SLAM?",
            check_mixed_format,
        ),
        (
            "Normal Robotics",
            "How can a beginner debug SLAM?",
            check_robotics_normal,
        ),
        (
            "Normal Daily",
            "What risks should I consider while using food delivery apps?",
            check_daily_normal,
        ),
    ]
    results = [_run_case(name, question, check) for name, question, check in cases]
    passed = all(results)
    print(f"\nOVERALL: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
