from __future__ import annotations

import json
import os
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from backend.evaluator_bridge import bridge_status, evaluate_answer  # noqa: E402
from backend.live_results_summarizer import (  # noqa: E402
    OFFICIAL_RESULTS_PATH,
    compute_official_results_summary,
    load_official_results,
    summarize_results_like,
)
from backend.rag_temp import get_combined_context  # noqa: E402


def _round_or_none(value):
    return None if value is None else round(float(value), 6)


def _main_metrics(summary):
    return {
        system: {
            "count": values["count"],
            "avg_accuracy": _round_or_none(values["avg_accuracy"]),
            "median_accuracy": _round_or_none(values["median_accuracy"]),
            "avg_latency": _round_or_none(values["avg_latency"]),
            "hallucination_count": values["hallucination_count"],
            "leakage_count": values["leakage_count"],
            "contamination_count": values["contamination_count"],
            "false_rejection_count": values["false_rejection_count"],
        }
        for system, values in summary["systems"].items()
    }


def _metrics_close(left, right, *, tolerance=1e-4):
    if left.keys() != right.keys():
        return False
    for system in left:
        if left[system].keys() != right[system].keys():
            return False
        for key, left_value in left[system].items():
            right_value = right[system][key]
            if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
                if abs(float(left_value) - float(right_value)) > tolerance:
                    return False
            elif left_value != right_value:
                return False
    return True


def validate_official_summary() -> bool:
    entries = load_official_results()
    website_summary = compute_official_results_summary()
    expected_summary = summarize_results_like(entries)

    print("Website official summary:")
    print(json.dumps(_main_metrics(website_summary), indent=2, sort_keys=True))
    print("\nExpected summarize_results.py-like summary:")
    print(json.dumps(_main_metrics(expected_summary), indent=2, sort_keys=True))

    pass_count = website_summary.get("official_results_count") == len(entries)
    pass_main = _metrics_close(_main_metrics(website_summary), _main_metrics(expected_summary))
    print(f"\nOfficial adapter count == results.json count: {'PASS' if pass_count else 'FAIL'}")
    print(f"Main metrics match calculated values: {'PASS' if pass_main else 'FAIL'}")
    return pass_count and pass_main


def validate_unverifiable_scoring() -> bool:
    os.environ["VOICE_ASSISTANT_SKIP_JUDGE"] = "1"
    question = "Explain Time-Reversal Localization Filter."
    answer_a = "The Time-Reversal Localization Filter is a robotics technique that reverses sensor time to localize robots."
    answer_b = "I cannot verify that this is a standard concept without a reliable paper, repository, or documentation."

    scored_a = evaluate_answer(
        question,
        answer_a,
        question_type="unverifiable",
        expected_intent="unverifiable",
        system_name="C",
    )
    scored_b = evaluate_answer(
        question,
        answer_b,
        question_type="unverifiable",
        expected_intent="unverifiable",
        system_name="C",
    )

    print("\nUnverifiable scoring:")
    print(json.dumps({
        "answer_a": {
            "accuracy": scored_a.get("accuracy"),
            "hallucination": scored_a.get("hallucination"),
            "evaluation_method": scored_a.get("evaluation_method"),
        },
        "answer_b": {
            "accuracy": scored_b.get("accuracy"),
            "hallucination": scored_b.get("hallucination"),
            "evaluation_method": scored_b.get("evaluation_method"),
        },
    }, indent=2, sort_keys=True))

    pass_unverifiable = (
        scored_a.get("hallucination") == 1
        and scored_b.get("hallucination") == 0
        and (scored_b.get("accuracy") or 0) > (scored_a.get("accuracy") or 0)
    )
    print(f"Unverifiable scoring behavior: {'PASS' if pass_unverifiable else 'FAIL'}")
    return pass_unverifiable


def validate_evaluator_loading() -> bool:
    status = bridge_status()
    fields = {
        "evaluator_file_exists": status.get("evaluator_file_exists"),
        "evaluator_file_loaded": status.get("evaluator_file_loaded"),
        "evaluator_callable_available": status.get("evaluator_callable_available"),
        "fallback_used": status.get("fallback_evaluator_used"),
        "evaluator_bridge_method": status.get("evaluator_bridge_method"),
        "evaluator_import_error": status.get("evaluator_import_error"),
    }
    print("\nEvaluator loading:")
    print(json.dumps(fields, indent=2, sort_keys=True))

    passed = bool(status.get("evaluator_file_exists")) and bool(status.get("evaluator_file_loaded"))
    if status.get("evaluator_callable_available"):
        passed = passed and not bool(status.get("fallback_evaluator_used"))
    print(f"Evaluator file-path loading: {'PASS' if passed else 'FAIL'}")
    return passed


def _rag_probe(question: str) -> dict:
    context = get_combined_context(question)
    return {
        "main_rag_used": context.get("main_rag_used"),
        "main_rag_candidate_count": context.get("main_rag_candidate_count"),
        "main_rag_relevance_score": context.get("main_rag_relevance_score"),
        "main_rag_relevance_reason": context.get("main_rag_relevance_reason"),
        "main_rag_relevance_category": context.get("main_rag_relevance_category"),
        "combined_context_chars": context.get("combined_context_chars"),
    }


def validate_rag_relevance() -> bool:
    daily_question = "What risks should I consider while using food delivery apps?"
    slam_question = "How can a beginner debug SLAM?"
    fake_question = "Explain Time-Reversal Localization Filter."

    daily = _rag_probe(daily_question)
    slam = _rag_probe(slam_question)
    fake = _rag_probe(fake_question)

    print("\nRAG relevance:")
    print(json.dumps({
        daily_question: daily,
        slam_question: slam,
        fake_question: fake,
    }, indent=2, sort_keys=True))

    daily_ok = (
        daily.get("main_rag_used") is False
        or daily.get("main_rag_relevance_reason") == "daily_relevant_context"
    )
    slam_ok = slam.get("main_rag_used") is True
    fake_ok = fake.get("main_rag_used") is False
    print(f"Daily app question avoids robotics RAG: {'PASS' if daily_ok else 'FAIL'}")
    print(f"SLAM question may use robotics RAG: {'PASS' if slam_ok else 'FAIL'}")
    print(f"Fake unverifiable term blocks generic RAG: {'PASS' if fake_ok else 'FAIL'}")
    return daily_ok and slam_ok and fake_ok


def main() -> int:
    if not OFFICIAL_RESULTS_PATH.exists():
        print(f"Official results file not found: {OFFICIAL_RESULTS_PATH}")
        return 1
    evaluator_ok = validate_evaluator_loading()
    rag_ok = validate_rag_relevance()
    official_ok = validate_official_summary()
    unverifiable_ok = validate_unverifiable_scoring()
    passed = evaluator_ok and rag_ok and official_ok and unverifiable_ok
    print(f"\nOVERALL: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
