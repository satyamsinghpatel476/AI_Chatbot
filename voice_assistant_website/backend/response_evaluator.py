from __future__ import annotations

import re
from typing import Any


FAKE_TECH_TERMS = [
    "Self-Aware Occupancy Grid",
    "SAOG",
    "AstroVision SLAM Engine",
    "Recursive Intuition Mapping",
]

ROBOTICS_TERMS = {
    "robot", "robotics", "slam", "lidar", "localization", "odometry",
    "imu", "encoder", "mapping", "navigation", "path planning", "ros",
    "ros2", "pid", "control", "sensor", "perception", "obstacle",
}

DAILY_TERMS = {
    "phone", "app", "apps", "uber", "ola", "zomato", "swiggy", "gps",
    "maps", "battery", "notification", "privacy", "password", "shopping",
    "delivery", "ride", "restaurant", "online", "study", "screen time",
}

AMBIGUOUS_TERMS = {"tracking", "navigation", "mapping", "localization", "sensor", "control"}


def _contains_any(text: str, terms: list[str] | set[str]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms)


def infer_question_type(question: str, metadata: dict[str, Any] | None = None) -> str:
    metadata = metadata or {}
    for key in ("question_type", "resolved_domain", "domain"):
        value = metadata.get(key)
        if value:
            normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
            if normalized in {
                "robotics", "daily", "general", "unknown", "mixed",
                "ambiguous", "unverifiable", "personal_save",
                "personal_recall", "learning_save", "learning_recall",
            }:
                return normalized

    text = question.lower()
    if any(phrase in text for phrase in ["remember that", "my name is", "i live in"]):
        return "personal_save"
    if any(phrase in text for phrase in ["what is my name", "where do i live", "about myself"]):
        return "personal_recall"
    if any(phrase in text for phrase in ["learn that", "teach:"]):
        return "learning_save"
    if any(phrase in text for phrase in ["what did i teach", "what does", "mean"]):
        return "learning_recall"
    if _contains_any(text, ROBOTICS_TERMS) and _contains_any(text, DAILY_TERMS):
        return "mixed"
    if _contains_any(text, ROBOTICS_TERMS):
        return "robotics"
    if _contains_any(text, DAILY_TERMS):
        return "daily"
    if _contains_any(text, AMBIGUOUS_TERMS):
        return "ambiguous"
    if any(phrase in text for phrase in ["unknown", "unverified", "cannot verify"]):
        return "unverifiable"
    return "general"


def evaluate_response(
    question: str,
    response: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Website-local fallback evaluator for live session dashboard metrics."""

    metadata = metadata or {}
    response_text = response or ""
    normalized_response = response_text.lower()
    question_type = infer_question_type(question, metadata)

    fake_terms_found = [
        term for term in FAKE_TECH_TERMS
        if term.lower() in normalized_response
    ]
    has_fake_terms = bool(fake_terms_found)
    empty_or_error = not response_text.strip() or metadata.get("error") is True
    generic_but_not_wrong = bool(re.search(
        r"\b(depends|in general|usually|typically|can vary)\b",
        normalized_response,
    )) and not has_fake_terms

    hallucination = 1 if has_fake_terms else 0
    contamination = 1 if has_fake_terms else 0
    leakage = 0
    false_rejection = 1 if (
        "cannot answer" in normalized_response
        and question_type not in {"unknown", "unverifiable"}
    ) else 0

    if empty_or_error:
        return {
            "metrics_evaluated": False,
            "evaluation_method": "skipped_error_or_empty_response",
            "question_type": question_type,
            "fake_terms_found": fake_terms_found,
        }

    if has_fake_terms:
        accuracy = 4
        correctness = 3
        relevance = 4
        completeness = 5
        clarity = 6
        calibration = 3
        task_fulfillment = 4
    elif generic_but_not_wrong:
        accuracy = 7
        correctness = 7
        relevance = 7
        completeness = 6
        clarity = 8
        calibration = 7
        task_fulfillment = 6
    else:
        accuracy = 8
        correctness = 8
        relevance = 8
        completeness = 8
        clarity = 8
        calibration = 8
        task_fulfillment = 8

    if false_rejection:
        accuracy = min(accuracy, 5)
        task_fulfillment = min(task_fulfillment, 4)
        relevance = min(relevance, 5)

    intent_accuracy = 1 if metadata.get("predicted_intent") or metadata.get("intent") else None
    domain_accuracy = 1 if metadata.get("resolved_domain") or metadata.get("domain") else None

    dimension_scores = {
        "correctness": correctness,
        "task_fulfillment": task_fulfillment,
        "relevance": relevance,
        "completeness": completeness,
        "clarity": clarity,
        "calibration": calibration,
    }
    return {
        "metrics_evaluated": True,
        "evaluation_method": "rule_based_fallback",
        "question_type": question_type,
        "accuracy": accuracy,
        "hallucination": hallucination,
        "leakage": leakage,
        "contamination": contamination,
        "false_rejection": false_rejection,
        "correctness": correctness,
        "task_fulfillment": task_fulfillment,
        "relevance": relevance,
        "completeness": completeness,
        "clarity": clarity,
        "calibration": calibration,
        "dimension_scores": dimension_scores,
        "intent_classification_accuracy": intent_accuracy,
        "domain_resolution_accuracy": domain_accuracy,
        "fake_terms_found": fake_terms_found,
    }
