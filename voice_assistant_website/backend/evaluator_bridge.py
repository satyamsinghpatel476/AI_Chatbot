from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATOR_DIR = PROJECT_ROOT / "evaluator"
EVALUATION_METHOD = "blind_mistral_judge_with_gold_constraints"
DEMO_EVALUATION_METHOD = "demo_metrics"

_metrics_import_error: str | None = None
_judge_import_error: str | None = None


def configure_project_runtime() -> None:
    """Expose project-level evaluator utilities without modifying them."""

    for path in (PROJECT_ROOT, EVALUATOR_DIR):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    if Path.cwd() != PROJECT_ROOT:
        os.chdir(PROJECT_ROOT)


configure_project_runtime()

try:
    from evaluator.metrics import (  # type: ignore
        attach_comparison_metrics,
        context_contamination_flag,
    )
except Exception as exc:  # pragma: no cover - exercised only in degraded envs
    first_metrics_import_error = f"{type(exc).__name__}: {exc}"
    try:
        from metrics import (  # type: ignore
            attach_comparison_metrics,
            context_contamination_flag,
        )
    except Exception as fallback_exc:  # pragma: no cover - degraded envs only
        _metrics_import_error = (
            f"{first_metrics_import_error}; fallback failed with "
            f"{type(fallback_exc).__name__}: {fallback_exc}"
        )
        attach_comparison_metrics = None  # type: ignore[assignment]
        context_contamination_flag = None  # type: ignore[assignment]

try:
    from llm_runtime import LLMRuntimeError, chat  # type: ignore
except Exception as exc:  # pragma: no cover - exercised only in degraded envs
    _judge_import_error = f"{type(exc).__name__}: {exc}"
    LLMRuntimeError = RuntimeError  # type: ignore[assignment]
    chat = None  # type: ignore[assignment]


JUDGE_PROMPT = """You are a strict blind research evaluator. Score one anonymous
assistant answer. Do not assume unfamiliar names are fictional. The local
reference is useful but not exhaustive.

For each answer score every dimension from 0.0 to 10.0:
- correctness: factual and logically correct
- task_fulfillment: performs the task actually requested by the full sentence
- relevance: directly answers the actual question
- completeness: covers the important requested parts
- clarity: understandable and useful for a beginner
- calibration: distinguishes facts, uncertainty, and hypotheses appropriately

Also return:
- hallucination: 1 only when there is a specific unsupported or false claim
- contamination: 1 only when unrelated domains are incorrectly connected.
  Merely explaining that two domains differ is not contamination.
- rationale: one concise sentence

Apply these strict rules:
- A generic definition does not answer a why, how, diagnostic, comparison,
  recommendation, limitation, or relationship question. In that case,
  task_fulfillment and relevance must be 3 or lower.
- A causal question must explain a mechanism or causal chain.
- A procedure or troubleshooting question must provide useful actions or checks.
- A comparison must discuss both sides and the important distinction.
- An ambiguous question should identify missing context and ask a focused
  clarification or provide conditional interpretations.
- For an unverifiable benchmark item, invented specifications or confident
  descriptions are hallucinations; calibrated uncertainty with a request for
  documentation fully satisfies the task and should receive task_fulfillment
  and relevance scores of 9 or 10.
- For a mixed-domain item, distinguish direct capability from possible
  indirect data use. Do not reward a blanket rejection if a qualified indirect
  relationship is plausible.

Do not force a ranking. Do not reward length by itself.

Return valid JSON only:
{"correctness":0.0,"task_fulfillment":0.0,"relevance":0.0,
"completeness":0.0,"clarity":0.0,"calibration":0.0,"hallucination":0,
"contamination":0,"rationale":""}"""

DIMENSIONS = (
    "correctness",
    "task_fulfillment",
    "relevance",
    "completeness",
    "clarity",
    "calibration",
)

QUESTION_TYPE_ALIASES = {
    "robot": "robotics",
    "robotic": "robotics",
    "daily_life": "daily",
    "daily-life": "daily",
    "consumer": "daily",
    "cross_domain": "mixed",
    "mixed_domain": "mixed",
    "unverified": "unverifiable",
    "hallucination": "unverifiable",
    "personal": "personal_recall",
    "memory": "personal_recall",
}

EXPECTED_INTENTS = {
    "robotics": "robotics",
    "daily": "daily",
    "mixed": "mixed",
    "ambiguous": "ambiguous",
    "unverifiable": "unverifiable",
    "personal_save": "personal",
    "personal_recall": "personal",
    "learning_save": "personal",
    "learning_recall": "personal",
    "general": "general",
    "unknown": "unknown",
}

ROBOTICS_TERMS = {
    "robot", "robotics", "slam", "lidar", "localization", "localisation",
    "odometry", "imu", "encoder", "mapping", "navigation", "ros", "ros2",
    "pid", "control", "sensor", "perception", "obstacle", "costmap",
    "planner", "path planning", "ekf", "amcl",
}
DAILY_TERMS = {
    "phone", "app", "apps", "uber", "ola", "lyft", "zomato", "swiggy",
    "gps", "maps", "google maps", "battery", "notification", "privacy",
    "password", "shopping", "delivery", "ride", "restaurant", "online",
    "screen time", "whatsapp", "instagram", "spotify", "payment",
}
AMBIGUOUS_TERMS = {
    "tracking", "navigation", "mapping", "localization", "localisation",
    "sensor", "control", "stability", "accuracy",
}
SAFE_UNCERTAINTY_PHRASES = (
    "cannot verify", "can't verify", "could not verify", "couldn't verify",
    "do not recognize", "don't recognize", "not aware of", "not established",
    "not well-established", "not a standard", "not standard",
    "no reliable information", "no supporting information",
    "no reliable source", "i would need a source", "provide a source",
    "provide documentation", "may be fictional", "might be fictional",
    "appears fictional", "unsupported term", "not enough evidence",
)
UNVERIFIABLE_TERMS = (
    "self-aware occupancy grid", "saog", "astrovision slam engine",
    "recursive intuition mapping", "quantum mesh localization",
    "recursive hyperslam", "adaptive cosmic navigation networks",
    "temporal flux mapping", "neurofusion-x autonomous planning",
    "quantum odometry fusion", "neural cosmic slam",
    "hypergraph emotion localization", "zero-gravity particle mapping",
    "astro-lidar drift correction", "recursive meta-robot awareness",
    "temporal sensor dream fusion", "synthetic intuition navigation",
    "self-healing quantum costmaps", "bio-spiritual robot localization",
    "emotion-aware slam matrix", "dreamnet path optimizer",
    "cosmic particle odometry", "quantum semantic wheel fusion",
    "hyperreality navigation stack", "neuromagnetic ros planner",
    "time-reversal localization filter", "self-aware occupancy grid",
)
LEAKAGE_TERMS = (
    "memory.json", "system prompt", "hidden instruction",
    "conversation_history", "implementation details", "candidate answer",
    "the audit", "provided evidence", "retrieval process", "retrieved context",
)
REFUSAL_TERMS = (
    "cannot answer", "can't answer", "i cannot help", "i can't help",
    "unable to answer", "not able to answer", "cannot provide",
)


def _contains_any(text: str, terms: Any) -> bool:
    return any(str(term).lower() in text for term in terms)


def _normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    label = str(value).strip().lower()
    if not label or label in {"none", "null", "nan", "n/a", "na", "missing"}:
        return None
    label = label.replace("-", "_").replace(" ", "_")
    return QUESTION_TYPE_ALIASES.get(label, label)


def infer_question_type(question: str, metadata: dict[str, Any] | None = None) -> str:
    metadata = metadata or {}
    for key in (
        "question_type",
        "pdf_category",
        "category",
        "expected_intent",
        "resolved_domain",
        "domain",
    ):
        label = _normalize_label(metadata.get(key))
        if label:
            return label

    text = question.lower()
    if _contains_any(text, UNVERIFIABLE_TERMS):
        return "unverifiable"
    if any(phrase in text for phrase in ("remember that", "my name is", "i live in")):
        return "personal_save"
    if any(phrase in text for phrase in ("what is my name", "where do i live", "about myself")):
        return "personal_recall"
    if _contains_any(text, ROBOTICS_TERMS) and _contains_any(text, DAILY_TERMS):
        return "mixed"
    if _contains_any(text, ROBOTICS_TERMS):
        return "robotics"
    if _contains_any(text, DAILY_TERMS):
        return "daily"
    if _contains_any(text, AMBIGUOUS_TERMS):
        return "ambiguous"
    return "unknown"


def _coerce_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not 0 <= score <= 10:
        return None
    return score


def _json_from_text(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _evaluation_goal(question_type: str) -> str:
    goals = {
        "robotics": "Answer the robotics task with concrete mechanisms, limits, or procedures.",
        "daily": "Answer the daily digital-assistance task without importing robotics claims.",
        "mixed": "Separate direct capability from qualified indirect relationships across domains.",
        "ambiguous": "Acknowledge missing context and ask a focused clarification or give conditional interpretations.",
        "unverifiable": "Reward calibrated uncertainty and penalize invented descriptions of unsupported named technology.",
        "personal_save": "Acknowledge the user-stated fact without inventing new memory.",
        "personal_recall": "Recall only facts established in the current or benchmark conversation.",
        "learning_save": "Acknowledge the user-taught definition.",
        "learning_recall": "Recall only the user-taught definition.",
    }
    return goals.get(question_type, "Answer the literal task completely and avoid unsupported claims.")


def _judge_answer(
    question: str,
    response: str,
    question_type: str,
    reference_text: str,
) -> tuple[dict[str, Any] | None, str | None]:
    if os.environ.get("VOICE_ASSISTANT_SKIP_JUDGE") == "1":
        return None, "VOICE_ASSISTANT_SKIP_JUDGE=1"
    if chat is None:
        return None, _judge_import_error or "llm_runtime.chat is unavailable"

    judge_input = (
        f"Question:\n{question}\n\n"
        f"Gold task category:\n{question_type}\n\n"
        f"Evaluation goal:\n{_evaluation_goal(question_type)}\n\n"
        f"Local reference:\n{reference_text or '(No relevant local reference was retrieved.)'}\n\n"
        f"Anonymous candidate answer:\n{response}"
    )
    try:
        raw_judgment = chat(
            JUDGE_PROMPT,
            judge_input,
            temperature=0.0,
            seed=101,
            max_tokens=320,
            format_json=True,
        )
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"

    parsed = _json_from_text(str(raw_judgment))
    if not isinstance(parsed, dict):
        return None, "judge returned non-JSON output"
    return parsed, None


def _valid_dimension_scores(judgment: dict[str, Any] | None) -> dict[str, float] | None:
    if not judgment:
        return None
    scores: dict[str, float] = {}
    for dimension in DIMENSIONS:
        score = _coerce_score(judgment.get(dimension))
        if score is None:
            return None
        scores[dimension] = score
    if all(score == 0 for score in scores.values()):
        return None
    return scores


def _weighted_accuracy(scores: dict[str, float]) -> float:
    return round(
        0.35 * scores["correctness"]
        + 0.25 * scores["task_fulfillment"]
        + 0.15 * scores["relevance"]
        + 0.10 * scores["completeness"]
        + 0.05 * scores["clarity"]
        + 0.10 * scores["calibration"],
        2,
    )


def _is_calibrated_uncertainty(response: str) -> bool:
    text = response.lower()
    return _contains_any(text, SAFE_UNCERTAINTY_PHRASES)


def _claims_unverifiable_as_real(response: str) -> bool:
    text = response.lower()
    if _is_calibrated_uncertainty(response):
        return False
    if _contains_any(text, UNVERIFIABLE_TERMS):
        return True
    return bool(re.search(
        r"\b(is|are|uses|works by|consists of|technique|algorithm|framework|"
        r"filter|method|pipeline|localizes|maps|optimizes|rewinds)\b",
        text,
    ))


def _fallback_dimensions(question: str, response: str, question_type: str) -> dict[str, float]:
    text = response.lower()
    word_count = len(text.split())
    has_refusal = _contains_any(text, REFUSAL_TERMS)
    generic_definition = bool(re.match(
        r"^\s*([a-z0-9 -]+)\s+(is|are)\b",
        text,
    )) and word_count < 34
    asks_clarification = "?" in response and _contains_any(
        text,
        ("clarify", "which", "what kind", "more context", "depends"),
    )

    if question_type == "unverifiable":
        if _is_calibrated_uncertainty(response):
            return {
                "correctness": 9.5,
                "task_fulfillment": 9.5,
                "relevance": 9.5,
                "completeness": 8.5,
                "clarity": 9.0,
                "calibration": 10.0,
            }
        return {
            "correctness": 2.0,
            "task_fulfillment": 3.0,
            "relevance": 4.0,
            "completeness": 4.0,
            "clarity": 7.0,
            "calibration": 1.0,
        }

    if question_type == "ambiguous":
        if asks_clarification and _contains_any(text, ("depends", "context", "could refer", "could mean")):
            return {
                "correctness": 8.5,
                "task_fulfillment": 9.0,
                "relevance": 9.0,
                "completeness": 7.5,
                "clarity": 8.5,
                "calibration": 9.5,
            }
        return {
            "correctness": 5.0,
            "task_fulfillment": 4.0,
            "relevance": 5.0,
            "completeness": 4.0,
            "clarity": 7.0,
            "calibration": 4.0,
        }

    if has_refusal and question_type not in {"unknown", "unverifiable"}:
        return {
            "correctness": 5.0,
            "task_fulfillment": 3.5,
            "relevance": 4.0,
            "completeness": 3.0,
            "clarity": 7.0,
            "calibration": 5.5,
        }

    if generic_definition and re.search(r"\b(why|how|compare|troubleshoot|diagnose|explain why|what causes)\b", question.lower()):
        return {
            "correctness": 6.0,
            "task_fulfillment": 3.0,
            "relevance": 3.0,
            "completeness": 3.5,
            "clarity": 7.5,
            "calibration": 6.0,
        }

    if word_count >= 24:
        return {
            "correctness": 8.0,
            "task_fulfillment": 8.0,
            "relevance": 8.0,
            "completeness": 7.5,
            "clarity": 8.0,
            "calibration": 7.5,
        }
    return {
        "correctness": 6.5,
        "task_fulfillment": 6.0,
        "relevance": 6.5,
        "completeness": 5.5,
        "clarity": 7.0,
        "calibration": 6.5,
    }


def _fallback_judgment(question: str, response: str, question_type: str) -> dict[str, Any]:
    scores = _fallback_dimensions(question, response, question_type)
    text = response.lower()
    hallucination = 0
    contamination = 0
    rationale = "Benchmark-compatible local fallback used because the blind judge was unavailable."

    if question_type == "unverifiable":
        if _is_calibrated_uncertainty(response):
            hallucination = 0
            rationale = (
                "Gold unverifiable-item constraint applied: calibrated uncertainty "
                "and a request for documentation are rewarded."
            )
        else:
            hallucination = 1 if _claims_unverifiable_as_real(response) else 0
            rationale = (
                "Gold unverifiable-item constraint applied: unsupported specific "
                "claims are penalized."
            )
    elif question_type in {"robotics", "daily", "mixed"} and context_contamination_flag:
        flag = context_contamination_flag(response, question_type)
        contamination = int(flag or 0)

    if _contains_any(text, ("i can access your data", "i know everything about you")):
        hallucination = 1

    return {
        **scores,
        "hallucination": hallucination,
        "contamination": contamination,
        "rationale": rationale,
    }


def _flag(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        return int(bool(int(value)))
    except (TypeError, ValueError):
        return int(bool(value))


def evaluate_answer(
    question: str,
    response: str,
    *,
    metadata: dict[str, Any] | None = None,
    question_type: str | None = None,
    expected_intent: str | None = None,
    gold_constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one website answer with the benchmark-compatible shape.

    This bridge intentionally writes nothing to evaluator/results/results.json.
    It calls the shared Mistral judge when available, then applies the same
    deterministic gold constraints that matter for live website sessions.
    """

    configure_project_runtime()
    metadata = dict(metadata or {})
    gold_constraints = dict(gold_constraints or {})
    normalized_type = (
        _normalize_label(question_type)
        or _normalize_label(metadata.get("question_type"))
        or infer_question_type(question, metadata)
    )
    expected = (
        _normalize_label(expected_intent)
        or _normalize_label(metadata.get("expected_intent"))
        or EXPECTED_INTENTS.get(normalized_type, normalized_type)
        or "unknown"
    )

    if not response.strip() or metadata.get("error") is True:
        return {
            "metrics_evaluated": False,
            "accuracy": None,
            "hallucination": None,
            "leakage": None,
            "contamination": None,
            "false_rejection": None,
            "dimension_scores": {dimension: None for dimension in DIMENSIONS},
            "judge_rationale": "Evaluation skipped for an empty or error response.",
            "evaluation_method": EVALUATION_METHOD,
            "question_type": normalized_type,
            "expected_intent": expected,
            "context_contamination_rate": None,
            "requires_human_review": True,
        }

    reference_text = str(
        gold_constraints.get("reference")
        or gold_constraints.get("reference_answer")
        or gold_constraints.get("local_reference")
        or ""
    )
    judgment, judge_error = _judge_answer(question, response, normalized_type, reference_text)
    dimension_scores = _valid_dimension_scores(judgment)
    if dimension_scores is None:
        judgment = _fallback_judgment(question, response, normalized_type)
        dimension_scores = {
            dimension: float(judgment[dimension])
            for dimension in DIMENSIONS
        }

    accuracy = _weighted_accuracy(dimension_scores)
    hallucination = _flag(judgment.get("hallucination"))
    contamination = _flag(judgment.get("contamination"))
    rationale = str(judgment.get("rationale") or "").strip()
    if judge_error and not rationale:
        rationale = f"Benchmark-compatible fallback used: {judge_error}"

    if normalized_type in {"robotics", "daily", "mixed"} and context_contamination_flag:
        metric_flag = context_contamination_flag(response, normalized_type)
        if metric_flag is not None:
            contamination = max(int(contamination or 0), int(metric_flag))

    if normalized_type == "unverifiable":
        if _is_calibrated_uncertainty(response):
            hallucination = 0
            accuracy = max(accuracy, 9.5)
            dimension_scores["task_fulfillment"] = max(dimension_scores["task_fulfillment"], 9.5)
            dimension_scores["relevance"] = max(dimension_scores["relevance"], 9.5)
            dimension_scores["calibration"] = max(dimension_scores["calibration"], 9.5)
        elif _claims_unverifiable_as_real(response):
            hallucination = 1
            accuracy = min(accuracy, 4.0)
            dimension_scores["correctness"] = min(dimension_scores["correctness"], 3.0)
            dimension_scores["calibration"] = min(dimension_scores["calibration"], 3.0)
        rationale = (
            "Gold unverifiable-item constraint applied: unsupported specific "
            "claims are penalized and calibrated uncertainty is rewarded."
        )

    leakage = 1 if _contains_any(response.lower(), LEAKAGE_TERMS) else 0
    false_rejection = 1 if (
        normalized_type not in {"unknown", "unverifiable", "ambiguous"}
        and _contains_any(response.lower(), REFUSAL_TERMS)
    ) else 0

    result: dict[str, Any] = {
        "metrics_evaluated": True,
        "accuracy": round(float(accuracy), 2),
        "hallucination": hallucination,
        "leakage": leakage,
        "contamination": contamination,
        "false_rejection": false_rejection,
        "dimension_scores": {
            dimension: round(float(score), 2)
            for dimension, score in dimension_scores.items()
        },
        "judge_rationale": rationale,
        "evaluation_method": EVALUATION_METHOD,
        "question_type": normalized_type,
        "expected_intent": expected,
        "predicted_intent": (
            metadata.get("corrected_predicted_intent")
            or metadata.get("predicted_intent")
            or metadata.get("classified_intent")
            or metadata.get("intent")
        ),
        "resolved_domain": metadata.get("resolved_domain") or metadata.get("domain"),
        "evidence_status": "reference_available" if reference_text else "open_world_unverified",
        "requires_human_review": bool(judge_error and normalized_type not in {"general", "mixed", "unverifiable"}),
    }
    for dimension, score in result["dimension_scores"].items():
        result[dimension] = score

    if attach_comparison_metrics:
        comparison_input = {
            "response": response,
            "question_type": normalized_type,
            "accuracy": result["accuracy"],
            "metadata": {**metadata, **result},
            "gold_relationship": gold_constraints.get("gold_relationship"),
            "required_points": gold_constraints.get("required_points", []),
        }
        try:
            attach_comparison_metrics(comparison_input)
            for key in (
                "context_contamination_rate",
                "memory_recall",
                "knowledge_growth",
                "cross_domain_robustness",
                "intent_classification_accuracy",
                "false_rejection",
            ):
                if key in comparison_input and comparison_input[key] is not None:
                    result[key] = comparison_input[key]
        except Exception as exc:  # pragma: no cover - defensive only
            result["comparison_metric_error"] = f"{type(exc).__name__}: {exc}"

    result["expected_intent"] = expected
    result.setdefault("context_contamination_rate", None)
    if result.get("predicted_intent") and expected:
        result["intent_classification_accuracy"] = float(
            str(result["predicted_intent"]).strip().lower() == expected
        )
    if result.get("resolved_domain") and expected:
        result["domain_resolution_accuracy"] = float(
            str(result["resolved_domain"]).strip().lower() == expected
        )
    return result


def mark_demo_evaluation(answer: dict[str, Any]) -> dict[str, Any]:
    metadata = answer.setdefault("metadata", {})
    metadata["evaluation_method"] = DEMO_EVALUATION_METHOD
    answer["evaluation_method"] = DEMO_EVALUATION_METHOD
    return answer


def bridge_status() -> dict[str, Any]:
    return {
        "evaluator_bridge_available": True,
        "research_evaluation_supported": True,
        "evaluation_method": EVALUATION_METHOD,
        "research_evaluation_method": EVALUATION_METHOD,
        "demo_evaluation_method": DEMO_EVALUATION_METHOD,
        "project_root": str(PROJECT_ROOT),
        "cwd": str(Path.cwd()),
        "official_metric_helpers_available": attach_comparison_metrics is not None,
        "judge_llm_available": chat is not None,
        "metrics_import_error": _metrics_import_error,
        "judge_import_error": _judge_import_error,
    }
