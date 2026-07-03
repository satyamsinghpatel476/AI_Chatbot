import json
import math
import os
import re
from collections import defaultdict

from llm_runtime import LLMRuntimeError, chat
from evaluator.metrics import cross_domain_robustness_components


RELATIONSHIP_JUDGE_PROMPT = """You are evaluating assistant responses for a
controlled research benchmark. Candidate labels are anonymous. Use the supplied
gold relationship and behavioral requirements. Do not infer system identity.

For each candidate return binary 0/1 fields:
- contaminated: carries an unrelated prior domain into the answer or asserts an
  incorrect cross-domain capability.
- false_rejection: rejects a valid direct, indirect, or conditional relationship
  as impossible, or refuses an answerable unrelated-context task.
- relationship_correct: identifies the gold relationship correctly.
- required_information: names the important information in required_points.
- limitations_stated: states relevant limitations or conditions.
- onboard_safety_preserved: does not replace onboard sensing, perception, or
  safety when they remain necessary.
- task_fulfilled: answers the current request rather than a nearby topic.
- safe_explanation: relationship_correct, task_fulfilled, and not contaminated.

For unrelated context switches, relationship_correct means the response answers
the current domain without carrying over the prior domain. For unrelated dual
questions, it means both parts are answered separately.

Return valid JSON only:
{"candidates":{"Candidate 1":{"contaminated":0,"false_rejection":0,
"relationship_correct":0,"required_information":0,"limitations_stated":0,
"onboard_safety_preserved":0,"task_fulfilled":0,"safe_explanation":0}}}"""

SINGLE_RELATIONSHIP_JUDGE_PROMPT = """Evaluate one anonymous assistant response
using the supplied gold relationship and behavioral requirements.

Return binary integer 0/1 fields:
- contaminated
- false_rejection
- relationship_correct
- required_information
- limitations_stated
- onboard_safety_preserved
- task_fulfilled
- safe_explanation

For unrelated context switches, a correct response answers only the current
domain. For unrelated dual questions, it answers both parts separately.
Return valid JSON only with exactly those eight fields."""


def _valid_binary_judgment(value):
    fields = [
        "contaminated",
        "false_rejection",
        "relationship_correct",
        "required_information",
        "limitations_stated",
        "onboard_safety_preserved",
        "task_fulfilled",
        "safe_explanation",
    ]
    return isinstance(value, dict) and all(
        value.get(field) in {0, 1} for field in fields
    )


def _single_judge_input(case, row):
    context = "\n".join(
        f"User: {item['user']}\nAssistant: {item['assistant']}"
        for item in case.get("context", [])
    ) or "(none)"
    return (
        f"Case ID: {case['id']}\n"
        f"Prior context:\n{context}\n\n"
        f"Current query:\n{case['query']}\n\n"
        f"Gold relationship: {case['gold_relationship']}\n"
        f"Required points: {json.dumps(case.get('required_points', []))}\n"
        f"Forbidden claims: {json.dumps(case.get('forbidden_claims', []))}\n\n"
        f"Anonymous response:\n{row['response']}"
    )


def retry_single_judgment(case, row, *, seed, timeout):
    try:
        raw = chat(
            SINGLE_RELATIONSHIP_JUDGE_PROMPT,
            _single_judge_input(case, row),
            temperature=0.0,
            seed=seed,
            max_tokens=260,
            format_json=True,
            ensure_complete=False,
            timeout=timeout,
        )
        judgment = json.loads(raw)
        return judgment if _valid_binary_judgment(judgment) else None
    except (LLMRuntimeError, json.JSONDecodeError, TypeError, ValueError):
        return None


def judge_relationship_case(case, candidate_rows, *, seed=0, timeout=180):
    if os.environ.get("FULL_RESEARCH_MODE") != "1":
        return heuristic_relationship_case(case, candidate_rows)

    mapping = {
        f"Candidate {index + 1}": row
        for index, row in enumerate(candidate_rows)
    }
    candidates = "\n\n".join(
        f"{label}:\n{row['response']}"
        for label, row in mapping.items()
    )
    single_prefix = _single_judge_input(case, candidate_rows[0]).split(
        "Anonymous response:",
        1,
    )[0]
    judge_input = single_prefix + f"Anonymous candidates:\n{candidates}"
    try:
        raw = chat(
            RELATIONSHIP_JUDGE_PROMPT,
            judge_input,
            temperature=0.0,
            seed=seed,
            max_tokens=max(420, 180 * len(candidate_rows)),
            format_json=True,
            ensure_complete=False,
            timeout=timeout,
        )
        parsed = json.loads(raw)
        judgments = parsed.get("candidates", parsed)
    except (LLMRuntimeError, json.JSONDecodeError, TypeError, ValueError):
        judgments = {}

    scored = []
    for candidate_index, (label, row) in enumerate(mapping.items()):
        judgment = judgments.get(label)
        if not _valid_binary_judgment(judgment):
            for retry_index in range(2):
                judgment = retry_single_judgment(
                    case,
                    row,
                    seed=(
                        seed
                        + candidate_index
                        + 10000
                        + retry_index * 1000
                    ),
                    timeout=timeout,
                )
                if _valid_binary_judgment(judgment):
                    break
        if not _valid_binary_judgment(judgment):
            judgment = {
                "contaminated": None,
                "false_rejection": None,
                "relationship_correct": None,
                "required_information": None,
                "limitations_stated": None,
                "onboard_safety_preserved": None,
                "task_fulfilled": None,
                "safe_explanation": None,
            }
            review = True
        else:
            review = False
        heuristic = cross_domain_robustness_components(row.get("response", ""), case)
        for field in [
            "contaminated",
            "false_rejection",
            "relationship_correct",
            "required_information",
            "limitations_stated",
            "onboard_safety_preserved",
            "task_fulfilled",
            "safe_explanation",
        ]:
            if heuristic.get(field) is not None:
                if field == "contaminated" and heuristic[field] == 0:
                    judgment[field] = 0
                elif field == "contaminated" and heuristic[field] == 1:
                    judgment[field] = 1
                elif field == "false_rejection":
                    judgment[field] = heuristic[field]
                elif judgment.get(field) is None:
                    judgment[field] = heuristic[field]

        components = [
            judgment["relationship_correct"],
            judgment["required_information"],
            judgment["limitations_stated"],
            judgment["onboard_safety_preserved"],
            judgment["task_fulfilled"],
        ]
        robustness = (
            sum(components) / len(components)
            if all(value is not None for value in components)
            else None
        )
        scored.append({
            **row,
            **judgment,
            "cross_domain_robustness": robustness,
            "requires_human_review": review,
            "evaluation_method": "anonymous_gold_conditioned_mistral_judge",
        })
    return scored


def heuristic_relationship_case(case, candidate_rows):
    scored = []
    for row in candidate_rows:
        judgment = cross_domain_robustness_components(
            row.get("response", ""),
            case,
        )
        components = [
            judgment["relationship_correct"],
            judgment["required_information"],
            judgment["limitations_stated"],
            judgment["onboard_safety_preserved"],
            judgment["task_fulfilled"],
        ]
        robustness = (
            sum(components) / len(components)
            if all(value is not None for value in components)
            else None
        )
        scored.append({
            **row,
            **judgment,
            "cross_domain_robustness": robustness,
            "requires_human_review": False,
            "evaluation_method": "deterministic_gold_conditioned_heuristic",
        })
    return scored


def score_relationship_rows(rows, *, seed=0, timeout=180, batch_size=4):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["id"]].append(row)

    scored = []
    for case_index, case_rows in enumerate(grouped.values()):
        case = case_rows[0]
        for offset in range(0, len(case_rows), batch_size):
            batch = case_rows[offset:offset + batch_size]
            scored.extend(judge_relationship_case(
                case,
                batch,
                seed=seed + case_index + offset,
                timeout=timeout,
            ))
    return scored


def _normalized(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def score_memory_rows(rows):
    expected_values = {
        _normalized(row["expected"]) for row in rows
    }
    scored = []
    uncertainty = [
        "don't know", "do not know", "not sure", "cannot recall",
        "can't recall", "unknown", "no saved",
    ]
    for row in rows:
        response = _normalized(row["response"])
        expected = _normalized(row["expected"])
        negated = bool(re.search(
            rf"\b(?:not|isn't|is not)\s+{re.escape(expected)}\b",
            response,
        ))
        has_expected = expected in response and not negated
        is_uncertain = any(phrase in response for phrase in uncertainty)
        other_values = {
            value for value in expected_values
            if value != expected and value in response
        }
        exact = int(has_expected and not is_uncertain and not other_values)
        missing = int(not has_expected and is_uncertain)
        incorrect = int(not exact and not missing)
        scored.append({
            **row,
            "exact_recall": exact,
            "incorrect_recall": incorrect,
            "missing_recall": missing,
            "cross_user_or_domain_leakage": int(bool(other_values)),
        })
    return scored


def _term_coverage(response, terms):
    tokens = set(re.findall(r"[a-z0-9]+", response.lower()))
    required = [
        set(re.findall(r"[a-z0-9]+", term.lower()))
        for term in terms
    ]
    matched = sum(bool(term_tokens & tokens) for term_tokens in required)
    return matched / max(len(required), 1)


def score_knowledge_rows(rows):
    uncertainty = [
        "cannot verify", "can't verify", "don't know", "do not know",
        "not familiar", "not established", "no information", "not taught",
        "need more context", "please share",
    ]
    scored = []
    for row in rows:
        exact_response = row["exact_recall_response"]
        paraphrase_response = row["paraphrased_recall_response"]
        unrelated_response = row["unrelated_query_response"]
        exact_coverage = _term_coverage(
            exact_response,
            row["required_terms"],
        )
        paraphrase_coverage = _term_coverage(
            paraphrase_response,
            row["required_terms"],
        )
        exact = int(exact_coverage >= 2 / 3)
        paraphrased = int(paraphrase_coverage >= 2 / 3)
        unrelated_rejected = int(any(
            phrase in unrelated_response.lower() for phrase in uncertainty
        ))
        false_memory = int(not unrelated_rejected)
        latency_values = [
            row.get("exact_recall_latency_ms"),
            row.get("paraphrased_recall_latency_ms"),
            row.get("unrelated_query_latency_ms"),
        ]
        latency_values = [
            float(value) for value in latency_values if value is not None
        ]
        scored.append({
            **row,
            "latency_ms": (
                sum(latency_values) / len(latency_values)
                if latency_values else None
            ),
            "exact_knowledge_recall": exact,
            "semantic_recall_accuracy": paraphrased,
            "unrelated_concept_rejection": unrelated_rejected,
            "false_memory": false_memory,
            "knowledge_growth_accuracy": (exact + paraphrased) / 2,
        })
    return scored
