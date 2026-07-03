import json
import re
from collections import Counter


INSTRUCTION_VERBS = {
    "explain",
    "answer",
    "what",
    "why",
    "how",
    "when",
    "where",
    "who",
    "which",
    "can",
    "should",
    "is",
    "are",
    "do",
    "does",
    "compare",
    "define",
    "describe",
    "tell",
    "evaluate",
    "analyze",
    "recommend",
    "suggest",
    "choose",
    "reduce",
    "improve",
    "debug",
    "check",
    "detect",
    "learn",
    "remember",
    "use",
    "run",
    "tune",
    "train",
    "perform",
}

FRAGMENT_ONLY_QUESTIONS = {
    ".",
    "terms.",
    "meaning separately.",
}

CONTEXT_DEPENDENT_FRAGMENTS = {
    "without contamination.",
    "direct, indirect, or unsupported.",
}


def _words(text):
    return re.findall(r"[A-Za-z0-9*+-]+", str(text or ""))


def _starts_with_instruction_verb(text):
    lowered = text.lower().strip()
    return any(
        re.match(rf"^{re.escape(verb)}\b", lowered)
        for verb in INSTRUCTION_VERBS
    )


def _is_personal_save_statement(text):
    lowered = text.lower().strip()
    patterns = [
        r"^my name is\s+\S+",
        r"^my favorite app is\s+\S+",
        r"^my favorite robot is\s+\S+",
        r"^i live in\s+\S+",
        r"^i study\s+\S+",
        r"^remember that\s+\S+",
        r"^remember i\s+\S+",
    ]
    return any(re.match(pattern, lowered) for pattern in patterns)


def _is_learning_save_statement(text):
    lowered = text.lower().strip()
    return bool(
        re.match(r"^learn that\s+\S+", lowered)
        or re.match(r"^teaching:\s*\S+", lowered)
    )


def _is_comparison_statement(text):
    lowered = text.lower().strip()
    return bool(
        re.match(r"^difference between\s+.+\s+and\s+.+[.!]?$", lowered)
        or re.match(r"^compare\s+.+\s+and\s+.+[.!]?$", lowered)
    )


def _is_contextual_instruction(text):
    lowered = text.lower().strip()
    action = (
        r"(?:answer|explain|describe|define|compare|tell|give|list|evaluate|"
        r"analyze|debug|detect)"
    )
    patterns = [
        rf"^earlier we discussed\s+.+\.\s*now\s+{action}\b.+",
        rf"^after discussing\s+.+,\s*{action}\b.+",
        rf"^now\s+{action}\b.+",
    ]
    return any(re.match(pattern, lowered) for pattern in patterns)


def _named_technical_entity(question):
    text = str(question or "").strip()
    text = re.sub(
        r"^\s*(?:explain|describe|define|analyze|evaluate|compare|tell(?: me about)?|check|detect|debug)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip(" .!?")
    words = re.findall(r"[A-Za-z0-9*+-]+", text)
    if not words:
        return False
    title_words = [
        word for word in words
        if word[:1].isupper() and word.lower() not in {"the", "a", "an"}
    ]
    return bool(
        re.search(r"[a-z][A-Z]|[A-Z]{2,}|[A-Za-z]+-[A-Za-z0-9-]+", text)
        or len(title_words) >= 2
    )


def validate_benchmark_question(question, category=None):
    text = str(question or "").strip()
    word_count = len(_words(text))
    if not text:
        return {
            "valid": False,
            "reason": "empty_question",
            "word_count": 0,
        }

    lowered = text.lower()
    if lowered in FRAGMENT_ONLY_QUESTIONS:
        return {
            "valid": False,
            "reason": "fragment_only",
            "word_count": word_count,
        }
    if lowered in CONTEXT_DEPENDENT_FRAGMENTS:
        return {
            "valid": False,
            "reason": "context_dependent_fragment",
            "word_count": word_count,
        }
    if word_count == 0:
        return {
            "valid": False,
            "reason": "fragment_only",
            "word_count": word_count,
        }

    has_question_mark = "?" in text
    if has_question_mark:
        return {
            "valid": True,
            "reason": None,
            "word_count": word_count,
        }

    starts_with_instruction = _starts_with_instruction_verb(text)
    if starts_with_instruction:
        return {
            "valid": True,
            "reason": None,
            "word_count": word_count,
        }
    if _is_personal_save_statement(text):
        return {
            "valid": True,
            "reason": None,
            "word_count": word_count,
        }
    if _is_learning_save_statement(text):
        return {
            "valid": True,
            "reason": None,
            "word_count": word_count,
        }
    if _is_comparison_statement(text):
        return {
            "valid": True,
            "reason": None,
            "word_count": word_count,
        }
    if _is_contextual_instruction(text):
        return {
            "valid": True,
            "reason": None,
            "word_count": word_count,
        }
    if (
        str(category or "").lower() == "unverifiable"
        and _named_technical_entity(text)
    ):
        return {
            "valid": True,
            "reason": None,
            "word_count": word_count,
        }

    return {
        "valid": False,
        "reason": "unsupported_format",
        "word_count": word_count,
    }


def invalid_benchmark_reason(question):
    result = validate_benchmark_question(question)
    return None if result["valid"] else result["reason"]


def is_valid_benchmark_question(question: str, category=None) -> bool:
    return validate_benchmark_question(question, category=category)["valid"]


def filter_valid_benchmark_questions(questions):
    valid = []
    skipped = []
    for index, question in enumerate(questions, 1):
        if isinstance(question, dict):
            question_text = str(question.get("question", "") or "").strip()
            category = str(question.get("category", "unknown") or "unknown")
        elif isinstance(question, str):
            question_text = str(question or "").strip()
            category = "unknown"
        else:
            skipped.append({
                "index": index,
                "category": "unknown",
                "question": str(question or ""),
                "reason": "unsupported_format",
                "word_count": 0,
            })
            continue

        result = validate_benchmark_question(question_text, category=category)
        if result["valid"]:
            valid.append(question)
        else:
            skipped.append({
                "index": index,
                "category": category,
                "question": question_text,
                "reason": result["reason"],
                "word_count": result["word_count"],
            })
    return valid, skipped


def save_skipped_questions(skipped, path="skipped_questions.json"):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(skipped, handle, indent=2, ensure_ascii=False)


def category_counts(items):
    counts = Counter()
    for item in items:
        if isinstance(item, dict):
            counts[str(item.get("category", "unknown") or "unknown")] += 1
        else:
            counts["unknown"] += 1
    return dict(sorted(counts.items()))
