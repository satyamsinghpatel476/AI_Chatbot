from __future__ import annotations

from collections import Counter
import re
from pathlib import Path
from typing import Any

from pypdf import PdfReader


VALID_STARTING_WORDS = {
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
    "explain",
    "describe",
    "define",
    "compare",
    "evaluate",
    "analyze",
    "debug",
    "detect",
    "tell",
    "give",
    "list",
    "use",
    "run",
    "tune",
    "train",
    "perform",
    "remember",
    "learn",
    "difference",
}

KNOWN_CATEGORIES = {"robotics", "daily", "mixed", "ambiguous", "unverifiable"}

_START_WORD_PATTERN = "|".join(sorted(VALID_STARTING_WORDS, key=len, reverse=True))
_NUMBERING_RE = re.compile(
    r"^\s*(?:[-*•]\s*)?(?:Q(?:uestion)?\s*)?(?:\d{1,4}|[A-Za-z])[\).:-]\s+",
    re.IGNORECASE,
)
_STRICT_NUMBERED_LINE_RE = re.compile(r"^\s*(\d{1,4})\.\s+(.*)$")
_INLINE_STRICT_NUMBER_RE = re.compile(r"(?<!^)\s+(?=\d{1,4}\.\s+)")
_CATEGORY_RE = re.compile(r"^\s*\[([A-Za-z_-]+)\]\s*(.*)$")
_START_RE = re.compile(rf"^[\"'(\[]*({_START_WORD_PATTERN})\b", re.IGNORECASE)
_NUMBERED_SPLIT_RE = re.compile(
    r"\s+(?=(?:Q(?:uestion)?\s*)?\d{1,4}[\).:-]\s+[A-Za-z])",
    re.IGNORECASE,
)
_QUESTION_SPLIT_RE = re.compile(rf"(?<=[?.!])\s+(?=(?:{_START_WORD_PATTERN})\b)", re.IGNORECASE)


def extract_text_from_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _normalize_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _prepare_text(text: str) -> str:
    clean_text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    clean_text = _NUMBERED_SPLIT_RE.sub("\n", clean_text)
    clean_text = _QUESTION_SPLIT_RE.sub("\n", clean_text)
    return clean_text


def _prepare_numbered_text(text: str) -> str:
    clean_text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    return _INLINE_STRICT_NUMBER_RE.sub("\n", clean_text)


def _strip_numbering(value: str) -> str:
    return _NUMBERING_RE.sub("", value).strip()


def _starts_like_question(value: str) -> bool:
    return bool(_START_RE.search(value.strip()))


def _word_count(value: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", value))


def _looks_like_fragment(value: str) -> bool:
    text = value.strip()
    if len(text) < 8:
        return True
    if _word_count(text) < 2:
        return True
    if re.fullmatch(r"(?:page\s*)?\d+", text, re.IGNORECASE):
        return True
    return False


def _clean_candidate(value: str) -> str | None:
    candidate = _strip_numbering(value)
    candidate = _normalize_spaces(candidate).strip(" -–—\t")
    if not candidate or _looks_like_fragment(candidate):
        return None
    if not (_starts_like_question(candidate) or candidate.endswith("?")):
        return None
    return candidate


def _dedupe_key(value: str) -> str:
    return re.sub(r"\W+", " ", value).strip().lower()


def _category_and_question(text: str) -> tuple[str | None, str]:
    category_match = _CATEGORY_RE.match(text)
    if not category_match:
        return None, text
    category = category_match.group(1).strip().lower().replace("-", "_")
    clean_question = category_match.group(2).strip()
    return category, clean_question


def _entry(number: int | None, text: str) -> dict[str, Any] | None:
    normalized_text = _normalize_spaces(text)
    if not normalized_text:
        return None
    category, clean_question = _category_and_question(normalized_text)
    clean_question = _normalize_spaces(clean_question)
    if not clean_question or _looks_like_fragment(clean_question):
        return None

    if number is None:
        raw = clean_question
    elif category:
        raw = f"{number}. [{category}] {clean_question}"
    else:
        raw = f"{number}. {clean_question}"

    return {
        "number": number,
        "category": category if category in KNOWN_CATEGORIES else category,
        "question": clean_question,
        "raw": raw,
    }


def _validation(entries: list[dict[str, Any]], numbered_entries_found: int) -> dict[str, Any]:
    numbers = [
        int(entry["number"])
        for entry in entries
        if isinstance(entry.get("number"), int)
    ]
    number_counts = Counter(numbers)
    duplicate_numbers = sorted(number for number, count in number_counts.items() if count > 1)
    missing_numbers: list[int] = []
    if numbers:
        existing = set(numbers)
        missing_numbers = [
            number
            for number in range(min(numbers), max(numbers) + 1)
            if number not in existing
        ]

    category_counts = Counter(
        str(entry.get("category"))
        for entry in entries
        if entry.get("category")
    )

    return {
        "numbered_entries_found": numbered_entries_found,
        "missing_numbers": missing_numbers,
        "duplicate_numbers": duplicate_numbers,
        "category_counts": dict(sorted(category_counts.items())),
    }


def extract_numbered_question_entries_from_text(text: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    current_number: int | None = None
    current_parts: list[str] = []
    seen_exact_numbered_entries: set[tuple[int, str]] = set()
    raw_numbered_entries_found = 0

    def flush() -> None:
        nonlocal current_number, current_parts
        if current_number is None or not current_parts:
            current_number = None
            current_parts = []
            return

        text_value = _normalize_spaces(" ".join(current_parts))
        exact_key = (current_number, text_value.lower())
        if exact_key not in seen_exact_numbered_entries:
            entry = _entry(current_number, text_value)
            if entry:
                entries.append(entry)
            seen_exact_numbered_entries.add(exact_key)

        current_number = None
        current_parts = []

    for raw_line in _prepare_numbered_text(text).splitlines():
        line = _normalize_spaces(raw_line)
        if not line:
            continue

        numbered_match = _STRICT_NUMBERED_LINE_RE.match(line)
        if numbered_match:
            flush()
            current_number = int(numbered_match.group(1))
            current_parts = [numbered_match.group(2).strip()]
            raw_numbered_entries_found += 1
            continue

        if current_number is not None:
            current_parts.append(line)

    flush()
    return entries, _validation(entries, raw_numbered_entries_found)


def extract_questions_from_text(text: str) -> list[str]:
    questions: list[str] = []
    seen: set[str] = set()
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        candidate = _clean_candidate(" ".join(buffer))
        buffer.clear()
        if not candidate:
            return
        key = _dedupe_key(candidate)
        if key and key not in seen:
            seen.add(key)
            questions.append(candidate)

    for raw_line in _prepare_text(text).splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue

        starts_new = bool(_NUMBERING_RE.match(line)) or _starts_like_question(_strip_numbering(line))
        if starts_new:
            flush()
            buffer.append(line)
        elif buffer and _word_count(line) >= 2:
            buffer.append(line)
        else:
            flush()
            continue

        if buffer and re.search(r"[?.!]$", line):
            flush()

    flush()
    return questions


def extract_question_entries_from_text(text: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    numbered_entries, numbered_validation = extract_numbered_question_entries_from_text(text)
    if len(numbered_entries) >= 5:
        return numbered_entries, {
            **numbered_validation,
            "extraction_mode": "numbered",
            "fallback_used": False,
        }

    fallback_questions = extract_questions_from_text(text)
    fallback_entries = [
        entry
        for entry in (_entry(None, question) for question in fallback_questions)
        if entry
    ]
    fallback_validation = _validation(fallback_entries, numbered_validation["numbered_entries_found"])
    return fallback_entries, {
        **fallback_validation,
        "numbered_entries_found": numbered_validation["numbered_entries_found"],
        "extraction_mode": "fallback_line_based",
        "fallback_used": True,
    }


def extract_questions_from_pdf_result(path: Path) -> dict[str, Any]:
    entries, validation = extract_question_entries_from_text(extract_text_from_pdf(path))
    return {
        "questions": entries,
        **validation,
    }


def extract_questions_from_pdf(path: Path) -> list[dict[str, Any]]:
    return extract_questions_from_pdf_result(path)["questions"]
