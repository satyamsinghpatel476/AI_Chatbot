from __future__ import annotations

import csv
import importlib
import importlib.util
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

from .domain_router import normalize_domain_label, route_query
from .session_store import session_store


ALLOWED_EXTENSIONS = {".pdf", ".txt", ".csv", ".docx", ".json"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAIN_RAG_INDEX_PATH = PROJECT_ROOT / "rag" / "index.faiss"
MAIN_RAG_CHUNKS_PATH = PROJECT_ROOT / "rag" / "chunks.pkl"
TEMP_CONTEXT_CHAR_LIMIT = 4500
MAIN_CONTEXT_CHAR_LIMIT = 800
MIXED_MAIN_CONTEXT_CHAR_LIMIT = 500
COMBINED_CONTEXT_CHAR_LIMIT = 9000
MAIN_RAG_RELEVANCE_GATE_ENABLED = True
MAIN_RAG_MIN_OVERLAP_RATIO = 0.18
_MAIN_RAG_CANDIDATES = (
    ("rag.retrieve", "retrieve"),
    ("rag.retriever", "retrieve"),
    ("src.rag.retrieve", "retrieve"),
    ("src.rag.retriever", "retrieve"),
    ("retrieve", "retrieve"),
    ("retriever", "retrieve"),
    ("research_core", "retrieve_local_knowledge"),
)
_MAIN_RAG_CACHE: dict[str, Any] = {
    "attempted": False,
    "function": None,
    "module": None,
    "error": None,
    "test_query_result_count": 0,
}
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "give",
    "how",
    "i",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "please",
    "should",
    "tell",
    "the",
    "this",
    "to",
    "using",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}
ROBOTICS_NAMED_TERMS = (
    "slam",
    "lidar",
    "ros",
    "ros2",
    "amcl",
    "ekf",
    "pid",
    "odometry",
    "costmap",
    "turtlebot",
    "turtlebot3",
    "path planning",
    "motion planning",
    "trajectory planning",
    "occupancy grid",
    "localization",
    "localisation",
    "loop closure",
    "obstacle avoidance",
    "dynamic obstacle avoidance",
    "inverse kinematics",
    "forward kinematics",
    "kalman filter",
    "particle filter",
    "sensor fusion",
    "robot navigation",
    "autonomous navigation",
)
ROBOTICS_DOMAIN_TERMS = ROBOTICS_NAMED_TERMS + (
    "robot",
    "robots",
    "robotics",
    "robotic",
    "mapping",
    "planner",
    "navigation stack",
    "wheel encoder",
    "manipulator",
    "robot arm",
    "differential drive",
    "imu",
)
DAILY_DOMAIN_TERMS = (
    "food delivery",
    "delivery app",
    "delivery apps",
    "ride sharing",
    "ride-sharing",
    "cloud storage",
    "social media",
    "consumer app",
    "shopping",
    "password",
    "privacy",
    "phone",
    "smartphone",
    "notification",
    "battery",
    "maps",
    "google maps",
    "zomato",
    "swiggy",
    "uber",
    "ola",
    "lyft",
    "whatsapp",
    "instagram",
    "spotify",
    "payment",
)
DOMAIN_RELATION_TERMS = (
    "relationship",
    "relation",
    "relate",
    "compare",
    "analogy",
    "similar",
    "difference",
    "different",
    "daily life",
    "daily-life",
    "consumer",
    "robotics",
)
UNVERIFIABLE_FULL_TERMS = (
    "quantum mesh localization",
    "recursive hyperslam",
    "time-reversal localization filter",
    "self-aware occupancy grid",
    "astrovision slam engine",
    "recursive intuition mapping",
    "adaptive cosmic navigation networks",
    "temporal flux mapping",
    "neurofusion-x autonomous planning",
    "quantum odometry fusion",
    "neural cosmic slam",
    "hypergraph emotion localization",
    "zero-gravity particle mapping",
    "astro-lidar drift correction",
    "recursive meta-robot awareness",
    "temporal sensor dream fusion",
    "synthetic intuition navigation",
    "self-healing quantum costmaps",
    "bio-spiritual robot localization",
    "emotion-aware slam matrix",
    "dreamnet path optimizer",
    "cosmic particle odometry",
    "quantum semantic wheel fusion",
    "hyperreality navigation stack",
    "neuromagnetic ros planner",
)
_TEMP_DOCUMENT_TRIGGERS = {
    "uploaded",
    "upload",
    "document",
    "file",
    "pdf",
    "doc",
    "text",
    "csv",
    "json",
    "attachment",
}


def safe_filename(filename: str) -> str:
    source = Path(filename or "upload.txt")
    suffix = source.suffix.lower()
    stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", source.stem).strip("._-")
    stem = stem[:80] or "upload"
    return f"{stem}_{uuid4().hex[:10]}{suffix}"


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def _extract_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _extract_docx(path: Path) -> str:
    from docx import Document

    document = Document(str(path))
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def _extract_csv(data: bytes) -> str:
    text = _decode_text(data)
    rows = []
    reader = csv.reader(io.StringIO(text))
    for index, row in enumerate(reader):
        if index >= 200:
            break
        rows.append(" | ".join(cell.strip() for cell in row))
    return "\n".join(rows)


def _extract_json(data: bytes) -> str:
    text = _decode_text(data)
    try:
        parsed: Any = json.loads(text)
    except json.JSONDecodeError:
        return text
    return json.dumps(parsed, indent=2, ensure_ascii=False)


def extract_text(path: Path, data: bytes, *, max_chars: int = 5000) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text = _extract_pdf(path)
    elif suffix == ".docx":
        text = _extract_docx(path)
    elif suffix == ".csv":
        text = _extract_csv(data)
    elif suffix == ".json":
        text = _extract_json(data)
    else:
        text = _decode_text(data)
    return " ".join(text.split())[:max_chars]


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", (text or "").lower())
        if token not in _STOPWORDS
    }


def _has_overlap(question: str, context: str, *, minimum: int = 1) -> bool:
    question_tokens = _tokens(question)
    if not question_tokens:
        return False
    return len(question_tokens.intersection(_tokens(context))) >= minimum


def _contains_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    lower_text = text.lower()
    return any(phrase in lower_text for phrase in phrases)


def _infer_question_category(question: str) -> str:
    text = question.lower()
    if _contains_phrase(text, UNVERIFIABLE_FULL_TERMS):
        return "unverifiable"
    has_robotics = _contains_phrase(text, ROBOTICS_DOMAIN_TERMS)
    has_daily = _contains_phrase(text, DAILY_DOMAIN_TERMS)
    asks_relation = _contains_phrase(text, DOMAIN_RELATION_TERMS)
    if (has_robotics and has_daily) or (asks_relation and has_robotics and "daily" in text):
        return "mixed"
    if has_robotics:
        return "robotics"
    if has_daily:
        return "daily"
    if any(term in text for term in ("tracking", "navigation", "mapping", "localization", "localisation", "sensor")):
        return "ambiguous"
    return "unknown"


def _retrieved_chunks(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        clean = " ".join(value.split())
        return [clean] if clean else []
    if isinstance(value, dict):
        chunks: list[str] = []
        for key in ("context", "text", "content", "page_content"):
            if key in value:
                chunks.extend(_retrieved_chunks(value[key]))
        for key in ("results", "documents", "docs", "contexts", "chunks"):
            if key in value:
                chunks.extend(_retrieved_chunks(value[key]))
        if chunks:
            return chunks
        clean = " ".join(_normalize_retrieved_context(value).split())
        return [clean] if clean else []
    if isinstance(value, (list, tuple)):
        chunks = []
        for item in value:
            chunks.extend(_retrieved_chunks(item))
        return [chunk for chunk in chunks if chunk]
    page_content = getattr(value, "page_content", None)
    if page_content:
        clean = " ".join(str(page_content).split())
        return [clean] if clean else []
    clean = " ".join(str(value).split())
    return [clean] if clean else []


def _query_term_stats(question: str, chunk: str) -> tuple[int, float, set[str]]:
    query_terms = _tokens(question)
    if not query_terms:
        return 0, 0.0, set()
    chunk_terms = _tokens(chunk)
    overlap = query_terms.intersection(chunk_terms)
    ratio = len(overlap) / max(len(query_terms), 1)
    return len(overlap), ratio, overlap


def _rag_relevance_details(
    question: str,
    retrieved_chunks: list[str],
    *,
    question_type: str | None = None,
    route_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    route_info = dict(route_info or {})
    route_resolved = normalize_domain_label(route_info.get("resolved_domain"))
    route_target = normalize_domain_label(route_info.get("target_domain"))
    category = (
        question_type
        or route_target
        or route_resolved
        or _infer_question_category(question)
    )
    category = str(category or "unknown").strip().lower()
    strict_isolation = bool(route_info.get("strict_isolation"))
    route_confidence = float(route_info.get("confidence") or 0.0)
    if not retrieved_chunks:
        return {
            "used": False,
            "score": 0.0,
            "reason": "no_retrieved_chunks",
            "category": category,
            "max_chars": 0,
        }

    top_chunk = retrieved_chunks[0]
    all_context = "\n\n".join(retrieved_chunks)
    question_text = question.lower()
    top_text = top_chunk.lower()
    context_text = all_context.lower()
    exact_fake_terms = [term for term in UNVERIFIABLE_FULL_TERMS if term in question_text]
    top_overlap, top_ratio, overlap_terms = _query_term_stats(question, top_chunk)
    named_robotics_match = any(
        term in question_text and term in context_text
        for term in ROBOTICS_NAMED_TERMS
    )
    lexical_match = (
        top_overlap >= 2
        or top_ratio >= MAIN_RAG_MIN_OVERLAP_RATIO
        or named_robotics_match
    )
    max_chars = MIXED_MAIN_CONTEXT_CHAR_LIMIT if category == "mixed" else MAIN_CONTEXT_CHAR_LIMIT
    score = round(top_ratio, 4)

    if exact_fake_terms:
        exact_found = any(term in context_text for term in exact_fake_terms)
        return {
            "used": exact_found,
            "score": 1.0 if exact_found else score,
            "reason": (
                "exact_unverifiable_term_found"
                if exact_found
                else "blocked_unverifiable_without_exact_term"
            ),
            "category": "unverifiable",
            "max_chars": MAIN_CONTEXT_CHAR_LIMIT if exact_found else 0,
            "overlap_terms": sorted(overlap_terms),
        }

    if route_resolved == "unverifiable" or category == "unverifiable":
        return {
            "used": False,
            "score": score,
            "reason": "blocked_unverifiable_without_exact_term",
            "category": "unverifiable",
            "max_chars": 0,
            "overlap_terms": sorted(overlap_terms),
        }

    if category == "unverifiable":
        return {
            "used": False,
            "score": score,
            "reason": "blocked_unverifiable_without_exact_term",
            "category": category,
            "max_chars": 0,
            "overlap_terms": sorted(overlap_terms),
        }

    if strict_isolation and category == "daily":
        return {
            "used": False,
            "score": score,
            "reason": "route_blocked_strict_daily_main_rag",
            "category": category,
            "max_chars": 0,
            "overlap_terms": sorted(overlap_terms),
        }

    if strict_isolation and category == "robotics" and not lexical_match:
        return {
            "used": False,
            "score": score,
            "reason": "route_blocked_strict_robotics_weak_rag",
            "category": category,
            "max_chars": 0,
            "overlap_terms": sorted(overlap_terms),
        }

    if category == "mixed" and not lexical_match:
        return {
            "used": False,
            "score": score,
            "reason": "route_blocked_mixed_weak_rag",
            "category": category,
            "max_chars": 0,
            "overlap_terms": sorted(overlap_terms),
        }

    if category == "ambiguous" and route_confidence < 0.7:
        return {
            "used": False,
            "score": score,
            "reason": "route_blocked_ambiguous_low_confidence",
            "category": category,
            "max_chars": 0,
            "overlap_terms": sorted(overlap_terms),
        }

    if category == "daily":
        asks_robotics_relation = (
            _contains_phrase(question_text, ROBOTICS_DOMAIN_TERMS)
            or _contains_phrase(question_text, DOMAIN_RELATION_TERMS)
        )
        top_has_daily = _contains_phrase(top_text, DAILY_DOMAIN_TERMS)
        top_has_robotics = _contains_phrase(top_text, ROBOTICS_DOMAIN_TERMS)
        if top_has_robotics and not asks_robotics_relation:
            return {
                "used": False,
                "score": score,
                "reason": "blocked_daily_question_robotics_context",
                "category": category,
                "max_chars": 0,
                "overlap_terms": sorted(overlap_terms),
            }
        if not top_has_daily and not asks_robotics_relation:
            return {
                "used": False,
                "score": score,
                "reason": "blocked_daily_question_without_daily_context",
                "category": category,
                "max_chars": 0,
                "overlap_terms": sorted(overlap_terms),
            }
        return {
            "used": bool(lexical_match),
            "score": score,
            "reason": "daily_relevant_context" if lexical_match else "insufficient_daily_overlap",
            "category": category,
            "max_chars": MAIN_CONTEXT_CHAR_LIMIT if lexical_match else 0,
            "overlap_terms": sorted(overlap_terms),
        }

    if category == "ambiguous" and not _contains_phrase(question_text, ROBOTICS_NAMED_TERMS):
        return {
            "used": False,
            "score": score,
            "reason": "blocked_ambiguous_without_explicit_robotics_term",
            "category": category,
            "max_chars": 0,
            "overlap_terms": sorted(overlap_terms),
        }

    if category == "unknown":
        strong_match = top_overlap >= 3 or top_ratio >= 0.30 or named_robotics_match
        return {
            "used": bool(strong_match),
            "score": score,
            "reason": "unknown_strong_relevance" if strong_match else "blocked_unknown_weak_relevance",
            "category": category,
            "max_chars": MAIN_CONTEXT_CHAR_LIMIT if strong_match else 0,
            "overlap_terms": sorted(overlap_terms),
        }

    return {
        "used": bool(lexical_match),
        "score": score,
        "reason": "relevance_gate_passed" if lexical_match else "insufficient_query_overlap",
        "category": category,
        "max_chars": max_chars if lexical_match else 0,
        "overlap_terms": sorted(overlap_terms),
    }


def is_rag_relevant(question: str, retrieved_chunks: list[str]) -> bool:
    return bool(_rag_relevance_details(question, retrieved_chunks).get("used"))


def _split_text(text: str, *, chunk_size: int = 900) -> list[str]:
    clean_text = " ".join((text or "").split())
    if not clean_text:
        return []
    if len(clean_text) <= chunk_size:
        return [clean_text]

    chunks = []
    start = 0
    while start < len(clean_text):
        end = min(start + chunk_size, len(clean_text))
        if end < len(clean_text):
            boundary = clean_text.rfind(". ", start, end)
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        chunks.append(clean_text[start:end].strip())
        start = end
    return [chunk for chunk in chunks if chunk]


def _temporary_context_for_question(question: str) -> str:
    documents = session_store.get_temporary_documents()
    if not documents:
        return ""

    question_tokens = _tokens(question)
    document_triggered = bool(question_tokens.intersection(_TEMP_DOCUMENT_TRIGGERS))
    scored_chunks: list[tuple[int, int, str]] = []
    fallback_chunks: list[str] = []

    for doc_index, document in enumerate(documents):
        filename = str(document.get("filename") or "uploaded file")
        for chunk_index, chunk in enumerate(_split_text(str(document.get("text") or ""))):
            labelled_chunk = f"[{filename}]\n{chunk}"
            fallback_chunks.append(labelled_chunk)
            chunk_tokens = _tokens(chunk)
            overlap = len(question_tokens.intersection(chunk_tokens))
            if overlap:
                scored_chunks.append((overlap, -(doc_index * 1000 + chunk_index), labelled_chunk))

    selected = [
        chunk
        for _overlap, _order, chunk in sorted(scored_chunks, reverse=True)
    ]
    if not selected and document_triggered:
        selected = fallback_chunks[:3]
    if not selected:
        return ""

    context_parts = []
    total_chars = 0
    for chunk in selected:
        next_total = total_chars + len(chunk) + 2
        if next_total > TEMP_CONTEXT_CHAR_LIMIT and context_parts:
            break
        context_parts.append(chunk)
        total_chars = next_total
    return "\n\n".join(context_parts)[:TEMP_CONTEXT_CHAR_LIMIT]


def _detect_main_rag_path() -> str:
    if MAIN_RAG_INDEX_PATH.exists() and MAIN_RAG_CHUNKS_PATH.exists():
        return (
            f"{PROJECT_ROOT / 'rag'} "
            f"(index={MAIN_RAG_INDEX_PATH.name}, chunks={MAIN_RAG_CHUNKS_PATH.name})"
        )
    for relative_path in (
        "rag/retrieve.py",
        "rag/retriever.py",
        "src/rag/retrieve.py",
        "src/rag/retriever.py",
    ):
        path = PROJECT_ROOT / relative_path
        if path.exists():
            return str(path)
    rag_dir = PROJECT_ROOT / "rag"
    return str(rag_dir) if rag_dir.exists() else ""


def _module_source_path(module_name: str) -> Path | None:
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError, ValueError):
        return None
    origin = getattr(spec, "origin", None)
    if not origin or origin == "namespace":
        return None
    path = Path(origin)
    return path if path.exists() else None


def _huggingface_model_cached(model_name: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return False
    model_names = [model_name]
    if "/" not in model_name:
        model_names.append(f"sentence-transformers/{model_name}")
    for candidate_name in model_names:
        for filename in ("config.json", "model.safetensors", "pytorch_model.bin"):
            try:
                cached_path = try_to_load_from_cache(candidate_name, filename)
            except Exception:
                continue
            if cached_path and cached_path is not object():
                return Path(str(cached_path)).exists()
    return False


def _module_safe_to_import(module_name: str) -> tuple[bool, str | None]:
    path = _module_source_path(module_name)
    if path is None:
        return True, None
    try:
        source = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return True, None

    model_names = re.findall(r"SentenceTransformer\(\s*['\"]([^'\"]+)['\"]", source)
    for model_name in model_names:
        model_path = Path(model_name)
        if model_path.exists():
            continue
        if os.environ.get("VOICE_ASSISTANT_ALLOW_RAG_MODEL_IMPORT") != "1":
            return (
                False,
                f"{module_name}: skipped model-backed retriever to avoid loading {model_name}; "
                "using lightweight fallback when available",
            )
        if _huggingface_model_cached(model_name):
            continue
        return (
            False,
            f"{module_name}: skipped to avoid downloading uncached model {model_name}",
        )
    return True, None


def _normalize_retrieved_context(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        parts = []
        for key in ("context", "text", "content", "page_content", "documents", "results"):
            if key in value:
                parts.append(_normalize_retrieved_context(value[key]))
        return "\n\n".join(part for part in parts if part)
    if isinstance(value, (list, tuple)):
        return "\n\n".join(_normalize_retrieved_context(item) for item in value if item is not None)
    page_content = getattr(value, "page_content", None)
    if page_content:
        return str(page_content)
    return str(value)


def _load_main_retrieve():
    if _MAIN_RAG_CACHE["attempted"]:
        return _MAIN_RAG_CACHE["function"]

    _MAIN_RAG_CACHE["attempted"] = True
    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    previous_env = {
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
    }
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    errors = []
    try:
        for module_name, function_name in _MAIN_RAG_CANDIDATES:
            safe_to_import, skip_reason = _module_safe_to_import(module_name)
            if not safe_to_import:
                errors.append(skip_reason)
                continue
            try:
                module = importlib.import_module(module_name)
                function = getattr(module, function_name, None)
            except Exception as exc:
                errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
                continue
            if callable(function):
                _MAIN_RAG_CACHE.update({
                    "function": function,
                    "module": module_name,
                    "error": None,
                })
                return function
            errors.append(f"{module_name}: missing callable {function_name}")
        _MAIN_RAG_CACHE["error"] = "; ".join(str(error) for error in errors if error) or None
        return None
    finally:
        for key, previous_value in previous_env.items():
            if previous_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous_value


def _count_retrieved_results(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return 1 if value.strip() else 0
    if isinstance(value, dict):
        for key in ("results", "documents", "docs", "contexts", "chunks"):
            nested = value.get(key)
            if isinstance(nested, (list, tuple)):
                return len(nested)
        return 1 if _normalize_retrieved_context(value).strip() else 0
    if isinstance(value, (list, tuple)):
        return len(value)
    return 1 if _normalize_retrieved_context(value).strip() else 0


def _main_rag_source() -> str:
    module_name = _MAIN_RAG_CACHE.get("module")
    if module_name:
        source_path = _module_source_path(str(module_name))
        if source_path:
            return str(source_path)
        return str(module_name)
    return _detect_main_rag_path()


def _main_project_test_query_result_count() -> int:
    retrieve = _load_main_retrieve()
    if retrieve is None:
        _MAIN_RAG_CACHE["test_query_result_count"] = 0
        return 0

    previous_env = {
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
    }
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        try:
            retrieved = retrieve("robotics navigation", k=3)
        except TypeError:
            retrieved = retrieve("robotics navigation")
    except Exception as exc:
        _MAIN_RAG_CACHE["error"] = f"{type(exc).__name__}: {exc}"
        _MAIN_RAG_CACHE["test_query_result_count"] = 0
        return 0
    finally:
        for key, previous_value in previous_env.items():
            if previous_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous_value

    result_count = _count_retrieved_results(retrieved)
    _MAIN_RAG_CACHE["test_query_result_count"] = result_count
    return result_count


def _main_project_context_for_question(
    question: str,
    *,
    route_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    retrieve = _load_main_retrieve()
    if retrieve is None:
        return {
            "context": "",
            "used": False,
            "candidate_count": 0,
            "relevance_score": 0.0,
            "relevance_reason": "main_retriever_unavailable",
            "relevance_category": (route_info or {}).get("target_domain") or _infer_question_category(question),
            "max_chars": 0,
        }
    try:
        retrieved = retrieve(question, k=3)
    except TypeError:
        try:
            retrieved = retrieve(question)
        except Exception as exc:
            _MAIN_RAG_CACHE["error"] = f"{type(exc).__name__}: {exc}"
            return {
                "context": "",
                "used": False,
                "candidate_count": 0,
                "relevance_score": 0.0,
                "relevance_reason": "main_retriever_error",
                "relevance_category": (route_info or {}).get("target_domain") or _infer_question_category(question),
                "max_chars": 0,
            }
    except Exception as exc:
        _MAIN_RAG_CACHE["error"] = f"{type(exc).__name__}: {exc}"
        return {
            "context": "",
            "used": False,
            "candidate_count": 0,
            "relevance_score": 0.0,
            "relevance_reason": "main_retriever_error",
            "relevance_category": (route_info or {}).get("target_domain") or _infer_question_category(question),
            "max_chars": 0,
        }

    chunks = _retrieved_chunks(retrieved)
    relevance = _rag_relevance_details(question, chunks, route_info=route_info)
    if not relevance.get("used"):
        return {
            "context": "",
            "used": False,
            "candidate_count": len(chunks),
            "relevance_score": relevance.get("score", 0.0),
            "relevance_reason": relevance.get("reason", "irrelevant_context"),
            "relevance_category": relevance.get("category"),
            "max_chars": 0,
        }

    max_chars = int(relevance.get("max_chars") or MAIN_CONTEXT_CHAR_LIMIT)
    context = "\n\n".join(chunks)
    context = " ".join(context.split())[:max_chars]
    return {
        "context": context,
        "used": bool(context),
        "candidate_count": len(chunks),
        "relevance_score": relevance.get("score", 0.0),
        "relevance_reason": relevance.get("reason", "relevance_gate_passed"),
        "relevance_category": relevance.get("category"),
        "max_chars": max_chars if context else 0,
    }


def main_project_rag_status(*, validate_import: bool = False) -> dict[str, Any]:
    if validate_import:
        _load_main_retrieve()
        _main_project_test_query_result_count()
    detected_path = _detect_main_rag_path()
    return {
        "main_rag_available": callable(_MAIN_RAG_CACHE.get("function")),
        "main_rag_path_detected": detected_path,
        "main_rag_index_detected": MAIN_RAG_INDEX_PATH.exists(),
        "main_rag_chunks_detected": MAIN_RAG_CHUNKS_PATH.exists(),
        "main_rag_source": _main_rag_source(),
        "main_rag_module": _MAIN_RAG_CACHE.get("module"),
        "main_rag_read_only": True,
        "main_rag_test_query_result_count": int(_MAIN_RAG_CACHE.get("test_query_result_count") or 0),
        "main_rag_error": _MAIN_RAG_CACHE.get("error"),
        "main_rag_relevance_gate_enabled": MAIN_RAG_RELEVANCE_GATE_ENABLED,
        "main_rag_max_chars": MAIN_CONTEXT_CHAR_LIMIT,
        "main_rag_mixed_max_chars": MIXED_MAIN_CONTEXT_CHAR_LIMIT,
    }


def get_combined_context(question: str, route_info: dict[str, Any] | None = None) -> dict[str, Any]:
    route_info = dict(route_info or route_query(question))
    temporary_upload_context = _temporary_context_for_question(question)
    main_rag = _main_project_context_for_question(question, route_info=route_info)
    main_project_rag_context = str(main_rag.get("context") or "")
    parts = [
        temporary_upload_context,
        main_project_rag_context,
    ]
    combined_context = "\n\n".join(part for part in parts if part)[:COMBINED_CONTEXT_CHAR_LIMIT]
    return {
        "main_project_rag_context": main_project_rag_context,
        "temporary_upload_context": temporary_upload_context,
        "combined_context": combined_context,
        "main_rag_used": bool(main_rag.get("used") and main_project_rag_context),
        "temporary_rag_used": bool(temporary_upload_context),
        "main_rag_candidate_count": int(main_rag.get("candidate_count") or 0),
        "main_rag_relevance_score": float(main_rag.get("relevance_score") or 0.0),
        "main_rag_relevance_reason": main_rag.get("relevance_reason"),
        "main_rag_relevance_category": main_rag.get("relevance_category"),
        "main_rag_relevance_gate_enabled": MAIN_RAG_RELEVANCE_GATE_ENABLED,
        "main_rag_max_chars": int(main_rag.get("max_chars") or 0),
        "combined_context_chars": len(combined_context),
        "route_info": route_info,
    }


async def save_and_extract_upload(file, upload_dir: Path, *, max_chars: int = 5000) -> dict[str, Any]:
    original_name = file.filename or "upload.txt"
    suffix = Path(original_name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(
            "Unsupported file type. Upload PDF, TXT, CSV, DOCX, or JSON."
        )

    upload_dir.mkdir(parents=True, exist_ok=True)
    stored_name = safe_filename(original_name)
    stored_path = upload_dir / stored_name
    data = await file.read()
    stored_path.write_bytes(data)
    text = extract_text(stored_path, data, max_chars=max_chars)
    return {
        "original_filename": original_name,
        "stored_filename": stored_name,
        "path": stored_path,
        "text": text,
        "characters": len(text),
    }
