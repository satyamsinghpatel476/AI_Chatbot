import json
import math
import os
import re
from collections import Counter


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_DIR = os.path.join(ROOT_DIR, "memory")
RAG_DOCUMENT_DIR = os.path.join(ROOT_DIR, "rag", "documents")
os.makedirs(MEMORY_DIR, exist_ok=True)

ROBOTICS_TERMS = {
    "robot", "robotics", "slam", "pid", "ros", "ros2", "gazebo",
    "localization", "mapping", "navigation", "path", "planning", "kalman",
    "ekf", "amcl", "lidar", "odometry", "imu", "encoder", "sensor",
    "controller", "control", "kinematics", "actuator", "middleware",
}
DAILY_TERMS = {
    "uber", "ola", "lyft", "ride", "taxi", "zomato", "swiggy", "food",
    "delivery", "maps", "spotify", "whatsapp", "instagram", "phonepe",
    "payment", "blinkit", "zepto", "groceries", "app",
}
STOPWORDS = {
    "a", "an", "and", "are", "about", "again", "can", "describe", "do",
    "explain", "for", "how", "i", "in", "is", "it", "me", "my", "of",
    "on", "please", "tell", "that", "the", "this", "to", "use", "what",
    "which", "with", "you",
}


def tokens(text):
    return re.findall(r"[a-z0-9*+-]+", text.lower())


def complete_sentence_response(text, max_words):
    """Keep a concise answer ending at a complete sentence boundary."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return text

    sentences = re.split(r"(?<=[.!?])\s+", text)
    selected = []
    word_count = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or sentence[-1] not in ".!?":
            continue
        if re.fullmatch(r"\d+[\.\)]", sentence):
            continue
        sentence_words = len(sentence.split())
        if selected and word_count + sentence_words > max_words:
            break
        if not selected and sentence_words > max_words:
            words = sentence.split()[:max_words]
            return " ".join(words).rstrip(",;:") + "."
        selected.append(sentence)
        word_count += sentence_words

    if selected:
        return " ".join(selected)

    words = text.split()[:max_words]
    return " ".join(words).rstrip(",;:") + "."


def infer_domain(text):
    if any(
        phrase in text.lower()
        for phrase in [
            "my name is", "i live in", "my favorite", "i study", "remember",
            "what is my name", "where do i live", "what do i study",
            "which robot do i like", "tell me about myself",
        ]
    ):
        return "personal"
    words = set(tokens(text))
    robotics = bool(words & ROBOTICS_TERMS)
    daily = bool(words & DAILY_TERMS)
    if robotics and daily:
        return "mixed"
    if robotics:
        return "robotics"
    if daily:
        return "daily"
    return "general"


def load_json(path, default):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def load_interactions(system_name, limit=12):
    path = os.path.join(MEMORY_DIR, f"{system_name}_interactions.json")
    interactions = load_json(path, [])
    return interactions[-limit:] if isinstance(interactions, list) else []


def append_interaction(system_name, user_text, assistant_text, limit=24):
    path = os.path.join(MEMORY_DIR, f"{system_name}_interactions.json")
    interactions = load_json(path, [])
    if not isinstance(interactions, list):
        interactions = []
    interactions.append({
        "user": user_text.strip(),
        "assistant": assistant_text.strip(),
    })
    save_json(path, interactions[-limit:])
    return interactions[-limit:]


def extract_personal_facts(text):
    patterns = [
        ("name", r"\bmy name is\s+(.+?)[.!?]*$"),
        ("location", r"\b(?:remember that )?i live in\s+(.+?)[.!?]*$"),
        ("study", r"\b(?:remember that )?i study\s+(.+?)[.!?]*$"),
        (
            "preferred_language",
            r"\b(?:remember that )?my preferred language is\s+(.+?)[.!?]*$",
        ),
        (
            "operating_system",
            r"\b(?:remember that )?my operating system is\s+(.+?)[.!?]*$",
        ),
    ]
    facts = {}
    for key, pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            facts[key] = match.group(1).strip(" .!?")

    favorite = re.search(
        r"\bmy favorite\s+([a-z ]+?)\s+is\s+(.+?)[.!?]*$",
        text,
        flags=re.IGNORECASE,
    )
    if favorite:
        facts[f"favorite_{favorite.group(1).strip().replace(' ', '_')}"] = (
            favorite.group(2).strip(" .!?")
        )
    return facts


def update_structured_memory(system_name, text):
    path = os.path.join(MEMORY_DIR, f"{system_name}_facts.json")
    facts = load_json(path, {})
    facts.update(extract_personal_facts(text))
    save_json(path, facts)
    return facts


def select_facts(system_name, query):
    path = os.path.join(MEMORY_DIR, f"{system_name}_facts.json")
    facts = load_json(path, {})
    q = query.lower()

    # For a new memory statement, expose only the fact stated in this turn.
    # Supplying every stored fact makes a simple acknowledgement drift into an
    # unnecessary biography and creates cross-topic contamination.
    stated_facts = extract_personal_facts(query)
    if stated_facts:
        return stated_facts

    requested_keys = set()
    if "name" in q:
        requested_keys.add("name")
    if "where do i live" in q or "my location" in q:
        requested_keys.add("location")
    if "what do i study" in q or "my study" in q:
        requested_keys.add("study")
    if "preferred language" in q:
        requested_keys.add("preferred_language")
    if "operating system" in q:
        requested_keys.add("operating_system")
    if "favorite app" in q:
        requested_keys.add("favorite_app")
    if "favorite robot" in q or "which robot do i like" in q:
        requested_keys.add("favorite_robot")
    if "about myself" in q:
        requested_keys.update(facts)

    if requested_keys:
        return {
            key: value
            for key, value in facts.items()
            if key in requested_keys
        }

    selected = {}
    for key, value in facts.items():
        readable_key = key.replace("_", " ")
        if any(word in q for word in readable_key.split()):
            selected[key] = value
    return selected


def clear_experiment_memory():
    for name in [
        "system_a_interactions.json",
        "system_b_interactions.json",
        "system_b_facts.json",
        "system_c_facts.json",
    ]:
        save_json(
            os.path.join(MEMORY_DIR, name),
            [] if "interactions" in name else {},
        )


def _document_chunks():
    chunks = []
    if not os.path.isdir(RAG_DOCUMENT_DIR):
        return chunks
    for filename in sorted(os.listdir(RAG_DOCUMENT_DIR)):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(RAG_DOCUMENT_DIR, filename)
        domain = (
            "daily"
            if filename.startswith("daily_")
            else "robotics"
        )
        with open(path, encoding="utf-8") as handle:
            sections = re.split(r"\n\s*\n", handle.read())
        for section in sections:
            section = section.strip()
            if section:
                chunks.append({
                    "source": filename,
                    "domain": domain,
                    "text": section,
                })
    return chunks


DOCUMENT_CHUNKS = _document_chunks()
_RETRIEVAL_CACHE = {}


def retrieve_local_knowledge(query, k=3, domain=None):
    """Small deterministic BM25-style retriever for the local research corpus."""
    cache_key = (" ".join(str(query or "").lower().split()), int(k), domain)
    if os.environ.get("DISABLE_RETRIEVAL_CACHE") != "1" and cache_key in _RETRIEVAL_CACHE:
        return [dict(item) for item in _RETRIEVAL_CACHE[cache_key]]

    query_terms = [term for term in tokens(query) if term not in STOPWORDS]
    if not query_terms or not DOCUMENT_CHUNKS:
        return []

    candidates = [
        item for item in DOCUMENT_CHUNKS
        if domain not in {"robotics", "daily"}
        or item.get("domain") == domain
    ]
    if not candidates:
        candidates = DOCUMENT_CHUNKS

    document_tokens = [tokens(item["text"]) for item in candidates]
    document_frequency = Counter()
    for terms in document_tokens:
        document_frequency.update(set(terms))

    scored = []
    total_documents = len(candidates)
    for item, terms in zip(candidates, document_tokens):
        counts = Counter(terms)
        score = 0.0
        for term in query_terms:
            if not counts[term]:
                continue
            inverse_frequency = math.log(
                1 + (total_documents - document_frequency[term] + 0.5)
                / (document_frequency[term] + 0.5)
            )
            score += inverse_frequency * counts[term] / (counts[term] + 1.2)
        normalized_query = " ".join(query_terms)
        normalized_text = " ".join(terms)
        if normalized_query and normalized_query in normalized_text:
            score += 2.0
        if score > 0:
            scored.append({**item, "score": score})

    scored.sort(key=lambda item: item["score"], reverse=True)
    result = scored[:k]
    if os.environ.get("DISABLE_RETRIEVAL_CACHE") != "1":
        _RETRIEVAL_CACHE[cache_key] = [dict(item) for item in result]
    return result
