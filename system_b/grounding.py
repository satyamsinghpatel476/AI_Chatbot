import os

from research_core import load_json, save_json


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTERACTION_FILE = os.path.join(ROOT_DIR, "memory", "system_b_interactions.json")

INTENT_GUIDANCE = {
    "robotics": (
        "The classifier sees a robotics-related request. Explain the requested "
        "mechanism, diagnosis, comparison, or procedure—not merely the named "
        "topic."
    ),
    "daily": (
        "The classifier sees an everyday-life or consumer-service request. "
        "Give practical, location-aware advice without claiming live data."
    ),
    "personal": (
        "The classifier sees a personal-memory request. Use supplied facts only "
        "when they directly answer the latest message."
    ),
    "mixed": (
        "The classifier sees more than one domain. Analyze the relationship "
        "between them explicitly; do not assume that two mentioned things are "
        "compatible or incompatible without explaining why."
    ),
}


def load_routed_interactions(limit=12):
    interactions = load_json(INTERACTION_FILE, [])
    if not isinstance(interactions, list):
        return []
    return interactions[-limit:]


def append_routed_interaction(user_text, assistant_text, intent, limit=24):
    interactions = load_json(INTERACTION_FILE, [])
    if not isinstance(interactions, list):
        interactions = []
    interactions.append({
        "user": user_text.strip(),
        "assistant": assistant_text.strip(),
        "intent": intent,
    })
    save_json(INTERACTION_FILE, interactions[-limit:])


def select_intent_context(interactions, current_intent, current_user=None):
    """Select prior turns by classifier labels rather than topic keywords."""
    if current_intent == "mixed":
        allowed = {"robotics", "daily", "personal", "mixed"}
    elif current_intent == "personal":
        allowed = {"personal", "mixed"}
    elif current_intent in {"robotics", "daily"}:
        allowed = {current_intent, "mixed"}
    else:
        allowed = set()

    normalized_current = (current_user or "").strip().lower()
    selected = [
        item for item in interactions
        if item.get("intent") in allowed
        and not str(item.get("assistant", "")).startswith("Model runtime error:")
        and (
            not normalized_current
            or str(item.get("user", "")).strip().lower() != normalized_current
        )
    ]
    return selected[-6:]


def classifier_guidance(intent, confidence):
    if confidence < 0.65:
        return (
            "The classifier is uncertain. Treat its label only as a weak hint "
            "and infer the user's actual request from the literal wording."
        )
    return INTENT_GUIDANCE.get(
        intent,
        "Use the literal request to determine the correct domain and task.",
    )
