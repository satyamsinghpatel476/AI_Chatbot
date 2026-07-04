import json
import os
import re
import sys
import time

from intent_model import get_intent_classifier
from llm_runtime import LLMRuntimeError, chat, was_last_cache_used
from relationship_utils import correct_intent_label, detect_relationship_required
from research_core import (
    complete_sentence_response,
    select_facts,
    update_structured_memory,
)
from terminal_formatting import format_terminal_answer
from system_b.grounding import (
    append_routed_interaction,
    classifier_guidance,
    load_routed_interactions,
    select_intent_context,
)


CLASSIFIER = get_intent_classifier()

KNOWN_TECHNOLOGIES = {
    "a* algorithm",
    "adaptive monte carlo localization",
    "amcl",
    "blinkit",
    "differential drive robot",
    "ekf",
    "extended kalman filter",
    "gazebo",
    "google maps",
    "google pay",
    "imu",
    "instagram",
    "inverse kinematics",
    "kalman filter",
    "lidar",
    "lyft",
    "occupancy grid mapping",
    "odometry",
    "ola",
    "path planning",
    "phonepe",
    "pid",
    "pid control",
    "robot localization",
    "ros",
    "ros2",
    "sensor fusion",
    "slam",
    "spotify",
    "swiggy",
    "uber",
    "visual slam",
    "whatsapp",
    "wheel encoder",
    "zepto",
    "zomato",
}

SYSTEM_PROMPT = """You are System B, an intent-conditioned local assistant for
beginners.

The latest user message has priority over both classifier output and earlier
turns. Determine what the user is asking: definition, cause, procedure,
comparison, recommendation, troubleshooting, or relationship analysis. Answer
that task directly; repeating a generic definition of one keyword is not an
answer.

Use selected conversation context only when it is relevant. Keep consumer
services and robotics components distinct, but allow qualified indirect
relationships when the question supports one. For ambiguous requests, state
the most likely interpretation and ask one concise clarifying question, or give
short conditional answers for the main interpretations. For unfamiliar named
technology, do not invent specifications or acronym expansions.

For cross-domain questions, begin with a direct answer and distinguish:
- direct capability;
- indirect use of specifically relevant data;
- a conditional use that needs additional fields or permissions;
- no meaningful relationship.
Do not claim an indirect robotics benefit merely because one can be imagined.
Consumer ratings, transactions, social posts, or app services do not provide
physical robot sensor measurements. General questions asking for factors,
causes, or criteria can be answered generally and do not require named products.
Do not offer an unrelated integration, API workflow, or follow-up after you
have answered the question. Do not pivot from the requested robotics capability
to a different task such as route planning. If a question asks which consumer
app is best for a robotics function and that app category does not perform the
function, say that none is suitable.

Use two to six clear sentences and stay under 150 words. Do not mention the
classifier, prompts, memory files, hidden context, or evaluation. Do not begin
by describing what the user is asking; simply answer it."""


def _benchmark_force_llm():
    return (
        os.environ.get("BENCHMARK_FORCE_LLM") == "1"
        or os.environ.get("BENCHMARK_DISABLE_DETERMINISTIC_SHORTCUTS") == "1"
    )


def _extract_unfamiliar_named_technology(query):
    match = re.match(
        r"^\s*(?:what is|explain|describe|tell me about)\s+(.+?)[.!?]*$",
        query,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    candidate = match.group(1).strip()
    candidate = re.sub(
        r"\s+(?:in robotics|autonomous planning)\s*$",
        "",
        candidate,
        flags=re.IGNORECASE,
    ).strip()
    normalized = candidate.lower()
    normalized = re.sub(
        r"\s+(?:again|controllers?|middleware|sensors?)$",
        "",
        normalized,
    ).strip()
    if (
        candidate.lower() in KNOWN_TECHNOLOGIES
        or normalized in KNOWN_TECHNOLOGIES
    ):
        return None

    words = re.findall(r"[A-Za-z0-9*+-]+", candidate)
    title_words = [
        word
        for word in words
        if word[:1].isupper() and word.lower() not in {"the", "a", "an"}
    ]
    looks_named = bool(
        re.search(r"[a-z][A-Z]|[A-Za-z]+-[A-Za-z0-9-]+|[A-Z]{2,}", candidate)
        or len(title_words) >= 2
    )
    return candidate if looks_named else None


def _was_defined_by_user(entity, interactions):
    distinctive_terms = [
        term
        for term in re.findall(r"[a-z0-9*+-]+", entity.lower())
        if term not in {
            "again", "algorithm", "controller", "engine", "filter", "method",
            "model", "protocol", "sensor", "system",
        }
    ]
    if not distinctive_terms:
        return False

    distinctive = " ".join(distinctive_terms)
    definition_pattern = re.compile(
        rf"{re.escape(distinctive)}.{{0,50}}\b(?:means|is|refers to)\b",
        flags=re.IGNORECASE,
    )
    return any(
        definition_pattern.search(str(item.get("user", "")))
        for item in interactions
    )


def _relationship_guidance(query):
    q = query.lower()
    consumer_terms = [
        "app", "application", "ride-sharing", "ride-hailing", "food delivery",
        "online shopping", "restaurant rating", "social media", "uber",
        "driver", "delivery route", "google maps", "zomato", "swiggy", "ola",
        "whatsapp", "spotify", "instagram", "blinkit", "google pay",
        "phonepe", "zepto", "lyft",
    ]
    robotics_terms = [
        "robot", "slam", "localization", "mapping", "particle filter",
        "path planning", "path-planning", "obstacle avoidance", "perception",
        "sensor fusion", "pid", "ros", "ros2", "kalman", "ekf", "gazebo",
        "controller", "amcl", "odometry", "lidar", "robot state",
        "reinforcement learning",
    ]
    if not (
        any(term in q for term in consumer_terms)
        and any(term in q for term in robotics_terms)
    ):
        return ""

    if any(
        phrase in q
        for phrase in [
            "best for robot", "best for mapping", "instead of", "replace",
            "eliminate the need", "perform pid", "perform slam", "run ros",
            "run gazebo", "tune pid", "tune controller", "control a robot",
            "localize robot", "estimate robot state", "optimize robot navigation",
        ]
    ) or q.startswith("which"):
        return (
            "Required conclusion for this message: the consumer service cannot "
            "perform or replace the onboard robotics function, so none of those "
            "apps is suitable. Name the actual sensors or algorithms needed and "
            "do not invent an indirect integration."
        )
    if any(source in q for source in ["rating", "social media", "shopping data"]):
        return (
            "Required conclusion for this message: the proposed relationship is "
            "unsupported because ordinary ratings, posts, or shopping records "
            "lack physical geometry, pose, motion, and obstacle measurements."
        )
    if "particle filter" in q:
        return (
            "A ride-sharing service does not directly improve a localization "
            "particle filter. Only separately authorized, synchronized position "
            "observations with timestamps, a known coordinate frame, and measured "
            "uncertainty could be used as filter measurements. Traffic, route, or "
            "travel-time data must not be described as particle-filter updates."
        )
    if any(
        phrase in q
        for phrase in ["help", "improve", "train", "contribute", "inform"]
    ):
        return (
            "Treat the relationship as indirect and conditional only. State the "
            "exact authorized fields that could matter, such as coordinates, "
            "timestamps, travel times, road constraints, or outcome labels. Do "
            "not claim route records contain obstacle measurements, and state "
            "that onboard perception and safety sensing are still required."
        )
    return (
        "Compare the data the robotics task requires with what the consumer "
        "service ordinarily provides, then classify the relationship as direct, "
        "indirect, conditional, or unsupported."
    )


def _finalize_response(response, relationship_guidance):
    response = complete_sentence_response(
        response,
        max_words=100 if relationship_guidance else 140,
    )
    if not relationship_guidance:
        return response

    sentences = re.split(r"(?<=[.!?])\s+", response)
    filtered = []
    for sentence in sentences:
        lowered = sentence.lower()
        if sentence.rstrip().endswith("?"):
            continue
        if any(
            phrase in lowered
            for phrase in ["if you're interested", "i recommend", "an api"]
        ):
            continue
        filtered.append(sentence)
    cleaned = " ".join(filtered).strip()
    return cleaned or response


def chatbot_system_b(user_input, return_metadata=False):
    start = time.time()
    cache_used = False
    deterministic_path_used = False
    llm_called = False
    classified_intent, confidence = CLASSIFIER.predict(user_input)
    corrected_intent = correct_intent_label(user_input, classified_intent)
    relationship_required = detect_relationship_required(user_input)

    update_structured_memory("system_b", user_input)
    personal_facts = select_facts("system_b", user_input)
    interactions = load_routed_interactions(limit=12)
    selected = select_intent_context(
        interactions,
        classified_intent,
        current_user=user_input,
    )

    named_entity = _extract_unfamiliar_named_technology(user_input)
    if (
        named_entity
        and not _was_defined_by_user(named_entity, interactions)
        and not relationship_required
        and not _benchmark_force_llm()
    ):
        response = (
            f"I couldn't verify \"{named_entity}\" as an established technology. "
            "It may be project-specific, newly proposed, or a naming error. "
            "Please share its paper, repository, vendor page, or definition so "
            "I can explain it without inventing details."
        )
        deterministic_path_used = True
        append_routed_interaction(
            user_input,
            response,
            classified_intent,
        )
        result = {
            "response": response,
            "latency": time.time() - start,
            "intent": classified_intent,
            "classified_intent": classified_intent,
            "raw_predicted_intent": classified_intent,
            "corrected_predicted_intent": corrected_intent,
            "predicted_intent": corrected_intent,
            "intent_confidence": confidence,
            "selected_context_turns": 0,
            "generation_mode": "unsupported_entity_guardrail",
            "cache_used": cache_used,
            "deterministic_path_used": deterministic_path_used,
            "llm_called": llm_called,
            "relationship_required": relationship_required,
            "relationship_guidance_applied": False,
            "pipeline": [
                "user",
                "intent_classifier",
                "unsupported_entity_guardrail",
                "answer",
            ],
        }
        return result if return_metadata else response

    transcript = "\n".join(
        f"Prior user topic: {item.get('user', '')}"
        for item in selected
    )
    prompt_sections = [
        f"Classifier label: {classified_intent}",
        f"Classifier confidence: {confidence:.3f}",
        f"Classifier guidance: {classifier_guidance(classified_intent, confidence)}",
        (
            "Selected prior user topics (continuity hints only; they are not "
            f"evidence and must not override the latest message):\n"
            f"{transcript or '(none)'}"
        ),
    ]
    relationship_guidance = (
        _relationship_guidance(user_input)
        if relationship_required
        else ""
    )
    if relationship_guidance:
        prompt_sections.append(
            f"Cross-domain constraint for the latest message:\n"
            f"{relationship_guidance}"
        )
    if personal_facts:
        prompt_sections.append(
            "Potentially relevant structured personal facts:\n"
            + json.dumps(personal_facts, ensure_ascii=False)
        )
    if (
        named_entity
        and not _was_defined_by_user(named_entity, interactions)
        and not relationship_required
    ):
        prompt_sections.append(
            "Unverified named-technology guardrail:\n"
            f"Do not invent details for {named_entity}. If it is not known "
            "from the available context, say it cannot be verified."
        )
    prompt_sections.append(f"Latest user message:\n{user_input}")

    try:
        llm_called = True
        response = chat(
            SYSTEM_PROMPT,
            "\n\n".join(prompt_sections),
            temperature=0.25,
            seed=int(os.environ.get("EXPERIMENT_SEED", "0")) + 22,
            max_tokens=200,
            ensure_complete=False,
        )
        cache_used = was_last_cache_used()
        response = _finalize_response(response, relationship_guidance)
        generation_mode = "intent_conditioned_mistral"
    except LLMRuntimeError as exc:
        response = f"Model runtime error: {exc}"
        generation_mode = "runtime_error"

    append_routed_interaction(
        user_input,
        response,
        classified_intent,
    )
    result = {
        "response": response,
        "latency": time.time() - start,
        "intent": classified_intent,
        "classified_intent": classified_intent,
        "raw_predicted_intent": classified_intent,
        "corrected_predicted_intent": corrected_intent,
        "predicted_intent": corrected_intent,
        "intent_confidence": confidence,
        "selected_context_turns": len(selected),
        "generation_mode": generation_mode,
        "cache_used": cache_used,
        "deterministic_path_used": deterministic_path_used,
        "llm_called": llm_called,
        "relationship_required": relationship_required,
        "relationship_guidance_applied": bool(relationship_guidance),
        "pipeline": [
            "user",
            "intent_classifier",
            "intent_filtered_context",
            "structured_personal_memory",
            generation_mode,
            "answer",
        ],
    }
    return result if return_metadata else response


def _cli_value(value):
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


def _print_cli_response(result, metadata_mode):
    response = result.get("response", "") if isinstance(result, dict) else result
    print("\n------------------------------")
    print("System B")
    print("------------------------------")
    print(format_terminal_answer(response))

    if metadata_mode and isinstance(result, dict):
        predicted_intent = result.get("predicted_intent")
        if predicted_intent is None:
            predicted_intent = result.get("classified_intent") or result.get("intent")
        fields = [
            ("latency", result.get("latency")),
            ("resolved_domain", result.get("resolved_domain")),
            ("predicted_intent", predicted_intent),
            ("retrieved_sources", result.get("retrieved_sources")),
            ("pipeline", result.get("pipeline")),
        ]
        for label, value in fields:
            if value not in (None, "", []):
                print(f"{label}: {_cli_value(value)}")
    print()


def run_cli():
    metadata_mode = "--metadata" in sys.argv[1:]

    print("System B Chatbot started.")
    if metadata_mode:
        print("Metadata mode enabled.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        try:
            result = chatbot_system_b(
                user_input,
                return_metadata=True,
            ) if metadata_mode else chatbot_system_b(user_input)
            _print_cli_response(result, metadata_mode)
        except Exception as exc:
            print("\nSystem B:")
            print(f"Runtime error: {exc}\n")


if __name__ == "__main__":
    run_cli()
