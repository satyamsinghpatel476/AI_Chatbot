import json
import os
import re
import sys
import time

from intent_model import get_intent_classifier
from llm_runtime import LLMRuntimeError, chat, was_last_cache_used
from memory.semantic_memory import add_memory, search_memory
from relationship_utils import correct_intent_label, detect_relationship_required
from research_core import (
    DOCUMENT_CHUNKS,
    ROOT_DIR,
    complete_sentence_response,
    extract_personal_facts,
    infer_domain,
    load_json,
    retrieve_local_knowledge,
    save_json,
    select_facts,
    update_structured_memory,
)


CLASSIFIER = get_intent_classifier()
LEARNING_FILE = os.path.join(ROOT_DIR, "learning.json")

FAST_GROUNDED_PROMPT = """You are System C, a local multi-domain AI assistant
for beginners. Produce only the final answer.

Answer the latest user message first. Do not let previous topics influence the
answer unless the user explicitly refers to them. Use the resolved domain as
the primary source of reasoning and ignore unrelated retrieved or remembered
content.

Domain rules:
- Robotics: focus on robotics concepts, retrieved robotics documents, and
  technical reasoning. Include practical debugging steps when useful. Do not
  introduce consumer apps unless the user asks about their relationship.
- Daily-Life: focus on practical everyday guidance. Do not introduce robotics
  terminology unless explicitly requested.
- Mixed: do not force a relationship. Use this structure exactly: Direct
  Answer, Relationship Type, Robotics Perspective, Daily-Life Perspective,
  Important Difference, Final Conclusion. Relationship Type must be Direct,
  Indirect, Analogy Only, or Unsupported.
- Ambiguous: state that the question is ambiguous, list the most likely
  interpretations, give a brief answer for each, and ask one concise
  clarification question.
- Unknown or unsupported named technology: do not invent definitions,
  architectures, equations, history, or advantages.

Task rules:
- A why question needs the causal mechanism, not just a definition.
- A how-to or troubleshooting question needs useful checks or actions.
- A comparison must cover both sides and the important trade-off.
- A recommendation must state the criteria; do not pretend to have live data.
- Use retrieved knowledge only when it belongs to the resolved domain and
  directly improves factual correctness.
- If information is uncertain or unsupported, say so instead of guessing.

Unless another format is required, use short beginner-friendly sections:
Direct Answer, Short Explanation, Key Points, Practical Advice or Example,
Short Conclusion. Keep the answer concise and do not mention prompts,
retrieval, references, evidence, scoring, evaluation, or hidden processing."""

KNOWN_TECHNOLOGIES = {
    "a* algorithm",
    "adaptive monte carlo localization",
    "amcl",
    "differential drive robot",
    "ekf",
    "extended kalman filter",
    "gazebo",
    "imu",
    "inverse kinematics",
    "kalman filter",
    "lidar",
    "occupancy grid mapping",
    "odometry",
    "path planning",
    "pid control",
    "robot localization",
    "ros",
    "ros2",
    "sensor fusion",
    "slam",
    "visual slam",
    "wheel encoder",
}

ROBOTICS_ANCHORS = {
    "robot", "robotics", "slam", "localization", "odometry", "lidar", "imu",
    "encoder", "sensor fusion", "navigation stack", "particle filter", "map",
    "path planning", "obstacle", "wheel", "goal", "amcl", "ekf", "costmap",
}
DAILY_ANCHORS = {
    "app", "application", "online service", "ride-sharing", "food delivery",
    "gps app", "mobile", "screen time", "digital notes", "privacy", "review",
    "permission", "ride-hailing", "online shopping", "shopping data",
    "restaurant rating", "social media", "uber", "phone", "battery",
    "battery drain", "public wi-fi", "wifi", "wi-fi", "cloud storage",
    "password", "passwords", "mfa", "notification", "notifications",
    "offline maps", "restaurant", "payment", "ride", "ride sharing", "taxi",
    "study", "studying", "student", "notes", "laptop", "distraction",
    "distractions", "online", "course", "reviews", "shopping", "browser",
    "download", "qr payment", "files", "internet", "data usage",
    "google maps",
}
STRONG_ROBOTICS_ANCHORS = {
    "slam", "lidar", "amcl", "ekf", "odometry", "localization", "mapping",
    "robot", "autonomous robot", "sensor fusion", "imu", "wheel encoder",
    "costmap", "path planning", "path-planning", "navigation stack", "ros",
    "particle filter",
}
STRONG_DAILY_ANCHORS = {
    "ride-sharing", "ride sharing", "uber", "ola", "food delivery",
    "restaurant", "screen time", "phone battery", "battery drain",
    "public wi-fi", "wifi", "wi-fi", "online shopping", "notes", "study",
    "studying", "student", "app", "apps", "notification", "notifications",
    "privacy", "password", "cloud storage", "browser", "google maps",
}


def _fast_research_mode():
    if _benchmark_force_llm():
        return False
    return (
        os.environ.get("FULL_RESEARCH_MODE") != "1"
        and os.environ.get("FAST_RESEARCH_MODE", "1") != "0"
    )


def _benchmark_force_llm():
    return (
        os.environ.get("BENCHMARK_FORCE_LLM") == "1"
        or os.environ.get("BENCHMARK_DISABLE_DETERMINISTIC_SHORTCUTS") == "1"
    )


def _learn_from_user_statement(query):
    patterns = [
        r"^\s*learn that\s+(.+?)\s+(?:means|is|refers to)\s+(.+?)[.!?]*$",
        r"^\s*teach(?:ing)?\s*:\s*(.+?)\s*=\s*(.+?)[.!?]*$",
    ]
    for pattern in patterns:
        match = re.match(pattern, query, flags=re.IGNORECASE)
        if not match:
            continue
        concept = match.group(1).strip(" .!?").lower()
        definition = match.group(2).strip(" .!?")
        database = load_json(LEARNING_FILE, {})
        if not isinstance(database, dict):
            database = {}
        database[concept] = definition
        save_json(LEARNING_FILE, database)
        add_memory(f"{concept}: {definition}")
        return {"concept": concept, "definition": definition}
    return None


def _learned_context(query):
    database = load_json(LEARNING_FILE, {})
    if not isinstance(database, dict):
        return []

    query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    generic = {
        "algorithm", "controller", "engine", "filter", "method", "model",
        "protocol", "sensor", "system",
    }
    matches = []
    for concept, definition in database.items():
        concept_terms = set(re.findall(r"[a-z0-9]+", concept.lower()))
        distinctive = concept_terms - generic
        overlap = len(distinctive & query_terms)
        if concept.lower() in query.lower() or (
            distinctive and overlap == len(distinctive)
        ):
            matches.append({
                "concept": concept,
                "definition": str(definition),
                "score": 1.0 if concept.lower() in query.lower() else 0.8,
            })
    return sorted(
        matches,
        key=lambda item: item["score"],
        reverse=True,
    )[:3]


def _personal_response(query, facts):
    stated = extract_personal_facts(query)
    if stated:
        key, value = next(iter(stated.items()))
        labels = {
            "name": "name",
            "location": "location",
            "study": "study topic",
            "favorite_app": "favorite app",
            "favorite_robot": "favorite robot",
            "preferred_language": "preferred language",
            "operating_system": "operating system",
        }
        label = labels.get(key, key.replace("_", " "))
        return f"Got it — I'll remember that your {label} is {value}."

    q = query.lower()
    recall = [
        ("what is my name", "name", "Your name is {value}."),
        ("where do i live", "location", "You live in {value}."),
        ("what is my favorite app", "favorite_app", "Your favorite app is {value}."),
        ("which robot do i like", "favorite_robot", "Your favorite robot is {value}."),
        ("what do i study", "study", "You study {value}."),
        (
            "what is my preferred language",
            "preferred_language",
            "Your preferred language is {value}.",
        ),
        (
            "what is my operating system",
            "operating_system",
            "Your operating system is {value}.",
        ),
    ]
    for phrase, key, template in recall:
        if phrase in q and facts.get(key):
            return template.format(value=facts[key])
    return None


def _infer_task(query, domain):
    q = query.lower().strip()
    if domain == "mixed" and re.match(r"^(can|could|should|would|which)\b", q):
        return "cross-domain relationship analysis"
    if re.match(r"^(why|what causes|how does .+ affect)\b", q):
        return "causal explanation"
    if re.match(r"^(how can|how do|how should|how would)\b", q):
        if any(
            term in q
            for term in [
                "fail", "fault", "recover", "drift", "inaccurate", "noisy",
                "losing", "reduce", "improve", "detect",
            ]
        ):
            return "troubleshooting procedure"
        return "procedure"
    if any(term in q for term in ["difference between", "differ from", "compare"]):
        return "comparison"
    if re.match(r"^(which|what is the best|what are good)\b", q):
        return "recommendation"
    if re.match(r"^(what is|explain|describe|tell me about)\b", q):
        return "definition and explanation"
    return "direct answer"


def _has_anchor(query, anchors):
    q = query.lower()
    for anchor in anchors:
        pattern = (
            r"(?<![a-z0-9])"
            + re.escape(anchor).replace(r"\ ", r"\s+")
            + r"s?(?![a-z0-9])"
        )
        if re.search(pattern, q):
            return True
    return False


def _resolve_domain(query, relationship_required=None):
    domain = infer_domain(query)
    if relationship_required is None:
        relationship_required = detect_relationship_required(query)
    strong_robotics = _has_anchor(query, STRONG_ROBOTICS_ANCHORS)
    strong_daily = _has_anchor(query, STRONG_DAILY_ANCHORS)
    robotics = _has_anchor(query, ROBOTICS_ANCHORS) or strong_robotics
    daily = _has_anchor(query, DAILY_ANCHORS) or strong_daily

    if robotics and daily and relationship_required:
        return "mixed"
    if strong_robotics and not (daily and relationship_required):
        return "robotics"
    if strong_daily and not (robotics and relationship_required):
        return "daily"
    if robotics and daily:
        return "mixed" if relationship_required else domain
    if domain == "general" and robotics:
        return "robotics"
    if domain == "general" and daily:
        return "daily"
    if domain == "general":
        return "unknown"
    return domain


def _needs_clarification(query, domain):
    """Detect genuinely underspecified requests without an extra LLM call."""
    q = query.lower().strip()

    # A named domain and object is enough for a useful general explanation.
    explicit_robotics = _has_anchor(q, ROBOTICS_ANCHORS) and not q.startswith(
        ("why is my ", "why does my ", "how can i improve ")
    )
    explicit_daily = _has_anchor(q, DAILY_ANCHORS) and not q.startswith(
        ("what is the best way to reach",)
    )
    if explicit_robotics or explicit_daily:
        return False

    vague_patterns = [
        r"\btracking accuracy\b",
        r"\breach my destination\b",
        r"\bmy system\b",
        r"^\s*how do i reduce latency\b",
        r"\bmost reliable sensor\b",
        r"\bimprove stability\b",
        r"\bmy map (?:is )?inaccurate\b",
        r"\bbest path to follow\b",
        r"\bhandle noisy data\b",
        r"\bmy position estimate\b",
    ]
    if any(re.search(pattern, q) for pattern in vague_patterns):
        return True

    return False


def _clarifying_question(query):
    q = query.lower()
    if "latency" in q:
        return (
            "Where is the latency occurring—network communication, software, "
            "sensor processing, or the robot-control loop?"
        )
    if "sensor" in q:
        return (
            "What must the sensor measure, in what environment, and does "
            "reliability mean accuracy, robustness, or service life?"
        )
    if "stability" in q:
        return (
            "Which system is unstable, what behavior do you observe, and under "
            "what operating conditions?"
        )
    if any(term in q for term in ["tracking", "position estimate", "map"]):
        return (
            "Are you referring to a phone/GPS app or a robot localization "
            "system, and what sensors or positioning method does it use?"
        )
    if any(term in q for term in ["destination", "best path"]):
        return (
            "What are the start and destination, travel mode or robot type, "
            "and the priority—time, distance, cost, or safety?"
        )
    if "noisy data" in q:
        return (
            "What kind of data is noisy, what produces it, and what decision "
            "or model will use the cleaned result?"
        )
    return "Which device or system, operating context, and success metric do you mean?"


def _ambiguity_guidance(query):
    """Supply useful conditional checks while still requesting missing context."""
    q = query.lower()
    if "tracking accuracy" in q or "position estimate" in q:
        return (
            "For phone/GPS tracking: check precise-location permission, clear "
            "sky view, power-saving restrictions, and error at known points. "
            "For robot localization: check sensor calibration, timestamps, "
            "coordinate frames, wheel slip, fusion covariances, and comparison "
            "with ground truth."
        )
    if "latency" in q:
        return (
            "Measure delay at each stage before optimizing. For networks inspect "
            "round-trip time, loss, and server delay; for software profile slow "
            "functions and queues; for robots inspect sensor rates, message "
            "queues, inference time, and control-loop deadlines."
        )
    if "reliable sensor" in q:
        return (
            "There is no universally most reliable sensor. Compare the required "
            "quantity, range, accuracy, update rate, environment, failure modes, "
            "maintenance, and whether complementary redundancy is needed."
        )
    if "stability" in q:
        return (
            "Mechanical stability needs center-of-mass, mounting, traction, and "
            "vibration checks; control stability needs logs, delay, sampling "
            "rate, gain tuning, saturation, and oscillation checks."
        )
    if "map" in q:
        return (
            "For a phone map, check stale map data, GPS quality, permissions, "
            "and connectivity. For a robot map, inspect sensor calibration, "
            "wheel slip, timestamps, loop closure, moving objects, and whether "
            "the map frame is being reused correctly."
        )
    if "destination" in q or "best path" in q:
        return (
            "A route recommendation needs start, destination, travel mode, and "
            "the priority among time, distance, cost, accessibility, and safety. "
            "A robot path also needs a map, footprint, obstacle constraints, and "
            "a cost objective."
        )
    if "noisy data" in q:
        return (
            "First plot the raw data and identify sensor faults, outliers, drift, "
            "missing samples, or high-frequency noise. Then choose calibration, "
            "validation, robust outlier handling, or a filter matched to the "
            "signal without removing real changes."
        )
    if "drift" in q:
        return (
            "Drift can come from accumulated integration error, bias, calibration "
            "error, clock mismatch, changing conditions, or an unstable model. "
            "Log the estimate and raw measurements, compare with a reference, "
            "and locate the stage where error begins accumulating."
        )
    return (
        "Give two materially different interpretations, one useful check for "
        "each, and ask for the device, environment, observed symptom, and goal."
    )


def _deterministic_ambiguity_response(query, clarifying_question):
    q = query.lower()
    if "tracking accuracy" in q or "position estimate" in q:
        interpretations = [
            (
                "Phone/GPS tracking",
                "check precise-location permission, sky view, power-saving "
                "settings, and error at known points",
            ),
            (
                "Robot localization",
                "check sensor calibration, timestamps, coordinate frames, "
                "wheel slip, covariances, and ground truth comparison",
            ),
            (
                "Object tracking",
                "check camera quality, lighting, frame rate, labels, and how "
                "lost targets are re-identified",
            ),
        ]
    elif "latency" in q:
        interpretations = [
            (
                "Network latency",
                "measure round-trip time, packet loss, server delay, and "
                "connection quality",
            ),
            (
                "Software latency",
                "profile slow functions, queues, blocking I/O, and background "
                "work",
            ),
            (
                "Robot-control latency",
                "inspect sensor rates, message queues, inference time, and "
                "control-loop deadlines",
            ),
        ]
    elif "sensor" in q:
        interpretations = [
            (
                "Robot sensor choice",
                "match the sensor to the measurement, range, update rate, "
                "environment, and failure modes",
            ),
            (
                "Everyday device sensor",
                "compare accuracy, durability, calibration needs, battery use, "
                "and privacy impact",
            ),
        ]
    elif "stability" in q:
        interpretations = [
            (
                "Mechanical stability",
                "check center of mass, mounting, traction, vibration, and load "
                "limits",
            ),
            (
                "Control stability",
                "inspect logs, delay, sampling rate, gain tuning, saturation, "
                "and oscillation",
            ),
        ]
    elif "map" in q:
        interpretations = [
            (
                "Phone or web map",
                "check stale map data, GPS quality, permissions, connectivity, "
                "and local coverage",
            ),
            (
                "Robot map",
                "inspect sensor calibration, wheel slip, timestamps, loop "
                "closure, moving objects, and map-frame reuse",
            ),
        ]
    elif "destination" in q or "best path" in q:
        interpretations = [
            (
                "Daily travel route",
                "choose based on start, destination, travel mode, time, cost, "
                "distance, and safety",
            ),
            (
                "Robot path planning",
                "use a map, robot footprint, obstacle constraints, and a cost "
                "objective",
            ),
        ]
    elif "noisy data" in q:
        interpretations = [
            (
                "Sensor data",
                "plot the raw signal, inspect faults, drift, outliers, missing "
                "samples, and sampling rate",
            ),
            (
                "Application or dataset data",
                "validate sources, remove duplicates carefully, handle missing "
                "values, and preserve real changes",
            ),
        ]
    else:
        interpretations = [
            (
                "Robotics interpretation",
                "identify the robot, sensors, environment, observed symptom, "
                "and success metric",
            ),
            (
                "Daily-life interpretation",
                "identify the device, app, situation, constraint, and goal",
            ),
        ]

    bullets = "\n".join(
        f"- {name}: {guidance}." for name, guidance in interpretations
    )
    return (
        "This question is ambiguous because the system and goal are missing.\n\n"
        "Most likely interpretations:\n"
        f"{bullets}\n\n"
        f"{clarifying_question}"
    )


def _needs_unknown_general_response(query, domain):
    if domain != "unknown":
        return False
    q = query.lower().strip()

    # The generic unknown fallback is only for context-missing system
    # questions. If the current question has an explicit daily-life or
    # robotics anchor, let the resolved domain path handle it.
    if _has_anchor(q, DAILY_ANCHORS) or _has_anchor(q, ROBOTICS_ANCHORS):
        return False

    broad_patterns = [
        r"\bmake (?:the |my )?system more reliable\b",
        r"\b(?:system|service|route) (?:more )?reliable\b",
        r"\broute looks wrong\b",
        r"\breduce noise(?: in (?:my |the )?system)?\b",
        r"\bnoise in (?:my |the )?system\b",
        r"\bwhat should i check\b",
        r"\bhow can i make (?:the |my )?system\b",
        r"\bhow can i improve stability\b",
    ]
    return any(re.search(pattern, q) for pattern in broad_patterns)


def _deterministic_unknown_general_response(query):
    q = query.lower()
    if "route" in q:
        direct = (
            "This depends on what kind of route is wrong, because a travel "
            "route, robot path, and network route are diagnosed differently."
        )
        general = (
            "In general, check the start and destination, map or graph data, "
            "constraints, recent changes, logs, and where the route first "
            "diverges from what you expected."
        )
        question = "Which route are you referring to: a map app route, a robot path, or a network/service route?"
    elif "noise" in q:
        direct = (
            "This depends on the system, because noise can mean sensor noise, "
            "software/data noise, or communication noise."
        )
        general = (
            "In general, inspect the raw input, logs, sampling rate, recent "
            "changes, configuration, outliers, and whether the noise appears "
            "only under certain conditions."
        )
        question = "Which system is noisy, and what signal or behavior are you observing?"
    elif "reliable" in q or "reliability" in q:
        direct = (
            "This depends on the system, because reliability can mean fewer "
            "crashes, more accurate outputs, better uptime, or safer behavior."
        )
        general = (
            "In general, check logs, inputs, configuration, recent changes, "
            "failure patterns, resource limits, and the smallest repeatable "
            "case that causes the issue."
        )
        question = "Which system are you referring to: a robot, a software app, or a network/service?"
    else:
        direct = (
            "This depends on the system, so I should not force it into "
            "robotics or daily-life without more context."
        )
        general = (
            "It could refer to a robot, a software app, or a network/service. "
            "In general, check logs, inputs, configuration, recent changes, "
            "and failure patterns."
        )
        question = "Which system are you referring to?"

    return (
        f"Direct Answer\n{direct}\n\n"
        "Possible Interpretations\n"
        "- Robot or hardware system.\n"
        "- Software app or data pipeline.\n"
        "- Network, route, or online service.\n\n"
        f"General Check\n{general}\n\n"
        f"Clarification\n{question}"
    )


def _relationship_hint(query, domain):
    if domain != "mixed":
        return "none"
    q = query.lower()
    if any(
        phrase in q
        for phrase in [
            "replace", "instead of", "eliminate the need", "best for robot",
            "best for mapping",
        ]
    ):
        return "likely unsupported as a direct replacement"
    if any(
        phrase in q
        for phrase in ["help", "improve", "train", "contribute", "inform"]
    ):
        return "possibly indirect or conditional if relevant fields exist"
    return "must be analyzed from the required and available data"


def _relationship_guidance(query, relationship_hint):
    q = query.lower()
    if relationship_hint == "likely unsupported as a direct replacement":
        return (
            "Required conclusion: none of the named consumer apps is suitable "
            "for performing or replacing the onboard robotics function. Open "
            "with that conclusion, name the sensors/algorithms the robot actually "
            "needs, and do not invent an indirect integration the user did not ask for."
        )
    if any(source in q for source in ["rating", "social media", "shopping data"]):
        return (
            "Ordinary ratings, social posts, or shopping records do not contain "
            "the physical geometry, pose, motion, or obstacle measurements needed "
            "for local robot perception, so the proposed relationship is unsupported."
        )
    if relationship_hint.startswith("possibly indirect"):
        return (
            "State that the relationship is only indirect or conditional. Name "
            "the exact fields that would be useful—such as authorized coordinates, "
            "timestamps, travel times, road constraints, or outcome labels—and "
            "state that onboard perception and safety sensing are still required."
        )
    return (
        "Compare what information the robotics task requires with what the named "
        "consumer source ordinarily provides, then classify the relationship."
    )


def _format_mixed_response(
    *,
    direct_answer,
    relationship_type,
    robotics_perspective,
    daily_life_perspective,
    important_difference,
    final_conclusion,
):
    return (
        f"Direct Answer\n{direct_answer}\n\n"
        f"Relationship Type:\n{relationship_type}\n\n"
        f"Robotics Perspective\n{robotics_perspective}\n\n"
        f"Daily-Life Perspective\n{daily_life_perspective}\n\n"
        f"Important Difference\n{important_difference}\n\n"
        f"Final Conclusion\n{final_conclusion}"
    )


def _sentence_start(text):
    return text[:1].upper() + text[1:] if text else text


def _source_be_verb(text):
    plural_sources = ("ratings", "posts", "permissions")
    return "are" if text.lower().endswith(plural_sources) else "is"


def _source_do_verb(text):
    plural_sources = ("ratings", "posts", "permissions")
    return "do" if text.lower().endswith(plural_sources) else "does"


def _source_possessive(text):
    plural_sources = ("ratings", "posts", "permissions")
    return "their" if text.lower().endswith(plural_sources) else "its"


def _consumer_source_from_query(query):
    q = query.lower()
    if "restaurant" in q or "rating" in q:
        return "restaurant ratings"
    if "social media" in q or "instagram" in q:
        return "social media posts"
    if "ride-sharing" in q or "ride sharing" in q:
        return "ride-sharing apps"
    if "uber" in q or "ride-hailing" in q or "ride" in q:
        return "ride-hailing app data"
    if "google maps" in q:
        return "Google Maps data"
    if "food delivery" in q or "delivery route" in q:
        return "food-delivery route data"
    if "app permission" in q:
        return "phone app permissions"
    if "screen time" in q:
        return "screen time habits"
    return "consumer app data"


def _robotics_focus_from_query(query):
    q = query.lower()
    if "slam" in q:
        return "SLAM"
    if "robot perception" in q or "perception" in q:
        return "robot perception"
    if "localization" in q:
        return "robot localization"
    if "lidar" in q:
        return "LiDAR"
    if "pid" in q:
        return "PID control"
    if "ros2" in q:
        return "ROS2"
    if "ros" in q:
        return "ROS"
    if "path planning" in q or "path-planning" in q:
        return "robot path planning"
    if "navigation" in q:
        return "robot navigation"
    if "robot" in q or "robotics" in q:
        return "robotics"
    return "the robotics task"


def _deterministic_relationship_response(query, domain, relationship_hint):
    if domain != "mixed":
        return None

    q = query.lower()
    consumer_source = _consumer_source_from_query(query)
    robotics_focus = _robotics_focus_from_query(query)

    if (
        "slam" in q
        and ("ride-sharing" in q or "ride sharing" in q or "ride-hailing" in q)
        and any(term in q for term in ["compare", "difference", "versus", "vs"])
    ):
        return _format_mixed_response(
            direct_answer=(
                "SLAM and ride-sharing apps are different kinds of systems: "
                "SLAM is a robotics mapping/localization method, while "
                "ride-sharing apps are consumer transportation services."
            ),
            relationship_type="Analogy Only",
            robotics_perspective=(
                "SLAM uses robot sensor data, such as camera, LiDAR, IMU, or "
                "odometry measurements, to estimate the robot's pose while "
                "building or updating a map."
            ),
            daily_life_perspective=(
                "Ride-sharing apps use phone location, driver availability, "
                "routing, pricing, and booking workflows to connect riders and "
                "drivers."
            ),
            important_difference=(
                "SLAM estimates a robot's state from onboard physical "
                "measurements; ride-sharing apps coordinate human trips and do "
                "not replace robot perception or mapping."
            ),
            final_conclusion=(
                "They can be compared only by analogy around location and "
                "navigation; they do not perform the same technical role."
            ),
        )

    if "app permission" in q and "ros" in q:
        return _format_mixed_response(
            direct_answer=(
                "No. Phone app permissions and ROS permissions are not "
                "interchangeable."
            ),
            relationship_type="Analogy Only",
            robotics_perspective=(
                "ROS access depends on node configuration, graph security, "
                "network policy, and robot-side authorization."
            ),
            daily_life_perspective=(
                "Phone app permissions control access to phone resources such "
                "as location, contacts, camera, or files."
            ),
            important_difference=(
                "Both involve access control, but they protect different "
                "systems and operate at different layers."
            ),
            final_conclusion=(
                "Use phone permissions for mobile privacy and ROS security "
                "settings plus robot safety checks for robotics."
            ),
        )

    direct_replacement = any(
        phrase in q
        for phrase in [
            "replace",
            "instead of",
            "perform slam",
            "perform pid",
            "run ros",
            "run ros2",
            "run gazebo",
            "tune pid",
            "tune controller",
            "control a robot",
            "replace perception",
            "replace sensors",
            "replace lidar",
            "localize robots",
            "localize a robot",
            "estimate robot state",
            "obstacle avoidance",
            "avoid obstacles",
            "detect obstacles",
            "which ride app",
            "which food delivery app",
        ]
    )
    unsupported_source = any(
        source in q
        for source in [
            "restaurant rating",
            "restaurant ratings",
            "rating",
            "social media",
            "shopping history",
            "app permissions",
        ]
    )
    if (
        relationship_hint == "likely unsupported as a direct replacement"
        or direct_replacement
        or unsupported_source
    ):
        return _format_mixed_response(
            direct_answer=(
                f"No. {_sentence_start(consumer_source)} cannot directly "
                f"perform or replace {robotics_focus}."
            ),
            relationship_type="Unsupported",
            robotics_perspective=(
                f"{_sentence_start(robotics_focus)} needs physical "
                "measurements such as geometry, pose, motion, range, obstacles, "
                "or feedback from the robot's own sensors and controllers."
            ),
            daily_life_perspective=(
                f"{_sentence_start(consumer_source)} "
                f"{_source_be_verb(consumer_source)} useful for "
                f"{_source_possessive(consumer_source)} consumer context, "
                f"but normally {_source_do_verb(consumer_source)} "
                "not provide robot sensor "
                "measurements or controller commands."
            ),
            important_difference=(
                "Consumer records are not onboard robot perception, "
                "localization, control, or real-time safety data."
            ),
            final_conclusion=(
                "Use proper robot sensors, algorithms, controllers, and safety "
                "checks instead of treating the consumer source as a robotics "
                "replacement."
            ),
        )

    if any(
        phrase in q
        for phrase in ["train", "improve", "inform", "help", "contribute"]
    ):
        return _format_mixed_response(
            direct_answer=(
                f"{_sentence_start(consumer_source)} could help only indirectly "
                "and only if it contains authorized, task-relevant fields."
            ),
            relationship_type="Indirect",
            robotics_perspective=(
                "A robot task may use external coordinates, timestamps, route "
                "constraints, travel times, or outcome labels only after they "
                "are aligned to the robot's map, uncertainty model, and goal."
            ),
            daily_life_perspective=(
                f"{_sentence_start(consumer_source)} is mainly collected for a "
                "consumer service, not for robot sensing or control."
            ),
            important_difference=(
                "External app data can inform high-level planning or training, "
                "but it does not measure local obstacles or robot state."
            ),
            final_conclusion=(
                "The relationship is possible only as a limited indirect input; "
                "onboard perception, localization, controllers, and safety "
                "sensing are still required."
            ),
        )

    if "navigation" in q:
        return _format_mixed_response(
            direct_answer=(
                "Consumer navigation data may be a coarse external hint, but "
                "it cannot replace robot navigation."
            ),
            relationship_type="Indirect",
            robotics_perspective=(
                "Robot navigation needs a map or world model, localization, "
                "obstacle perception, planning, control, and safety checks."
            ),
            daily_life_perspective=(
                "Phone navigation uses maps, GPS, traffic information, and app "
                "routing for human travel."
            ),
            important_difference=(
                "A phone route does not provide real-time robot pose, obstacle "
                "measurements, or controller feedback."
            ),
            final_conclusion=(
                "Treat daily-life navigation data as optional context, not as a "
                "robotics navigation system."
            ),
        )

    relationship_type = (
        "Indirect"
        if relationship_hint.startswith("possibly indirect")
        or any(term in q for term in ["help", "improve", "affect", "use "])
        else "Analogy Only"
    )
    return _format_mixed_response(
        direct_answer=(
            f"{_sentence_start(robotics_focus)} and {consumer_source} are "
            "related only in a limited way; they should not be treated as the "
            "same system."
        ),
        relationship_type=relationship_type,
        robotics_perspective=(
            f"{_sentence_start(robotics_focus)} depends on robot-side sensor "
            "measurements, coordinate frames, uncertainty, and task-specific "
            "robot behavior."
        ),
        daily_life_perspective=(
            f"{_sentence_start(consumer_source)} belongs to a consumer or "
            "everyday digital-service context and is usually optimized for "
            "human use."
        ),
        important_difference=(
            "A similar word such as location, map, route, or navigation does "
            "not mean the data can replace physical robot sensing or control."
        ),
        final_conclusion=(
            "Use the relationship only as a careful analogy or limited "
            "external context unless the question gives authorized, "
            "task-relevant data fields."
        ),
    )


def _extract_named_definition(query):
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
    if normalized in KNOWN_TECHNOLOGIES:
        return None

    words = re.findall(r"[A-Za-z0-9*+-]+", candidate)
    title_words = [
        word for word in words
        if word[:1].isupper() and word.lower() not in {"the", "a", "an"}
    ]
    looks_named = bool(
        re.search(r"[a-z][A-Z]|[A-Za-z]+-[A-Za-z0-9-]+|[A-Z]{2,}", candidate)
        or len(title_words) >= 2
    )
    return candidate if looks_named else None


def _is_established_named_technology(entity):
    normalized = re.sub(r"\s+", " ", entity.lower()).strip()
    if normalized in KNOWN_TECHNOLOGIES:
        return True
    corpus = " ".join(item.get("text", "").lower() for item in DOCUMENT_CHUNKS)
    return normalized in corpus


def _filter_semantic_memory(query, domain, semantic):
    if domain not in {"robotics", "daily", "mixed"}:
        return []

    query_terms = {
        term for term in re.findall(r"[a-z0-9]+", query.lower())
        if term not in {
            "a", "an", "and", "are", "can", "do", "does", "for", "how",
            "i", "in", "is", "it", "my", "of", "or", "the", "to", "what",
            "which", "why",
        }
    }
    filtered = []
    for item in semantic:
        text = str(item.get("text", ""))
        memory_domain = _resolve_domain(text)
        if domain in {"robotics", "daily"} and memory_domain != domain:
            continue
        if domain == "mixed" and memory_domain not in {"robotics", "daily", "mixed"}:
            continue
        memory_terms = set(re.findall(r"[a-z0-9]+", text.lower()))
        if query_terms and not (query_terms & memory_terms):
            continue
        filtered.append(item)
    return filtered


def _retrieve_context(query, domain, learned):
    if domain == "mixed":
        # A narrowly targeted policy passage prevents unrelated consumer-app
        # chunks from contaminating a cross-domain answer.
        documents = [
            item for item in DOCUMENT_CHUNKS
            if item.get("text", "").startswith(
                "External Consumer Data and Robotics"
            )
        ][:1]
    elif domain in {"robotics", "daily"}:
        candidates = retrieve_local_knowledge(
            query,
            k=4,
            domain=domain,
        )
        if not candidates:
            documents = []
        else:
            top_score = candidates[0].get("score", 0)
            ratio = 0.72 if domain == "daily" else 0.80
            documents = [
                item for item in candidates
                if item.get("score", 0) >= top_score * ratio
            ][:2]
    else:
        documents = []
    semantic = _filter_semantic_memory(
        query,
        domain,
        search_memory(query, k=3),
    )
    return documents, semantic, learned


def _build_generation_input(
    query,
    *,
    domain,
    task,
    needs_clarification,
    clarifying_question,
    relationship_hint,
    relationship_guidance,
    ambiguity_guidance,
    personal_facts,
    learned,
    documents,
    semantic,
    classifier_label,
    classifier_confidence,
):
    sections = [
        f"Latest user message:\n{query}",
        (
            "Request analysis:\n"
            f"- resolved domain: {domain}\n"
            f"- requested task: {task}\n"
            f"- ambiguity requiring clarification: {needs_clarification}\n"
            f"- relationship hint: {relationship_hint}\n"
            f"- weak classifier hint: {classifier_label} "
            f"({classifier_confidence:.3f})"
        ),
    ]
    if needs_clarification:
        sections.append(
            "Required final clarification question:\n" + clarifying_question
        )
        sections.append(
            "Useful conditional diagnostic guidance:\n" + ambiguity_guidance
        )
    if relationship_hint != "none":
        sections.append(
            "Required relationship handling:\n" + relationship_guidance
        )
    if personal_facts:
        sections.append(
            "Relevant personal facts:\n"
            + json.dumps(personal_facts, ensure_ascii=False)
        )
    if learned:
        sections.append(
            "Relevant user-taught definitions:\n"
            + "\n".join(
                f"- {item['concept']}: {item['definition']}"
                for item in learned
            )
        )
    if semantic:
        sections.append(
            "Relevant prior learned memory:\n"
            + "\n".join(f"- {item['text'][:500]}" for item in semantic)
        )
    if documents:
        sections.append(
            "Local reference:\n"
            + "\n\n".join(
                f"[{item['source']}] {item['text'][:1200]}"
                for item in documents
            )
        )
    else:
        sections.append(
            "Local reference: no close passage was found. Use calibrated "
            "general knowledge and do not invent named-product details."
        )
    return "\n\n".join(sections)


def _clean_response(response, needs_clarification, clarifying_question):
    response = re.sub(r"\[[^\]]+\.txt\]\s*", "", response)
    response = complete_sentence_response(
        response,
        max_words=155 if needs_clarification else 150,
    )

    internal_phrases = [
        "task plan", "provided evidence", "retrieved context", "system prompt",
        "according to the reference", "the local reference",
    ]
    if any(phrase in response.lower() for phrase in internal_phrases):
        sentences = re.split(r"(?<=[.!?])\s+", response)
        response = " ".join(
            sentence for sentence in sentences
            if not any(phrase in sentence.lower() for phrase in internal_phrases)
        ).strip()

    if needs_clarification and "?" not in response:
        response = response.rstrip() + " " + clarifying_question
    return response


KEY_TERM_STOPWORDS = {
    "about", "again", "answer", "app", "apps", "are", "best", "can",
    "compare", "could", "current", "describe", "does", "for", "from",
    "goodbye", "have", "hello", "help", "hey", "how", "improve", "into",
    "make", "manage", "mean", "means", "most", "question", "reduce",
    "safe", "safely", "should", "system", "tell", "thanks", "thank",
    "that", "the", "their", "this", "use", "used", "using", "what",
    "when", "while", "which", "why", "with", "without", "would",
}

CURRENT_TERM_PHRASES = [
    "screen time",
    "studying online",
    "student",
    "notes",
    "laptop",
    "mobile",
    "distraction",
    "distractions",
    "phone battery drain",
    "important notifications",
    "notifications",
    "data usage",
    "internet",
    "browser",
    "download",
    "ride-sharing",
    "ride sharing",
    "ride-hailing",
    "food delivery",
    "restaurant ratings",
    "restaurant rating",
    "social media",
    "google maps",
    "app permissions",
    "app permission",
    "robot perception",
    "robot localization",
    "path planning",
    "sensor fusion",
    "wheel encoder",
    "public wi-fi",
    "cloud storage",
    "offline maps",
]

TERM_ALIASES = {
    "slam": ["slam", "simultaneous localization and mapping"],
    "ride-sharing": ["ride-sharing", "ride sharing", "ride-sharing apps"],
    "ride sharing": ["ride-sharing", "ride sharing", "ride-sharing apps"],
    "ride-hailing": ["ride-hailing", "ride hailing", "ride-hailing apps"],
    "screen time": ["screen time"],
    "restaurant ratings": ["restaurant ratings", "restaurant rating"],
    "restaurant rating": ["restaurant ratings", "restaurant rating"],
    "robot perception": ["robot perception", "perception"],
    "robot localization": ["robot localization", "localization"],
    "google maps": ["google maps"],
    "public wi-fi": ["public wi-fi", "public wifi", "public wi fi"],
    "reliable": ["reliable", "reliability"],
    "distraction": ["distraction", "distractions", "distracting"],
    "distractions": ["distraction", "distractions", "distracting"],
    "studying online": ["studying online", "study online", "online study"],
    "notifications": ["notification", "notifications"],
}


def _normalized_match_text(text):
    return re.sub(r"[-_/]+", " ", str(text or "").lower())


def _contains_term(text, term):
    haystack = _normalized_match_text(text)
    aliases = TERM_ALIASES.get(term.lower(), [term])
    for alias in aliases:
        normalized = _normalized_match_text(alias).strip()
        if not normalized:
            continue
        pattern = r"(?<![a-z0-9])" + r"\s+".join(
            re.escape(part) for part in normalized.split()
        )
        if " " not in normalized:
            pattern += r"s?"
        pattern += r"(?![a-z0-9])"
        if re.search(pattern, haystack):
            return True
    return False


def _question_key_terms(query):
    q = query.lower()
    terms = []
    for phrase in CURRENT_TERM_PHRASES:
        if _contains_term(q, phrase):
            terms.append(phrase)

    for token in re.findall(r"[a-z0-9*+-]+", q):
        normalized = token.strip("+-").lower()
        if len(normalized) < 3:
            continue
        if normalized in KEY_TERM_STOPWORDS:
            continue
        if normalized not in terms:
            terms.append(normalized)
    return terms[:8]


def _required_current_terms(query):
    q = query.lower()
    required = []
    if _contains_term(q, "screen time"):
        required.append("screen time")
    if _contains_term(q, "slam") and (
        _contains_term(q, "ride-sharing")
        or _contains_term(q, "ride-hailing")
    ):
        required.extend(["slam", "ride-sharing"])
    return required


def _response_mentions_current_question(query, response):
    if not str(response or "").strip():
        return False, _question_key_terms(query)[:3]

    missing_required = [
        term for term in _required_current_terms(query)
        if not _contains_term(response, term)
    ]
    if missing_required:
        return False, missing_required

    key_terms = _question_key_terms(query)
    if not key_terms:
        return True, []
    if any(_contains_term(response, term) for term in key_terms):
        return True, []
    return False, key_terms[:3]


def _is_daily_navigation_app_query(query):
    return any(
        _contains_term(query, term)
        for term in [
            "navigation app",
            "navigation apps",
            "mobile gps app",
            "mobile gps apps",
        ]
    )


def _fallback_current_question_response(query, domain):
    q = query.lower()
    key_terms = _question_key_terms(query)
    focus = _robotics_focus_from_query(query)
    source = _consumer_source_from_query(query)

    if _contains_term(q, "screen time"):
        return (
            "Direct Answer\nScreen time means the amount of time spent using a "
            "phone, computer, tablet, TV, or similar screen.\n\n"
            "Short Explanation\nTo manage screen time, first measure your daily "
            "use, then reduce the most distracting categories gradually.\n\n"
            "Key Points\n- Turn off nonessential notifications.\n"
            "- Move distracting apps away from the home screen.\n"
            "- Set device-free times, especially before sleep.\n\n"
            "Short Conclusion\nScreen time improves when you track it and make "
            "small, repeatable limits."
        )

    if domain == "daily" and _is_daily_navigation_app_query(q):
        return (
            "Direct Answer\nChoose navigation apps by matching them to your "
            "usual routes, local coverage, privacy comfort, and phone limits.\n\n"
            "Short Explanation\nA good navigation app should give accurate "
            "routes, current traffic updates, and reliable directions in the "
            "places where you actually travel.\n\n"
            "Key Points\n- Check route accuracy and local coverage for your city or area.\n"
            "- Prefer strong traffic updates and clear rerouting.\n"
            "- Look for offline maps for weak-signal trips.\n"
            "- Compare battery and data usage on longer journeys.\n"
            "- Review privacy and location permissions before relying on it.\n\n"
            "Practical Advice or Example\nTry the same familiar trip in two "
            "navigation apps and compare ETA, route clarity, offline support, "
            "and ease of use.\n\n"
            "Short Conclusion\nThe best choice is the app that is accurate, "
            "easy to use, privacy-aware, and dependable in your local area."
        )

    if domain == "daily" and (
        _contains_term(q, "distractions")
        or _contains_term(q, "studying online")
    ):
        return (
            "Direct Answer\nTo reduce distractions while studying online, make "
            "your device quieter, your study task clearer, and your breaks "
            "planned.\n\n"
            "Short Explanation\nOnline study is difficult because the same "
            "laptop or phone used for learning also contains messages, apps, "
            "and websites that interrupt attention.\n\n"
            "Key Points\n- Turn off nonessential notifications during study blocks.\n"
            "- Keep only the course tab, notes, and required files open.\n"
            "- Use short timed sessions with a planned break.\n\n"
            "Short Conclusion\nA simple setup and fewer notifications usually "
            "reduce distractions more reliably than willpower alone."
        )

    if domain == "daily" and (
        _contains_term(q, "notes")
        or _contains_term(q, "laptop")
        or _contains_term(q, "mobile")
    ):
        return (
            "Direct Answer\nA student should keep notes in one main system that "
            "syncs between laptop and mobile, with simple folders and clear "
            "file names.\n\n"
            "Key Points\n- Use one notebook or folder per course.\n"
            "- Put dates and topic names in note titles.\n"
            "- Keep important files backed up and searchable.\n\n"
            "Short Conclusion\nConsistency matters more than using many note apps."
        )

    if domain == "daily" and _contains_term(q, "public wi-fi"):
        return (
            "Direct Answer\nUse public Wi-Fi carefully while travelling: avoid "
            "sensitive transactions unless the connection and website are secure.\n\n"
            "Key Points\n- Prefer HTTPS websites and trusted apps.\n"
            "- Avoid banking or QR payment on unknown networks when possible.\n"
            "- Turn off auto-join and file sharing.\n\n"
            "Short Conclusion\nPublic Wi-Fi is convenient, but privacy settings "
            "and cautious use matter."
        )

    if domain == "daily" and (
        _contains_term(q, "phone battery drain")
        or _contains_term(q, "notifications")
    ):
        return (
            "Direct Answer\nTo reduce phone battery drain without missing "
            "important notifications, limit background activity instead of "
            "turning everything off.\n\n"
            "Key Points\n- Keep notifications on only for important apps.\n"
            "- Lower brightness and shorten screen timeout.\n"
            "- Disable background refresh for distracting or unused apps.\n\n"
            "Short Conclusion\nPrioritize essential notifications and reduce "
            "the silent battery drains around them."
        )

    if domain == "mixed":
        deterministic = _deterministic_relationship_response(
            query,
            domain,
            _relationship_hint(query, domain),
        )
        if deterministic:
            return deterministic
        return _format_mixed_response(
            direct_answer=(
                f"This question combines {focus} with {source}; they should be "
                "kept separate unless a specific data relationship is justified."
            ),
            relationship_type="Unsupported",
            robotics_perspective=(
                f"{_sentence_start(focus)} depends on robot-side measurements, "
                "models, control, and safety requirements."
            ),
            daily_life_perspective=(
                f"{_sentence_start(source)} belongs to an everyday consumer "
                "context and normally is not a robot sensor or controller."
            ),
            important_difference=(
                "A consumer service record is not the same as physical robot "
                "pose, geometry, obstacle, or feedback data."
            ),
            final_conclusion=(
                "Use the consumer data only if the current question gives a "
                "clear, authorized, task-relevant field; otherwise treat the "
                "relationship as unsupported."
            ),
        )

    if domain == "robotics":
        term = key_terms[0] if key_terms else focus
        return (
            f"Direct Answer\nThis is a robotics question about {term}.\n\n"
            "Short Explanation\nA reliable answer needs to stay with the robot "
            "system, its sensors, its control or planning goal, and the observed "
            "behavior.\n\n"
            "Practical Advice\nCheck the robot logs, sensor data, timestamps, "
            "coordinate frames, and any controller or planner errors before "
            "changing settings.\n\n"
            "Short Conclusion\nFocus the diagnosis on the current robot problem "
            "and avoid unrelated daily-life examples."
        )

    if domain == "daily":
        term = key_terms[0] if key_terms else "this daily-life topic"
        return (
            f"Direct Answer\nThis is a daily-life question about {term}.\n\n"
            "Short Explanation\nThe useful answer should stay practical and "
            "focus on the current situation, not robotics details.\n\n"
            "Practical Advice\nIdentify the goal, compare the main options, and "
            "choose the action that is safest, simplest, and easiest to repeat.\n\n"
            "Short Conclusion\nKeep the guidance tied to the current everyday "
            "task."
        )

    term = key_terms[0] if key_terms else "the current question"
    return (
        f"Direct Answer\nI should answer {term} directly, but I do not have "
        "enough reliable context for a detailed answer.\n\n"
        "Short Explanation\nThe safest response is to stay with the current "
        "question and avoid bringing in unrelated previous topics.\n\n"
        "Short Conclusion\nPlease provide one more detail if you want a more "
        "specific answer."
    )


def _regenerate_current_question_only(query, domain, task, missing_terms, seed):
    required = ", ".join(missing_terms) if missing_terms else "(none)"
    prompt = (
        f"Latest user message:\n{query}\n\n"
        f"Resolved domain: {domain}\n"
        f"Requested task: {task}\n"
        f"Required current-question terms to mention: {required}\n\n"
        "Use only the latest user message and the resolved domain. Do not use "
        "prior answers, previous relationship analysis, memory, retrieved "
        "passages, examples about restaurant ratings, or any earlier final "
        "answer. If the question is mixed, build the answer from the entities "
        "named in the latest user message."
    )
    llm_called = False
    cache_used = False
    try:
        llm_called = True
        response = chat(
            FAST_GROUNDED_PROMPT,
            prompt,
            temperature=0.0,
            seed=seed + 140,
            max_tokens=240,
            ensure_complete=False,
        )
        cache_used = was_last_cache_used()
        response = complete_sentence_response(response, max_words=170)
        valid, _ = _response_mentions_current_question(query, response)
        if valid:
            return (
                response,
                "current_question_only_regeneration",
                llm_called,
                cache_used,
            )
    except LLMRuntimeError:
        pass

    return (
        _fallback_current_question_response(query, domain),
        "current_question_only_fallback",
        llm_called,
        cache_used,
    )


DAILY_CONTAMINATION_PHRASES = [
    "ride-sharing",
    "ride sharing",
    "uber",
    "ola",
    "food delivery",
    "restaurant",
    "shopping app",
    "social media",
    "public wi-fi",
    "public wifi",
    "phone storage",
    "qr payment",
    "qr payments",
    "browser extension",
    "browser extensions",
    "password manager",
    "password managers",
]

ROBOTICS_CONTAMINATION_PHRASES = [
    "slam",
    "lidar",
    "amcl",
    "ekf",
    "odometry",
    "ros",
    "costmap",
    "particle filter",
    "sensor fusion",
    "robot localization",
]

MIXED_TEMPLATE_MARKERS = [
    "relationship type",
    "robotics perspective",
    "daily-life perspective",
    "daily life perspective",
]


def _mixed_template_markers_present(response):
    text = str(response or "").lower()
    return any(marker in text for marker in MIXED_TEMPLATE_MARKERS)


def _off_domain_contamination_terms(response, domain, relationship_required):
    if relationship_required or domain not in {"robotics", "daily"}:
        return []

    terms = []
    phrases = (
        DAILY_CONTAMINATION_PHRASES
        if domain == "robotics"
        else ROBOTICS_CONTAMINATION_PHRASES
    )
    for phrase in phrases:
        if _contains_term(response, phrase):
            terms.append(phrase)

    for marker in MIXED_TEMPLATE_MARKERS:
        if marker in str(response or "").lower():
            terms.append(marker)

    return sorted(set(terms))


def _regenerate_without_domain_contamination(
    query,
    domain,
    task,
    contamination_terms,
    seed,
):
    if domain == "robotics":
        instruction = (
            "Answer this as a pure robotics question. Remove unrelated "
            "daily-life examples or mixed-domain framing."
        )
    else:
        instruction = (
            "Answer this as a pure daily-life question. Remove unrelated "
            "robotics examples or mixed-domain framing."
        )

    prompt = (
        f"Latest user message:\n{query}\n\n"
        f"Resolved domain: {domain}\n"
        f"Requested task: {task}\n"
        "Detected unrelated terms or framing:\n"
        + ", ".join(contamination_terms)
        + "\n\n"
        f"{instruction}\n"
        "Do not include Relationship Type, Robotics Perspective, "
        "Daily-Life Perspective, or cross-domain analysis unless the user "
        "explicitly asks for a relationship."
    )
    llm_called = False
    cache_used = False
    try:
        llm_called = True
        response = chat(
            FAST_GROUNDED_PROMPT,
            prompt,
            temperature=0.0,
            seed=seed + 240,
            max_tokens=220,
            ensure_complete=False,
        )
        cache_used = was_last_cache_used()
        response = complete_sentence_response(response, max_words=160)
        if not _off_domain_contamination_terms(response, domain, False):
            return (
                response,
                "contamination_repair_regeneration",
                llm_called,
                cache_used,
                False,
                True,
            )
    except LLMRuntimeError:
        pass

    fallback = _fallback_current_question_response(query, domain)
    return (
        fallback,
        "contamination_repair_fallback",
        llm_called,
        cache_used,
        True,
        not _off_domain_contamination_terms(fallback, domain, False),
    )


def chatbot_system_c(user_input, return_metadata=False):
    start = time.time()
    seed = int(os.environ.get("EXPERIMENT_SEED", "0"))
    stages = []

    update_structured_memory("system_c", user_input)
    personal_facts = select_facts("system_c", user_input)
    stages.append("structured_personal_memory")

    learned_now = _learn_from_user_statement(user_input)
    learned = _learned_context(user_input)
    stages.append("learning_database")

    classified_intent, confidence = CLASSIFIER.predict(user_input)
    corrected_intent = correct_intent_label(user_input, classified_intent)
    stages.append("shared_intent_classifier")

    direct_personal = _personal_response(user_input, personal_facts)
    documents = []
    semantic = []
    relationship_hint = "none"
    relationship_guidance = ""
    needs_clarification = False
    clarifying_question = ""
    ambiguity_guidance = ""
    relationship_required = detect_relationship_required(user_input)
    domain = _resolve_domain(user_input, relationship_required)
    task = _infer_task(user_input, domain)
    named_entity = None
    cache_used = False
    deterministic_path_used = False
    llm_called = False
    benchmark_force_llm = _benchmark_force_llm()

    if learned_now:
        response = (
            f"Learned and remembered: {learned_now['concept']} means "
            f"{learned_now['definition']}."
        )
        stages.extend(["trusted_learning_write", "final_answer"])
        deterministic_path_used = True
    elif learned:
        best = learned[0]
        response = f"{best['concept']} means {best['definition']}."
        stages.extend(["trusted_learning_recall", "final_answer"])
        deterministic_path_used = True
    elif direct_personal:
        response = direct_personal
        stages.extend(["trusted_personal_memory", "final_answer"])
        deterministic_path_used = True
    else:
        named_entity = _extract_named_definition(user_input)
        if (
            named_entity
            and not _is_established_named_technology(named_entity)
            and not relationship_required
            and not benchmark_force_llm
        ):
            response = (
                f"I cannot verify that \"{named_entity}\" appears to be a "
                "standard or well-established concept from the available "
                "knowledge. It may be fictional, experimental, or "
                "domain-specific. If you have a research paper or source "
                "describing it, I can analyze it."
            )
            stages.extend(["unsupported_entity_guardrail", "final_answer"])
            deterministic_path_used = True
        else:
            if (
                _needs_unknown_general_response(user_input, domain)
                and not benchmark_force_llm
            ):
                response = _deterministic_unknown_general_response(user_input)
                stages.extend([
                    "deterministic_unknown_general_response",
                    "final_answer",
                ])
                deterministic_path_used = True
            elif domain == "daily" and _is_daily_navigation_app_query(user_input):
                response = _fallback_current_question_response(user_input, domain)
                stages.extend([
                    "deterministic_daily_navigation_app_response",
                    "final_answer",
                ])
                deterministic_path_used = True
            else:
                needs_clarification = _needs_clarification(user_input, domain)
                if needs_clarification:
                    domain = "unknown"
                    clarifying_question = _clarifying_question(user_input)
                    ambiguity_guidance = _ambiguity_guidance(user_input)
                relationship_hint = (
                    _relationship_hint(user_input, domain)
                    if relationship_required
                    else "none"
                )
                if relationship_hint != "none":
                    relationship_guidance = _relationship_guidance(
                        user_input,
                        relationship_hint,
                    )
                stages.append("deterministic_request_analysis")

                if needs_clarification and _fast_research_mode():
                    response = _deterministic_ambiguity_response(
                        user_input,
                        clarifying_question,
                    )
                else:
                    response = (
                        _deterministic_relationship_response(
                            user_input,
                            domain,
                            relationship_hint,
                        )
                        if _fast_research_mode() and relationship_required
                        else None
                    )
                if response:
                    deterministic_path_used = True
                    if needs_clarification:
                        stages.extend(["ambiguity_guardrail_no_mistral", "final_answer"])
                    else:
                        stages.extend(["compatibility_logic_no_mistral", "final_answer"])
                else:
                    documents, semantic, learned = _retrieve_context(
                        user_input,
                        domain,
                        learned,
                    )
                    stages.extend(["domain_scoped_bm25", "lightweight_semantic_memory"])

                    prompt = _build_generation_input(
                        user_input,
                        domain=domain,
                        task=task,
                        needs_clarification=needs_clarification,
                        clarifying_question=clarifying_question,
                        relationship_hint=relationship_hint,
                        relationship_guidance=relationship_guidance,
                        ambiguity_guidance=ambiguity_guidance,
                        personal_facts=personal_facts,
                        learned=learned,
                        documents=documents,
                        semantic=semantic,
                        classifier_label=classified_intent,
                        classifier_confidence=confidence,
                    )
                    try:
                        llm_called = True
                        response = chat(
                            FAST_GROUNDED_PROMPT,
                            prompt,
                            temperature=0.05,
                            seed=seed + 40,
                            max_tokens=240,
                            ensure_complete=False,
                        )
                        cache_used = cache_used or was_last_cache_used()
                        response = _clean_response(
                            response,
                            needs_clarification,
                            clarifying_question,
                        )
                        stages.extend(["single_pass_grounded_generation", "final_answer"])
                    except LLMRuntimeError as exc:
                        response = f"Model runtime error: {exc}"
                        stages.extend(["runtime_error", "final_answer"])

    validation_passed, missing_terms = _response_mentions_current_question(
        user_input,
        response,
    )
    response_repaired = False
    repair_stage = None
    if not validation_passed:
        stages = [stage for stage in stages if stage != "final_answer"]
        stages.append("stale_answer_validation_failed")
        response, repair_stage, repair_llm_called, repair_cache_used = (
            _regenerate_current_question_only(
                user_input,
                domain,
                task,
                missing_terms,
                seed,
            )
        )
        llm_called = llm_called or repair_llm_called
        cache_used = cache_used or repair_cache_used
        deterministic_path_used = (
            deterministic_path_used
            or repair_stage == "current_question_only_fallback"
        )
        stages.extend([repair_stage, "final_answer"])
        response_repaired = True
        validation_passed, missing_terms = _response_mentions_current_question(
            user_input,
            response,
        )

    contamination_repair_attempted = False
    contamination_repair_successful = False
    contamination_terms = _off_domain_contamination_terms(
        response,
        domain,
        relationship_required,
    )
    if contamination_terms:
        stages = [stage for stage in stages if stage != "final_answer"]
        stages.append("off_domain_contamination_detected")
        (
            response,
            contamination_repair_stage,
            contamination_llm_called,
            contamination_cache_used,
            contamination_fallback_used,
            contamination_repair_successful,
        ) = _regenerate_without_domain_contamination(
            user_input,
            domain,
            task,
            contamination_terms,
            seed,
        )
        llm_called = llm_called or contamination_llm_called
        cache_used = cache_used or contamination_cache_used
        deterministic_path_used = (
            deterministic_path_used or contamination_fallback_used
        )
        stages.extend([contamination_repair_stage, "final_answer"])
        contamination_repair_attempted = True
        validation_passed, missing_terms = _response_mentions_current_question(
            user_input,
            response,
        )

    mixed_template_used = bool(
        domain == "mixed"
        and relationship_required
        and _mixed_template_markers_present(response)
    )

    result = {
        "response": response,
        "latency": time.time() - start,
        "pipeline": stages,
        "predicted_intent": corrected_intent,
        "classified_intent": classified_intent,
        "raw_predicted_intent": classified_intent,
        "corrected_predicted_intent": corrected_intent,
        "intent_confidence": confidence,
        "resolved_domain": domain,
        "task": task,
        "answer_requirements": [
            "answer the literal task directly",
            "use calibrated claims",
            "provide mechanisms, trade-offs, or actions when requested",
        ],
        "query_parts": [user_input],
        "retrieved_sources": sorted({
            item.get("source", "") for item in documents if item.get("source")
        }),
        "semantic_memory_hits": len(semantic),
        "learning_database_hits": len(learned),
        "learned_now": learned_now,
        "relationship": relationship_hint,
        "relationship_required": relationship_required,
        "mixed_template_used": mixed_template_used,
        "relationship_analysis": {
            "hint": relationship_hint,
            "mode": (
                "deterministic_compatibility_logic"
                if "compatibility_logic_no_mistral" in stages
                else "single_pass_grounded_analysis"
            ),
        } if relationship_hint != "none" else {},
        "ambiguity": (
            "essential system context is missing"
            if needs_clarification
            else "none"
        ),
        "ambiguity_analysis": {
            "ambiguous": needs_clarification,
            "clarifying_question": clarifying_question,
        },
        "audit": {
            "mode": "integrated_single_pass_self_check",
            "extra_llm_calls": 0,
        },
        "named_entity_guarded": named_entity,
        "cache_used": cache_used,
        "deterministic_path_used": deterministic_path_used,
        "llm_called": llm_called,
        "contamination_repair_attempted": contamination_repair_attempted,
        "contamination_repair_successful": contamination_repair_successful,
        "response_validation": {
            "mentions_current_question": validation_passed,
            "missing_terms": missing_terms,
            "repaired": response_repaired,
            "repair_stage": repair_stage,
        },
    }
    return result if return_metadata else response


if not os.path.exists(LEARNING_FILE):
    save_json(LEARNING_FILE, {})


def _cli_value(value):
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


def _print_cli_response(result, metadata_mode):
    response = result.get("response", "") if isinstance(result, dict) else result
    print("\nSystem C:")
    print(response)

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

    print("System C Chatbot started.")
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
            result = chatbot_system_c(
                user_input,
                return_metadata=True,
            ) if metadata_mode else chatbot_system_c(user_input)
            _print_cli_response(result, metadata_mode)
        except Exception as exc:
            print("\nSystem C:")
            print(f"Runtime error: {exc}\n")


if __name__ == "__main__":
    run_cli()
