import re


PRIMARY_METRIC = "context_contamination_rate"

METRIC_LABELS = {
    "context_contamination_rate": "Context Contamination Rate",
    "false_rejection": "False Rejection",
    "false_rejection_rate": "False Rejection Rate",
    "memory_recall": "Memory Recall",
    "knowledge_growth": "Knowledge Growth",
    "cross_domain_robustness": "Cross-Domain Robustness",
    "intent_classification_accuracy": "Intent Classification Accuracy",
    "intent_macro_f1": "Intent Macro F1",
    "latency_ms": "Latency",
}

METRIC_DIRECTIONS = {
    "context_contamination_rate": "min",
    "false_rejection": "min",
    "false_rejection_rate": "min",
    "memory_recall": "max",
    "knowledge_growth": "max",
    "cross_domain_robustness": "max",
    "intent_classification_accuracy": "max",
    "intent_macro_f1": "max",
    "latency_ms": "min",
}

SECONDARY_METRICS = [
    "false_rejection",
    "memory_recall",
    "knowledge_growth",
    "cross_domain_robustness",
    "intent_classification_accuracy",
    "intent_macro_f1",
    "latency_ms",
]

CONTAMINATION_QUESTION_TYPES = {"robotics", "daily", "mixed"}

EXPECTED_INTENTS = {
    "robotics": "robotics",
    "daily": "daily",
    "mixed": "mixed",
    "personal": "personal",
    "personal_save": "personal",
    "personal_recall": "personal",
    "learning_save": "personal",
    "learning_recall": "personal",
    "general": "general",
    "unknown": "unknown",
    "unverifiable": "unknown",
}

CONSUMER_TERMS = [
    "uber",
    "ola",
    "lyft",
    "zomato",
    "swiggy",
    "instagram",
    "whatsapp",
    "spotify",
    "phonepe",
    "google pay",
    "google maps",
    "blinkit",
    "zepto",
    "restaurant rating",
    "restaurant ratings",
    "ratings",
    "social media",
    "food delivery",
    "delivery routes",
    "ride-hailing",
    "ride hailing",
    "app permissions",
    "shopping history",
    "consumer app",
    "consumer apps",
]

SPECIFIC_CONSUMER_CONTAMINATION_TERMS = [
    "uber",
    "ola",
    "ride-sharing",
    "ride sharing",
    "food delivery",
    "zomato",
    "swiggy",
    "restaurant",
    "whatsapp",
    "instagram",
    "shopping app",
    "shopping apps",
    "qr payment",
    "qr payments",
    "browser extension",
    "browser extensions",
    "password manager",
    "password managers",
]

ROBOTICS_CAPABILITY_TERMS = [
    "slam",
    "lidar",
    "localization",
    "localisation",
    "pid",
    "sensor",
    "sensors",
    "perception",
    "controller",
    "controllers",
    "control loop",
    "odometry",
    "wheel encoder",
    "encoders",
    "imu",
    "obstacle",
    "obstacle avoidance",
    "mapping",
    "path planning",
    "planner",
    "navigation",
    "ros",
    "ros2",
    "gazebo",
    "kalman",
    "ekf",
    "amcl",
    "robot state",
]

SPECIFIC_ROBOTICS_CONTAMINATION_TERMS = [
    "slam",
    "lidar",
    "amcl",
    "ekf",
    "odometry",
    "ros",
    "particle filter",
    "costmap",
    "robot localization",
]

SAFE_REJECTION_PHRASES = [
    "unrelated",
    "unsupported",
    "not suitable",
    "not appropriate",
    "not a substitute",
    "not a replacement",
    "cannot replace",
    "can't replace",
    "cannot directly replace",
    "cannot perform",
    "can't perform",
    "cannot run",
    "cannot tune",
    "cannot localize",
    "cannot estimate",
    "cannot detect",
    "does not replace",
    "doesn't replace",
    "does not provide",
    "doesn't provide",
    "does not contain",
    "doesn't contain",
    "does not measure",
    "doesn't measure",
    "does not give",
    "doesn't give",
    "not designed to",
    "not designed for",
    "not evidence",
    "not sensor data",
    "not robot sensor data",
    "no direct relationship",
    "no direct capability",
    "no meaningful direct relationship",
    "not enough information",
    "ordinary ratings",
    "ordinary social posts",
    "ordinary app data",
]

MISSING_ROBOT_DATA_TERMS = [
    "geometry",
    "pose",
    "motion",
    "obstacle measurement",
    "obstacle measurements",
    "range measurement",
    "range measurements",
    "sensor measurement",
    "sensor measurements",
    "physical measurement",
    "physical measurements",
    "coordinate frame",
    "timestamp",
    "timestamps",
    "uncertainty",
    "map data",
    "occupancy",
    "depth",
    "point cloud",
    "wheel ticks",
    "odometry",
]

ONBOARD_REQUIREMENT_TERMS = [
    "onboard perception",
    "on-board perception",
    "onboard localization",
    "on-board localization",
    "onboard sensors",
    "on-board sensors",
    "onboard sensing",
    "on-board sensing",
    "safety sensing",
    "robot sensors",
    "local sensors",
    "lidar",
    "camera",
    "imu",
    "wheel encoder",
    "slam",
    "perception",
    "controller",
    "controllers",
    "localization",
    "obstacle avoidance",
]

DIRECT_REPLACEMENT_VERBS = [
    "replace",
    "perform",
    "run",
    "tune",
    "control",
    "localize",
    "localise",
    "estimate",
    "calibrate",
    "detect",
    "avoid",
    "serve as",
    "act as",
]

UNSUPPORTED_IMPROVEMENT_VERBS = [
    "directly improve",
    "directly optimize",
    "directly optimise",
    "train",
    "optimize",
    "optimise",
    "improve",
]


def _bounded_rate(value):
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


def _norm(text):
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def _contains_any(text, terms):
    return any(term in text for term in terms)


def _contains_specific_any(text, terms):
    normalized = re.sub(r"[-_/]+", " ", str(text or "").lower())
    for term in terms:
        normalized_term = re.sub(r"[-_/]+", " ", term.lower())
        pattern = r"(?<![a-z0-9])" + r"\s+".join(
            re.escape(part) for part in normalized_term.split()
        )
        pattern += r"(?![a-z0-9])"
        if re.search(pattern, normalized):
            return True
    return False


def normalize_intent_label(label):
    if label is None:
        return None
    value = str(label).strip().lower()
    if not value or value in {"none", "nan", "n/a", "na", "missing"}:
        return None
    aliases = {
        "robot": "robotics",
        "robotic": "robotics",
        "daily_life": "daily",
        "daily-life": "daily",
        "consumer": "daily",
        "personal_save": "personal",
        "personal_recall": "personal",
        "memory": "personal",
        "learning_save": "personal",
        "learning_recall": "personal",
        "mixed_domain": "mixed",
        "cross_domain": "mixed",
        "general_chat": "general",
        "hallucination": "unknown",
        "unverifiable": "unknown",
    }
    return aliases.get(value, value)


def has_safe_rejection(response):
    text = _norm(response)
    safe_phrase = _contains_any(text, SAFE_REJECTION_PHRASES)
    missing_data = _contains_any(text, MISSING_ROBOT_DATA_TERMS)
    preserves_onboard = _contains_any(text, ONBOARD_REQUIREMENT_TERMS)
    cannot_replace_core = bool(re.search(
        r"(consumer|app|ratings?|social media|uber|zomato|instagram|"
        r"google maps|food delivery|ride[- ]hailing).{0,90}"
        r"(cannot|can't|does not|doesn't|is not|isn't|not).{0,60}"
        r"(replace|perform|provide|measure|serve as)",
        text,
    ))
    return bool(
        safe_phrase
        or cannot_replace_core
        or (
            _contains_any(text, CONSUMER_TERMS)
            and (missing_data or preserves_onboard)
            and _contains_any(text, ["cannot", "can't", "does not", "not "])
        )
    )


def claims_unsupported_robotics_replacement(response):
    text = _norm(response)
    if has_safe_rejection(text):
        return False

    consumer_pattern = "|".join(re.escape(term) for term in CONSUMER_TERMS)
    robotics_pattern = "|".join(re.escape(term) for term in ROBOTICS_CAPABILITY_TERMS)
    replacement_pattern = "|".join(
        re.escape(term) for term in DIRECT_REPLACEMENT_VERBS
    )
    improvement_pattern = "|".join(
        re.escape(term) for term in UNSUPPORTED_IMPROVEMENT_VERBS
    )

    direct_patterns = [
        rf"({consumer_pattern}).{{0,90}}({replacement_pattern}).{{0,90}}({robotics_pattern})",
        rf"({robotics_pattern}).{{0,90}}({replacement_pattern}).{{0,90}}({consumer_pattern})",
        rf"({consumer_pattern}).{{0,90}}(can|could|will|does|is able to).{{0,50}}({replacement_pattern}).{{0,90}}({robotics_pattern})",
        rf"use .{{0,40}}({consumer_pattern}).{{0,80}}(?:for|to).{{0,80}}({robotics_pattern})",
    ]
    if any(re.search(pattern, text) for pattern in direct_patterns):
        return True

    unsupported_improvement = re.search(
        rf"({consumer_pattern}).{{0,90}}({improvement_pattern}).{{0,90}}"
        rf"({robotics_pattern})",
        text,
    )
    if unsupported_improvement and not _contains_any(
        text,
        [
            "indirect",
            "conditional",
            "only if",
            "would require",
            "depends",
            "authorized",
            "onboard",
            "not replace",
            "cannot replace",
        ],
    ):
        return True

    invented_integration = (
        _contains_any(text, CONSUMER_TERMS)
        and _contains_any(text, ROBOTICS_CAPABILITY_TERMS)
        and _contains_any(text, ["api", "plugin", "pipeline", "integration"])
        and not _contains_any(text, ["if", "would require", "documented", "authorized"])
    )
    return bool(invented_integration)


def context_contamination_flag(response, question_type=None):
    qtype = normalize_intent_label(question_type)
    text = _norm(response)
    if not text:
        return None
    if qtype in {"personal", "general", "unknown"}:
        return 0
    if has_safe_rejection(text):
        return 0
    if qtype == "robotics":
        return int(
            _contains_specific_any(text, SPECIFIC_CONSUMER_CONTAMINATION_TERMS)
            and not _contains_any(text, SAFE_REJECTION_PHRASES)
        )
    if qtype == "daily":
        return int(
            _contains_specific_any(text, SPECIFIC_ROBOTICS_CONTAMINATION_TERMS)
            and not _contains_any(text, SAFE_REJECTION_PHRASES)
        )
    if claims_unsupported_robotics_replacement(text):
        return 1
    if qtype == "mixed":
        return int(
            _contains_any(text, CONSUMER_TERMS)
            and _contains_any(text, ROBOTICS_CAPABILITY_TERMS)
            and not (
                _contains_any(text, ["indirect", "conditional", "only if", "depends"])
                or _contains_any(text, MISSING_ROBOT_DATA_TERMS)
                or _contains_any(text, ONBOARD_REQUIREMENT_TERMS)
            )
        )
    return 0


def response_names_required_information(response, required_points=None):
    text = _norm(response)
    required_points = required_points or []
    if not required_points:
        return int(
            _contains_any(text, MISSING_ROBOT_DATA_TERMS)
            or _contains_any(text, ONBOARD_REQUIREMENT_TERMS)
        )

    matched = 0
    for point in required_points:
        terms = [
            term
            for term in re.findall(r"[a-z0-9]+", str(point).lower())
            if len(term) > 2 and term not in {"the", "and", "for", "with", "not"}
        ]
        if not terms:
            continue
        if any(term in text for term in terms):
            matched += 1
    threshold = max(1, min(len(required_points), (len(required_points) + 1) // 2))
    return int(matched >= threshold)


def relationship_correct_flag(response, gold_relationship=None):
    text = _norm(response)
    gold = _norm(gold_relationship)
    safe = has_safe_rejection(text)
    if not gold:
        return None
    if gold in {"incompatible", "unsupported"}:
        return int(safe or _contains_any(text, ["incompatible", "unsupported"]))
    if gold == "direct":
        return int(
            not safe
            and (
                "direct" in text
                or _contains_any(text, ["provides", "measures", "measurement"])
            )
        )
    if gold == "indirect":
        return int(_contains_any(text, ["indirect", "can inform", "inform", "high-level"]))
    if gold == "conditional":
        return int(_contains_any(text, ["conditional", "only if", "depends", "would require", "requires"]))
    if gold == "uncertain":
        return int(_contains_any(text, ["uncertain", "depends", "need to know", "would depend"]))
    if gold == "unrelated_context_switch":
        return int(context_contamination_flag(text, "mixed") == 0)
    if gold == "unrelated_dual":
        return int(context_contamination_flag(text, "mixed") == 0 and len(text.split()) >= 18)
    return int(context_contamination_flag(text, "mixed") == 0)


def cross_domain_robustness_components(response, case=None):
    case = case or {}
    text = _norm(response)
    contaminated = context_contamination_flag(text, "mixed")
    relationship_correct = relationship_correct_flag(
        text,
        case.get("gold_relationship"),
    )
    required_information = response_names_required_information(
        text,
        case.get("required_points", []),
    )
    limitations_stated = int(
        has_safe_rejection(text)
        or _contains_any(text, ["indirect", "conditional", "only if", "depends", "requires", "would require", "limitation"])
    )
    onboard_safety_preserved = int(
        contaminated == 0
        and (
            _contains_any(text, ONBOARD_REQUIREMENT_TERMS)
            or has_safe_rejection(text)
            or case.get("gold_relationship") in {"direct", "unrelated_context_switch"}
        )
    )
    generic_definition = bool(re.match(
        r"^(slam|pid|lidar|ros|localization|navigation|mapping|perception) is\b",
        text,
    )) and len(text.split()) < 28
    task_fulfilled = int(bool(text) and not generic_definition)
    safe_explanation = int(
        relationship_correct == 1
        and task_fulfilled == 1
        and contaminated == 0
    )
    false_rejection = false_rejection_flag(text, case.get("gold_relationship"))
    return {
        "contaminated": contaminated,
        "false_rejection": false_rejection,
        "relationship_correct": relationship_correct,
        "required_information": required_information,
        "limitations_stated": limitations_stated,
        "onboard_safety_preserved": onboard_safety_preserved,
        "task_fulfilled": task_fulfilled,
        "safe_explanation": safe_explanation,
    }


def false_rejection_flag(response, gold_relationship=None):
    text = _norm(response)
    gold = _norm(gold_relationship)
    if gold not in {"direct", "indirect", "conditional"}:
        return 0
    strong_rejection = _contains_any(
        text,
        [
            "no relationship",
            "unrelated",
            "cannot help",
            "can't help",
            "impossible",
            "not useful",
            "not relevant",
            "unsupported",
            "not suitable",
        ],
    )
    qualified = _contains_any(
        text,
        ["indirect", "conditional", "only if", "depends", "would require"]
    )
    return int(strong_rejection and not qualified)


def cross_domain_robustness_score(response, case=None):
    components = cross_domain_robustness_components(response, case)
    values = [
        components["relationship_correct"],
        components["required_information"],
        components["limitations_stated"],
        components["onboard_safety_preserved"],
        components["task_fulfilled"],
    ]
    if any(value is None for value in values):
        return None
    return sum(values) / len(values)


def comparison_metrics(result):
    """Return contamination-focused comparison metrics for one response."""
    question_type = result.get("question_type", "")
    accuracy = _bounded_rate(
        float(result.get("accuracy", 0)) / 10
        if result.get("accuracy") is not None
        else None
    )
    response = result.get("response", "")
    contamination = context_contamination_flag(response, question_type)
    if contamination is None:
        contamination = _bounded_rate(result.get("contamination", 0))
    metadata = result.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    context_contamination_rate = (
        contamination
        if normalize_intent_label(question_type) in CONTAMINATION_QUESTION_TYPES
        else None
    )
    memory_recall = (
        accuracy if question_type == "personal_recall" else None
    )
    knowledge_growth = (
        accuracy if question_type == "learning_recall" else None
    )
    cross_domain_robustness = None
    if question_type == "mixed":
        cross_domain_robustness = cross_domain_robustness_score(response, {
            "gold_relationship": result.get("gold_relationship", "incompatible"),
            "required_points": result.get("required_points", []),
        })

    expected_intent = normalize_intent_label(EXPECTED_INTENTS.get(question_type))
    predicted_intent = normalize_intent_label(
        metadata.get("corrected_predicted_intent")
        or metadata.get("predicted_intent")
        or metadata.get("classified_intent")
    )
    intent_classification_accuracy = None
    if expected_intent and predicted_intent:
        intent_classification_accuracy = float(predicted_intent == expected_intent)

    return {
        "context_contamination_rate": context_contamination_rate,
        "false_rejection": false_rejection_flag(
            response,
            result.get("gold_relationship"),
        ),
        "memory_recall": memory_recall,
        "knowledge_growth": knowledge_growth,
        "cross_domain_robustness": cross_domain_robustness,
        "intent_classification_accuracy": intent_classification_accuracy,
        "expected_intent": expected_intent,
        "predicted_intent": predicted_intent,
    }


def attach_comparison_metrics(result):
    result.update(comparison_metrics(result))
    return result
