import re


ROBOTICS_RELATIONSHIP_TERMS = {
    "slam",
    "lidar",
    "amcl",
    "ekf",
    "odometry",
    "localization",
    "mapping",
    "robot",
    "robotics",
    "autonomous robot",
    "sensor fusion",
    "imu",
    "wheel encoder",
    "costmap",
    "path planning",
    "navigation stack",
    "ros",
    "particle filter",
}

DAILY_RELATIONSHIP_TERMS = {
    "ride-sharing",
    "ride sharing",
    "ride apps",
    "ride app",
    "uber",
    "ola",
    "food delivery",
    "restaurant",
    "screen time",
    "phone battery",
    "public wi-fi",
    "public wifi",
    "online shopping",
    "google maps",
    "notes",
    "study",
    "studying",
    "student",
    "app",
    "apps",
    "notifications",
    "privacy",
    "password",
    "cloud storage",
    "browser",
    "daily-life",
    "daily life",
    "consumer",
}

STRONG_ROBOTICS_INTENT_TERMS = {
    "slam",
    "lidar",
    "amcl",
    "ekf",
    "odometry",
    "wheel odometry",
    "wheel encoder",
    "imu",
    "sensor fusion",
    "robot",
    "robotics",
    "autonomous robot",
    "localization",
    "mapping",
    "local planner",
    "global planner",
    "path planning",
    "navigation stack",
    "ros",
    "particle filter",
    "costmap",
}

STRONG_DAILY_INTENT_TERMS = {
    "ride-sharing",
    "ride sharing",
    "uber",
    "ola",
    "food delivery",
    "restaurant",
    "phone storage",
    "app permissions",
    "digital notes",
    "notes",
    "study",
    "studying",
    "student",
    "laptop",
    "mobile",
    "qr payments",
    "qr payment",
    "cloud storage",
    "browser extensions",
    "browser extension",
    "password managers",
    "password manager",
    "screen time",
    "public wi-fi",
    "public wifi",
    "online shopping",
    "navigation apps",
    "navigation app",
    "mobile gps apps",
    "mobile gps app",
    "phone battery",
}


def _contains_term(text, term):
    normalized = re.sub(r"[-_/]+", " ", str(text or "").lower())
    normalized_term = re.sub(r"[-_/]+", " ", term.lower())
    pattern = r"(?<![a-z0-9])" + r"\s+".join(
        re.escape(part) for part in normalized_term.split()
    )
    if " " not in normalized_term:
        pattern += r"s?"
    pattern += r"(?![a-z0-9])"
    return bool(re.search(pattern, normalized))


def _has_any(text, terms):
    return any(_contains_term(text, term) for term in terms)


def detect_relationship_required(question: str) -> bool:
    """Return True only for explicit cross-domain relationship requests."""
    q = str(question or "").lower()
    has_robotics = _has_any(q, ROBOTICS_RELATIONSHIP_TERMS)
    has_daily = _has_any(q, DAILY_RELATIONSHIP_TERMS)
    has_both_domains = has_robotics and has_daily

    explicit_cross_domain = any(
        phrase in q
        for phrase in [
            "cross-domain",
            "cross domain",
            "without mixing",
            "without contaminating",
            "contamination between",
            "direct or indirect",
            "supported or unsupported",
        ]
    )
    if explicit_cross_domain and has_both_domains:
        return True

    relationship_phrases = [
        "compare",
        "difference between",
        "similarity",
        "similarities",
        "relate",
        "related",
        "relationship",
        "connected",
        "versus",
        "instead of",
    ]
    if has_both_domains and (
        any(phrase in q for phrase in relationship_phrases)
        or re.search(r"\bvs\.?\b", q)
    ):
        return True

    action_patterns = [
        r"\bcan\b.+\b(?:improve|replace|help|affect)\b",
        r"\bcould\b.+\b(?:improve|replace|help|affect)\b",
        r"\bwould\b.+\b(?:improve|replace|help|affect)\b",
        r"\bdoes\b.+\b(?:improve|replace|help|affect)\b",
        r"\buse\b.+\bfor\b",
        r"\busing\b.+\bfor\b",
        r"\bhelp\b.+\bwith\b",
        r"\baffect\b",
    ]
    if has_both_domains and any(re.search(pattern, q) for pattern in action_patterns):
        return True

    return False


def _term_hits(text, terms):
    return [term for term in terms if _contains_term(text, term)]


def correct_intent_label(question, predicted_intent):
    """Deterministically correct noisy classifier labels for reporting."""
    q = str(question or "")
    relationship_required = detect_relationship_required(q)
    robotics_hits = _term_hits(q, STRONG_ROBOTICS_INTENT_TERMS)
    daily_hits = _term_hits(q, STRONG_DAILY_INTENT_TERMS)

    if robotics_hits and daily_hits:
        if relationship_required:
            return "mixed"
        return "robotics" if len(robotics_hits) >= len(daily_hits) else "daily"
    if robotics_hits:
        return "robotics"
    if daily_hits:
        return "daily"

    normalized = str(predicted_intent or "").strip().lower()
    aliases = {
        "robot": "robotics",
        "robotic": "robotics",
        "daily-life": "daily",
        "daily_life": "daily",
        "consumer": "daily",
        "mixed_domain": "mixed",
        "cross_domain": "mixed",
    }
    return aliases.get(normalized, normalized or None)
