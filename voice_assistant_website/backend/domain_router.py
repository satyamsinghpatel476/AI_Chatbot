from __future__ import annotations

import re
from typing import Any


DOMAINS = {"robotics", "daily", "mixed", "ambiguous", "unverifiable", "unknown"}

STRICT_ISOLATION_CUES = (
    "without contamination",
    "without mixing",
    "do not mix",
    "don't mix",
    "ignore previous",
    "ignore earlier",
    "earlier we discussed",
    "after discussing",
    "now explain",
    "answer only",
    "keep separate",
    "separate from previous topic",
)
DOMAIN_SWITCH_CUES = (
    "earlier we discussed",
    "after discussing",
    "previously",
    "previous topic",
    "ignore previous",
    "ignore earlier",
    "now explain",
    "now answer",
    "now tell",
)
ROBOTICS_CUES = (
    "slam",
    "ros",
    "ros2",
    "lidar",
    "odometry",
    "ekf",
    "pid",
    "robot",
    "robots",
    "robotics",
    "rover",
    "localization",
    "localisation",
    "mapping",
    "path planning",
    "sensor fusion",
    "occupancy grid",
    "navigation algorithm",
    "navigation algorithms",
    "autonomous robot",
    "autonomous robots",
    "robot navigation",
    "robot localization",
)
DAILY_CUES = (
    "food delivery",
    "delivery app",
    "delivery apps",
    "ride-sharing",
    "ride sharing",
    "online privacy",
    "public wi-fi",
    "public wifi",
    "cloud storage",
    "phone storage",
    "screen time",
    "digital notes",
    "shopping app",
    "shopping apps",
    "social media",
    "online scam",
    "online scams",
    "mobile data",
    "safe downloads",
    "navigation app",
    "navigation apps",
    "daily life",
    "beginner app",
    "uber",
    "ola",
    "lyft",
    "zomato",
    "swiggy",
    "whatsapp",
    "instagram",
    "phone",
    "smartphone",
    "privacy",
    "password",
)
MIXED_CUES = (
    "relation",
    "relationship",
    "compare",
    "comparison",
    "replace",
    "directly improve",
    "indirectly improve",
    "analogy",
    "robotics and daily",
    "daily and robotics",
    "daily-life and robotics",
    "robotics and daily-life",
    "related",
)
UNVERIFIABLE_CUES = (
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
UNVERIFIABLE_STYLE_CUES = (
    "quantum",
    "hyper",
    "astro",
    "cosmic",
    "dream",
    "intuition",
    "self-aware",
    "bio-spiritual",
    "neuromagnetic",
)

DAILY_STRONG_CONTAMINATION_TERMS = (
    "slam",
    "ros",
    "lidar",
    "odometry",
    "ekf",
    "pid",
    "robotics perspective",
    "robot perception",
    "robot sensor",
    "robot sensors",
    "robot sensing",
    "robot control",
    "robot state",
    "robot's map",
    "onboard perception",
    "robot localization",
    "localization",
    "occupancy grid",
    "sensor fusion",
    "controller",
    "controllers",
)
ROBOTICS_STRONG_CONTAMINATION_TERMS = (
    "food delivery",
    "ride-sharing",
    "ride sharing",
    "uber",
    "shopping app",
    "shopping apps",
    "social media",
    "phone storage",
    "online scam",
    "online scams",
)


def normalize_domain_label(value: Any) -> str | None:
    if value is None:
        return None
    label = str(value).strip().lower()
    if not label or label in {"none", "null", "nan", "n/a", "na", "missing"}:
        return None
    label = label.replace("-", "_").replace(" ", "_")
    aliases = {
        "daily_life": "daily",
        "daily_digital_assistance": "daily",
        "general_daily": "daily",
        "consumer": "daily",
        "robot": "robotics",
        "robotic": "robotics",
        "robotics_support": "robotics",
        "cross_domain": "mixed",
        "mixed_domain": "mixed",
        "uncertain": "ambiguous",
        "fake": "unverifiable",
        "unverified": "unverifiable",
    }
    normalized = aliases.get(label, label)
    return normalized if normalized in DOMAINS or normalized in {"daily", "robotics"} else normalized


def contains_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in phrases)


def matching_terms(text: str, terms: tuple[str, ...]) -> list[str]:
    lowered = text.lower()
    matches = []
    for term in terms:
        pattern = rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])"
        if re.search(pattern, lowered):
            matches.append(term)
    return matches


def _focus_current_request(question: str) -> str:
    text = question.lower()
    focus = text
    for marker in (
        "now explain",
        "now answer",
        "now tell",
        "now describe",
        "answer a beginner",
        "answer beginner",
        "answer only",
        "explain",
    ):
        index = text.rfind(marker)
        if index >= 0:
            focus = text[index:]
            break

    for boundary in (
        " without contamination",
        " without mixing",
        " do not mix",
        " don't mix",
        " keep separate",
        " separate from previous topic",
    ):
        boundary_index = focus.find(boundary)
        if boundary_index > 0:
            focus = focus[:boundary_index]
            break
    return focus


def _unverifiable_like(text: str) -> bool:
    lowered = text.lower()
    if contains_phrase(lowered, UNVERIFIABLE_CUES):
        return True
    has_robotics_shape = contains_phrase(
        lowered,
        ("slam", "localization", "localisation", "mapping", "odometry", "navigation", "planner"),
    )
    has_fictional_style = contains_phrase(lowered, UNVERIFIABLE_STYLE_CUES)
    return bool(has_robotics_shape and has_fictional_style)


def _domain_hits(text: str) -> tuple[list[str], list[str], list[str]]:
    robotics_hits = matching_terms(text, ROBOTICS_CUES)
    daily_hits = matching_terms(text, DAILY_CUES)
    mixed_hits = matching_terms(text, MIXED_CUES)
    return robotics_hits, daily_hits, mixed_hits


def route_query(question: str, previous_context: str = "") -> dict[str, Any]:
    text = question or ""
    lowered = text.lower()
    focus = _focus_current_request(text)
    strict_isolation = contains_phrase(lowered, STRICT_ISOLATION_CUES)
    domain_switch = strict_isolation or contains_phrase(lowered, DOMAIN_SWITCH_CUES)

    if _unverifiable_like(lowered):
        return {
            "resolved_domain": "unverifiable",
            "target_domain": "unknown",
            "strict_isolation": strict_isolation,
            "domain_switch": domain_switch,
            "reason": "unverifiable_named_or_fictional_technical_concept",
            "confidence": 0.82,
        }

    focus_robotics, focus_daily, focus_mixed = _domain_hits(focus)
    full_robotics, full_daily, full_mixed = _domain_hits(lowered)

    mixed_requested = bool(
        full_mixed
        and (
            (full_robotics and full_daily)
            or "replace" in lowered
            or "relationship" in lowered
            or "related" in lowered
        )
        and not strict_isolation
    )
    if mixed_requested:
        return {
            "resolved_domain": "mixed",
            "target_domain": "mixed",
            "strict_isolation": False,
            "domain_switch": domain_switch,
            "reason": "explicit_cross_domain_relationship_request",
            "confidence": 0.88,
        }

    if focus_daily and not focus_robotics:
        return {
            "resolved_domain": "daily",
            "target_domain": "daily",
            "strict_isolation": strict_isolation,
            "domain_switch": domain_switch,
            "reason": "current_request_daily_domain",
            "confidence": 0.86 if strict_isolation else 0.74,
        }
    if focus_robotics and not focus_daily:
        return {
            "resolved_domain": "robotics",
            "target_domain": "robotics",
            "strict_isolation": strict_isolation,
            "domain_switch": domain_switch,
            "reason": "current_request_robotics_domain",
            "confidence": 0.88 if strict_isolation else 0.78,
        }

    if full_daily and not full_robotics:
        return {
            "resolved_domain": "daily",
            "target_domain": "daily",
            "strict_isolation": strict_isolation,
            "domain_switch": domain_switch,
            "reason": "daily_domain_cues",
            "confidence": 0.72,
        }
    if full_robotics and not full_daily:
        return {
            "resolved_domain": "robotics",
            "target_domain": "robotics",
            "strict_isolation": strict_isolation,
            "domain_switch": domain_switch,
            "reason": "robotics_domain_cues",
            "confidence": 0.76,
        }
    if full_daily and full_robotics:
        target = "unknown" if strict_isolation else "mixed"
        return {
            "resolved_domain": "ambiguous" if strict_isolation else "mixed",
            "target_domain": target,
            "strict_isolation": strict_isolation,
            "domain_switch": domain_switch,
            "reason": "both_domains_present_without_clear_current_target" if strict_isolation else "both_domains_present",
            "confidence": 0.58 if strict_isolation else 0.68,
        }

    previous_domain = None
    if previous_context:
        prev_robotics, prev_daily, _prev_mixed = _domain_hits(previous_context)
        if prev_robotics and not prev_daily:
            previous_domain = "robotics"
        elif prev_daily and not prev_robotics:
            previous_domain = "daily"

    return {
        "resolved_domain": "ambiguous" if domain_switch else "unknown",
        "target_domain": "unknown",
        "strict_isolation": strict_isolation,
        "domain_switch": domain_switch or bool(previous_domain),
        "reason": "domain_switch_without_clear_target" if domain_switch else "insufficient_domain_evidence",
        "confidence": 0.45 if domain_switch else 0.25,
    }


def contamination_terms_for_route(answer: str, route: dict[str, Any]) -> list[str]:
    target_domain = normalize_domain_label(route.get("target_domain"))
    if target_domain == "daily":
        return matching_terms(answer, DAILY_STRONG_CONTAMINATION_TERMS)
    if target_domain == "robotics":
        return matching_terms(answer, ROBOTICS_STRONG_CONTAMINATION_TERMS)
    return []


def needs_soft_repair(answer: str, route: dict[str, Any]) -> bool:
    if normalize_domain_label(route.get("resolved_domain")) == "mixed" and not route.get("strict_isolation"):
        return False
    confidence = float(route.get("confidence") or 0.0)
    if not route.get("strict_isolation") and confidence < 0.7:
        return False
    return bool(contamination_terms_for_route(answer, route))
