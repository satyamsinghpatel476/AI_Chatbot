import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chatbot_system_c import chatbot_system_c
from evaluator.metrics import context_contamination_flag
from system_b.chatbot_system_b import chatbot_system_b


def fail(message, failures):
    failures.append(message)


def check_equal(label, actual, expected, failures):
    if actual != expected:
        fail(f"{label}: got {actual!r}, expected {expected!r}", failures)


def contains_all(text, terms):
    lowered = text.lower()
    return all(term in lowered for term in terms)


def contains_any(text, terms):
    lowered = text.lower()
    return any(term in lowered for term in terms)


def run_case(name, question, expected_intent, expected_domain=None):
    failures = []
    b = chatbot_system_b(question, return_metadata=True)
    c = chatbot_system_c(question, return_metadata=True)

    check_equal(
        "System B corrected_intent",
        b.get("corrected_predicted_intent"),
        expected_intent,
        failures,
    )
    check_equal(
        "System C corrected_intent",
        c.get("corrected_predicted_intent"),
        expected_intent,
        failures,
    )
    if expected_domain is not None:
        check_equal(
            "System C resolved_domain",
            c.get("resolved_domain"),
            expected_domain,
            failures,
        )
    return failures, b, c


def main():
    total = 0
    failed = 0

    cases = [
        (
            "robotics_slam_failure",
            "Why does SLAM fail in real-world environments?",
            "robotics",
            "robotics",
            "robotics",
        ),
        (
            "robotics_ekf_debug",
            "How can a beginner debug EKF sensor fusion?",
            "robotics",
            "robotics",
            "robotics",
        ),
        (
            "daily_navigation_apps",
            "How can a beginner choose between different navigation apps?",
            "daily",
            "daily",
            "daily",
        ),
        (
            "daily_ride_sharing_risks",
            "What risks should I consider while using ride-sharing apps?",
            "daily",
            "daily",
            "daily",
        ),
        (
            "mixed_uber_gps_slam",
            "Can Uber GPS replace SLAM?",
            "mixed",
            "mixed",
            "mixed",
        ),
    ]

    for name, question, intent, domain, contamination_type in cases:
        total += 1
        failures, _, c = run_case(name, question, intent, domain)
        response = c.get("response", "")
        contamination = context_contamination_flag(response, contamination_type)

        if name.startswith("robotics") and contamination:
            fail("System C robotics response marked contaminated", failures)
        if name == "daily_ride_sharing_risks" and contamination:
            fail("System C daily response marked robotics-contaminated", failures)
        if name == "daily_navigation_apps":
            if not contains_all(
                response,
                ["route", "traffic", "offline", "privacy", "battery"],
            ):
                fail(
                    "Navigation-app answer missing route/traffic/offline/"
                    "privacy/battery guidance",
                    failures,
                )
            if contains_any(response, ["checkout total", "food delivery"]):
                fail(
                    "Navigation-app answer mentioned checkout total or food delivery",
                    failures,
                )
        if name == "mixed_uber_gps_slam":
            check_equal(
                "relationship_required",
                c.get("relationship_required"),
                True,
                failures,
            )
            check_equal(
                "mixed_template_used",
                c.get("mixed_template_used"),
                True,
                failures,
            )

        if failures:
            failed += 1
            print(f"FAIL {name}: {question}")
            for item in failures:
                print(f"  - {item}")
            print(f"  response: {response[:240]}")
        else:
            print(f"PASS {name}: {question}")

    print("\nSummary")
    print(f"PASS: {total - failed}")
    print(f"FAIL: {failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
