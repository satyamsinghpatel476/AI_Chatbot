from chatbot_system_c import chatbot_system_c


PURE_ROBOTICS = [
    "What are the limitations of SLAM?",
    "How can a beginner debug AMCL localization?",
    "Why does LiDAR mapping fail in real-world environments?",
    "What are the common causes of error in EKF sensor fusion?",
]

PURE_DAILY = [
    "How can I reduce screen time?",
    "How can I choose a ride-sharing app?",
    "Why do food delivery prices change?",
    "How can I safely use public Wi-Fi?",
]

TRUE_MIXED = [
    "Compare SLAM and ride-sharing apps.",
    "Can food delivery routes improve robot path planning?",
    "Can GPS from ride apps replace SLAM?",
    "What is the relationship between LiDAR mapping and Google Maps?",
]

MIXED_MARKERS = [
    "relationship type",
    "daily-life perspective",
    "daily life perspective",
]

ROBOTICS_TERMS = [
    "slam",
    "lidar",
    "amcl",
    "ekf",
    "odometry",
]

RELATIONSHIP_WORDS = [
    "relationship type",
    "direct",
    "indirect",
    "unsupported",
    "analogy",
]


def contains_any(text, terms):
    lowered = text.lower()
    return any(term in lowered for term in terms)


def check_case(question, expectations):
    result = chatbot_system_c(question, return_metadata=True)
    response = result.get("response", "")
    failures = []

    expected_domain = expectations.get("domain")
    if result.get("resolved_domain") != expected_domain:
        failures.append(
            f"resolved_domain={result.get('resolved_domain')!r}, "
            f"expected {expected_domain!r}"
        )

    if result.get("mixed_template_used") != expectations.get("mixed_template"):
        failures.append(
            f"mixed_template_used={result.get('mixed_template_used')!r}, "
            f"expected {expectations.get('mixed_template')!r}"
        )

    if "relationship_required" in expectations and (
        result.get("relationship_required")
        != expectations["relationship_required"]
    ):
        failures.append(
            f"relationship_required={result.get('relationship_required')!r}, "
            f"expected {expectations['relationship_required']!r}"
        )

    forbidden = expectations.get("forbidden", [])
    if contains_any(response, forbidden):
        failures.append(f"response contains forbidden terms: {forbidden}")

    required_any = expectations.get("required_any", [])
    if required_any and not contains_any(response, required_any):
        failures.append(f"response lacks any of: {required_any}")

    return failures, result


def main():
    checks = []
    for question in PURE_ROBOTICS:
        checks.append((
            question,
            {
                "domain": "robotics",
                "mixed_template": False,
                "relationship_required": False,
                "forbidden": MIXED_MARKERS,
            },
        ))
    for question in PURE_DAILY:
        checks.append((
            question,
            {
                "domain": "daily",
                "mixed_template": False,
                "relationship_required": False,
                "forbidden": ROBOTICS_TERMS + MIXED_MARKERS,
            },
        ))
    for question in TRUE_MIXED:
        checks.append((
            question,
            {
                "domain": "mixed",
                "mixed_template": True,
                "relationship_required": True,
                "required_any": RELATIONSHIP_WORDS,
            },
        ))

    passed = 0
    failed = 0
    for index, (question, expectations) in enumerate(checks, 1):
        failures, result = check_case(question, expectations)
        if failures:
            failed += 1
            print(f"FAIL {index}: {question}")
            for failure in failures:
                print(f"  - {failure}")
            print(f"  response: {result.get('response', '')[:240]}")
        else:
            passed += 1
            print(f"PASS {index}: {question}")

    print("\nSummary")
    print(f"PASS: {passed}")
    print(f"FAIL: {failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
