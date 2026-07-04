import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_hygiene import validate_benchmark_question


VALID_CASES = [
    "My name is Satyam.",
    "My favorite app is Uber.",
    "My favorite robot is TurtleBot3.",
    "Remember that I live in Chennai.",
    "What is SLAM?",
    "What is ROS2?",
    "What is odometry?",
    "Difference between ROS and ROS2.",
    "Explain PID.",
    "Define EKF.",
]

INVALID_CASES = [
    "terms.",
    "meaning separately.",
    "without contamination.",
    "direct, indirect, or unsupported.",
    ".",
]


def check_case(question, expected_valid):
    result = validate_benchmark_question(question)
    passed = bool(result.get("valid")) is expected_valid
    status = "PASS" if passed else "FAIL"
    print(
        f"{status}: {question!r} -> "
        f"valid={result.get('valid')} reason={result.get('reason')}"
    )
    return passed


def main():
    passed = 0
    failed = 0

    print("Valid cases")
    for question in VALID_CASES:
        if check_case(question, True):
            passed += 1
        else:
            failed += 1

    print("\nInvalid cases")
    for question in INVALID_CASES:
        if check_case(question, False):
            passed += 1
        else:
            failed += 1

    print("\nSummary")
    print(f"PASS: {passed}")
    print(f"FAIL: {failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
