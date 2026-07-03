from chatbot_system_c import chatbot_system_c
from research_core import clear_experiment_memory


CASES = [
    (
        "My name is Satyam.",
        ["remember", "satyam"],
        "save_name",
    ),
    (
        "What is my name?",
        ["satyam"],
        "recall_name",
    ),
    (
        "My favorite app is Uber.",
        ["remember", "uber"],
        "save_favorite_app",
    ),
    (
        "What is my favorite app?",
        ["uber"],
        "recall_favorite_app",
    ),
    (
        "My favorite robot is TurtleBot3.",
        ["remember", "turtlebot3"],
        "save_favorite_robot",
    ),
    (
        "Which robot do I like?",
        ["turtlebot3"],
        "recall_favorite_robot",
    ),
]


def run_case(question, expected_terms, label):
    result = chatbot_system_c(question, return_metadata=True)
    response = str(result.get("response", ""))
    lowered = response.lower()
    missing = [term for term in expected_terms if term not in lowered]
    passed = not missing
    status = "PASS" if passed else "FAIL"
    print(f"{status} {label}: {question}")
    print(f"  Response: {response}")
    if missing:
        print(f"  Missing: {', '.join(missing)}")
    return passed


def main():
    clear_experiment_memory()

    passed = 0
    failed = 0
    try:
        for question, expected_terms, label in CASES:
            if run_case(question, expected_terms, label):
                passed += 1
            else:
                failed += 1

        print("\nSummary")
        print(f"PASS: {passed}")
        print(f"FAIL: {failed}")
        if failed:
            raise SystemExit(1)
    finally:
        clear_experiment_memory()


if __name__ == "__main__":
    main()
