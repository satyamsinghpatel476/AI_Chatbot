import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKIPPED_PATH = PROJECT_ROOT / "skipped_questions.json"


def main():
    if not SKIPPED_PATH.exists():
        print(
            "No skipped_questions.json found. Run scripts/prebenchmark_check.py "
            "or evaluator first."
        )
        return

    try:
        with open(SKIPPED_PATH, encoding="utf-8") as handle:
            skipped = json.load(handle)
    except json.JSONDecodeError as exc:
        print(f"Could not read skipped_questions.json: {exc}")
        return

    if not isinstance(skipped, list):
        print("skipped_questions.json is not a list.")
        return

    print(f"Total skipped: {len(skipped)}")
    if not skipped:
        return

    print()
    for item in skipped:
        if not isinstance(item, dict):
            print("Index: unknown")
            print("Category: unknown")
            print(f"Question: {item}")
            print("Reason: unsupported_format")
            print("Word count: unknown")
            print()
            continue
        print(f"Index: {item.get('index')}")
        print(f"Category: {item.get('category', 'unknown')}")
        print(f"Question: {item.get('question', '')}")
        print(f"Reason: {item.get('reason', 'unsupported_format')}")
        print(f"Word count: {item.get('word_count', 'unknown')}")
        print()


if __name__ == "__main__":
    main()
