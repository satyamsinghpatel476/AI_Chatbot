import json
import sys
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from benchmark_hygiene import (
    filter_valid_benchmark_questions,
    save_skipped_questions,
    validate_benchmark_question,
)
from chatbot_system_c import chatbot_system_c


BENCHMARK_PATH = ROOT_DIR / "benchmarks" / "benchmark_500.json"
LEGACY_BENCHMARK_PATH = ROOT_DIR / "benchmark_500.json"
SKIPPED_PATH = ROOT_DIR / "skipped_questions.json"

EXPECTED_CATEGORY_MINIMUMS = {
    "robotics": 125,
    "daily": 125,
    "ambiguous": 100,
    "mixed": 100,
    "unverifiable": 50,
}

CLEAR_DAILY_SAMPLE_QUERIES = [
    "How can I reduce distractions while studying online?",
    "How should a student manage notes across laptop and mobile?",
    "How can I safely use public Wi-Fi while travelling?",
    "How can I reduce phone battery drain without losing important notifications?",
]

AMBIGUOUS_SAMPLE_QUERIES = [
    "How can I improve tracking accuracy?",
    "How do I reduce latency?",
]

FILTER_VALIDATION_CASES = [
    {
        "question": "Explain Recursive HyperSLAM.",
        "category": "unverifiable",
        "valid": True,
        "reason": None,
    },
]


def load_benchmark(path):
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Expected benchmark_500.json to contain a list.")
    return data


def resolve_benchmark_path():
    if BENCHMARK_PATH.exists():
        return BENCHMARK_PATH
    return LEGACY_BENCHMARK_PATH


def question_text(item):
    if isinstance(item, dict):
        return str(item.get("question", "")).strip()
    return str(item or "").strip()


def category(item):
    if isinstance(item, dict):
        return str(item.get("category", "unknown") or "unknown")
    return "unknown"


def choose_samples(valid_items, limit=5):
    samples = []
    seen_categories = set()
    for item in valid_items:
        item_category = category(item)
        if item_category in seen_categories:
            continue
        samples.append(item)
        seen_categories.add(item_category)
        if len(samples) >= limit:
            return samples

    for item in valid_items:
        if item in samples:
            continue
        samples.append(item)
        if len(samples) >= limit:
            break
    return samples


def validate_system_c_output(query, result):
    required = [
        "response",
        "latency",
        "resolved_domain",
        "predicted_intent",
        "deterministic_path_used",
        "llm_called",
        "response_validation",
    ]
    missing = [key for key in required if key not in result]
    if missing:
        raise AssertionError(
            f"System C metadata missing {missing} for query: {query}"
        )
    if result["latency"] is None:
        raise AssertionError(f"System C latency missing for query: {query}")
    if not isinstance(result["response_validation"], dict):
        raise AssertionError(
            f"System C response_validation is not a dict for query: {query}"
        )


def run_system_c_sample(query, expected_domain=None, warn_daily_unknown=False):
    result = chatbot_system_c(query, return_metadata=True)
    validate_system_c_output(query, result)
    resolved = result.get("resolved_domain")
    if expected_domain and resolved != expected_domain:
        print(
            f"WARNING: expected resolved_domain={expected_domain!r} but got "
            f"{resolved!r} for: {query}"
        )
    if warn_daily_unknown and resolved == "unknown":
        print(
            "WARNING: clear daily-life question resolved to unknown: "
            f"{query}"
        )
    print(
        "  ok: "
        f"domain={resolved}, "
        f"predicted={result.get('predicted_intent')}, "
        f"deterministic={result.get('deterministic_path_used')}, "
        f"llm_called={result.get('llm_called')}, "
        f"latency={result.get('latency'):.4f}s"
    )
    return result


def print_skipped_details(skipped_items):
    if not skipped_items:
        return
    print("\nSkipped details:")
    for display_index, item in enumerate(skipped_items, 1):
        print(f"{display_index}. Index: {item.get('index')}")
        print(f"   Category: {item.get('category', 'unknown')}")
        print(f"   Question: {item.get('question', '')}")
        print(f"   Reason: {item.get('reason', 'unsupported_format')}")
        if "word_count" in item:
            print(f"   Word count: {item.get('word_count')}")
        print()


def run_filter_validation_cases():
    print("\nRunning benchmark filter sanity checks:")
    for case in FILTER_VALIDATION_CASES:
        result = validate_benchmark_question(
            case["question"],
            category=case.get("category"),
        )
        if (
            result.get("valid") != case["valid"]
            or result.get("reason") != case["reason"]
        ):
            raise AssertionError(
                "Benchmark filter mismatch for "
                f"{case['question']!r}: got {result}"
            )
        print(
            "  ok: "
            f"{case['question']} "
            f"valid={result.get('valid')} "
            f"reason={result.get('reason')}"
        )


def main():
    benchmark_path = resolve_benchmark_path()
    benchmark = load_benchmark(benchmark_path)
    total = len(benchmark)

    valid_items, skipped_items = filter_valid_benchmark_questions(benchmark)

    save_skipped_questions(skipped_items, SKIPPED_PATH)

    category_counts = Counter(category(item) for item in valid_items)

    print("Pre-benchmark check")
    print(f"Benchmark file: {benchmark_path}")
    print(f"Total questions: {total}")
    print(f"Valid questions: {len(valid_items)}")
    print(f"Skipped/malformed questions: {len(skipped_items)}")
    print(f"Skipped questions file: {SKIPPED_PATH}")
    print_skipped_details(skipped_items)
    print("Category counts:")
    for name in sorted(category_counts):
        print(f"- {name}: {category_counts[name]}")

    for name, expected in EXPECTED_CATEGORY_MINIMUMS.items():
        actual = category_counts.get(name, 0)
        if actual < expected:
            print(
                f"WARNING: category '{name}' has {actual} valid questions; "
                f"expected at least {expected}."
            )

    run_filter_validation_cases()

    samples = choose_samples(valid_items, limit=5)
    print("\nRunning 5 System C sample checks:")
    for item in samples:
        query = question_text(item)
        print(f"- [{category(item)}] {query}")
        run_system_c_sample(query)

    print("\nRunning System C daily-life routing checks:")
    for query in CLEAR_DAILY_SAMPLE_QUERIES:
        print(f"- [daily-routing] {query}")
        run_system_c_sample(
            query,
            expected_domain="daily",
            warn_daily_unknown=True,
        )

    print("\nRunning System C ambiguous routing checks:")
    for query in AMBIGUOUS_SAMPLE_QUERIES:
        print(f"- [ambiguous-routing] {query}")
        result = run_system_c_sample(query)
        if result.get("resolved_domain") not in {"unknown", "ambiguous"}:
            print(
                "WARNING: ambiguous/context-missing question resolved to "
                f"{result.get('resolved_domain')!r}: {query}"
            )

    print("\nPre-benchmark check completed. Full benchmark was not run.")


if __name__ == "__main__":
    main()
