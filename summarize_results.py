import argparse
import json
import statistics
from pathlib import Path


DEFAULT_RESULTS = Path("evaluator/results/results.json")
SYSTEMS = ["A", "B", "C"]


def load_results(path):
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Expected results file to contain a list.")
    return data


def number(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean(values):
    valid = [number(value) for value in values if number(value) is not None]
    return sum(valid) / len(valid) if valid else None


def median(values):
    valid = [number(value) for value in values if number(value) is not None]
    return statistics.median(valid) if valid else None


def fmt(value):
    if value is None:
        return "Not evaluated in this run"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def system_rows(entries, system):
    rows = []
    for entry in entries:
        result = entry.get("results", {}).get(system)
        if isinstance(result, dict):
            rows.append({
                "question": entry.get("question"),
                **result,
            })
    return rows


def dimension_averages(rows):
    keys = [
        "correctness",
        "task_fulfillment",
        "relevance",
        "completeness",
        "clarity",
        "calibration",
    ]
    output = {}
    for key in keys:
        output[key] = mean(
            row.get("dimension_scores", {}).get(key)
            for row in rows
            if isinstance(row.get("dimension_scores"), dict)
        )
    return output


def category_accuracy(rows):
    categories = [
        "robotics",
        "daily",
        "general",
        "unknown",
        "mixed",
        "ambiguous",
        "unverifiable",
    ]
    output = {}
    for category in categories:
        values = [
            row.get("accuracy")
            for row in rows
            if row.get("question_type") == category
        ]
        if values:
            output[category] = mean(values)
    return output


def subset_metric(rows, question_types, metric):
    subset = [
        row.get(metric)
        for row in rows
        if row.get("question_type") in question_types
    ]
    if not subset:
        return None
    return mean(subset)


def print_system_summary(system, rows):
    print(f"\nSystem {system}")
    print(f"Questions evaluated: {len(rows)}")
    print(f"Avg accuracy: {fmt(mean(row.get('accuracy') for row in rows))}")
    print(f"Median accuracy: {fmt(median(row.get('accuracy') for row in rows))}")
    print(f"Avg latency: {fmt(mean(row.get('latency') for row in rows))}")
    print(f"Hallucination count: {sum(1 for row in rows if row.get('hallucination'))}")
    print(f"Leakage count: {sum(1 for row in rows if row.get('leakage'))}")
    print(f"Contamination count: {sum(1 for row in rows if row.get('contamination'))}")
    print(f"False rejection count: {sum(1 for row in rows if row.get('false_rejection'))}")

    print("Dimension score averages:")
    for key, value in dimension_averages(rows).items():
        print(f"- {key}: {fmt(value)}")

    print("Category-wise accuracy:")
    category_values = category_accuracy(rows)
    if category_values:
        for category, value in category_values.items():
            print(f"- {category}: {fmt(value)}")
    else:
        print("- Not evaluated in this run")

    print("Special metrics:")
    memory = subset_metric(rows, {"personal_save", "personal_recall"}, "memory_recall")
    learning = subset_metric(rows, {"learning_save", "learning_recall"}, "knowledge_growth")
    mixed = subset_metric(rows, {"mixed"}, "cross_domain_robustness")
    intent = mean(
        row.get("intent_classification_accuracy")
        for row in rows
        if row.get("predicted_intent") is not None
        or row.get("corrected_predicted_intent") is not None
    )
    print(f"- Memory Recall: {fmt(memory)}")
    print(f"- Knowledge Growth: {fmt(learning)}")
    print(f"- Cross-Domain Robustness: {fmt(mixed)}")
    print(f"- Intent Classification: {fmt(intent)}")
    if system == "C":
        strict = mean(row.get("domain_resolution_accuracy_strict") for row in rows)
        relaxed = mean(row.get("domain_resolution_accuracy_relaxed") for row in rows)
        legacy = mean(row.get("domain_resolution_accuracy") for row in rows)
        print(f"- Domain Resolution: {fmt(legacy)}")
        print(f"- Domain Resolution Strict: {fmt(strict)}")
        print(f"- Domain Resolution Relaxed: {fmt(relaxed)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=str(DEFAULT_RESULTS))
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        raise SystemExit(f"Results file not found: {path}")

    entries = load_results(path)
    print(f"Results file: {path}")
    print(f"Benchmark entries: {len(entries)}")
    for system in SYSTEMS:
        rows = system_rows(entries, system)
        if rows:
            print_system_summary(system, rows)


if __name__ == "__main__":
    main()
