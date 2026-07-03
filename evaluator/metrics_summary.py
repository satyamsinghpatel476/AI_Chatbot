import argparse
import csv
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "results"
DEFAULT_RESULTS_JSON = RESULT_DIR / "results.json"
DEFAULT_SUMMARY_JSON = RESULT_DIR / "metrics_summary.json"
DEFAULT_SUMMARY_CSV = RESULT_DIR / "metrics_summary.csv"

SYSTEMS = [
    ("A", "System A"),
    ("B", "System B"),
    ("C", "System C"),
]

CATEGORY_COLUMNS = [
    ("Robotics", "robotics"),
    ("Daily", "daily"),
    ("Ambiguous", "ambiguous"),
    ("Mixed", "mixed"),
    ("Unverifiable", "unverifiable"),
]

COMPOSITE_WEIGHTS = {
    "accuracy_component": 0.35,
    "contamination_resistance": 0.25,
    "hallucination_resistance": 0.15,
    "intent_component": 0.15,
    "latency_component": 0.10,
}

NULL_STRINGS = {"", "none", "null", "nan", "n/a", "na", "missing"}


def as_number(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_label(value):
    if value is None:
        return None
    label = str(value).strip().lower()
    if label in NULL_STRINGS:
        return None
    aliases = {
        "robot": "robotics",
        "robotic": "robotics",
        "daily_life": "daily",
        "daily-life": "daily",
        "consumer": "daily",
        "mixed_domain": "mixed",
        "cross_domain": "mixed",
        "hallucination": "unverifiable",
        "unverified": "unverifiable",
    }
    return aliases.get(label, label)


def mean(values):
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def clamp(value, low=0.0, high=10.0):
    return max(low, min(high, value))


def rounded(value, places=4):
    if value is None or value == "N/A":
        return "N/A" if value == "N/A" else None
    return round(float(value), places)


def format_value(value, places=2):
    if value is None or value == "N/A":
        return "N/A"
    return f"{float(value):.{places}f}"


def format_intent(value):
    if value == "N/A" or value is None:
        return "N/A"
    return f"{float(value):.3f}"


def get_system_result(entry, system_key, system_name):
    results = entry.get("results", {})
    if not isinstance(results, dict):
        return None
    result = results.get(system_key)
    if result is None:
        result = results.get(system_name)
    return result if isinstance(result, dict) else None


def get_question_type(entry, result):
    return normalize_label(
        result.get("question_type")
        or entry.get("question_type")
        or entry.get("category")
    )


def get_latency(result):
    latency = as_number(result.get("latency"))
    if latency is not None:
        return latency
    latency_ms = as_number(result.get("latency_ms"))
    if latency_ms is not None:
        return latency_ms / 1000.0
    return None


def flag_value(result, primary_key, fallback_key=None):
    value = as_number(result.get(primary_key))
    if value is None and fallback_key:
        value = as_number(result.get(fallback_key))
    return value


def count_flags(values):
    return sum(1 for value in values if value is not None and value > 0)


def collect_system_rows(entries, system_key, system_name):
    rows = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        result = get_system_result(entry, system_key, system_name)
        if result is None:
            continue
        rows.append({
            "entry": entry,
            "result": result,
            "question_type": get_question_type(entry, result),
            "accuracy": as_number(result.get("accuracy")),
            "latency": get_latency(result),
            "hallucination": flag_value(result, "hallucination"),
            "contamination": flag_value(
                result,
                "contamination",
                "context_contamination_rate",
            ),
            "expected_intent": normalize_label(result.get("expected_intent")),
            "predicted_intent": normalize_label(
                result.get("corrected_predicted_intent")
                or result.get("predicted_intent")
            ),
        })
    return rows


def intent_accuracy(rows):
    evaluable = [
        row for row in rows
        if row["expected_intent"] is not None
        and row["predicted_intent"] is not None
    ]
    if not evaluable:
        return "N/A", 0, 0
    correct = sum(
        1 for row in evaluable
        if row["expected_intent"] == row["predicted_intent"]
    )
    return correct / len(evaluable), correct, len(evaluable)


def context_switching_score(rows):
    mixed_rows = [row for row in rows if row["question_type"] == "mixed"]
    mixed_accuracy = mean([row["accuracy"] for row in mixed_rows])
    if mixed_accuracy is None:
        return "N/A", 0, None
    contamination_count = count_flags(
        [row["contamination"] for row in mixed_rows]
    )
    score = mixed_accuracy - (contamination_count * 0.5)
    return clamp(score), contamination_count, mixed_accuracy


def category_accuracy(rows):
    output = {}
    for label, question_type in CATEGORY_COLUMNS:
        output[label] = mean(
            [
                row["accuracy"]
                for row in rows
                if row["question_type"] == question_type
            ]
        )
    return output


def normalized_scores(values_by_system):
    valid_values = [
        value for value in values_by_system.values()
        if value is not None
    ]
    if not valid_values:
        return {system: None for system in values_by_system}
    worst = max(valid_values)
    if worst <= 0:
        return {
            system: (0.0 if value is not None else None)
            for system, value in values_by_system.items()
        }
    return {
        system: (
            clamp((value / worst) * 10.0)
            if value is not None
            else None
        )
        for system, value in values_by_system.items()
    }


def weighted_composite(components):
    available = {
        name: value
        for name, value in components.items()
        if value is not None
    }
    if not available:
        return None
    total_weight = sum(COMPOSITE_WEIGHTS[name] for name in available)
    if total_weight <= 0:
        return None
    return sum(
        (COMPOSITE_WEIGHTS[name] / total_weight) * value
        for name, value in available.items()
    )


def summarize(entries):
    summaries = {}
    for system_key, system_name in SYSTEMS:
        rows = collect_system_rows(entries, system_key, system_name)
        accuracies = [row["accuracy"] for row in rows]
        latencies = [row["latency"] for row in rows]
        hallucinations = [row["hallucination"] for row in rows]
        contaminations = [row["contamination"] for row in rows]

        intent, intent_correct, intent_total = intent_accuracy(rows)
        context_score, mixed_contaminations, mixed_accuracy = (
            context_switching_score(rows)
        )

        summaries[system_name] = {
            "rows": rows,
            "avg_accuracy": mean(accuracies),
            "avg_latency": mean(latencies),
            "hallucinations": count_flags(hallucinations),
            "hallucination_rate": mean(hallucinations),
            "contaminations": count_flags(contaminations),
            "contamination_rate": mean(contaminations),
            "intent_accuracy": intent,
            "intent_correct": intent_correct,
            "intent_evaluable_samples": intent_total,
            "context_switching_score": context_score,
            "mixed_accuracy": mixed_accuracy,
            "mixed_contaminations": mixed_contaminations,
            "category_accuracy": category_accuracy(rows),
        }

    contamination_norm = normalized_scores({
        system: summary["contamination_rate"]
        for system, summary in summaries.items()
    })
    hallucination_norm = normalized_scores({
        system: summary["hallucination_rate"]
        for system, summary in summaries.items()
    })
    latency_norm = normalized_scores({
        system: summary["avg_latency"]
        for system, summary in summaries.items()
    })

    for system, summary in summaries.items():
        intent = summary["intent_accuracy"]
        components = {
            "accuracy_component": summary["avg_accuracy"],
            "contamination_resistance": (
                None
                if contamination_norm[system] is None
                else 10.0 - contamination_norm[system]
            ),
            "hallucination_resistance": (
                None
                if hallucination_norm[system] is None
                else 10.0 - hallucination_norm[system]
            ),
            "intent_component": (
                None if intent == "N/A" else float(intent) * 10.0
            ),
            "latency_component": (
                None
                if latency_norm[system] is None
                else 10.0 - latency_norm[system]
            ),
        }
        summary["composite_components"] = components
        summary["overall_composite_score"] = weighted_composite(components)

    return summaries


def best_system(summaries, key, higher_is_better=True):
    values = {
        system: summary.get(key)
        for system, summary in summaries.items()
        if isinstance(summary.get(key), (int, float))
    }
    if not values:
        return {"system": "N/A", "value": "N/A"}
    best_value = (
        max(values.values()) if higher_is_better else min(values.values())
    )
    winners = [
        system for system, value in values.items()
        if value == best_value
    ]
    return {
        "system": ", ".join(winners),
        "value": best_value,
    }


def build_conclusion(summaries):
    return {
        "best_overall_system": best_system(
            summaries,
            "overall_composite_score",
            higher_is_better=True,
        ),
        "best_accuracy_system": best_system(
            summaries,
            "avg_accuracy",
            higher_is_better=True,
        ),
        "fastest_system": best_system(
            summaries,
            "avg_latency",
            higher_is_better=False,
        ),
        "lowest_hallucination_system": best_system(
            summaries,
            "hallucinations",
            higher_is_better=False,
        ),
        "lowest_contamination_system": best_system(
            summaries,
            "contaminations",
            higher_is_better=False,
        ),
        "best_context_switching_system": best_system(
            summaries,
            "context_switching_score",
            higher_is_better=True,
        ),
    }


def printable_system_rows(summaries):
    rows = []
    for _, system_name in SYSTEMS:
        summary = summaries[system_name]
        rows.append({
            "System": system_name,
            "Avg Accuracy": format_value(summary["avg_accuracy"]),
            "Avg Latency": format_value(summary["avg_latency"]),
            "Hallucinations": str(summary["hallucinations"]),
            "Contaminations": str(summary["contaminations"]),
            "Intent Accuracy": format_intent(summary["intent_accuracy"]),
            "Context Switching Score": format_value(
                summary["context_switching_score"]
            ),
            "Overall Composite Score": format_value(
                summary["overall_composite_score"]
            ),
        })
    return rows


def printable_category_rows(summaries):
    rows = []
    for _, system_name in SYSTEMS:
        category_scores = summaries[system_name]["category_accuracy"]
        row = {"System": system_name}
        for label, _ in CATEGORY_COLUMNS:
            row[label] = format_value(category_scores[label])
        rows.append(row)
    return rows


def print_table(rows, columns):
    widths = {
        column: max(len(column), *(len(str(row[column])) for row in rows))
        for column in columns
    }
    header = " | ".join(column.ljust(widths[column]) for column in columns)
    separator = "-+-".join("-" * widths[column] for column in columns)
    print(header)
    print(separator)
    for row in rows:
        print(
            " | ".join(
                str(row[column]).ljust(widths[column])
                for column in columns
            )
        )


def json_ready_summary(summaries, conclusion):
    systems = {}
    for system, summary in summaries.items():
        systems[system] = {
            "avg_accuracy": rounded(summary["avg_accuracy"]),
            "avg_latency": rounded(summary["avg_latency"]),
            "hallucinations": summary["hallucinations"],
            "hallucination_rate": rounded(summary["hallucination_rate"]),
            "contaminations": summary["contaminations"],
            "contamination_rate": rounded(summary["contamination_rate"]),
            "intent_accuracy": rounded(summary["intent_accuracy"]),
            "intent_correct": summary["intent_correct"],
            "intent_evaluable_samples": summary["intent_evaluable_samples"],
            "context_switching_score": rounded(
                summary["context_switching_score"]
            ),
            "mixed_accuracy": rounded(summary["mixed_accuracy"]),
            "mixed_contaminations": summary["mixed_contaminations"],
            "overall_composite_score": rounded(
                summary["overall_composite_score"]
            ),
            "composite_components": {
                key: rounded(value)
                for key, value in summary["composite_components"].items()
            },
            "category_accuracy": {
                label: rounded(value)
                for label, value in summary["category_accuracy"].items()
            },
        }
    return {
        "input_file": str(DEFAULT_RESULTS_JSON),
        "systems": systems,
        "conclusion": {
            key: {
                "system": value["system"],
                "value": rounded(value["value"]),
            }
            for key, value in conclusion.items()
        },
    }


def write_summary_csv(path, summaries, conclusion):
    fieldnames = [
        "section",
        "system",
        "metric",
        "value",
        "avg_accuracy",
        "avg_latency",
        "hallucinations",
        "contaminations",
        "intent_accuracy",
        "context_switching_score",
        "overall_composite_score",
        "robotics",
        "daily",
        "ambiguous",
        "mixed",
        "unverifiable",
    ]
    rows = []
    for _, system in SYSTEMS:
        summary = summaries[system]
        categories = summary["category_accuracy"]
        rows.append({
            "section": "system_summary",
            "system": system,
            "avg_accuracy": rounded(summary["avg_accuracy"]),
            "avg_latency": rounded(summary["avg_latency"]),
            "hallucinations": summary["hallucinations"],
            "contaminations": summary["contaminations"],
            "intent_accuracy": rounded(summary["intent_accuracy"]),
            "context_switching_score": rounded(
                summary["context_switching_score"]
            ),
            "overall_composite_score": rounded(
                summary["overall_composite_score"]
            ),
            "robotics": rounded(categories["Robotics"]),
            "daily": rounded(categories["Daily"]),
            "ambiguous": rounded(categories["Ambiguous"]),
            "mixed": rounded(categories["Mixed"]),
            "unverifiable": rounded(categories["Unverifiable"]),
        })

    for metric, result in conclusion.items():
        rows.append({
            "section": "conclusion",
            "metric": metric,
            "system": result["system"],
            "value": rounded(result["value"]),
        })

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summaries, conclusion):
    system_columns = [
        "System",
        "Avg Accuracy",
        "Avg Latency",
        "Hallucinations",
        "Contaminations",
        "Intent Accuracy",
        "Context Switching Score",
        "Overall Composite Score",
    ]
    category_columns = [
        "System",
        "Robotics",
        "Daily",
        "Ambiguous",
        "Mixed",
        "Unverifiable",
    ]

    print("\nSystem comparison")
    print_table(printable_system_rows(summaries), system_columns)

    print("\nCategory-wise accuracy")
    print_table(printable_category_rows(summaries), category_columns)

    print("\nAutomatic conclusion")
    labels = {
        "best_overall_system": "Best overall system",
        "best_accuracy_system": "Best accuracy system",
        "fastest_system": "Fastest system",
        "lowest_hallucination_system": "Lowest hallucination system",
        "lowest_contamination_system": "Lowest contamination system",
        "best_context_switching_system": "Best context switching system",
    }
    for key, label in labels.items():
        result = conclusion[key]
        print(f"{label}: {result['system']} ({format_value(result['value'])})")


def load_entries(path):
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("entries"), list):
        return data["entries"]
    raise ValueError("Expected results.json to contain a list of result entries.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze existing evaluator results without generating answers."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_RESULTS_JSON),
        help="Path to evaluator results.json.",
    )
    parser.add_argument(
        "--json-output",
        default=str(DEFAULT_SUMMARY_JSON),
        help="Path for metrics_summary.json.",
    )
    parser.add_argument(
        "--csv-output",
        default=str(DEFAULT_SUMMARY_CSV),
        help="Path for metrics_summary.csv.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    json_output = Path(args.json_output)
    csv_output = Path(args.csv_output)

    entries = load_entries(input_path)
    summaries = summarize(entries)
    conclusion = build_conclusion(summaries)

    print_summary(summaries, conclusion)

    json_output.parent.mkdir(parents=True, exist_ok=True)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    with json_output.open("w", encoding="utf-8") as handle:
        output = json_ready_summary(summaries, conclusion)
        output["input_file"] = str(input_path)
        json.dump(output, handle, indent=2)
    write_summary_csv(csv_output, summaries, conclusion)

    print(f"\nWrote {json_output}")
    print(f"Wrote {csv_output}")


if __name__ == "__main__":
    main()
