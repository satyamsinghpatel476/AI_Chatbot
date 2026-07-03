import csv
import json
import math
import os
import statistics

try:
    from evaluator.metrics import (
        METRIC_LABELS,
        PRIMARY_METRIC,
        comparison_metrics,
    )
except ModuleNotFoundError:
    from metrics import (
        METRIC_LABELS,
        PRIMARY_METRIC,
        comparison_metrics,
    )


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.join(BASE_DIR, "results", "runs")
SUMMARY_JSON = os.path.join(BASE_DIR, "results", "research_summary.json")
SUMMARY_CSV = os.path.join(BASE_DIR, "results", "research_summary.csv")


def mean_ci(values):
    if not values:
        return None, None
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, 0.0
    margin = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return mean, margin


def run_metrics(entries, system):
    rows = [
        entry["results"][system]
        for entry in entries
        if system in entry.get("results", {})
    ]
    if not rows:
        return None

    per_response = [comparison_metrics(row) for row in rows]
    run_summary = {}
    for metric in METRIC_LABELS:
        values = [
            item[metric]
            for item in per_response
            if item.get(metric) is not None
        ]
        run_summary[f"{metric}_pct"] = (
            statistics.mean(values) * 100 if values else None
        )
    return run_summary


def load_runs():
    runs = []
    if not os.path.isdir(RUN_DIR):
        return runs
    for filename in sorted(os.listdir(RUN_DIR)):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(RUN_DIR, filename)
        try:
            with open(path, encoding="utf-8") as f:
                entries = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        runs.append({"file": filename, "entries": entries})
    return runs


def main():
    runs = load_runs()
    if not runs:
        raise SystemExit("No archived runs found. Complete evaluator.py runs first.")

    output = {
        "run_count": len(runs),
        "primary_metric": PRIMARY_METRIC,
        "metric_labels": METRIC_LABELS,
        "systems": {},
        "note": "Values are measured across archived runs; CI is 95%.",
    }
    flat_rows = []

    for system in ["A", "B", "C"]:
        per_run = [
            metrics
            for run in runs
            if (metrics := run_metrics(run["entries"], system)) is not None
        ]
        system_summary = {}
        for metric in [f"{name}_pct" for name in METRIC_LABELS]:
            values = [
                row[metric] for row in per_run if row.get(metric) is not None
            ]
            mean, ci = mean_ci(values)
            system_summary[metric] = {
                "mean": round(mean, 3) if mean is not None else None,
                "ci95": round(ci, 3) if ci is not None else None,
            }
            flat_rows.append({
                "system": system,
                "metric": metric,
                "mean": system_summary[metric]["mean"],
                "ci95": system_summary[metric]["ci95"],
                "runs": len(values),
            })
        output["systems"][system] = system_summary

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["system", "metric", "mean", "ci95", "runs"],
        )
        writer.writeheader()
        writer.writerows(flat_rows)

    print(f"Wrote {SUMMARY_JSON}")
    print(f"Wrote {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
