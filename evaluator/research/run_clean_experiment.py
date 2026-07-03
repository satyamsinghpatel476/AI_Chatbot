import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from evaluator.research.benchmarks import BENCHMARK_DIR, write_benchmarks
from evaluator.research.scoring import (
    score_knowledge_rows,
    score_memory_rows,
    score_relationship_rows,
)
from evaluator.research.statistics import (
    describe,
    describe_latency,
    mcnemar_exact,
    paired_continuous,
)
from evaluator.metrics import normalize_intent_label
from llm_runtime import prepare_uncached_benchmark_runtime


ROOT_DIR = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT_DIR / "evaluator" / "results"
DEFAULT_EXPERIMENT_DIR = Path("/tmp/ai_robotics_assistant_clean_experiment")
SYSTEMS = ["A", "B", "C"]
ABLATIONS = [f"C{index}" for index in range(7)]
SUITE_FILES = {
    "context": "context_contamination.json",
    "memory": "memory_recall.json",
    "knowledge": "knowledge_growth.json",
    "intent": "intent_classification.json",
    "cross": "cross_domain_robustness.json",
}
SUITE_ALIASES = {
    "all": ["context", "memory", "knowledge", "intent", "cross"],
    "context": ["context"],
    "contamination": ["context"],
    "memory": ["memory"],
    "knowledge": ["knowledge"],
    "intent": ["intent"],
    "cross": ["cross"],
    "robustness": ["cross"],
}


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def run_command(command, *, env=None):
    subprocess.run(
        command,
        cwd=ROOT_DIR,
        env=env,
        check=True,
    )


def configure_uncached_research_runtime():
    prepare_uncached_benchmark_runtime()
    os.environ["BENCHMARK_FORCE_LLM"] = "1"
    os.environ["BENCHMARK_DISABLE_DETERMINISTIC_SHORTCUTS"] = "1"
    os.environ["BENCHMARK_FORCE_RERUN"] = "1"
    print(
        "Research benchmark cache disabled. The Mistral cache file was "
        "deleted if present, existing raw worker outputs will be ignored, "
        "and systems will be re-run."
    )


def worker_command(
    *,
    system,
    mode,
    benchmark,
    state_dir,
    output,
    temperature,
    seed,
    timeout,
    base_state_dir=None,
):
    command = [
        sys.executable,
        "-m",
        "evaluator.research.worker",
        "--system",
        system,
        "--mode",
        mode,
        "--benchmark",
        str(benchmark),
        "--state-dir",
        str(state_dir),
        "--output",
        str(output),
        "--temperature",
        str(temperature),
        "--seed",
        str(seed),
        "--timeout",
        str(timeout),
    ]
    if base_state_dir:
        command.extend(["--base-state-dir", str(base_state_dir)])
    return command


def output_matches_cases(rows, expected_ids, system):
    if not isinstance(rows, list):
        return False
    if len(rows) != len(expected_ids):
        return False
    row_ids = [row.get("id") for row in rows]
    systems = {row.get("system") for row in rows}
    return row_ids == expected_ids and systems <= {system}


def ensure_worker_output(expected_ids=None, **kwargs):
    output = Path(kwargs["output"])
    force_rerun = os.environ.get("BENCHMARK_FORCE_RERUN") == "1"
    if output.exists() and not force_rerun:
        rows = read_json(output)
        if expected_ids is None or output_matches_cases(
            rows,
            expected_ids,
            kwargs["system"],
        ):
            return rows
    run_command(worker_command(**kwargs))
    return read_json(output)


def selected_suite_names(suite):
    key = str(suite or "all").strip().lower()
    if key not in SUITE_ALIASES:
        raise ValueError(
            "--suite must be one of: "
            + ", ".join(sorted(SUITE_ALIASES))
        )
    return SUITE_ALIASES[key]


def parse_systems(value):
    selected = [
        item.strip().upper()
        for item in str(value or ",".join(SYSTEMS)).split(",")
        if item.strip()
    ]
    invalid = [item for item in selected if item not in {"A", "B", "C"}]
    if invalid:
        raise ValueError(f"Unknown systems: {', '.join(invalid)}")
    return selected or list(SYSTEMS)


def research_mode_tag():
    return (
        "full"
        if os.environ.get("FULL_RESEARCH_MODE") == "1"
        else "fast"
    )


def experiment_metadata(seed, temperature, timeout, suite, limit):
    model = {}
    try:
        raw = subprocess.check_output(
            ["curl", "-fsS", "http://127.0.0.1:11434/api/tags"],
            text=True,
        )
        tags = json.loads(raw).get("models", [])
        model = next(
            (item for item in tags if item.get("name") == "mistral:latest"),
            {},
        )
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        pass
    return {
        "created_at": datetime.now().isoformat(),
        "seed": seed,
        "temperature": temperature,
        "timeout_seconds": timeout,
        "model": model,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "system_order": SYSTEMS,
        "suite": suite,
        "limit": limit,
        "fast_research_mode": os.environ.get("FULL_RESEARCH_MODE") != "1"
        and os.environ.get("FAST_RESEARCH_MODE", "1") != "0",
        "llm_cache_disabled": os.environ.get("DISABLE_LLM_CACHE") == "1",
        "benchmark_force_llm": os.environ.get("BENCHMARK_FORCE_LLM") == "1",
        "benchmark_force_rerun": os.environ.get("BENCHMARK_FORCE_RERUN") == "1",
        "historical_results_included": False,
    }


def prepare_benchmark_paths(experiment_dir, suite, limit, smoke):
    selected = selected_suite_names(suite)
    benchmark_paths = {
        name: Path(BENCHMARK_DIR) / filename
        for name, filename in SUITE_FILES.items()
        if name in selected
    }

    if limit is not None:
        limited_dir = experiment_dir / "limited_benchmarks"
        limited_dir.mkdir(exist_ok=True)
        loaded = {
            name: read_json(path)
            for name, path in benchmark_paths.items()
        }
        selected_rows = {name: [] for name in loaded}
        total = 0
        while total < limit and any(
            len(selected_rows[name]) < len(rows)
            for name, rows in loaded.items()
        ):
            for name, rows in loaded.items():
                if total >= limit:
                    break
                index = len(selected_rows[name])
                if index < len(rows):
                    selected_rows[name].append(rows[index])
                    total += 1
        for name, rows in selected_rows.items():
            target = limited_dir / SUITE_FILES[name]
            write_json(target, rows)
            benchmark_paths[name] = target
        return benchmark_paths

    if smoke:
        smoke_dir = experiment_dir / "smoke_benchmarks"
        smoke_dir.mkdir(exist_ok=True)
        limits = {
            "context": 4,
            "memory": 2,
            "knowledge": 2,
            "intent": 12,
            "cross": 5,
        }
        for name, path in list(benchmark_paths.items()):
            target = smoke_dir / path.name
            write_json(target, read_json(path)[:limits[name]])
            benchmark_paths[name] = target

    return benchmark_paths


def stable_raw_runs(
    experiment_dir,
    seed,
    temperature,
    timeout,
    smoke,
    suite,
    limit,
):
    mode_tag = research_mode_tag()
    raw_dir = experiment_dir / f"raw_{mode_tag}"
    state_root = experiment_dir / f"state_{mode_tag}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    state_root.mkdir(parents=True, exist_ok=True)

    benchmark_paths = prepare_benchmark_paths(
        experiment_dir,
        suite,
        limit,
        smoke,
    )

    all_rows = defaultdict(list)
    for system in SYSTEMS:
        system_state = state_root / system
        for mode in ["context", "memory", "cross", "intent"]:
            if mode not in benchmark_paths:
                continue
            cases = read_json(benchmark_paths[mode])
            output = raw_dir / f"{system}_{mode}.json"
            rows = ensure_worker_output(
                system=system,
                mode=mode,
                benchmark=benchmark_paths[mode],
                state_dir=system_state / mode,
                output=output,
                temperature=temperature,
                seed=seed,
                timeout=timeout,
                expected_ids=[row["id"] for row in cases],
            )
            all_rows[mode].extend(rows)

        if "knowledge" not in benchmark_paths:
            continue
        knowledge_cases = read_json(benchmark_paths["knowledge"])
        teach_state = system_state / "knowledge_teach"
        teach_output = raw_dir / f"{system}_knowledge_teach.json"
        teach_rows = ensure_worker_output(
            system=system,
            mode="knowledge-teach",
            benchmark=benchmark_paths["knowledge"],
            state_dir=teach_state,
            output=teach_output,
            temperature=temperature,
            seed=seed,
            timeout=timeout,
            expected_ids=[row["id"] for row in knowledge_cases],
        )
        base_state = system_state / "knowledge_base"
        if base_state.exists():
            shutil.rmtree(base_state)
        shutil.copytree(teach_state, base_state)
        recall_output = raw_dir / f"{system}_knowledge_recall.json"
        recall_rows = ensure_worker_output(
            system=system,
            mode="knowledge-recall",
            benchmark=benchmark_paths["knowledge"],
            state_dir=system_state / "knowledge_recall_work",
            base_state_dir=base_state,
            output=recall_output,
            temperature=temperature,
            seed=seed,
            timeout=timeout,
            expected_ids=[row["id"] for row in knowledge_cases],
        )
        teach_by_id = {row["id"]: row for row in teach_rows}
        for row in recall_rows:
            row.update(teach_by_id.get(row["id"], {}))
        all_rows["knowledge"].extend(recall_rows)
    return all_rows


def score_intent_rows(rows):
    scored = []
    for row in rows:
        predicted = normalize_intent_label(row.get("predicted_intent"))
        gold = normalize_intent_label(row["gold_intent"])
        correct = (
            int(predicted == gold)
            if predicted is not None
            else None
        )
        scored.append({
            **row,
            "gold_intent": gold,
            "predicted_intent": predicted,
            "intent_correct": correct,
        })
    return scored


def score_stable_runs(raw, experiment_dir, seed, timeout):
    scored_dir = experiment_dir / f"scored_{research_mode_tag()}"
    scored_dir.mkdir(exist_ok=True)
    paths = {
        "context": scored_dir / "context.json",
        "cross": scored_dir / "cross.json",
        "memory": scored_dir / "memory.json",
        "knowledge": scored_dir / "knowledge.json",
        "intent": scored_dir / "intent.json",
    }

    context = score_relationship_rows(
        raw["context"],
        seed=seed + 1000,
        timeout=timeout,
    )
    write_json(paths["context"], context)

    cross = score_relationship_rows(
        raw["cross"],
        seed=seed + 2000,
        timeout=timeout,
    )
    write_json(paths["cross"], cross)

    memory = score_memory_rows(raw["memory"])
    knowledge = score_knowledge_rows(raw["knowledge"])
    intent = score_intent_rows(raw["intent"])
    write_json(paths["memory"], memory)
    write_json(paths["knowledge"], knowledge)
    write_json(paths["intent"], intent)
    return {
        "context": context,
        "cross": cross,
        "memory": memory,
        "knowledge": knowledge,
        "intent": intent,
    }


def row_metric(rows, system, field):
    return [
        row.get(field)
        for row in rows
        if row.get("system") == system
    ]


def intent_summary(rows, system):
    system_rows = [
        row for row in rows
        if row["system"] == system and row.get("predicted_intent") is not None
    ]
    if not system_rows:
        return {
            "n": 0,
            "accuracy": None,
            "accuracy_ci95_low": None,
            "accuracy_ci95_high": None,
            "macro_f1": None,
            "macro_precision": None,
            "macro_recall": None,
        }
    gold = [row["gold_intent"] for row in system_rows]
    predicted = [row["predicted_intent"] for row in system_rows]
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold,
        predicted,
        labels=["robotics", "daily", "personal", "mixed", "general", "unknown"],
        average="macro",
        zero_division=0,
    )
    correct = [
        int(gold_value == predicted_value)
        for gold_value, predicted_value in zip(gold, predicted)
    ]
    accuracy_description = describe(correct, seed=20260623)
    return {
        "n": len(system_rows),
        "accuracy": float(accuracy_score(gold, predicted)),
        "accuracy_ci95_low": accuracy_description["ci95_low"],
        "accuracy_ci95_high": accuracy_description["ci95_high"],
        "macro_f1": float(f1),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
    }


def build_confusion_rows(intent_rows):
    labels = ["robotics", "daily", "personal", "mixed", "general", "unknown"]
    output = []
    for system in SYSTEMS:
        rows = [
            row for row in intent_rows
            if row["system"] == system and row.get("predicted_intent") is not None
        ]
        if not rows:
            output.append({
                "system": system,
                "gold_intent": "N/A",
                **{label: None for label in labels},
            })
            continue
        matrix = confusion_matrix(
            [row["gold_intent"] for row in rows],
            [row["predicted_intent"] for row in rows],
            labels=labels,
        )
        for gold, counts in zip(labels, matrix):
            output.append({
                "system": system,
                "gold_intent": gold,
                **{
                    label: int(count)
                    for label, count in zip(labels, counts)
                },
            })
    return output


def build_category_summary(scored, seed):
    rows = []
    for system in SYSTEMS:
        context = {
            "context_contamination_rate": describe(
                row_metric(scored["context"], system, "contaminated"),
                seed=seed,
            ),
            "false_rejection_rate": describe(
                row_metric(scored["context"], system, "false_rejection"),
                seed=seed + 1,
            ),
            "safe_explanation_rate": describe(
                row_metric(scored["context"], system, "safe_explanation"),
                seed=seed + 2,
            ),
        }
        memory = {
            "memory_recall": describe(
                row_metric(scored["memory"], system, "exact_recall"),
                seed=seed + 3,
            ),
            "incorrect_recall_rate": describe(
                row_metric(scored["memory"], system, "incorrect_recall"),
                seed=seed + 4,
            ),
            "missing_recall_rate": describe(
                row_metric(scored["memory"], system, "missing_recall"),
                seed=seed + 5,
            ),
        }
        knowledge = {
            "knowledge_growth": describe(
                row_metric(scored["knowledge"], system, "knowledge_growth_accuracy"),
                seed=seed + 6,
            ),
            "semantic_recall_accuracy": describe(
                row_metric(scored["knowledge"], system, "semantic_recall_accuracy"),
                seed=seed + 7,
            ),
            "false_memory_rate": describe(
                row_metric(scored["knowledge"], system, "false_memory"),
                seed=seed + 8,
            ),
        }
        cross = {
            "cross_domain_robustness": describe(
                row_metric(scored["cross"], system, "cross_domain_robustness"),
                seed=seed + 9,
            ),
            "false_rejection_rate": describe(
                row_metric(scored["cross"], system, "false_rejection"),
                seed=seed + 10,
            ),
        }
        intent = intent_summary(scored["intent"], system)
        latency_values = []
        for suite_rows in scored.values():
            latency_values.extend(row_metric(suite_rows, system, "latency_ms"))
        latency = describe_latency(latency_values, seed=seed + 11)
        sections = {
            "context_contamination": context,
            "memory_recall": memory,
            "knowledge_growth": knowledge,
            "cross_domain_robustness": cross,
            "intent_classification": {
                "intent_classification_accuracy": intent,
                "intent_macro_f1": {
                    "n": intent["n"],
                    "mean": intent["macro_f1"],
                    "median": None,
                    "std": None,
                    "ci95_low": None,
                    "ci95_high": None,
                    "p95": None,
                },
            },
            "latency": {"latency_ms": latency},
        }
        for category, metrics in sections.items():
            for metric, summary in metrics.items():
                if metric == "intent_classification_accuracy":
                    rows.append({
                        "system": system,
                        "category": category,
                        "metric": "intent_classification_accuracy",
                        "n": summary["n"],
                        "mean": summary["accuracy"],
                        "median": None,
                        "std": None,
                        "ci95_low": summary["accuracy_ci95_low"],
                        "ci95_high": summary["accuracy_ci95_high"],
                        "p95": None,
                        "macro_f1": summary["macro_f1"],
                        "macro_precision": summary["macro_precision"],
                        "macro_recall": summary["macro_recall"],
                    })
                else:
                    rows.append({
                        "system": system,
                        "category": category,
                        "metric": metric,
                        **summary,
                    })
    return rows


def paired_rows(rows, first, second, field):
    by_case = defaultdict(dict)
    for row in rows:
        by_case[row["id"]][row["system"]] = row.get(field)
    first_values = []
    second_values = []
    for values in by_case.values():
        if first in values and second in values:
            first_values.append(values[first])
            second_values.append(values[second])
    return first_values, second_values


def statistical_tests(scored):
    tests = {}
    for first, second in [("A", "B"), ("A", "C"), ("B", "C")]:
        key = f"{first}_vs_{second}"
        tests[key] = {}
        for suite, field, binary in [
            ("context", "contaminated", True),
            ("context", "false_rejection", True),
            ("memory", "exact_recall", True),
            ("knowledge", "knowledge_growth_accuracy", False),
            ("cross", "cross_domain_robustness", False),
            ("intent", "intent_correct", True),
        ]:
            first_values, second_values = paired_rows(
                scored[suite],
                first,
                second,
                field,
            )
            tests[key][f"{suite}.{field}"] = (
                mcnemar_exact(first_values, second_values)
                if binary
                else paired_continuous(first_values, second_values)
            )
        latency_tests = {}
        for suite in ["context", "memory", "knowledge", "cross"]:
            first_values, second_values = paired_rows(
                scored[suite],
                first,
                second,
                "latency_ms",
            )
            latency_tests[suite] = paired_continuous(
                first_values,
                second_values,
            )
        tests[key]["latency_ms"] = latency_tests
    return tests


def flatten_fresh_results(scored):
    rows = []
    for suite, suite_rows in scored.items():
        for row in suite_rows:
            rows.append({
                "suite": suite,
                "case_id": row["id"],
                "system": row["system"],
                "subtype": row.get("subtype") or row.get("gold_relationship"),
                "query": row.get("query") or row.get("recall"),
                "response": row.get("response")
                or row.get("exact_recall_response"),
                "latency_ms": row.get("latency_ms"),
                "total_latency": row.get("total_latency"),
                "total_latency_ms": row.get("total_latency_ms"),
                "llm_called": row.get("llm_called"),
                "cache_used": row.get("cache_used"),
                "deterministic_path_used": row.get("deterministic_path_used"),
                "contaminated": row.get("contaminated"),
                "false_rejection": row.get("false_rejection"),
                "safe_explanation": row.get("safe_explanation"),
                "memory_recall": row.get("exact_recall"),
                "knowledge_growth": row.get("knowledge_growth_accuracy"),
                "semantic_recall_accuracy": row.get("semantic_recall_accuracy"),
                "false_memory": row.get("false_memory"),
                "cross_domain_robustness": row.get("cross_domain_robustness"),
                "intent_correct": row.get("intent_correct"),
                "predicted_intent": row.get("predicted_intent"),
                "gold_intent": row.get("gold_intent"),
                "requires_human_review": row.get("requires_human_review"),
            })
    return rows


def stratified_ablation_benchmarks(experiment_dir, smoke):
    target = experiment_dir / "ablation_benchmarks"
    target.mkdir(exist_ok=True)
    hard_cases = [
        (
            "abl-hard-01",
            "Can Uber GPS replace SLAM for a delivery robot?",
            "incompatible",
            ["cannot replace SLAM", "onboard localization", "uncertainty"],
        ),
        (
            "abl-hard-02",
            "Can restaurant ratings improve obstacle avoidance?",
            "incompatible",
            ["ratings are not geometry", "obstacle measurements", "onboard perception"],
        ),
        (
            "abl-hard-03",
            "Can social media replace robot perception?",
            "incompatible",
            ["social posts are not sensor data", "perception sensors"],
        ),
        (
            "abl-hard-04",
            "Can food delivery routes train robot navigation?",
            "conditional",
            ["authorized routes", "timestamps", "domain mismatch", "onboard sensing"],
        ),
        (
            "abl-hard-05",
            "Can Google Maps replace LiDAR on a mobile robot?",
            "incompatible",
            ["cannot replace LiDAR", "range measurements", "local obstacles"],
        ),
        (
            "abl-hard-06",
            "Can ride-hailing drivers improve robot path planning?",
            "conditional",
            ["authorized trajectories", "labels", "not obstacle sensing"],
        ),
        (
            "abl-hard-07",
            "Are app permissions and ROS permissions interchangeable?",
            "incompatible",
            ["different permission systems", "ROS configuration", "safety"],
        ),
        (
            "abl-hard-08",
            "When I say navigation tracking, do I mean phone navigation or robot navigation?",
            "uncertain",
            ["ambiguous", "phone navigation", "robot navigation"],
        ),
    ]
    context_subset = [
        {
            "id": case_id,
            "suite": "context_contamination",
            "subtype": "hard_ablation_cross_domain",
            "context": [],
            "query": query,
            "current_domain": "mixed",
            "gold_relationship": relationship,
            "required_points": required,
            "forbidden_claims": [
                "consumer apps are physical robot sensors",
                "external app data replaces onboard safety or perception",
            ],
        }
        for case_id, query, relationship, required in hard_cases
    ]
    cross_subset = [
        {
            **row,
            "id": row["id"].replace("abl-hard", "abl-cross"),
            "suite": "cross_domain_robustness",
        }
        for row in context_subset
    ]
    if smoke:
        context_subset = context_subset[:4]
        cross_subset = cross_subset[:4]
    paths = {
        "context": target / "context.json",
        "cross": target / "cross.json",
        "memory": Path(BENCHMARK_DIR) / "memory_recall.json",
        "knowledge": Path(BENCHMARK_DIR) / "knowledge_growth.json",
    }
    write_json(paths["context"], context_subset)
    write_json(paths["cross"], cross_subset)
    if smoke:
        for name in ["memory", "knowledge"]:
            reduced = target / f"{name}.json"
            write_json(reduced, read_json(paths[name])[:2])
            paths[name] = reduced
    return paths


def run_ablations(experiment_dir, seed, temperature, timeout, smoke):
    paths = stratified_ablation_benchmarks(experiment_dir, smoke)
    mode_tag = research_mode_tag()
    raw_dir = experiment_dir / f"ablation_raw_{mode_tag}"
    state_root = experiment_dir / f"ablation_state_{mode_tag}"
    raw = defaultdict(list)
    for system in ABLATIONS:
        for mode in ["context", "memory", "cross"]:
            rows = ensure_worker_output(
                system=system,
                mode=mode,
                benchmark=paths[mode],
                state_dir=state_root / system / mode,
                output=raw_dir / f"{system}_{mode}.json",
                temperature=temperature,
                seed=seed,
                timeout=timeout,
            )
            raw[mode].extend(rows)
        teach_state = state_root / system / "knowledge_teach"
        teach = ensure_worker_output(
            system=system,
            mode="knowledge-teach",
            benchmark=paths["knowledge"],
            state_dir=teach_state,
            output=raw_dir / f"{system}_knowledge_teach.json",
            temperature=temperature,
            seed=seed,
            timeout=timeout,
        )
        base = state_root / system / "knowledge_base"
        if base.exists():
            shutil.rmtree(base)
        shutil.copytree(teach_state, base)
        recall = ensure_worker_output(
            system=system,
            mode="knowledge-recall",
            benchmark=paths["knowledge"],
            state_dir=state_root / system / "knowledge_recall_work",
            base_state_dir=base,
            output=raw_dir / f"{system}_knowledge_recall.json",
            temperature=temperature,
            seed=seed,
            timeout=timeout,
        )
        teach_by_id = {row["id"]: row for row in teach}
        for row in recall:
            row.update(teach_by_id.get(row["id"], {}))
        raw["knowledge"].extend(recall)

    scored_dir = experiment_dir / f"ablation_scored_{research_mode_tag()}"
    scored_dir.mkdir(exist_ok=True)
    context = score_relationship_rows(
        raw["context"],
        seed=seed + 3000,
        timeout=timeout,
    )
    write_json(scored_dir / "context.json", context)
    cross = score_relationship_rows(
        raw["cross"],
        seed=seed + 4000,
        timeout=timeout,
    )
    write_json(scored_dir / "cross.json", cross)
    memory = score_memory_rows(raw["memory"])
    knowledge = score_knowledge_rows(raw["knowledge"])
    scored = {
        "context": context,
        "cross": cross,
        "memory": memory,
        "knowledge": knowledge,
    }
    write_json(scored_dir / "memory.json", memory)
    write_json(scored_dir / "knowledge.json", knowledge)

    summary = []
    for system in ABLATIONS:
        latency = []
        for rows in scored.values():
            latency.extend(row_metric(rows, system, "latency_ms"))
        metrics = {
            "context_contamination_rate": row_metric(
                context,
                system,
                "contaminated",
            ),
            "false_rejection_rate": row_metric(
                context,
                system,
                "false_rejection",
            ) + row_metric(cross, system, "false_rejection"),
            "memory_recall": row_metric(memory, system, "exact_recall"),
            "knowledge_growth": row_metric(
                knowledge,
                system,
                "knowledge_growth_accuracy",
            ),
            "cross_domain_robustness": row_metric(
                cross,
                system,
                "cross_domain_robustness",
            ),
        }
        for metric, values in metrics.items():
            summary.append({
                "ablation": system,
                "metric": metric,
                **describe(values, seed=seed),
            })
        summary.append({
            "ablation": system,
            "metric": "latency_ms",
            **describe_latency(latency, seed=seed),
        })
    return scored, summary


def main():
    global SYSTEMS

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT_DIR))
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--suite", default="all")
    parser.add_argument("--systems", default="A,B,C")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--run-ablation", action="store_true")
    args = parser.parse_args()

    configure_uncached_research_runtime()
    SYSTEMS = parse_systems(args.systems)
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be a positive integer")
    selected_suite_names(args.suite)
    if args.fast:
        os.environ["FAST_RESEARCH_MODE"] = "1"
        os.environ.pop("FULL_RESEARCH_MODE", None)
    else:
        os.environ.setdefault("FAST_RESEARCH_MODE", "1")

    experiment_dir = Path(args.experiment_dir)
    if args.fresh and experiment_dir.exists():
        shutil.rmtree(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    write_benchmarks()
    metadata = experiment_metadata(
        args.seed,
        args.temperature,
        args.timeout,
        args.suite,
        args.limit,
    )
    metadata["smoke"] = args.smoke
    write_json(experiment_dir / "metadata.json", metadata)

    raw = stable_raw_runs(
        experiment_dir,
        args.seed,
        args.temperature,
        args.timeout,
        args.smoke,
        args.suite,
        args.limit,
    )
    scored = score_stable_runs(
        raw,
        experiment_dir,
        args.seed,
        args.timeout,
    )
    fresh_rows = flatten_fresh_results(scored)
    category_summary = build_category_summary(scored, args.seed)
    tests = statistical_tests(scored)
    confusion_rows = build_confusion_rows(scored["intent"])

    write_json(RESULT_DIR / "fresh_run_results.json", {
        "metadata": metadata,
        "rows": fresh_rows,
    })
    write_csv(RESULT_DIR / "fresh_run_results.csv", fresh_rows)
    write_csv(RESULT_DIR / "category_summary.csv", category_summary)
    write_csv(
        RESULT_DIR / "intent_confusion_matrix.csv",
        confusion_rows,
    )
    write_json(RESULT_DIR / "statistical_tests.json", tests)

    should_run_ablation = (
        not args.skip_ablation
        and (args.limit is None or args.run_ablation)
        and bool({"all", "context", "contamination", "cross", "robustness"} & {
            str(args.suite).strip().lower()
        })
    )
    if should_run_ablation:
        _, ablation_summary = run_ablations(
            experiment_dir,
            args.seed,
            args.temperature,
            args.timeout,
            args.smoke or args.limit is not None,
        )
        write_csv(
            RESULT_DIR / "ablation_summary.csv",
            ablation_summary,
        )
    elif not (RESULT_DIR / "ablation_summary.csv").exists():
        write_csv(RESULT_DIR / "ablation_summary.csv", [])

    print(f"Controlled experiment completed: {experiment_dir}")


if __name__ == "__main__":
    main()
