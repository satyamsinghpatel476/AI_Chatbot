import argparse
import json
import os
import signal
import shutil
import time

from evaluator.research.ablation import SystemCAblation
from evaluator.research.state import (
    configure_stable_system,
    copy_state,
    inject_filler_history,
    initialize_state_dir,
    reload_loaded_system,
    reset_loaded_system,
    seed_context,
)
from llm_runtime import prepare_uncached_benchmark_runtime


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


class SystemCallTimeout(TimeoutError):
    pass


def configure_research_worker_runtime():
    prepare_uncached_benchmark_runtime()
    os.environ["BENCHMARK_FORCE_LLM"] = "1"
    os.environ["BENCHMARK_DISABLE_DETERMINISTIC_SHORTCUTS"] = "1"


def _timeout_handler(signum, frame):
    raise SystemCallTimeout("system call timed out")


def call_with_timeout(func, timeout):
    previous = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, max(1, float(timeout)))
    try:
        return func()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def build_system(name, state_dir, temperature, seed, timeout):
    if name in {"A", "B", "C"}:
        module, function = configure_stable_system(
            name,
            state_dir,
            temperature,
            seed,
            timeout,
        )
        return {
            "name": name,
            "module": module,
            "call": function,
            "reset": lambda: reset_loaded_system(name, state_dir),
            "reload": lambda: reload_loaded_system(name, state_dir),
            "timeout": timeout,
        }

    if name.startswith("C") and name[1:].isdigit():
        level = int(name[1:])
        system = SystemCAblation(
            level,
            state_dir,
            temperature=temperature,
            seed=seed,
            timeout=timeout,
        )
        return {
            "name": name,
            "module": system,
            "call": system,
            "reset": system.reset,
            "reload": system.reload,
            "timeout": timeout,
        }
    raise ValueError(f"Unknown system: {name}")


def invoke(system, query):
    started = time.perf_counter()
    try:
        result = call_with_timeout(
            lambda: system["call"](query, return_metadata=True),
            system.get("timeout", 180),
        )
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return {
            "response": f"System runtime error: {exc}",
            "latency_ms": elapsed * 1000,
            "total_latency": elapsed,
            "total_latency_ms": elapsed * 1000,
            "llm_called": False,
            "cache_used": False,
            "deterministic_path_used": False,
            "metadata": {
                "runtime_error": str(exc),
                "cache_used": False,
                "deterministic_path_used": False,
                "llm_called": False,
            },
        }
    elapsed = time.perf_counter() - started
    metadata = {
        key: value
        for key, value in result.items()
        if key not in {"response", "latency"}
    }
    if "latency" in result:
        metadata["function_reported_latency"] = result.get("latency")
    metadata.setdefault("cache_used", False)
    metadata.setdefault("deterministic_path_used", False)
    metadata.setdefault("llm_called", False)
    if os.environ.get("DISABLE_LLM_CACHE") == "1":
        metadata["cache_used"] = False
    llm_called = bool(metadata.get("llm_called"))
    cache_used = bool(metadata.get("cache_used"))
    deterministic_path_used = bool(metadata.get("deterministic_path_used"))
    latency_warning = None
    if llm_called and elapsed < 0.05:
        latency_warning = (
            f"WARNING: System {system['name']} reported llm_called=true "
            f"but total latency was {elapsed:.4f}s."
        )
        print(latency_warning)
        metadata["latency_warning"] = latency_warning
    return {
        "response": result.get("response", ""),
        "latency_ms": elapsed * 1000,
        "total_latency": elapsed,
        "total_latency_ms": elapsed * 1000,
        "llm_called": llm_called,
        "cache_used": cache_used,
        "deterministic_path_used": deterministic_path_used,
        "metadata": metadata,
    }


def run_context(system, cases, state_dir, output_path=None):
    rows = []
    for case in cases:
        system["reset"]()
        if system["name"] in {"A", "B"}:
            seed_context(
                system["name"],
                state_dir,
                case.get("context", []),
                system["module"],
            )
        output = invoke(system, case["query"])
        rows.append({**case, "system": system["name"], **output})
        if output_path:
            save_json(output_path, rows)
    return rows


def run_cross_domain(system, cases, output_path=None):
    rows = []
    for case in cases:
        system["reset"]()
        output = invoke(system, case["query"])
        rows.append({**case, "system": system["name"], **output})
        if output_path:
            save_json(output_path, rows)
    return rows


def run_memory(system, cases, state_dir, output_path=None):
    rows = []
    for case in cases:
        system["reset"]()
        save_output = invoke(system, case["save"])
        if system["name"] in {"A", "B"}:
            inject_filler_history(
                system["name"],
                state_dir,
                case.get("filler_turns", 30),
            )
        recall_output = invoke(system, case["recall"])
        rows.append({
            **case,
            "system": system["name"],
            "save_response": save_output["response"],
            "save_latency_ms": save_output["latency_ms"],
            "save_total_latency": save_output["total_latency"],
            "save_llm_called": save_output["llm_called"],
            "save_cache_used": save_output["cache_used"],
            "save_deterministic_path_used": save_output["deterministic_path_used"],
            "save_metadata": save_output["metadata"],
            **recall_output,
        })
        if output_path:
            save_json(output_path, rows)
    return rows


def run_knowledge_teach(system, cases, state_dir, output_path=None):
    system["reset"]()
    rows = []
    for case in cases:
        output = invoke(system, case["teach"])
        rows.append({
            "id": case["id"],
            "system": system["name"],
            "teach_response": output["response"],
            "teach_latency_ms": output["latency_ms"],
            "teach_total_latency": output["total_latency"],
            "teach_llm_called": output["llm_called"],
            "teach_cache_used": output["cache_used"],
            "teach_deterministic_path_used": output["deterministic_path_used"],
            "teach_metadata": output["metadata"],
        })
        if output_path:
            save_json(output_path, rows)
    if system["name"] in {"A", "B"}:
        inject_filler_history(system["name"], state_dir, 30)
    return rows


def run_knowledge_recall(system, cases, state_dir, base_state_dir, output_path=None):
    rows = []
    for case in cases:
        case_outputs = {}
        aggregate_total_latency = 0.0
        aggregate_llm_called = False
        aggregate_cache_used = False
        aggregate_deterministic_path_used = False
        for query_kind in [
            "exact_recall",
            "paraphrased_recall",
            "unrelated_query",
        ]:
            copy_state(base_state_dir, state_dir)
            system["reload"]()
            output = invoke(system, case[query_kind])
            case_outputs[f"{query_kind}_response"] = output["response"]
            case_outputs[f"{query_kind}_latency_ms"] = output["latency_ms"]
            case_outputs[f"{query_kind}_total_latency"] = output["total_latency"]
            case_outputs[f"{query_kind}_llm_called"] = output["llm_called"]
            case_outputs[f"{query_kind}_cache_used"] = output["cache_used"]
            case_outputs[
                f"{query_kind}_deterministic_path_used"
            ] = output["deterministic_path_used"]
            case_outputs[f"{query_kind}_metadata"] = output["metadata"]
            aggregate_total_latency += output["total_latency"]
            aggregate_llm_called = (
                aggregate_llm_called or output["llm_called"]
            )
            aggregate_cache_used = (
                aggregate_cache_used or output["cache_used"]
            )
            aggregate_deterministic_path_used = (
                aggregate_deterministic_path_used
                or output["deterministic_path_used"]
            )
        rows.append({
            **case,
            "system": system["name"],
            "total_latency": aggregate_total_latency,
            "total_latency_ms": aggregate_total_latency * 1000,
            "llm_called": aggregate_llm_called,
            "cache_used": aggregate_cache_used,
            "deterministic_path_used": aggregate_deterministic_path_used,
            **case_outputs,
        })
        if output_path:
            save_json(output_path, rows)
    return rows


def run_intent(system, cases, output_path=None):
    rows = []
    classifier = getattr(system["module"], "CLASSIFIER", None)
    if classifier is None and hasattr(system["module"], "classifier"):
        classifier = system["module"].classifier
    for case in cases:
        if classifier is None:
            predicted = None
            confidence = None
            latency_ms = None
        else:
            started = time.perf_counter()
            try:
                predicted, confidence = call_with_timeout(
                    lambda: classifier.predict(case["query"]),
                    system.get("timeout", 180),
                )
            except Exception:
                predicted = None
                confidence = None
            latency_ms = (time.perf_counter() - started) * 1000
        rows.append({
            **case,
            "system": system["name"],
            "predicted_intent": predicted,
            "intent_confidence": confidence,
            "latency_ms": latency_ms,
            "total_latency": (
                latency_ms / 1000.0 if latency_ms is not None else None
            ),
            "total_latency_ms": latency_ms,
            "llm_called": False,
            "cache_used": False,
            "deterministic_path_used": False,
        })
        if output_path:
            save_json(output_path, rows)
    return rows


def main():
    configure_research_worker_runtime()
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "context",
            "cross",
            "memory",
            "knowledge-teach",
            "knowledge-recall",
            "intent",
        ],
    )
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--base-state-dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    initialize_state_dir(args.state_dir)
    system = build_system(
        args.system,
        args.state_dir,
        args.temperature,
        args.seed,
        args.timeout,
    )
    cases = load_json(args.benchmark)

    if args.mode == "context":
        rows = run_context(system, cases, args.state_dir, args.output)
    elif args.mode == "cross":
        rows = run_cross_domain(system, cases, args.output)
    elif args.mode == "memory":
        rows = run_memory(system, cases, args.state_dir, args.output)
    elif args.mode == "knowledge-teach":
        rows = run_knowledge_teach(system, cases, args.state_dir, args.output)
    elif args.mode == "knowledge-recall":
        if not args.base_state_dir:
            raise SystemExit("--base-state-dir is required for knowledge recall")
        rows = run_knowledge_recall(
            system,
            cases,
            args.state_dir,
            args.base_state_dir,
            args.output,
        )
    else:
        rows = run_intent(system, cases, args.output)

    save_json(args.output, rows)
    print(f"Wrote {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
