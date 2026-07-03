from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "evaluator" / "results" / "results.json"
SYSTEM_KEYS = ("A", "B", "C")


@dataclass(frozen=True)
class LoadResult:
    path: Path
    entries: list[dict[str, Any]]
    errors: list[str]
    warnings: list[str]
    loaded_at: str
    source_modified_at: str | None
    file_size_bytes: int | None


def _format_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def load_results(path: Path | str = DEFAULT_RESULTS_PATH) -> LoadResult:
    """Load the latest evaluator results from disk on every call."""
    result_path = Path(path)
    loaded_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    errors: list[str] = []
    warnings: list[str] = []
    source_modified_at: str | None = None
    file_size_bytes: int | None = None

    if not result_path.exists():
        return LoadResult(
            path=result_path,
            entries=[],
            errors=[f"Missing results file: {result_path}"],
            warnings=warnings,
            loaded_at=loaded_at,
            source_modified_at=source_modified_at,
            file_size_bytes=file_size_bytes,
        )

    try:
        stat = result_path.stat()
        source_modified_at = _format_timestamp(stat.st_mtime)
        file_size_bytes = stat.st_size
        raw_text = result_path.read_text(encoding="utf-8")
    except OSError as exc:
        return LoadResult(
            path=result_path,
            entries=[],
            errors=[f"Could not read {result_path}: {exc}"],
            warnings=warnings,
            loaded_at=loaded_at,
            source_modified_at=source_modified_at,
            file_size_bytes=file_size_bytes,
        )

    if not raw_text.strip():
        return LoadResult(
            path=result_path,
            entries=[],
            errors=[f"Empty JSON file: {result_path}"],
            warnings=warnings,
            loaded_at=loaded_at,
            source_modified_at=source_modified_at,
            file_size_bytes=file_size_bytes,
        )

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return LoadResult(
            path=result_path,
            entries=[],
            errors=[
                "Malformed JSON in "
                f"{result_path} at line {exc.lineno}, column {exc.colno}: "
                f"{exc.msg}"
            ],
            warnings=warnings,
            loaded_at=loaded_at,
            source_modified_at=source_modified_at,
            file_size_bytes=file_size_bytes,
        )

    if not isinstance(payload, list):
        return LoadResult(
            path=result_path,
            entries=[],
            errors=[
                "Unexpected results schema: expected a list of benchmark "
                f"entries in {result_path}"
            ],
            warnings=warnings,
            loaded_at=loaded_at,
            source_modified_at=source_modified_at,
            file_size_bytes=file_size_bytes,
        )

    entries: list[dict[str, Any]] = []
    skipped = 0
    for entry in payload:
        if isinstance(entry, dict):
            entries.append(entry)
        else:
            skipped += 1

    if skipped:
        warnings.append(f"Skipped {skipped} non-object benchmark entries.")

    return LoadResult(
        path=result_path,
        entries=entries,
        errors=errors,
        warnings=warnings,
        loaded_at=loaded_at,
        source_modified_at=source_modified_at,
        file_size_bytes=file_size_bytes,
    )


def validate_entries(entries: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    if not entries:
        errors.append("No benchmark entries found in results.json.")
        return errors

    system_counts = {system: 0 for system in SYSTEM_KEYS}
    missing_examples: list[str] = []

    for index, entry in enumerate(entries, start=1):
        results = entry.get("results")
        if not isinstance(results, dict):
            if len(missing_examples) < 5:
                missing_examples.append(f"entry {index}: missing results object")
            continue

        missing_systems: list[str] = []
        for system in SYSTEM_KEYS:
            if isinstance(results.get(system), dict):
                system_counts[system] += 1
            else:
                missing_systems.append(system)

        if missing_systems and len(missing_examples) < 5:
            missing_examples.append(
                f"entry {index}: missing System {', '.join(missing_systems)}"
            )

    for system, count in system_counts.items():
        if count == 0:
            errors.append(f"No System {system} results found in results.json.")

    if missing_examples:
        errors.append(
            "Some benchmark entries are missing required A, B, C results: "
            + "; ".join(missing_examples)
        )

    return errors
