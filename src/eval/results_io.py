"""Helpers for writing benchmark results."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


CSV_FIELDS = [
    "kernel_name",
    "strategy",
    "level",
    "M",
    "N",
    "K",
    "target",
    "device",
    "compile_passed",
    "correctness_passed",
    "max_abs_error",
    "mean_abs_error",
    "latency_ms_mean",
    "latency_ms_std",
    "num_warmup",
    "num_trials",
    "tuning_time_sec",
    "max_trials_global",
    "max_trials_per_task",
    "num_trials_per_iter",
    "cost_model",
    "task_scheduler",
    "seed",
    "metaschedule_work_dir",
    "scheduled_module_path",
    "error_type",
    "error_message",
    "timestamp",
    "bad_baseline",
]


def save_json_result(result: dict[str, Any], output_dir: Path) -> Path:
    """Write one JSON file for this run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = _json_filename(result)
    path = output_dir / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def append_csv_result(result: dict[str, Any], output_dir: Path) -> Path:
    """Append this run to results.csv, creating the file and header if needed."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "results.csv"
    fieldnames = _fieldnames_for_result(path, result)

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow({field: _csv_value(result.get(field)) for field in fieldnames})

    return path


def _json_filename(result: dict[str, Any]) -> str:
    timestamp = str(result["timestamp"]).replace(":", "").replace("+", "Z")
    target = _safe_fragment(str(result["target"]))
    kernel = _safe_fragment(str(result["kernel_name"]))
    strategy = _safe_fragment(str(result.get("strategy", "baseline")))
    return (
        f"{kernel}_{strategy}_M{result['M']}_N{result['N']}_K{result['K']}_"
        f"{target}_{timestamp}.json"
    )


def _safe_fragment(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def _fieldnames_for_result(path: Path, result: dict[str, Any]) -> list[str]:
    result_fields = [
        field
        for field in CSV_FIELDS
        if field in result or field in ("strategy", "level", "compile_passed")
    ]
    result_fields.extend(
        sorted(field for field in result if field not in result_fields and _is_scalar(result[field]))
    )

    if not path.exists() or path.stat().st_size == 0:
        return result_fields

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            existing_fields = next(reader)
        except StopIteration:
            return result_fields

    missing_fields = [field for field in result_fields if field not in existing_fields]
    if not missing_fields:
        return existing_fields

    expanded_fields = existing_fields + missing_fields
    _rewrite_csv_with_fields(path, expanded_fields)
    return expanded_fields


def _rewrite_csv_with_fields(path: Path, fieldnames: list[str]) -> None:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _csv_value(value: Any) -> Any:
    if _is_scalar(value):
        return value
    return json.dumps(value, sort_keys=True)
