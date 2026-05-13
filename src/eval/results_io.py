"""Helpers for writing benchmark results."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


CSV_FIELDS = [
    "kernel_name",
    "M",
    "N",
    "K",
    "target",
    "device",
    "correctness_passed",
    "max_abs_error",
    "mean_abs_error",
    "latency_ms_mean",
    "latency_ms_std",
    "num_warmup",
    "num_trials",
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
    write_header = not path.exists()

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: result.get(field) for field in CSV_FIELDS})

    return path


def _json_filename(result: dict[str, Any]) -> str:
    timestamp = str(result["timestamp"]).replace(":", "").replace("+", "Z")
    target = _safe_fragment(str(result["target"]))
    kernel = _safe_fragment(str(result["kernel_name"]))
    return (
        f"{kernel}_M{result['M']}_N{result['N']}_K{result['K']}_"
        f"{target}_{timestamp}.json"
    )


def _safe_fragment(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)
