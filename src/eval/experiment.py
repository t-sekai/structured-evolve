"""Shared experiment pipeline for workload scheduling strategies."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from src.eval.benchmark import build_and_run_workload, measure_latency_ms_for_args
from src.eval.correctness import check_against_reference
from src.eval.results_io import append_csv_result, save_json_result
from src.kernels.workloads import Workload
from src.strategies.base import SchedulingStrategy, StrategyBuildConfig


def run_workload_experiment(
    *,
    workload: Workload,
    strategy: SchedulingStrategy,
    strategy_config: StrategyBuildConfig,
    target_name: str,
    num_warmup: int,
    num_trials: int,
    output_dir: Path,
    benchmark_invocations: int = 1,
    min_repeat_ms: int | None = None,
    bad_baseline: bool = False,
    extra_metadata: Mapping[str, Any] | None = None,
    postprocess_result: Callable[[dict[str, Any]], Mapping[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """Run one workload through compile, correctness, timing, and persistence."""
    timestamp = datetime.now(timezone.utc).isoformat()
    strategy_config = _with_default_work_dir(
        strategy=strategy,
        config=strategy_config,
        output_dir=output_dir,
        shape_id=workload.shape_id,
        target_name=target_name,
        workload_name=workload.name,
        timestamp=timestamp,
    )
    strategy_config = replace(strategy_config, workload_name=workload.name)

    try:
        rng = np.random.default_rng(seed=0)
        inputs_np = workload.make_inputs(rng)
        reference_np = workload.reference(inputs_np)

        run_state = build_and_run_workload(
            workload=workload,
            input_arrays=inputs_np,
            target_name=target_name,
            strategy=strategy,
            strategy_config=strategy_config,
        )

        output_for_check = run_state.output_np
        if bad_baseline:
            output_for_check = np.zeros_like(run_state.output_np)

        correctness = check_against_reference(output_for_check, reference_np)
        latency = measure_latency_ms_for_args(
            lib=run_state.lib,
            device=run_state.device,
            run_args=run_state.run_args,
            num_warmup=num_warmup,
            num_trials=num_trials,
            benchmark_invocations=benchmark_invocations,
            min_repeat_ms=min_repeat_ms,
        )

        result = _base_result(
            workload=workload,
            strategy=strategy,
            target_name=target_name,
            device=run_state.device_description,
            timestamp=run_state.timestamp,
            num_warmup=num_warmup,
            num_trials=num_trials,
            benchmark_invocations=benchmark_invocations,
            min_repeat_ms=min_repeat_ms,
            bad_baseline=bad_baseline,
            extra_metadata=extra_metadata,
        )
        result.update(
            {
                "compile_passed": True,
                "correctness_passed": correctness.passed,
                "max_abs_error": correctness.max_abs_error,
                "mean_abs_error": correctness.mean_abs_error,
                "latency_ms_mean": latency.mean,
                "latency_ms_std": latency.std,
                "error_type": "",
                "error_message": "",
            }
        )
        result.update(run_state.build_metadata)

    except Exception as err:  # pylint: disable=broad-except
        result = _base_result(
            workload=workload,
            strategy=strategy,
            target_name=target_name,
            device="",
            timestamp=timestamp,
            num_warmup=num_warmup,
            num_trials=num_trials,
            benchmark_invocations=benchmark_invocations,
            min_repeat_ms=min_repeat_ms,
            bad_baseline=bad_baseline,
            extra_metadata=extra_metadata,
        )
        result.update(
            {
                "compile_passed": False,
                "correctness_passed": False,
                "max_abs_error": None,
                "mean_abs_error": None,
                "latency_ms_mean": None,
                "latency_ms_std": None,
                "error_type": type(err).__name__,
                "error_message": _truncate(str(err)),
            }
        )

    if postprocess_result is not None:
        postprocessed = postprocess_result(result)
        if postprocessed:
            result.update(_jsonable_metadata(postprocessed))

    return _persist_result(result, output_dir)


def record_rejected_workload_experiment(
    *,
    workload: Workload,
    strategy: SchedulingStrategy,
    target_name: str,
    num_warmup: int,
    num_trials: int,
    benchmark_invocations: int,
    min_repeat_ms: int | None,
    output_dir: Path,
    rejection_stage: str,
    rejection_reason: str,
    extra_metadata: Mapping[str, Any] | None = None,
    postprocess_result: Callable[[dict[str, Any]], Mapping[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """Persist a candidate rejected before its full benchmark and tuning path."""
    result = _base_result(
        workload=workload,
        strategy=strategy,
        target_name=target_name,
        device="",
        timestamp=datetime.now(timezone.utc).isoformat(),
        num_warmup=num_warmup,
        num_trials=num_trials,
        benchmark_invocations=benchmark_invocations,
        min_repeat_ms=min_repeat_ms,
        bad_baseline=False,
        extra_metadata=extra_metadata,
    )
    result.update(
        {
            "compile_passed": False,
            "correctness_passed": False,
            "max_abs_error": None,
            "mean_abs_error": None,
            "latency_ms_mean": None,
            "latency_ms_std": None,
            "cascade_rejected": True,
            "rejection_stage": rejection_stage,
            "rejection_reason": _truncate(rejection_reason),
            "tuning_skipped": True,
            "error_type": "CascadeRejected",
            "error_message": _truncate(f"{rejection_stage}: {rejection_reason}"),
        }
    )
    if postprocess_result is not None:
        postprocessed = postprocess_result(result)
        if postprocessed:
            result.update(_jsonable_metadata(postprocessed))
    return _persist_result(result, output_dir)


def _base_result(
    *,
    workload: Workload,
    strategy: SchedulingStrategy,
    target_name: str,
    device: str,
    timestamp: str,
    num_warmup: int,
    num_trials: int,
    benchmark_invocations: int,
    min_repeat_ms: int | None,
    bad_baseline: bool,
    extra_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result = {
        "task_name": workload.name,
        "kernel_name": workload.kernel_name,
        "workload_name": workload.name,
        "strategy": strategy.name,
        "level": strategy.level,
        "M": None,
        "N": None,
        "K": None,
        "shape": workload.shape_id,
        "problem_size": workload.problem_size,
        "target": target_name,
        "device": device,
        "num_warmup": num_warmup,
        "num_trials": num_trials,
        "benchmark_invocations": benchmark_invocations,
        "min_repeat_ms": min_repeat_ms,
        "timestamp": timestamp,
        "bad_baseline": bad_baseline,
    }
    result.update(_jsonable_metadata(workload.metadata))
    if extra_metadata:
        result.update(_jsonable_metadata(extra_metadata))
    return result


def _persist_result(result: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    json_path = save_json_result(result, output_dir)
    csv_path = append_csv_result(result, output_dir)
    result["json_result"] = str(json_path)
    result["csv_result"] = str(csv_path)
    return result


def _with_default_work_dir(
    *,
    strategy: SchedulingStrategy,
    config: StrategyBuildConfig,
    output_dir: Path,
    workload_name: str,
    shape_id: str,
    target_name: str,
    timestamp: str,
) -> StrategyBuildConfig:
    if config.work_dir is not None:
        return config

    safe_timestamp = timestamp.replace(":", "").replace("+", "Z")
    work_dir = (
        output_dir
        / "work_dirs"
        / strategy.name
        / workload_name
        / f"{shape_id}_{target_name}_{safe_timestamp}"
    )
    return replace(config, work_dir=work_dir)


def _truncate(value: str, max_len: int = 1000) -> str:
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def _jsonable_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize common path-like metadata before result persistence."""
    return {key: _jsonable_value(value) for key, value in metadata.items()}


def _jsonable_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_jsonable_value(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable_value(item) for key, item in value.items()}
    return value
