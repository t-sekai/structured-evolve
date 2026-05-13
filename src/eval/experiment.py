"""Shared experiment pipeline for all matmul scheduling strategies."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.benchmark import build_and_run_matmul, measure_latency_ms
from src.eval.correctness import check_against_reference
from src.eval.results_io import append_csv_result, save_json_result
from src.kernels.matmul_tir import KERNEL_NAME
from src.strategies.base import MatmulStrategy, StrategyBuildConfig


def run_matmul_experiment(
    *,
    strategy: MatmulStrategy,
    strategy_config: StrategyBuildConfig,
    M: int,
    N: int,
    K: int,
    target_name: str,
    num_warmup: int,
    num_trials: int,
    output_dir: Path,
    bad_baseline: bool = False,
) -> dict[str, Any]:
    """Run one strategy through the common compile, correctness, timing, and save path."""
    timestamp = datetime.now(timezone.utc).isoformat()
    strategy_config = _with_default_work_dir(
        strategy=strategy,
        config=strategy_config,
        output_dir=output_dir,
        M=M,
        N=N,
        K=K,
        target_name=target_name,
        timestamp=timestamp,
    )

    try:
        rng = np.random.default_rng(seed=0)
        a_np = rng.standard_normal((M, K), dtype=np.float32)
        b_np = rng.standard_normal((K, N), dtype=np.float32)
        reference_np = a_np @ b_np

        run_state = build_and_run_matmul(
            a_np=a_np,
            b_np=b_np,
            M=M,
            N=N,
            K=K,
            target_name=target_name,
            strategy=strategy,
            strategy_config=strategy_config,
        )

        output_for_check = run_state.output_np
        if bad_baseline:
            output_for_check = np.zeros_like(run_state.output_np)

        correctness = check_against_reference(output_for_check, reference_np)
        latency = measure_latency_ms(
            lib=run_state.lib,
            device=run_state.device,
            a_tvm=run_state.a_tvm,
            b_tvm=run_state.b_tvm,
            c_tvm=run_state.c_tvm,
            num_warmup=num_warmup,
            num_trials=num_trials,
        )

        result = _base_result(
            strategy=strategy,
            M=M,
            N=N,
            K=K,
            target_name=target_name,
            device=run_state.device_description,
            timestamp=run_state.timestamp,
            num_warmup=num_warmup,
            num_trials=num_trials,
            bad_baseline=bad_baseline,
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
            strategy=strategy,
            M=M,
            N=N,
            K=K,
            target_name=target_name,
            device="",
            timestamp=timestamp,
            num_warmup=num_warmup,
            num_trials=num_trials,
            bad_baseline=bad_baseline,
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

    json_path = save_json_result(result, output_dir)
    csv_path = append_csv_result(result, output_dir)
    result["json_result"] = str(json_path)
    result["csv_result"] = str(csv_path)
    return result


def _base_result(
    *,
    strategy: MatmulStrategy,
    M: int,
    N: int,
    K: int,
    target_name: str,
    device: str,
    timestamp: str,
    num_warmup: int,
    num_trials: int,
    bad_baseline: bool,
) -> dict[str, Any]:
    return {
        "kernel_name": KERNEL_NAME,
        "strategy": strategy.name,
        "level": strategy.level,
        "M": M,
        "N": N,
        "K": K,
        "target": target_name,
        "device": device,
        "num_warmup": num_warmup,
        "num_trials": num_trials,
        "timestamp": timestamp,
        "bad_baseline": bad_baseline,
    }


def _with_default_work_dir(
    *,
    strategy: MatmulStrategy,
    config: StrategyBuildConfig,
    output_dir: Path,
    M: int,
    N: int,
    K: int,
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
        / f"M{M}_N{N}_K{K}_{target_name}_{safe_timestamp}"
    )
    return replace(config, work_dir=work_dir)


def _truncate(value: str, max_len: int = 1000) -> str:
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."
