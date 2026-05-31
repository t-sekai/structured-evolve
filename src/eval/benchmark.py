"""TVM compilation, execution, and timing helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import tvm

from src.kernels.workloads import Workload
from src.strategies.base import SchedulingStrategy, StrategyBuildConfig


@dataclass(frozen=True)
class RunState:
    lib: tvm.runtime.Module
    device: tvm.runtime.Device
    device_description: str
    strategy_name: str
    strategy_level: str
    scheduled_module: tvm.IRModule
    build_metadata: dict[str, Any]
    run_args: tuple[Any, ...]
    input_tensors: tuple[Any, ...]
    output_tensor: Any
    output_np: np.ndarray
    timestamp: str


@dataclass(frozen=True)
class LatencyStats:
    mean: float
    std: float
    samples_ms: list[float]


def build_and_run_workload(
    *,
    workload: Workload,
    input_arrays: tuple[np.ndarray, ...],
    target_name: str,
    strategy: SchedulingStrategy,
    strategy_config: StrategyBuildConfig,
) -> RunState:
    """Build a TensorIR workload for a target, run once, and return runtime state."""
    device = _get_device(target_name)
    target = _make_target(target_name)

    ir_module = workload.create_ir_module()
    build_result = strategy.build(
        workload=workload,
        ir_module=ir_module,
        target=target,
        target_name=target_name,
        config=strategy_config,
    )

    input_tensors = tuple(_array(array, device) for array in input_arrays)
    output_tvm = _empty(
        workload.output_shape,
        dtype=workload.output_dtype,
        device=device,
    )
    run_args = (*input_tensors, output_tvm)

    build_result.lib["main"](*run_args)
    device.sync()
    output_np = output_tvm.numpy()

    return RunState(
        lib=build_result.lib,
        device=device,
        device_description=_device_description(target_name, device),
        strategy_name=strategy.name,
        strategy_level=strategy.level,
        scheduled_module=build_result.scheduled_module,
        build_metadata=build_result.metadata,
        run_args=run_args,
        input_tensors=input_tensors,
        output_tensor=output_tvm,
        output_np=output_np,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def measure_latency_ms_for_args(
    *,
    lib: tvm.runtime.Module,
    device: tvm.runtime.Device,
    run_args: tuple[Any, ...],
    num_warmup: int,
    num_trials: int,
    benchmark_invocations: int = 1,
    min_repeat_ms: int | None = None,
) -> LatencyStats:
    """Measure runtime in milliseconds for an arbitrary TVM entrypoint argument list."""
    for _ in range(num_warmup):
        lib["main"](*run_args)
    device.sync()

    evaluator_kwargs = {
        "number": benchmark_invocations,
        "repeat": num_trials,
    }
    if min_repeat_ms is not None:
        evaluator_kwargs["min_repeat_ms"] = min_repeat_ms
    evaluator = lib.time_evaluator("main", device, **evaluator_kwargs)
    timing_result = evaluator(*run_args)
    samples_ms = [float(sample * 1_000.0) for sample in timing_result.results]
    return LatencyStats(
        mean=float(np.mean(samples_ms)),
        std=float(np.std(samples_ms)),
        samples_ms=samples_ms,
    )


def _get_device(target_name: str) -> tvm.runtime.Device:
    if target_name == "llvm":
        return tvm.cpu(0)
    if target_name == "cuda":
        device = tvm.cuda(0)
        if not device.exist:
            raise RuntimeError(
                "CUDA target requested, but tvm.cuda(0) is not available. "
                "Check that TVM was built with CUDA and that a CUDA device is visible."
            )
        return device
    raise ValueError(f"Unsupported target: {target_name}")


def _array(array: np.ndarray, device: tvm.runtime.Device) -> Any:
    if hasattr(tvm, "nd"):
        return tvm.nd.array(array, device)
    return tvm.runtime.tensor(array, device=device)


def _empty(shape: tuple[int, ...], *, dtype: str, device: tvm.runtime.Device) -> Any:
    if hasattr(tvm, "nd"):
        return tvm.nd.empty(shape, dtype=dtype, device=device)
    return tvm.runtime.empty(shape, dtype=dtype, device=device)


def _make_target(target_name: str) -> tvm.target.Target:
    host = _make_host_target()
    if target_name == "cuda":
        device_target = tvm.target.Target.from_device(tvm.cuda(0))
        return tvm.target.Target(device_target, host=host)
    if target_name == "llvm":
        return host
    raise ValueError(f"Unsupported target: {target_name}")


def _make_host_target() -> tvm.target.Target:
    return tvm.target.Target(
        {"kind": "llvm", "mcpu": "generic", "num-cores": os.cpu_count() or 1}
    )


def _device_description(target_name: str, device: tvm.runtime.Device) -> str:
    del target_name
    return str(device)
