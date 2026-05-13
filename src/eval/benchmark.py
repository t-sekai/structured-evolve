"""TVM compilation, execution, and timing helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import tvm

from src.kernels.matmul_tir import create_matmul_ir_module
from src.strategies.base import MatmulStrategy, StrategyBuildConfig
from src.strategies.fixed import FixedScheduleStrategy


@dataclass(frozen=True)
class RunState:
    lib: tvm.runtime.Module
    device: tvm.runtime.Device
    device_description: str
    strategy_name: str
    strategy_level: str
    scheduled_module: tvm.IRModule
    build_metadata: dict[str, Any]
    a_tvm: Any
    b_tvm: Any
    c_tvm: Any
    output_np: np.ndarray
    timestamp: str


@dataclass(frozen=True)
class LatencyStats:
    mean: float
    std: float
    samples_ms: list[float]


def build_and_run_matmul(
    *,
    a_np: np.ndarray,
    b_np: np.ndarray,
    M: int,
    N: int,
    K: int,
    target_name: str,
    strategy: MatmulStrategy | None = None,
    strategy_config: StrategyBuildConfig | None = None,
) -> RunState:
    """Build the TensorIR matmul for a target, run once, and return runtime state."""
    strategy = strategy or FixedScheduleStrategy()
    strategy_config = strategy_config or StrategyBuildConfig()

    device = _get_device(target_name)
    target = _make_target(target_name)

    ir_module = create_matmul_ir_module(M, N, K)
    build_result = strategy.build(
        ir_module=ir_module,
        target=target,
        target_name=target_name,
        config=strategy_config,
    )

    a_tvm = _array(a_np, device)
    b_tvm = _array(b_np, device)
    c_tvm = _empty((M, N), dtype="float32", device=device)

    build_result.lib["main"](a_tvm, b_tvm, c_tvm)
    device.sync()
    output_np = c_tvm.numpy()

    return RunState(
        lib=build_result.lib,
        device=device,
        device_description=_device_description(target_name, device),
        strategy_name=strategy.name,
        strategy_level=strategy.level,
        scheduled_module=build_result.scheduled_module,
        build_metadata=build_result.metadata,
        a_tvm=a_tvm,
        b_tvm=b_tvm,
        c_tvm=c_tvm,
        output_np=output_np,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def measure_latency_ms(
    *,
    lib: tvm.runtime.Module,
    device: tvm.runtime.Device,
    a_tvm: Any,
    b_tvm: Any,
    c_tvm: Any,
    num_warmup: int,
    num_trials: int,
) -> LatencyStats:
    """Measure runtime in milliseconds using TVM's built-in time evaluator."""
    for _ in range(num_warmup):
        lib["main"](a_tvm, b_tvm, c_tvm)
    device.sync()

    evaluator = lib.time_evaluator("main", device, number=1, repeat=num_trials)
    timing_result = evaluator(a_tvm, b_tvm, c_tvm)
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
