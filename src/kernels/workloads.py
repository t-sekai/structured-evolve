"""Workload descriptors for schedulable TensorIR kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
import tvm

from src.kernels.conv2d_tir import (
    KERNEL_NAME as CONV2D_KERNEL_NAME,
    Conv2DShape,
    apply_schedule_for_target as apply_conv2d_schedule_for_target,
    conv2d_reference,
    create_conv2d_ir_module,
    make_conv2d_inputs,
)
from src.kernels.matmul_tir import (
    KERNEL_NAME as MATMUL_KERNEL_NAME,
    apply_schedule_for_target as apply_matmul_schedule_for_target,
    create_matmul_ir_module,
)

MATMUL_WORKLOAD = "matmul"
CONV2D_WORKLOAD = "conv2d"
WORKLOAD_NAMES = (MATMUL_WORKLOAD, CONV2D_WORKLOAD)


@dataclass(frozen=True)
class Workload:
    """One concrete TensorIR workload instance."""

    name: str
    kernel_name: str
    shape_id: str
    problem_size: int
    output_shape: tuple[int, ...]
    output_dtype: str
    metadata: Mapping[str, object]
    create_ir_module: Callable[[], tvm.IRModule]
    make_inputs: Callable[[np.random.Generator], tuple[np.ndarray, ...]]
    reference: Callable[[tuple[np.ndarray, ...]], np.ndarray]
    # The fixed baseline belongs to the workload because block names and loop
    # structure are kernel-specific.
    fixed_schedule: Callable[[tvm.IRModule, str], tvm.IRModule]
    prompt_context: str
    primary_block_name: str


def make_matmul_workload(M: int, N: int, K: int) -> Workload:
    """Return the existing matmul workload as a generic descriptor."""
    _validate_positive({"M": M, "N": N, "K": K})

    def make_inputs(rng: np.random.Generator) -> tuple[np.ndarray, ...]:
        a_np = rng.standard_normal((M, K), dtype=np.float32)
        b_np = rng.standard_normal((K, N), dtype=np.float32)
        return a_np, b_np

    def reference(inputs: tuple[np.ndarray, ...]) -> np.ndarray:
        a_np, b_np = inputs
        return a_np @ b_np

    shape_id = f"M{M}_N{N}_K{K}"
    return Workload(
        name=MATMUL_WORKLOAD,
        kernel_name=MATMUL_KERNEL_NAME,
        shape_id=shape_id,
        problem_size=M * N * K,
        output_shape=(M, N),
        output_dtype="float32",
        metadata={
            "M": M,
            "N": N,
            "K": K,
        },
        create_ir_module=lambda: create_matmul_ir_module(M, N, K),
        make_inputs=make_inputs,
        reference=reference,
        fixed_schedule=apply_matmul_schedule_for_target,
        prompt_context=(
            f"matmul shape M={M}, N={N}, K={K}. Preserve numerical correctness "
            "for C = A @ B."
        ),
        primary_block_name="C",
    )


def make_conv2d_workload(
    *,
    batch: int,
    in_channels: int,
    height: int,
    width: int,
    out_channels: int,
    kernel_h: int,
    kernel_w: int,
    stride: int = 1,
    padding: int = 0,
) -> Workload:
    """Return an NCHW Conv2D workload descriptor."""
    shape = Conv2DShape(
        batch=batch,
        in_channels=in_channels,
        height=height,
        width=width,
        out_channels=out_channels,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        stride=stride,
        padding=padding,
    )
    metadata = {
        "batch": shape.batch,
        "in_channels": shape.in_channels,
        "height": shape.height,
        "width": shape.width,
        "out_channels": shape.out_channels,
        "kernel_h": shape.kernel_h,
        "kernel_w": shape.kernel_w,
        "stride": shape.stride,
        "padding": shape.padding,
        "out_height": shape.out_height,
        "out_width": shape.out_width,
    }
    return Workload(
        name=CONV2D_WORKLOAD,
        kernel_name=CONV2D_KERNEL_NAME,
        shape_id=shape.shape_id,
        problem_size=shape.problem_size,
        output_shape=(
            shape.batch,
            shape.out_channels,
            shape.out_height,
            shape.out_width,
        ),
        output_dtype="float32",
        metadata=metadata,
        create_ir_module=lambda: create_conv2d_ir_module(shape),
        make_inputs=lambda rng: make_conv2d_inputs(shape, rng),
        reference=lambda inputs: conv2d_reference(shape, inputs),
        fixed_schedule=apply_conv2d_schedule_for_target,
        prompt_context=(
            "NCHW Conv2D forward workload with "
            f"batch={shape.batch}, in_channels={shape.in_channels}, "
            f"height={shape.height}, width={shape.width}, "
            f"out_channels={shape.out_channels}, kernel={shape.kernel_h}x{shape.kernel_w}, "
            f"stride={shape.stride}, padding={shape.padding}. Preserve numerical "
            "correctness for the conv output tensor."
        ),
        primary_block_name="conv",
    )


def make_workload(
    *,
    workload_name: str,
    M: int,
    N: int,
    K: int,
    workload_params: Mapping[str, object] | None = None,
) -> Workload:
    """Create a workload from CLI/config fields."""
    if workload_name == MATMUL_WORKLOAD:
        return make_matmul_workload(M, N, K)
    if workload_name == CONV2D_WORKLOAD:
        params = dict(workload_params or {})
        return make_conv2d_workload(
            batch=int(params.get("batch", 1)),
            in_channels=int(params.get("in_channels", 3)),
            height=int(params.get("height", 16)),
            width=int(params.get("width", 16)),
            out_channels=int(params.get("out_channels", 8)),
            kernel_h=int(params.get("kernel_h", 3)),
            kernel_w=int(params.get("kernel_w", 3)),
            stride=int(params.get("stride", 1)),
            padding=int(params.get("padding", 1)),
        )
    choices = ", ".join(WORKLOAD_NAMES)
    raise ValueError(f"Unknown workload '{workload_name}'. Available workloads: {choices}")


def _validate_positive(values: Mapping[str, int]) -> None:
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
