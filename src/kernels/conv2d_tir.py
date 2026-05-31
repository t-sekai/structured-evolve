"""TensorIR definition for NCHW Conv2D workloads."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tvm
from tvm import te

try:
    from tvm import s_tir as _s_tir
except ImportError:
    _s_tir = None

KERNEL_NAME = "conv2d_nchw"


@dataclass(frozen=True)
class Conv2DShape:
    """Static NCHW Conv2D shape with square stride and padding."""

    batch: int
    in_channels: int
    height: int
    width: int
    out_channels: int
    kernel_h: int
    kernel_w: int
    stride: int = 1
    padding: int = 0

    @property
    def out_height(self) -> int:
        return (self.height + 2 * self.padding - self.kernel_h) // self.stride + 1

    @property
    def out_width(self) -> int:
        return (self.width + 2 * self.padding - self.kernel_w) // self.stride + 1

    @property
    def shape_id(self) -> str:
        return (
            f"B{self.batch}_CI{self.in_channels}_H{self.height}_W{self.width}_"
            f"CO{self.out_channels}_KH{self.kernel_h}_KW{self.kernel_w}_"
            f"S{self.stride}_P{self.padding}"
        )

    @property
    def problem_size(self) -> int:
        return (
            self.batch
            * self.out_channels
            * self.out_height
            * self.out_width
            * self.in_channels
            * self.kernel_h
            * self.kernel_w
        )


def create_conv2d_ir_module(shape: Conv2DShape) -> tvm.IRModule:
    """Create TensorIR for NCHW Conv2D forward inference."""
    _validate_shape(shape)

    data = te.placeholder(
        (shape.batch, shape.in_channels, shape.height, shape.width),
        name="data",
        dtype="float32",
    )
    weight = te.placeholder(
        (shape.out_channels, shape.in_channels, shape.kernel_h, shape.kernel_w),
        name="weight",
        dtype="float32",
    )

    if shape.padding > 0:
        padded_h = shape.height + 2 * shape.padding
        padded_w = shape.width + 2 * shape.padding
        data_in = te.compute(
            (shape.batch, shape.in_channels, padded_h, padded_w),
            lambda n, c, h, w: te.if_then_else(
                te.all(
                    h >= shape.padding,
                    h < shape.height + shape.padding,
                    w >= shape.padding,
                    w < shape.width + shape.padding,
                ),
                data[n, c, h - shape.padding, w - shape.padding],
                te.const(0.0, "float32"),
            ),
            name="data_pad",
        )
    else:
        data_in = data

    rc = te.reduce_axis((0, shape.in_channels), name="rc")
    ry = te.reduce_axis((0, shape.kernel_h), name="ry")
    rx = te.reduce_axis((0, shape.kernel_w), name="rx")
    conv = te.compute(
        (shape.batch, shape.out_channels, shape.out_height, shape.out_width),
        lambda n, co, y, x: te.sum(
            data_in[n, rc, y * shape.stride + ry, x * shape.stride + rx]
            * weight[co, rc, ry, rx],
            axis=[rc, ry, rx],
        ),
        name="conv",
    )
    prim_func = te.create_prim_func([data, weight, conv]).with_attr(
        "global_symbol", "main"
    )
    return tvm.IRModule.from_expr(prim_func)


def apply_schedule_for_target(ir_module: tvm.IRModule, target_name: str) -> tvm.IRModule:
    """Apply a conservative fixed Conv2D schedule."""
    if target_name == "llvm":
        return _apply_llvm_schedule(ir_module)
    if target_name == "cuda":
        return _apply_cuda_schedule(ir_module)
    raise ValueError(f"Unsupported target: {target_name}")


def _apply_llvm_schedule(ir_module: tvm.IRModule) -> tvm.IRModule:
    sch = _create_schedule(ir_module)
    try:
        block = _get_block(sch, "conv", func_name="main")
        loops = list(sch.get_loops(block))
        if len(loops) >= 4:
            sch.parallel(loops[1])
            sch.vectorize(loops[3])
        return sch.mod
    except Exception:
        return ir_module


def _apply_cuda_schedule(ir_module: tvm.IRModule) -> tvm.IRModule:
    """Keep the CUDA baseline as the canonical TE lowering."""
    return ir_module


def make_conv2d_inputs(shape: Conv2DShape, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    """Create deterministic Conv2D input tensors."""
    data = rng.standard_normal(
        (shape.batch, shape.in_channels, shape.height, shape.width),
        dtype=np.float32,
    )
    weight = rng.standard_normal(
        (shape.out_channels, shape.in_channels, shape.kernel_h, shape.kernel_w),
        dtype=np.float32,
    )
    return data, weight


def conv2d_reference(shape: Conv2DShape, inputs: tuple[np.ndarray, ...]) -> np.ndarray:
    """Compute a NumPy NCHW Conv2D reference."""
    data, weight = inputs
    if shape.padding > 0:
        data = np.pad(
            data,
            ((0, 0), (0, 0), (shape.padding, shape.padding), (shape.padding, shape.padding)),
            mode="constant",
        )

    output = np.zeros(
        (shape.batch, shape.out_channels, shape.out_height, shape.out_width),
        dtype=np.float32,
    )
    for n in range(shape.batch):
        for co in range(shape.out_channels):
            for y in range(shape.out_height):
                y0 = y * shape.stride
                for x in range(shape.out_width):
                    x0 = x * shape.stride
                    window = data[
                        n,
                        :,
                        y0 : y0 + shape.kernel_h,
                        x0 : x0 + shape.kernel_w,
                    ]
                    output[n, co, y, x] = np.sum(window * weight[co], dtype=np.float32)
    return output


def _create_schedule(ir_module: tvm.IRModule):
    if hasattr(tvm, "tir") and hasattr(tvm.tir, "Schedule"):
        return tvm.tir.Schedule(ir_module)
    if _s_tir is not None:
        return _s_tir.Schedule(ir_module)
    raise RuntimeError("This TVM build does not expose a TensorIR schedule API.")


def _get_block(sch, name: str, *, func_name: str):
    if hasattr(sch, "get_block"):
        return sch.get_block(name, func_name=func_name)
    return sch.get_sblock(name, func_name=func_name)


def _validate_shape(shape: Conv2DShape) -> None:
    for name, value in (
        ("batch", shape.batch),
        ("in_channels", shape.in_channels),
        ("height", shape.height),
        ("width", shape.width),
        ("out_channels", shape.out_channels),
        ("kernel_h", shape.kernel_h),
        ("kernel_w", shape.kernel_w),
        ("stride", shape.stride),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if shape.padding < 0:
        raise ValueError(f"padding must be non-negative, got {shape.padding}")
    if shape.out_height <= 0 or shape.out_width <= 0:
        raise ValueError(
            "Conv2D output dimensions must be positive, got "
            f"out_height={shape.out_height}, out_width={shape.out_width}"
        )
