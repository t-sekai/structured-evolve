"""TensorIR definition for a simple matrix multiplication workload."""

from __future__ import annotations

import tvm
from tvm import te

try:
    from tvm import s_tir as _s_tir
except ImportError:
    _s_tir = None

KERNEL_NAME = "matmul_tir"
CUDA_TILE = 16


def create_matmul_ir_module(M: int, N: int, K: int) -> tvm.IRModule:
    """Create TensorIR for C[M, N] = sum_k A[M, K] * B[K, N]."""
    _validate_shape(M, N, K)

    A = te.placeholder((M, K), name="A", dtype="float32")
    B = te.placeholder((K, N), name="B", dtype="float32")
    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
        name="C",
    )
    prim_func = te.create_prim_func([A, B, C]).with_attr("global_symbol", "main")
    return tvm.IRModule.from_expr(prim_func)


def apply_schedule_for_target(ir_module: tvm.IRModule, target_name: str) -> tvm.IRModule:
    """Apply a tiny target-specific schedule so the scaffold runs on CPU and CUDA."""
    if target_name == "llvm":
        return _apply_llvm_schedule(ir_module)
    if target_name == "cuda":
        return _apply_cuda_schedule(ir_module)
    raise ValueError(f"Unsupported target: {target_name}")


def _apply_llvm_schedule(ir_module: tvm.IRModule) -> tvm.IRModule:
    """Keep the CPU baseline as the canonical TE lowering."""
    return ir_module


def _apply_cuda_schedule(ir_module: tvm.IRModule) -> tvm.IRModule:
    """Map one CUDA thread to one output element inside a 16x16 block tile."""
    sch = _create_schedule(ir_module)
    block = _get_block(sch, "C", func_name="main")
    i, j, k = sch.get_loops(block)

    i_block, i_thread = sch.split(i, factors=[None, CUDA_TILE])
    j_block, j_thread = sch.split(j, factors=[None, CUDA_TILE])
    sch.reorder(i_block, j_block, i_thread, j_thread, k)

    sch.bind(i_block, "blockIdx.y")
    sch.bind(j_block, "blockIdx.x")
    sch.bind(i_thread, "threadIdx.y")
    sch.bind(j_thread, "threadIdx.x")
    return sch.mod


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


def _validate_shape(M: int, N: int, K: int) -> None:
    for name, value in (("M", M), ("N", N), ("K", K)):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
