"""TensorIR definition for a simple matrix multiplication workload."""

from __future__ import annotations

import tvm
from tvm.script import ir as I

try:
    from tvm.script import tir as T
except ImportError:
    from tvm.script import tirx as T

try:
    from tvm import s_tir as _s_tir
except ImportError:
    _s_tir = None

KERNEL_NAME = "matmul_tir"
CUDA_TILE = 16


def create_matmul_ir_module(M: int, N: int, K: int) -> tvm.IRModule:
    """Create TensorIR for C[M, N] = sum_k A[M, K] * B[K, N]."""
    _validate_shape(M, N, K)

    if hasattr(T, "sblock"):

        @I.ir_module
        class MatmulModule:
            @T.prim_func
            def main(
                A: T.Buffer((M, K), "float32"),
                B: T.Buffer((K, N), "float32"),
                C: T.Buffer((M, N), "float32"),
            ) -> None:
                T.func_attr(
                    {
                        "global_symbol": "main",
                        "tir.noalias": True,
                        "tirx.noalias": True,
                    }
                )
                for i, j, k in T.grid(M, N, K):
                    with T.sblock("C"):
                        vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                        with T.init():
                            C[vi, vj] = T.float32(0.0)
                        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

        return MatmulModule

    @I.ir_module
    class MatmulModule:
        @T.prim_func
        def main(
            A: T.Buffer((M, K), "float32"),
            B: T.Buffer((K, N), "float32"),
            C: T.Buffer((M, N), "float32"),
        ) -> None:
            T.func_attr(
                {
                    "global_symbol": "main",
                    "tir.noalias": True,
                    "tirx.noalias": True,
                }
            )
            for i, j, k in T.grid(M, N, K):
                with T.block("C"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = T.float32(0.0)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

    return MatmulModule


def apply_schedule_for_target(ir_module: tvm.IRModule, target_name: str) -> tvm.IRModule:
    """Apply a tiny target-specific schedule so the scaffold runs on CPU and CUDA."""
    if target_name == "cuda":
        return _apply_cuda_schedule(ir_module)
    if target_name == "llvm":
        return ir_module
    raise ValueError(f"Unsupported target: {target_name}")


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
