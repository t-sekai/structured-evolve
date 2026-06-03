"""Ambitious CUDA Level-1 seed for matmul evolution."""

from __future__ import annotations

import tvm


def apply_schedule(ir_module: tvm.IRModule, target_name: str) -> tvm.IRModule:
    if target_name != "cuda":
        return ir_module

    try:
        return _thread_tiled_multi_output(ir_module)
    except Exception:
        try:
            return _simple_split_bind(ir_module)
        except Exception:
            return ir_module


def _thread_tiled_multi_output(ir_module: tvm.IRModule) -> tvm.IRModule:
    """CUDA seed with per-thread output tiles, avoiding fragile shared staging."""
    sch = tvm.s_tir.Schedule(ir_module)
    block = sch.get_sblock("C", func_name="main")
    i, j, k = sch.get_loops(block)

    i_block, i_thread, i_inner = sch.split(i, factors=[None, 8, 4])
    j_block, j_thread, j_inner = sch.split(j, factors=[None, 16, 2])
    k_block, k_inner = sch.split(k, factors=[None, 32])
    sch.reorder(
        i_block,
        j_block,
        i_thread,
        j_thread,
        i_inner,
        j_inner,
        k_block,
        k_inner,
    )

    sch.bind(i_block, "blockIdx.y")
    sch.bind(j_block, "blockIdx.x")
    sch.bind(i_thread, "threadIdx.y")
    sch.bind(j_thread, "threadIdx.x")
    sch.unroll(k_inner)
    return sch.mod


def _simple_split_bind(ir_module: tvm.IRModule) -> tvm.IRModule:
    sch = tvm.s_tir.Schedule(ir_module)
    block = sch.get_sblock("C", func_name="main")
    i, j, k = sch.get_loops(block)
    i_block, i_thread = sch.split(i, factors=[None, 16])
    j_block, j_thread = sch.split(j, factors=[None, 16])
    sch.reorder(i_block, j_block, i_thread, j_thread, k)
    sch.bind(i_block, "blockIdx.y")
    sch.bind(j_block, "blockIdx.x")
    sch.bind(i_thread, "threadIdx.y")
    sch.bind(j_thread, "threadIdx.x")
    return sch.mod

