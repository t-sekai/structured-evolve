"""Buildable CUDA Level-1 schedule seed for NCHW Conv2D."""

from __future__ import annotations

import tvm


def apply_schedule(ir_module: tvm.IRModule, target_name: str) -> tvm.IRModule:
    if target_name != "cuda":
        return ir_module

    sch = tvm.s_tir.Schedule(ir_module)
    try:
        _inline_if_present(sch, "data_pad")
        block = sch.get_sblock("conv", func_name="main")
        loops = list(sch.get_loops(block))
        n, co, y, x = loops[:4]
        reduction_loops = loops[4:]

        co_block, co_thread = sch.split(co, factors=[None, 2])
        y_block, y_thread = sch.split(y, factors=[None, 4])
        x_block, x_thread = sch.split(x, factors=[None, 8])
        block_z = sch.fuse(n, co_block)
        sch.reorder(
            block_z,
            y_block,
            x_block,
            co_thread,
            y_thread,
            x_thread,
            *reduction_loops,
        )
        sch.bind(block_z, "blockIdx.z")
        sch.bind(y_block, "blockIdx.y")
        sch.bind(x_block, "blockIdx.x")
        sch.bind(co_thread, "threadIdx.z")
        sch.bind(y_thread, "threadIdx.y")
        sch.bind(x_thread, "threadIdx.x")
        return sch.mod
    except Exception:
        return ir_module


def _inline_if_present(sch, name: str) -> None:
    try:
        block = sch.get_sblock(name, func_name="main")
    except Exception:
        return
    sch.compute_inline(block)
