"""Ambitious CUDA Level-1 seed for NCHW Conv2D evolution."""

from __future__ import annotations

import tvm


def apply_schedule(ir_module: tvm.IRModule, target_name: str) -> tvm.IRModule:
    if target_name != "cuda":
        return ir_module

    variants = (
        dict(co_thread=2, y_thread=4, y_inner=2, x_thread=8, x_inner=2, rc_tile=8),
        dict(co_thread=4, y_thread=4, y_inner=1, x_thread=8, x_inner=2, rc_tile=8),
        dict(co_thread=2, y_thread=8, y_inner=1, x_thread=8, x_inner=1, rc_tile=8),
    )
    for params in variants:
        try:
            return _local_accumulator_thread_tiled(ir_module, **params)
        except Exception:
            pass

    try:
        return _simple_split_bind(ir_module)
    except Exception:
        return ir_module


def _local_accumulator_thread_tiled(
    ir_module: tvm.IRModule,
    *,
    co_thread: int,
    y_thread: int,
    y_inner: int,
    x_thread: int,
    x_inner: int,
    rc_tile: int,
) -> tvm.IRModule:
    sch = tvm.s_tir.Schedule(ir_module)
    _inline_if_present(sch, "data_pad")
    block = sch.get_sblock("conv", func_name="main")
    n, co, y, x, rc, ry, rx = sch.get_loops(block)

    co_block, co_t, co_i = sch.split(co, factors=[None, co_thread, 1])
    y_block, y_t, y_i = sch.split(y, factors=[None, y_thread, y_inner])
    x_block, x_t, x_i = sch.split(x, factors=[None, x_thread, x_inner])
    rc_outer, rc_inner = sch.split(rc, factors=[None, rc_tile])
    block_z = sch.fuse(n, co_block)
    sch.reorder(
        block_z,
        y_block,
        x_block,
        co_t,
        y_t,
        x_t,
        rc_outer,
        rc_inner,
        ry,
        rx,
        co_i,
        y_i,
        x_i,
    )

    conv_local = sch.cache_write(block, 0, "local")
    sch.reverse_compute_at(conv_local, x_block)

    sch.bind(block_z, "blockIdx.z")
    sch.bind(y_block, "blockIdx.y")
    sch.bind(x_block, "blockIdx.x")
    sch.bind(co_t, "threadIdx.z")
    sch.bind(y_t, "threadIdx.y")
    sch.bind(x_t, "threadIdx.x")
    sch.unroll(rc_inner)
    sch.unroll(ry)
    sch.unroll(rx)
    return sch.mod


def _simple_split_bind(ir_module: tvm.IRModule) -> tvm.IRModule:
    sch = tvm.s_tir.Schedule(ir_module)
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


def _inline_if_present(sch, name: str) -> None:
    try:
        block = sch.get_sblock(name, func_name="main")
    except Exception:
        return
    sch.compute_inline(block)
