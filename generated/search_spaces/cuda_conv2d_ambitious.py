"""Ambitious CUDA Level-2 search-space seed for NCHW Conv2D evolution."""

from __future__ import annotations

import tvm


def generate_design_space(sch: tvm.s_tir.Schedule):
    spaces = []

    # Variant 1: 2 output channels x 8x16 spatial tile, 64 CUDA threads.
    _append_local_accumulator_variant(
        spaces,
        sch,
        co_thread=2,
        y_thread=4,
        y_inner=2,
        x_thread=8,
        x_inner=2,
        rc_tile=8,
    )

    # Variant 2: more channel parallelism, still retaining a local accumulator.
    _append_local_accumulator_variant(
        spaces,
        sch,
        co_thread=4,
        y_thread=4,
        y_inner=1,
        x_thread=8,
        x_inner=2,
        rc_tile=8,
    )

    # Variant 3: wider y parallelism for large feature maps.
    _append_local_accumulator_variant(
        spaces,
        sch,
        co_thread=2,
        y_thread=8,
        y_inner=1,
        x_thread=8,
        x_inner=1,
        rc_tile=8,
    )

    # Variant 4: wider x parallelism, useful when x extent dominates the tile.
    _append_local_accumulator_variant(
        spaces,
        sch,
        co_thread=1,
        y_thread=4,
        y_inner=2,
        x_thread=16,
        x_inner=1,
        rc_tile=8,
    )

    _append_simple_fallback(spaces, sch)
    return spaces


def _append_local_accumulator_variant(
    spaces,
    sch,
    *,
    co_thread: int,
    y_thread: int,
    y_inner: int,
    x_thread: int,
    x_inner: int,
    rc_tile: int,
) -> None:
    candidate = sch.copy()
    try:
        _inline_if_present(candidate, "data_pad")
        block = candidate.get_sblock("conv", func_name="main")
        n, co, y, x, rc, ry, rx = candidate.get_loops(block)
        co_block, co_t, co_i = candidate.split(co, factors=[None, co_thread, 1])
        y_block, y_t, y_i = candidate.split(y, factors=[None, y_thread, y_inner])
        x_block, x_t, x_i = candidate.split(x, factors=[None, x_thread, x_inner])
        rc_outer, rc_inner = candidate.split(rc, factors=[None, rc_tile])
        block_z = candidate.fuse(n, co_block)
        candidate.reorder(
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

        conv_local = candidate.cache_write(block, 0, "local")
        candidate.reverse_compute_at(conv_local, x_block)

        candidate.bind(block_z, "blockIdx.z")
        candidate.bind(y_block, "blockIdx.y")
        candidate.bind(x_block, "blockIdx.x")
        candidate.bind(co_t, "threadIdx.z")
        candidate.bind(y_t, "threadIdx.y")
        candidate.bind(x_t, "threadIdx.x")
        candidate.unroll(rc_inner)
        candidate.unroll(ry)
        candidate.unroll(rx)
        spaces.append(candidate)
    except Exception:
        pass


def _append_simple_fallback(spaces, sch) -> None:
    candidate = sch.copy()
    try:
        _inline_if_present(candidate, "data_pad")
        block = candidate.get_sblock("conv", func_name="main")
        loops = list(candidate.get_loops(block))
        n, co, y, x = loops[:4]
        reduction_loops = loops[4:]
        co_block, co_thread = candidate.split(co, factors=[None, 2])
        y_block, y_thread = candidate.split(y, factors=[None, 4])
        x_block, x_thread = candidate.split(x, factors=[None, 8])
        block_z = candidate.fuse(n, co_block)
        candidate.reorder(
            block_z,
            y_block,
            x_block,
            co_thread,
            y_thread,
            x_thread,
            *reduction_loops,
        )
        candidate.bind(block_z, "blockIdx.z")
        candidate.bind(y_block, "blockIdx.y")
        candidate.bind(x_block, "blockIdx.x")
        candidate.bind(co_thread, "threadIdx.z")
        candidate.bind(y_thread, "threadIdx.y")
        candidate.bind(x_thread, "threadIdx.x")
        spaces.append(candidate)
    except Exception:
        pass


def _inline_if_present(sch, name: str) -> None:
    try:
        block = sch.get_sblock(name, func_name="main")
    except Exception:
        return
    sch.compute_inline(block)
