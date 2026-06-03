"""Ambitious CUDA Level-2 search-space seed for matmul evolution."""

from __future__ import annotations

import tvm


def generate_design_space(sch: tvm.s_tir.Schedule):
    spaces = []

    # Variant 1: 32x32 block tile, 128 compute threads, 4x2 per-thread outputs.
    _append_thread_tiled_variant(
        spaces,
        sch,
        i_thread=8,
        i_inner=4,
        j_thread=16,
        j_inner=2,
        k_inner=32,
    )

    # Variant 2: 32x32 block tile, 64 compute threads, 4x4 per-thread outputs.
    _append_thread_tiled_variant(
        spaces,
        sch,
        i_thread=8,
        i_inner=4,
        j_thread=8,
        j_inner=4,
        k_inner=32,
    )

    # Variant 3: 16x32 block tile, smaller y tile for skinny-M workloads.
    _append_thread_tiled_variant(
        spaces,
        sch,
        i_thread=4,
        i_inner=4,
        j_thread=16,
        j_inner=2,
        k_inner=32,
    )

    # Variant 4: simple fallback split/bind schedule.
    fallback = sch.copy()
    try:
        block = fallback.get_sblock("C", func_name="main")
        i, j, k = fallback.get_loops(block)
        i_block, i_thread = fallback.split(i, factors=[None, 16])
        j_block, j_thread = fallback.split(j, factors=[None, 16])
        fallback.reorder(i_block, j_block, i_thread, j_thread, k)
        fallback.bind(i_block, "blockIdx.y")
        fallback.bind(j_block, "blockIdx.x")
        fallback.bind(i_thread, "threadIdx.y")
        fallback.bind(j_thread, "threadIdx.x")
        spaces.append(fallback)
    except Exception:
        pass

    return spaces


def _append_thread_tiled_variant(
    spaces,
    sch,
    *,
    i_thread: int,
    i_inner: int,
    j_thread: int,
    j_inner: int,
    k_inner: int,
) -> None:
    candidate = sch.copy()
    try:
        block = candidate.get_sblock("C", func_name="main")
        i, j, k = candidate.get_loops(block)
        i_block, i_t, i_in = candidate.split(i, factors=[None, i_thread, i_inner])
        j_block, j_t, j_in = candidate.split(j, factors=[None, j_thread, j_inner])
        k_block, k_in = candidate.split(k, factors=[None, k_inner])
        candidate.reorder(
            i_block,
            j_block,
            i_t,
            j_t,
            i_in,
            j_in,
            k_block,
            k_in,
        )
        candidate.bind(i_block, "blockIdx.y")
        candidate.bind(j_block, "blockIdx.x")
        candidate.bind(i_t, "threadIdx.y")
        candidate.bind(j_t, "threadIdx.x")
        candidate.unroll(k_in)
        spaces.append(candidate)
    except Exception:
        pass
