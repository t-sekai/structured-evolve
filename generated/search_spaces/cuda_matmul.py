"""Minimal CUDA Level-2 search-space candidate for matmul."""

from __future__ import annotations

import tvm


def generate_design_space(sch: tvm.s_tir.Schedule):
    spaces = []

    for tile_i, tile_j in ((16, 16), (8, 32)):
        tiled = sch.copy()
        try:
            block = tiled.get_sblock("C", func_name="main")
            i, j, k = tiled.get_loops(block)
            i_block, i_thread = tiled.split(i, factors=[None, tile_i])
            j_block, j_thread = tiled.split(j, factors=[None, tile_j])
            tiled.reorder(i_block, j_block, i_thread, j_thread, k)
            tiled.bind(i_block, "blockIdx.y")
            tiled.bind(j_block, "blockIdx.x")
            tiled.bind(i_thread, "threadIdx.y")
            tiled.bind(j_thread, "threadIdx.x")
            spaces.append(tiled)
        except Exception:
            pass

    return spaces
