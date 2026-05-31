"""Minimal CUDA Level-2 search-space candidate for NCHW Conv2D."""

from __future__ import annotations

import tvm


def generate_design_space(sch: tvm.s_tir.Schedule):
    spaces = []

    for tile_y, tile_x in ((4, 8), (8, 8)):
        tiled = sch.copy()
        try:
            block = tiled.get_sblock("conv", func_name="main")
            loops = list(tiled.get_loops(block))
            n, co, y, x = loops[:4]
            reduction_loops = loops[4:]
            y_block, y_thread = tiled.split(y, factors=[None, tile_y])
            x_block, x_thread = tiled.split(x, factors=[None, tile_x])
            tiled.reorder(n, co, y_block, x_block, y_thread, x_thread, *reduction_loops)
            tiled.bind(y_block, "blockIdx.y")
            tiled.bind(x_block, "blockIdx.x")
            tiled.bind(y_thread, "threadIdx.y")
            tiled.bind(x_thread, "threadIdx.x")
            spaces.append(tiled)
        except Exception:
            pass

    return spaces
