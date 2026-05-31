"""Minimal CUDA Level-2 search-space candidate for NCHW Conv2D."""

from __future__ import annotations

import tvm


def generate_design_space(sch: tvm.s_tir.Schedule):
    spaces = []

    for tile_co, tile_y, tile_x in ((1, 4, 8), (2, 4, 8), (2, 8, 8), (4, 4, 8)):
        tiled = sch.copy()
        try:
            _inline_if_present(tiled, "data_pad")
            block = tiled.get_sblock("conv", func_name="main")
            loops = list(tiled.get_loops(block))
            n, co, y, x = loops[:4]
            reduction_loops = loops[4:]
            co_block, co_thread = tiled.split(co, factors=[None, tile_co])
            y_block, y_thread = tiled.split(y, factors=[None, tile_y])
            x_block, x_thread = tiled.split(x, factors=[None, tile_x])
            block_z = tiled.fuse(n, co_block)
            tiled.reorder(
                block_z,
                y_block,
                x_block,
                co_thread,
                y_thread,
                x_thread,
                *reduction_loops,
            )
            tiled.bind(block_z, "blockIdx.z")
            tiled.bind(y_block, "blockIdx.y")
            tiled.bind(x_block, "blockIdx.x")
            tiled.bind(co_thread, "threadIdx.z")
            tiled.bind(y_thread, "threadIdx.y")
            tiled.bind(x_thread, "threadIdx.x")
            spaces.append(tiled)
        except Exception:
            pass

    return spaces


def _inline_if_present(sch, name: str) -> None:
    try:
        block = sch.get_sblock(name, func_name="main")
    except Exception:
        return
    sch.compute_inline(block)
