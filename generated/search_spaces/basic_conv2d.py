"""Minimal Level-2 search-space candidate for NCHW Conv2D."""

from __future__ import annotations

import tvm


def generate_design_space(sch: tvm.s_tir.Schedule):
    spaces = [sch]

    for tile_y, tile_x in ((2, 4), (4, 4)):
        tiled = sch.copy()
        try:
            block = tiled.get_sblock("conv", func_name="main")
            loops = list(tiled.get_loops(block))
            n, co, y, x = loops[:4]
            reduction_loops = loops[4:]
            y_outer, y_inner = tiled.split(y, factors=[None, tile_y])
            x_outer, x_inner = tiled.split(x, factors=[None, tile_x])
            tiled.reorder(n, co, y_outer, x_outer, y_inner, x_inner, *reduction_loops)
            tiled.parallel(co)
            tiled.vectorize(x_inner)
            spaces.append(tiled)
        except Exception:
            pass

    return spaces
