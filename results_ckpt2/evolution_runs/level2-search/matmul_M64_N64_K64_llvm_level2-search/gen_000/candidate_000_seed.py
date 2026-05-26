"""Minimal Level-2 search-space candidate for matmul.

Generated Level-2 candidates should expose generate_design_space(sch). TVM
MetaSchedule will tune among the returned schedules using its normal
SearchStrategy, Builder, Runner, Database, and CostModel flow.
"""

from __future__ import annotations

import tvm


def generate_design_space(sch: tvm.s_tir.Schedule):
    spaces = [sch]

    tiled = sch.copy()
    try:
        block = tiled.get_sblock("C", func_name="main")
        i, j, k = tiled.get_loops(block)
        i_outer, i_inner = tiled.split(i, factors=[None, 4])
        j_outer, j_inner = tiled.split(j, factors=[None, 4])
        tiled.reorder(i_outer, j_outer, i_inner, j_inner, k)
        tiled.parallel(i_outer)
        tiled.vectorize(j_inner)
        spaces.append(tiled)
    except Exception:
        pass

    return spaces
