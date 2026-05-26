from __future__ import annotations

import tvm
import tvm.s_tir as s_tir


def apply_schedule(ir_module: tvm.IRModule, target_name: str) -> tvm.IRModule:
    sch: s_tir.Schedule = s_tir.Schedule(ir_module)
    blk = sch.get_block("C")
    loops = sch.get_loops(blk)
    i, j, k = loops[0], loops[1], loops[2]

    i_outer, i_inner = sch.split(i, factor=8)
    j_outer, j_inner = sch.split(j, factor=8)
    k_outer, k_inner = sch.split(k, factor=8)

    sch.reorder(i_outer, j_outer, i_inner, j_inner, k_outer, k_inner)

    if target_name == "cuda":
        sch.bind(i_outer, "blockIdx.x")
        sch.bind(j_outer, "blockIdx.y")
        sch.bind(i_inner, "threadIdx.x")
        sch.bind(j_inner, "threadIdx.y")
        sch.unroll(k_inner)
    else:
        sch.parallel(i_outer)
        sch.unroll(k_inner)

    return sch.mod
