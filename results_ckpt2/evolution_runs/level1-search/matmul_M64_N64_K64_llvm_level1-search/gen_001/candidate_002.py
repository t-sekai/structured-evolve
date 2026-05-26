from __future__ import annotations

import tvm
import tvm.s_tir

def apply_schedule(ir_module: tvm.IRModule, target_name: str) -> tvm.IRModule:
    sch = tvm.s_tir.Schedule(ir_module)
    block = sch.get_block("C")
    loops = sch.get_loops(block)
    if len(loops) >= 2:
        i_outer, i_inner = sch.split(loops[0], factor=8)
        j_outer, j_inner = sch.split(loops[1], factor=8)
        sch.reorder(i_outer, j_outer, i_inner, j_inner, *loops[2:])
        sch.parallel(i_outer)
        sch.vectorize(i_inner)
    return sch.mod
