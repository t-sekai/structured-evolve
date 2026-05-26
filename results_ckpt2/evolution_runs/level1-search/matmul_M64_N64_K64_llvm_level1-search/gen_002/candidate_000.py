from __future__ import annotations

import tvm

def apply_schedule(ir_module: tvm.IRModule, target_name: str) -> tvm.IRModule:
    sch = tvm.s_tir.Schedule(ir_module)
    blk = sch.get_block("C")
    i, j, k = sch.get_loops(blk)
    io, ii = sch.split(i, factor=16)
    jo, ji = sch.split(j, factor=16)
    sch.reorder(io, jo, ii, ji, k)
    if target_name == "cuda":
        sch.bind(io, "blockIdx.x")
        sch.bind(jo, "blockIdx.y")
        sch.bind(ii, "threadIdx.x")
        sch.bind(ji, "threadIdx.y")
    else:
        sch.parallel(io)
        sch.vectorize(ii)
        sch.unroll(k)
    return sch.mod
