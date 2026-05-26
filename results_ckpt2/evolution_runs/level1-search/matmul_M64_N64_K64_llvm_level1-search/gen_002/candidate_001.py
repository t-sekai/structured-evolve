from __future__ import annotations

import tvm


def apply_schedule(ir_module: tvm.IRModule, target_name: str) -> tvm.IRModule:
    sch = tvm.s_tir.Schedule(ir_module)
    block = sch.get_block("C")
    i, j, k = sch.get_loops(block)
    io, ii = sch.split(i, factor=32)
    jo, ji = sch.split(j, factor=32)
    ko, ki = sch.split(k, factor=32)
    sch.reorder(io, jo, ko, ii, ji, ki)
    if target_name == "cuda":
        sch.bind(io, "blockIdx.x")
        sch.bind(jo, "blockIdx.y")
        sch.bind(ii, "threadIdx.x")
        sch.bind(ji, "threadIdx.y")
    else:
        sch.parallel(io)
    return sch.mod
