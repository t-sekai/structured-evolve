from __future__ import annotations

import tvm


def apply_schedule(ir_module: tvm.IRModule, target_name: str) -> tvm.IRModule:
    try:
        sch = tvm.s_tir.Schedule(ir_module)
        block = sch.get_sblock("C", func_name="main")
        i, j, k = sch.get_loops(block)

        if target_name == "cuda":
            i_block, i_thread = sch.split(i, factors=[None, 16])
            j_block, j_thread = sch.split(j, factors=[None, 16])
            sch.reorder(i_block, j_block, i_thread, j_thread, k)
            sch.bind(i_block, "blockIdx.y")
            sch.bind(j_block, "blockIdx.x")
            sch.bind(i_thread, "threadIdx.y")
            sch.bind(j_thread, "threadIdx.x")
        else:
            i_outer, i_inner = sch.split(i, factors=[None, 16])
            j_outer, j_inner = sch.split(j, factors=[None, 16])
            sch.reorder(i_outer, j_outer, i_inner, j_inner, k)
            sch.parallel(i_outer)
            sch.vectorize(j_inner)
        return sch.mod
    except Exception:
        return ir_module
