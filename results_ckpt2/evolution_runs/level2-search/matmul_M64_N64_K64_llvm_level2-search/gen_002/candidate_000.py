from __future__ import annotations

import tvm


def generate_design_space(sch: tvm.s_tir.Schedule):
    spaces = [sch]
    try:
        tiled = sch.copy()
        block = tiled.get_sblock("C", func_name="main")
        i, j, k = tiled.get_loops(block)
        i_o, i_i = tiled.split(i, factors=[None, 4])
        j_o, j_i = tiled.split(j, factors=[None, 4])
        tiled.reorder(i_o, j_o, i_i, j_i, k)
        tiled.parallel(i_o)
        tiled.vectorize(j_i)
        spaces.append(tiled)
    except Exception:
        pass
    try:
        tiled = sch.copy()
        block = tiled.get_sblock("C", func_name="main")
        i, j, k = tiled.get_loops(block)
        i_o, i_i = tiled.split(i, factors=[None, 8])
        j_o, j_i = tiled.split(j, factors=[None, 8])
        k_o, k_i = tiled.split(k, factors=[None, 8])
        tiled.reorder(i_o, j_o, k_o, i_i, j_i, k_i)
        tiled.parallel(i_o)
        tiled.vectorize(j_i)
        tiled.unroll(i_i)
        spaces.append(tiled)
    except Exception:
        pass
    try:
        tiled = sch.copy()
        block = tiled.get_sblock("C", func_name="main")
        cache = tiled.cache_write(block, 0, "global")
        i, j, k = tiled.get_loops(block)
        i_o, i_i = tiled.split(i, factors=[None, 4])
        j_o, j_i = tiled.split(j, factors=[None, 4])
        tiled.reorder(i_o, j_o, i_i, j_i, k)
        tiled.compute_at(cache, i_o)
        tiled.parallel(i_o)
        tiled.vectorize(j_i)
        spaces.append(tiled)
    except Exception:
        pass
    return spaces
