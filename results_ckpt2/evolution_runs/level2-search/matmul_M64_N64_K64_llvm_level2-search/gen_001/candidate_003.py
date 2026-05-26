from __future__ import annotations

import tvm


def generate_design_space(sch: tvm.s_tir.Schedule):
    spaces = [sch]

    # Variant 1: 4x4 tiling with parallel and vectorize
    try:
        s = sch.copy()
        block = s.get_sblock("C", func_name="main")
        i, j, k = s.get_loops(block)
        i_o, i_i = s.split(i, factors=[None, 4])
        j_o, j_i = s.split(j, factors=[None, 4])
        s.reorder(i_o, j_o, i_i, j_i, k)
        s.parallel(i_o)
        s.vectorize(j_i)
        spaces.append(s)
    except Exception:
        pass

    # Variant 2: 8x8 tiling with parallel and vectorize
    try:
        s = sch.copy()
        block = s.get_sblock("C", func_name="main")
        i, j, k = s.get_loops(block)
        i_o, i_i = s.split(i, factors=[None, 8])
        j_o, j_i = s.split(j, factors=[None, 8])
        s.reorder(i_o, j_o, i_i, j_i, k)
        s.parallel(i_o)
        s.vectorize(j_i)
        spaces.append(s)
    except Exception:
        pass

    # Variant 3: 4x4 tiling with unroll on inner k loop
    try:
        s = sch.copy()
        block = s.get_sblock("C", func_name="main")
        i, j, k = s.get_loops(block)
        i_o, i_i = s.split(i, factors=[None, 4])
        j_o, j_i = s.split(j, factors=[None, 4])
        k_o, k_i = s.split(k, factors=[None, 4])
        s.reorder(i_o, j_o, k_o, i_i, j_i, k_i)
        s.parallel(i_o)
        s.unroll(k_i)
        spaces.append(s)
    except Exception:
        pass

    return spaces
