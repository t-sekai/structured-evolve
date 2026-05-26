from __future__ import annotations

import tvm


def generate_design_space(sch: tvm.s_tir.Schedule):
    spaces = [sch]

    try:
        s1 = sch.copy()
        blk = s1.get_sblock("C", func_name="main")
        i, j, k = s1.get_loops(blk)
        i_o, i_i = s1.split(i, factors=[None, 4])
        j_o, j_i = s1.split(j, factors=[None, 4])
        s1.reorder(i_o, j_o, i_i, j_i, k)
        s1.parallel(i_o)
        s1.vectorize(j_i)
        spaces.append(s1)
    except Exception:
        pass

    try:
        s2 = sch.copy()
        blk = s2.get_sblock("C", func_name="main")
        i, j, k = s2.get_loops(blk)
        i_o, i_i = s2.split(i, factors=[None, 8])
        j_o, j_i = s2.split(j, factors=[None, 8])
        k_o, k_i = s2.split(k, factors=[None, 8])
        s2.reorder(i_o, j_o, k_o, i_i, j_i, k_i)
        s2.parallel(i_o)
        s2.unroll(i_i)
        spaces.append(s2)
    except Exception:
        pass

    try:
        s3 = sch.copy()
        blk = s3.get_sblock("C", func_name="main")
        i, j, k = s3.get_loops(blk)
        i_o, i_i = s3.split(i, factors=[None, 16])
        j_o, j_i = s3.split(j, factors=[None, 16])
        fused = s3.fuse(i_o, j_o)
        s3.parallel(fused)
        s3.vectorize(j_i)
        spaces.append(s3)
    except Exception:
        pass

    return spaces
