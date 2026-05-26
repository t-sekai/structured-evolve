from __future__ import annotations
import tvm

def generate_design_space(sch: tvm.s_tir.Schedule):
    spaces = [sch]

    # Variant 1: small tile 4x4 with parallel and vectorize
    try:
        tiled = sch.copy()
        block = tiled.get_sblock("C", func_name="main")
        i, j, k = tiled.get_loops(block)
        i_o, i_i = tiled.split(i, factor=4)
        j_o, j_i = tiled.split(j, factor=4)
        tiled.reorder(i_o, j_o, i_i, j_i, k)
        tiled.parallel(i_o)
        tiled.vectorize(j_i)
        spaces.append(tiled)
    except Exception:
        pass

    # Variant 2: larger tile 8x8 with reduction split and unroll
    try:
        tiled = sch.copy()
        block = tiled.get_sblock("C", func_name="main")
        i, j, k = tiled.get_loops(block)
        i_o, i_i = tiled.split(i, factor=8)
        j_o, j_i = tiled.split(j, factor=8)
        k_o, k_i = tiled.split(k, factor=8)
        tiled.reorder(i_o, j_o, k_o, i_i, j_i, k_i)
        tiled.parallel(i_o)
        tiled.vectorize(j_i)
        tiled.unroll(k_i)
        spaces.append(tiled)
    except Exception:
        pass

    # Variant 3: cache reads/writes with tile 4x4
    try:
        tiled = sch.copy()
        block = tiled.get_sblock("C", func_name="main")
        a = tiled.cache_read(block, 0, "global")
        b = tiled.cache_read(block, 1, "global")
        c = tiled.cache_write(block, 0, "global")
        i, j, k = tiled.get_loops(block)
        i_o, i_i = tiled.split(i, factor=4)
        j_o, j_i = tiled.split(j, factor=4)
        tiled.reorder(i_o, j_o, i_i, j_i, k)
        tiled.parallel(i_o)
        tiled.vectorize(j_i)
        spaces.append(tiled)
    except Exception:
        pass

    return spaces
