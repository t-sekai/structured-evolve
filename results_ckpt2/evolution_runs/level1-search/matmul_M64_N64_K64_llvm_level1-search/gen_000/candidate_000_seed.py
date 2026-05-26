"""Minimal Level-1 schedule candidate.

Generated candidates should expose apply_schedule(ir_module, target_name) and
return a tvm.IRModule. This identity candidate is useful as a smoke test.
"""

from __future__ import annotations

import tvm


def apply_schedule(ir_module: tvm.IRModule, target_name: str) -> tvm.IRModule:
    del target_name
    return ir_module
