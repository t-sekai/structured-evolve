"""Level-1 strategy for directly generated schedule candidates."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import tempfile
from pathlib import Path
from types import ModuleType

import tvm

from src.kernels.workloads import Workload
from src.strategies.base import StrategyBuildConfig, StrategyBuildResult


APPLY_FN_NAME = "apply_schedule"
_SCHEDULE_PRIMITIVE_SEEN: dict[str, set[int]] = {}


class GeneratedScheduleStrategy:
    """Load a Python schedule candidate and compile its scheduled IRModule."""

    name = "generated-schedule"
    level = "level_1_direct_schedule"

    def build(
        self,
        *,
        workload: Workload,
        ir_module: tvm.IRModule,
        target: tvm.target.Target,
        target_name: str,
        config: StrategyBuildConfig,
    ) -> StrategyBuildResult:
        if config.generated_schedule_path is None:
            raise ValueError(
                "GeneratedScheduleStrategy requires --generated-schedule-path."
            )

        candidate_path = Path(config.generated_schedule_path)
        _install_schedule_compat()
        module = _load_candidate_module(_normalized_candidate_source(candidate_path))
        apply_schedule = getattr(module, APPLY_FN_NAME, None)
        if not callable(apply_schedule):
            raise TypeError(
                f"{candidate_path} must define callable {APPLY_FN_NAME} "
                "(ir_module, target_name)."
            )

        scheduled_module = apply_schedule(ir_module, target_name)
        if not isinstance(scheduled_module, tvm.IRModule):
            raise TypeError(
                f"{APPLY_FN_NAME} must return tvm.IRModule, got "
                f"{type(scheduled_module).__name__}."
            )

        lib = tvm.build(scheduled_module, target=target)
        return StrategyBuildResult(
            lib=lib,
            scheduled_module=scheduled_module,
            metadata={
                "schedule_source": str(candidate_path),
                "generated_schedule_path": str(candidate_path),
                "generated_schedule_sha256": _sha256(candidate_path),
                "generated_schedule_apply_fn": APPLY_FN_NAME,
                "tuning_time_sec": 0.0,
                "max_trials_global": 0,
                "num_trials_per_iter": 0,
                "workload_name": workload.name,
            },
        )


def _load_candidate_module(path: Path) -> ModuleType:
    if not path.exists():
        raise FileNotFoundError(f"Generated schedule candidate not found: {path}")
    if not path.is_file():
        raise ValueError(f"Generated schedule path is not a file: {path}")

    module_name = f"_generated_schedule_{hashlib.sha256(str(path).encode()).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load generated schedule candidate: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalized_candidate_source(path: Path) -> Path:
    """Patch common LLM API slips in a temporary candidate copy."""
    text = path.read_text(encoding="utf-8")
    normalized = text.replace(".mod()", ".mod")
    if normalized == text:
        return path

    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=f".{path.stem}.normalized.py",
        delete=False,
    )
    normalized_path = Path(temp_file.name)
    temp_file.close()
    normalized_path.write_text(normalized, encoding="utf-8")
    return normalized_path


def _install_schedule_compat() -> None:
    """Support common upstream TVM schedule spellings in generated candidates."""
    if not hasattr(tvm, "s_tir") and hasattr(tvm, "tir") and hasattr(tvm.tir, "Schedule"):
        tvm.s_tir = tvm.tir
        sys.modules.setdefault("tvm.s_tir", tvm.tir)
    if not hasattr(tvm, "s_tir") or not hasattr(tvm.s_tir, "Schedule"):
        return

    schedule_cls = tvm.s_tir.Schedule

    if not hasattr(tvm, "tir"):
        tvm.tir = tvm.s_tir
        sys.modules.setdefault("tvm.tir", tvm.s_tir)

    if not hasattr(schedule_cls, "get_block") and hasattr(schedule_cls, "get_sblock"):

        def get_block(self, name: str, func_name: str | None = None):
            return self.get_sblock(name, func_name=func_name or "main")

        schedule_cls.get_block = get_block
    if not hasattr(schedule_cls, "get_sblock") and hasattr(schedule_cls, "get_block"):

        def get_sblock(self, name: str, func_name: str | None = None):
            return self.get_block(name, func_name=func_name or "main")

        schedule_cls.get_sblock = get_sblock

    original_split = getattr(schedule_cls, "split", None)
    if original_split is not None and not getattr(original_split, "_accepts_factor", False):

        def split(self, loop, factors=None, factor=None, **kwargs):
            if factors is None:
                if factor is None:
                    raise TypeError("split() requires factors= or factor=")
                factors = [None, factor]
            return original_split(self, loop, factors=factors, **kwargs)

        split._accepts_factor = True
        schedule_cls.split = split

    _wrap_schedule_primitive(
        schedule_cls,
        "vectorize",
        allow_once_attr="_structured_evolve_vectorized",
    )
    _wrap_schedule_primitive(
        schedule_cls,
        "parallel",
        allow_once_attr="_structured_evolve_parallelized",
    )


def _wrap_schedule_primitive(schedule_cls, name: str, *, allow_once_attr: str) -> None:
    original = getattr(schedule_cls, name, None)
    if original is None or getattr(original, "_structured_evolve_safe", False):
        return

    def wrapped(self, loop, *args, **kwargs):
        seen = _SCHEDULE_PRIMITIVE_SEEN.setdefault(allow_once_attr, set())
        schedule_id = id(self)
        if schedule_id in seen:
            return None
        try:
            result = original(self, loop)
        except Exception:
            return None
        seen.add(schedule_id)
        return result

    wrapped._structured_evolve_safe = True
    setattr(schedule_cls, name, wrapped)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
