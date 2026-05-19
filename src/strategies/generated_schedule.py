"""Level-1 strategy for directly generated schedule candidates."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType

import tvm

from src.strategies.base import StrategyBuildConfig, StrategyBuildResult


APPLY_FN_NAME = "apply_schedule"


class GeneratedScheduleStrategy:
    """Load a Python schedule candidate and compile its scheduled IRModule."""

    name = "generated-schedule"
    level = "level_1_direct_schedule"

    def build(
        self,
        *,
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
        module = _load_candidate_module(candidate_path)
        apply_schedule = getattr(module, APPLY_FN_NAME, None)
        if not callable(apply_schedule):
            raise TypeError(
                f"{candidate_path} must define callable {APPLY_FN_NAME}"
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
