"""Strategy for rebuilding an exact scheduled TensorIR module."""

from __future__ import annotations

from pathlib import Path

import tvm

from src.strategies.base import StrategyBuildConfig, StrategyBuildResult


class SavedScheduledModuleStrategy:
    """Build a losslessly serialized module selected during search."""

    name = "saved-scheduled-module"
    level = "level_2_exact_winner_schedule"

    def build(
        self,
        *,
        ir_module: tvm.IRModule,
        target: tvm.target.Target,
        target_name: str,
        config: StrategyBuildConfig,
    ) -> StrategyBuildResult:
        del ir_module
        if config.saved_scheduled_module_json_path is None:
            raise ValueError(
                "SavedScheduledModuleStrategy requires a serialized scheduled-module "
                "JSON artifact from Level 2 search."
            )

        json_path = Path(config.saved_scheduled_module_json_path)
        if not json_path.is_file():
            raise FileNotFoundError(f"Scheduled-module JSON artifact not found: {json_path}")

        scheduled_module = tvm.ir.load_json(json_path.read_text(encoding="utf-8"))
        if not isinstance(scheduled_module, tvm.IRModule):
            raise TypeError(
                "Serialized scheduled-module artifact must contain a tvm.IRModule, "
                f"got {type(scheduled_module).__name__}."
            )

        lib = tvm.build(scheduled_module, target=target)
        return StrategyBuildResult(
            lib=lib,
            scheduled_module=scheduled_module,
            metadata={
                "schedule_source": str(json_path),
                "scheduled_module_path": (
                    str(config.saved_scheduled_module_path)
                    if config.saved_scheduled_module_path is not None
                    else None
                ),
                "scheduled_module_json_path": str(json_path),
                "exact_schedule_reused": True,
                "tuning_time_sec": 0.0,
                "max_trials_global": 0,
                "num_trials_per_iter": 0,
                "target_name": target_name,
            },
        )
