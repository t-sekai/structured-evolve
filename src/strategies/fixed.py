"""Current hand-written baseline schedule."""

from __future__ import annotations

import tvm

from src.kernels.matmul_tir import apply_schedule_for_target
from src.strategies.base import StrategyBuildConfig, StrategyBuildResult


class FixedScheduleStrategy:
    """Keep the checkpoint-1 schedule as a comparable fixed baseline."""

    name = "fixed"
    level = "baseline_fixed"

    def build(
        self,
        *,
        ir_module: tvm.IRModule,
        target: tvm.target.Target,
        target_name: str,
        config: StrategyBuildConfig,
    ) -> StrategyBuildResult:
        del config
        scheduled_module = apply_schedule_for_target(ir_module, target_name)
        lib = tvm.build(scheduled_module, target=target)
        return StrategyBuildResult(
            lib=lib,
            scheduled_module=scheduled_module,
            metadata={
                "schedule_source": "src.kernels.matmul_tir.apply_schedule_for_target",
                "tuning_time_sec": 0.0,
                "max_trials_global": 0,
                "num_trials_per_iter": 0,
            },
        )
