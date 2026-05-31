"""Current hand-written baseline schedules."""

from __future__ import annotations

import tvm

from src.kernels.workloads import Workload
from src.strategies.base import StrategyBuildConfig, StrategyBuildResult


class FixedScheduleStrategy:
    """Keep simple workload-specific schedules as comparable fixed baselines."""

    name = "fixed"
    level = "baseline_fixed"

    def build(
        self,
        *,
        workload: Workload,
        ir_module: tvm.IRModule,
        target: tvm.target.Target,
        target_name: str,
        config: StrategyBuildConfig,
    ) -> StrategyBuildResult:
        del config
        scheduled_module = workload.fixed_schedule(ir_module, target_name)
        lib = tvm.build(scheduled_module, target=target)
        return StrategyBuildResult(
            lib=lib,
            scheduled_module=scheduled_module,
            metadata={
                "schedule_source": f"built-in fixed schedule for {workload.name}",
                "tuning_time_sec": 0.0,
                "max_trials_global": 0,
                "num_trials_per_iter": 0,
                "workload_name": workload.name,
            },
        )
