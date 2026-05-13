"""TVM MetaSchedule baseline strategy."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import tvm

from src.strategies.base import StrategyBuildConfig, StrategyBuildResult

try:
    from tvm.s_tir.meta_schedule.measure_callback import AddToDatabase, UpdateCostModel
    from tvm.s_tir.meta_schedule.tir_integration import compile_tir, tune_tir
except ImportError:  # pragma: no cover - compatibility with upstream TVM namespace
    from tvm.meta_schedule.measure_callback import AddToDatabase, UpdateCostModel
    from tvm.meta_schedule.tir_integration import compile_tir, tune_tir


class MetaScheduleStrategy:
    """Tune the canonical TensorIR module with TVM MetaSchedule."""

    name = "metaschedule"
    level = "baseline_metaschedule"

    def build(
        self,
        *,
        ir_module: tvm.IRModule,
        target: tvm.target.Target,
        target_name: str,
        config: StrategyBuildConfig,
    ) -> StrategyBuildResult:
        if config.work_dir is None:
            raise ValueError("MetaScheduleStrategy requires a work_dir.")
        if config.max_trials_global <= 0:
            raise ValueError("MetaScheduleStrategy requires max_trials_global > 0.")
        if config.num_trials_per_iter <= 0:
            raise ValueError("MetaScheduleStrategy requires num_trials_per_iter > 0.")

        work_dir = Path(config.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        tune_kwargs = {
            "work_dir": str(work_dir),
            "max_trials_global": config.max_trials_global,
            "num_trials_per_iter": config.num_trials_per_iter,
            "cost_model": config.cost_model,
            "task_scheduler": config.task_scheduler,
            "seed": config.seed,
            "num_tuning_cores": config.num_tuning_cores,
            "measure_callbacks": [AddToDatabase(), UpdateCostModel()],
            "post_optimization": config.post_optimization,
        }
        if config.max_trials_per_task is not None:
            tune_kwargs["max_trials_per_task"] = config.max_trials_per_task

        start = perf_counter()
        with target:
            database = tune_tir(ir_module, target, **tune_kwargs)
            schedule = compile_tir(database, ir_module, target)
        tuning_time_sec = perf_counter() - start

        scheduled_module = schedule.mod
        scheduled_module_path = work_dir / "scheduled_module.txt"
        scheduled_module_path.write_text(str(scheduled_module), encoding="utf-8")

        lib = tvm.build(scheduled_module, target=target)
        return StrategyBuildResult(
            lib=lib,
            scheduled_module=scheduled_module,
            metadata={
                "schedule_source": "tvm.s_tir.meta_schedule.tir_integration.tune_tir",
                "tuning_time_sec": tuning_time_sec,
                "metaschedule_work_dir": str(work_dir),
                "metaschedule_database_workload": str(work_dir / "database_workload.json"),
                "metaschedule_database_tuning_record": str(
                    work_dir / "database_tuning_record.json"
                ),
                "scheduled_module_path": str(scheduled_module_path),
                "max_trials_global": config.max_trials_global,
                "max_trials_per_task": config.max_trials_per_task,
                "num_trials_per_iter": config.num_trials_per_iter,
                "cost_model": config.cost_model,
                "task_scheduler": config.task_scheduler,
                "seed": config.seed,
                "num_tuning_cores": config.num_tuning_cores,
                "post_optimization": config.post_optimization,
                "target_for_tuning": str(target),
                "target_name": target_name,
            },
        )
