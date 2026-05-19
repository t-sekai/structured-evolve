"""Level-2 strategy for generated MetaSchedule search spaces."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from time import perf_counter
from types import ModuleType
from typing import Any, Callable

import tvm

from src.strategies.base import StrategyBuildConfig, StrategyBuildResult

try:
    from tvm.s_tir.meta_schedule.builder import LocalBuilder
    from tvm.s_tir.meta_schedule.measure_callback import AddToDatabase, UpdateCostModel
    from tvm.s_tir.meta_schedule.runner import LocalRunner
    from tvm.s_tir.meta_schedule.space_generator import SpaceGenerator
    from tvm.s_tir.meta_schedule.tir_integration import compile_tir, tune_tir
except ImportError:  # pragma: no cover - compatibility with upstream TVM namespace
    from tvm.meta_schedule.builder import LocalBuilder
    from tvm.meta_schedule.measure_callback import AddToDatabase, UpdateCostModel
    from tvm.meta_schedule.runner import LocalRunner
    from tvm.meta_schedule.space_generator import SpaceGenerator
    from tvm.meta_schedule.tir_integration import compile_tir, tune_tir


SPACE_FN_NAME = "generate_design_space"
CREATE_FN_NAME = "create_space_generator"


class GeneratedSearchSpaceStrategy:
    """Tune within a generated MetaSchedule design space."""

    name = "generated-search-space"
    level = "level_2_evolved_search_space"

    def build(
        self,
        *,
        ir_module: tvm.IRModule,
        target: tvm.target.Target,
        target_name: str,
        config: StrategyBuildConfig,
    ) -> StrategyBuildResult:
        if config.generated_search_space_path is None:
            raise ValueError(
                "GeneratedSearchSpaceStrategy requires --generated-search-space-path."
            )
        if config.work_dir is None:
            raise ValueError("GeneratedSearchSpaceStrategy requires a work_dir.")
        if config.max_trials_global <= 0:
            raise ValueError("GeneratedSearchSpaceStrategy requires max_trials_global > 0.")
        if config.num_trials_per_iter <= 0:
            raise ValueError("GeneratedSearchSpaceStrategy requires num_trials_per_iter > 0.")

        candidate_path = Path(config.generated_search_space_path)
        module = _load_candidate_module(candidate_path)
        space, fallback_space_fn = _space_from_module(module=module, target_name=target_name)

        work_dir = Path(config.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        tune_kwargs: dict[str, Any] = {
            "work_dir": str(work_dir),
            "max_trials_global": config.max_trials_global,
            "num_trials_per_iter": config.num_trials_per_iter,
            "cost_model": config.cost_model,
            "task_scheduler": config.task_scheduler,
            "seed": config.seed,
            "num_tuning_cores": config.num_tuning_cores,
            "measure_callbacks": [AddToDatabase(), UpdateCostModel()],
            "post_optimization": config.post_optimization,
            "space": space,
            "builder": LocalBuilder(max_workers=1, timeout_sec=10.0),
            "runner": LocalRunner(timeout_sec=10.0),
        }
        if config.max_trials_per_task is not None:
            tune_kwargs["max_trials_per_task"] = config.max_trials_per_task

        start = perf_counter()
        with target:
            database = tune_tir(ir_module, target, **tune_kwargs)
            schedule = compile_tir(database, ir_module, target)
        tuning_time_sec = perf_counter() - start

        used_fallback_schedule = schedule is None
        if schedule is None:
            schedule = _fallback_schedule(
                ir_module=ir_module,
                generate_design_space=fallback_space_fn,
            )

        scheduled_module = schedule.mod
        scheduled_module_path = work_dir / "scheduled_module.txt"
        scheduled_module_path.write_text(str(scheduled_module), encoding="utf-8")

        lib = tvm.build(scheduled_module, target=target)
        return StrategyBuildResult(
            lib=lib,
            scheduled_module=scheduled_module,
            metadata={
                "schedule_source": str(candidate_path),
                "generated_search_space_path": str(candidate_path),
                "generated_search_space_sha256": _sha256(candidate_path),
                "generated_search_space_entrypoint": _entrypoint_name(module),
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
                "used_fallback_schedule": used_fallback_schedule,
            },
        )


def _space_from_module(*, module: ModuleType, target_name: str):
    create_space_generator = getattr(module, CREATE_FN_NAME, None)
    if callable(create_space_generator):
        return create_space_generator(target_name), None

    generate_design_space = getattr(module, SPACE_FN_NAME, None)
    if callable(generate_design_space):
        wrapped = _wrap_space_fn(generate_design_space)
        return SpaceGenerator.create(wrapped), wrapped

    raise TypeError(
        f"Generated search-space candidate must define callable {SPACE_FN_NAME}(sch) "
        f"or {CREATE_FN_NAME}(target_name)."
    )


def _wrap_space_fn(func: Callable):
    def wrapped(sch):
        result = func(sch)
        if result is None:
            return [sch]
        if isinstance(result, list):
            return result
        if isinstance(result, tuple):
            return list(result)
        return result

    return wrapped


def _fallback_schedule(
    *,
    ir_module: tvm.IRModule,
    generate_design_space: Callable | None,
):
    """Return a buildable schedule when a tiny MetaSchedule run records no winner."""
    sch = tvm.s_tir.Schedule(ir_module)
    if generate_design_space is None:
        return sch

    try:
        candidates = generate_design_space(sch)
    except Exception:
        return sch

    if not isinstance(candidates, list):
        candidates = [candidates]
    for candidate in reversed(candidates):
        if hasattr(candidate, "mod"):
            return candidate
    return sch


def _entrypoint_name(module: ModuleType) -> str:
    if callable(getattr(module, CREATE_FN_NAME, None)):
        return CREATE_FN_NAME
    return SPACE_FN_NAME


def _load_candidate_module(path: Path) -> ModuleType:
    if not path.exists():
        raise FileNotFoundError(f"Generated search-space candidate not found: {path}")
    if not path.is_file():
        raise ValueError(f"Generated search-space path is not a file: {path}")

    module_name = f"_generated_search_space_{hashlib.sha256(str(path).encode()).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load generated search-space candidate: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
