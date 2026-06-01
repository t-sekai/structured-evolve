"""Benchmark-suite abstractions built on first-class experiment methods."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class MatmulTaskCase:
    """One matmul workload shape on one TVM target."""

    M: int
    N: int
    K: int
    target: str

    @property
    def case_id(self) -> str:
        return f"matmul_M{self.M}_N{self.N}_K{self.K}_{self.target}"


@dataclass(frozen=True)
class Conv2DTaskCase:
    """One NCHW Conv2D workload shape on one TVM target."""

    batch: int
    in_channels: int
    height: int
    width: int
    out_channels: int
    kernel_h: int
    kernel_w: int
    stride: int
    padding: int
    target: str

    @property
    def case_id(self) -> str:
        return (
            f"conv2d_B{self.batch}_CI{self.in_channels}_H{self.height}_W{self.width}_"
            f"CO{self.out_channels}_KH{self.kernel_h}_KW{self.kernel_w}_"
            f"S{self.stride}_P{self.padding}_{self.target}"
        )

    @property
    def workload_params(self) -> dict[str, int]:
        return {
            "batch": self.batch,
            "in_channels": self.in_channels,
            "height": self.height,
            "width": self.width,
            "out_channels": self.out_channels,
            "kernel_h": self.kernel_h,
            "kernel_w": self.kernel_w,
            "stride": self.stride,
            "padding": self.padding,
        }


@dataclass(frozen=True)
class MethodCase:
    """One experiment method configuration to run for each task case."""

    name: str
    generated_schedule_path: Path | None = None
    generated_search_space_path: Path | None = None
    max_trials_global: int = 64
    max_trials_per_task: int | None = None
    num_trials_per_iter: int = 64
    cost_model: str = "xgb"
    task_scheduler: str = "gradient"
    seed: int | None = 0
    num_tuning_cores: int | str = "physical"
    post_optimization: bool = False
    generations: int = 1
    population_size: int = 2
    survivors: int = 1
    search_num_warmup: int | None = None
    search_num_trials: int | None = None
    search_benchmark_invocations: int | None = None
    search_min_repeat_ms: int | None = None
    search_max_trials_global: int | None = None
    search_num_trials_per_iter: int | None = None
    level2_final_evaluation_policy: str = "fresh-retune"
    include_evaluator_feedback: bool = True
    enable_rejection_cascade: bool = True
    enable_elite_carry_forward: bool = False
    enable_diverse_inspiration: bool = True
    dry_run: bool = True


@dataclass(frozen=True)
class SuiteRunConfig:
    """Shared settings for a benchmark suite run."""

    output_dir: Path
    suite_dir: Path
    experiment_id: str
    suite_name: str
    num_warmup: int
    num_trials: int
    benchmark_invocations: int = 1
    min_repeat_ms: int | None = None
    bad_baseline: bool = False
    bedrock_client: Any = None


def default_experiment_id(prefix: str = "suite") -> str:
    """Return a stable human-readable id for a new suite run."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{timestamp}"


def run_benchmark_suite(
    *,
    tasks: Iterable[MatmulTaskCase | Conv2DTaskCase],
    methods: Iterable[MethodCase],
    config: SuiteRunConfig,
) -> list[dict[str, Any]]:
    """Run all task/method pairs and write a compact suite manifest/summary."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.suite_dir.mkdir(parents=True, exist_ok=True)

    task_list = list(tasks)
    method_list = list(methods)
    _write_json(
        config.suite_dir / "manifest.json",
        {
            "experiment_id": config.experiment_id,
            "suite_name": config.suite_name,
            "output_dir": str(config.output_dir),
            "suite_dir": str(config.suite_dir),
            "num_warmup": config.num_warmup,
            "num_trials": config.num_trials,
            "benchmark_invocations": config.benchmark_invocations,
            "min_repeat_ms": config.min_repeat_ms,
            "tasks": [{**asdict(task), "case_id": task.case_id} for task in task_list],
            "methods": [_method_manifest(method) for method in method_list],
        },
    )

    results: list[dict[str, Any]] = []
    for task in task_list:
        for method_case in method_list:
            result = run_suite_case(
                task=task,
                method_case=method_case,
                config=config,
            )
            results.append(result)

    _write_summary(config.suite_dir / "summary.csv", results)
    _write_json(config.suite_dir / "summary.json", results)
    return results


def run_suite_case(
    *,
    task: MatmulTaskCase | Conv2DTaskCase,
    method_case: MethodCase,
    config: SuiteRunConfig,
) -> dict[str, Any]:
    """Run one experiment method against one task case."""
    from src.eval.method import (
        MethodRunConfig,
        default_level2_search_space_path,
        run_experiment_method,
    )

    generated_search_space_path = method_case.generated_search_space_path
    workload_name = _task_workload_name(task)
    if method_case.name == "level2-candidate" and generated_search_space_path is None:
        generated_search_space_path = default_level2_search_space_path(
            task.target,
            workload_name=workload_name,
        )

    return run_experiment_method(
        method=method_case.name,
        config=MethodRunConfig(
            M=getattr(task, "M", 256),
            N=getattr(task, "N", 256),
            K=getattr(task, "K", 256),
            target_name=task.target,
            output_dir=config.output_dir,
            workload_name=workload_name,
            workload_params=_task_workload_params(task),
            num_warmup=config.num_warmup,
            num_trials=config.num_trials,
            benchmark_invocations=config.benchmark_invocations,
            min_repeat_ms=config.min_repeat_ms,
            experiment_id=config.experiment_id,
            suite_name=config.suite_name,
            run_id=f"{task.case_id}_{method_case.name}",
            run_kind="benchmark_suite",
            benchmark_group="final_benchmark",
            bad_baseline=config.bad_baseline,
            generated_schedule_path=method_case.generated_schedule_path,
            generated_search_space_path=generated_search_space_path,
            max_trials_global=method_case.max_trials_global,
            max_trials_per_task=method_case.max_trials_per_task,
            num_trials_per_iter=method_case.num_trials_per_iter,
            cost_model=method_case.cost_model,
            task_scheduler=method_case.task_scheduler,
            seed=method_case.seed,
            num_tuning_cores=method_case.num_tuning_cores,
            post_optimization=method_case.post_optimization,
            level1_seed_candidate_path=(
                method_case.generated_schedule_path or Path("generated/schedules/identity.py")
            ),
            level2_seed_candidate_path=(
                method_case.generated_search_space_path
                or default_level2_search_space_path(
                    task.target,
                    workload_name=workload_name,
                )
            ),
            generations=method_case.generations,
            population_size=method_case.population_size,
            survivors=method_case.survivors,
            search_num_warmup=method_case.search_num_warmup,
            search_num_trials=method_case.search_num_trials,
            search_benchmark_invocations=method_case.search_benchmark_invocations,
            search_min_repeat_ms=method_case.search_min_repeat_ms,
            search_max_trials_global=method_case.search_max_trials_global,
            search_num_trials_per_iter=method_case.search_num_trials_per_iter,
            level2_final_evaluation_policy=method_case.level2_final_evaluation_policy,
            include_evaluator_feedback=method_case.include_evaluator_feedback,
            enable_rejection_cascade=method_case.enable_rejection_cascade,
            enable_elite_carry_forward=method_case.enable_elite_carry_forward,
            enable_diverse_inspiration=method_case.enable_diverse_inspiration,
            dry_run=method_case.dry_run,
            bedrock_client=config.bedrock_client,
        ),
    )


def _method_manifest(method: MethodCase) -> dict[str, Any]:
    data = asdict(method)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def _task_workload_name(task: MatmulTaskCase | Conv2DTaskCase) -> str:
    if isinstance(task, Conv2DTaskCase):
        return "conv2d"
    return "matmul"


def _task_workload_params(task: MatmulTaskCase | Conv2DTaskCase) -> dict[str, int]:
    if isinstance(task, Conv2DTaskCase):
        return task.workload_params
    return {}


def _write_summary(path: Path, results: list[dict[str, Any]]) -> None:
    if not results:
        return
    fields = [
        "experiment_id",
        "suite_name",
        "run_id",
        "experiment_method",
        "strategy",
        "level",
        "M",
        "N",
        "K",
        "batch",
        "in_channels",
        "height",
        "width",
        "out_channels",
        "kernel_h",
        "kernel_w",
        "stride",
        "padding",
        "out_height",
        "out_width",
        "target",
        "compile_passed",
        "correctness_passed",
        "latency_ms_mean",
        "latency_ms_std",
        "selection_role",
        "final_evaluation_policy",
        "benchmark_invocations",
        "min_repeat_ms",
        "tuning_time_sec",
        "evolution_time_sec",
        "evolution_run_dir",
        "evolution_history_path",
        "evolution_best_path",
        "best_candidate_path",
        "best_fitness_score",
        "json_result",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field) for field in fields})


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
