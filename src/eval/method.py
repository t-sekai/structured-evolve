"""Experiment methods built on the shared matmul evaluation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from src.eval.experiment import run_matmul_experiment
from src.evolution.bedrock_client import BedrockClient
from src.evolution.loop import run_schedule_evolution
from src.evolution.space_loop import run_search_space_evolution
from src.strategies import StrategyBuildConfig, get_strategy


METHOD_NAMES = (
    "fixed",
    "metaschedule",
    "level1-candidate",
    "level2-candidate",
    "level1-search",
    "level2-search",
)

METHOD_TO_STRATEGY = {
    "fixed": "fixed",
    "metaschedule": "metaschedule",
    "level1-candidate": "generated-schedule",
    "level2-candidate": "generated-search-space",
}


@dataclass(frozen=True)
class MethodRunConfig:
    """Configuration for one method on one matmul task."""

    M: int
    N: int
    K: int
    target_name: str
    output_dir: Path
    num_warmup: int = 3
    num_trials: int = 10
    experiment_id: str | None = None
    suite_name: str | None = None
    run_id: str | None = None
    run_kind: str = "single_experiment"
    benchmark_group: str = "final_benchmark"
    selection_role: str | None = None
    bad_baseline: bool = False
    tuning_work_dir: Path | None = None
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
    evolution_run_dir: Path | None = None
    level1_seed_candidate_path: Path = Path("generated/schedules/identity.py")
    level2_seed_candidate_path: Path | None = None
    generations: int = 1
    population_size: int = 2
    survivors: int = 1
    search_num_warmup: int | None = None
    search_num_trials: int | None = None
    search_max_trials_global: int | None = None
    search_num_trials_per_iter: int | None = None
    dry_run: bool = True
    bedrock_client: BedrockClient | None = None


def default_experiment_id(prefix: str = "experiment") -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{timestamp}"


def run_experiment_method(
    *,
    method: str,
    config: MethodRunConfig,
) -> dict[str, Any]:
    """Run one first-class experiment method and return its final result row."""
    if method in METHOD_TO_STRATEGY:
        return _run_existing_candidate(method=method, config=config)
    if method == "level1-search":
        return _run_level1_search(config=config)
    if method == "level2-search":
        return _run_level2_search(config=config)
    raise ValueError(f"Unknown method '{method}'. Available methods: {', '.join(METHOD_NAMES)}")


def _run_existing_candidate(*, method: str, config: MethodRunConfig) -> dict[str, Any]:
    strategy = get_strategy(METHOD_TO_STRATEGY[method])
    generated_search_space_path = config.generated_search_space_path
    if method == "level2-candidate" and generated_search_space_path is None:
        generated_search_space_path = default_level2_search_space_path(config.target_name)
    strategy_config = StrategyBuildConfig(
        work_dir=config.tuning_work_dir,
        max_trials_global=config.max_trials_global,
        max_trials_per_task=config.max_trials_per_task,
        num_trials_per_iter=config.num_trials_per_iter,
        seed=config.seed,
        num_tuning_cores=config.num_tuning_cores,
        cost_model=config.cost_model,
        task_scheduler=config.task_scheduler,
        post_optimization=config.post_optimization,
        generated_schedule_path=config.generated_schedule_path,
        generated_search_space_path=generated_search_space_path,
    )
    return run_matmul_experiment(
        strategy=strategy,
        strategy_config=strategy_config,
        M=config.M,
        N=config.N,
        K=config.K,
        target_name=config.target_name,
        num_warmup=config.num_warmup,
        num_trials=config.num_trials,
        output_dir=config.output_dir,
        bad_baseline=config.bad_baseline,
        extra_metadata=_base_metadata(
            method=method,
            config=config,
            selection_role=config.selection_role or _selection_role(strategy.level),
        ),
    )


def _run_level1_search(*, config: MethodRunConfig) -> dict[str, Any]:
    run_dir = _evolution_run_dir(config, "level1-search")
    search_config = _search_config(config)

    start = perf_counter()
    history = run_schedule_evolution(
        seed_candidate_path=config.level1_seed_candidate_path,
        run_dir=run_dir,
        output_dir=config.output_dir,
        generations=config.generations,
        population_size=config.population_size,
        survivors=config.survivors,
        target_name=config.target_name,
        M=config.M,
        N=config.N,
        K=config.K,
        num_warmup=search_config.search_num_warmup or config.num_warmup,
        num_trials=search_config.search_num_trials or config.num_trials,
        bedrock_client=config.bedrock_client,
        dry_run=config.dry_run,
        experiment_id=config.experiment_id,
        suite_name=config.suite_name,
        benchmark_group="evolution_search",
        experiment_method="level1-search",
    )
    evolution_time_sec = perf_counter() - start
    best = _best_or_raise(history)
    best_path = Path(best["candidate"]["path"])

    return _run_evolved_best(
        config=replace(config, generated_schedule_path=best_path),
        method="level1-search",
        strategy_name="generated-schedule",
        run_dir=run_dir,
        history=history,
        best=best,
        evolution_time_sec=evolution_time_sec,
    )


def _run_level2_search(*, config: MethodRunConfig) -> dict[str, Any]:
    run_dir = _evolution_run_dir(config, "level2-search")
    search_config = _search_config(config)

    start = perf_counter()
    history = run_search_space_evolution(
        seed_candidate_path=(
            config.level2_seed_candidate_path
            or default_level2_search_space_path(config.target_name)
        ),
        run_dir=run_dir,
        output_dir=config.output_dir,
        generations=config.generations,
        population_size=config.population_size,
        survivors=config.survivors,
        target_name=config.target_name,
        M=config.M,
        N=config.N,
        K=config.K,
        num_warmup=search_config.search_num_warmup or config.num_warmup,
        num_trials=search_config.search_num_trials or config.num_trials,
        max_trials_global=search_config.search_max_trials_global or config.max_trials_global,
        num_trials_per_iter=(
            search_config.search_num_trials_per_iter or config.num_trials_per_iter
        ),
        cost_model=config.cost_model,
        task_scheduler=config.task_scheduler,
        seed=config.seed,
        num_tuning_cores=config.num_tuning_cores,
        bedrock_client=config.bedrock_client,
        dry_run=config.dry_run,
        experiment_id=config.experiment_id,
        suite_name=config.suite_name,
        benchmark_group="evolution_search",
        experiment_method="level2-search",
    )
    evolution_time_sec = perf_counter() - start
    best = _best_or_raise(history)
    best_path = Path(best["candidate"]["path"])

    return _run_evolved_best(
        config=replace(config, generated_search_space_path=best_path),
        method="level2-search",
        strategy_name="generated-search-space",
        run_dir=run_dir,
        history=history,
        best=best,
        evolution_time_sec=evolution_time_sec,
    )


def _run_evolved_best(
    *,
    config: MethodRunConfig,
    method: str,
    strategy_name: str,
    run_dir: Path,
    history: list[dict[str, Any]],
    best: dict[str, Any],
    evolution_time_sec: float,
) -> dict[str, Any]:
    strategy = get_strategy(strategy_name)
    strategy_config = StrategyBuildConfig(
        max_trials_global=config.max_trials_global,
        max_trials_per_task=config.max_trials_per_task,
        num_trials_per_iter=config.num_trials_per_iter,
        seed=config.seed,
        num_tuning_cores=config.num_tuning_cores,
        cost_model=config.cost_model,
        task_scheduler=config.task_scheduler,
        post_optimization=config.post_optimization,
        generated_schedule_path=config.generated_schedule_path,
        generated_search_space_path=config.generated_search_space_path,
    )
    result = run_matmul_experiment(
        strategy=strategy,
        strategy_config=strategy_config,
        M=config.M,
        N=config.N,
        K=config.K,
        target_name=config.target_name,
        num_warmup=config.num_warmup,
        num_trials=config.num_trials,
        output_dir=config.output_dir,
        bad_baseline=config.bad_baseline,
        extra_metadata=(
            _base_metadata(
                method=method,
                config=config,
                selection_role="best_of_search",
            )
            | _evolution_metadata(
                run_dir=run_dir,
                history=history,
                best=best,
                evolution_time_sec=evolution_time_sec,
            )
        ),
    )
    result["evolution_history"] = str(run_dir / "history.json")
    result["evolution_best"] = str(run_dir / "best.json")
    return result


def default_level2_search_space_path(target_name: str) -> Path:
    """Return the built-in Level-2 seed for one target."""
    if target_name == "llvm":
        return Path("generated/search_spaces/basic_matmul.py")
    if target_name == "cuda":
        return Path("generated/search_spaces/cuda_matmul.py")
    raise ValueError(f"Unsupported target: {target_name}")


def _base_metadata(
    *,
    method: str,
    config: MethodRunConfig,
    selection_role: str,
) -> dict[str, Any]:
    return {
        "experiment_id": config.experiment_id,
        "suite_name": config.suite_name,
        "run_id": config.run_id,
        "run_kind": config.run_kind,
        "benchmark_group": config.benchmark_group,
        "experiment_method": method,
        "selection_role": selection_role,
    }


def _evolution_metadata(
    *,
    run_dir: Path,
    history: list[dict[str, Any]],
    best: dict[str, Any],
    evolution_time_sec: float,
) -> dict[str, Any]:
    compile_failures = sum(1 for row in history if not row["result"].get("compile_passed"))
    correct_candidates = sum(1 for row in history if row["result"].get("correctness_passed"))
    return {
        "evolution_run_dir": run_dir,
        "evolution_history_path": run_dir / "history.json",
        "evolution_best_path": run_dir / "best.json",
        "evolution_time_sec": evolution_time_sec,
        "num_candidates_evaluated": len(history),
        "num_compile_failures": compile_failures,
        "num_correct_candidates": correct_candidates,
        "best_candidate_id": best["candidate"]["candidate_id"],
        "best_candidate_path": best["candidate"]["path"],
        "best_fitness_score": best["fitness"]["score"],
        "best_fitness_reason": best["fitness"]["reason"],
    }


def _selection_role(level: str) -> str:
    if level.startswith("baseline"):
        return "baseline"
    return "candidate"


def _evolution_run_dir(config: MethodRunConfig, method: str) -> Path:
    if config.evolution_run_dir is not None:
        return config.evolution_run_dir
    run_id = config.run_id or default_experiment_id(method)
    return config.output_dir / "evolution_runs" / method / run_id


def _best_or_raise(history: list[dict[str, Any]]) -> dict[str, Any]:
    if not history:
        raise RuntimeError("Search produced no evaluated candidates.")
    return history[0]


def _search_config(config: MethodRunConfig) -> MethodRunConfig:
    return config
