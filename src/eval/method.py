"""Experiment methods built on the shared workload evaluation pipeline."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from src.eval.experiment import run_workload_experiment
from src.evolution.bedrock_client import BedrockClient
from src.evolution.loop import run_schedule_evolution
from src.evolution.space_loop import run_search_space_evolution
from src.kernels.workloads import (
    CONV2D_WORKLOAD,
    MATMUL_WORKLOAD,
    Workload,
    make_workload,
)
from src.strategies import StrategyBuildConfig, get_strategy


METHOD_NAMES = (
    "fixed",
    "metaschedule",
    "level1-candidate",
    "level2-candidate",
    "level1-search",
    "level2-search",
)

STRATEGY_METHODS = {
    "fixed": "fixed",
    "metaschedule": "metaschedule",
    "level1-candidate": "generated-schedule",
    "level2-candidate": "generated-search-space",
}

LEVEL2_FINAL_EVALUATION_POLICIES = (
    "fresh-retune",
    "exact-winner",
)

DEFAULT_LEVEL2_SEARCH_SPACE_PATHS = {
    MATMUL_WORKLOAD: {
        "llvm": Path("generated/search_spaces/basic_matmul.py"),
        "cuda": Path("generated/search_spaces/cuda_matmul.py"),
    },
    CONV2D_WORKLOAD: {
        "llvm": Path("generated/search_spaces/basic_conv2d.py"),
        "cuda": Path("generated/search_spaces/cuda_conv2d.py"),
    },
}

IDENTITY_SCHEDULE_PATH = Path("generated/schedules/identity.py")

DEFAULT_LEVEL1_SCHEDULE_PATHS = {
    MATMUL_WORKLOAD: {
        "cuda": Path("generated/schedules/cuda_matmul.py"),
    },
    CONV2D_WORKLOAD: {
        "cuda": Path("generated/schedules/cuda_conv2d.py"),
    },
}


@dataclass(frozen=True)
class MethodRunConfig:
    """Configuration for one method on one workload task.

    M/N/K remain compatibility fields for matmul runs and legacy evolution prompts.
    Non-matmul workloads should pass their real shape through workload_params.
    """

    M: int
    N: int
    K: int
    target_name: str
    output_dir: Path
    workload_name: str = "matmul"
    workload_params: dict[str, Any] = field(default_factory=dict)
    num_warmup: int = 3
    num_trials: int = 10
    benchmark_invocations: int = 1
    min_repeat_ms: int | None = None
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
    if method in STRATEGY_METHODS:
        return _run_strategy_method(method=method, config=config)
    if method == "level1-search":
        return _run_level1_search(config=config)
    if method == "level2-search":
        return _run_level2_search(config=config)
    raise ValueError(f"Unknown method '{method}'. Available methods: {', '.join(METHOD_NAMES)}")


def _run_strategy_method(*, method: str, config: MethodRunConfig) -> dict[str, Any]:
    strategy = get_strategy(STRATEGY_METHODS[method])
    generated_schedule_path = config.generated_schedule_path
    if method == "level1-candidate":
        generated_schedule_path = _level1_schedule_path(config)
    generated_search_space_path = config.generated_search_space_path
    if method == "level2-candidate" and generated_search_space_path is None:
        generated_search_space_path = default_level2_search_space_path(
            config.target_name,
            workload_name=config.workload_name,
        )
    workload = _workload(config)
    strategy_config = _strategy_build_config(
        config=config,
        workload=workload,
        work_dir=config.tuning_work_dir,
        generated_schedule_path=generated_schedule_path,
        generated_search_space_path=generated_search_space_path,
    )
    return run_workload_experiment(
        workload=workload,
        strategy=strategy,
        strategy_config=strategy_config,
        target_name=config.target_name,
        num_warmup=config.num_warmup,
        num_trials=config.num_trials,
        benchmark_invocations=config.benchmark_invocations,
        min_repeat_ms=config.min_repeat_ms,
        output_dir=config.output_dir,
        bad_baseline=config.bad_baseline,
        extra_metadata=_base_metadata(
            method=method,
            config=config,
            selection_role=config.selection_role or _selection_role(strategy.level),
        ),
    )


def _strategy_build_config(
    *,
    config: MethodRunConfig,
    workload: Workload,
    work_dir: Path | None,
    generated_schedule_path: Path | None,
    generated_search_space_path: Path | None,
) -> StrategyBuildConfig:
    """Translate method-level knobs into the strategy-level build config."""
    return StrategyBuildConfig(
        work_dir=work_dir,
        max_trials_global=config.max_trials_global,
        max_trials_per_task=config.max_trials_per_task,
        num_trials_per_iter=config.num_trials_per_iter,
        seed=config.seed,
        num_tuning_cores=config.num_tuning_cores,
        cost_model=config.cost_model,
        task_scheduler=config.task_scheduler,
        post_optimization=config.post_optimization,
        generated_schedule_path=generated_schedule_path,
        generated_search_space_path=generated_search_space_path,
        workload_name=workload.name,
    )


def _run_level1_search(*, config: MethodRunConfig) -> dict[str, Any]:
    run_dir = _evolution_run_dir(config, "level1-search")
    workload = _workload(config)

    start = perf_counter()
    history = run_schedule_evolution(
        seed_candidate_path=_level1_seed_candidate_path(config),
        run_dir=run_dir,
        output_dir=config.output_dir,
        generations=config.generations,
        population_size=config.population_size,
        survivors=config.survivors,
        target_name=config.target_name,
        M=config.M,
        N=config.N,
        K=config.K,
        workload=workload,
        num_warmup=_search_value(config.search_num_warmup, config.num_warmup),
        num_trials=_search_value(config.search_num_trials, config.num_trials),
        benchmark_invocations=_search_benchmark_invocations(config),
        min_repeat_ms=_search_min_repeat_ms(config),
        bedrock_client=config.bedrock_client,
        dry_run=config.dry_run,
        experiment_id=config.experiment_id,
        suite_name=config.suite_name,
        benchmark_group="evolution_search",
        experiment_method="level1-search",
        include_evaluator_feedback=config.include_evaluator_feedback,
        enable_rejection_cascade=config.enable_rejection_cascade,
        enable_elite_carry_forward=config.enable_elite_carry_forward,
        enable_diverse_inspiration=config.enable_diverse_inspiration,
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
    _validate_level2_final_evaluation_policy(config.level2_final_evaluation_policy)
    run_dir = _evolution_run_dir(config, "level2-search")
    workload = _workload(config)

    start = perf_counter()
    history = run_search_space_evolution(
        seed_candidate_path=(
            config.level2_seed_candidate_path
            or default_level2_search_space_path(
                config.target_name,
                workload_name=config.workload_name,
            )
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
        workload=workload,
        num_warmup=_search_value(config.search_num_warmup, config.num_warmup),
        num_trials=_search_value(config.search_num_trials, config.num_trials),
        benchmark_invocations=_search_benchmark_invocations(config),
        min_repeat_ms=_search_min_repeat_ms(config),
        max_trials_global=_search_value(
            config.search_max_trials_global,
            config.max_trials_global,
        ),
        num_trials_per_iter=(
            _search_value(config.search_num_trials_per_iter, config.num_trials_per_iter)
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
        include_evaluator_feedback=config.include_evaluator_feedback,
        enable_rejection_cascade=config.enable_rejection_cascade,
        enable_elite_carry_forward=config.enable_elite_carry_forward,
        enable_diverse_inspiration=config.enable_diverse_inspiration,
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
    workload = _workload(config)
    strategy_config = _strategy_build_config(
        config=config,
        workload=workload,
        work_dir=None,
        generated_schedule_path=config.generated_schedule_path,
        generated_search_space_path=config.generated_search_space_path,
    )
    selection_role = "best_of_search"
    final_evaluation_metadata: dict[str, Any] = {}
    if method == "level2-search":
        strategy_name, strategy_config, selection_role, final_evaluation_metadata = (
            _level2_final_evaluation(
                config=config,
                best=best,
                fresh_retune_config=strategy_config,
            )
        )
    strategy = get_strategy(strategy_name)
    result = run_workload_experiment(
        workload=workload,
        strategy=strategy,
        strategy_config=strategy_config,
        target_name=config.target_name,
        num_warmup=config.num_warmup,
        num_trials=config.num_trials,
        benchmark_invocations=config.benchmark_invocations,
        min_repeat_ms=config.min_repeat_ms,
        output_dir=config.output_dir,
        bad_baseline=config.bad_baseline,
        extra_metadata=(
            _base_metadata(
                method=method,
                config=config,
                selection_role=selection_role,
            )
            | _evolution_metadata(
                run_dir=run_dir,
                history=history,
                best=best,
                evolution_time_sec=evolution_time_sec,
            )
            | final_evaluation_metadata
        ),
    )
    result["evolution_history"] = str(run_dir / "history.json")
    result["evolution_best"] = str(run_dir / "best.json")
    return result


def _level2_final_evaluation(
    *,
    config: MethodRunConfig,
    best: dict[str, Any],
    fresh_retune_config: StrategyBuildConfig,
) -> tuple[str, StrategyBuildConfig, str, dict[str, Any]]:
    policy = config.level2_final_evaluation_policy
    _validate_level2_final_evaluation_policy(policy)
    best_result = best["result"]
    metadata = {
        "final_evaluation_policy": policy,
        "search_winner_scheduled_module_path": best_result.get("scheduled_module_path"),
        "search_winner_scheduled_module_json_path": best_result.get(
            "scheduled_module_json_path"
        ),
        "search_winner_metaschedule_work_dir": best_result.get("metaschedule_work_dir"),
        "search_winner_metaschedule_database_tuning_record": best_result.get(
            "metaschedule_database_tuning_record"
        ),
        "search_winner_metaschedule_database_workload": best_result.get(
            "metaschedule_database_workload"
        ),
        "search_winner_used_fallback_schedule": best_result.get("used_fallback_schedule"),
    }
    if policy == "fresh-retune":
        return (
            "generated-search-space",
            fresh_retune_config,
            "best_of_search_fresh_retune",
            metadata
            | {
                "final_evaluation_semantics": "retune_winning_search_space_generator",
                "exact_schedule_reused": False,
            },
        )

    return (
        "saved-scheduled-module",
        StrategyBuildConfig(
            saved_scheduled_module_path=_optional_path(
                best_result.get("scheduled_module_path")
            ),
            saved_scheduled_module_json_path=_optional_path(
                best_result.get("scheduled_module_json_path")
            ),
            workload_name=config.workload_name,
        ),
        "best_of_search_exact_winner",
        metadata
        | {
            "final_evaluation_semantics": "reuse_search_time_scheduled_module",
            "exact_schedule_reused": True,
        },
    )


def _validate_level2_final_evaluation_policy(policy: str) -> None:
    if policy not in LEVEL2_FINAL_EVALUATION_POLICIES:
        choices = ", ".join(LEVEL2_FINAL_EVALUATION_POLICIES)
        raise ValueError(
            f"Unknown Level 2 final-evaluation policy {policy!r}. Available: {choices}"
        )


def _optional_path(value: Any) -> Path | None:
    if value is None or value == "":
        return None
    return Path(str(value))


def default_level2_search_space_path(
    target_name: str,
    *,
    workload_name: str = "matmul",
) -> Path:
    """Return the built-in Level-2 seed for one target."""
    try:
        return DEFAULT_LEVEL2_SEARCH_SPACE_PATHS[workload_name][target_name]
    except KeyError as err:
        workloads = ", ".join(DEFAULT_LEVEL2_SEARCH_SPACE_PATHS)
        raise ValueError(
            f"Unsupported Level-2 seed for workload={workload_name!r}, "
            f"target={target_name!r}. Available workloads: {workloads}"
        ) from err


def _level1_seed_candidate_path(config: MethodRunConfig) -> Path:
    if config.level1_seed_candidate_path != IDENTITY_SCHEDULE_PATH:
        return config.level1_seed_candidate_path
    return _default_level1_schedule_path(config) or config.level1_seed_candidate_path


def _level1_schedule_path(config: MethodRunConfig) -> Path | None:
    if config.generated_schedule_path not in (None, IDENTITY_SCHEDULE_PATH):
        return config.generated_schedule_path
    return _default_level1_schedule_path(config) or config.generated_schedule_path


def _default_level1_schedule_path(config: MethodRunConfig) -> Path | None:
    return DEFAULT_LEVEL1_SCHEDULE_PATHS.get(config.workload_name, {}).get(
        config.target_name
    )


def _workload(config: MethodRunConfig) -> Workload:
    return make_workload(
        workload_name=config.workload_name,
        M=config.M,
        N=config.N,
        K=config.K,
        workload_params=config.workload_params,
    )


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
        "evaluator_feedback_enabled": config.include_evaluator_feedback,
        "rejection_cascade_enabled": config.enable_rejection_cascade,
        "elite_carry_forward_enabled": config.enable_elite_carry_forward,
        "diverse_inspiration_enabled": config.enable_diverse_inspiration,
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
        return _validate_explicit_evolution_run_dir(config.evolution_run_dir)
    experiment_id = _safe_path_component(config.experiment_id or default_experiment_id(method))
    run_id = config.run_id or default_experiment_id(method)
    run_name = f"{_safe_path_component(run_id)}__{_evolution_config_slug(config)}"
    base_dir = config.output_dir / "evolution_runs" / method / experiment_id / run_name
    return _reserve_unique_run_dir(base_dir)


def _validate_explicit_evolution_run_dir(path: Path) -> Path:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"--evolution-run-dir points to a non-empty directory: {path}. "
            "Choose a fresh directory so this run does not mix with prior artifacts."
        )
    return path


def _reserve_unique_run_dir(base_dir: Path) -> Path:
    try:
        base_dir.mkdir(parents=True, exist_ok=False)
        return base_dir
    except FileExistsError:
        pass
    for index in range(1, 1000):
        candidate = base_dir.with_name(f"{base_dir.name}__{index:03d}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            pass
    raise RuntimeError(f"Could not find a fresh evolution run directory near {base_dir}")


def _safe_path_component(value: Any) -> str:
    safe = re.sub(r"[^A-Za-z0-9._=-]+", "_", str(value)).strip("._")
    return safe or "run"


def _evolution_config_slug(config: MethodRunConfig) -> str:
    max_trials = _search_value(config.search_max_trials_global, config.max_trials_global)
    trials_per_iter = _search_value(
        config.search_num_trials_per_iter,
        config.num_trials_per_iter,
    )
    parts = [
        f"g{config.generations}",
        f"p{config.population_size}",
        f"s{config.survivors}",
        f"mg{max_trials}",
        f"it{trials_per_iter}",
        f"cm{config.cost_model}",
        f"ts{config.task_scheduler}",
        f"seed{config.seed}",
        f"fb{int(config.include_evaluator_feedback)}",
        f"rc{int(config.enable_rejection_cascade)}",
        f"elite{int(config.enable_elite_carry_forward)}",
        f"div{int(config.enable_diverse_inspiration)}",
        f"policy{config.level2_final_evaluation_policy}",
    ]
    model_id = _bedrock_model_id(config)
    if model_id:
        digest = hashlib.sha1(model_id.encode("utf-8")).hexdigest()[:8]
        parts.append(f"model{_short_path_component(model_id)}-{digest}")
    if config.dry_run:
        parts.append("dry")
    return _safe_path_component("__".join(parts))


def _bedrock_model_id(config: MethodRunConfig) -> str | None:
    client_config = getattr(config.bedrock_client, "config", None)
    model_id = getattr(client_config, "model_id", None)
    if isinstance(model_id, str) and model_id:
        return model_id
    return None


def _short_path_component(value: str, max_length: int = 48) -> str:
    token = value.rsplit("/", 1)[-1].rsplit(":", 1)[0]
    safe = _safe_path_component(token)
    if len(safe) <= max_length:
        return safe
    return safe[:max_length].rstrip("._=-") or "value"


def _best_or_raise(history: list[dict[str, Any]]) -> dict[str, Any]:
    if not history:
        raise RuntimeError("Search produced no evaluated candidates.")
    return history[0]


def _search_value(override: Any | None, fallback: Any) -> Any:
    if override is not None:
        return override
    return fallback


def _search_benchmark_invocations(config: MethodRunConfig) -> int:
    if config.search_benchmark_invocations is not None:
        return config.search_benchmark_invocations
    return config.benchmark_invocations


def _search_min_repeat_ms(config: MethodRunConfig) -> int | None:
    if config.search_min_repeat_ms is not None:
        return config.search_min_repeat_ms
    return config.min_repeat_ms
