"""Level-2 evolution loop for generated MetaSchedule search spaces."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.eval.experiment import record_rejected_matmul_experiment, run_matmul_experiment
from src.evolution.bedrock_client import BedrockClient
from src.evolution.candidate import Candidate
from src.evolution.cascade import preflight_generated_search_space
from src.evolution.fitness import FitnessResult, score_result
from src.evolution.prompts import (
    SYSTEM_PROMPT,
    evaluator_feedback,
    search_space_mutation_prompt,
    strip_code_fences,
)
from src.evolution.selection import annotate_source, plan_metadata, plan_next_generation
from src.strategies import StrategyBuildConfig, get_strategy


def run_search_space_evolution(
    *,
    seed_candidate_path: Path,
    run_dir: Path,
    output_dir: Path,
    generations: int,
    population_size: int,
    survivors: int,
    target_name: str,
    M: int,
    N: int,
    K: int,
    num_warmup: int,
    num_trials: int,
    benchmark_invocations: int,
    min_repeat_ms: int | None,
    max_trials_global: int,
    num_trials_per_iter: int,
    cost_model: str,
    task_scheduler: str,
    seed: int | None,
    num_tuning_cores: int | str,
    bedrock_client: BedrockClient | None,
    dry_run: bool,
    experiment_id: str | None = None,
    suite_name: str | None = None,
    benchmark_group: str | None = None,
    experiment_method: str | None = None,
    include_evaluator_feedback: bool = True,
    enable_rejection_cascade: bool = True,
    enable_elite_carry_forward: bool = False,
    enable_diverse_inspiration: bool = False,
) -> list[dict[str, Any]]:
    """Run an OpenEvolve-style loop over MetaSchedule search-space files."""
    if generations < 0:
        raise ValueError(f"generations must be non-negative, got {generations}")
    if population_size <= 0:
        raise ValueError(f"population_size must be positive, got {population_size}")
    if survivors <= 0:
        raise ValueError(f"survivors must be positive, got {survivors}")
    if survivors > population_size:
        raise ValueError("survivors cannot exceed population_size")
    if dry_run and bedrock_client is not None:
        raise ValueError("dry_run and bedrock_client are mutually exclusive")
    if not dry_run and bedrock_client is None:
        raise ValueError("Provide a BedrockClient or set dry_run=True")

    run_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    strategy = get_strategy("generated-search-space")
    run_id = run_dir.name

    _write_json(
        run_dir / "manifest.json",
        {
            "experiment_id": experiment_id,
            "suite_name": suite_name,
            "run_id": run_id,
            "run_kind": "evolution",
            "experiment_method": experiment_method,
            "evolution_kind": strategy.level,
            "strategy": strategy.name,
            "seed_candidate_path": str(seed_candidate_path),
            "run_dir": str(run_dir),
            "output_dir": str(output_dir),
            "generations": generations,
            "population_size": population_size,
            "survivors": survivors,
            "target": target_name,
            "M": M,
            "N": N,
            "K": K,
            "num_warmup": num_warmup,
            "num_trials": num_trials,
            "benchmark_invocations": benchmark_invocations,
            "min_repeat_ms": min_repeat_ms,
            "max_trials_global": max_trials_global,
            "num_trials_per_iter": num_trials_per_iter,
            "cost_model": cost_model,
            "task_scheduler": task_scheduler,
            "seed": seed,
            "num_tuning_cores": num_tuning_cores,
            "dry_run": dry_run,
            **_ablation_metadata(
                include_evaluator_feedback=include_evaluator_feedback,
                enable_rejection_cascade=enable_rejection_cascade,
                enable_elite_carry_forward=enable_elite_carry_forward,
                enable_diverse_inspiration=enable_diverse_inspiration,
            ),
        },
    )

    seed_dir = run_dir / "gen_000"
    seed_dir.mkdir(parents=True, exist_ok=True)
    seed_path = seed_dir / "candidate_000_seed.py"
    shutil.copyfile(seed_candidate_path, seed_path)
    active = [
        Candidate(candidate_id="g000_c000_seed", generation=0, path=seed_path, origin="seed")
    ]

    history: list[dict[str, Any]] = []
    for generation in range(generations + 1):
        generation_dir = run_dir / f"gen_{generation:03d}"
        generation_dir.mkdir(parents=True, exist_ok=True)
        evaluated = [
            _evaluate_candidate(
                candidate=candidate,
                strategy=strategy,
                generation_dir=generation_dir,
                target_name=target_name,
                M=M,
                N=N,
                K=K,
                num_warmup=num_warmup,
                num_trials=num_trials,
                benchmark_invocations=benchmark_invocations,
                min_repeat_ms=min_repeat_ms,
                max_trials_global=max_trials_global,
                num_trials_per_iter=num_trials_per_iter,
                cost_model=cost_model,
                task_scheduler=task_scheduler,
                seed=seed,
                num_tuning_cores=num_tuning_cores,
                output_dir=output_dir,
                run_id=run_id,
                run_dir=run_dir,
                experiment_id=experiment_id,
                suite_name=suite_name,
                benchmark_group=benchmark_group,
                experiment_method=experiment_method,
                enable_rejection_cascade=enable_rejection_cascade,
                include_evaluator_feedback=include_evaluator_feedback,
                enable_elite_carry_forward=enable_elite_carry_forward,
                enable_diverse_inspiration=enable_diverse_inspiration,
            )
            for candidate in active
        ]
        evaluated.sort(key=lambda row: row["fitness"]["score"], reverse=True)
        for rank, row in enumerate(evaluated, start=1):
            row["generation_rank"] = rank
        history.extend(evaluated)
        _write_json(generation_dir / "ranking.json", evaluated)

        if generation == generations:
            break

        active = _make_next_generation(
            evaluated=evaluated,
            survivors=survivors,
            generation=generation + 1,
            generation_dir=run_dir / f"gen_{generation + 1:03d}",
            population_size=population_size,
            target_name=target_name,
            M=M,
            N=N,
            K=K,
            bedrock_client=bedrock_client,
            dry_run=dry_run,
            include_evaluator_feedback=include_evaluator_feedback,
            enable_rejection_cascade=enable_rejection_cascade,
            enable_elite_carry_forward=enable_elite_carry_forward,
            enable_diverse_inspiration=enable_diverse_inspiration,
        )

    history.sort(key=lambda row: row["fitness"]["score"], reverse=True)
    _write_json(run_dir / "history.json", history)
    _write_json(run_dir / "best.json", history[0] if history else {})
    return history


def _evaluate_candidate(
    *,
    candidate: Candidate,
    strategy,
    generation_dir: Path,
    target_name: str,
    M: int,
    N: int,
    K: int,
    num_warmup: int,
    num_trials: int,
    benchmark_invocations: int,
    min_repeat_ms: int | None,
    max_trials_global: int,
    num_trials_per_iter: int,
    cost_model: str,
    task_scheduler: str,
    seed: int | None,
    num_tuning_cores: int | str,
    output_dir: Path,
    run_id: str,
    run_dir: Path,
    experiment_id: str | None,
    suite_name: str | None,
    benchmark_group: str | None,
    experiment_method: str | None,
    enable_rejection_cascade: bool,
    include_evaluator_feedback: bool,
    enable_elite_carry_forward: bool,
    enable_diverse_inspiration: bool,
) -> dict[str, Any]:
    metadata = {
        **_candidate_metadata(
            candidate=candidate,
            strategy=strategy,
            run_id=run_id,
            run_dir=run_dir,
            experiment_id=experiment_id,
            suite_name=suite_name,
            benchmark_group=benchmark_group,
            experiment_method=experiment_method,
        ),
        **_ablation_metadata(
            include_evaluator_feedback=include_evaluator_feedback,
            enable_rejection_cascade=enable_rejection_cascade,
            enable_elite_carry_forward=enable_elite_carry_forward,
            enable_diverse_inspiration=enable_diverse_inspiration,
        ),
    }
    if enable_rejection_cascade:
        cascade = preflight_generated_search_space(
            candidate.path,
            m=M,
            n=N,
            k=K,
            target_name=target_name,
        )
        metadata = {**metadata, **cascade.metadata}
        if not cascade.passed:
            result = record_rejected_matmul_experiment(
                strategy=strategy,
                M=M,
                N=N,
                K=K,
                target_name=target_name,
                num_warmup=num_warmup,
                num_trials=num_trials,
                benchmark_invocations=benchmark_invocations,
                min_repeat_ms=min_repeat_ms,
                output_dir=output_dir,
                rejection_stage=cascade.stage,
                rejection_reason=cascade.reason,
                extra_metadata=metadata,
                postprocess_result=_fitness_metadata,
            )
            return _evaluated_candidate(candidate, result)
    else:
        metadata = {
            **metadata,
            "cascade_rejected": False,
            "rejection_stage": "",
            "rejection_reason": "",
            "tuning_skipped": False,
        }

    safe_id = candidate.candidate_id.replace("/", "_")
    result = run_matmul_experiment(
        strategy=strategy,
        strategy_config=StrategyBuildConfig(
            work_dir=generation_dir / "work_dirs" / safe_id,
            generated_search_space_path=candidate.path,
            max_trials_global=max_trials_global,
            num_trials_per_iter=num_trials_per_iter,
            cost_model=cost_model,
            task_scheduler=task_scheduler,
            seed=seed,
            num_tuning_cores=num_tuning_cores,
        ),
        M=M,
        N=N,
        K=K,
        target_name=target_name,
        num_warmup=num_warmup,
        num_trials=num_trials,
        benchmark_invocations=benchmark_invocations,
        min_repeat_ms=min_repeat_ms,
        output_dir=output_dir,
        extra_metadata={
            **metadata,
            "cascade_rejected": False,
            "rejection_stage": "",
            "rejection_reason": "",
            "tuning_skipped": False,
        },
        postprocess_result=_fitness_metadata,
    )
    return _evaluated_candidate(candidate, result)


def _evaluated_candidate(candidate: Candidate, result: dict[str, Any]) -> dict[str, Any]:
    fitness = FitnessResult(
        score=float(result["fitness_score"]),
        reason=str(result["fitness_reason"]),
    )
    return {
        "candidate": _candidate_dict(candidate),
        "fitness": asdict(fitness),
        "result": result,
    }


def _make_next_generation(
    *,
    evaluated: list[dict[str, Any]],
    survivors: int,
    generation: int,
    generation_dir: Path,
    population_size: int,
    target_name: str,
    M: int,
    N: int,
    K: int,
    bedrock_client: BedrockClient | None,
    dry_run: bool,
    include_evaluator_feedback: bool,
    enable_rejection_cascade: bool,
    enable_elite_carry_forward: bool,
    enable_diverse_inspiration: bool,
) -> list[Candidate]:
    generation_dir.mkdir(parents=True, exist_ok=True)
    plan = plan_next_generation(
        evaluated,
        survivors=survivors,
        population_size=population_size,
        enable_elite_carry_forward=enable_elite_carry_forward,
        enable_diverse_inspiration=enable_diverse_inspiration,
    )
    _write_json(
        generation_dir / "selection.json",
        {
            "generation": generation,
            **_ablation_metadata(
                include_evaluator_feedback=include_evaluator_feedback,
                enable_rejection_cascade=enable_rejection_cascade,
                enable_elite_carry_forward=enable_elite_carry_forward,
                enable_diverse_inspiration=enable_diverse_inspiration,
            ),
            **plan_metadata(plan),
        },
    )
    next_candidates: list[Candidate] = []
    for index, elite in enumerate(plan.elites):
        parent = elite["candidate"]
        candidate_id = f"g{generation:03d}_c{index:03d}_elite"
        candidate_path = generation_dir / f"candidate_{index:03d}_elite.py"
        shutil.copyfile(Path(parent["path"]), candidate_path)
        next_candidates.append(
            Candidate(
                candidate_id=candidate_id,
                generation=generation,
                path=candidate_path,
                parent_id=parent["candidate_id"],
                origin="elite",
            )
        )

    for index, source in enumerate(plan.mutation_sources, start=len(plan.elites)):
        parent_row = annotate_source(source)
        parent = parent_row["candidate"]
        parent_path = Path(parent["path"])
        parent_code = parent_path.read_text(encoding="utf-8")
        candidate_id = f"g{generation:03d}_c{index:03d}"
        candidate_path = generation_dir / f"candidate_{index:03d}.py"
        prompt_path = generation_dir / f"candidate_{index:03d}.prompt.txt"
        response_path = generation_dir / f"candidate_{index:03d}.response.txt"

        prompt = search_space_mutation_prompt(
            parent_code=parent_code,
            target_name=target_name,
            M=M,
            N=N,
            K=K,
            generation=generation,
            candidate_index=index,
            parent_feedback=(
                evaluator_feedback(
                    parent_row,
                    include_level2_artifacts=True,
                )
                if include_evaluator_feedback
                else ""
            ),
        )
        prompt_path.write_text(prompt + "\n", encoding="utf-8")

        if dry_run:
            code = _dry_run_mutation(parent_code=parent_code, generation=generation, index=index)
            response_path.write_text(code, encoding="utf-8")
        else:
            assert bedrock_client is not None
            response = bedrock_client.generate(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
            )
            response_path.write_text(response, encoding="utf-8")
            code = strip_code_fences(response)

        candidate_path.write_text(_ensure_trailing_newline(code), encoding="utf-8")
        next_candidates.append(
            Candidate(
                candidate_id=candidate_id,
                generation=generation,
                path=candidate_path,
                parent_id=parent["candidate_id"],
                prompt_path=prompt_path,
                response_path=response_path,
                origin=(
                    "inspiration_mutation"
                    if source.selection_source_role == "diverse_inspiration"
                    else "mutation"
                ),
                inspiration_id=(
                    parent["candidate_id"]
                    if source.selection_source_role == "diverse_inspiration"
                    else None
                ),
            )
        )
    return next_candidates


def _dry_run_mutation(*, parent_code: str, generation: int, index: int) -> str:
    return (
        f"# Dry-run search-space mutation generated locally for generation {generation}, "
        f"candidate {index}.\n"
        + parent_code
    )


def _candidate_dict(candidate: Candidate) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "generation": candidate.generation,
        "path": str(candidate.path),
        "parent_id": candidate.parent_id,
        "prompt_path": str(candidate.prompt_path) if candidate.prompt_path else None,
        "response_path": str(candidate.response_path) if candidate.response_path else None,
        "origin": candidate.origin,
        "inspiration_id": candidate.inspiration_id,
    }


def _candidate_metadata(
    *,
    candidate: Candidate,
    strategy,
    run_id: str,
    run_dir: Path,
    experiment_id: str | None,
    suite_name: str | None,
    benchmark_group: str | None,
    experiment_method: str | None,
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "suite_name": suite_name,
        "run_id": run_id,
        "run_kind": "evolution",
        "benchmark_group": benchmark_group,
        "experiment_method": experiment_method,
        "generation": candidate.generation,
        "candidate_id": candidate.candidate_id,
        "parent_id": candidate.parent_id,
        "candidate_origin": candidate.origin,
        "inspiration_id": candidate.inspiration_id,
        "selection_role": "candidate",
        "candidate_path": candidate.path,
        "prompt_path": candidate.prompt_path,
        "response_path": candidate.response_path,
        "evolution_run_dir": run_dir,
        "evolution_kind": strategy.level,
    }


def _fitness_metadata(result: dict[str, Any]) -> dict[str, Any]:
    fitness = score_result(result)
    return {
        "fitness_score": fitness.score,
        "fitness_reason": fitness.reason,
    }


def _ablation_metadata(
    *,
    include_evaluator_feedback: bool,
    enable_rejection_cascade: bool,
    enable_elite_carry_forward: bool,
    enable_diverse_inspiration: bool,
) -> dict[str, Any]:
    return {
        "evaluator_feedback_enabled": include_evaluator_feedback,
        "rejection_cascade_enabled": enable_rejection_cascade,
        "elite_carry_forward_enabled": enable_elite_carry_forward,
        "diverse_inspiration_enabled": enable_diverse_inspiration,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")


def _ensure_trailing_newline(value: str) -> str:
    if value.endswith("\n"):
        return value
    return value + "\n"
