"""Level-2 evolution loop for generated MetaSchedule search spaces."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.eval.experiment import run_matmul_experiment
from src.evolution.bedrock_client import BedrockClient
from src.evolution.candidate import Candidate
from src.evolution.fitness import score_result
from src.evolution.prompts import (
    SYSTEM_PROMPT,
    search_space_mutation_prompt,
    strip_code_fences,
)
from src.strategies import StrategyBuildConfig, get_strategy


def run_search_space_evolution(
    *,
    seed_candidate_path: Path,
    run_dir: Path,
    generations: int,
    population_size: int,
    survivors: int,
    target_name: str,
    M: int,
    N: int,
    K: int,
    num_warmup: int,
    num_trials: int,
    max_trials_global: int,
    num_trials_per_iter: int,
    cost_model: str,
    task_scheduler: str,
    seed: int | None,
    num_tuning_cores: int | str,
    bedrock_client: BedrockClient | None,
    dry_run: bool,
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
    strategy = get_strategy("generated-search-space")

    seed_dir = run_dir / "gen_000"
    seed_dir.mkdir(parents=True, exist_ok=True)
    seed_path = seed_dir / "candidate_000_seed.py"
    shutil.copyfile(seed_candidate_path, seed_path)
    active = [Candidate(candidate_id="g000_c000_seed", generation=0, path=seed_path)]

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
                max_trials_global=max_trials_global,
                num_trials_per_iter=num_trials_per_iter,
                cost_model=cost_model,
                task_scheduler=task_scheduler,
                seed=seed,
                num_tuning_cores=num_tuning_cores,
            )
            for candidate in active
        ]
        evaluated.sort(key=lambda row: row["fitness"]["score"], reverse=True)
        history.extend(evaluated)
        _write_json(generation_dir / "ranking.json", evaluated)

        if generation == generations:
            break

        parents = [row["candidate"] for row in evaluated[:survivors]]
        active = _make_next_generation(
            parents=parents,
            generation=generation + 1,
            generation_dir=run_dir / f"gen_{generation + 1:03d}",
            population_size=population_size,
            target_name=target_name,
            M=M,
            N=N,
            K=K,
            bedrock_client=bedrock_client,
            dry_run=dry_run,
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
    max_trials_global: int,
    num_trials_per_iter: int,
    cost_model: str,
    task_scheduler: str,
    seed: int | None,
    num_tuning_cores: int | str,
) -> dict[str, Any]:
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
        output_dir=generation_dir / "results",
    )
    fitness = score_result(result)
    return {
        "candidate": _candidate_dict(candidate),
        "fitness": asdict(fitness),
        "result": result,
    }


def _make_next_generation(
    *,
    parents: list[dict[str, Any]],
    generation: int,
    generation_dir: Path,
    population_size: int,
    target_name: str,
    M: int,
    N: int,
    K: int,
    bedrock_client: BedrockClient | None,
    dry_run: bool,
) -> list[Candidate]:
    generation_dir.mkdir(parents=True, exist_ok=True)
    next_candidates: list[Candidate] = []
    for index in range(population_size):
        parent = parents[index % len(parents)]
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
