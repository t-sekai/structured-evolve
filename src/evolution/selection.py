"""Selection planning for evolution generations."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MutationSource:
    """One evaluated candidate selected as a mutation parent or inspiration."""

    result: dict[str, Any]
    selection_source_role: str


@dataclass(frozen=True)
class GenerationPlan:
    """Elites to carry forward plus sources to mutate for the next generation."""

    elites: list[dict[str, Any]]
    mutation_sources: list[MutationSource]


def plan_next_generation(
    evaluated: list[dict[str, Any]],
    *,
    survivors: int,
    population_size: int,
    enable_elite_carry_forward: bool = False,
    enable_diverse_inspiration: bool = True,
) -> GenerationPlan:
    """Choose carried-forward elites and mutation sources for one generation."""
    if not evaluated:
        raise ValueError("Cannot plan a generation without evaluated candidates")
    if survivors < 1:
        raise ValueError("survivors must be positive")
    if population_size < 1:
        raise ValueError("population_size must be positive")

    parent_pool = evaluated[: min(survivors, len(evaluated))]
    elites = (
        parent_pool[: min(survivors, population_size)]
        if enable_elite_carry_forward
        else []
    )
    mutation_slots = population_size - len(elites)
    if not mutation_slots:
        return GenerationPlan(elites=elites, mutation_sources=[])

    parent_digests = {_source_digest(result) for result in parent_pool}
    inspiration = None
    if enable_diverse_inspiration:
        inspiration = next(
            (
                result
                for result in evaluated[len(parent_pool) :]
                if _is_valid(result) and _source_digest(result) not in parent_digests
            ),
            None,
        )

    mutation_sources: list[MutationSource] = []
    parent_index = 0
    if not elites:
        mutation_sources.append(MutationSource(parent_pool[parent_index], "elite_parent"))
        parent_index += 1

    if inspiration is not None and len(mutation_sources) < mutation_slots:
        mutation_sources.append(MutationSource(inspiration, "diverse_inspiration"))

    while len(mutation_sources) < mutation_slots:
        parent = parent_pool[parent_index % len(parent_pool)]
        mutation_sources.append(MutationSource(parent, "elite_parent"))
        parent_index += 1

    return GenerationPlan(elites=elites, mutation_sources=mutation_sources)


def annotate_source(source: MutationSource) -> dict[str, Any]:
    """Attach the source role to an evaluated row before prompt feedback."""
    return {**source.result, "selection_source_role": source.selection_source_role}


def plan_metadata(plan: GenerationPlan) -> dict[str, Any]:
    """Return JSON-safe selection metadata."""
    return {
        "elites": [
            {
                "candidate_id": result.get("candidate", {}).get("candidate_id"),
                "fitness_score": _fitness_score(result),
            }
            for result in plan.elites
        ],
        "mutation_sources": [
            {
                "candidate_id": source.result.get("candidate", {}).get("candidate_id"),
                "selection_source_role": source.selection_source_role,
                "fitness_score": _fitness_score(source.result),
            }
            for source in plan.mutation_sources
        ],
    }


def _source_digest(result: dict[str, Any]) -> str:
    path = Path(str(result["candidate"]["path"]))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_valid(result: dict[str, Any]) -> bool:
    persisted = result.get("result", {})
    return bool(persisted.get("compile_passed")) and bool(persisted.get("correctness_passed"))


def _fitness_score(result: dict[str, Any]) -> Any:
    return result.get("fitness", {}).get("score")
