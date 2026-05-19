"""Common interfaces for pluggable matmul scheduling strategies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import tvm


@dataclass(frozen=True)
class StrategyBuildConfig:
    """Configuration shared by scheduling strategies.

    Later LLM-generated schedule and search-space strategies should add their own
    fields around this object rather than changing the evaluation pipeline.
    """

    work_dir: Path | None = None
    max_trials_global: int = 64
    max_trials_per_task: int | None = None
    num_trials_per_iter: int = 64
    seed: int | None = 0
    num_tuning_cores: int | str = "physical"
    cost_model: str = "xgb"
    task_scheduler: str = "gradient"
    post_optimization: bool = False
    generated_schedule_path: Path | None = None
    generated_search_space_path: Path | None = None


@dataclass(frozen=True)
class StrategyBuildResult:
    """Compiled TVM module plus metadata produced by a scheduling strategy."""

    lib: tvm.runtime.Module
    scheduled_module: tvm.IRModule
    metadata: dict[str, Any]


class MatmulStrategy(Protocol):
    """A strategy that turns the canonical matmul IRModule into a compiled module."""

    name: str
    level: str

    def build(
        self,
        *,
        ir_module: tvm.IRModule,
        target: tvm.target.Target,
        target_name: str,
        config: StrategyBuildConfig,
    ) -> StrategyBuildResult:
        """Schedule, tune, or otherwise compile a matmul IRModule."""
        raise NotImplementedError
