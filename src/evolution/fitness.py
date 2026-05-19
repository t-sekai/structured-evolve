"""Fitness scoring for benchmark result dictionaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FitnessResult:
    """A scalar score plus the reason it received that score."""

    score: float
    reason: str


def score_result(result: dict[str, Any]) -> FitnessResult:
    """Score one schedule benchmark result.

    Correct, faster candidates receive higher scores. Invalid candidates are
    kept in the population record with strongly negative scores so failures are
    visible instead of silently disappearing.
    """
    if not result.get("compile_passed", False):
        error_type = result.get("error_type") or "compile_failed"
        return FitnessResult(score=-1000.0, reason=str(error_type))

    if not result.get("correctness_passed", False):
        max_abs_error = result.get("max_abs_error")
        return FitnessResult(score=-100.0, reason=f"incorrect max_abs_error={max_abs_error}")

    latency = result.get("latency_ms_mean")
    if latency is None:
        return FitnessResult(score=-10.0, reason="missing_latency")

    latency = float(latency)
    if latency <= 0.0:
        return FitnessResult(score=-10.0, reason=f"invalid_latency={latency}")

    return FitnessResult(score=1.0 / latency, reason="correct")

