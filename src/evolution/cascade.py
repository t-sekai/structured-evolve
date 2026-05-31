"""Cheap rejection checks for evolution candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import tvm

from src.eval.benchmark import _array, _empty, _get_device, _make_target
from src.eval.correctness import check_against_reference
from src.kernels.workloads import Workload, make_matmul_workload
from src.strategies.generated_search_space import _load_candidate_module, _space_from_module


@dataclass(frozen=True)
class CascadeCheckResult:
    """Result of a cheap preflight check."""

    passed: bool
    stage: str
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


def syntax_check(candidate_path: Path) -> CascadeCheckResult:
    """Reject source files that Python cannot parse."""
    try:
        source = candidate_path.read_text(encoding="utf-8")
        compile(source, str(candidate_path), "exec")
    except Exception as error:  # pylint: disable=broad-except
        return CascadeCheckResult(
            passed=False,
            stage="syntax",
            reason=_concise_error(error),
            metadata={"syntax_check_passed": False},
        )
    return CascadeCheckResult(
        passed=True,
        stage="accepted",
        reason="syntax_check_passed",
        metadata={"syntax_check_passed": True},
    )


def preflight_generated_search_space(
    candidate_path: Path,
    *,
    m: int,
    n: int,
    k: int,
    target_name: str,
) -> CascadeCheckResult:
    """Compile and correctness-check direct Level 2 design-space variants."""
    return preflight_generated_workload_search_space(
        candidate_path,
        workload=make_matmul_workload(m, n, k),
        target_name=target_name,
    )


def preflight_generated_workload_search_space(
    candidate_path: Path,
    *,
    workload: Workload,
    target_name: str,
) -> CascadeCheckResult:
    """Compile and correctness-check direct Level 2 design-space variants."""
    syntax = syntax_check(candidate_path)
    if not syntax.passed:
        return syntax

    try:
        ir_module = workload.create_ir_module()
        module = _load_candidate_module(candidate_path)
        _, generate_design_space = _space_from_module(
            module=module,
            target_name=target_name,
        )
    except Exception as error:  # pylint: disable=broad-except
        return CascadeCheckResult(
            passed=False,
            stage="compile",
            reason=f"design_space_load_failed: {_concise_error(error)}",
            metadata={
                **syntax.metadata,
                "preflight_compile_passed": False,
                "preflight_correctness_passed": False,
            },
        )

    if generate_design_space is None:
        return CascadeCheckResult(
            passed=True,
            stage="accepted",
            reason="preflight_skipped_for_factory_search_space",
            metadata={
                **syntax.metadata,
                "preflight_skipped": True,
                "preflight_reason": "factory_search_space_requires_tune_context",
            },
        )

    try:
        schedules = list(generate_design_space(tvm.s_tir.Schedule(ir_module)))
    except Exception as error:  # pylint: disable=broad-except
        return CascadeCheckResult(
            passed=False,
            stage="compile",
            reason=f"design_space_generation_failed: {_concise_error(error)}",
            metadata={
                **syntax.metadata,
                "preflight_compile_passed": False,
                "preflight_correctness_passed": False,
            },
        )

    if not schedules:
        return CascadeCheckResult(
            passed=False,
            stage="compile",
            reason="design_space_generation_failed: no schedules returned",
            metadata={
                **syntax.metadata,
                "preflight_compile_passed": False,
                "preflight_correctness_passed": False,
                "preflight_variant_count": 0,
            },
        )

    target = _make_target(target_name)
    device = _get_device(target_name)
    rng = np.random.default_rng(0)
    inputs_np = workload.make_inputs(rng)
    reference = workload.reference(inputs_np)
    compile_errors: list[str] = []
    correctness_errors: list[str] = []

    for index, schedule in enumerate(schedules):
        try:
            lib = tvm.build(schedule.mod, target=target)
        except Exception as error:  # pylint: disable=broad-except
            compile_errors.append(_concise_error(error))
            continue

        try:
            input_tensors = tuple(_array(array, device) for array in inputs_np)
            output = _empty(
                workload.output_shape,
                dtype=workload.output_dtype,
                device=device,
            )
            lib["main"](*input_tensors, output)
            device.sync()
            correctness = check_against_reference(output.numpy(), reference)
        except Exception as error:  # pylint: disable=broad-except
            correctness_errors.append(_concise_error(error))
            continue

        if correctness.passed:
            return CascadeCheckResult(
                passed=True,
                stage="accepted",
                reason="preflight_compile_and_correctness_passed",
                metadata={
                    **syntax.metadata,
                    "preflight_skipped": False,
                    "preflight_compile_passed": True,
                    "preflight_correctness_passed": True,
                    "preflight_variant_index": index,
                    "preflight_variant_count": len(schedules),
                },
            )
        correctness_errors.append(
            f"variant {index}: max_abs_error={correctness.max_abs_error:.6g}"
        )

    if correctness_errors:
        return CascadeCheckResult(
            passed=False,
            stage="correctness",
            reason=f"preflight_correctness_failed: {correctness_errors[0]}",
            metadata={
                **syntax.metadata,
                "preflight_compile_passed": True,
                "preflight_correctness_passed": False,
                "preflight_variant_count": len(schedules),
            },
        )

    return CascadeCheckResult(
        passed=False,
        stage="compile",
        reason=f"preflight_compile_failed: {compile_errors[0]}",
        metadata={
            **syntax.metadata,
            "preflight_compile_passed": False,
            "preflight_correctness_passed": False,
            "preflight_variant_count": len(schedules),
        },
    )


def _concise_error(error: Exception, *, limit: int = 240) -> str:
    text = f"{type(error).__name__}: {error}".replace("\n", " ")
    return text if len(text) <= limit else f"{text[: limit - 3]}..."
