"""Correctness checking against NumPy reference outputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CorrectnessResult:
    passed: bool
    max_abs_error: float
    mean_abs_error: float


def check_against_reference(
    actual: np.ndarray,
    reference: np.ndarray,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> CorrectnessResult:
    """Compare an output tensor against the NumPy reference."""
    abs_error = np.abs(actual - reference)
    max_abs_error = float(np.max(abs_error))
    mean_abs_error = float(np.mean(abs_error))

    try:
        np.testing.assert_allclose(actual, reference, rtol=rtol, atol=atol)
        passed = True
    except AssertionError:
        passed = False

    return CorrectnessResult(
        passed=passed,
        max_abs_error=max_abs_error,
        mean_abs_error=mean_abs_error,
    )
