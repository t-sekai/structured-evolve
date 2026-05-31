"""Evaluation utilities for correctness, timing, result persistence, and suites."""

from src.eval.suite import (
    Conv2DTaskCase,
    MethodCase,
    MatmulTaskCase,
    SuiteRunConfig,
    run_benchmark_suite,
    run_suite_case,
)

__all__ = [
    "MethodCase",
    "Conv2DTaskCase",
    "MatmulTaskCase",
    "SuiteRunConfig",
    "run_benchmark_suite",
    "run_suite_case",
]
