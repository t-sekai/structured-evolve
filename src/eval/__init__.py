"""Evaluation utilities for correctness, timing, result persistence, and suites."""

from src.eval.suite import (
    MethodCase,
    MatmulTaskCase,
    SuiteRunConfig,
    run_benchmark_suite,
    run_suite_case,
)

__all__ = [
    "MethodCase",
    "MatmulTaskCase",
    "SuiteRunConfig",
    "run_benchmark_suite",
    "run_suite_case",
]
