#!/usr/bin/env python3
"""Run a benchmark matrix across workload tasks and experiment methods."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.suite import (
    Conv2DTaskCase,
    MethodCase,
    MatmulTaskCase,
    SuiteRunConfig,
    default_experiment_id,
    run_benchmark_suite,
)


METHOD_NAMES = (
    "fixed",
    "metaschedule",
    "level1-candidate",
    "level2-candidate",
    "level1-search",
    "level2-search",
)

WORKLOAD_NAMES = ("matmul", "conv2d")

LEVEL2_FINAL_EVALUATION_POLICIES = (
    "fresh-retune",
    "exact-winner",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reproducible checkpoint-style comparisons."
    )
    parser.add_argument(
        "--shape",
        action="append",
        default=None,
        help="Matmul shape as M,N,K or MxNxK. Can be passed multiple times.",
    )
    parser.add_argument(
        "--conv2d-shape",
        action="append",
        default=None,
        help=(
            "Conv2D shape as B,CI,H,W,CO,KH,KW,stride,pad. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--workload",
        choices=WORKLOAD_NAMES,
        default="matmul",
        help="Workload family for --shape/--conv2d-shape.",
    )
    parser.add_argument(
        "--target",
        action="append",
        choices=("llvm", "cuda"),
        default=None,
        help="Target to run. Can be passed multiple times. Defaults to llvm.",
    )
    parser.add_argument(
        "--methods",
        default="fixed,metaschedule,level1-candidate,level2-candidate",
        help=f"Comma-separated experiment methods. Available: {', '.join(METHOD_NAMES)}.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--suite-dir", type=Path, default=None)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--suite-name", default="benchmark_suite")
    parser.add_argument("--num-warmup", type=int, default=3)
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument(
        "--benchmark-invocations",
        type=int,
        default=1,
        help="Kernel invocations per timing sample before any TVM minimum-duration adjustment.",
    )
    parser.add_argument(
        "--min-repeat-ms",
        type=int,
        default=None,
        help="Optional minimum duration in milliseconds for each timing sample.",
    )
    parser.add_argument(
        "--generated-schedule-path",
        type=Path,
        default=Path("generated/schedules/identity.py"),
    )
    parser.add_argument(
        "--generated-search-space-path",
        type=Path,
        default=None,
        help="Existing Level 2 search-space candidate. Defaults to a target-specific seed.",
    )
    parser.add_argument("--max-trials-global", type=int, default=64)
    parser.add_argument("--max-trials-per-task", type=int, default=None)
    parser.add_argument("--num-trials-per-iter", type=int, default=64)
    parser.add_argument("--cost-model", choices=("xgb", "random", "mlp"), default="xgb")
    parser.add_argument(
        "--task-scheduler",
        choices=("gradient", "round-robin"),
        default="gradient",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-tuning-cores", default="physical")
    parser.add_argument("--post-optimization", action="store_true")
    parser.add_argument("--bad-baseline", action="store_true")

    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--population-size", type=int, default=2)
    parser.add_argument("--survivors", type=int, default=1)
    parser.add_argument("--search-num-warmup", type=int, default=None)
    parser.add_argument("--search-num-trials", type=int, default=None)
    parser.add_argument("--search-benchmark-invocations", type=int, default=None)
    parser.add_argument("--search-min-repeat-ms", type=int, default=None)
    parser.add_argument("--search-max-trials-global", type=int, default=None)
    parser.add_argument("--search-num-trials-per-iter", type=int, default=None)
    parser.add_argument(
        "--level2-final-evaluation-policy",
        choices=LEVEL2_FINAL_EVALUATION_POLICIES,
        default="fresh-retune",
    )
    parser.add_argument(
        "--disable-evaluator-feedback",
        action="store_true",
        help="Ablate compact parent evaluator feedback in mutation prompts.",
    )
    parser.add_argument(
        "--disable-rejection-cascade",
        action="store_true",
        help="Ablate cheap syntax/compile/correctness rejection before full evaluation.",
    )
    parser.add_argument(
        "--enable-elite-carry-forward",
        action="store_true",
        help="Carry elite survivors directly into the next generation.",
    )
    parser.add_argument(
        "--disable-diverse-inspiration",
        dest="enable_diverse_inspiration",
        action="store_false",
        default=True,
        help="Ablate the valid non-elite diverse inspiration source.",
    )
    parser.add_argument("--use-bedrock", action="store_true")
    parser.add_argument("--bedrock-model-id", default=None)
    parser.add_argument("--bedrock-region", default=None)
    parser.add_argument("--api-config", type=Path, default=Path("config/api_keys.toml"))
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "num_warmup",
        "num_trials",
        "benchmark_invocations",
        "max_trials_global",
        "num_trials_per_iter",
        "generations",
        "population_size",
        "survivors",
    ):
        value = getattr(args, name)
        if name == "generations":
            if value < 0:
                raise ValueError(f"--generations must be non-negative, got {value}")
        elif value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive, got {value}")
    for name in (
        "max_trials_per_task",
        "search_num_warmup",
        "search_num_trials",
        "search_benchmark_invocations",
        "search_max_trials_global",
        "search_num_trials_per_iter",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive, got {value}")
    for name in ("min_repeat_ms", "search_min_repeat_ms"):
        value = getattr(args, name)
        if value is not None and value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative, got {value}")
    if args.survivors > args.population_size:
        raise ValueError("--survivors cannot exceed --population-size")
    _parse_tuning_cores(args.num_tuning_cores)


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        experiment_id = args.experiment_id or default_experiment_id("benchmark")
        suite_dir = args.suite_dir or args.output_dir / "suites" / experiment_id
        results = run_benchmark_suite(
            tasks=_task_cases(args),
            methods=_method_cases(args),
            config=SuiteRunConfig(
                output_dir=args.output_dir,
                suite_dir=suite_dir,
                experiment_id=experiment_id,
                suite_name=args.suite_name,
                num_warmup=args.num_warmup,
                num_trials=args.num_trials,
                benchmark_invocations=args.benchmark_invocations,
                min_repeat_ms=args.min_repeat_ms,
                bad_baseline=args.bad_baseline,
                bedrock_client=_bedrock_client(args) if args.use_bedrock else None,
            ),
        )
        failures = [
            result
            for result in results
            if not result.get("compile_passed") or not result.get("correctness_passed")
        ]
        print(f"experiment_id: {experiment_id}")
        print(f"suite_dir: {suite_dir}")
        print(f"results_csv: {args.output_dir / 'results.csv'}")
        print(f"runs: {len(results)}")
        print(f"failures: {len(failures)}")
        return 1 if failures else 0
    except Exception as err:  # pylint: disable=broad-except
        print(f"ERROR: {err}", file=sys.stderr)
        return 1


def _task_cases(args: argparse.Namespace) -> list[MatmulTaskCase | Conv2DTaskCase]:
    targets = args.target or ["llvm"]
    tasks: list[MatmulTaskCase | Conv2DTaskCase] = []
    if args.workload == "matmul":
        shape_values = args.shape or ["128,128,128"]
        for shape_value in shape_values:
            M, N, K = _parse_shape(shape_value)
            for target in targets:
                tasks.append(MatmulTaskCase(M=M, N=N, K=K, target=target))
        return tasks

    shape_values = args.conv2d_shape or ["1,3,16,16,8,3,3,1,1"]
    for shape_value in shape_values:
        shape = _parse_conv2d_shape(shape_value)
        for target in targets:
            tasks.append(Conv2DTaskCase(*shape, target=target))
    return tasks


def _method_cases(args: argparse.Namespace) -> list[MethodCase]:
    names = [name.strip() for name in args.methods.split(",") if name.strip()]
    if not names:
        raise ValueError("--methods must include at least one method")
    unknown = [name for name in names if name not in METHOD_NAMES]
    if unknown:
        choices = ", ".join(METHOD_NAMES)
        raise ValueError(f"Unknown methods {unknown}. Available: {choices}")

    return [
        MethodCase(
            name=name,
            generated_schedule_path=args.generated_schedule_path,
            generated_search_space_path=args.generated_search_space_path,
            max_trials_global=args.max_trials_global,
            max_trials_per_task=args.max_trials_per_task,
            num_trials_per_iter=args.num_trials_per_iter,
            cost_model=args.cost_model,
            task_scheduler=args.task_scheduler,
            seed=args.seed,
            num_tuning_cores=_parse_tuning_cores(args.num_tuning_cores),
            post_optimization=args.post_optimization,
            generations=args.generations,
            population_size=args.population_size,
            survivors=args.survivors,
            search_num_warmup=args.search_num_warmup,
            search_num_trials=args.search_num_trials,
            search_benchmark_invocations=args.search_benchmark_invocations,
            search_min_repeat_ms=args.search_min_repeat_ms,
            search_max_trials_global=args.search_max_trials_global,
            search_num_trials_per_iter=args.search_num_trials_per_iter,
            level2_final_evaluation_policy=args.level2_final_evaluation_policy,
            include_evaluator_feedback=not args.disable_evaluator_feedback,
            enable_rejection_cascade=not args.disable_rejection_cascade,
            enable_elite_carry_forward=args.enable_elite_carry_forward,
            enable_diverse_inspiration=args.enable_diverse_inspiration,
            dry_run=not args.use_bedrock,
        )
        for name in names
    ]


def _bedrock_client(args: argparse.Namespace):
    from src.evolution.bedrock_client import BedrockClient

    return BedrockClient.from_sources(
        config_path=args.api_config,
        model_id=args.bedrock_model_id,
        region_name=args.bedrock_region,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )


def _parse_shape(value: str) -> tuple[int, int, int]:
    normalized = value.lower().replace("x", ",")
    pieces = [piece.strip() for piece in normalized.split(",") if piece.strip()]
    if len(pieces) != 3:
        raise ValueError(f"Shape must be M,N,K or MxNxK, got {value!r}")
    try:
        M, N, K = (int(piece) for piece in pieces)
    except ValueError as err:
        raise ValueError(f"Shape must contain integers, got {value!r}") from err
    for name, dimension in (("M", M), ("N", N), ("K", K)):
        if dimension <= 0:
            raise ValueError(f"{name} must be positive, got {dimension}")
    return M, N, K


def _parse_conv2d_shape(value: str) -> tuple[int, int, int, int, int, int, int, int, int]:
    normalized = value.lower().replace("x", ",")
    pieces = [piece.strip() for piece in normalized.split(",") if piece.strip()]
    if len(pieces) != 9:
        raise ValueError(
            "Conv2D shape must be B,CI,H,W,CO,KH,KW,stride,pad, "
            f"got {value!r}"
        )
    try:
        shape = tuple(int(piece) for piece in pieces)
    except ValueError as err:
        raise ValueError(f"Conv2D shape must contain integers, got {value!r}") from err
    names = (
        "batch",
        "in_channels",
        "height",
        "width",
        "out_channels",
        "kernel_h",
        "kernel_w",
        "stride",
    )
    for name, dimension in zip(names, shape[:8], strict=True):
        if dimension <= 0:
            raise ValueError(f"{name} must be positive, got {dimension}")
    if shape[8] < 0:
        raise ValueError(f"padding must be non-negative, got {shape[8]}")
    return shape


def _parse_tuning_cores(value: str) -> int | str:
    if value in ("physical", "logical"):
        return value
    try:
        cores = int(value)
    except ValueError as err:
        raise ValueError(
            "--num-tuning-cores must be 'physical', 'logical', or a positive integer"
        ) from err
    if cores <= 0:
        raise ValueError(f"--num-tuning-cores must be positive, got {cores}")
    return cores


if __name__ == "__main__":
    raise SystemExit(main())
