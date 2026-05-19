#!/usr/bin/env python3
"""Run a matmul scheduling strategy through the shared benchmark pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.experiment import run_matmul_experiment
from src.strategies import StrategyBuildConfig, available_strategy_names, get_strategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile, run, check, and benchmark a TVM TensorIR matmul strategy."
    )
    parser.add_argument("--M", type=int, default=256, help="Rows of A and C.")
    parser.add_argument("--N", type=int, default=256, help="Columns of B and C.")
    parser.add_argument("--K", type=int, default=256, help="Reduction dimension.")
    parser.add_argument(
        "--strategy",
        choices=available_strategy_names(),
        default="fixed",
        help="Scheduling strategy to evaluate. 'fixed' preserves the checkpoint-1 baseline.",
    )
    parser.add_argument(
        "--target",
        choices=("llvm", "cuda"),
        default="llvm",
        help="TVM target to compile for.",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=3,
        help="Untimed warmup runs before benchmarking.",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=10,
        help="Number of timed trials for TVM time_evaluator.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for JSON and CSV results.",
    )
    parser.add_argument(
        "--tuning-work-dir",
        type=Path,
        default=None,
        help="Optional MetaSchedule work directory. Defaults under OUTPUT_DIR/work_dirs.",
    )
    parser.add_argument(
        "--generated-schedule-path",
        type=Path,
        default=None,
        help=(
            "Python file for --strategy generated-schedule. It must define "
            "apply_schedule(ir_module, target_name) -> tvm.IRModule."
        ),
    )
    parser.add_argument(
        "--generated-search-space-path",
        type=Path,
        default=None,
        help=(
            "Python file for --strategy generated-search-space. It must define "
            "generate_design_space(sch) or create_space_generator(target_name)."
        ),
    )
    parser.add_argument(
        "--max-trials-global",
        type=int,
        default=64,
        help="Global MetaSchedule tuning trial budget.",
    )
    parser.add_argument(
        "--max-trials-per-task",
        type=int,
        default=None,
        help="Optional per-task MetaSchedule tuning trial budget.",
    )
    parser.add_argument(
        "--num-trials-per-iter",
        type=int,
        default=64,
        help="MetaSchedule candidates measured per tuning iteration.",
    )
    parser.add_argument(
        "--cost-model",
        choices=("xgb", "random", "mlp"),
        default="xgb",
        help="MetaSchedule cost model.",
    )
    parser.add_argument(
        "--task-scheduler",
        choices=("gradient", "round-robin"),
        default="gradient",
        help="MetaSchedule task scheduler.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="MetaSchedule random seed.",
    )
    parser.add_argument(
        "--num-tuning-cores",
        default="physical",
        help="MetaSchedule tuning cores: physical, logical, or an integer.",
    )
    parser.add_argument(
        "--post-optimization",
        action="store_true",
        help="Enable TVM MetaSchedule post-optimization when supported.",
    )
    parser.add_argument(
        "--bad-baseline",
        action="store_true",
        help="Check an intentionally invalid all-zero output to exercise failure reporting.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "M",
        "N",
        "K",
        "num_warmup",
        "num_trials",
        "max_trials_global",
        "num_trials_per_iter",
    ):
        value = getattr(args, name)
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive, got {value}")
    if args.max_trials_per_task is not None and args.max_trials_per_task <= 0:
        raise ValueError(
            f"--max-trials-per-task must be positive, got {args.max_trials_per_task}"
        )
    _parse_tuning_cores(args.num_tuning_cores)


def main() -> int:
    args = parse_args()

    try:
        validate_args(args)

        strategy = get_strategy(args.strategy)
        strategy_config = StrategyBuildConfig(
            work_dir=args.tuning_work_dir,
            max_trials_global=args.max_trials_global,
            max_trials_per_task=args.max_trials_per_task,
            num_trials_per_iter=args.num_trials_per_iter,
            seed=args.seed,
            num_tuning_cores=_parse_tuning_cores(args.num_tuning_cores),
            cost_model=args.cost_model,
            task_scheduler=args.task_scheduler,
            post_optimization=args.post_optimization,
            generated_schedule_path=args.generated_schedule_path,
            generated_search_space_path=args.generated_search_space_path,
        )
        result = run_matmul_experiment(
            strategy=strategy,
            strategy_config=strategy_config,
            M=args.M,
            N=args.N,
            K=args.K,
            target_name=args.target,
            num_warmup=args.num_warmup,
            num_trials=args.num_trials,
            output_dir=args.output_dir,
            bad_baseline=args.bad_baseline,
        )

        status = "PASS" if result["correctness_passed"] else "FAIL"
        print(f"strategy: {result['strategy']}")
        print(f"level: {result['level']}")
        print(f"compile: {'PASS' if result['compile_passed'] else 'FAIL'}")
        print(f"correctness: {status}")
        if result["compile_passed"]:
            print(f"max_abs_error: {result['max_abs_error']:.6g}")
            print(f"mean_abs_error: {result['mean_abs_error']:.6g}")
            print(f"latency_ms_mean: {result['latency_ms_mean']:.6f}")
            print(f"latency_ms_std: {result['latency_ms_std']:.6f}")
            if result.get("tuning_time_sec") is not None:
                print(f"tuning_time_sec: {result['tuning_time_sec']:.6f}")
        else:
            print(f"error_type: {result['error_type']}")
            print(f"error_message: {result['error_message']}")
        print(f"json_result: {result['json_result']}")
        print(f"csv_result: {result['csv_result']}")
        if not result["compile_passed"]:
            return 1
        return 0 if result["correctness_passed"] else 2

    except Exception as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1


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
