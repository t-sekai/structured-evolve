#!/usr/bin/env python3
"""Run one first-class experiment method on one matmul task."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


METHOD_NAMES = (
    "fixed",
    "metaschedule",
    "level1-candidate",
    "level2-candidate",
    "level1-search",
    "level2-search",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one experiment method. Search methods perform search first, "
            "then re-benchmark the best candidate as the final result."
        )
    )
    parser.add_argument(
        "--method",
        choices=METHOD_NAMES,
        default="fixed",
        help="Experiment method to run.",
    )
    parser.add_argument("--M", type=int, default=256, help="Rows of A and C.")
    parser.add_argument("--N", type=int, default=256, help="Columns of B and C.")
    parser.add_argument("--K", type=int, default=256, help="Reduction dimension.")
    parser.add_argument("--target", choices=("llvm", "cuda"), default="llvm")
    parser.add_argument("--num-warmup", type=int, default=3)
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--suite-name", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-kind", default="single_experiment")
    parser.add_argument("--benchmark-group", default="final_benchmark")
    parser.add_argument("--selection-role", default=None)
    parser.add_argument("--bad-baseline", action="store_true")

    parser.add_argument("--tuning-work-dir", type=Path, default=None)
    parser.add_argument(
        "--generated-schedule-path",
        type=Path,
        default=Path("generated/schedules/identity.py"),
        help="Existing Level 1 schedule candidate for level1-candidate.",
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

    parser.add_argument("--evolution-run-dir", type=Path, default=None)
    parser.add_argument(
        "--level1-seed-candidate-path",
        type=Path,
        default=Path("generated/schedules/identity.py"),
    )
    parser.add_argument(
        "--level2-seed-candidate-path",
        type=Path,
        default=None,
        help="Level 2 evolution seed. Defaults to a target-specific seed.",
    )
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--population-size", type=int, default=2)
    parser.add_argument("--survivors", type=int, default=1)
    parser.add_argument("--search-num-warmup", type=int, default=None)
    parser.add_argument("--search-num-trials", type=int, default=None)
    parser.add_argument("--search-max-trials-global", type=int, default=None)
    parser.add_argument("--search-num-trials-per-iter", type=int, default=None)
    parser.add_argument(
        "--use-bedrock",
        action="store_true",
        help="Use AWS Bedrock for evolution mutations. Otherwise deterministic dry-run mutations are used.",
    )
    parser.add_argument("--bedrock-model-id", default=None)
    parser.add_argument("--bedrock-region", default=None)
    parser.add_argument("--api-config", type=Path, default=Path("config/api_keys.toml"))
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
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
        "search_max_trials_global",
        "search_num_trials_per_iter",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive, got {value}")
    if args.survivors > args.population_size:
        raise ValueError("--survivors cannot exceed --population-size")
    _parse_tuning_cores(args.num_tuning_cores)


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)

        from src.eval.method import MethodRunConfig, default_experiment_id, run_experiment_method

        experiment_id = args.experiment_id or default_experiment_id(args.method)
        run_id = args.run_id or _default_run_id(args)
        result = run_experiment_method(
            method=args.method,
            config=MethodRunConfig(
                M=args.M,
                N=args.N,
                K=args.K,
                target_name=args.target,
                output_dir=args.output_dir,
                num_warmup=args.num_warmup,
                num_trials=args.num_trials,
                experiment_id=experiment_id,
                suite_name=args.suite_name,
                run_id=run_id,
                run_kind=args.run_kind,
                benchmark_group=args.benchmark_group,
                selection_role=args.selection_role,
                bad_baseline=args.bad_baseline,
                tuning_work_dir=args.tuning_work_dir,
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
                evolution_run_dir=args.evolution_run_dir,
                level1_seed_candidate_path=args.level1_seed_candidate_path,
                level2_seed_candidate_path=args.level2_seed_candidate_path,
                generations=args.generations,
                population_size=args.population_size,
                survivors=args.survivors,
                search_num_warmup=args.search_num_warmup,
                search_num_trials=args.search_num_trials,
                search_max_trials_global=args.search_max_trials_global,
                search_num_trials_per_iter=args.search_num_trials_per_iter,
                dry_run=not args.use_bedrock,
                bedrock_client=_bedrock_client(args) if args.use_bedrock else None,
            ),
        )
        _print_result(result)
        if not result["compile_passed"]:
            return 1
        return 0 if result["correctness_passed"] else 2
    except Exception as err:  # pylint: disable=broad-except
        print(f"ERROR: {err}", file=sys.stderr)
        return 1


def _print_result(result: dict) -> None:
    status = "PASS" if result["correctness_passed"] else "FAIL"
    print(f"experiment_id: {result.get('experiment_id')}")
    print(f"run_id: {result.get('run_id')}")
    print(f"method: {result.get('experiment_method')}")
    print(f"strategy: {result['strategy']}")
    print(f"level: {result['level']}")
    print(f"compile: {'PASS' if result['compile_passed'] else 'FAIL'}")
    print(f"correctness: {status}")
    if result["compile_passed"]:
        print(f"latency_ms_mean: {result['latency_ms_mean']:.6f}")
        print(f"latency_ms_std: {result['latency_ms_std']:.6f}")
        if result.get("tuning_time_sec") is not None:
            print(f"tuning_time_sec: {result['tuning_time_sec']:.6f}")
        if result.get("evolution_time_sec") is not None:
            print(f"evolution_time_sec: {result['evolution_time_sec']:.6f}")
    else:
        print(f"error_type: {result['error_type']}")
        print(f"error_message: {result['error_message']}")
    if result.get("best_candidate_path"):
        print(f"best_candidate_path: {result['best_candidate_path']}")
    print(f"json_result: {result['json_result']}")
    print(f"csv_result: {result['csv_result']}")


def _bedrock_client(args: argparse.Namespace):
    from src.evolution.bedrock_client import BedrockClient

    return BedrockClient.from_sources(
        config_path=args.api_config,
        model_id=args.bedrock_model_id,
        region_name=args.bedrock_region,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )


def _default_run_id(args: argparse.Namespace) -> str:
    return f"matmul_M{args.M}_N{args.N}_K{args.K}_{args.target}_{args.method}"


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
