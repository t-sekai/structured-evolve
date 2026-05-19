#!/usr/bin/env python3
"""Run Level-2 evolution over generated MetaSchedule search spaces."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evolution.bedrock_client import BedrockClient
from src.evolution.config import DEFAULT_API_CONFIG_PATH
from src.evolution.space_loop import run_search_space_evolution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evolve generated TVM MetaSchedule search spaces for matmul."
    )
    parser.add_argument(
        "--seed-candidate-path",
        type=Path,
        default=Path("generated/search_spaces/basic_matmul.py"),
        help="Initial search-space candidate file.",
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--population-size", type=int, default=2)
    parser.add_argument("--survivors", type=int, default=1)
    parser.add_argument("--M", type=int, default=64)
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--target", choices=("llvm", "cuda"), default="llvm")
    parser.add_argument("--num-warmup", type=int, default=1)
    parser.add_argument("--num-trials", type=int, default=3)
    parser.add_argument("--max-trials-global", type=int, default=8)
    parser.add_argument("--num-trials-per-iter", type=int, default=4)
    parser.add_argument("--cost-model", choices=("xgb", "random", "mlp"), default="random")
    parser.add_argument(
        "--task-scheduler",
        choices=("gradient", "round-robin"),
        default="round-robin",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-tuning-cores", default="1")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--use-bedrock", action="store_true")
    parser.add_argument("--bedrock-model-id", default=None)
    parser.add_argument("--bedrock-region", default=None)
    parser.add_argument("--api-config", type=Path, default=DEFAULT_API_CONFIG_PATH)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "generations",
        "population_size",
        "survivors",
        "M",
        "N",
        "K",
        "num_warmup",
        "num_trials",
        "max_trials_global",
        "num_trials_per_iter",
    ):
        value = getattr(args, name)
        if name == "generations":
            if value < 0:
                raise ValueError(f"--generations must be non-negative, got {value}")
        elif value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive, got {value}")
    if args.survivors > args.population_size:
        raise ValueError("--survivors cannot exceed --population-size")
    if args.dry_run == args.use_bedrock:
        raise ValueError("Choose exactly one of --dry-run or --use-bedrock")
    if not args.seed_candidate_path.exists():
        raise FileNotFoundError(f"Seed candidate not found: {args.seed_candidate_path}")
    _parse_tuning_cores(args.num_tuning_cores)


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        run_dir = args.run_dir or _default_run_dir()
        bedrock_client = None
        if args.use_bedrock:
            bedrock_client = BedrockClient.from_sources(
                config_path=args.api_config,
                model_id=args.bedrock_model_id,
                region_name=args.bedrock_region,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )

        history = run_search_space_evolution(
            seed_candidate_path=args.seed_candidate_path,
            run_dir=run_dir,
            generations=args.generations,
            population_size=args.population_size,
            survivors=args.survivors,
            target_name=args.target,
            M=args.M,
            N=args.N,
            K=args.K,
            num_warmup=args.num_warmup,
            num_trials=args.num_trials,
            max_trials_global=args.max_trials_global,
            num_trials_per_iter=args.num_trials_per_iter,
            cost_model=args.cost_model,
            task_scheduler=args.task_scheduler,
            seed=args.seed,
            num_tuning_cores=_parse_tuning_cores(args.num_tuning_cores),
            bedrock_client=bedrock_client,
            dry_run=args.dry_run,
        )
        best = history[0]
        print(f"run_dir: {run_dir}")
        print(f"best_candidate: {best['candidate']['candidate_id']}")
        print(f"best_path: {best['candidate']['path']}")
        print(f"best_score: {best['fitness']['score']:.6g}")
        print(f"best_reason: {best['fitness']['reason']}")
        print(f"best_compile: {best['result']['compile_passed']}")
        print(f"best_correctness: {best['result']['correctness_passed']}")
        print(f"best_latency_ms_mean: {best['result']['latency_ms_mean']}")
        print(f"best_tuning_time_sec: {best['result'].get('tuning_time_sec')}")
        return 0
    except Exception as err:  # pylint: disable=broad-except
        print(f"ERROR: {err}", file=sys.stderr)
        return 1


def _default_run_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("generated") / "evolution" / f"search_space_{timestamp}"


def _parse_tuning_cores(value: str) -> int | str:
    if value in ("physical", "logical"):
        return value
    cores = int(value)
    if cores <= 0:
        raise ValueError(f"--num-tuning-cores must be positive, got {cores}")
    return cores


if __name__ == "__main__":
    raise SystemExit(main())
