#!/usr/bin/env python3
"""Run Level-1 OpenEvolve-style search over direct schedule candidates."""

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
from src.evolution.loop import run_schedule_evolution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evolve direct TVM schedule candidates for matmul."
    )
    parser.add_argument(
        "--seed-candidate-path",
        type=Path,
        default=Path("generated/schedules/identity.py"),
        help="Initial schedule candidate file.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Evolution run directory. Defaults under generated/evolution.",
    )
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--population-size", type=int, default=4)
    parser.add_argument("--survivors", type=int, default=2)
    parser.add_argument("--M", type=int, default=64)
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument(
        "--target",
        choices=("llvm", "cuda"),
        default="llvm",
        help="TVM target to evaluate.",
    )
    parser.add_argument("--num-warmup", type=int, default=1)
    parser.add_argument("--num-trials", type=int, default=3)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate deterministic local mutations instead of calling Bedrock.",
    )
    parser.add_argument(
        "--use-bedrock",
        action="store_true",
        help="Call AWS Bedrock for schedule mutations.",
    )
    parser.add_argument(
        "--bedrock-model-id",
        default=None,
        help="Bedrock model id. Overrides config and BEDROCK_MODEL_ID.",
    )
    parser.add_argument(
        "--bedrock-region",
        default=None,
        help="AWS region. Overrides config and AWS_REGION/AWS_DEFAULT_REGION.",
    )
    parser.add_argument(
        "--api-config",
        type=Path,
        default=DEFAULT_API_CONFIG_PATH,
        help="TOML file for local API keys and Bedrock defaults.",
    )
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

        history = run_schedule_evolution(
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
        return 0
    except Exception as err:  # pylint: disable=broad-except
        print(f"ERROR: {err}", file=sys.stderr)
        return 1


def _default_run_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("generated") / "evolution" / f"schedule_{timestamp}"


if __name__ == "__main__":
    raise SystemExit(main())
