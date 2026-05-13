#!/usr/bin/env python3
"""Run the checkpoint-1 TVM matmul baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.benchmark import build_and_run_matmul, measure_latency_ms
from src.eval.correctness import check_against_reference
from src.eval.results_io import append_csv_result, save_json_result
from src.kernels.matmul_tir import KERNEL_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile, run, check, and benchmark a simple TVM TensorIR matmul."
    )
    parser.add_argument("--M", type=int, default=256, help="Rows of A and C.")
    parser.add_argument("--N", type=int, default=256, help="Columns of B and C.")
    parser.add_argument("--K", type=int, default=256, help="Reduction dimension.")
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
        "--bad-baseline",
        action="store_true",
        help="Check an intentionally invalid all-zero output to exercise failure reporting.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("M", "N", "K", "num_warmup", "num_trials"):
        value = getattr(args, name)
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive, got {value}")


def main() -> int:
    args = parse_args()

    try:
        validate_args(args)

        rng = np.random.default_rng(seed=0)
        a_np = rng.standard_normal((args.M, args.K), dtype=np.float32)
        b_np = rng.standard_normal((args.K, args.N), dtype=np.float32)
        reference_np = a_np @ b_np

        # Future MetaSchedule/OpenEvolve work should replace this fixed schedule path
        # with generated schedules, search spaces, or search-space generators.
        run_state = build_and_run_matmul(
            a_np=a_np,
            b_np=b_np,
            M=args.M,
            N=args.N,
            K=args.K,
            target_name=args.target,
        )

        output_for_check = run_state.output_np
        if args.bad_baseline:
            output_for_check = np.zeros_like(run_state.output_np)

        correctness = check_against_reference(output_for_check, reference_np)
        latency = measure_latency_ms(
            lib=run_state.lib,
            device=run_state.device,
            a_tvm=run_state.a_tvm,
            b_tvm=run_state.b_tvm,
            c_tvm=run_state.c_tvm,
            num_warmup=args.num_warmup,
            num_trials=args.num_trials,
        )

        result = {
            "kernel_name": KERNEL_NAME,
            "M": args.M,
            "N": args.N,
            "K": args.K,
            "target": args.target,
            "device": run_state.device_description,
            "correctness_passed": correctness.passed,
            "max_abs_error": correctness.max_abs_error,
            "mean_abs_error": correctness.mean_abs_error,
            "latency_ms_mean": latency.mean,
            "latency_ms_std": latency.std,
            "num_warmup": args.num_warmup,
            "num_trials": args.num_trials,
            "timestamp": run_state.timestamp,
            "bad_baseline": args.bad_baseline,
        }

        json_path = save_json_result(result, args.output_dir)
        csv_path = append_csv_result(result, args.output_dir)

        status = "PASS" if correctness.passed else "FAIL"
        print(f"correctness: {status}")
        print(f"max_abs_error: {correctness.max_abs_error:.6g}")
        print(f"mean_abs_error: {correctness.mean_abs_error:.6g}")
        print(f"latency_ms_mean: {latency.mean:.6f}")
        print(f"latency_ms_std: {latency.std:.6f}")
        print(f"json_result: {json_path}")
        print(f"csv_result: {csv_path}")
        return 0 if correctness.passed else 2

    except Exception as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
