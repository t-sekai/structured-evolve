#!/usr/bin/env python3
"""Sweep MetaSchedule hyperparameters and rank successful runs by latency."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_EXPERIMENT = REPO_ROOT / "scripts" / "run_experiment.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a grid search over TVM MetaSchedule hyperparameters using "
            "scripts/run_experiment.py, then write a ranked sweep summary."
        )
    )
    parser.add_argument("--workload", choices=("matmul", "conv2d"), default="matmul")
    parser.add_argument("--shape", default="128,128,128", help="Matmul shape M,N,K.")
    parser.add_argument(
        "--conv2d-shape",
        default="1,3,16,16,8,3,3,1,1",
        help="Conv2D shape B,CI,H,W,CO,KH,KW,stride,pad.",
    )
    parser.add_argument("--target", choices=("llvm", "cuda"), default="llvm")
    parser.add_argument(
        "--method",
        choices=("metaschedule", "level2-candidate"),
        default="level2-candidate",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/hparam_sweep"))
    parser.add_argument("--experiment-id", default=None)

    parser.add_argument("--max-trials-global", default="16,32,64")
    parser.add_argument("--num-trials-per-iter", default="4,8,16")
    parser.add_argument("--cost-model", default="random,xgb")
    parser.add_argument("--task-scheduler", default="round-robin,gradient")
    parser.add_argument("--seed", default="0,1,2")
    parser.add_argument("--num-tuning-cores", default="1,physical")
    parser.add_argument("--max-runs", type=int, default=None)

    parser.add_argument("--num-warmup", type=int, default=3)
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--benchmark-invocations", type=int, default=1)
    parser.add_argument("--min-repeat-ms", type=int, default=None)
    parser.add_argument("--post-optimization", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--pythonpath",
        default=None,
        help="Extra PYTHONPATH entries for child experiments, separated by ':'.",
    )
    parser.add_argument(
        "--library-path",
        default=None,
        help="Extra LD_LIBRARY_PATH entries for child experiments, separated by ':'.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment_id = args.experiment_id or _default_experiment_id(args)
    sweep_dir = args.output_dir / "sweeps" / experiment_id
    sweep_dir.mkdir(parents=True, exist_ok=True)

    combinations = list(_hyperparameter_grid(args))
    if args.max_runs is not None:
        combinations = combinations[: args.max_runs]
    if not combinations:
        raise ValueError("Sweep grid is empty.")

    summary_path = sweep_dir / "sweep_results.csv"
    completed = _completed_run_ids(summary_path) if args.resume else set()
    rows: list[dict[str, Any]] = _read_existing_rows(summary_path) if args.resume else []

    for index, params in enumerate(combinations, start=1):
        run_id = _run_id(args, index, params)
        if run_id in completed:
            print(f"[{index}/{len(combinations)}] skip completed {run_id}")
            continue

        command = _experiment_command(
            args=args,
            experiment_id=experiment_id,
            run_id=run_id,
            sweep_dir=sweep_dir,
            params=params,
        )
        print(f"[{index}/{len(combinations)}] run {run_id}: {_format_params(params)}")
        if args.dry_run:
            print(" ".join(command))
            continue

        completed_run = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=_child_env(args),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        result = _load_result_from_stdout(completed_run.stdout)
        row = _summary_row(
            run_id=run_id,
            params=params,
            returncode=completed_run.returncode,
            stdout=completed_run.stdout,
            stderr=completed_run.stderr,
            result=result,
        )
        rows.append(row)
        _write_csv(summary_path, rows)
        _write_text(sweep_dir / f"{run_id}.stdout.log", completed_run.stdout)
        _write_text(sweep_dir / f"{run_id}.stderr.log", completed_run.stderr)

    if args.dry_run:
        return 0

    ranked = _rank_successes(rows)
    best = ranked[0] if ranked else None
    best_path = sweep_dir / "best_config.json"
    with best_path.open("w", encoding="utf-8") as f:
        json.dump({"experiment_id": experiment_id, "best": best, "ranked": ranked}, f, indent=2)
        f.write("\n")

    print(f"sweep_dir: {sweep_dir}")
    print(f"summary_csv: {summary_path}")
    print(f"best_config_json: {best_path}")
    if best is None:
        print("best: none; no compile-correct runs completed")
        return 1
    print(
        "best: "
        f"run_id={best['run_id']} "
        f"latency_ms_mean={best['latency_ms_mean']} "
        f"max_trials_global={best['max_trials_global']} "
        f"num_trials_per_iter={best['num_trials_per_iter']} "
        f"cost_model={best['cost_model']} "
        f"task_scheduler={best['task_scheduler']} "
        f"seed={best['seed']} "
        f"num_tuning_cores={best['num_tuning_cores']}"
    )
    return 0


def _hyperparameter_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    grid = []
    for max_trials, trials_per_iter, cost_model, scheduler, seed, cores in itertools.product(
        _parse_int_list(args.max_trials_global),
        _parse_int_list(args.num_trials_per_iter),
        _parse_str_list(args.cost_model),
        _parse_str_list(args.task_scheduler),
        _parse_int_list(args.seed),
        _parse_str_list(args.num_tuning_cores),
    ):
        if trials_per_iter > max_trials:
            continue
        grid.append(
            {
                "max_trials_global": max_trials,
                "num_trials_per_iter": trials_per_iter,
                "cost_model": cost_model,
                "task_scheduler": scheduler,
                "seed": seed,
                "num_tuning_cores": cores,
            }
        )
    return grid


def _experiment_command(
    *,
    args: argparse.Namespace,
    experiment_id: str,
    run_id: str,
    sweep_dir: Path,
    params: dict[str, Any],
) -> list[str]:
    command = [
        args.python,
        str(RUN_EXPERIMENT),
        "--method",
        args.method,
        "--workload",
        args.workload,
        "--target",
        args.target,
        "--output-dir",
        str(args.output_dir),
        "--experiment-id",
        experiment_id,
        "--run-id",
        run_id,
        "--run-kind",
        "hyperparameter_sweep",
        "--benchmark-group",
        "hyperparameter_sweep",
        "--num-warmup",
        str(args.num_warmup),
        "--num-trials",
        str(args.num_trials),
        "--benchmark-invocations",
        str(args.benchmark_invocations),
        "--tuning-work-dir",
        str(sweep_dir / "work_dirs" / run_id),
        "--max-trials-global",
        str(params["max_trials_global"]),
        "--num-trials-per-iter",
        str(params["num_trials_per_iter"]),
        "--cost-model",
        str(params["cost_model"]),
        "--task-scheduler",
        str(params["task_scheduler"]),
        "--seed",
        str(params["seed"]),
        "--num-tuning-cores",
        str(params["num_tuning_cores"]),
    ]
    if args.min_repeat_ms is not None:
        command.extend(["--min-repeat-ms", str(args.min_repeat_ms)])
    if args.post_optimization:
        command.append("--post-optimization")
    if args.workload == "matmul":
        M, N, K = _parse_shape(args.shape)
        command.extend(["--M", str(M), "--N", str(N), "--K", str(K)])
    else:
        command.extend(["--conv2d-shape", args.conv2d_shape])
    return command


def _child_env(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    pythonpath_entries = _path_entries(args.pythonpath, _default_pythonpath_entries())
    library_entries = _path_entries(args.library_path, _default_library_path_entries())
    _prepend_env_path(env, "PYTHONPATH", pythonpath_entries)
    _prepend_env_path(env, "LD_LIBRARY_PATH", library_entries)
    return env


def _default_pythonpath_entries() -> list[Path]:
    return [REPO_ROOT / "third_party" / "tvm" / "python"]


def _default_library_path_entries() -> list[Path]:
    return [
        REPO_ROOT / "third_party" / "tvm" / "build",
        REPO_ROOT / "third_party" / "tvm" / "build-cuda",
        REPO_ROOT / "third_party" / "tvm" / "build-v011-cuda-shim",
        REPO_ROOT / "third_party" / "tvm" / "build-v011-cuda",
        REPO_ROOT / "third_party" / "tvm" / "build-cuda-llvm17",
    ]


def _path_entries(raw: str | None, defaults: list[Path]) -> list[str]:
    entries: list[Path | str]
    if raw is None:
        entries = defaults
    else:
        entries = [piece for piece in raw.split(":") if piece]
    normalized = []
    for entry in entries:
        path = Path(entry)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if path.exists():
            normalized.append(str(path))
    return normalized


def _prepend_env_path(env: dict[str, str], key: str, entries: list[str]) -> None:
    if not entries:
        return
    existing = env.get(key)
    if existing:
        env[key] = os.pathsep.join(entries + [existing])
    else:
        env[key] = os.pathsep.join(entries)


def _summary_row(
    *,
    run_id: str,
    params: dict[str, Any],
    returncode: int,
    stdout: str,
    stderr: str,
    result: dict[str, Any] | None,
) -> dict[str, Any]:
    row = {"run_id": run_id, "returncode": returncode, **params}
    if result is None:
        row.update(
            {
                "compile_passed": False,
                "correctness_passed": False,
                "latency_ms_mean": "",
                "latency_ms_std": "",
                "tuning_time_sec": "",
                "json_result": "",
                "error_type": "MissingResultJson",
                "error_message": _last_error(stdout, stderr),
            }
        )
        return row

    for key in (
        "compile_passed",
        "correctness_passed",
        "latency_ms_mean",
        "latency_ms_std",
        "tuning_time_sec",
        "json_result",
        "error_type",
        "error_message",
        "shape",
        "target",
        "device",
    ):
        row[key] = result.get(key, "")
    return row


def _load_result_from_stdout(stdout: str) -> dict[str, Any] | None:
    json_path = None
    for line in stdout.splitlines():
        if line.startswith("json_result:"):
            json_path = Path(line.split(":", 1)[1].strip())
    if json_path is None:
        return None
    if not json_path.is_absolute():
        json_path = REPO_ROOT / json_path
    if not json_path.exists():
        return None
    with json_path.open("r", encoding="utf-8") as f:
        result = json.load(f)
    result["json_result"] = str(json_path)
    return result


def _rank_successes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    successes = [
        row
        for row in rows
        if _is_true(row.get("compile_passed"))
        and _is_true(row.get("correctness_passed"))
        and row.get("latency_ms_mean") not in ("", None)
    ]
    return sorted(successes, key=lambda row: float(row["latency_ms_mean"]))


def _completed_run_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", newline="", encoding="utf-8") as f:
        return {row["run_id"] for row in csv.DictReader(f) if row.get("run_id")}


def _read_existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "returncode",
        "max_trials_global",
        "num_trials_per_iter",
        "cost_model",
        "task_scheduler",
        "seed",
        "num_tuning_cores",
        "compile_passed",
        "correctness_passed",
        "latency_ms_mean",
        "latency_ms_std",
        "tuning_time_sec",
        "shape",
        "target",
        "device",
        "json_result",
        "error_type",
        "error_message",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _run_id(args: argparse.Namespace, index: int, params: dict[str, Any]) -> str:
    shape = args.shape if args.workload == "matmul" else args.conv2d_shape
    safe_shape = shape.replace(",", "x").replace(" ", "")
    return (
        f"{args.workload}_{safe_shape}_{args.target}_{args.method}_"
        f"run{index:03d}_mtg{params['max_trials_global']}_"
        f"tpi{params['num_trials_per_iter']}_{params['cost_model']}_"
        f"{params['task_scheduler']}_seed{params['seed']}_cores{params['num_tuning_cores']}"
    )


def _default_experiment_id(args: argparse.Namespace) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"hparam_sweep_{args.workload}_{args.target}_{args.method}_{timestamp}"


def _parse_shape(value: str) -> tuple[int, int, int]:
    pieces = [piece.strip() for piece in value.lower().replace("x", ",").split(",") if piece.strip()]
    if len(pieces) != 3:
        raise ValueError(f"--shape must be M,N,K or MxNxK, got {value!r}")
    M, N, K = (int(piece) for piece in pieces)
    if min(M, N, K) <= 0:
        raise ValueError(f"--shape dimensions must be positive, got {value!r}")
    return M, N, K


def _parse_int_list(value: str) -> list[int]:
    return [int(piece) for piece in _parse_str_list(value)]


def _parse_str_list(value: str) -> list[str]:
    pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
    if not pieces:
        raise ValueError("Comma-separated list cannot be empty.")
    return pieces


def _format_params(params: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in params.items())


def _is_true(value: Any) -> bool:
    return value is True or str(value).lower() == "true"


def _last_error(stdout: str, stderr: str) -> str:
    lines = [line for line in (stderr + "\n" + stdout).splitlines() if line.strip()]
    return lines[-1][:500] if lines else ""


if __name__ == "__main__":
    raise SystemExit(main())
