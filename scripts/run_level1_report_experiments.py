#!/usr/bin/env python3
"""Run Level-1 schedule-generation experiments for the report."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_EXPERIMENT = REPO_ROOT / "scripts" / "run_experiment.py"


@dataclass(frozen=True)
class Shape:
    M: int
    N: int
    K: int

    @property
    def label(self) -> str:
        return f"{self.M}x{self.N}x{self.K}"

    @property
    def shape_id(self) -> str:
        return f"M{self.M}_N{self.N}_K{self.K}"


@dataclass(frozen=True)
class Ablation:
    name: str
    flags: tuple[str, ...] = ()


BASELINE_METHODS = ("fixed", "metaschedule", "level1-candidate")
ABLATIONS = (
    Ablation("full"),
    Ablation("no_evaluator_feedback", ("--disable-evaluator-feedback",)),
    Ablation("no_rejection_cascade", ("--disable-rejection-cascade",)),
    Ablation("elite_carry_forward", ("--enable-elite-carry-forward",)),
    Ablation("diverse_inspiration", ("--enable-diverse-inspiration",)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("llvm", "cuda"), default="cuda")
    parser.add_argument(
        "--shape",
        action="append",
        default=None,
        help="Matmul shape as M,N,K or MxNxK. Can be passed multiple times.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/level1_report"))
    parser.add_argument("--experiment-id", default="level1_report")
    parser.add_argument("--python", default=str(REPO_ROOT / "tvm-build-venv" / "bin" / "python"))
    parser.add_argument("--num-warmup", type=int, default=3)
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--search-num-warmup", type=int, default=1)
    parser.add_argument("--search-num-trials", type=int, default=3)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--population-size", type=int, default=4)
    parser.add_argument("--survivors", type=int, default=2)
    parser.add_argument("--max-trials-global", type=int, default=8)
    parser.add_argument("--num-trials-per-iter", type=int, default=8)
    parser.add_argument("--cost-model", choices=("xgb", "random", "mlp"), default="xgb")
    parser.add_argument("--task-scheduler", choices=("gradient", "round-robin"), default="round-robin")
    parser.add_argument("--num-tuning-cores", default="1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-bedrock", action="store_true")
    parser.add_argument("--bedrock-model-id", default=None)
    parser.add_argument("--bedrock-region", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--only-shape", default=None, help="Run one shape label from the configured matrix.")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-searches", action="store_true")
    parser.add_argument("--budget-sweep", action="store_true")
    parser.add_argument("--sweep-generations", default="1,3,5")
    parser.add_argument("--sweep-population-sizes", default="2,4")
    parser.add_argument("--sweep-survivors", default="1,2")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    shapes = _shapes(args)
    report_dir = args.output_dir / "report_artifacts" / args.experiment_id
    report_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment_id": args.experiment_id,
        "target": args.target,
        "shapes": [shape.label for shape in shapes],
        "use_bedrock": args.use_bedrock,
        "generations": args.generations,
        "population_size": args.population_size,
        "survivors": args.survivors,
        "budget_sweep": args.budget_sweep,
    }
    _write_json(report_dir / "manifest.json", manifest)

    if not args.summarize_only:
        runs = _planned_runs(args, shapes, report_dir)
        completed = (
            _completed_run_ids(args.output_dir / "results.csv", args.experiment_id)
            if args.resume
            else set()
        )
        for index, run in enumerate(runs, start=1):
            if run["run_id"] in completed:
                print(f"[{index}/{len(runs)}] skip completed {run['run_id']}")
                continue
            print(f"[{index}/{len(runs)}] run {run['run_id']}")
            if args.dry_run:
                print(" ".join(run["command"]))
                continue
            proc = subprocess.run(
                run["command"],
                cwd=REPO_ROOT,
                env=_child_env(),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            _write_text(report_dir / f"{run['run_id']}.stdout.log", proc.stdout)
            _write_text(report_dir / f"{run['run_id']}.stderr.log", proc.stderr)
            if proc.returncode not in (0, 1, 2):
                print(f"  unexpected return code {proc.returncode}; see logs")

    if args.dry_run:
        return 0

    rows = _dedup_rows(_read_results(args.output_dir / "results.csv", args.experiment_id))
    _write_summary_tables(report_dir, rows)
    if args.budget_sweep:
        _write_budget_sweep_outputs(report_dir, rows)
    _write_generation_plot(report_dir, rows)
    _write_qualitative_examples(report_dir, rows)
    print(f"report_dir: {report_dir}")
    print(f"baseline_table: {report_dir / 'baseline_table.csv'}")
    print(f"ablation_table: {report_dir / 'ablation_table.csv'}")
    print(f"generation_plot: {report_dir / 'generation_best_latency.svg'}")
    if args.budget_sweep:
        print(f"budget_sweep_table: {report_dir / 'budget_sweep_table.csv'}")
        print(f"budget_sweep_plot: {report_dir / 'budget_sweep_latency.svg'}")
    print(f"qualitative_examples: {report_dir / 'qualitative_examples.md'}")
    return 0


def _planned_runs(args: argparse.Namespace, shapes: list[Shape], report_dir: Path) -> list[dict[str, Any]]:
    if args.budget_sweep:
        return _budget_sweep_runs(args, shapes, report_dir)

    runs: list[dict[str, Any]] = []
    for shape in shapes:
        if not args.skip_baselines:
            for method in BASELINE_METHODS:
                run_id = f"{shape.shape_id}_{args.target}_{method}"
                command = _base_command(args, shape, method, run_id, "level1_baseline")
                if method == "metaschedule":
                    command.extend(
                        [
                            "--max-trials-global",
                            str(args.max_trials_global),
                            "--num-trials-per-iter",
                            str(args.num_trials_per_iter),
                            "--cost-model",
                            args.cost_model,
                            "--task-scheduler",
                            args.task_scheduler,
                            "--num-tuning-cores",
                            args.num_tuning_cores,
                            "--seed",
                            str(args.seed),
                            "--tuning-work-dir",
                            str(report_dir / "work_dirs" / run_id),
                        ]
                    )
                runs.append({"run_id": run_id, "command": command})

        if not args.skip_searches:
            for ablation in ABLATIONS:
                run_id = f"{shape.shape_id}_{args.target}_level1_search_{ablation.name}"
                command = _base_command(args, shape, "level1-search", run_id, "level1_ablation")
                command.extend(
                    [
                        "--generations",
                        str(args.generations),
                        "--population-size",
                        str(args.population_size),
                        "--survivors",
                        str(args.survivors),
                        "--search-num-warmup",
                        str(args.search_num_warmup),
                        "--search-num-trials",
                        str(args.search_num_trials),
                        "--evolution-run-dir",
                        str(report_dir / "evolution_runs" / run_id),
                    ]
                )
                if args.use_bedrock:
                    command.append("--use-bedrock")
                    if args.bedrock_model_id:
                        command.extend(["--bedrock-model-id", args.bedrock_model_id])
                    if args.bedrock_region:
                        command.extend(["--bedrock-region", args.bedrock_region])
                    if args.temperature is not None:
                        command.extend(["--temperature", str(args.temperature)])
                    if args.max_tokens is not None:
                        command.extend(["--max-tokens", str(args.max_tokens)])
                command.extend(ablation.flags)
                runs.append({"run_id": run_id, "command": command})
    return runs


def _budget_sweep_runs(
    args: argparse.Namespace,
    shapes: list[Shape],
    report_dir: Path,
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for shape, generations, population_size, survivors in itertools.product(
        shapes,
        _parse_int_list(args.sweep_generations),
        _parse_int_list(args.sweep_population_sizes),
        _parse_int_list(args.sweep_survivors),
    ):
        if survivors > population_size:
            continue
        run_id = (
            f"{shape.shape_id}_{args.target}_level1_budget_"
            f"g{generations}_p{population_size}_s{survivors}"
        )
        command = _base_command(args, shape, "level1-search", run_id, "level1_budget_sweep")
        command.extend(
            [
                "--generations",
                str(generations),
                "--population-size",
                str(population_size),
                "--survivors",
                str(survivors),
                "--search-num-warmup",
                str(args.search_num_warmup),
                "--search-num-trials",
                str(args.search_num_trials),
                "--evolution-run-dir",
                str(report_dir / "evolution_runs" / run_id),
            ]
        )
        if args.use_bedrock:
            command.append("--use-bedrock")
            if args.bedrock_model_id:
                command.extend(["--bedrock-model-id", args.bedrock_model_id])
            if args.bedrock_region:
                command.extend(["--bedrock-region", args.bedrock_region])
            if args.temperature is not None:
                command.extend(["--temperature", str(args.temperature)])
            if args.max_tokens is not None:
                command.extend(["--max-tokens", str(args.max_tokens)])
        runs.append({"run_id": run_id, "command": command})
    return runs


def _base_command(
    args: argparse.Namespace,
    shape: Shape,
    method: str,
    run_id: str,
    benchmark_group: str,
) -> list[str]:
    return [
        args.python,
        str(RUN_EXPERIMENT),
        "--method",
        method,
        "--target",
        args.target,
        "--M",
        str(shape.M),
        "--N",
        str(shape.N),
        "--K",
        str(shape.K),
        "--num-warmup",
        str(args.num_warmup),
        "--num-trials",
        str(args.num_trials),
        "--output-dir",
        str(args.output_dir),
        "--experiment-id",
        args.experiment_id,
        "--run-id",
        run_id,
        "--run-kind",
        "level1_report",
        "--benchmark-group",
        benchmark_group,
    ]


def _write_summary_tables(report_dir: Path, rows: list[dict[str, str]]) -> None:
    final_rows = [
        row
        for row in rows
        if row.get("benchmark_group") in {"level1_baseline", "final_benchmark"}
        or (
            row.get("benchmark_group") == "level1_ablation"
            and row.get("selection_role", "").startswith("best_of_search")
        )
    ]
    baseline_rows = [
        _summary_row(row)
        for row in final_rows
        if row.get("experiment_method") in BASELINE_METHODS
        or row.get("experiment_method") == "level1-search"
    ]
    _write_csv(report_dir / "baseline_table.csv", baseline_rows)

    ablation_rows = []
    for row in final_rows:
        if row.get("experiment_method") != "level1-search":
            continue
        ablation_rows.append(
            {
                **_summary_row(row),
                "evaluator_feedback_enabled": row.get("evaluator_feedback_enabled"),
                "rejection_cascade_enabled": row.get("rejection_cascade_enabled"),
                "elite_carry_forward_enabled": row.get("elite_carry_forward_enabled"),
                "diverse_inspiration_enabled": row.get("diverse_inspiration_enabled"),
                "num_candidates_evaluated": row.get("num_candidates_evaluated"),
                "num_compile_failures": row.get("num_compile_failures"),
                "num_correct_candidates": row.get("num_correct_candidates"),
                "best_candidate_path": row.get("best_candidate_path"),
                "best_fitness_score": row.get("best_fitness_score"),
            }
        )
    _write_csv(report_dir / "ablation_table.csv", ablation_rows)


def _write_budget_sweep_outputs(report_dir: Path, rows: list[dict[str, str]]) -> None:
    final_rows = [
        row
        for row in rows
        if row.get("benchmark_group") == "level1_budget_sweep"
        and row.get("selection_role", "").startswith("best_of_search")
    ]
    table_rows = []
    for row in final_rows:
        parsed = _parse_budget_run_id(row.get("run_id", ""))
        table_rows.append(
            {
                **parsed,
                **_summary_row(row),
                "num_candidates_evaluated": row.get("num_candidates_evaluated"),
                "num_compile_failures": row.get("num_compile_failures"),
                "num_correct_candidates": row.get("num_correct_candidates"),
                "best_candidate_path": row.get("best_candidate_path"),
                "best_fitness_score": row.get("best_fitness_score"),
            }
        )
    table_rows.sort(
        key=lambda row: (
            row.get("shape", ""),
            int(row.get("generations") or 0),
            int(row.get("population_size") or 0),
            int(row.get("survivors") or 0),
        )
    )
    _write_csv(report_dir / "budget_sweep_table.csv", table_rows)
    _write_budget_sweep_plot(report_dir, table_rows)


def _write_budget_sweep_plot(report_dir: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    points = [
        row
        for row in rows
        if _float_or_none(str(row.get("latency_ms_mean") or "")) is not None
        and _int_or_none(str(row.get("num_candidates_evaluated") or "")) is not None
    ]
    if not points:
        return
    plt.figure(figsize=(7, 4.5))
    for row in points:
        x = int(str(row["num_candidates_evaluated"]))
        y = float(str(row["latency_ms_mean"]))
        label = f"g{row['generations']} p{row['population_size']} s{row['survivors']}"
        plt.scatter([x], [y])
        plt.annotate(label, (x, y), fontsize=7, xytext=(4, 3), textcoords="offset points")
    plt.xlabel("Candidates evaluated")
    plt.ylabel("Final best latency (ms)")
    plt.title("Level 1 Budget Sweep")
    plt.tight_layout()
    plt.savefig(report_dir / "budget_sweep_latency.svg")
    plt.close()


def _summary_row(row: dict[str, str]) -> dict[str, Any]:
    return {
        "run_id": row.get("run_id"),
        "shape": row.get("shape"),
        "target": row.get("target"),
        "method": row.get("experiment_method"),
        "strategy": row.get("strategy"),
        "compile_passed": row.get("compile_passed"),
        "correctness_passed": row.get("correctness_passed"),
        "latency_ms_mean": row.get("latency_ms_mean"),
        "latency_ms_std": row.get("latency_ms_std"),
        "tuning_time_sec": row.get("tuning_time_sec"),
        "evolution_time_sec": row.get("evolution_time_sec"),
        "error_type": row.get("error_type"),
        "error_message": row.get("error_message"),
    }


def _write_generation_plot(report_dir: Path, rows: list[dict[str, str]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    groups: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        if row.get("benchmark_group") != "evolution_search":
            continue
        if row.get("correctness_passed") != "True":
            continue
        latency = _float_or_none(row.get("latency_ms_mean"))
        generation = _int_or_none(row.get("generation"))
        if latency is None or generation is None:
            continue
        key = row.get("run_id", "")
        groups.setdefault(key, []).append((generation, latency))

    if not groups:
        return

    plt.figure(figsize=(8, 4.8))
    for key, points in sorted(groups.items()):
        best_so_far = []
        best = None
        for generation, latency in sorted(points):
            best = latency if best is None else min(best, latency)
            best_so_far.append((generation, best))
        xs, ys = zip(*best_so_far)
        plt.plot(xs, ys, marker="o", label=_short_label(key))
    plt.xlabel("Generation")
    plt.ylabel("Best correct latency so far (ms)")
    plt.title("Level 1 Direct Schedule Search")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(report_dir / "generation_best_latency.svg")
    plt.close()


def _dedup_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    latest: dict[tuple[str, ...], dict[str, str]] = {}
    for index, row in enumerate(rows):
        key = (
            row.get("run_id", ""),
            row.get("benchmark_group", ""),
            row.get("selection_role", ""),
            row.get("generation", ""),
            row.get("candidate_id", ""),
            row.get("experiment_method", ""),
        )
        latest[key] = {**row, "_row_index": str(index)}
    deduped = []
    for row in sorted(latest.values(), key=lambda item: int(item["_row_index"])):
        row.pop("_row_index", None)
        deduped.append(row)
    return deduped


def _write_qualitative_examples(report_dir: Path, rows: list[dict[str, str]]) -> None:
    final_searches = [
        row
        for row in rows
        if row.get("experiment_method") == "level1-search"
        and row.get("selection_role", "").startswith("best_of_search")
    ]
    best = min(
        (row for row in final_searches if _float_or_none(row.get("latency_ms_mean")) is not None),
        key=lambda row: float(row["latency_ms_mean"]),
        default=None,
    )
    failed = next(
        (
            row
            for row in rows
            if row.get("experiment_method") == "level1-search"
            and row.get("compile_passed") != "True"
            and row.get("candidate_path")
        ),
        None,
    )
    lines = ["# Level 1 Qualitative Examples", ""]
    if best:
        lines.extend(_example_section("Best final Level 1 schedule", best.get("best_candidate_path")))
    if failed:
        lines.extend(
            [
                "## Representative failed candidate",
                "",
                f"- candidate_path: {failed.get('candidate_path')}",
                f"- error_type: {failed.get('error_type')}",
                f"- error_message: {failed.get('error_message')}",
                "",
            ]
        )
        lines.extend(_code_excerpt(failed.get("candidate_path")))
    _write_text(report_dir / "qualitative_examples.md", "\n".join(lines).rstrip() + "\n")


def _example_section(title: str, path_value: str | None) -> list[str]:
    lines = ["## " + title, "", f"- path: {path_value}", ""]
    lines.extend(_code_excerpt(path_value))
    return lines


def _code_excerpt(path_value: str | None, max_lines: int = 120) -> list[str]:
    if not path_value:
        return ["(unavailable)", ""]
    path = Path(path_value)
    if not path.exists():
        return [f"(missing: {path})", ""]
    text = "\n".join(path.read_text(encoding="utf-8").splitlines()[:max_lines])
    return ["```python", text, "```", ""]


def _read_results(path: Path, experiment_id: str) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f) if row.get("experiment_id") == experiment_id]


def _completed_run_ids(path: Path, experiment_id: str) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", newline="", encoding="utf-8") as f:
        return {
            row.get("run_id", "")
            for row in csv.DictReader(f)
            if row.get("experiment_id") == experiment_id
        }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _shapes(args: argparse.Namespace) -> list[Shape]:
    raw_shapes = args.shape or ["256,256,256", "8,256,1024", "17,1031,2917"]
    shapes = [_parse_shape(value) for value in raw_shapes]
    if args.only_shape:
        shapes = [shape for shape in shapes if shape.label == args.only_shape]
    if not shapes:
        raise ValueError("No shapes selected.")
    return shapes


def _parse_shape(value: str) -> Shape:
    pieces = [piece.strip() for piece in value.lower().replace("x", ",").split(",") if piece.strip()]
    if len(pieces) != 3:
        raise ValueError(f"Shape must be M,N,K or MxNxK, got {value!r}")
    M, N, K = (int(piece) for piece in pieces)
    if min(M, N, K) <= 0:
        raise ValueError(f"Shape dimensions must be positive, got {value!r}")
    return Shape(M=M, N=N, K=K)


def _parse_int_list(value: str) -> list[int]:
    pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
    if not pieces:
        raise ValueError("Integer list cannot be empty.")
    values = [int(piece) for piece in pieces]
    if any(item <= 0 for item in values):
        raise ValueError(f"Integer list values must be positive: {value!r}")
    return values


def _parse_budget_run_id(run_id: str) -> dict[str, Any]:
    pieces = run_id.split("_")
    parsed = {"generations": "", "population_size": "", "survivors": ""}
    for piece in pieces:
        if len(piece) > 1 and piece[0] == "g" and piece[1:].isdigit():
            parsed["generations"] = piece[1:]
        elif len(piece) > 1 and piece[0] == "p" and piece[1:].isdigit():
            parsed["population_size"] = piece[1:]
        elif len(piece) > 1 and piece[0] == "s" and piece[1:].isdigit():
            parsed["survivors"] = piece[1:]
    return parsed


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    _prepend_env_path(env, "PATH", [REPO_ROOT / "tvm-build-venv" / "bin"])
    _prepend_env_path(env, "PYTHONPATH", [REPO_ROOT / "third_party" / "tvm" / "python"])
    _prepend_env_path(
        env,
        "LD_LIBRARY_PATH",
        [
            REPO_ROOT / "third_party" / "tvm" / "build",
            REPO_ROOT / "third_party" / "tvm" / "build-cuda",
            REPO_ROOT / "third_party" / "tvm" / "build-v011-cuda-shim",
            REPO_ROOT / "third_party" / "tvm" / "build-v011-cuda",
            REPO_ROOT / "third_party" / "tvm" / "build-cuda-llvm17",
        ],
    )
    return env


def _prepend_env_path(env: dict[str, str], key: str, paths: list[Path]) -> None:
    entries = [str(path) for path in paths if path.exists()]
    existing = env.get(key)
    env[key] = os.pathsep.join(entries + ([existing] if existing else []))


def _float_or_none(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _int_or_none(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _short_label(run_id: str) -> str:
    return run_id.replace("_level1_search_", " ").replace("_cuda", "")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
