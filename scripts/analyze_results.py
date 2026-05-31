#!/usr/bin/env python3
"""Create analysis tables and matplotlib figures from results/results.csv."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize benchmark and evolution results for analysis."
    )
    parser.add_argument("--results-csv", type=Path, default=Path("results/results.csv"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--suite-name", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--baseline-strategy",
        default="fixed",
        help="Strategy used as the speedup denominator.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if not args.results_csv.exists():
            raise FileNotFoundError(f"Results CSV not found: {args.results_csv}")

        output_dir = args.output_dir or _default_output_dir(args.results_csv.parent)
        output_dir.mkdir(parents=True, exist_ok=True)

        df = _load_results(args.results_csv)
        df = _apply_filters(
            df,
            experiment_id=args.experiment_id,
            suite_name=args.suite_name,
            run_id=args.run_id,
        )
        if df.empty:
            raise ValueError("No rows left after applying filters.")

        benchmark_df = _final_benchmark_rows(df)

        summary = _summary_by_strategy(benchmark_df)
        summary_path = output_dir / "summary_by_strategy.csv"
        summary.to_csv(summary_path, index=False)

        speedups = _speedups(benchmark_df, baseline_strategy=args.baseline_strategy)
        speedups_path = output_dir / "speedups.csv"
        speedups.to_csv(speedups_path, index=False)

        evolution = _evolution_best_by_generation(df)
        evolution_path = output_dir / "evolution_best_by_generation.csv"
        evolution.to_csv(evolution_path, index=False)

        figures_dir = output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        _write_latency_figures(speedups, figures_dir)
        _write_evolution_figure(evolution, figures_dir / "evolution_fitness.svg")

        print(f"analysis_dir: {output_dir}")
        print(f"summary: {summary_path}")
        print(f"speedups: {speedups_path}")
        print(f"evolution: {evolution_path}")
        return 0
    except Exception as err:  # pylint: disable=broad-except
        print(f"ERROR: {err}", file=sys.stderr)
        return 1


def _load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for column in (
        "M",
        "N",
        "K",
        "problem_size",
        "generation",
        "fitness_score",
        "latency_ms_mean",
        "latency_ms_std",
        "tuning_time_sec",
        "max_trials_global",
        "num_trials_per_iter",
    ):
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in ("compile_passed", "correctness_passed", "bad_baseline"):
        if column in df:
            df[column] = df[column].map(_to_bool)
    if "shape" not in df and {"M", "N", "K"}.issubset(df.columns):
        df["shape"] = df.apply(
            lambda row: f"M{int(row.M)}_N{int(row.N)}_K{int(row.K)}",
            axis=1,
        )
    return df


def _apply_filters(
    df: pd.DataFrame,
    *,
    experiment_id: str | None,
    suite_name: str | None,
    run_id: str | None,
) -> pd.DataFrame:
    filtered = df
    for column, value in (
        ("experiment_id", experiment_id),
        ("suite_name", suite_name),
        ("run_id", run_id),
    ):
        if value is not None and column in filtered:
            filtered = filtered[filtered[column] == value]
    return filtered.copy()


def _final_benchmark_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep final benchmark rows separate from search-time candidate evaluations."""
    if "benchmark_group" not in df:
        return df.copy()
    final_rows = df[df["benchmark_group"] == "final_benchmark"].copy()
    if final_rows.empty:
        return df.copy()
    return final_rows


def _summary_by_strategy(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        column
        for column in ("target", "shape", "experiment_method", "strategy", "level")
        if column in df
    ]
    grouped = df.groupby(group_cols, dropna=False)
    summary = grouped.agg(
        runs=("strategy", "size"),
        compile_rate=("compile_passed", "mean"),
        correctness_rate=("correctness_passed", "mean"),
        latency_ms_mean=("latency_ms_mean", "mean"),
        latency_ms_median=("latency_ms_mean", "median"),
        latency_ms_best=("latency_ms_mean", "min"),
        latency_ms_std=("latency_ms_mean", "std"),
        tuning_time_sec_mean=("tuning_time_sec", "mean"),
        evolution_time_sec_mean=("evolution_time_sec", "mean"),
    )
    return summary.reset_index()


def _speedups(df: pd.DataFrame, *, baseline_strategy: str) -> pd.DataFrame:
    compile_mask = _bool_mask(df, "compile_passed", default=True)
    correctness_mask = _bool_mask(df, "correctness_passed", default=True)
    valid = df[compile_mask & correctness_mask & df["latency_ms_mean"].notna()].copy()
    if valid.empty:
        return pd.DataFrame()

    group_cols = [
        column
        for column in ("target", "shape", "experiment_method", "strategy", "level")
        if column in valid
    ]
    grouped = (
        valid.groupby(group_cols, dropna=False)
        .agg(
            latency_ms_mean=("latency_ms_mean", "mean"),
            latency_ms_best=("latency_ms_mean", "min"),
            tuning_time_sec_mean=("tuning_time_sec", "mean"),
            evolution_time_sec_mean=("evolution_time_sec", "mean"),
            runs=("strategy", "size"),
        )
        .reset_index()
    )
    method_col = "experiment_method" if "experiment_method" in grouped else "strategy"
    baseline = grouped[grouped[method_col].fillna(grouped["strategy"]) == baseline_strategy][
        ["target", "shape", "latency_ms_mean"]
    ].rename(columns={"latency_ms_mean": "baseline_latency_ms_mean"})
    result = grouped.merge(baseline, on=["target", "shape"], how="left")
    result["speedup_vs_baseline"] = (
        result["baseline_latency_ms_mean"] / result["latency_ms_mean"]
    )
    return result


def _evolution_best_by_generation(df: pd.DataFrame) -> pd.DataFrame:
    required = {"run_id", "generation", "fitness_score"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    evolution = df[df["generation"].notna() & df["fitness_score"].notna()].copy()
    if evolution.empty:
        return pd.DataFrame()
    if "experiment_method" not in evolution:
        evolution["experiment_method"] = evolution.apply(_method_from_evolution_row, axis=1)
    else:
        missing_method = evolution["experiment_method"].isna() | (
            evolution["experiment_method"].astype(str).str.strip() == ""
        )
        evolution.loc[missing_method, "experiment_method"] = evolution.loc[
            missing_method
        ].apply(_method_from_evolution_row, axis=1)
    group_cols = [
        column
        for column in (
            "run_id",
            "experiment_method",
            "evolution_kind",
            "target",
            "shape",
            "generation",
        )
        if column in evolution
    ]
    grouped = evolution.groupby(group_cols, dropna=False)
    return grouped.agg(
        best_fitness_score=("fitness_score", "max"),
        best_latency_ms=("latency_ms_mean", "min"),
        candidates=("candidate_id", "nunique"),
        correctness_rate=("correctness_passed", "mean"),
        compile_rate=("compile_passed", "mean"),
    ).reset_index()


def _method_from_evolution_row(row: pd.Series) -> str:
    run_id = str(row.get("run_id", ""))
    evolution_kind = str(row.get("evolution_kind", ""))
    if "level1-search" in run_id or "level_1" in evolution_kind:
        return "level1-search"
    if "level2-search" in run_id or "level_2" in evolution_kind:
        return "level2-search"
    return "search"


def _write_latency_figures(speedups: pd.DataFrame, figures_dir: Path) -> None:
    if speedups.empty:
        return
    plt = _matplotlib_pyplot()
    for (target, shape), rows in speedups.groupby(["target", "shape"], dropna=False):
        plot_rows = rows[rows["latency_ms_mean"].notna()].copy()
        if plot_rows.empty:
            continue
        plot_rows["label"] = plot_rows.apply(_row_label, axis=1)
        plot_rows = plot_rows.sort_values("latency_ms_mean", ascending=True)

        fig_height = max(3.2, 0.45 * len(plot_rows) + 1.3)
        fig, ax = plt.subplots(figsize=(8.0, fig_height), constrained_layout=True)
        ax.barh(plot_rows["label"], plot_rows["latency_ms_mean"], color="#2563eb")
        ax.set_title(f"Mean Latency by Method ({target}, {shape})")
        ax.set_xlabel("Mean latency (ms)")
        ax.set_ylabel("Experiment method")
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        for index, value in enumerate(plot_rows["latency_ms_mean"]):
            ax.text(value, index, f" {value:.4g}", va="center", fontsize=8)

        filename = f"latency_{_safe_name(str(target))}_{_safe_name(str(shape))}.svg"
        fig.savefig(figures_dir / filename, format="svg")
        plt.close(fig)


def _write_evolution_figure(evolution: pd.DataFrame, path: Path) -> None:
    if evolution.empty:
        return
    plt = _matplotlib_pyplot()
    plot_rows = evolution[
        evolution["generation"].notna() & evolution["best_fitness_score"].notna()
    ].copy()
    if plot_rows.empty:
        return

    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    label_cols = [
        column
        for column in ("experiment_method", "target", "shape")
        if column in plot_rows
    ]
    if label_cols:
        grouped = plot_rows.groupby(label_cols, dropna=False)
    else:
        grouped = [("all", plot_rows)]

    for label_values, rows in grouped:
        rows = rows.sort_values("generation")
        label = _series_label(label_cols, label_values)
        ax.plot(
            rows["generation"],
            rows["best_fitness_score"],
            marker="o",
            linewidth=1.8,
            label=label,
        )

    ax.set_title("Best Search Fitness by Generation")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness score (1 / latency for correct candidates)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.xaxis.get_major_locator().set_params(integer=True)
    if len(plot_rows) > 1 and label_cols:
        ax.legend(
            fontsize=8,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=False,
        )
    fig.savefig(path, format="svg")
    plt.close(fig)


def _to_bool(value: object) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no"):
        return False
    return None


def _row_label(row: pd.Series) -> str:
    method = row.get("experiment_method")
    if method is not None and not pd.isna(method):
        return str(method)
    return str(row.get("strategy"))


def _series_label(columns: list[str], values: object) -> str:
    if not columns:
        return "all"
    if len(columns) == 1:
        values = (values,)
    pieces = []
    for column, value in zip(columns, values):
        if pd.isna(value):
            continue
        pieces.append(_compact_label_piece(column, value))
    return ", ".join(pieces) if pieces else "all"


def _compact_label_piece(column: str, value: object) -> str:
    text = str(value)
    if column == "experiment_method":
        return text
    if column == "target":
        return text
    if column == "shape":
        return text.replace("_", " ")
    return text


def _bool_mask(df: pd.DataFrame, column: str, *, default: bool) -> pd.Series:
    if column not in df:
        return pd.Series(default, index=df.index)
    return df[column].fillna(False).astype(bool)


def _default_output_dir(results_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return results_dir / "analysis" / timestamp


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def _matplotlib_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise ImportError(
            "matplotlib is required to generate figures. Install project "
            "dependencies with `pip install -r requirements.txt`."
        ) from err
    return plt


if __name__ == "__main__":
    raise SystemExit(main())
