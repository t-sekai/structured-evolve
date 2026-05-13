# Checkpoint 1 Report

## Project Question and Goals

This project asks where LLM evolution should intervene in a structured ML kernel scheduling pipeline. The core hypothesis is that LLMs may be more useful when they generate or evolve structured search spaces for TVM MetaSchedule, instead of directly emitting a single schedule.

The planned comparison is:

1. Default TVM MetaSchedule baseline
2. Level 1: direct LLM-generated TVM schedule code
3. Level 2: LLM-generated search spaces refined by TVM MetaSchedule
4. Level 3: LLM-evolved search-space generators refined by TVM MetaSchedule

GPT 5.5 was limitedly used during this checkpoint for the write up and for code polishing.

## Experiments Planned

The first real experiments will use matrix multiplication (GEMM) as the initial workload and compare correctness, measured latency, and the ability of each method to produce valid TVM programs. Later experiments can add more ML kernels and larger shape sets.

Planned stages:

1. Establish a default TVM MetaSchedule baseline.
2. Add direct LLM schedule generation for the same TensorIR workload.
3. Add LLM-generated search spaces that TVM MetaSchedule can tune.
4. Add OpenEvolve to evolve reusable search-space generator programs.

## Success Criteria

For each scheduling strategy, a run is successful if it:

- Produces a TVM program that compiles for the requested target.
- Produces numerically correct output against a NumPy reference.
- Records latency using a consistent timing path.
- Saves structured results that can be compared across methods.

The longer-term success criterion is whether Level 2 or Level 3 can produce faster or more robust schedules than direct LLM schedule generation while preserving correctness.

## Current Evaluation Code

Checkpoint 1 implements a minimal runnable baseline. It currently:

- Generates random float32 inputs for matrix multiplication.
- Computes a NumPy reference output.
- Defines a simple TensorIR matmul kernel.
- Compiles the kernel with Apache TVM for `llvm` or `cuda`.
- Runs the compiled kernel.
- Checks correctness with NumPy tolerance-based comparison.
- Measures latency with TVM's `time_evaluator`.
- Writes one JSON file per run and appends a CSV summary.
- Supports `--bad-baseline` to intentionally check an all-zero output and demonstrate failure reporting.

This checkpoint does not implement OpenEvolve, LLM generation, or TVM MetaSchedule tuning.

## Code and Result Files

- `scripts/run_baseline.py`: command-line entry point for compile, run, check, benchmark, and save.
- `scripts/run_all.sh`: small helper that runs the LLVM baseline and runs CUDA when available.
- `src/kernels/matmul_tir.py`: TensorIR matmul definition and minimal target schedule hook.
- `src/eval/benchmark.py`: TVM build, run, device selection, and timing helpers.
- `src/eval/correctness.py`: NumPy reference comparison helpers.
- `src/eval/results_io.py`: JSON and CSV result writing.
- `results/`: output directory for generated benchmark files.

## Results 

We tested our TVM baseline on a NVIDIA RTX A5000 with AMD EPYC 7402 24-Core Processor and 64GB RAM. Matrix multiplication results for CPU and GPU runs are stored in the ```results_ckpt1``` folder.

## What Remains

Next work should add:

- A TVM MetaSchedule baseline for the same workload and shapes.
- A clean interface for plugging in generated schedule candidates.
- A representation for LLM-generated search spaces.
- An OpenEvolve loop for evolving search-space generator code.
- More workloads, shape sets, and repeated evaluation runs.
