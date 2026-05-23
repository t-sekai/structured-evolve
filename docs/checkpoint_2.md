# Checkpoint 2 Report

## What Changed Since Checkpoint 1

Checkpoint 1 established a runnable TVM matmul evaluation baseline. Since then, we implemented search/evolution methods into our codebase.

The current user-facing scripts are:

- `scripts/run_experiment.py`: run one method on one matmul task.
- `scripts/run_benchmark_suite.py`: run many methods across many shapes/targets.
- `scripts/analyze_results.py`: create summary tables and SVG figures from
  `results/results.csv`.

The current experiment methods are:

- `fixed`: checkpoint-1 fixed schedule baseline.
- `metaschedule`: TVM MetaSchedule baseline.
- `level1-candidate`: benchmark one existing direct schedule candidate.
- `level2-candidate`: benchmark one existing generated search-space candidate.
- `level1-search`: search over direct schedule candidates, then benchmark the
  best candidate.
- `level2-search`: search over generated search-space candidates, then benchmark
  the best candidate.

## Current Scope

For this checkpoint, Level 1 and Level 2 are CPU-only. CUDA support for these
methods is still in progress. CPU results should be generated with `--target llvm`.

The current workload is TensorIR matrix multiplication with configurable
`M`, `N`, and `K`. The evaluation checks:

- Compilation success.
- Numerical correctness against a NumPy reference.
- Mean and standard deviation of runtime latency.
- TVM MetaSchedule tuning time where applicable.
- Search/evolution wall-clock time and candidate counts where applicable.

## Results

We ran a preliminary CPU search-vs-baseline suite:

```bash
python scripts/run_benchmark_suite.py \
  --experiment-id ckpt2_cpu_search_vs_baseline \
  --suite-name checkpoint_2_cpu_search_vs_baseline \
  --shape 64,64,64 \
  --target llvm \
  --methods fixed,metaschedule,level1-search,level2-search \
  --generations 2 \
  --population-size 4 \
  --survivors 2 \
  --search-num-warmup 1 \
  --search-num-trials 3 \
  --num-warmup 3 \
  --num-trials 10 \
  --max-trials-global 8 \
  --num-trials-per-iter 4 \
  --cost-model random \
  --task-scheduler round-robin \
  --num-tuning-cores 1
```

These results should be treated as preliminary. They seem suspicious and likely
reflect a bug or mismatch in the current search/evaluation path rather than a
reliable conclusion about the methods. We are investigating the issue before
using these numbers for final claims.

### CPU Candidate and Baseline Comparison

| Method | Shape | Target | Correct? | Latency Mean (ms) | Latency Std (ms) | Tuning Time (s) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fixed | 64x64x64 | llvm | Yes | 0.1192 | 0.0017 | 0.0 | Baseline fixed schedule. |
| metaschedule | 64x64x64 | llvm | Yes | 0.0368 | 0.0048 | 21.44 | TVM MetaSchedule baseline with a small random-search budget. |

### CPU Search Results

| Method | Shape | Target | Candidates Evaluated | Best Candidate | Search Time (s) | Final Latency Mean (ms) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| level1-search | 64x64x64 | llvm | 9 | `g000_c000_seed` | 34.56 | 0.2224 | Suspicious: best candidate is the seed, 8/9 candidates failed compilation, and final latency is worse than the fixed baseline. |
| level2-search | 64x64x64 | llvm | 9 | `g001_c001` | 143.72 | 0.0558 | Suspicious: all candidates passed, but the reported final latency is worse than the best search-time candidate and worse than MetaSchedule. |

### Figures

Generated analysis artifacts are under
`results/analysis/ckpt2_cpu_search_vs_baseline/`.

### Preliminary Interpretation

The current numbers are not yet trustworthy. The suspicious signs are:

- Level 1 search mostly generates candidates that fail to compile, so the best
  candidate remains the seed.
- Level 1 final benchmark latency is worse than the fixed baseline even though
  the seed should behave similarly to a simple direct schedule candidate.
- Level 2 search reports strong search-time fitness for some candidates, but the
  final benchmark of the selected best candidate is slower than the MetaSchedule
  baseline.
- The analysis summary currently mixes search-time candidate rows and final
  benchmark rows unless filtered carefully, which may make aggregate plots or
  averages misleading.

Because of these issues, we are using this run mainly as evidence that the
pipeline is producing intermediate artifacts and structured results, not yet as
evidence that either search method improves performance.

## Next Steps

1. Add CUDA support for Level 1 and Level 2 methods.
2. Debug the suspicious CPU results before making performance claims.
3. Re-run the CPU commands above after fixing the suspected measurement/search
   issue.
4. Increase search budget if early search runs do not produce meaningful
   candidate diversity.
5. Add more matmul shapes and later expand to additional kernels.
6. Use the generated analysis tables/figures to compare latency, correctness,
   tuning time, search time, and failure rates across methods.
