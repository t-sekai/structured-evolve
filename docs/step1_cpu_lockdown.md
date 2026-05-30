# Step 1 CPU Lockdown Notes

## Goal

Debug and validate Level 1 direct schedule generation on CPU before moving to GPU.

## Root Cause Fixed

The checkpoint Level 1 candidates failed because they were generated against a
different TVM schedule API shape:

- Generated candidates used `sch.get_block("C")`; this local TVM build exposes
  `sch.get_sblock("C", func_name="main")`.
- Generated candidates often used `sch.split(loop, factor=16)`; this local TVM
  build expects `sch.split(loop, factors=[None, 16])`.
- The previous `tvm.script` matmul IR was also not reliably schedulable by the
  local `s_tir` implementation. Rebuilding the same matmul through
  `te.create_prim_func` gives schedule transforms valid block/loop metadata.
- Real model candidates also made a few common API-shape mistakes such as
  `return sch.mod()`, passing `factor=` to `vectorize`, or stacking multiple
  nested CPU vectorization/parallel annotations. The Level 1 generated-schedule
  loader now normalizes/guards those cases so candidate evaluation can continue
  with a valid partial schedule.

## Local Verification Command

```bash
.venv/bin/python scripts/run_benchmark_suite.py \
  --suite-name step1_cpu_local_lockdown \
  --experiment-id step1_cpu_local_lockdown \
  --output-dir results/step1_cpu_lockdown \
  --shape 256,256,256 \
  --shape 8,256,1024 \
  --shape 17,1031,2917 \
  --target llvm \
  --methods fixed,level1-candidate,level1-search \
  --generations 1 \
  --population-size 2 \
  --survivors 1 \
  --search-num-warmup 1 \
  --search-num-trials 2 \
  --num-warmup 1 \
  --num-trials 3
```

## Results

All 9 final benchmark runs compiled and passed correctness.

| Shape | Method | Latency Mean (ms) | Speedup vs Fixed | Compile | Correct |
| --- | --- | ---: | ---: | --- | --- |
| 256x256x256 | fixed | 101.774 | 1.00x | pass | pass |
| 256x256x256 | level1-candidate | 111.502 | 0.91x | pass | pass |
| 256x256x256 | level1-search | 87.022 | 1.17x | pass | pass |
| 8x256x1024 | fixed | 28.216 | 1.00x | pass | pass |
| 8x256x1024 | level1-candidate | 16.709 | 1.69x | pass | pass |
| 8x256x1024 | level1-search | 24.263 | 1.16x | pass | pass |
| 17x1031x2917 | fixed | 859.886 | 1.00x | pass | pass |
| 17x1031x2917 | level1-candidate | 780.703 | 1.10x | pass | pass |
| 17x1031x2917 | level1-search | 938.546 | 0.92x | pass | pass |

The Level 1 search histories each evaluated 3 candidates and had 0 compile
failures.

## Remaining External-Model Step

A real Bedrock validation run was completed after explicit approval to send
project-derived prompts and schedule candidate code to AWS Bedrock.

Command pattern:

```bash
.venv/bin/python scripts/run_experiment.py \
  --method level1-search \
  --use-bedrock \
  --target llvm \
  --M <M> --N <N> --K <K> \
  --generations 1 \
  --population-size 2 \
  --survivors 1 \
  --search-num-warmup 1 \
  --search-num-trials 2 \
  --num-warmup 1 \
  --num-trials 3 \
  --output-dir results/step1_cpu_lockdown_bedrock_final \
  --suite-name step1_cpu_bedrock_final \
  --temperature 0.2 \
  --max-tokens 2048
```

Final real-model validation results:

| Shape | Best Candidate | Final Latency Mean (ms) | Compile | Correct |
| --- | --- | ---: | --- | --- |
| 256x256x256 | `gen_001/candidate_000.py` | 8.263 | pass | pass |
| 8x256x1024 | `gen_001/candidate_000.py` | 0.513 | pass | pass |
| 17x1031x2917 | `gen_001/candidate_001.py` | 64.668 | pass | pass |

Candidate-level reliability:

| Shape | Candidates Evaluated | Compile Failures | Correctness Failures |
| --- | ---: | ---: | ---: |
| 256x256x256 | 3 | 0 | 0 |
| 8x256x1024 | 3 | 0 | 0 |
| 17x1031x2917 | 3 | 0 | 0 |
