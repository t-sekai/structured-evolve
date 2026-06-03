# Structured Evolution for Search-Space Generation in ML Kernel Scheduling
Kevin Chan (tsekchan@stanford.edu), Newton Chen (hsinchen@stanford.edu)

## Abstract

We study where a zero-shot large language model should intervene in an ML kernel scheduling pipeline, and whether its flexibility and expressiveness can outperform a rule-based autotuning baseline, Apache TVM MetaSchedule. Rather than asking an LLM only to emit one concrete low-level schedule, we compare that direct approach with a structured alternative in which the LLM generates a MetaSchedule design space and TVM searches within it. We build this comparison on fixed TensorIR matrix-multiplication kernels, automatic syntax/compile/correctness/latency gates, and an OpenEvolve-style mutation-and-selection loop over generated programs. Our results do not support the hypothesis that LLM-generated search spaces consistently outperform direct schedule generation or standard MetaSchedule: MetaSchedule remains strongest on regular matrix multiplication, direct schedule evolution is less reliable but finds large speedups on irregular shapes where MetaSchedule fails, and search-space evolution is more conservative and sensitive to prompt, seed, and feedback quality. These findings suggest that the main promise of LLM-guided scheduling is not replacing mature autotuners on their best cases, but expanding scheduling coverage for irregular or underspecified workloads. 

## Project Proposal

[View Proposal](CS348K_Project_Proposal.pdf)

## First Checkpoint (Week 6)

[View Checkpoint](docs/checkpoint_1.md)

## Second Checkpoint (Week 8)

[View Checkpoint](docs/checkpoint_2.md)

## Final Report (Week 10)

[View Report](structured_evolve_final_report.pdf)

## Install

First follow the TVM guide to install TVM from source: [TVM Source Install Guide](https://tvm.apache.org/docs/install/from_source.html#install-from-source)

With your built TVM in `tvm-build-venv`:

```bash
python -c "import tvm; print(tvm.__version__)"
pip install -r requirements.txt
```


## How to Use This Repo

The user-facing workflow has three scripts:

| Script | Purpose |
| --- | --- |
| `scripts/run_experiment.py` | Run one experiment method on one matmul task. Search methods generate candidates first, then benchmark the best candidate. |
| `scripts/run_benchmark_suite.py` | Run many methods across many shapes/targets with shared metadata. |
| `scripts/analyze_results.py` | Read `results/results.csv` and write summary tables and SVG figures. |

The code path is:

```text
scripts
  -> src/eval/method.py       # first-class methods: baselines, candidates, search
  -> src/eval/experiment.py   # shared compile/correctness/timing/result pipeline
  -> src/eval/benchmark.py    # TVM build/run/timing helpers
  -> src/strategies/*         # pluggable scheduling/tuning implementations
  -> src/eval/results_io.py   # JSON + CSV persistence
```

Available experiment methods:

| Method | What It Does |
| --- | --- |
| `fixed` | Checkpoint-1 hand-written schedule baseline. |
| `metaschedule` | TVM MetaSchedule baseline. |
| `level1-candidate` | Benchmark one existing Level 1 schedule file. |
| `level2-candidate` | Benchmark one existing Level 2 search-space file refined by MetaSchedule. |
| `level1-search` | Search over direct schedule candidates, then benchmark the best candidate. |
| `level2-search` | Search over generated search-space candidates, then benchmark the best candidate. |

## Single Experiments

Run the fixed CPU baseline:

```bash
python scripts/run_experiment.py --target llvm
```

Run the fixed CUDA baseline from checkpoint 1:

```bash
python scripts/run_experiment.py --target cuda
```

Run a small TVM MetaSchedule CPU baseline:

```bash
python scripts/run_experiment.py --method metaschedule --target llvm --M 128 --N 128 --K 128 --max-trials-global 16 --num-trials-per-iter 4 --cost-model random --num-tuning-cores 1
```

Run a Level 1 direct schedule candidate:

```bash
python scripts/run_experiment.py --method level1-candidate --generated-schedule-path generated/schedules/identity.py --target llvm
```

Generated schedule candidates are Python files that define:

```python
def apply_schedule(ir_module: tvm.IRModule, target_name: str) -> tvm.IRModule:
    ...
```

Run a dry-run Level 1 search experiment:

```bash
python scripts/run_experiment.py --method level1-search --generations 1 --population-size 2 --survivors 1 --M 16 --N 16 --K 16 --num-warmup 1 --num-trials 2
```

Run the same Level 1 method with AWS Bedrock for candidate mutations:

```bash
cp config/api_keys.example.toml config/api_keys.toml
# Fill in config/api_keys.toml with your AWS credentials and Bedrock model id.

python scripts/run_experiment.py --method level1-search --use-bedrock --generations 2 --population-size 4 --survivors 2 --target llvm
```

Search artifacts are written under `results/evolution_runs/...` by default.
Search-time candidate measurements and final best-candidate measurements are
both appended to the shared `results/results.csv`.

For Bedrock API-key auth, put the token in the ignored local config as:

```toml
[aws]
bedrock_bearer_token = "ABSK..."
region = "us-west-2"
```

Command-line flags like `--bedrock-model-id`, `--bedrock-region`,
`--temperature`, and `--max-tokens` override values in `config/api_keys.toml`.

The Bedrock wrapper expects a chat-style model invocation. If a specific
Bedrock model id uses a provider-specific request schema, adjust
`src/evolution/bedrock_client.py`.

Run a Level 2 generated search-space candidate refined by TVM MetaSchedule:

```bash
python scripts/run_experiment.py --method level2-candidate --generated-search-space-path generated/search_spaces/basic_matmul.py --target llvm --M 16 --N 16 --K 16 --max-trials-global 4 --num-trials-per-iter 2 --cost-model random --task-scheduler round-robin --num-tuning-cores 1
```

For `--target cuda`, Level 2 candidate and search methods default to
`generated/search_spaces/cuda_matmul.py`. LLVM defaults to
`generated/search_spaces/basic_matmul.py`. Pass an explicit path to override the
target-specific seed.

Run a dry-run Level 2 search experiment:

```bash
python scripts/run_experiment.py --method level2-search --generations 1 --population-size 1 --survivors 1 --M 16 --N 16 --K 16 --max-trials-global 4 --num-trials-per-iter 2 --num-tuning-cores 1
```

For real comparisons, use the default `--cost-model xgb`, increase
`--max-trials-global`, and keep the same shape, target, warmup, and timing
settings across methods.

Demonstrate a correctness failure:

```bash
python scripts/run_experiment.py --target cuda --bad-baseline
```

## Benchmark Suites

Run a benchmark matrix for checkpoint-style comparisons:

```bash
python scripts/run_benchmark_suite.py \
  --shape 128,128,128 \
  --shape 256,256,256 \
  --target llvm \
  --methods fixed,metaschedule,level1-candidate,level2-candidate \
  --max-trials-global 16 \
  --num-trials-per-iter 4 \
  --cost-model random \
  --num-tuning-cores 1
```

Include search methods when you want end-to-end search time in the same
experiment table:

```bash
python scripts/run_benchmark_suite.py \
  --shape 128,128,128 \
  --target llvm \
  --methods fixed,metaschedule,level1-search,level2-search \
  --generations 2 \
  --population-size 4 \
  --survivors 2
```

## Analysis

Create analysis tables and SVG figures from the shared CSV:

```bash
python scripts/analyze_results.py --results-csv results/results.csv
```

## Metrics

Each run reports and saves:

- `correctness_passed`: whether the TVM output matches the NumPy reference
- `max_abs_error`: maximum absolute elementwise error
- `mean_abs_error`: mean absolute elementwise error
- `latency_ms_mean`: mean runtime in milliseconds across timed trials
- `latency_ms_std`: standard deviation of runtime in milliseconds across timed trials
- `tuning_time_sec`: TVM MetaSchedule tuning time where applicable
- `evolution_time_sec`: end-to-end search time for search methods
- `num_candidates_evaluated`: number of candidates evaluated by a search method
- `best_candidate_path`: artifact path for the best evolved candidate

JSON files are written per run. A shared `results.csv` is created or appended
in the output directory. Runs add analysis metadata such as `experiment_id`,
`run_id`, `experiment_method`, `generation`, `candidate_id`, `fitness_score`,
and `selection_role` so the same CSV can drive tables and plots.
