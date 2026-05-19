# Structured Evolution for Search-Space Generation in ML Kernel Scheduling
Kevin Chan (tsekchan@stanford.edu), Newton Chen (hsinchen@stanford.edu)

## Summary

We are going to build and evaluate a structured LLM-guided system for generating faster ML kernels by combining OpenEvolve-style evolutionary search with Apache TVM MetaSchedule. We will demonstrate success with benchmark plots comparing three levels of LLM intervention: direct kernel/schedule generation, LLM-generated search spaces refined by TVM's autotuner, and LLM-evolved search-space generators refined by TVM's autotuner. The goal is to study whether LLMs are most effective as kernel writers, search-space designers, or search-space-generator designers, while keeping TVM MetaSchedule as a strong, structured baseline.

## Project Proposal

[View Proposal](CS348K_Project_Proposal.pdf)

## First Checkpoint (Week 6)

[View Checkpoint](docs/checkpoint_1.md)

## Install

First follow the TVM guide to install TVM from source: [TVM Source Install Guide](https://tvm.apache.org/docs/install/from_source.html#install-from-source)

With your built TVM in `tvm-build-venv`:

```bash
python -c "import tvm; print(tvm.__version__)"
pip install -r requirements.txt
```


## Example Commands

Run the fixed CPU baseline from checkpoint 1:

```bash
python scripts/run_baseline.py --target llvm
```

Run the fixed CUDA baseline from checkpoint 1:

```bash
python scripts/run_baseline.py --target cuda
```

Run a small TVM MetaSchedule CPU baseline:

```bash
python scripts/run_baseline.py --strategy metaschedule --target llvm --M 128 --N 128 --K 128 --max-trials-global 16 --num-trials-per-iter 4 --cost-model random --num-tuning-cores 1
```

Run a Level 1 direct schedule candidate:

```bash
python scripts/run_baseline.py --strategy generated-schedule --generated-schedule-path generated/schedules/identity.py --target llvm
```

Generated schedule candidates are Python files that define:

```python
def apply_schedule(ir_module: tvm.IRModule, target_name: str) -> tvm.IRModule:
    ...
```

Run a dry-run Level 1 OpenEvolve-style schedule evolution loop:

```bash
python scripts/evolve_schedules.py --dry-run --generations 1 --population-size 2 --survivors 1 --M 16 --N 16 --K 16 --num-warmup 1 --num-trials 2
```

Run the same loop with AWS Bedrock for candidate mutations:

```bash
cp config/api_keys.example.toml config/api_keys.toml
# Fill in config/api_keys.toml with your AWS credentials and Bedrock model id.

python scripts/evolve_schedules.py --use-bedrock --generations 2 --population-size 4 --survivors 2 --target llvm
```

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
python scripts/run_baseline.py --strategy generated-search-space --generated-search-space-path generated/search_spaces/basic_matmul.py --target llvm --M 16 --N 16 --K 16 --max-trials-global 4 --num-trials-per-iter 2 --cost-model random --task-scheduler round-robin --num-tuning-cores 1
```

Run a dry-run Level 2 OpenEvolve-style search-space evolution loop:

```bash
python scripts/evolve_search_spaces.py --dry-run --generations 1 --population-size 1 --survivors 1 --M 16 --N 16 --K 16 --max-trials-global 4 --num-trials-per-iter 2 --num-tuning-cores 1
```

For real comparisons, use the default `--cost-model xgb`, increase
`--max-trials-global`, and keep the same shape, target, warmup, and timing
settings across strategies.

Demonstrate a correctness failure:

```bash
python scripts/run_baseline.py --target cuda --bad-baseline
```

Run the small baseline suite:

```bash
bash scripts/run_all.sh
```

## Metrics

Each run reports and saves:

- `correctness_passed`: whether the TVM output matches the NumPy reference
- `max_abs_error`: maximum absolute elementwise error
- `mean_abs_error`: mean absolute elementwise error
- `latency_ms_mean`: mean runtime in milliseconds across timed trials
- `latency_ms_std`: standard deviation of runtime in milliseconds across timed trials

JSON files are written per run. A shared `results.csv` is created or appended in the output directory.
