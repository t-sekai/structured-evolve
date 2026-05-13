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

Run the CPU baseline:

```bash
python scripts/run_baseline.py --target llvm
```

Run the CUDA baseline:

```bash
python scripts/run_baseline.py --target cuda
```

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