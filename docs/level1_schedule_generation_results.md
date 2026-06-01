# Level 1 Schedule Generation Results

This note summarizes the CUDA Level 1 direct schedule generation experiments
used for the report. Raw experiment outputs are under `results/`, which is
ignored by git.

## Main Ablation

Experiment id:

```text
level1_report_cuda_all_fixed
```

Artifacts:

```text
results/level1_report/report_artifacts/level1_report_cuda_all_fixed/baseline_table.csv
results/level1_report/report_artifacts/level1_report_cuda_all_fixed/ablation_table.csv
results/level1_report/report_artifacts/level1_report_cuda_all_fixed/generation_best_latency.svg
results/level1_report/report_artifacts/level1_report_cuda_all_fixed/qualitative_examples.md
```

Best correct Level 1 results:

| Shape | Fixed CUDA latency (ms) | Best Level 1 latency (ms) | Best variant |
| --- | ---: | ---: | --- |
| `M256_N256_K256` | `0.0209982` | `0.0197151` | full |
| `M8_N256_K1024` | `0.1168825` | `0.0289178` | diverse inspiration |
| `M17_N1031_K2917` | `0.8433340` | `0.2713596` | elite carry-forward |

Selected ablation details:

| Shape | Variant | Latency mean (ms) | Compile failures | Correct candidates |
| --- | --- | ---: | ---: | ---: |
| `M256_N256_K256` | full | `0.0197151` | `2` | `11` |
| `M256_N256_K256` | no evaluator feedback | `0.0205856` | `1` | `12` |
| `M256_N256_K256` | no rejection cascade | `0.0200442` | `5` | `8` |
| `M256_N256_K256` | elite carry-forward | `0.0206458` | `1` | `12` |
| `M256_N256_K256` | diverse inspiration | `0.0204989` | `8` | `5` |
| `M8_N256_K1024` | full | `0.0291228` | `2` | `11` |
| `M8_N256_K1024` | no evaluator feedback | `0.0290618` | `0` | `13` |
| `M8_N256_K1024` | no rejection cascade | `0.0343320` | `3` | `10` |
| `M8_N256_K1024` | elite carry-forward | `0.0305435` | `1` | `12` |
| `M8_N256_K1024` | diverse inspiration | `0.0289178` | `1` | `12` |
| `M17_N1031_K2917` | full | `0.2828349` | `1` | `12` |
| `M17_N1031_K2917` | no evaluator feedback | `0.8728700` | `0` | `13` |
| `M17_N1031_K2917` | no rejection cascade | `0.1883703` | `0` | `13` |
| `M17_N1031_K2917` | elite carry-forward | `0.2713596` | `0` | `13` |
| `M17_N1031_K2917` | diverse inspiration | `0.2496382` | `0` | `13` |

CUDA MetaSchedule failed for these shapes in the local TVM build during
design-space initialization, so these results compare Level 1 direct schedules
against the fixed CUDA baseline and Level 1 seed, not against a successful
MetaSchedule baseline.

## Budget Sweep

Experiment id:

```text
level1_budget_sweep_cuda_8x256x1024
```

Artifacts:

```text
results/level1_report/report_artifacts/level1_budget_sweep_cuda_8x256x1024/budget_sweep_table.csv
results/level1_report/report_artifacts/level1_budget_sweep_cuda_8x256x1024/budget_sweep_latency.svg
results/level1_report/report_artifacts/level1_budget_sweep_cuda_8x256x1024/generation_best_latency.svg
```

Best successful budget:

| Generations | Population | Survivors | Candidates | Latency mean (ms) | Latency std (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `3` | `2` | `1` | `7` | `0.0291259` | `0.0002970` |

Full budget ranking:

| Generations | Population | Survivors | Candidates | Latency mean (ms) |
| ---: | ---: | ---: | ---: | ---: |
| `3` | `2` | `1` | `7` | `0.0291259` |
| `1` | `2` | `1` | `3` | `0.0291961` |
| `5` | `4` | `1` | `21` | `0.0292827` |
| `1` | `2` | `2` | `3` | `0.0294041` |
| `3` | `4` | `2` | `13` | `0.0299291` |
| `5` | `2` | `2` | `11` | `0.0309851` |
| `5` | `2` | `1` | `11` | `0.0312829` |
| `3` | `2` | `2` | `7` | `0.0347963` |
| `5` | `4` | `2` | `21` | `0.0284154` |
| `1` | `4` | `1` | `5` | `0.0302783` |
| `1` | `4` | `2` | `5` | `0.0619068` |
| `3` | `4` | `1` | `13` | `0.0358267` |

The budget sweep suggests that the `8x256x1024` shape reaches the fast schedule
family with a small search budget. Larger budgets did not improve final latency
in this run and added more model calls. The best measured config was
`generations=3`, `population_size=2`, `survivors=1`.
