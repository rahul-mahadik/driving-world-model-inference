# Results

## Operator configuration benchmark

The [RTX PRO 4000 dataset](results/2026-11-23-operator-generalization-pro-4000/README.md) contains **4,680 GPU measurements**: 60 workloads × 26 configurations including eager × three fresh processes. The 24 Triton configurations span block size and warp count; compiled PyTorch supplies a strong reference. All recorded numerical checks passed.

[Download and reproduce](results/2026-11-23-operator-generalization-pro-4000/README.md#download-and-run) · [Gymnasium interface](docs/operator-generalization.md#use-the-gymnasium-interface) · [Frozen protocol](protocols/operator-generalization-v2-compile-isolation-v1.json)

## Transfer to RTX 4090

Configuration rankings changed between the RTX PRO 4000 and RTX 4090 cohorts. Across 18 test workloads, mean Spearman rank correlation was **0.301** (median 0.280). Only **6 of 18** source-optimal choices remained target-optimal. Transferring the source-table optimum produced mean target regret of **0.107481**, a difference in latency normalized to eager, not a percentage slowdown.

All 1,404 target measurements passed numerical checks. Source choices were frozen before target measurement; no model was retrained on the target. This is one GPU/runtime pair across six constructed shape families.

![Frozen choices on the second GPU](results/2026-11-23-operator-generalization-pro-4000/figures/transfer/cross-gpu-selection-regret.svg)

[Per-workload rankings and transfer data](results/2026-11-23-operator-generalization-pro-4000/README.md#transfer-to-rtx-4090)

## Eight-trial policy comparison

Each method received eight trials on each of 18 held-out workloads, with 20 episodes per workload. Lower regret is better; zero means the selected configuration matches the finite table's best measured cost.

| Policy | Mean final table regret | Repeated actions / 2,880 turns |
| --- | ---: | ---: |
| Compiled-query baseline | 0.027931 | 2,520 |
| Random | 0.055878 | 0 |
| Optuna TPE | 0.031880 | 1,434 |
| Train-only ridge selector | 0.031822 | 0 |

The compiled-query baseline starts with eager, queries compiled and retains the observed incumbent; its final choice can therefore be eager. It had the lowest mean regret. The learned selector reduced regret relative to random search, but did not improve on the compiled-query baseline. Its small difference from TPE does not establish superiority. The two LLM selectors ran one episode per workload and appear in a [separate matched comparison](results/2026-11-23-operator-generalization-pro-4000/README.md#matched-llm-comparison).

These comparisons query the same three recorded GPU-process repeats. Policy episodes measure search behavior; they do not add independent hardware measurements. Regret evaluates the selected configuration using all three process repeats, so its curve can rise when a noisy observation changes the incumbent.

![Eight-trial configuration-search regret](results/2026-11-23-operator-generalization-pro-4000/figures/cpu-20/policy-regret-vs-budget.svg)

## Fresh frozen-selection performance

All 42 distinct selected workload/configuration pairs were remeasured in three fresh processes with eager and compiled references, producing 378 correct measurements. The descriptive geometric mean of per-workload median compiled/selected latency ratios was 1.074× for the compiled-query baseline, 1.053× for random, 1.064× for TPE and 1.058× for ridge. The mini and nano selections measured 1.051× and 0.988×. These are the frozen choices, including eager fallbacks, rather than a new search run.

[Per-process variability and figures](results/2026-11-23-operator-generalization-pro-4000/README.md#fresh-frozen-selection-performance)

## Other coverage

The earlier [12-workload operator dataset](results/2026-11-20-operator-search-rtx-pro-4500/README.md) retains its separate four-trial comparison and fresh confirmation. It is not pooled with the expanded suite.

Vista model measurements and their original protocols remain in the [evidence archive](research/archive/README.md). Numerical operator checks do not establish full-model video quality or speed.
