# PPO and native-autotuning supplement

Five PPO policies were trained on the measured operator benchmark. This release records every checkpoint, training episode and held-out decision, together with a separate live comparison against native Triton autotuning.

## PPO on the measured table

The training split contains 30 workloads and 2,340 recorded GPU measurements. Each of five fixed seeds completed 102,400 environment steps (512,000 total). All checkpoints were frozen before TEST evaluation. There was no hyperparameter search, held-out training or best-seed selection.

Each trained checkpoint and one untrained initialization received twenty eight-query episodes on each of eighteen held-out workloads. These **2,160 evaluation episodes** reuse measured timings from the RTX PRO 4000 table; they add no independent GPU samples. The test split contains six shape families, each in three dtypes.

| Method | Mean final table regret |
| --- | ---: |
| Compiled-query baseline | 0.027931 |
| Optuna TPE | 0.031880 |
| Ridge selector | 0.031822 |
| Random | 0.055878 |
| PPO, mean of all five training seeds | 0.035556 |
| Untrained initialization, seed 3101 | 0.218278 |

PPO improved over initialization but did not beat TPE or the compiled-query baseline in mean regret. The five trained-seed means are 0.033024, 0.025118, 0.027289, 0.035376 and 0.056971 (sample standard deviation 0.012672). The paired mean PPO-minus-TPE difference is 0.003676; its descriptive six-family bootstrap interval is [−0.034030, 0.042650]. These data do not establish policy superiority.

![Held-out search curves](../../reports/figures/ppo-regret.svg)

[Training settings and implementation](../../docs/ppo-training.md) · [Seed spread](../../reports/figures/ppo-seed-spread.svg) · [Analysis JSON](replay-analysis.json)

## Live GPU comparison

The fixed PPO checkpoint for seed 3101 was evaluated on an NVIDIA GeForce RTX 4090 with PyTorch 2.10.0 and Triton 3.6.0. It received fresh GPU feedback for eight queries per workload, without policy updates. Each of three fresh processes evaluated the same eighteen held-out workloads, using inputs bound to the original benchmark. The GPU cohort is separate from the RTX PRO 4000 training table.

Native `triton.autotune` evaluated all 24 Triton configurations through the same whole-operator wrappers. A second comparator could also retain eager or compiled code: three selection probes chose among those references and the native-selected kernel before final confirmation. PPO can also retain eager or compiled code, so this fallback comparator better matches its available choices.

| Comparator | Comparator latency / PPO latency | Descriptive 95% interval |
| --- | ---: | ---: |
| Native autotuning, Triton kernels only | 1.089 | [1.019, 1.163] |
| Native autotuning with eager/compiled fallback | 0.948 | [0.882, 1.009] |
| Compiled reference | 1.039 | [0.981, 1.141] |

Ratios above one favor PPO. PPO's mean advantage over native-only selection did not extend to native selection with reference fallback. The comparison does not establish an equal-budget win: native selection uses 24 configuration trials and three fallback probes, while PPO uses eight queries plus an eager reset.

Every choice was followed by independent eager, compiled, native-selected and PPO-selected confirmations. All **216 confirmations passed numerical correctness checks**; all 1,296 native configuration trials and 432 PPO queries completed without rejection. Each measurement uses five warmups and thirty synchronized CUDA-event samples. Initialization, compilation, correctness checks and warmups are outside the timed invocation. Selection costs are recorded separately. Independent CPU inference with the frozen checkpoint reproduced all 432 recorded actions; the [verification receipt](frozen-policy-verification.json) binds the checkpoint, manifest and verifier.

The summary takes the median of three within-process latency ratios per workload, then their geometric mean. Intervals resample six shape families with their three dtypes together. They describe this allocation and fixed checkpoint; repeated processes do not establish population-level variation across GPUs. No fresh full-grid oracle was measured, so these results report latency differences rather than live regret. The gates check isolated-operator numerical agreement, not perceptual quality or end-to-end Vista performance.

Median instrumented selection time per workload/process was 10.27 seconds for native tuning and 4.38 seconds for PPO; the reference fallback probes added a median 1.70 seconds. The methods share compilation caches within a process and alternate selection order. PPO's evaluator journals every CUDA sample, while native tuning journals each configuration. These wall times include that unequal logging overhead and do not isolate intrinsic search efficiency or cold compilation. Offline PPO training and final confirmation costs are excluded.

![Fresh confirmed operator latencies](../../reports/figures/ppo-native-speedup.svg)

[Instrumented tuning cost](../../reports/figures/ppo-tuning-cost.svg) · [Live analysis JSON](live-analysis.json)

### Live data and reproduction

The [live evidence archive](live-data.tar.gz) is 4,666,784 bytes, SHA-256 `739b8775ed7e456c33bc01050e6645400e0fa3c0c2e7ac35e2549cc4d9a951d7`. Its [inventory](live-archive-members.json) records all 24 members, expanding to 61.76 MB. It includes the three scientific journals, all observations and actions, configuration timings, correctness results, confirmation samples, exact native source, input bindings, scientific protocol and offline audit code.

```sh
tar -xzf results/2026-11-24-ppo-native-autotune/live-data.tar.gz -C results
PYTHONPATH=extensions/kernel_tuning python \
  results/ppo-native-live-2026-11-23/reproduce_analysis.py \
  --output live-summary.json
python scripts/plot_ppo_supplement.py --kind live \
  --analysis live-summary.json --live-gpu-label 'RTX 4090' --output live-figures
```

The offline audit reconstructs native choices, fallback choices, PPO incumbents, observations and rewards from the recorded measurements. Public files replace physical GPU identifiers and the corresponding cohort hash consistently. All measurement values, trial IDs, inputs and decisions are retained, with original-to-public file hashes in the export receipt. Private provider and host-operation files are excluded. The public analysis reproduces the raw analysis after this declared cohort mapping.

## Download and reproduce PPO replay

The [PPO evidence archive](ppo-data.tar.gz) is 29,074,068 bytes, SHA-256 `ff7a0e812d2b3a06272def63556c68580be5097b2a0e7694ba642d3041031917`. It expands to 518 MB. Its [inventory](ppo-archive-members.json) records all 71 member hashes.

The archive includes five trained checkpoints, the untrained reference, all training and evaluation journals, four original CPU baseline reports, the exact pre-training source, and a measured-row mapping to the public GPU dataset. Checkpoints and scientific records are byte-exact; one diagnostic log has its workstation prefix removed. Original measurement IDs are retained.

From the repository root:

```sh
python -m pip install -e './extensions/kernel_tuning[rl,search]'
tar -xzf results/2026-11-24-ppo-native-autotune/ppo-data.tar.gz -C results
PYTHONPATH=extensions/kernel_tuning python \
  results/ppo-training-2026-11-23/reproduce_analysis.py \
  --output replay-summary.json
python scripts/plot_ppo_supplement.py --kind replay \
  --analysis replay-summary.json --output replay-figures
```

The archived reports use one original measured-table hash. The archive's `source-bindings.json` maps each row to the public table, where physical GPU identifiers were pseudonymized. Numerical measurements, inputs and choices are unchanged. Training again on the public table creates a new protocol with its own public-table/source hashes; see the [training commands](../../docs/ppo-training.md#reproduce-training-and-evaluation).

The archive retains the original summary and a metadata correction: summarization itself performs no training, while its five trained input reports do. The corrected summary describes that distinction. An offline verification recomputed every one of the 512,000 recorded training transitions and every evaluation trajectory.
