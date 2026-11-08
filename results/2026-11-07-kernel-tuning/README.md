# A100 operator kernels and RL configuration search

The [separate operator study](summary.json) passed 75 GPU tests with no skipped tests. Two-order confirmation found lower incremental peak allocation than eager for 6 of 6 tested operator workloads; per-workload eager and compiled comparisons retain measured regressions; the original failed numerical test contract is documented separately. A tabular Q learner completed GPU-backed configuration search and equal-budget learned/random/grid evaluation. RL superiority, unseen-shape generalization and whole-video benefits remain unvalidated. Historical compiler timing and quality claims are unchanged. GitHub publication remains unverified.

Inputs are seeded random tensors with small engineering shapes or selected historical Vista profile shapes, not captured model activations. No full Vista checkpoint was executed. Latency uses CUDA events around one invocation and excludes compilation, warmups, correctness, input creation and output transfer. Allocation is the peak live allocator increase over resident inputs and includes returned output; it is not full-model GPU memory.

RL is finite-budget kernel-configuration search, not a driving simulator. Training and evaluation use the same workload keys with fresh measurements. Training cost is separately reported. Best-observed incumbents are descriptive; selected RL configurations have no independent confirmation, so `rl_improvement_validated:false` remains explicit.

The first environment attempt failed because CUDA and the inventory tool represented the same physical UUID differently. Versioned support v2 accepts only the optional physical-GPU prefix and retains both raw forms privately. Failed/partial artifacts remain archived and are not credited as completed experiments.

The original 188-test suite had 185 passes, one failure and two skips. The constant-input GroupNorm test failed eager equivalence. A reviewed test-only revision checks the independent analytic SiLU(bias) reference at one float32 epsilon absolute tolerance; the original failure remains a failed contract. The custom result matched the double-statistics oracle more closely on this diagnostic, but instruction-level causality was not profiled. The two no-GPU tests now mock unavailable CUDA on every host instead of skipping on GPU hosts. Kernels and benchmark tolerances were unchanged. Later passes do not establish constant-input eager or bitwise equivalence.

[Correctness](correctness.json) · [Smoke](smoke.json) · [Profile shapes](profile.json) · [Test-contract revision](test_revision.json) · [RL training and evaluation](rl.json) · [Runtime](runtime.json) · [Protocol](protocol.json) · [Failures](failures.json) · [Provenance](provenance.json) · [Backup integrity](backup-integrity.json)

## Measured operator results

All rows below use float16 inputs and the selected block size 1024 / 4 warps. Ranges show the medians from two fresh confirmation orders; they are not confidence intervals. The memory column compares peak incremental live allocation, including output, with the eager operator.

| Operator | Input shape | Eager latency (ms) | Custom Triton latency (ms) | Compiled PyTorch latency (ms) | Less allocation vs eager |
| --- | --- | ---: | ---: | ---: | ---: |
| GEGLU | 50 × 576 × 10240 | 1.648 | 0.586–0.588 | 0.594 | 50.00% |
| GEGLU | 288 × 25 × 10240 | 0.454–0.455 | 0.196 | 0.202 | 50.00% |
| GroupNorm + SiLU | 50 × 320 × 72 × 128 | 3.177 | 1.032 | 0.678 | 74.86% |
| GroupNorm + SiLU | 50 × 640 × 36 × 64 | 1.638–1.639 | 0.566–0.567 | 0.401 | 74.93% |

GEGLU used the same peak allocation as the compiled control. GroupNorm + SiLU used about 0.586% more allocation and was 41–52% slower than compiled PyTorch. The two small smoke workloads also used less allocation than eager, but the custom kernels were approximately 11–18% slower on those small inputs. These regressions remain part of the result.

## Measured RL execution

The configuration environment chooses block size and warp count, evaluates the candidate on the GPU, and returns correctness-gated latency and memory feedback. Its normalized cost is `(latency / eager_latency + 0.25 × memory / eager_memory) / 1.25`; the reward is clipped `1 − cost`, with invalid results penalized. A candidate replaces the eager incumbent only when its measured cost is lower.

The Q learner completed 12 training episodes with 96 candidate measurements and 12 fresh eager baselines. Evaluation ran 12 more episodes across learned, random and grid policies, with the same eight-candidate budget per episode: another 96 candidate measurements and 12 fresh baselines. Together these are **216 actual CUDA trials / 6,480 timing samples**. Training cost is separate. Every recorded action, reward, incumbent and learned Q-value was independently reconstructed.

The learned policy retained eager for all evaluated GEGLU episodes. Normalization's learned combined-cost outcomes overlapped grid search. This validates a working GPU-backed tuning environment, without establishing RL superiority, unseen-workload generalization or an independently confirmed RL-selected speedup.

## Validation and preserved attempts

The revised suite passed **188 tests, including 75 GPU tests**, with no skips. Smoke and profile benchmarks contribute another **468 trials / 14,040 timing samples**. Across the benchmark and RL stages, all **684 trial IDs were unique** and all completed trials passed their stated correctness gates. The original failed constant-input contract is described above and is not counted as passing.

[Operational attempt details](operational-attempts.json) retain the creation-response ambiguity, early SSH readiness failures, local preflight error, initial backup checker ordering failure and one recovered compute-cleanup response failure. These are operational history, not kernel experiment successes. All original private receipts and failed scientific attempts remain preserved locally. The A100 and its new boot volume were released after verified backup; the historical evidence disk remains retained.

