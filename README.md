# DriveBench: Verifiable Rewards for Lossy World-Model Inference

**Research thesis: verifiable rewards for lossy inference optimization of video world models.**

[Project website](https://drivebench-research.rahul-a-mahadik.chatgpt.site) · [Research protocol](docs/research-protocol.md) · [Historical measurements](RESULTS.md) · [Reproduction](docs/reproduce-release.md) · [Citation](CITATION.cff)

DriveBench measures the tradeoff between inference cost and output fidelity, and exposes that feedback to optimization agents. The research question is whether an evaluator can give useful, difficult-to-game rewards when an optimization deliberately changes a model's output. A faster result is useful only if the verifier measures the intended fidelity constraints.

**Status: research in progress.** Historical GPU measurements are published. The modern Vista baseline is being implemented and has not yet been measured on a GPU. Empirical verifier error rates, a large measured lookup table, repeated policy comparisons, hardware transfer and adversarial code evaluation remain open. No scientific finding from those phases is claimed.

The former one-episode, eight-configuration LLM/random/grid study was **canceled before execution on November 14, 2026**. Its plans are retained for provenance, not offered as the active research design. See the [plan registry](configs/research/active-roadmap-v2.json) and [cancellation record](docs/canceled-plans.md). More legacy eager-versus-compiled repeats are not planned.

## Research sequence

| Phase | Question | Required evidence | Current state |
|---|---|---|---|
| 1. Modern baseline | Does compilation still help after native SDPA, half precision and channels-last? | Legacy/modern baseline table; fresh-process costs, break-even and runtime attribution | Implementation; GPU measurements absent |
| 2. Quality verifier | Does the gate reject meaningful degradation while accepting valid variation? | Frozen development calibration and independently labeled held-out false-accept/false-reject rates | Design; uncalibrated |
| 3. Tabulated search | What cost–memory–quality frontier exists in a conditional inference space? | Repeated measured cells, provenance, failures and per-cell uncertainty | Design; no lookup dataset yet |
| 4. Policies and transfer | Do policies beat TPE, and do configurations transfer across GPUs? | At least 20 search seeds per method, live confirmation, rank correlation and transfer regret | Unrun |
| 5. Code actions | Can a protected evaluator reject gaming while measuring a real kernel substitution? | Profile-selected operators, adversarial tests and end-to-end Vista delta | Design; no arbitrary-code service |
| 6. Findings | Which hypothesis survives the measurements? | Finding-led report, including null or negative results | Pending evidence |

The first live step is a bounded memory-fit and execution-cost check on cheaper hardware, without changing the declared workload. A 48 GB Ampere device is a candidate, not a proven fit; an 80 GB device is the fallback. The full table follows measured cost, not a fixed promise to spend on 1,000 configurations.

The phases are gates, not a checklist of already implemented capabilities. The target of at least 1,000 configurations is a future coverage goal; an enumerable Cartesian product is not a measured benchmark. A second model is deferred until the baseline, verifier and tabulated comparison are sound.

## Modern Vista baseline

The new [modern adapter](docs/modern-vista.md) is separate from the preserved legacy implementation. It must preserve the pretrained action-conditioning projections and checkpoint keys when replacing attention with native PyTorch SDPA. Channels-last and precision are explicit, measured choices, not presumed wins.

The [measurement protocol](docs/modern-baseline-measurement.md) uses few fixed scenes and repeated fresh processes, discloses compiler-cache state, and separates initialization, the cold prefix and steady-state requests. Compilation-only cost is reported only when directly instrumented; first-request latency is not relabeled as pure compilation. Break-even may be infinite. Profiling and performance measurement are separate runs.

The legacy eager row is a labeled historical cohort when same-device remeasurement is unavailable. A historical GPU/runtime row is not a paired modern comparison. The modern eager and modern compiled comparison must share the same physical device, full runtime, weights, inputs and workload.

## Quality, tables and code actions

- [Verifier design](docs/verifier-research-design.md): seed and precision variation are calibration data, not automatic quality labels; final thresholds are frozen before held-out labels are opened.
- [Tabulated study design](docs/tabulated-study-design.md): conditional configurations, failed and missing cells, per-cell repeats, policy costs and hardware transfer.
- [Code-action threat model](docs/code-action-threat-model.md): protected evaluator and explicit adversarial cases. The current configuration interface is not a secure arbitrary-code sandbox.

The proposed final verifier combines a distributional video metric with trajectory-following assessment. Both need validated feature/estimator implementations and adequate data; neither a four-scene distribution score nor a hard-coded tolerance establishes quality preservation. False-accept and false-reject rates must include denominators, uncertainty, label disagreements and invalid-evaluator outcomes.

## Existing software interfaces

The released benchmark records latency, throughput, allocator memory, failures and source/runtime/input identity. `AgentSession` supplies bounded tools and evaluator-owned final scoring for typed configuration actions. The operator tier provides `BudgetedTuningEnv`, a CUDA evaluator, compiled PyTorch references and example Triton configurations. These reusable contracts do not imply a trained RL policy or superiority over autotuning.

Install the first-party packages:

```sh
git clone https://github.com/rahul-mahadik/driving-world-model-inference.git
cd driving-world-model-inference
python3 -m venv .venv
. .venv/bin/activate
python -m pip install . ./extensions/kernel_tuning
```

CPU demos remain available as **software smoke tests only**, in the [installation guide](docs/reproduce-release.md). Their synthetic outputs are not empirical benchmark evidence. Real Vista runs require separately licensed assets, a declared NVIDIA environment, evaluator-owned checks and resource limits. The public package does not purchase or provision compute.

## Measured historical evidence

- [H100 legacy inference comparison](results/2026-10-30-h100-tradeoff/README.md): 200 requests per role on the pinned Torch 2.0.1/xFormers stack; 23.85% lower median latency and 7.68% higher peak allocated memory. Held-out quality validation was not completed.
- [Numerical audit](results/2026-10-30-h100-tradeoff/numerical-audit/README.md): all 400 original timing rows and a standalone reconstruction of 48 recorded fields. Reanalysis is not a new GPU reproduction.
- [A100 operator experiments](results/2026-11-07-kernel-tuning/README.md): historical operator correctness and timing; no demonstrated RL advantage or whole-video gain.

All historical result files, failed attempts and original qualification flags remain unchanged. They are not evidence for the modern baseline or the proposed verifier. No leaderboard is presented before measured coverage exists.

First-party code is [MIT licensed](LICENSE). Model weights, datasets and third-party software retain their [separate licenses](NOTICE.md). Contributions should include reproducible measurement and appropriate claim boundaries; see [CONTRIBUTING.md](CONTRIBUTING.md).
