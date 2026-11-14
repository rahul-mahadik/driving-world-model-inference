# Verifiable rewards for lossy video-world-model inference

**Status: prospective research program, not a completed benchmark study.** The former one-episode/eight-configuration/three-policy study is [canceled](canceled-plans.md). Its original [protocol](archive/research-protocol-v1.md), configurations and evidence remain immutable historical records. The [active registry](../configs/research/active-roadmap-v2.json) identifies the new program.

## Thesis and falsifiable questions

An optimization may legitimately change video outputs while reducing inference cost. The evaluator must distinguish acceptable variation from real degradation and reward only supported improvements. The central artifact is a measured, reproducible reward with a validated error rate, not a claim that an LLM or RL method must win.

The candidate findings are hypotheses: configurations may fail to transfer across devices; a calibrated verifier may reject manipulations missed by tensor tolerances; or policy priors may fail near memory and compatibility boundaries. Write-up order follows the observed evidence, including null results. Do not choose a headline in advance or suppress contradictory cases.

## Phase 1 — establish the modern reference

Create an explicit modern Vista adapter with native PyTorch SDPA, declared half precision and channels-last. Preserve the original checkpoint, trained action projections, sampler and conditioning semantics. A dependency or operator incompatibility is a recorded outcome; retain the old stack under a legacy label rather than silently dropping weights, precision checks or functionality.

Measure modern eager and modern compiled on one physical GPU with the same full modern runtime. Use a small fixed scene panel and many fresh-process restart blocks, with balanced role order. Disclose disk compiler cache, OS cache, weight loading, warmups and request boundaries. The complete prospective schedule and statistical unit are specified in [modern baseline measurement](modern-baseline-measurement.md). Do not multiply the effective scene count by restarts or treat event samples as independent workloads.

Report initialization, cold first request, the full cold prefix, steady-state latency, memory and failures. Attribute compile-only duration only through direct instrumentation; report a labeled compilation-attributable excess or null when it cannot be identified. Calculate request-count break-even from total cold and steady-state costs, including uncertainty and the fraction of resamples with no finite crossing.

Profile separately: conditioning, denoiser and VAE decode at request level; then operator/device-time attribution. Use non-overlapping spans or interval unions and show unassigned time. Do not sum nested CPU spans or asynchronous kernels into wall time. Select future custom-kernel targets from this profile, with compiled PyTorch as the baseline.

**Exit:** a legacy eager / modern eager / modern compiled table, cold-cost and break-even fields, and a runtime-attribution chart backed by retained records. Historical legacy rows stay labeled cross-cohort; modern pairwise comparisons require matched hardware/runtime. If modern eager eliminates the former compilation benefit, report that as a finding. No additional legacy eager-versus-compiled timing campaign is planned.

## Phase 2 — calibrate and validate the verifier

Measure reference distributions under seed changes and declared precision changes. These estimate variability; they are not ground-truth declarations that every changed sample is equally good. Independently label reference variants and prospective degraded configurations, including few-step and aggressive-cache candidates whose actual outputs may or may not be bad.

Use scene-disjoint calibration, threshold-selection and held-out verifier-audit sets. Freeze the metric versions, feature extractor/trajectory estimator, aggregation, thresholds, controls and analysis before opening final labels. Development may use cheap seed-matched proxies. The evaluator-owned final check adds distributional video distance and trajectory-following assessment, with enough independent scenes and an estimator validated against trusted motion labels. Protect metric assets and final inputs from candidates.

Report false accepts and false rejects against independent labels, not against labels produced by the same metric. Include counts, denominators, scene-cluster uncertainty, disagreement, estimator failure and invalid outcomes. A small zero-error sample does not prove a low error rate. See [verifier design](verifier-research-design.md) for sample-size and calibration details.

**Exit:** measured held-out verifier error rates and uncertainty. Until then, reward remains experimental and no quality-preservation claim is admitted.

## Phase 3 — measure a conditional search space

The target axes are steps, sampler, guidance batching, feature caching, token merging, VAE tiling, precision, compile mode and batch size. Each requires an implemented adapter, effective-setting readback and declared interactions. Do not count unsupported combinations as valid configurations. Preserve invalid configurations, OOM cliffs and their costs instead of silently dropping them.

Target at least 1,000 configurations for the full dataset, after an initial complete pilot panel. Prospectively freeze sampled configurations, scenes, generation seeds, repeated measurements, fidelity levels and an incomplete-run policy. A domain with 1,000 possible entries is not a 1,000-cell measured table. Publish actual coverage, fresh-process and timing repetitions, raw samples, input/runtime/protocol bindings, quality results, failures and per-cell uncertainty. Unknown cells remain unknown; do not interpolate them and call the result measured.

Use constrained speedup for feasible results plus a graded, predeclared penalty for measured constraint violation. Report the three-dimensional latency/memory/quality Pareto frontier rather than hiding tradeoffs in a scalar. Missing or invalid evaluator output has no usable performance reward. An explicitly registered synthetic surrogate may be used for software tests only.

**Exit:** a released lookup table with measured coverage and per-cell noise estimates. A table is an offline replay environment; it does not recreate compiler state, thermal drift, arbitrary code edits or unmeasured configurations. See [tabulated study design](tabulated-study-design.md).

## Phase 4 — compare policies and test transfer

Run random search, Sobol, TPE, successive halving, and an LLM with and without trial history for at least 20 declared search seeds per task on the same table. A history ablation must use the same base prompt, model snapshot, token/turn allowance and observation budget except for history. Table replay removes further GPU measurement cost; LLM calls, CPU computation and table construction still cost resources.

Use cumulative simulated measurement cost and invalid-action cost, not just query count. Successive halving requires actually measured fidelity levels; it cannot receive lower-fidelity observations absent from the table. Policies see only their allowed observations. Publish regret-versus-budget, probability of failed repeat/live confirmation, and budget spent on invalid actions. State the measured finite-panel oracle and its uncertainty; it is not a global optimum.

Lock selections before fresh live confirmation. No resampling a failed winner into a successful report. Rebuild a smaller matched configuration panel on a second device such as A100 or L40S, keeping runtimes/workloads controlled and hardware cohorts separate. Report rank correlation with tied/censored/OOM handling and transfer regret for the first-device choice. Twenty seeds estimate optimizer randomness on that task, not twenty new workloads.

**Exit:** a policy comparison against TPE and a hardware-transfer plot, whether the results are positive, null or negative. Comparative significance and family-level claims need adequate independent tasks, not merely repeated random seeds.

## Phase 5 — protected code actions

Only after profiling, choose real operator targets and allow multi-turn code edits with compile errors, correctness differences and profiler feedback. Execution requires an evaluator-owned container/VM boundary, no candidate network, no credentials, inaccessible evaluator references/metric assets/hidden-input manifest, only the per-request input tensors needed for computation, constrained writable outputs, hard wall/memory/process limits, and candidate-independent measurement. Static analysis is a filter, not a security boundary. The current configuration-only library is not such a sandbox.

Use the six hack archetypes from the specified Makora paper as explicit tests, plus lossy-specific attacks against perceptual metrics and unsampled video frames. Record exact reference/source revisions and test each defense rather than claim universal resistance. The [threat model](code-action-threat-model.md) distinguishes the paper's taxonomy from other lists and explains the additional video-specific cases.

Measure one accepted substitution inside Vista and report the whole-request delta against compiled PyTorch, even if it is null. Run a pinned KernelBench subset under its own task/correctness contract for comparability; do not pool those scores with video quality scores.

**Exit:** all preregistered malicious fixtures (including held-out variants for every declared class) are rejected by the real isolated evaluator, legitimate controls and false-rejection rates are reported, and one end-to-end substitution is measured. Passing mocked tests is not this exit condition.

## Phase 6 — write a finding-led report

Lead with the strongest supported finding, followed by method, uncertainty, negative cases and limits. Release exact code/configuration, dependency locks, distributions and failure/cost ledgers where asset licenses permit. Keep historical cohorts and old qualification flags unchanged. Add a second model only after Phases1–4 are supported by real evidence.

## Budget and storage are experimental constraints

The broad program is not admitted by this document. All existing spending limits remain binding until explicitly revised. A planning example of 1,000 configurations × 4 scenes × 2 generation seeds × 3 repeats is 24,000 full requests. At the historical 45–60-second reference range and $4.746 per GPU-hour, inference alone is approximately $1,424–$1,898. This excludes compile/setup, quality metrics, confirmation, second hardware, API calls, storage, failed attempts and cleanup; it is not a current quote or prediction for the modern stack.

At 176,947,328 bytes per historical raw request, that example is about4.25TB before metadata. A durable evidence policy and storage destination must be frozen before measurement. Do not delete prior raw evidence to make room, claim hashes substitute for unavailable outputs, or promise the full roadmap within the old sub-$500 cap. First obtain real Phase1 cost measurements, then cost the larger campaign.

Publish actual dollars and GPU time separately from credit purchases and reservations. Reuse existing bounded workers and cleanup primitives, with fresh immutable allocation contracts and input/runtime identities; do not revive canceled workflows or retired transports.

## Budget-first hardware feasibility

The November 14 hardware revision prioritizes a cheaper GPU and keeps the cumulative sub-$500 cap. A first bounded modern-stack check will establish whether the unchanged full workload fits on a 48 GB Ampere GPU and record complete first-request, warm-request, memory and failure costs. No 24 GB fit is assumed. If 48 GB fails, retain the OOM attempt and consider an 80 GB device; do not silently change frame count, resolution, steps, decoder chunking or precision to rescue the same task. A changed workload or memory policy needs a prospective separate configuration and quality check.

The feasibility stage measures dollars per **structurally completed request** and the cold setup cost, not hourly rent alone. Quality-valid request cost stays unavailable until a calibrated verifier exists. Compare contemporaneous rates and measured latency. Measurements on different physical devices remain separate cohorts; once a device is chosen, the modern eager/compiled restart study stays on that device and full runtime. The historical H100 row stays descriptive only. The first measured configuration panel should be small and complete; the 1,000+ target is a later budget/coverage gate. No larger campaign or top-up is authorized by this protocol.
