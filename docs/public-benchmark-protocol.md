# Public benchmark protocol — version 1

## Purpose and tracks

The deliverable is reusable measurement and search infrastructure. Implementations and policies are submissions, not presumed improvements. Report failed and slower candidates. Do not attribute historical compiler speedups to example kernels, new policies or new hardware.

Two distinct tracks are provided:

- **Video inference:** the existing Vista/nuScenes harness and historical cohort protocols. A request generates one declared future from image/trajectory inputs. End-to-end latency, sequential throughput, full-model allocator memory, and output-quality comparisons have their own boundaries. Licensed assets must be obtained separately. Historical held-out quality validation remains incomplete.
- **Operator configuration search:** `operator-search-v1`, packaged with the tuning extension. This uses standalone seeded tensor inputs and correctness checks against reference operators. An operator score does not establish end-to-end video performance, visual quality or driving safety.

## Task identity

Every comparison must bind the exact suite bytes (SHA-256), source hashes, candidate list/order, objective and weight, timing repetition/warmup settings, physical GPU, driver and software runtime. Separate devices/runtimes form separate cohorts; never pool their samples into a common speed ratio. A protocol hash is a configuration identifier, not proof a comparison was fair or source code trustworthy.

The released operator suite has six training, two validation and four test tasks. Splits are disjoint by operator/shape/grouping, even when only the dtype or seed changes. Test shapes match selected historical profile shapes, so they are not historically untouched workloads. Test access is open; prevention of policy selection on test results is a research procedure, not access control. Custom protocols must declare task groups before measurements.

Policies receive shape/operator features and the action history. The generic interface supports arbitrary candidate JSON; the supplied operator adapter supports compiled PyTorch and example Triton launch choices. Other executors must supply an evaluator. Do not silently change semantics, model weights, precision or quality thresholds to win a speed comparison.

## Budget and objective

The bundled operator suite defaults to four candidate measurements among nine settings, plus one fresh reference measurement per episode. Each attempted configuration, including a repeated, incorrect or error result, costs one trial. Repeats are fresh measurements. Protocol violations and unexpected evaluator exceptions preserve available journals and abort; they cannot become successful scores.

The reference starts as the incumbent and requires positive latency and allocated bytes, including for the latency-only objective. Zero-allocation references need an explicit alternative normalization contract. For a correctness-passing candidate,

`cost = (L/L_ref + w * M/M_ref) / (1+w)`.

For the operator suite, `L` is median measured operator latency and `M` is incremental peak allocated memory including the output. External inference tasks declare their own request-latency and memory boundaries; their scores cannot be pooled with operator scores. The default `w=0` is latency-only. Choosing a nonzero weight changes the task and may legitimately prefer slower, smaller-memory results; always publish both raw metrics. This declared scalar objective is not a universal utility function or Pareto claim.

The reward is `previous_best_cost - new_best_cost`; returned incorrect/error results earn zero and consume budget. The undiscounted return equals final improvement over the reference. A finite trial budget is part of the task, so an agent must not bootstrap beyond the final decision. Exposed history distinguishes untried, incorrect/error and measured outcomes. The environment is a finite-budget best-configuration search problem; bandit methods, conventional autotuning and RL agents can all participate.

## Measurement and acceptance

The provided CUDA evaluator uses an otherwise idle single NVIDIA device and a pinned runtime, with five warmups and thirty timed repetitions per measurement by default. CUDA events bracket one operator invocation and can include stream gaps. Host wall time is recorded separately. Compilation, initialization, input preparation, warmups, correctness and export are excluded from reported operator latency and included in job wall time. Record cold-start/compilation costs separately when they matter to deployment.

Inputs are seeded random tensors, not real model activations. Acceptance checks output shape, dtype, finiteness, dtype-specific allclose, and input immutability. Float16 uses rtol/atol0.002; bfloat16 uses0.02; float32 uses rtol0.00002/atol0.000003. These are operator tolerances, not calibrated video-quality thresholds. Preserve initial failures rather than tuning acceptance after seeing outcomes.

Memory measures the allocator's peak live allocation increase above resident inputs, including output and temporaries. It is not whole-model memory or GPU-reserved memory. Record reserved memory separately. Throughput from a serial invocation's latency describes that invocation; do not label it concurrent or saturated serving throughput.

External evaluators must document their measurement and quality gates, bind input/reference identities and retain evidence. Matching fields and `correct=True` establish a contract assertion, not a third-party audit. There is no automatic validation of arbitrary participant code.

The provided inference bridge has a separate cold-request contract: it constructs an engine per trial, records construction time separately, and measures context preparation through synchronized host-ready outputs, including first-use compilation. It requires a caller-supplied quality gate. Its CPU example uses synthetic frames and output-array bytes; it is not GPU performance evidence. See the [inference integration guide](../extensions/kernel_tuning/docs/extending.md) before defining a real model task.

## Policy comparison and reporting

Use validation for selecting search algorithms, hyperparameters, objectives and budgets. Freeze these choices before test. Compare the same tasks, action order, trial budget, objective, measurement settings and cohort. Supplied baselines are fixed grid prefixes and seeded random search without replacement. The grid starts with the compiled candidate; its ordering is part of the specification. No claim is made that these are the strongest possible baselines.

Evaluate fresh repeated episodes; interleave policy order in a prespecified schedule to reduce drift confounding. The single-policy CLI records one policy per invocation and does not automatically balance separate invocations. Advanced comparisons can call `evaluate_policy` in a scheduled order while sharing a serial evaluator/cohort. Never run independent GPU environments concurrently on one device when interpreting isolated latency.

Report every episode and its individual trials, per-task results, failed-call counts, reference and candidate calls, actual wall time, and all training cost. The harness reports per-task descriptive medians, not significance or generalization. Do not average unrelated latencies into one speedup. A learned policy's training calls/time are additional costs; amortization assumptions must be explicit.

An episode incumbent is a **best observed** result, subject to selection bias and noise. The evaluator does not automatically confirm it. A claimed improvement needs independent fresh measurements of the frozen selected configuration and reference, appropriate control comparisons, and enough independent repetitions to establish a signal. A separate exhaustive sweep can provide an oracle for small finite spaces; its cost must not disappear from the report.

## Result and release status

New runs retain plan, suite/source/protocol hashes, raw trial records, environment journals and completed/failed receipts. Existing paths are refused. Synthetic demos label every result as synthetic and cannot establish GPU performance or correctness. Interrupted or partial output is not completion.

Historical V10–V16 evidence stays immutable. The new V17 interface is CPU-tested and its current suite is not GPU-validated. Public availability and project licensing are separate release decisions. No leaderboard, RL superiority or quality-preserving end-to-end gain is implied by a successful run.
