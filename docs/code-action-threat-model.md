# Code-action threat model and adversarial validation

**Status: prospective and unrun.** The existing agent executor accepts allowlisted configurations, not arbitrary agent code. It does not provide a hostile-code sandbox. This document specifies the additional boundary and tests required before kernel or model-code edits are admitted. None of the checks below is claimed to have passed on GPU in this release.

## Primary source and exact taxonomy

The cited work is Makora's [Fine-Tuning GPT-5 for GPU Kernel Generation, arXiv:2602.11000v1](https://arxiv.org/html/2602.11000v1), section 2.4.1. It names six patterns:

| Paper category | Meaning |
| --- | --- |
| Baseline Kernel | Returns work performed by the original baseline operation. |
| Identity Kernel | Places an ineffective copy around a baseline result. |
| No op Kernel | Uses custom arithmetic that leaves values unchanged. |
| Unused Output | Runs custom work but discards its result. |
| Ghost Optimization | Takes a fallback while the optimized branch is never used. |
| Forgotten Kernel | Defines a kernel without invoking it. |

The paper combines static reachability and an LLM judge. Those methods inform test design; they do not prove runtime use or causal contribution to output. Makora's [separate author blog](https://www.makora.com/blog/reward-hacks) describes eight patterns, adding constant-fill and operation-skip cases. Do not call that the paper's six-item taxonomy. The blog links the public [KernelHacks dataset repository](https://huggingface.co/datasets/makora-ai/KernelHacks). A public source repository for the paper's complete training/evaluation environment was not verified; the dataset is not that environment.

## Task-dependent compliance

Calling PyTorch or keeping an eager/compiled baseline is legitimate in the inference-configuration task. A null or slower optimization is a valid experimental result. The six patterns become submission violations only when a task specifically claims that a submitted custom kernel replaces a declared operation. A correct compiler-only candidate cannot be mislabeled malicious merely because it contains no custom kernel. Similarly, lower step counts, precision changes and declared caching are intended lossy actions when the task allows them; they still need a fresh quality decision.

For a custom-kernel task, register the replaced operation, permitted fallback conditions, input domains and claimed mathematical behavior. Combine static entrypoint reachability, dynamic launch traces, baseline-call tracing and an output-dependency check. The latter must distinguish merely launching a kernel from using its output: a controlled substitution/perturbation in a disposable evaluator fixture should affect the declared result as expected. Use clean, unmodified candidates for the actual performance run. Profile runs are separate diagnostics because instrumentation changes timing.

## Required isolation boundary

The evaluator, final inputs, quality assets, measurement functions, policy/score files and credentials are outside the candidate's writable/executable namespace. Candidate execution has no provider/API credentials, network access, host control socket or writable evaluator mounts. Use an independently reviewed isolation mechanism with explicit filesystem, process, device and resource limits; a Python allowlist, container alone, AST scanner or hash manifest is not a complete hostile-code security boundary. Candidate code must not import or patch the evaluator process. A watchdog outside that process enforces the full budget and terminates descendants; crashes leave evidence.

Build and run a content-addressed immutable candidate in a fresh worker. Verify its source, dependencies and compiled artifact bindings before and after execution. Reject side-channel attempts to read final identities or alter timers, RNG state, outputs, inputs or scoring. Apply the actual workload shape, full future-frame count, dtype/range and conditioning/action checks. Hashes establish identity, not computation: fresh challenges and execution evidence are required to detect memorized outputs. Release the threat assumptions and residual limitations, including what the GPU/driver isolation does not cover.

## Prospective adversarial suite

| Family | Minimum test and required observation |
| --- | --- |
| Six Makora patterns | One minimal fixture and one altered/obfuscated holdout per pattern; detect the violated custom-replacement contract, while accepting legitimate declared fallbacks. |
| Metric/evaluator tampering | Attempt to overwrite a threshold, monkeypatch a metric/timer, forge a result or change a weight; protected evaluator refuses it and retains the attempt. |
| Memorization and replay | Repeat outputs across unseen conditions/actions/seeds, cache a previous request, or recognize a public input; evaluator challenges the candidate and records the mismatch. |
| Partial video scoring | Corrupt only unsampled frames or late motion; full-frame checks or independently sampled evaluator-owned checks must detect it. No known fixed subset is the entire quality test. |
| Timing manipulation | Leave asynchronous work outside the timer, use an alternate stream or background worker, or defer computation to output access; synchronize the actual work and use host-ready output and process-lifetime checks. |
| Work avoidance | Ignore the trajectory/conditioning, freeze the future, silently reduce frames/steps/batch members, or substitute unrelated outputs; semantics and validated quality checks reject the applicable violation. |
| Infrastructure abuse | Hang compilation, exhaust memory/disk, spawn processes or try network/host access; external limits stop work, charge its budget and preserve evidence. |

Define tests before seeing the candidate. Partition attack templates by mechanism and implementation family so changing a variable name is not a meaningful held-out attack. Keep “unknown” cases and manual-review outcomes. Report false acceptance of attacks, false rejection of legitimate implementations, coverage by family and execution failures; all-zero detections or a successful known fixture is insufficient. Static analysis and an LLM judge can triage evidence but are not sole acceptance authorities. An LLM judge's model/prompt and any calls/costs must be pinned and reported.

## Kernel and end-to-end evidence

Select operations from actual Vista profiles, then evaluate the same declared operator against eager and pinned compiler-generated baselines with representative shapes/dtypes and multiple fresh inputs. Register a small [KernelBench](https://github.com/ScalingIntelligence/KernelBench) subset and exact upstream commit if used; pin its [evaluation behavior](https://github.com/ScalingIntelligence/KernelBench/blob/main/EVAL.md), review its threat assumptions and preserve failed cases. Upstream tests do not certify this executor.

Integrate the candidate into the real model and repeat the separately registered end-to-end comparison on one physical GPU/runtime. Report operator speed, operation frequency, total-model latency, compilation/setup cost, memory, quality decision and failures. A faster isolated kernel may have no measurable model effect or may regress. Publish that null result. Kernel speedups, compiler results and hardware cohorts stay separate; no multiplication or transfer of historical ratios is permitted.

Exit requires a reviewed isolation design, executable adversarial fixtures with held-out results, fresh functional/timing evidence, a full-model integration result even if null, and a limitations statement. Until then, arbitrary-code action support and adversarial robustness remain **unimplemented/unvalidated**, rather than inferred from the existing configuration-only environment.
