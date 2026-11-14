# Benchmark and environment readiness

DriveBench is public measurement and policy-evaluation software. A useful benchmark does not require its example optimization to win, and an RL environment does not require a trained policy. The relevant questions are whether another user can execute the task, whether the measurements and reward match its declared contract, and what evidence supports the resulting claims.

## Three interfaces, separate evidence

| Component | What is provided | What has been exercised | Remaining limit |
| --- | --- | --- | --- |
| Historical Vista benchmark | Pinned inference, input preparation, timing/memory records, output comparisons and reports | Real H100/A100/H200 cohorts described in [the results index](../RESULTS.md) | The reported H100 compiler timing improvement has no completed joint held-out quality validation |
| Public operator benchmark and Gymnasium task | Versioned tensor workloads, finite candidate configurations, correctness checks, fresh trial measurements and policy evaluation | CPU integration of the public suite; earlier GPU operator and Q-learning experiments on their own workloads | Those earlier experiments do not validate the newer suite or establish policy superiority |
| Configuration-agent environment | Proposal snapshots, tool feedback, budgets, protected evaluator callbacks, final-selection barrier and scoring | Synthetic engine → tools → batch receipts → evidence reader → scoring | Real Vista/LPIPS/LLM validation through the new workflow remains unrun; arbitrary-code execution is not supported |

The operator task is finite-budget configuration search. It supports classic RL algorithms through Gymnasium, but bandit methods and ordinary autotuning are appropriate alternatives. The tool interface supports an LLM choosing configuration edits sequentially. It does not claim that a sixteen-setting task needs an LLM, that the LLM was trained with RL, or that its reward is safe against an attacker controlling the evaluator host.

## What makes a result comparable

Before measuring, record the workload, reference, candidate set, objective, source/runtime identities, inputs and budgets. Keep physical GPUs and runtimes in separate cohorts. Report latency boundaries and memory definitions next to the numbers: operator CUDA-event latency, cold inference requests and warm full-model requests are different measurements.

Preserve all attempts, including unsupported settings, incorrect outputs, OOMs, timeouts and partial runs. A missing or failed check cannot be converted into a passing score. The best observed trial is subject to selection bias; an improvement claim needs fresh confirmation after the candidate is fixed. Selection and confirmation costs belong in the report.

For model-level quality, freeze the proxy definitions and justified margins using development data before candidate selection. Seed variability describes the reference; it is not permission to relax a margin until a candidate passes. Final inputs remain outside the policy observation. Proxy acceptance is narrower than visual equivalence, trajectory fidelity or driving safety.

The complete measurement and comparison rules are in [the benchmark protocol](public-benchmark-protocol.md) and [agent guide](agent-environment.md). A custom evaluator is responsible for its own measurement and quality implementation. A `correct=True` field or a SHA-256 identity alone is not an independent audit of it.

## Reproducing and reporting a study

A reusable result should include:

1. The exact code revision, installation/runtime recipe, licensed-asset acquisition instructions and content identities. Do not redistribute data or weights without permission.
2. The frozen workload/split, candidate choices, reference settings, measurement settings and planned policy order. Declare whether evaluation inputs were previously public or used during development.
3. Candidate and reference calls, failures, initialization/compilation time, actual wall time and training/model-API cost. Equal candidate budgets are not automatically equal total cost.
4. Per-task measurements and raw samples or a documented route to independently regenerate them. Retain evidence inventories and hashes; do not pool unrelated devices into one speedup.
5. The locked selections, fresh final comparison, quality outcome and limitations. Report a zero, rejected or inconclusive outcome without requiring a winner.

The current full-Vista proposal is a small feasibility study: eight development scenes, four final scenes, two seeds and two repeats, with four trials each for LLM, random and grid policies. It keeps the model workload fixed but has limited independent scene coverage. A successful run would demonstrate this particular integration, not broad calibration, generalization, robust quality preservation or a statistically established advantage of LLM search.

## Trust boundary

The configuration agent receives a public JSON observation and returns an allowlisted action. It has no shell, provider tools or direct filesystem access through the supplied selector. The operator owns source code, metric assets, private inputs, process termination and output verification. Run the worker under an external timeout, verify cleanup and keep evaluator storage outside policy write access.

The library does not provision cloud resources, accept legal agreements, add funds or implement an operating-system sandbox. Supporting agent-written Python or custom GPU kernels in a scoring process would require an additional isolation boundary and adversarial validation. Those capabilities should not be inferred from the current configuration environment.
