# Measured configuration table and search comparison

**Status: prospective; no new table measurements or policy results.** This plan separates a large possible configuration domain from the number of configurations actually implemented, measured and quality-audited. It does not claim 1,000 supported or evaluated Vista configurations. The [verifier design](verifier-research-design.md) and [code-action boundary](code-action-threat-model.md) are prerequisites for their respective claims.

## Conditional configuration domain

The current trusted Vista agent adapter admits FP16/BF16, four decoder chunk sizes and a boolean denoiser compile path: at most 16 combinations. Existing flags elsewhere in the code are not proof that an agent-visible axis works. Each new axis needs a pinned implementation, correctness/semantic tests, supported values, compatibility rules and actual runtime evidence before it enters the measured domain.

| Requested axis | Required distinction before measurement |
| --- | --- |
| Steps and sampler | Pin noise schedule, integration algorithm and number of actual model evaluations; changing sampler is not just changing a label. |
| Guidance batching/schedule | Separate conditional/unconditional execution strategy from guidance strength and time schedule. Preserve or explicitly declare changed semantics. |
| Feature cache | Pin cached tensors, refresh schedule and invalidation scope. Separate legitimate within-request reuse from forbidden output replay across requests. |
| Token merging | Implement and record location, merge rate, matching rule and shape constraints; treat it as a lossy change. |
| VAE tiling | Specify spatial tiles, overlaps and stitching separately from the already available temporal decode chunk size. |
| Precision | Record actual compute, accumulation and output dtypes plus fallback behavior, not only the requested autocast flag. |
| Compilation | Pin compiled scope, mode, backend, graph breaks/recompiles and cold versus warm costs. |
| Batch size | Define futures/request, latency/request, latency/future and futures/second. Batch throughput does not inherit a single-future latency comparison. |

Canonicalize inactive options to avoid duplicates; freeze conditional legality before sampling. Do not create a fictitious Cartesian product of incompatible options. Publish the domain size, implemented size, attempted count, successful count and full-quality-evaluated count separately. OOM boundaries are empirical observations on a specified device/runtime/workload; do not infer them for unmeasured neighbors.

## Staged coverage

1. Complete the bounded verifier pilot and measure real time/storage per phase. This determines whether a full-quality gate is usable and what further measurement fits.
2. Freeze a small **16-configuration pilot panel** from actually implemented options before seeing their outcomes. For each configuration, use four optimizer-development scene groups, two generation seeds and three timing repeats: 24 requests/configuration, 384 candidate requests total. Interleave a matched reference under a predeclared block order. If it is not already a measured panel cell, its additional requests are budgeted explicitly. Three repeats estimate local timing variability, not broad quality validity; four scenes cannot support a distributional quality claim. Cold initialization/compilation and warmup costs are separately measured and charged.
3. Only after a fresh cost admission, expand to a frozen sampled set of at least 1,000 implemented configurations, with an explicit fidelity/repeat matrix and publication coverage target. This is a future objective, not an implication of a large schema. A smaller complete table may be released under its actual size. Expanding an existing table creates a new version and a new oracle universe.
4. Run offline policy comparisons on the complete measured panel and then fresh full-fidelity confirmations of locked finalists on separate scenes. Additional hardware is a separate optional cohort, admitted only with remaining budget.

[NAS-Bench-101](https://proceedings.mlr.press/v97/ying19a) makes cheap search comparisons possible by publishing actual repeated measurements at declared budgets. It is inspiration for the interface, not a license to treat unmeasured configurations or a surrogate's predictions as ground truth.

## Required row and failure semantics

Each append-only row identifies configuration/source/weights; physical GPU and complete runtime; fidelity/workload/batch definition; scene group, action, generation seed and repeat; process/block/order; initialization/compile/warmup/generation/export costs; measured latency and memory kind/units; quality-policy version and decision; raw/provenance hashes; and terminal status. Public artifacts sanitize private provider/transport identities while retaining stable anonymous device/cohort bindings. Preserve privately retained raw evidence and publish an auditable numerical reconstruction where licensing permits.

Allowed statuses distinguish success, correctness failure, quality failure, quality inconclusive, OOM, compiler error, timeout, transport/infrastructure failure, unsupported combination, and **not measured**. A timeout is censored duration, not an exact latency. Missing data are never converted to an OOM, a quality pass, a zero reward or an invented measurement. Preserve planned cells so coverage and selection bias are visible. No automatic retries erase failed attempts; any preregistered repeat is a separate row with its own cost.

An offline benchmark exposes only a fixed measured candidate panel and declared measured fidelity cells. Queries outside it return `unavailable` and are reported separately; they cannot enter a claimed full-space regret curve. The oracle is the best eligible configuration in that exact panel, with uncertainty in its measured value. Report **best-observed-panel regret**, not regret to an unknown global optimum. If no candidate passes the verifier, feasibility is zero and feasible regret is undefined, rather than manufactured as zero.

For a noisy table, use actual replicate draws under a frozen sampling rule and report finite-replicate reuse. Replaying table observations is explicitly offline simulation; it is not fresh GPU evidence. Repeated optimizer seeds do not create additional hardware measurements. Surrogate-imputed rows must be labeled predictions in a distinct artifact and cannot silently fill the benchmark table.

## Objective and fair comparisons

Keep feasibility separate from performance: correctness, execution integrity and a validated quality decision are mandatory. For eligible candidates, use a predeclared speed objective and measured noise floor. Expose graded constraint violations as development feedback, with frozen units/scales/weights, but do not let a large speedup compensate for a failed mandatory final gate. Publish latency–quality–memory Pareto sets, including uncertainty and missingness, rather than hiding tradeoffs in one scalar. Existing historical scores remain unchanged; any new reward is versioned and future-only.

Compare random search, scrambled Sobol, TPE, successive halving, and an LLM with and without history. Freeze their implementations, defaults, public observations and budgets. Use at least **20 paired optimizer seeds per method** on each admitted table task; these seeds measure algorithm variability. Include multiple held-out workloads/tasks before claiming generalization. For deterministic methods, disclose identical trajectories instead of counting copies as independent trials. LLM comparisons also pin model/version/prompt, history construction, sampling parameters and API cost; “without history” removes past feedback, not current task information.

Charge measured GPU-seconds, setup/compile costs and failures, alongside query count. Give all methods the same available domain and cost limits. Invalid attempts, repeated failed configurations and duplicate queries consume their registered budget. Sobol's mapping to conditional categorical options and TPE's treatment of inactive parameters must not grant one method a larger valid search space. Successive halving is eligible only if the table contains real lower-fidelity cells and the promotion rule/costs are frozen; a single-fidelity lookup cannot simulate it by inventing cheap outcomes. Measure proxy-to-full ranking behavior before asserting the proxy is useful.

Report regret versus charged budget, feasible success rate, repeat-failure rate, invalid-attempt cost fraction, and wall/API overhead. Use paired policy seeds and scene/task-cluster confidence intervals; do not pool all frames or table lookups as independent observations. Lock every method's selected finalists before any final feedback. Confirm each unique finalist and matched references in fresh runs; keep every selecting method's mapping and deduplicate shared candidates transparently. Use at least eight new scene groups, two seeds and three repeats (48 requests per selected configuration) for a first timing-confirmation pilot; use the larger verifier-specific count if required for its quality bound. The smaller confirmation does not bypass that requirement.

## Hardware transfer

A proposed second A100 or L40S cohort measures the same fixed panel or a prospectively stratified subset, including slow, fast and failure-prone configurations. Selection must not include only H100 winners. Use the same workload and verifier rules, but rebind runtime and compute baselines on that hardware. Report Spearman/Kendall rank agreement with scene/configuration uncertainty, feasibility changes and transfer regret against the best measured eligible candidate on the second device. OOM/unsupported cases are feasibility outcomes, not arbitrary ranks. One GPU of each type supports a limited case study, not a hardware-generalization claim.

## Cost feasibility

At the recorded planning point, approximately $257.90 had been spent against the strictly-below-$500 project cap, leaving **less than $242.10** once ongoing storage, settlement and any new costs are included. This is planning arithmetic, not a fresh provider balance or allocation approval. Using historical full-model 45–60 second requests and $4.746/GPU-hour gives these compute-only illustrations:

| Measurement count | Inference hours | Compute-only illustration |
| --- | ---: | ---: |
| 1,000 configs × 1 request | 12.5–16.7 | $59.33–79.10 |
| 1,000 configs × 4 scenes × 2 seeds × 3 repeats | 300–400 | $1,423.80–1,898.40 |
| 16 configs × 4 scenes × 2 seeds × 3 repeats | 4.8–6.4 | $22.78–30.37 |

These exclude compilation, warmups, repeated reference work, metrics, failed attempts, model-API calls, second hardware, evidence storage/transfer and cleanup. Altered configurations may be faster or slower; historical timing is not a safe upper bound. Therefore the full 1,000-configuration, repeated, quality-validated, multi-policy and multi-hardware roadmap is **not currently funded by the remaining cap**. A one-shot 1,000-row sweep is not an acceptable substitute for those claims.

Admit stages only after exact counts, worst-case time limits, raw-byte inventories, backup destination and cleanup reserves fit a new immutable contract. The staged pilot can justify the next scope or demonstrate that it is unaffordable. If the cap prevents expansion, release the smaller measured table, failed gates and deferred claims honestly. All proposed stages remain unrun.
