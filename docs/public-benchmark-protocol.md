# Public benchmark measurement contract

This guide describes the shared measurement and evaluation rules. Each released study also has a frozen protocol that fixes its workload, settings and analysis. Use the [operator-generalization guide](operator-generalization.md) and [dataset report](../results/2026-11-23-operator-generalization-pro-4000/README.md) for the current benchmark.

## Suites and interfaces

| Suite | Workloads and actions | Episode budget |
| --- | --- | --- |
| `operator-generalization-v2` | 60 workloads: 30 training, 12 validation, 18 test; compiled PyTorch and 24 Triton configurations, with eager as the initial reference | Eight candidate trials in the released study |
| `operator-search-v1` | Earlier 12-workload suite: six training, two validation, four test; nine candidate settings | Four candidate trials by default |

The current suite contains GEGLU and GroupNorm+SiLU, each with ten shapes in float16, bfloat16 and float32. All dtypes of an exact shape stay in one split. Both operators occur in every split. These public, constructed splits support shape-family evaluation; policy training and selection must respect the declared split even though users can inspect the full table.

`BudgetedTuningEnv` supports measured-table replay and live evaluation through the same Gymnasium interface. Its generic task and candidate descriptors also accept external evaluators. The [configuration-agent tools](agent-environment.md) provide inspection, proposals, profiling, trials and final submission for a separate model-level workflow. Vista latency and quality follow that workflow's own contract, described below.

## Comparison identity

Record the suite bytes and SHA-256, source revision and hashes, candidate list and order, objective and weight, warmup/repetition settings, inputs, physical GPU, driver and software runtime. Keep different devices or runtimes in separate cohorts. A source or protocol hash identifies the recorded bytes; implementation review and measurements establish their behavior.

Define workload semantics and acceptance criteria before measurement. Changes to model weights, precision, sampling steps or resolution need explicit task definitions and quality criteria. A speed ratio only compares measurements with compatible boundaries and inputs.

## Budget and reward

A live episode begins with a fresh reference measurement. Replay obtains its reference from the measured table. The reference starts as the incumbent. Each candidate attempt, including repeated, rejected, incorrect or error results, consumes one trial. Live repeated queries produce fresh measurements; replay uses the available recorded repeats. Unexpected evaluator exceptions preserve the journal and abort the episode.

For a numerically accepted candidate, the normalized cost is

`cost = (L/L_ref + w * M/M_ref) / (1+w)`.

For operator tasks, `L` is median invocation latency and `M` is incremental peak allocated memory, including the output. The reference requires positive latency and allocated bytes, including when `w=0`. A zero-allocation reference needs a different normalization contract. The current study uses `w=0`, so cost depends only on latency. A nonzero weight changes the objective; report both latency and memory alongside it.

Reward is `previous_best_cost - new_best_cost`. Incorrect, error and rejected outcomes receive zero reward and still consume budget. Undiscounted return equals the final improvement over the reference. Exhausting the budget terminates the task, so RL training must not bootstrap beyond that decision. Observations distinguish untried actions, failed actions and measured outcomes, and include workload features, query history, incumbent and remaining budget.

## Timing, memory and numerical checks

Run live measurements serially on an otherwise idle NVIDIA GPU with the declared runtime. The operator evaluator uses five warmups and 30 timed invocations per measurement. CUDA events bracket one operator call and can include stream gaps. Initialization, input preparation, compilation, correctness checks and warmups are excluded from invocation latency and recorded in preparation or job wall time. Report cold-start and compilation costs separately when they affect the deployment being studied.

The CUDA evaluator requires `TORCHINDUCTOR_COMPILE_THREADS=1`, resets compiler state at workload transitions and permits one evaluator at a time per process. Fresh-process repeats provide separate process observations; the 30 invocations within a measurement are subsamples of that process.

Inputs are seeded tensors. Acceptance checks output shape and dtype, finiteness, numerical agreement with eager, and input immutability. Relative/absolute tolerances are 0.002/0.002 for float16, 0.02/0.02 for bfloat16 and 0.00002/0.000003 for float32. These are operator tolerances. Model-level perceptual quality needs separately calibrated criteria.

Memory is the allocator's peak live-allocation increase above resident inputs, including outputs and temporaries. Report reserved memory separately. Whole-model memory and saturated serving throughput need their own measurement procedures; the reciprocal of serial invocation latency describes serial operator throughput.

## Measured-table replay

The `measured_replay` backend returns recorded measurements with their original trial IDs and process-repeat identities. Repeated queries cycle through the finite repeats. Replay adds no independent GPU samples, regardless of the number of policy episodes.

Policies receive task/configuration descriptions and their own queried observations. Table regret is computed afterward from the complete measured table and is not exposed to the policy. The public table remains inspectable, and the local Python interface assumes trusted policy code. A refusal, invalid action or failed selector API call consumes a turn with zero reward and no candidate query. Report selector API cost separately.

The [current methodology](operator-generalization.md) defines process sampling, table cost, oracle regret and confirmation. Earlier replay uses the procedures in [measured operator search](measured-operator-search.md).

## Policy comparisons

Fix policies, training data, hyperparameters, budgets and objectives before test evaluation. Use validation for any tuning and disclose its use. Compare matched tasks, candidate order, objective, measurement settings and runtime. Record the initial reference as well as candidate queries when accounting for cost.

For live comparisons, prespecify policy order and interleave repeated episodes to reduce time-drift confounding. The single-policy CLI does not balance separate invocations automatically. The [serial comparison example](../extensions/kernel_tuning/examples/compare_search_policies.py) demonstrates matching task seeds, budgets and both grid/random execution orders; its included integration tests use synthetic measurements.

Publish individual trials, per-task outcomes, failures, repeated actions, training cost and actual wall time. A best-observed incumbent is affected by selection noise. Freeze choices before fresh confirmation against the reference and relevant baselines. Report the cost of any exhaustive sweep used to construct an oracle, and distinguish measured-table regret from freshly confirmed latency ratios.

The current release includes compiled-query, random, TPE, ridge, PPO and pretrained LLM selectors. Their [result reports](../RESULTS.md) specify the episode counts and comparison panels. The [live PPO/native-autotuning supplement](../results/2026-11-24-ppo-native-autotune/README.md#live-gpu-comparison) uses different search budgets for PPO and native tuning and reports those costs explicitly.

## Model-level adapters

The `InferenceEvaluator` bridge constructs a fresh engine for each trial and measures one cold first request. Engine construction is separate; context preparation through synchronized host-ready outputs includes first-use compilation. It adds no warmups or repetitions and requires a caller-supplied quality gate. See the [integration contract](../extensions/kernel_tuning/docs/extending.md).

The configuration-agent Vista worker has a warm-request boundary: predecoded conditioning through host-ready frames. Compilation and output persistence are excluded from latency and included in wall time. Its development proxies and final scoring require evaluator-owned inputs and a quality policy. The [agent guide](agent-environment.md) describes that interface's current validation coverage.

Neither operator agreement nor synthetic integration tests establish video quality or end-to-end model speedups. Historical Vista results retain their recorded hardware, request and quality boundaries in the [archive](../research/archive/README.md).

## Evidence and execution responsibilities

Each run retains its plan, identity hashes, raw trials, environment journals and completion or failure receipts. Use new output paths and preserve partial attempts. Synthetic examples identify their generated measurements as test fixtures.

An external evaluator implements its own measurement, input/reference binding and quality checks. The environment validates the returned contract but does not audit arbitrary participant code. Protect evaluator storage and final inputs from policy writes, and provide external process limits and cleanup. The library does not provision cloud resources or provide isolation for arbitrary agent-written code. The [coverage table](benchmark-readiness.md) distinguishes completed operator studies from model-level and code-action capabilities.
