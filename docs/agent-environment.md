# Agent environment for inference optimization

The agent interface supports sequential configuration optimization. An LLM or another policy inspects development feedback, proposes a configuration, requests a profile or trial, and submits one immutable candidate. The evaluator controls inputs, measurement, correctness checks, quality policy, final scoring and budgets. The [Gymnasium operator environment](../extensions/kernel_tuning/README.md) supports smaller configuration-search tasks.

The included controller demonstration uses `InferenceEngine` with a synthetic CPU backend, a metric test double and scripted actions. Validation currently covers this CPU demonstration; full-Vista, empirical quality-calibration, LLM-driven model and adversarial isolation studies have not been completed. See the [validation status](benchmark-readiness.md).

The [study reproduction recipe](agent-study-reproduction.md) lists the evaluator-owned inputs, generation and metric commands, selection barrier, and artifacts required for a real run.

## Try the controller without a GPU

Install the core and extension, then run from any directory:

```sh
python -m pip install . ./extensions/kernel_tuning
drivebench-agent-demo --output ./agent-contract-demo
```

The new directory contains conditioning/input hashes, raw arrays for every request including warmups, a frozen fixture quality policy, reference/control metrics, an append-only action ledger, and the final assessment and score. `completed` records controller completion; the assessment determines acceptance. The perceptual metric is labeled as a test double. Rerunning into the same directory is refused.

## Agent tools

| Tool | Agent arguments | Trusted effect |
| --- | --- | --- |
| `inspect` | None | Public task settings, allowed values, latest proposal and budget state |
| `propose` | Nested `generation`/`optimizations` overrides; optional parent proposal ID | Validate allowlist and save a content-addressed configuration snapshot |
| `profile` | Proposal ID | Execute a charged diagnostic batch; profiler feedback is not a final speed score |
| `trial` | Proposal ID | Execute a charged development batch; no final inputs or final reward |
| `submit` | Proposal ID | Irrevocably select that snapshot before running the separately reserved final evaluation |

A GPU task must declare its physical device, software/source/weight identities, workload, development and final splits, proxy-quality policy, timing blocks, noise floor, time limits and evaluation budget before selection. The controller reserves calls and wall time for final evaluation. Invalid tools, failed measurements and OOM attempts consume their admitted budgets. Recovery aborts interrupted pending operations while preserving the original deadline and spent budget.

`AgentSession` accepts an executor callback that runs outside the policy and returns finite JSON with the exact cohort ID. Results or scores supplied in agent arguments are rejected. `final_evaluations` counts declared executor units; the executor and outer job guard enforce the requests within each unit. The fixed wall deadline includes policy waiting and failed work. An outer process timeout must interrupt hung compiler/CUDA calls.

## Vista configuration task

The Vista task exposes these supported options:

- FP16 or BF16 autocast.
- Decoder chunk sizes 1, 2, 7 or 14.
- Eager or the existing pinned default Inductor denoiser path.

These options give 16 combinations. Fifty denoising steps, 24 future frames at 576×1024, one future per request, trajectories, seeds, weights and disabled caches remain fixed. Model paths and compiler runtime selection are evaluator-owned. Actions are configuration changes; agent Python, custom kernels, package installs and code edits are outside the accepted action schema.

`vista_agent.run_batch` keeps one engine alive within a batch. It records initialization separately, preserves warmup arrays, synchronizes each measured request, checks raw outputs/counters/fallbacks and saves each result before continuing. Run each batch in a fresh worker with an external deadline. Latency starts with normalized conditioning arrays and ends with host-ready RGB; image-file decoding, construction, warmup/compilation and output export are timed separately. Stage timings describe batch components; kernel-level attribution requires a profiler trace.

`drivebench-vista-batch --spec /private/batch.json --spec-sha256 <pinned-hash>` is the worker entrypoint. Supply a private specification with model/runtime/source identities, a hashed manifest, explicit scene/seed/repeat requests, a fixed deadline and a new output directory. The worker enforces its schema, verifies original image/action hashes, decodes only conditioning images, and saves preparation and execution receipts separately. Run it through an external process timeout and verify worker exit before accepting completion. Hardware allocation and provider credentials are managed externally.

## Quality and reward

`agent_metrics.measure_pair` verifies raw-array hashes, excludes conditioning frames, computes RGB and temporal errors, and requires a supplied, version-bound LPIPS evaluator. Missing perceptual results stop the assessment.

`agent_evidence` reconstructs scoring records from pinned worker receipts and raw arrays. It verifies prepared conditioning inputs, joins matching scene/seed/repeat cells, and preserves generation timestamps. Final comparisons require the pinned selection barrier and execution order. The operator supplies metric callbacks, evaluates negative controls and verifies external worker completion.

`agent_scoring` requires a complete reference scene × seed × replicate matrix. Same-seed distances measure repeat noise; cross-seed distances describe reference variability, reported separately from candidate acceptance limits. Proxy margins are maximum-per-output limits justified by development evidence and frozen before candidate selection. Negative controls—frozen, black and reversed future frames—use the same development matrix. Missing cells or controls, or reference repeat noise above the limits, make calibration inconclusive. Threshold revisions must precede a new candidate-selection and evaluation run.

For three-policy comparisons, `selection_only_executor` records the submitted configuration and defers final evaluation. `freeze_final_plan` replays the three closed session ledgers, checks selections against eligible provisional trials, and writes an immutable ordered batch plan after every selection is locked. The plan binds the quality policy, source/input manifests, identities and equal trial budgets. The caller enforces it and withholds final feedback until all selections are fixed.

The [single-agent feasibility interface](agent-study-reproduction.md#single-agent-feasibility-interface) applies the same selection and final-evidence checks to one prospectively registered session and its reference.

Final quality assessment requires every declared final cell. A bound violation fails; missing evidence is inconclusive. An accepted timing score requires a complete passing assessment. Acceptance records agreement with the frozen proxy bounds. Visual quality, driving safety and trajectory fidelity require separate evaluation; `quality_preserving_claim` remains false.

After the final proxy assessment passes, the timing score is `max(0, 1 - worst_block_median(candidate/reference) - noise_floor)`. Blocks partition final scenes, with both execution orders for every scene and seed across repeats. Set the noise floor from reference timing variation before search. This descriptive score has no confidence-bound interpretation. Report when policies share reference measurements, since their comparisons are dependent.

## Trust boundary and LLM integration

An LLM policy receives the operator's public observation and returns a structured JSON action. The controller, GPU worker, private final inputs, metric assets and credentials stay outside its execution environment. Provider adapters must exclude host shell, filesystem, browser, MCP and cloud-control tools. Configuration allowlists prevent changes to checkpoints, evaluation code and timing functions.

The optional `ResponsesActionPolicy` accepts a configured official OpenAI client and an explicit model name. It requests strict structured output without provider tools, disables retries, and records usage metadata; the SDK is an optional dependency. See the [structured-output contract](https://developers.openai.com/api/docs/guides/structured-outputs).

API calls require a monetary budget in addition to call/token limits. `APICostLedger` and `BudgetedResponsesClient` reserve a caller-priced maximum before each request and retain that reservation after success or failure. Restarted policies use the same durable ledger. Verify prices, taxes, endpoint/tier and tokenizer or model-context bounds when setting reservations. Provider billing remains external to this ledger.

Construct `AgentSession`, a public-observation builder, and `ResponsesActionPolicy(client, model=your_pinned_model, max_calls=your_admitted_limit)`, then pass the selector to `run_agent`. The operator supplies the SDK and API credentials. Importing the package makes no requests.

The public-observation builder must exclude private calibration/final records, filesystem paths, authentication tokens and full worker exceptions. Keep evaluator storage inaccessible to the agent process. Policy/ledger hashes detect changes relative to a pinned root, but do not authenticate a directory an attacker can replace. The library relies on external process isolation. Supporting Python or kernel edits requires a separately audited execution and compilation boundary.

## Running a Vista validation study

A Vista study must declare its workloads, calibration data and policy-comparison budgets.

Calibration, trials and confirmation must share one fresh physical GPU and runtime. Withhold final inputs from the optimizer and report any prior public exposure. Freeze manifests, metric assets, source hashes, the calibration/noise procedure, prompts/model, study mode, policy budget and final execution order before the run. Bind the actual runtime after allocation, then freeze quality margins and the noise floor after development calibration and before candidate selection.

Before allocating paid hardware, verify current prices, funds, resource state and raw-backup capacity. Arm a new fixed-deadline resource guard with backup and cleanup reserves. Preserve failures and warmup costs, and publish the complete trajectory, score, costs and denied actions. Register matched budgets before any later policy comparison.
