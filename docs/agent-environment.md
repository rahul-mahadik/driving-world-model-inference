# Agent environment for inference optimization

The agent interface supports sequential configuration optimization. An LLM or another policy inspects development feedback, proposes a configuration, requests a profile or trial, and submits one immutable candidate. A trusted evaluator owns measurement, inputs, correctness, quality policy, final scoring and budgets. The existing Gymnasium/operator search remains available as a cheaper configuration-search tier.

**Current evidence for this interface: CPU contracts only.** The included episode uses the actual `InferenceEngine` with its explicitly synthetic backend, a numeric metric test double and scripted actions. No full-Vista run, real quality calibration, LLM optimization episode, adversarial sandbox validation or learned-policy advantage through this interface has been demonstrated. The original H100 speed result, failed quality qualification and all historical cohorts retain their original scope. See [readiness and claim boundaries](benchmark-readiness.md).

The [study reproduction recipe](agent-study-reproduction.md) lists the evaluator-owned inputs, generation and metric commands, selection barrier, and artifacts required for a real run.

## Try the controller without a GPU

Install the core and extension, then run from any directory:

```sh
python -m pip install . ./extensions/kernel_tuning
drivebench-agent-demo --output ./agent-contract-demo
```

The new directory contains conditioning/input hashes, raw arrays for every executed request including warmups, a frozen fixture quality policy, reference/control metrics, an append-only action ledger, the final assessment and score. A controller state of `completed` means its workflow finished; the separately recorded quality assessment and score determine acceptance. The fixture labels its fake perceptual metric explicitly; it does not download LPIPS, run Vista or call an LLM. Rerunning into the same directory is refused.

## Agent tools

| Tool | Agent arguments | Trusted effect |
| --- | --- | --- |
| `inspect` | None | Public task settings, allowed values, latest proposal and budget state |
| `propose` | Nested `generation`/`optimizations` overrides; optional parent proposal ID | Validate allowlist and save a content-addressed configuration snapshot |
| `profile` | Proposal ID | Execute a charged diagnostic batch; profiler feedback is not a final speed score |
| `trial` | Proposal ID | Execute a charged development batch; no final inputs or final reward |
| `submit` | Proposal ID | Irrevocably select that snapshot before running the separately reserved final evaluation |

A fresh real-world task must declare its physical GPU, software/source/weight identities, workload, development and final splits, proxy-quality policy, timing blocks, noise floor, time limits and evaluation budget before selection. The controller reserves both calls and wall time for final evaluation. Invalid tools, failed measurements and OOM attempts consume their admitted budgets. An interrupted pending operation aborts recovery; reopening cannot reset the deadline or refund work.

`AgentSession` accepts a trusted executor callback. The callback runs outside the policy and returns finite JSON with the exact cohort ID. Results or scores supplied in agent arguments are rejected. `final_evaluations` counts declared executor units, not individual frames: the executor and outer job guard must enforce the corresponding internal request count. The immutable wall deadline includes policy waiting and failed work, and an outer process timeout must interrupt a hung compiler/CUDA call.

## First Vista task

The first proposed model episode tunes these existing, supported options:

- FP16 or BF16 autocast.
- Decoder chunk sizes 1, 2, 7 or 14.
- Eager or the existing pinned default Inductor denoiser path.

This is a configuration task with 16 combinations. The agent sees feedback and can combine settings sequentially; a small discrete search space does not establish an advantage over grid or random search. Fifty denoising steps, 24 future frames at 576×1024, one future per request, trajectories, seeds, weights and disabled caches remain fixed. Model paths and compiler runtime selection are evaluator-owned. The same tool protocol can serve an operator executor, but this first Vista task does not accept agent Python, custom kernels, package installs or arbitrary code edits.

`vista_agent.run_batch` keeps one engine alive within a batch. It records initialization separately, preserves warmup arrays, synchronizes each measured request, checks raw outputs/counters/fallbacks and saves every result before continuing. Each batch is intended to run in a fresh externally bounded worker. The new measured boundary starts with normalized conditioning arrays and ends with host-ready RGB; image-file decoding, construction, warmup/compilation and output export are excluded and charged separately. **This boundary differs from historical results; the old H100 ratio cannot be carried over.** Stage timings are diagnostic, not a complete kernel profiler trace.

`drivebench-vista-batch --spec /private/batch.json --spec-sha256 <pinned-hash>` is the trusted worker entrypoint. The operator supplies a private, exact-schema specification with model/runtime/source identities, a hashed manifest, explicit scene/seed/repeat requests, fixed deadline and a new output directory. It verifies original image/action hashes, decodes only conditioning images, and saves preparation and worker receipts separately. Run it through an external process timeout and verify worker exit before accepting its completion. The library neither allocates a GPU nor manages provider credentials.

## Quality and reward

`agent_metrics.measure_pair` verifies raw-array hashes, excludes conditioning frames, computes RGB and temporal errors, and requires an explicitly supplied, version-bound LPIPS evaluator. Missing perceptual results fail closed. It does not silently replace LPIPS with another metric.

`agent_evidence` reconstructs scoring records from pinned worker receipts and raw arrays. It verifies the prepared conditioning inputs, joins matching scene/seed/repeat cells, and preserves the original generation timestamps. Final comparisons also require the pinned selection barrier and its execution order. Metric callbacks and negative-control evaluation remain trusted operator responsibilities; this loader does not certify that an external worker exited or that a GPU study was performed.

`agent_scoring` requires a complete reference scene × seed × replicate matrix. Same-seed repeat distances characterize repeat noise; cross-seed distances describe reference variability. These are distinct quantities. A large seed spread does not automatically permit large candidate error. Proxy margins are explicit maximum-per-output limits, selected and justified using development evidence and then frozen before candidate selection. Negative controls (frozen, black and reversed future frames) must be evaluated on the same development matrix. Insufficient coverage, missing controls, or reference repeat noise exceeding the limits make calibration inconclusive. Any threshold revision is a new policy, never an adjustment after seeing final candidate results.

For a comparison of multiple search policies, `selection_only_executor` makes `submit` record a selection without evaluating final data. `freeze_final_plan` replays all three closed session ledgers, checks selected configurations against eligible measured provisional trials, and writes one immutable ordered batch plan only after every selection is locked. It binds the frozen quality policy, source and input manifests, identities and equal trial budgets. The caller must enforce that plan and keep final feedback hidden until every selection is locked. This adapter is bookkeeping and does not allocate hardware or launch jobs.

Final quality assessment requires every declared final cell. A breach fails; missing evidence is inconclusive. Neither receives an accepted timing score. Passing means only that the evaluated outputs met the frozen proxy bounds. Perceptual similarity is not proof of visual equivalence, driving safety, or trajectory fidelity. Actual visual acceptance must be reviewed separately; `quality_preserving_claim` remains false.

The timing score is `max(0, 1 - worst_block_median(candidate/reference) - noise_floor)` only after the complete final proxy assessment passes. Blocks partition final scenes; every scene and seed has both execution orders across repeats. The floor is prespecified from reference timing noise before search, never fitted to a winning candidate. This is a conservative descriptive score, not a statistical confidence bound. Comparisons sharing fresh reference measurements across policies are dependent and must be disclosed.

## Trust boundary and LLM integration

The first task uses an LLM as a structured action selector. It receives only the operator's public observation and returns JSON. The controller, GPU worker, private final inputs, metric assets and credentials stay outside the policy's execution environment. A provider adapter must expose no host shell, filesystem, browser, MCP or cloud-control tools. Config allowlists prevent an action from changing checkpoints, evaluation code or timing functions.

The optional `ResponsesActionPolicy` accepts an already configured official OpenAI client and an explicit model name; the package has no mandatory SDK dependency. It requests strict structured output with no provider tools, disables request retries, and preserves bounded usage metadata. Calls still incur the provider's usage charges; call/token limits are not a dollar billing limit. Use a separately admitted API budget. `APICostLedger` and `BudgetedResponsesClient` reserve a caller-priced maximum before each outgoing request and never refund it after success or failure. A restarted policy remains bound to the same durable ledger. Prices, taxes, endpoint/tier and the chosen tokenizer or model-context upper bound still need independent verification; a local reservation is not a provider billing control. See the [official structured-output contract](https://developers.openai.com/api/docs/guides/structured-outputs). A model asked to produce actions is an LLM policy; it is not evidence that the policy was trained with RL.

The integration is explicit: construct a trusted `AgentSession`, a public-observation builder, and `ResponsesActionPolicy(client, model=your_pinned_model, max_calls=your_admitted_limit)`, then pass the selector to `run_agent`. The optional SDK and API credentials are operator-owned; importing the package never makes a request.

`run_agent` requires an explicit public-observation builder. Never pass private calibration/final records, raw filesystem paths, authentication tokens or full worker exceptions to it. Run/evaluator storage must be inaccessible to any agent process. Policy/ledger hashes detect accidental mutation relative to a pinned root; they are not signatures against an attacker able to replace the whole directory. The library does not create an OS sandbox. Arbitrary Python/kernel edits require a separately audited isolation and compilation boundary and are outside this configuration-only release.

## Full-model demonstration still to execute

The proposed feasibility study calibrates on eight development scenes with two seeds and two repeats. An LLM, random search and grid search each receive four candidate trials on two development scenes. Their locked selections then receive fresh confirmation on four separate final scenes, with two seeds and two repeats. This small demonstration cannot establish a broadly calibrated quality limit, policy superiority, or generalization. It uses a new, explicitly smaller frozen policy; it does not replace the original study. All policies use one physical GPU and runtime, receive the same allowed settings and development measurements, and preserve failures and warmup costs. The final inputs must be withheld from the optimizer; existing public/historically visible inputs cannot be described as globally secret or novel. The study must report that limitation.

Before any paid run: freeze the concrete manifests, metric assets, source hashes, calibration/noise procedure, prompts/model, three policy budgets and final execution order. Bind the actual runtime after allocation and freeze the justified margins/noise floor after development calibration but before any candidate selection; verify raw backup capacity; audit current prices/funds/resources; arm a new immutable resource guard with complete backup and cleanup reserves. No allocation is performed by this library. Publish the complete trajectory, score, failures, cost, denied actions and equal-budget baselines. One episode demonstrates integration, not RL superiority or robust quality preservation.
