# Reproduce a configuration-agent study

This recipe describes the trusted operator's boundary for a new Vista study. It does not download restricted data, obtain credentials, provision a GPU or run merely by installing the package. The [small CPU demo](agent-environment.md#try-the-controller-without-a-gpu) is the installation check; it is not a substitute for the steps below.

Use a new output root and one dedicated physical GPU/runtime for calibration, search and final comparison. Keep previous runs intact. Obtain Vista weights and driving inputs separately under their terms, following [Vista integration](vista-integration.md). Keep the historical Vista runtime separate from the newer operator Torch/Triton environment. Record the actual source, model, input and runtime identities rather than copying an identity from a previous result.

The trusted controller and worker launchers are included in the checkout; no private orchestration source is needed. Read [host launcher setup](../tools/host/README.md) for its Linux/Docker/runtime requirements. The portable adapter accepts an operator-built image and interpreter pinned in the host template's `runtime` object; it never pulls an image. The preserved H100/A100 adapters instead require their original pinned local image and interpreter paths. The repository does not distribute prebuilt Vista environments. A historical environment proof is not valid for a new GPU. This remains an operator-run integration, not a one-command cloud deployment.

## Prepare evaluator-owned artifacts

The policy receives none of these private paths or raw records. Keep secrets outside the source repository and study publication tree.

| Artifact | Contents and role |
| --- | --- |
| Generation source manifest | JSON `{"source_files": {"/absolute/file": "sha256", ...}}`, including the worker's required sources and upstream model/config files |
| Input manifest bundle | JSON `{"manifests": [...]}` containing the exact manifest pins admitted for the study; each pin is the generation spec's `manifest` object |
| Batch specification | Exact-schema `vista_batch.SPEC_KEYS`: model/settings/runtime/source bindings, manifest, explicit scene/seed/repeat requests, new output directory, deadline and free-space reserve |
| Perceptual setup receipt | Hash-pinned offline LPIPS package and model-weight assets prepared by the existing [perceptual setup script](../scripts/prepare_perceptual_evaluation.py) |
| Quality plan | Development/final scene lists, seeds, repeats, explicit proxy margins, sample floors, timing partitions and the prospective noise-floor rule |
| Session policies | Identical candidate/trial budgets for LLM, random and grid; wall limits, final reserve, allowlisted values and quality/source/input identity pins |

All path/hash pins use an absolute `path` and exact file-byte `sha256`. A source manifest is not a generic promise that the files are unchanged: the batch worker and metrics worker verify their own required source sets. `vista_batch.required_source_paths(model)` and `agent_metrics_worker.required_source_paths()` expose those required paths. Do not derive a final policy's input list from feedback or silently replace a missing artifact.

For a real metric spec, `metric` contains `kind: "lpips"`, the pinned setup receipt, `device: "cpu"` or `"cuda"`, and exact versions for `torch`, `torchvision`, `numpy`, `scipy` and `lpips` (0.1.4). The worker validates assets and disables network use during metric initialization/evaluation. Pre-stage the assets; no fallback metric is accepted. `{"kind": "synthetic_fixture"}` is reserved for explicitly synthetic batches and is not LPIPS.

## Generate and calibrate

Run each batch in a fresh externally bounded process. The installed entrypoint and equivalent module form are:

```sh
drivebench-vista-batch --spec /private/study/calibration-batch.json --spec-sha256 "$CALIBRATION_SPEC_SHA256"
python -m drivebench_tuning.vista_batch --spec /private/study/calibration-batch.json --spec-sha256 "$CALIBRATION_SPEC_SHA256"
```

Replace the example path with the prepared artifact and set `CALIBRATION_SPEC_SHA256` to its reviewed file-byte digest. The two forms are alternatives, not instructions to run the same batch twice. The outer launcher must enforce the deadline even if CUDA hangs. Verify worker exit and cleanup before accepting `adapter-completed.json`. Preserve raw measured arrays, warmup arrays, per-request records, preparation receipts, failures and completion hashes.

Then run the metric worker with a `kind: "calibrate"` specification:

```sh
drivebench-vista-metrics --spec /private/study/calibration-metrics.json --spec-sha256 "$METRIC_SPEC_SHA256"
```

Set `METRIC_SPEC_SHA256` to that reviewed metric spec's digest. Every metric spec has `schema_version`, `kind`, `synthetic`, `created_at`, `deadline_unix`, `minimum_free_bytes`, `output`, `metric`, `references`, `sources` and `study_bindings`. Each `references` item pins a batch `directory`, `role`, `expected_spec_sha256` and `expected_completion_sha256`. `study_bindings` pins `source_manifest` and `input_manifest` files. The exact field sets live in [the worker](../extensions/kernel_tuning/drivebench_tuning/agent_metrics_worker.py); unknown or missing fields are rejected.

Calibration adds `quality_plan`. Declare `noise_floor_method: "maximum_paired_repeat_relative_difference"`; the worker freezes the larger of the prespecified floor and maximum paired reference-repeat relative difference. This is a descriptive noise floor, not a confidence bound. Declare the proxy margins prospectively; the worker does not choose or relax them to obtain a pass.

Calibration outputs include `quality-policy.json`, `calibration-records.json`, `timing-noise.json`, `metric-provenance.json`, control receipts and `completed.json`. The same-seed-repeat and cross-seed metrics are separate. A successful worker completion can still contain `calibration_status: "inconclusive"`; search must not start unless calibration is `ready`.

Frozen, black and reversed future controls are generated one at a time. After a successful control's recipe, output hash and metric receipt are durable, only its derived scratch array is removed. Original generated arrays remain untouched; failed scratch is preserved. A control receipt retains a recipe for independently regenerating the derivative. Its timestamp describes control generation, while original model outputs retain their original generation times.

## Search, lock and confirm

The trusted trial executor runs fresh reference and candidate batches on the declared development subset, then invokes a `kind: "trial"` metric spec. It adds `candidates`, `quality_policy`, `selection_started_at`, `candidate_sha256` and explicit `trial_cells` to the common fields. Results include paired records, proxy maxima, failures and median reference/candidate latency. `screening_status: "no_observed_breach"` is provisional subset screening, not a final quality pass.

Expose only the declared task, legal settings, remaining budgets and permitted development feedback to the policy. Use `AgentSession` and `run_agent` for the tool loop; the [agent guide](agent-environment.md#trust-boundary-and-llm-integration) describes selector and cost-ledger integration. A policy's own text cannot supply measured timing or a successful quality result.

The installed `drivebench-agent-trials` command connects this interface to the generation and metrics workers. Its episode spec has `schema_version`, `output`, `created_at`, `deadline_unix`, `session_policy`, `batch_template`, `metric_template`, `transport`, `trial_timeouts`, `selector` and `max_turns`. The transport pins `root`, `launcher`, `launcher_sha256`, host `python`, `host_template` and `max_jobs`. The exact schemas are in [agent_trials.py](../extensions/kernel_tuning/drivebench_tuning/agent_trials.py). For a complete nested schema reference, see `episode()` and `test_concrete_grid_episode_uses_fresh_paired_workers_and_only_selects` in [the integration tests](../extensions/kernel_tuning/tests/test_agent_trials.py). Their in-process transport and synthetic batches are test fixtures, not an admitted real GPU specification. The optional episode `trial_schedule` prospectively fixes one seed/replicate choice for each trial, covering every fixed trial scene; use the same schedule for compared policies. The optional transport `package_root` selects a pinned evaluator-source copy within the host root, so a new cohort can stage its adapter separately from preserved source. Both additions are operator-owned, frozen before execution, and do not let the policy select code or trial inputs. All episode artifacts belong to the trusted operator; never send the whole spec or raw worker receipts to an LLM.

For a policy running outside the GPU host, set `selector` to `{"kind": "external-json"}`. Initialize once, then pass one pinned JSON action per call through your existing authorized transport:

```sh
drivebench-agent-trials --spec /private/study/episode.json --spec-sha256 "$EPISODE_SHA256" --operation initialize
drivebench-agent-trials --spec /private/study/episode.json --spec-sha256 "$EPISODE_SHA256" --operation observe
drivebench-agent-trials --spec /private/study/episode.json --spec-sha256 "$EPISODE_SHA256" --operation act \
  --action /private/study/action-001.json --action-sha256 "$ACTION_SHA256"
```

Set both digest variables to the reviewed file-byte hashes. The action envelope is `{"tool": "...", "arguments": {...}}`: `inspect` takes `{}`, `propose` takes `{"overrides": {...}}` and an optional `parent_id`, and `trial`/`submit` take `{"proposal_id": "..."}`. This cohort has a zero profiling budget. Allowed configuration values, proposal identifiers and prior trial feedback come from the public observation; full tool behavior is documented in the [agent guide](agent-environment.md#agent-tools). The command prints the deliberate public observation projection, which is suitable for the policy. Keep the LLM API credential and its cost ledger on the calling machine in this mode. No background loop runs between calls and the immutable episode deadline does not move. Failed or interrupted workers halt further jobs; changing the output name is not a reason to retry them automatically.

For the two deterministic baseline controllers, use an explicit `{"kind": "grid", "seed": 1901}` or `{"kind": "random", "seed": 1901}` selector and `--operation run`. The optional `openai-responses` selector runs on the trusted host and requires the separately installed SDK, a pinned model and prospective API cost policy; it has no fallback policy. Both routes use the same trial evaluator and budgets. `completed.json` from a search episode records **selection only**; it is not final quality or timing evidence.

For an LLM/random/grid comparison, `selection_only_executor` records each submitted choice without running final inputs. Close all three sessions, then call `freeze_final_plan` with their externally pinned session policies and the frozen quality/source/input identities. It admits only selections backed by an eligible completed trial. Its immutable plan fixes role order and scene/seed/repeat cells. Execute that exact plan serially; final feedback must not influence another policy's selection.

For each selected candidate, use a `kind: "final"` metric spec. It adds `candidates`, `quality_policy`, `selection_started_at`, `candidate_sha256` and pinned `final_plan`. It reconciles the final plan's aggregate source/input pins with batch provenance and requires generation after all selections were locked. The output contains `paired-records.json`, `quality-assessment.json` and `timing-score.json`.

A failed or incomplete quality check receives no accepted timing score. An accepted descriptive score does not establish visual equivalence or policy superiority. Sharing the same fresh reference batches across selected policies makes their comparisons dependent; report that design explicitly.

## Export review material and inventory evidence

The trusted artifact worker prepares review material and a terminal file inventory. It does not generate model outputs, perform a visual review, verify a backup or retire resources:

```sh
drivebench-agent-artifacts --spec /private/study/export.json --spec-sha256 "$EXPORT_SPEC_SHA256"
drivebench-agent-artifacts --spec /private/study/inventory.json --spec-sha256 "$INVENTORY_SPEC_SHA256"
```

Each variable is the reviewed file-byte digest of its separate spec. Both specs use `schema_version`, `kind`, `synthetic`, `created_at`, `deadline_unix`, `minimum_free_bytes`, `maximum_output_bytes`, `output` and `sources`. Run them through an external timeout and verify terminal cleanup. The output byte limit includes the completion receipt on success. It is a checked bound, not an operating-system quota: a failure may retain the last attempted bounded artifact plus failure metadata beyond that limit. The exact field sets are in [agent_artifacts.py](../extensions/kernel_tuning/drivebench_tuning/agent_artifacts.py).

An `export` spec adds pinned `final_plan`, `quality_policy`, `review_plan` and three `comparisons` entries, one per LLM/random/grid policy with its final `metric_completion`. The review plan fixes scene/seed/repeat cells before the selection barrier; it cannot choose attractive cases after seeing results. Select only one replicate for each scene/seed pair because the preserved renderer groups cases by scene, seed and policy. The worker verifies the final metric-to-array provenance and replays the recorded assessment and timing score, without recomputing LPIPS. It uses the existing renderer for blinded contact sheets and motion previews, and adds every future frame as a native-resolution A/B PNG pair. These PNGs preserve rounded 8-bit display pixels, not the original float32 array values; original arrays remain untouched.

Only give reviewers the `public/` packet. Its name denotes the blinded review view, not permission to publish licensed frames. Keep `private/` and the surrounding spec/provenance receipts evaluator-only; seal the reviewer's actual observations before revealing the answer key. A completed export explicitly reports `visual_review_performed: false` and cannot establish quality preservation.

An `inventory` spec adds explicit absolute `roots`, `extra_files`, `required_pins` and `maximum_input_bytes`. Include every current-study tree, including failed and partial attempts. Keep the new inventory output outside those roots; overlapping roots and symlinks are rejected. The worker hashes all declared files twice, requires the same file set and bytes, and retains partial progress on failure. `inventory-verified.json` verifies that declared local scope; it does not certify that the planned study completed or that all needed roots were declared. Copy the evidence under the admitted backup process, independently rehash the destination, and separately verify process/provider cleanup.

## Publish evidence, not a completion label

Verify worker exit, source/input invariants, the final evidence inventory and an independently hashed backup before releasing resources. Publish the exact protocol/configuration and sanitized trajectories, measured outcomes, rejected attempts, costs and limitations. Retain raw arrays according to the declared study contract and external asset terms. A completed process, a passing CPU test suite and a completed real-model evaluation are separate milestones.

The current project proposal is the explicitly small 8-development/4-final-scene feasibility study described in [readiness](benchmark-readiness.md). Its real GPU/LPIPS/LLM execution remains unrun. This recipe does not relabel historical timings or complete the old same-GPU held-out study.
