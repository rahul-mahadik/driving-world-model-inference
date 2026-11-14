# Modern baseline measurement: startup, warm execution and attribution

**Status: implemented and CPU contract-tested; modern Vista GPU measurements are unrun.** This protocol supersedes the canceled three-policy study for future work. It does not reuse that study's allocation, deadlines, quality claims or source bindings. The existing historical evidence stays unchanged.

The initial measurement table has three rows:

| Baseline | Implementation | Evidence in this release |
|---|---|---|
| `legacy_eager` | Original Vista eager, Torch 2.0.1 / CUDA 11.8, native xFormers | Historical October 30 H100 cohort only; 200 requests over 50 scenes. Startup and restart measurements are unavailable in this table. |
| `modern_eager` | Explicit native SDPA modern port, pinned modern stack and parameter/layout policy | Unrun on GPU. |
| `modern_compiled` | Exactly the same modern port, with only denoiser compilation enabled | Unrun on GPU. |

The historical row gives context. It is not a matched control for the modern pair: hardware instance, software, scenes and dates differ. The analyzer does not pool it, calculate a causal cross-runtime speedup, or fill missing cold measurements from the old warm latencies. There is no new `legacy_compiled` run.

## Prespecified fresh-process design

The prospective default is **three fixed scenes, twelve paired process blocks, two modern roles per block**: 24 fresh processes. Each process generates three measured requests, one per scene, after at least three and at most twelve recorded cold-prefix requests. This gives 72 warm measurements and 144–360 total requests before separate profiles. Pair orders are seeded and reversed in adjacent blocks (AB/BA). Scene membership, seeds, action inputs, model weights, generation parameters and source/runtime identities are frozen before evidence.

All modern blocks require one explicitly declared dedicated GPU and an identical full modern runtime. The new first feasibility candidate is an RTX A6000 48 GB; an A100 80 GB is a separate higher-memory fallback only if a new plan declares it. No fit or capacity is assumed. The frozen hardware policy declares the exact reported GPU name, minimum GPU memory, minimum host RAM and whether MIG is supported. The collector binds that complete policy to a fresh modern preflight proof and verifies the shared physical UUID; it never pools devices or silently changes resolution/step count. The same weights, native attention, parameter dtypes, memory layout, sampler, conditioning and request settings apply to both roles. The compile flag is the sole permitted role difference. Modernization itself requires a separate correctness/quality audit; a CPU import or unit test is not that audit.

The statistical unit is the **four-process AB/BA reversal group**. Twelve paired blocks contain six such randomized groups; keep both orders and every request/scene inside each resampled group. The group bootstrap estimates runtime variability conditional on these three fixed scenes. It does not turn 72 requests into 72 independent scenes, establish broad scene generalization, or certify tail latency. Three within-process requests are insufficient for a useful p95; it remains unavailable.

## Measured boundaries

The new `drivebench_tuning.restart_measurement` single-worker entry point lives in the existing extension and wraps `drivebench.benchmark.run_benchmark` and `InferenceEngine`; it does not create a resource or job orchestration framework. A scoped factory instantiates `ModernVistaBackend` explicitly. The unchanged legacy validator sees its existing `vista` marker internally; `restart.json` binds the original modern configuration and the actual backend identity in `model_identity.json`. It is not relabeled as legacy execution. The frozen `drivebench` package and its source fingerprint remain unchanged. Receipts separately pin the four new measurement/profile/analysis helpers and both modern model-port files before and after execution; paired comparisons require the same combined implementation identity.

Report these separately:

1. **Model initialization:** the existing synchronized engine-construction timer. It includes checkpoint loading and model initialization inside that boundary. Interpreter/import/environment-probe/CUDA-context startup preceding it is not included.
2. **First request:** the first recorded cold-prefix request, from input loading/preprocessing through host-ready output. It includes lazy compilation if it occurs.
3. **Complete cold prefix:** initialization plus every recorded warmup request, including failures and stabilization attempts. Successful analysis requires a complete prefix and no compilation during the subsequent measured region.
4. **Warm steady state:** the equal-scene mean and median per process, excluding initialization, cold-prefix requests and synchronous output export; allocator peak allocated/reserved bytes are reported separately.
5. **Measured worker-body elapsed time:** also retained for cost accounting. It begins after plan validation, cache setup and the start record, and excludes interpreter startup; full process/allocation lifetime must come from the outer runner. It is not substituted for model request latency.

`compile_only_s` is **null**. First-request minus warm latency, or compiled-minus-eager cold prefix, mixes compilation with allocation, autotuning and other setup. It must not be called pure compilation time. Positive captured-graph evidence is required for the compiled role, but it does not prove full-graph execution or the absence of all graph breaks.

A fresh PID alone does not make compilation cold. Before any Torch import, each worker creates new empty, process-owned `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` directories and disables optional remote compiler caches. Paths and the initial empty condition are recorded. They are preserved with the attempt. This does not clear OS page caches, checkpoint downloads, CUDA driver caches or persistent hardware state. PyTorch documents that compiler artifacts can persist across runs and affect startup latency. [Official compilation-cache guidance](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html).

Synchronization is already part of the existing engine and timing boundary. Asynchronous submission alone is not completed GPU work. [Official PyTorch benchmark guidance](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html).

## Break-even and uncertainty

For process `r`, let `I_r` be initialization time, `C_r` the sum of its `k_r` cold-prefix requests, and `W_r` its mean warm latency. For horizons at least as large as both recorded prefixes, use:

```
T_r(N) = I_r + C_r + (N - k_r) W_r
```

The first strict crossing is computed from the mean paired intercept difference and mean paired warm saving. It is an **extrapolated workload model**, not an observed run lasting that many requests. All measured cold-prefix requests count as served requests; they are not charged twice.

Resample the six complete AB/BA reversal groups 4,000 times with a frozen seed. Those draws are not additional replication. At least four complete reversal groups are required by this analyzer; serial dependence spanning groups remains a limitation. If a draw has negative warm saving, or equal warm cost without a startup advantage, its lasting amortized crossing is infinite. Equal warm cost with strictly lower startup is already faster at the minimum modeled horizon. Keep these draws, report their fraction, and allow an unbounded upper confidence limit. Do not condition the interval on only the winning draws. No finite break-even, and no statistically resolved warm gain, are legitimate outcomes. Failures, missing blocks, source/runtime drift, changed scenes and recompilation invalidate the complete-case estimate; none is silently dropped or replaced.

The worker analyzer checks its own evidence. **Independent existing host-runner receipts for process exit, idle GPU and cleanup remain mandatory before accepting a live result.** Its output explicitly leaves that verification false; CPU fixtures cannot make it true.

## Cheaper full-workload feasibility first

Do not start all 24 workers before confirming memory fit and observed cost. A separate `--mode feasibility --blocks 1` plan uses one predeclared real scene, one new eager process and one new compiled process, each with the same full 50-step, 25-total-frame, 576×1024 workload, three to twelve cold-prefix requests, and one measured warm request. It therefore needs 8–26 generation requests, including lazy compilation and stabilization. Use dedicated one-scene configurations with `benchmark.requests: 1`, `save_outputs: 1`; do not shorten the model workload. This is an execution and memory feasibility smoke, not a quality evaluation or speedup/noise estimate.

The collector still records failures/OOM and preserves outputs. Report peak allocated/reserved memory, cold prefix, first request, warm request and observed worker-body cost at an explicitly supplied rate. **Cost per quality-valid request remains null** until a defensible quality verifier exists. The smoke does not automatically admit the larger restart study; actual memory headroom, whole-process/allocation overhead and fresh prices determine a new bounded decision.

Materialize that smoke separately with its declared hardware and a freshly checked rate:

```bash
python -m drivebench_tuning.restart_measurement plan \
  --eager-config configs/research/modern-eager-feasibility.yaml \
  --compiled-config configs/research/modern-compiled-feasibility.yaml \
  --mode feasibility --blocks 1 --gpu-name "NVIDIA RTX A6000" \
  --minimum-memory-mib 45000 --compute-usd-per-hour 0.888 \
  --output outputs/feasibility-plan.json
```

The shown rate is a dated illustration, not a current quote. Both configurations still use the unchanged full model workload.

For price context, a provider catalog checked on November 14 quoted RTX A6000 $0.888/h, A100 80 GB $2.386/h and H100 $5.241/h. At these compute-only rates, RTX A6000 could be about 5.90× slower than H100 and still have the same compute cost per request; A100 about 2.20×. This is arithmetic, not a measured speed or memory-fit prediction. Fresh rates and actual completed-work costs must replace the illustration.

## Reproduction interface

First install the base package and the existing kernel-tuning extension in the separately pinned modern environment. Precache the complete CLIP assets and set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` before starting the process; the modern backend rejects network-dependent initialization. Prepare the required checkpoint/data and run the new declared-device verifier through the existing protected runner. The historical H100-only verifier is unchanged and is not used for a cheaper-device admission. The examples below are commands for a future admitted run, not evidence that one occurred.

For the proposed RTX A6000 policy, the prospective command is:

```bash
python -m drivebench_tuning.modern_verify_environment \
  --gpu-name "NVIDIA RTX A6000" --minimum-memory-mib 45000 \
  --minimum-host-ram-gib 32 --mig-unsupported --dedicated \
  --output outputs/modern-environment \
  --write-lock outputs/modern-environment/runtime-lock.json
```

This collects actual CUDA tensor, native runtime, Nsight trace and SM-counter evidence, plus idle/single-device/runtime continuity checks. It does not load Vista or establish model memory fit or quality. The worker archives the full proof and lock and rejects an old H100-only proof. Use the same policy when materializing the plan; `--minimum-host-ram-gib 32` and MIG unsupported are the plan defaults. A MIG-capable device needs a new plan with `--mig-supported` and a verifier invocation without `--mig-unsupported`; observed MIG must still be disabled.

Create two fully resolved configurations that differ only in `optimizations.compile_denoiser`; use `model.backend: modern_vista` and the exact `modern_runtime` contract described in [the modern runtime documentation](modern-vista.md). Select exactly three scene records in the same manifest. Set `benchmark.requests: 3`, `warmup_requests: 3`, `max_warmup_requests: 12`, `headline: false`, `candidate_count: 1`, and `input_condition: uncached`.

```bash
python -m drivebench_tuning.restart_measurement plan \
  --eager-config configs/research/modern-eager.yaml \
  --compiled-config configs/research/modern-compiled.yaml \
  --blocks 12 --gpu-name "NVIDIA RTX A6000" --minimum-memory-mib 45000 \
  --output outputs/restart-plan.json
```

For each scheduled run, the existing bounded host runner starts a **new interpreter** with a frozen plan SHA256 and immutable externally enforced deadline:

```bash
python -m drivebench_tuning.restart_measurement run \
  --plan outputs/restart-plan.json --plan-sha256 PLAN_SHA256 \
  --run-id RUN_ID --deadline-unix IMMUTABLE_DEADLINE \
  --output outputs/restarts/RUN_ID
```

The worker does not enforce a stuck CUDA call's wall deadline itself. That termination and owned-container cleanup belong to the existing external runner. It refuses a second measurement in the same interpreter, preimported Torch, reused output directories, changed plans or expired boundaries. Failed attempts remain on disk.

Analyze all scheduled runs; an incomplete set is rejected. An optional hash-pinned historical summary fills only the `legacy_eager` row:

```bash
python -m drivebench_tuning.restart_measurement analyze \
  --plan outputs/restart-plan.json --plan-sha256 PLAN_SHA256 \
  --runs outputs/restarts/* --output outputs/restart-summary.json \
  --legacy-summary results/2026-10-30-h100-tradeoff/summary.json \
  --legacy-summary-sha256 HISTORICAL_SUMMARY_SHA256
```

## Runtime attribution without double counting

Profile separate requests; profiled execution never enters the timing table. The new `drivebench_tuning.modern_profile` wrapper reuses the existing profiler with an explicit modern backend factory. It supports only the in-process Torch tool: legacy Nsight child commands would lose that adapter and are rejected. The wrapper emits the Chrome trace, original modern configuration, actual model/source identity, archived environment proof/lock and a single request annotation around preprocessing through synchronized generation. It rejects source/input drift, missing CUDA kernel activity inside that annotation and compilation during the profiled request. The new analyzer provides **three separate diagnostic views** of one explicitly selected profiled-request window:

- host stage submission ranges;
- exclusive host operator ranges, subtracting nested parents;
- recorded GPU kernels/transfers by device timeline, with overlapping streams counted once in an explicit overlap bin.

Each view partitions the window into named work, overlap/ambiguity, and unassigned/idle intervals. Never add the views. Host-stage ranges are not GPU stage latency; kernel names are not an inferred high-level operator mapping. No CPU timestamp correlation or unverified kernel attribution is invented. `unassigned_or_idle` means the view lacks a named event, not proof that the whole GPU was idle. The old analyzer's inclusive operator totals remain useful diagnostics, but must not be stacked into a latency breakdown. [Official profiler documentation](https://docs.pytorch.org/docs/2.15/profiler.html).

```bash
python -m drivebench_tuning.modern_profile \
  --plan outputs/restart-plan.json --plan-sha256 PLAN_SHA256 \
  --role modern_eager --tool torch --deadline-unix IMMUTABLE_DEADLINE \
  --output outputs/profile

python -m drivebench_tuning.attribution \
  --trace outputs/profile/torch-trace.json \
  --start-us REQUEST_START_US --end-us REQUEST_END_US \
  --output outputs/attribution.json --figure outputs/attribution.png
```

Read window endpoints from `modern_profile.json` → `request_window`; they are reconstructed from the single recorded request range, not chosen to hide work. The renderer uses the existing optional `plots` dependency. All charts retain `profiled:true` and `headline_eligible:false`.

## The later configuration corpus is a separate cost decision

At least 1,000 configurations means at least `1000 × scenes × repeats` candidate measurements, before reference outputs, calibration, warmups, profiles, failures, quality review, copying or cleanup. The calculator makes this product explicit:

```bash
python -m drivebench_tuning.study_cost \
  --configurations 1000 --scenes 3 --repeats 2 \
  --output outputs/corpus-cost-projection.json
```

For **1,000 × 3 × 2**, that is **6,000 candidate requests**. The historical 45–60-second range would imply **75–100 GPU-hours** and roughly **$356–475 of compute** at the old $4.746/hour H100 price, before all excluded costs. It is an illustration, not a latency lower bound: modern/lossy settings may be faster, and compilation/failures may be slower. Fresh measurements and quotes must replace these assumptions before admission.

Override the timing/rate assumptions for an observed cheaper-GPU scenario with `--seconds-per-request LOW HIGH --compute-usd-per-hour RATE`; storage and retention are also explicit CLI inputs. Overrides are caller assumptions, not verified prices.

A saved full 25-frame float32 array at 576×1024×3 uses about **176.95 MB** including its array header. Those 6,000 arrays require about **1.062 TB per copy**, or **2.123 TB for two copies**, before references and metadata. At the historical $0.20/GB-month NVMe rate, one such raw copy would cost approximately **$212 for 30 days**; destination capacity rounding and backup-location rates are extra. Do not assume compression savings or that the retained 300 GB volume can hold the corpus. A compact feature/metric table can accompany raw outputs, but any raw retention/disposal policy needs a prospective evidence contract.

This scale is not funded or admitted by the canceled run's budget. Phase 1 results should first inform the next phase's configuration count, storage location, retention period and complete budget.
