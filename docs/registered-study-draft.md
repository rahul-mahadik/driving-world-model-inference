# Registered compiler study proposal

`configs/full_study_registered_draft.yaml` is an unmaterialized 18-variant draft.
The existing specification and frozen `d8a328…` package are unchanged. Its
`draft_status` field is descriptive: the plan command does not enforce that field.
Resolve the provisional reference from the complete bootstrap development review,
save a separately named approved specification, and only then create a plan.

The primary reference stays upstream. The proposed secondary reference is
`compiled-conditioning`, contingent on all 32 development clips and the fixed
blinded review supporting explicitly declared limits. If it fails, select
`conditioning-reuse` before planning and retain compiler failures on the curve.
Neither choice changes the historical strict-zero failure. Approximate acceptance
and passed execution/numerical gates are separate requirements.

| Group | Variants | Purpose |
|---|---|---|
| Original and exact engineering controls | upstream, eager, conditioning-reuse | Original behavior and independent conditioning reuse |
| Compiler ablation | compiled-default, compiled-conditioning | Registered compiler alone and with 2 GiB conditioning reuse |
| Boundary control | compiled-boundary-1 | Output block 9 boundary with interval 1, no feature reuse |
| Compiled reduced-step controls | steps-48, steps-47, steps-46, steps-45, steps-40, steps-30, steps-20 | Required step curve plus genuinely reduced controls near predicted feature runtime |
| Combined feature reuse | feature-2, feature-3, feature-4 | Compiled + conditioning reuse + output block 9 intervals 2/3/4 |
| Eager ablations | eager-feature-2, eager-steps-47 | Feature reuse and reduced steps without compilation, both with conditioning reuse |

All primary generation settings stay at 25 frames, 576×1024, FP16 and 50 steps,
except the explicitly named step variants. Candidate execution remains serial
(`max_batch_size: 1`). Main quality uses one recorded trajectory; the fixed action
subsets separately use two distinct trajectories at the same seed. This does not
claim actual joint batching, trajectory adherence, or counterfactual ground truth.

The compiled bootstrap variants match generation, model selectors, optimization
settings, manifest order, one candidate, all 32 retained outputs and 3–5 warmups.
Use the adoption commands in [compiled-bootstrap.md](compiled-bootstrap.md) after
the approved specification is planned; do not replay those three completed runs.
Adopted raw arrays remain external and are never automatically retired.

## Quality and selection sequence

Only execute these commands after the provisional reference decision; no command
below was executed to prepare this document:

```bash
python -m drivebench.study plan \
  --spec configs/full_study_registered_approved_v1.yaml --output outputs/full-study-v1
# Run development with the three --adopt/--adopt-source pairs in compiled-bootstrap.md.
python -m drivebench.study aggregate --study outputs/full-study-v1 --phase development
```

All 18 variants receive all 32 development clips. Complete the fixed blinded
reviews and declare any nonzero limits using this development evidence. Freeze
separate reference-attributed policies with `drivebench.freeze`; policies against
upstream cannot automatically accept a comparison against the optimized reference.
The original exact-zero policy remains bound to a verified exact run. Use the
reference-specific policy and current-source numerical-proof commands documented
in [full-study.md](full-study.md). A reviewed policy does not convert a failed
numerical report into a passed report.

The proposed accepted-compiler selection needs five roles:
`upstream`, `conditioning-reuse`, `compiled-conditioning`, one `feature-*` winner,
and its nearest measured `steps-*` control. The exact conditioning run supplies
the exact-zero anchor; the two references and reduced-step control remain visible.
Reducing this selection to four by silently dropping a required role is not a
valid budget saving. Four variants can suffice for a different, explicitly
justified fallback selection.

The runtime-match criterion is already predeclared: the paired scene-bootstrap
95% latency-ratio interval must lie entirely inside `[0.98, 1.02]`. Otherwise call
the control **nearest measured control** and report its gap. The criterion hash is
`6e83d60f341a36afc27355ef80f2cc4d13af46cb1e3f0c2c9cdc72859903f8fc`.
Step 50 is never a reduced-step control. This rule is separate from quality limits.

The revised draft adds **only steps-48** after the completed one-clip feature
screens. Their measured warm times were 42.271 / 41.218 / 40.728 seconds for
intervals 2 / 3 / 4, versus 43.595 seconds for compiled 50 steps. A linear planning
estimate using its 38.765-second denoising stage predicts 42.045 / 41.270 / 40.494
seconds for 48 / 47 / 46 steps, with estimated gaps of +0.54% / −0.12% / +0.58%.
These are control-selection estimates, not measured reduced-step runtimes or
established matches. The probes used no conditioning cache; the study applies the
same 2 GiB cache to paired compiled controls and feature variants. Full 32-scene
measurements still determine equivalence. Adding 49 would not cover a new measured
feature target. Existing 45/40/30/20 controls stay predeclared.

The [proposal receipt](../reports/study-proposals/reduced-step-controls-v2/receipt.json)
binds the transferred probe metadata, revised specification and budget, and
preserves full copies and hashes of the prior 17-variant draft. No study plan or
held-out generation was materialized by this revision. It adds about 0.48 hours
and $2.24 to the conservative budget, including warmup/loading allowances.

An example selection command, with placeholders replaced only by actual accepted
development evidence, is:

```bash
python -m drivebench.study freeze --study outputs/full-study-v1 \
  --exact-runs conditioning-reuse \
  --variants conditioning-reuse compiled-conditioning feature-2 steps-48 \
  --reviewed-policy upstream=/workspace/outputs/policies/reviewed-upstream-v1.yaml \
  --reviewed-policy compiled-conditioning=/workspace/outputs/policies/reviewed-compiled-v1.yaml \
  --numerical-proof upstream=/workspace/outputs/study-validation/upstream-current.json \
  --numerical-proof compiled-conditioning=/workspace/outputs/study-validation/compiled-conditioning-current.json \
  --timing-candidates 4 --timing-workload candidate_set
```

The feature and step names above are examples, not a preselected winner. Upstream
and the optimized reference are added to selection automatically. Complete and
review the 16-scene development action phase before proceeding to held-out data:

```bash
python -m drivebench.study quality --study outputs/full-study-v1 --phase development_actions
python -m drivebench.study aggregate --study outputs/full-study-v1 --phase development_actions
python -m drivebench.study quality --study outputs/full-study-v1 --phase evaluation
python -m drivebench.study aggregate --study outputs/full-study-v1 --phase evaluation
python -m drivebench.study quality --study outputs/full-study-v1 --phase evaluation_second_seed
python -m drivebench.study aggregate --study outputs/full-study-v1 --phase evaluation_second_seed
python -m drivebench.study quality --study outputs/full-study-v1 --phase evaluation_actions
python -m drivebench.study aggregate --study outputs/full-study-v1 --phase evaluation_actions
```

This supplies 128 held-out recorded clips, 16 second-seed clips, and 16 held-out
same-seed action pairs per selected variant. No held-out result may select another
threshold, feature interval, step count or reference. The report joins metadata
condition categories and every blinded review; review completion alone is not
acceptance. Failed acceptance remains a measured curve point.

## Timing and allocation boundaries

The proposed primary headline pair is upstream versus compiled-conditioning at
the same 50-step generation settings, if its new numerical and quality gates pass.
The draft declares four candidates for its main schedule, but that schedule will
remain unused if timing moves to a replacement GPU. The main timing command has
no runtime overlay and cannot replace the immutable study-01 environment paths.
Use separate candidate-4 and single-future companions on the new GPU, each bound
to the original selection and its own fresh proofs and pilot. Each finalist
receives four interleaved 50-request blocks per workload. These repeat the first
50 held-out requests, not all 128 distinct clips; unique clip and scene counts
must accompany the 200-request totals.

The standalone companion now accepts `--runtime-config` on `prepare-pilot` and
`plan`, bound to a YAML file containing only `environment_verification` and
`environment_lock` paths. Its CPU tests cover compatibility and provenance; actual
new-allocation execution remains pending. Do not rewrite the original study plan.
Use `prepare-pilot` with this fresh proof/lock overlay and current numerical proof,
then `calibrate_pilot plan/run/freeze` with a predeclared floor of 0, followed by
companion `plan/run` with the same overlay. Execute this sequence separately with
candidate counts 4 and 1 and distinct output namespaces. The companion interface
and canonical-path reporting command are documented in [full-study.md](full-study.md).
Each workload binds its own policy and passed current-source numerical evidence.
The numerical validator needs at least two candidates to prove isolation even
for a single-future benchmark.

The frozen package has known limits: its all-mode validator rejects feature
intervals greater than 1, and its cache branch is incompatible with the explicit
feature boundary. Its pilot comparison also requires identical generation,
including step count. Therefore neither a feature finalist nor a 40-vs-50-step
comparison can be promoted through those old gates. Their 32-development and
128-held-out quality/runtime curves remain descriptive, with the original strict
gate false. No additional 200-request feature or reduced-step timing is proposed.
Any later promotion would need separately audited, source-bound qualification;
reviewed quality limits alone are insufficient.

The candidate-4 pilot and full mixed upstream/compiler comparison need roughly
39 hours on the same GPU UUID including the warmup/loading allowance below.
The existing 16-hour allocation cannot supply that evidence. Preserve its hard
stop; start the candidate-4 pilot only in a newly authorized allocation with
enough uninterrupted budget and time. Quality phases may resume on a separately
verified equivalent replacement GPU. Headline timing blocks cannot be spliced
across different GPUs. Keep all evaluation and profiling outside timed blocks.

## Remaining bounded coverage

The 18-variant quality study does **not** schedule all input/throughput diagnostics.
The existing `scripts/prepare_execution_screen.py` already defines the following
small development jobs and strict output comparisons. Local inspection found
their configurations, not completed run evidence. Materialize a fresh screen
namespace when ready; this helper creates files and never launches GPU work:

```bash
python scripts/prepare_execution_screen.py --base-config configs/study_eager_development.yaml \
  --manifest data/manifests/development.jsonl \
  --output outputs/execution-screen-remaining-v1 --runtime-project /workspace
```

| Required coverage | Existing bounded screen jobs | Claim boundary |
|---|---|---|
| Input processing and warm filesystem | `input-uncached-*`, `input-warm-filesystem-*` | Checksums warm OS pages in both; uncached means no decoded/context reuse, not cold storage |
| Fresh versus retained conditioning | `input-conditioning-fresh-*`, `input-repeated-uncached-*`, `input-repeated-conditioning-*` | Retained conditioning is measured separately from preparing a new scene |
| Independent loading and transfer flags | `input-prefetch-*`, `input-pinned-*`, `input-pinned-async-*` | Pinned-async includes the two-slot transfer pool; async and double buffering are not independent switches |
| Combined input pipeline | `input-pipeline-*` | Compare against baseline and each isolated flag; no assumed additive speedups |
| Continuous independent scenes | Four-scene `*-throughput` runs with `save_outputs: 0` | Four measured requests give raw throughput, not a headline p95 or broad distribution |
| Complete candidate sets 4/8/16 | `candidates-{4,8,16}-eager`, `candidates-{4,8,16}-conditioning-b1` | Two complete measured requests plus one warmup; serial batch size 1, no headline percentile claim |
| Memory boundary | Existing `probe_candidate_memory.py` full-length direct batch attempts | Report successful actual batches, failures and peak memory separately from candidate-set latency |

For each of the nine input variants, run `*-correctness` (2 saved requests plus
one warmup), its saved-output strict check from the plan, then `*-throughput`
(4 requests plus one warmup, no export). These execute the real benchmark loading
path, which a direct model validation alone does not exercise. Example:

```bash
python -m drivebench.benchmark --config outputs/execution-screen-remaining-v1/configs/input-uncached-correctness.yaml
python -m drivebench.benchmark --config outputs/execution-screen-remaining-v1/configs/input-pinned-correctness.yaml
python scripts/check_screening_outputs.py \
  --reference outputs/execution-screen-remaining-v1/runs/input-uncached-correctness \
  --candidate outputs/execution-screen-remaining-v1/runs/input-pinned-correctness \
  --output outputs/execution-screen-remaining-v1/proofs/input-pinned-outputs-strict.json
python -m drivebench.benchmark --config outputs/execution-screen-remaining-v1/configs/input-pinned-throughput.yaml
python -m drivebench.benchmark --config outputs/execution-screen-remaining-v1/configs/candidates-16-conditioning-b1.yaml
```

Respect each plan prerequisite and use separate source-bound host receipts, fresh
outputs and timeouts. Do not execute the batch-2 branches when their strict
isolation evidence fails. Async staging currently synchronizes before inference;
the safe double buffer exists, but only CPU prefetch is expected to overlap with
the next GPU generation. A GPU transfer/compute-overlap claim requires an actual
trace. Model residency is implemented and always enabled; offload is explicitly
unsupported and is not a measured ablation. Decoder chunks and compiler/shape
probes already have separate artifacts; their coverage must be judged from actual
status and settings, not merely from available config names.

The selected input and serial scaling screens add about 5.4 GPU-hours including
their small current-source eager/cache validation gates and loading allowance.
Candidate 8/16 completion remains required even if candidate 4 is the only
multiple-future headline workload. Any optimized candidate-8/16 speed claim needs
the corresponding actual optimized runs; the eager/cache screens establish only
the behavior of those two implementations.

## Budget and disk audit

[registered-study-budget-draft.json](../reports/registered-study-budget-draft.json)
contains the arithmetic and specification hash. These are planning assumptions,
not new measurements: 57.3 seconds per eager future and 43.6 seconds per compiled
future, serial candidate scaling, $4.6608/hour, and **no further savings** assumed
for reduced steps, feature reuse or conditioning reuse.

| Stage | Generation hours |
|---|---:|
| All 18 variants × 32 development clips | 7.58 |
| Five selected variants × 128 held-out clips | 8.73 |
| Five selected variants × 16 second-seed clips | 1.09 |
| Five selected variants × two 16-scene, two-action phases | 4.36 |
| Two finalists × 200 single-future requests | 5.61 |
| Two finalists × 200 four-candidate requests | 22.42 |
| Separate four-block upstream pilots for both workloads | 15.92 |
| **Generation total** | **65.71** |

Allow up to 8.65 hours for five warmups per process, 2.57 hours for 120 seconds
of additional cold/loading overhead across 77 processes, and 5.40 hours for the
bounded coverage screens. This subtotal is 82.32 hours. An explicit 8–15-hour
reserve for LPIPS, review, additional proofs and failures gives **90.32–97.32 hours,
approximately $421–454**. Actual prior allocations, already executed discovery,
storage and idle charges must be added to the cumulative ledger. The completed
upstream and two bootstrap arrays avoid 1.28 hours of replay; their original cost
still belongs in the experiment total. This is a tight envelope under $500,
not a guarantee, and it exceeds the account's initial funding.

A full candidate-4 confirmatory timing rerun can add roughly 25 hours (~$116).
Reserve budget before interpreting a rejection: if confirmation cannot fit the
remaining authorization, report it unresolved instead of claiming a regression
or quality-preserving improvement. Keep candidate 8/16, memory and I/O screens
exploratory; extending each to 200 requests is outside this budget. Removing one
compiled held-out variant saves only about 2.52 generation hours, and cannot
justify dropping a required reference/control.

Each float future is about 0.165 GiB. A 32-future raw shard is 5.27 GiB; the largest
active reference/reference/candidate trio is 15.82 GiB for recorded or 16×2 action
phases. The three external bootstrap runs retain another 15.82 GiB; fixed retained
raw outputs across phases add roughly 8 GiB. The draft reserves 12 GiB before each
generation, and compiler caches/review assets need additional space. The bounded
I/O/scaling screen can retain another 12.2 GiB. On the 200 GiB volume, transfer
completed diagnostic directories to the local archive with verified hashes before
retiring their remote arrays; preserve metadata and strict output proofs. Never
delete adopted bootstrap arrays needed for later pairing. Do not rely on a raw
file hash to reconstruct a retired video or claim that it can support new metrics.
