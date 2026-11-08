# Resumable full study

The immutable study plan separates development selection from held-out evaluation.
Run these commands inside the verified native CUDA11.8/sm90 container at `/workspace`,
using `outputs/venvs/vista-upstream-cu118-sm90/bin/python`, the persistent model caches,
and the exact CUDA environment recorded by the active environment lock. Keep one
serialized GPU worker. The orchestrator launches generation and LPIPS in separate,
sequential processes; LPIPS imports only from its verified isolated package target.
No evaluation work should overlap headline timing.

Finish all Python source changes, including the variability-calibration module,
before numerical proofs. The numerical gate fingerprints the entire package.
The study also snapshots its generation sources and refuses changed generation
code on resume. Historical upstream runs may be adopted with their independently
recorded source receipt; this does not assert that old and new implementations
have identical source.

An optional pre-plan compiler bootstrap can run the same32 development requests
with `drivebench.benchmark` and retain all32 outputs. Compare compiled-default and
compiled-plus-conditioning-cache against the existing upstream development run,
including LPIPS and the declared blinded review, then choose the secondary
reference before materializing the main plan. This uses development data only.
Adopt those completed runs by repeating `--adopt VARIANT=RUN` and
`--adopt-source VARIANT=COMPLETED_HOST_RECEIPT` below. Keep shard size32, candidate
count1, save_outputs32, generation settings, compiler selectors and input-condition
settings identical to the planned variants. Each receipt must bind unchanged
source files through successful completion. Adoption retains the external arrays
and avoids duplicate generation; study metrics are still recomputed with their
own receipts. Any final nonzero policy for the study is bound to its complete
development aggregate, even when an earlier bootstrap review informed reference
selection.

```bash
python -m drivebench.study plan --spec configs/full_study.yaml --output outputs/full-study-v1
python -m drivebench.study quality --study outputs/full-study-v1 --phase development \
  --adopt upstream=/workspace/outputs/study-bootstrap/upstream \
  --adopt-source upstream=/workspace/outputs/host/study-upstream-development-attempt-1.json
python -m drivebench.study aggregate --study outputs/full-study-v1 --phase development
```

Development runs all12 declared variants on all32 distinct scenes: upstream,
eager, exact conditioning reuse, steps47/46/45/40/30/20, and output-block9 reuse intervals2/3/4.
The whole output block was selected from measured denoiser timings before planning;
its6.44458s of52.6014s denoising motivated47/46/45 as candidate runtime controls.
Those counts are hypotheses until actual full development measurements confirm the
latency gaps; the middle block was only0.53380s.
The optimized50-step run supplies the50-step curve point. A separate versioned
companion specification can set `quality.candidate_count: 2` for actual batch2
quality across32development and128held-out clips; the single-candidate main study
cannot establish quality of a joint2-candidate batch. The companion requires its
own complete paired outputs, development selection and held-out gate. Compare feature reuse
against the nearest **measured genuinely reduced-step** latency; report the actual
gap. The50-step reference is excluded as a reduced-step control. Predeclare an
additional count such as49 or42 when block profiling suggests it is needed, while
retaining the required50/40/30/20 curve. A nearest
neighbor is not automatically an equal-runtime match. If the gap is material,
predeclare a new development experiment before any held-out selection.

Runtime equivalence was predeclared at2026-10-27T00:15:29Z before materializing the
study. Rank the
predeclared reduced-step controls by absolute log ratio of development median
latency; exclude50 steps. Report the selected ratio and its paired scene-cluster
95% bootstrap interval from all32 real development scenes (2,000 resamples, seed0,
retaining each scene's paired requests/repeats). Label it **runtime matched** only
if the entire latency-ratio interval is inside the inclusive margin[0.98,1.02].
Otherwise label it **nearest measured control**, including when the point estimate
is within2% but the interval extends outside the margin. The closest control stays
on the curve with its measured gap. This criterion concerns runtime only and
imposes no generation quality threshold. It does not establish exact equality.
Any additional control must be declared on development data before held-out
generation. The standalone report embeds the complete versioned criterion and its
SHA256, computed from canonical sorted compact JSON:
`6e83d60f341a36afc27355ef80f2cc4d13af46cb1e3f0c2c9cdc72859903f8fc`.

Use bounded `drivebench.experiments` discovery for compiler modes, BF16, decoder
chunking, prefetch, requested batches, and candidate counts4/8/16. Keep failures,
unsupported behavior, actual batch sizes and OOM fallback visible. Discovery
results do not become headline results merely because their requested settings
look faster. Freeze the actual finalist workload after development discovery.

Complete the fixed blinded development review subset from the exported contact
sheets and full-future GIFs. Keep private keys away from the reviewer until ratings
are saved. Record agent-assisted review explicitly; never label it human review.
Use recorded ground truth only for recorded actions. Report numerical, LPIPS,
temporal, ground-truth and review findings for approximate variants even when they
fail the exact policy. New nonzero quality limits require a separate explicit
reviewed development policy and must precede held-out generation; no limit may be
widened using held-out outcomes. Exact zero-error equivalence needs no subjective
acceptability threshold, but does not remove the planned visual review work.

Produce strict development numerical reports with agreement, candidate reorder,
unrelated-action isolation, and exercised cache hits for the exact finalists.
The reports must use this final source and runtime; preserve all their arrays.
The following freeze command illustrates finalist names; choose them from the
completed development evidence and substitute the actual numerical report paths.
A separate development variability policy must be frozen before held-out headline
runs; the pilot and legacy performance-calibration policies are distinct. Create a
baseline development config with the chosen workload and its current passed
`numerical_validation` report, then run:

```bash
python -m drivebench.calibrate_pilot plan --config <validated-baseline-development-config.yaml> \
  --output outputs/study-variability-v1 --relative-tolerance-floor 0
python -m drivebench.calibrate_pilot run --pilot outputs/study-variability-v1
python -m drivebench.calibrate_pilot freeze --pilot outputs/study-variability-v1 \
  --output outputs/study-variability-v1/policy.json
```

A zero floor predeclares no additional allowance; the frozen policy is still the
conservative maximum of measured bidirectional block drift and upper95% scene-
bootstrap bounds. This rule is committed before measurement and never tuned using
held-out data. Four fresh50-request blocks cover all32development scenes. After
headline timing, apply it with `calibrate_pilot compare --policy ... --baseline ...
--candidate ... --output ...`; any rejection needs independently collected
`--confirm-baseline` and `--confirm-candidate` runs. The policy remains tied to the
same baseline implementation, workload, GPU UUID, runtime and source. Later pilot
policy verification checks the immutable numerical report and the contemporaneous
benchmark gate; it does not reread the retained numerical output arrays on every
comparison. Keep those arrays and their original hashes for independent audit. A replacement
allocation requires a newly measured pilot and proofs before its headline timings.

```bash
python -m drivebench.study freeze --study outputs/full-study-v1 \
  --exact-runs eager conditioning-reuse \
  --variants eager conditioning-reuse feature-2 steps-40 \
  --numerical-proof eager=/workspace/outputs/study-validation/eager/agreement.json \
  --numerical-proof conditioning-reuse=/workspace/outputs/study-validation/cache/agreement.json \
  --timing-candidates 1 --timing-workload single_future
python -m drivebench.study quality --study outputs/full-study-v1 --phase development_actions
python -m drivebench.study aggregate --study outputs/full-study-v1 --phase development_actions
python -m drivebench.study quality --study outputs/full-study-v1 --phase evaluation
python -m drivebench.study aggregate --study outputs/full-study-v1 --phase evaluation
python -m drivebench.study quality --study outputs/full-study-v1 --phase evaluation_second_seed
python -m drivebench.study aggregate --study outputs/full-study-v1 --phase evaluation_second_seed
python -m drivebench.study quality --study outputs/full-study-v1 --phase evaluation_actions
python -m drivebench.study aggregate --study outputs/full-study-v1 --phase evaluation_actions
python -m drivebench.study timing --study outputs/full-study-v1 --variants eager conditioning-reuse
```

Both16-scene action subsets are selected by request-ID hash in the original plan.
Recorded and perturbed actions deliberately share the recorded candidate seed.
Each comparison verifies distinct trajectory hashes and computes the change in
output caused by changing the action at the same noise seed, then compares that
response between implementations. These image-response diagnostics do not measure
trajectory adherence. Ordinary main workloads retain independent candidate seeds.

Approximate finalists can additionally use separately frozen nonzero development
limits. Preserve the exact-zero policy and its strict proofs. Create the additional
policy with `python -m drivebench.freeze --development-evaluation
outputs/full-study-v1/quality/development/aggregate/evaluation-vs-upstream.json
--thresholds <explicit-reviewed-development-draft.yaml> --output
<new-versioned-policy.yaml>`, then add `--reviewed-policy
upstream=<new-versioned-policy.yaml>` to the selection freeze command above.
The optional argument can repeat for the optimized reference with its own
development evaluation. Selection verifies all32 development scenes, current
metric/run hashes, reference attribution and chronology; it rejects changed
policies or policies frozen after any held-out attempt. Exact variants continue
using the original zero policy. Nonzero limits apply only to their calibrated
variants against their declared reference, never automatically to the other
reference. Both policies and provenance hashes remain in selection and aggregate
evidence. Metric limits alone do not establish accepted blinded review: the
nonzero quality gate remains pending until separately verified review acceptance
is joined. No policy is adjusted using held-out outcomes.

Every quality phase can resume with the same command. `--max-shards N` stops after
the firstN planned shards. Metrics, per-output SHA256s, blinded previews and receipts
are persisted before unselected arrays are deleted. One fixed request per phase
retains its full float outputs for each variant; external adopted arrays are never
deleted. Raw deletion is explicit: new metrics cannot be computed on deleted videos
without regenerating them. Aggregate raw timings remain exploratory and are never
promoted to headline timing.

A replacement allocation can resume **quality only** with `--runtime-config new-runtime.yaml`:

```yaml
environment_verification: /workspace/outputs/environment-study-02/verification.json
environment_lock: /workspace/outputs/environment-study-02/reference-lock.json
```

The replacement must pass the full environment gate and match all frozen runtime
fields except GPU UUID and PCI bus address. All original and replacement identities
remain in the aggregate provenance. Timing never uses this relaxation: every
headline block and its numerical proof must use the same GPU UUID/runtime.

Timing uses four independently warmed balanced50-request blocks per finalist,
200 total requests each. It repeats the same first50 held-out requests across
blocks; unique clip and scene counts are reported, so this does not imply128 unique
timing clips. Each block preserves cold-load/warmup time separately, all request
latencies, actual work, memory and failures. Scene-cluster paired bootstrap keeps
repeated blocks within their scene and reports both p50/p95 uncertainty and block
median drift. The200-request study aggregate is not four200-request runs and does
not automatically calibrate the separate legacy regression gate. A regression
rejection still needs an independent confirmatory run.

At57.5 seconds per50-step future, with upstream/eager/reuse/feature2/steps40 as the
five held-out variants, a conservative serial estimate is:

| Stage | Requested futures | Baseline-equivalent generation hours |
|---|---:|---:|
|32development×12variants|384|5.40|
|128held-out+16second seed×5variants|720|11.04|
|16development+16held-out×2same-seed actions×5variants|320|4.91|
|200timed requests×2finalists, one future/request|400|6.39|
|Alternative: same timing with four futures/request|1600|25.56|

Totals are27.74h (~$129) for one-future timing or46.90h (~$219) for four-future
headline timing at$4.6608/h, before loading, warmups, LPIPS, discovery, variability
calibration, transfers and failures. Reduced-step cost is scaled linearly only for
planning; no feature-reuse/batching speedup is assumed. Four50-request baseline
variability blocks add3.19h for one future or12.78h for four. A single16h allocation
cannot complete all of this serial work. Respect its existing hard stop and resume
on a fresh authorized allocation; do not mix GPU identities to manufacture one
headline timing run. Use `drivebench.study estimate` for other declared finalists.

The current study selection supports exactly one timing workload and candidate
count, with one immutable schedule. It cannot supply both single-future and
four-candidate headline timing by rerunning the same directory with new settings.
Predeclare a separate timing study or a separately audited timing-only launcher
bound to the same frozen finalist selection. Each workload needs its own numerical
proof, development variability pilot, interleaved four50-request blocks and output
namespace. Both workloads retain their actual GPU/runtime identity; never splice
blocks from replacement allocations. Candidate4 headline timing plus single-future
headline timing for two finalists adds6.39h of single-future measurement and3.19h
of its baseline pilot to the four-candidate estimate. Together with both pilots,
the conservative total is about69.3h (~$323), before discovery, warmups, evaluation
and failures. Candidate8/16 screens can remain explicitly exploratory with bounded
request counts, actual memory/OOM/batch records and no200-request claim. A request
latency for a candidate set describes the whole set, not latency per future.

The standalone timing-only companion binds the existing immutable selection and
quality evidence, and does not generate another quality dataset:

For nonzero numerical tolerances, first follow the mandatory
[exact semantic sidecar and guarded pilot workflow](exact-numerical-semantics.md).
The original validation tolerance must never relax isolation or cache equality.

```bash
python scripts/run_timing_companion.py prepare-pilot --study outputs/full-study-v1 \
  --output outputs/single-future-pilot-baseline.yaml --baseline <baseline> \
  --candidate-count 1 --numerical-proof <single-future-numerical.json>
python scripts/run_timing_companion.py pilot-plan --config outputs/single-future-pilot-baseline.yaml \
  --output outputs/single-future-pilot-v1 --relative-tolerance-floor 0
python scripts/run_timing_companion.py pilot-run --pilot outputs/single-future-pilot-v1
python scripts/run_timing_companion.py pilot-freeze --pilot outputs/single-future-pilot-v1 \
  --output outputs/single-future-pilot-v1/policy.json
python scripts/run_timing_companion.py plan --study outputs/full-study-v1 \
  --output outputs/full-study-single-future-timing-v1 --baseline <baseline> \
  --candidate <finalist> --candidate-count 1 --pilot-policy outputs/single-future-pilot-v1/policy.json \
  --proof <baseline>=<single-future-numerical.json> \
  --proof <finalist>=<single-future-numerical.json>
python scripts/run_timing_companion.py run \
  --companion outputs/full-study-single-future-timing-v1
```

It verifies the active package, helper, proofs and workload-specific pilot, then
uses the study execution lock for its separate balanced eight-block schedule.
Both finalists receive200 measured requests; the pilot still precedes those runs.

## Local report and publication projection

After copying the metadata, receipts and review artifacts into a local mirror,
generate the report without reading retained quality videos. Timing companions
with nonzero numerical tolerances also require their retained proof NPZ arrays:
the authoritative semantic check recomputes exact equality while reporting.

```bash
python scripts/report_full_study.py --study outputs/full-study-v1 \
  --output outputs/full-study-report-v1 --workspace-root "$PWD" \
  --coverage reports/scene-coverage.json \
  --review-ratings outputs/review-ratings-v1.json
```

Add the two independent timing companions when they exist:

```bash
python scripts/report_full_study.py --study outputs/full-study-v1 \
  --output outputs/full-study-report-v2 --workspace-root /workspace \
  --coverage reports/scene-coverage.json --review-ratings outputs/review-ratings-v1.json \
  --timing-companion candidate4=outputs/full-study-candidate4-timing-v1 \
  --timing-companion single=outputs/full-study-single-future-timing-v1
```

Run this reporting command CPU-only in the canonical `/workspace` environment.
The current timing-companion helper validates its complete parent selection,
source/helper identities, quality-policy evidence, optional replacement-runtime
bindings, and pilot. The report reconstructs both 200-request aggregates from
their four independent blocks and recomputes paired pilot comparisons and
absolute scene-bootstrap intervals. Missing canonical paths, changed artifacts,
mixed GPUs or incomplete blocks fail closed. An absent main study timing directory
is valid when these companions supply timing. Copy the completed report and its
sanitized `public/` projection to the local archive afterward.

Companion timing remains separate from quality acceptance. The report joins
verified 128-clip held-out and 16-clip second-seed metrics with their frozen,
reference-specific limits and bound accepted votes for the actual blinded
candidate panel. Every underlying quality shard must match the planned source
files, exact held-out partition and timing implementation; runtime compatibility
permits only the documented GPU UUID/PCI replacement. Source receipts for adopted
shards are rechecked. Completed CSVs containing rejected votes do not pass. Combined
claims additionally require a passed variability comparison and negative upper
95% bounds for both p50 and p95 relative latency changes. Any accepted claim
states the recorded-trajectory quality scope; it does not imply all candidates
were quality-evaluated on all 128 clips or establish trajectory adherence.
Both workloads' raw measurements, cold starts, uncertainty and provenance hashes
are exported; canonical paths, raw configs, UUIDs and reviewer notes remain private.

The review-ratings argument is optional while review remains pending. Its JSON
binds separately saved ratings to the immutable exported review; never edit the
original blank CSV. Use actual SHA256 values for both files:

```json
{
  "reviewer_kind": "agent_assisted",
  "reviews": [{
    "index": "/workspace/outputs/full-study-v1/quality/development/shards/000/metrics/blinded-review/index.html",
    "index_sha256": "<actual SHA256>",
    "csv": "/workspace/outputs/reviews/development-000-rated.csv",
    "csv_sha256": "<actual SHA256>"
  }]
}
```

Include every exported review index. Rated CSVs must contain the same unique
`case_id` set, with `acceptable_A` and `acceptable_B` each `yes` or `no`; retain
diagnostic notes locally. Use `human` only for actual human review. A completed
review subset alone does not establish quality preservation or driving adherence.

The local `index.html` links every saved paired evaluation, timing block and
blinded review. `results.json` includes both-reference quality summaries, raw paired
metrics and output hashes, same-seed action-response diagnostics, second-seed
results, condition sample counts and uncertainty, actual reduced-step runtime gaps,
cold/warm timing boundaries and raw measured requests. It rechecks artifact hashes,
policies, review files and timing-block reconstruction; missing or stale evidence
remains explicit. Description mentions and nearby objects are metadata categories,
not camera-visible or validated weather labels.

Publish only the generated `public/` directory and its projection manifest. It is
an allowlisted numerical/aggregate-plot projection without accounts, private paths,
GPU UUIDs, configurations, commands, weights, driving images/videos, private review
keys or freeform reviewer notes. The report conservatively withholds combined
headline claims until the separate frozen quality and variability-policy evidence
is joined; eligible timing alone does not establish a quality-preserving speedup.
Use a fresh output directory by default. `--replace` only replaces files owned by
the previous matching projection manifest and removes its old plot files; unknown
or changed files cause a refusal. `--no-plots` retains all numeric evidence when a
plotting library is unavailable. Synthetic fixtures always remain visibly labeled
and cannot establish GPU performance or quality.
