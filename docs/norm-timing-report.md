# Separate norm-qualified timing report and blind review sealing

This procedure documents tooling. It does not assert a completed numerical
qualification, GPU measurement, visual rating, quality acceptance or publication.
`scripts/report_norm_timing.py` is a new supplementary reporter; the old reporter,
old companion and frozen d8 package remain unchanged.

## Review preparation and retrieval

Use one independently rated native bundle for every selected candidate-versus-
upstream shard in development, evaluation 128, second-seed 16, development-actions 16
and evaluation-actions 16. With one 32-clip development shard and four 32-clip
evaluation shards, this is eight comparisons, four cases per comparison, each
with all 24 future frames showing both panels. These cases are a fixed subset;
they do not represent exhaustive human review of every clip.

The coordinator retrieves each `native-review/OPAQUE/public/` subtree and its
SHA256 inventory. Verify the full independent copy, including all 96 native paired
PNG frames, pages, manifest, index and blank CSV. Keep `private-receipt.json` and
original A/B keys in a separate private coordinator directory. The reviewer gets
only the public subtree, an opaque token, the rubric, and the original anonymous
index's SHA256. Do not send original source paths or variant names to the reviewer.

Use a protocol declared before native export, with explicit UTC `created_at`.
The already-predeclared bootstrap rubric may be reused if its scope is explicitly
adopted for main review; its old votes and seals cannot be reused. Every new
comparison requires its own inspection, CSV, seal and external pin. Retain rejected
or uncertain votes. This tool records a reviewer attestation and read chronology;
it cannot prove a reviewer never learned identities elsewhere.

The reviewer copies the blank CSV to a separate path outside `public/`, keeps its
case IDs/header, and records A/B/tie/uncertain preference, yes/no/uncertain panel
acceptance and concrete observations. Inspect all 24 frames per case; use the native
pixels for ambiguous objects/geometry. Agent reviews must say `agent_assisted`.

After all observations are saved, and before opening any answer key:

```bash
outputs/venvs/vista-upstream-cu118-sm90/bin/python scripts/report_norm_timing.py seal \
  --public "${OPAQUE_PUBLIC_DIRECTORY:?}" \
  --ratings "${SEPARATE_RATED_CSV:?}" \
  --protocol "${PREDECLARED_PROTOCOL:?}" \
  --original-index-sha256 "${COORDINATOR_SUPPLIED_INDEX_SHA256:?}" \
  --started-at "${ACTUAL_REVIEW_START_UTC:?}" \
  --reviewer-kind agent_assisted --output "${FRESH_SEAL_PATH:?}"
```

The seal binds every public file, the rated CSV, protocol, opaque case identities,
24-frame coverage and actual uncertainty count. Output is exclusive and private.
Its `private_keys_read_before_seal: false` and `all_panels_reviewed: true` fields
are explicit attestations: run the command only when those statements are true.
No private index path or mapping is read by `seal`.

The reviewer sends the seal SHA256 to the coordinator through the task message
channel. The coordinator records that externally supplied hash in a separate
immutable pin, still before any attribution:

```bash
outputs/venvs/vista-upstream-cu118-sm90/bin/python scripts/report_norm_timing.py pin \
  --seal "${SEALED_REVIEW:?}" --sha256 "${EXTERNALLY_RECEIVED_SEAL_SHA256:?}" \
  --output "${FRESH_EXTERNAL_PIN:?}"
```

Preserve the received message/hash with the private evidence inventory. A hash
newly calculated after opening the keys is not a substitute for this ordering.
The pin declares `private_keys_read_before_pin: false`.

## Coordinator registry

After every required review is sealed and pinned, create a NEW private registry.
Each entry binds one original per-shard metric, its native export receipt, the
independent seal and its external pin. Paths are canonical `/workspace` paths;
all hashes below are actual full SHA256 values, not placeholders. The registry
must exactly cover the selected candidate-versus-baseline comparison in all five
phases, without duplicate or bootstrap entries:

```json
{
  "schema_version": 1,
  "receipt_type": "norm_timing_sealed_review_registry_v1",
  "reviews": [
    {
      "metric": "/workspace/outputs/full-study-v1/quality/PHASE/shards/SHARD/metrics/upstream--CANDIDATE.json",
      "metric_sha256": "ACTUAL_SHA256",
      "native_receipt": "/workspace/outputs/full-study-v1/quality/PHASE/native-review/OPAQUE/private-receipt.json",
      "native_receipt_sha256": "ACTUAL_SHA256",
      "seal": "/workspace/outputs/reviews/main-v1/OPAQUE.seal.json",
      "seal_sha256": "EXTERNALLY_PINNED_ACTUAL_SHA256",
      "pin": "/workspace/outputs/reviews/main-v1/OPAQUE.pin.json",
      "pin_sha256": "ACTUAL_SHA256"
    }
  ]
}
```

Use the actual metric/native paths, not guessed path strings. Neither registry
creation nor rating completion establishes acceptance. A separate coordinator
may prepare file mappings without displaying keys to the reviewers. The reporter
validates the complete registry, each export/CSV/protocol, native frame hashes and
export→review→seal→pin timestamps before it writes a durable
`attribution-start-NNN.json` receipt and calls any private-key attribution helper.

## Report execution

Run only after a new norm companion is complete and `verify_completed()` succeeds.
For single and candidate 4, use the same selected implementation and quality
registry; different finalists need separate registries/reports and their own
frozen quality coverage. Run authoritatively on the canonical workspace, CPU-only,
with a fresh output directory:

```bash
CUDA_VISIBLE_DEVICES= outputs/venvs/vista-upstream-cu118-sm90/bin/python scripts/report_norm_timing.py report \
  --study outputs/full-study-v1 --workspace-root /workspace \
  --companion single=/workspace/outputs/norm-timing/single \
  --companion candidate4=/workspace/outputs/norm-timing/candidate4 \
  --review-registry outputs/reviews/main-v1/norm-registry.json \
  --output outputs/norm-timing-report-v1
```

The companion verifier recomputes its proof/qualification/pilot/source/runtime,
eight independent original 50-request blocks and paired scene-cluster uncertainty.
Budget CPU verification wall time; this is not a small metadata-only report.
The reporter does not run inference. It projects original per-block numerical
pass, headline eligibility and new norm eligibility separately. It includes
200 raw observations per finalist, unique clip/scene counts, p50/p95 confidence
intervals, cold/load/warmup data, peak memory, stage timings, and explicit RGB
error-norm limits/diagnostics. It never rewrites a legacy strict false flag.

The combined claim requires actual accepted review votes, complete main 32
development, 128 held-out, 16 second-seed and both 16-scene fixed-seed action phases,
matching implementations/runtime/source/LPIPS weights and frozen reference-specific
metric limits. Action response errors remain diagnostics without an invented
response-error threshold; there is no trajectory-adherence or safety claim.
Both p50 and p95 improvement must exceed the measured pilot tolerance and the
paired confidence interval. A no-regression pass alone is not speedup.

An apparent regression stays `rerun_required`. The companion's existing
`confirm_completed(primary, confirmation)` API can verify a separately authorized
same-identity confirmation; preserve that result separately. This report does not
replace an unfavorable primary with a more favorable rerun or turn
“regression not confirmed” into an improvement claim. If confirmation is infeasible
under the budget, report the unresolved status.

The public subtree contains only allowlisted numerical results, HTML tables and
hashes, without paths, GPU UUIDs, weights, frames, keys or reviewer notes. The
private audit retains exact evidence paths/error explanations. Copy and verify
both independently; publish only the public projection after the normal privacy
scan. Keep the ordinary full-study curves/profiling/diagnostic report linked next
to this supplement. Local preparation is not a GitHub deployment.

## Execution and evidence constraints

The new timing pilot and each comparison block require three to five independent
warmups, complete 50-step counters, serial candidate batches, and positive compiler
execution evidence. A quality-only bootstrap run cannot replace these timing blocks.

The consumer binds current strict baseline proof, full-runtime norm qualification,
frozen upstream variability pilot, immutable selection and runtime overlay. It
reconstructs raw observations without changing flags, preserves repeated clip
identities within scene-cluster bootstrap groups, and demands independent process
paths/timing vectors. Completed blocks resume; a started incomplete block is
preserved and refuses automatic retry. Deadline admission is a full-block estimate
plus 240 seconds, backed by an external watchdog, not an internal kill guarantee.
Each `_load` can perform expensive CPU revalidation; measure that cost before
choosing the complete block budget. See [the companion protocol](norm-qualified-timing.md)
for the frozen-contract and verification-cost requirements.
