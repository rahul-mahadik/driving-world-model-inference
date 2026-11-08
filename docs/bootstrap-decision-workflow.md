# Bootstrap review, reference decision and remaining diagnostics

This procedure prepares decisions; it records no new visual ratings, LPIPS
measurements or acceptance. Keep the frozen package `d8a328…`, historical strict
zero failures, completed run directories and original blank review CSVs intact.
Do not open `private_key.json` until the separate ratings have been saved and
hashed. Reviews performed by an agent must say `agent_assisted`, not human review.
The predeclared rubric is `outputs/bootstrap-review-protocol-v1.json`, created
before review outputs were available; preserve and hash that original protocol.

## Exact qualification artifacts

The two evaluations belong under
`outputs/study-bootstrap/qualification-v1/`:

| Comparison | Metrics JSON | Blinded review directory |
|---|---|---|
| Compiler alone against upstream | `compiled-default-vs-upstream.json` | `compiled-default-vs-upstream-review/` |
| Compiler plus conditioning reuse against upstream | `compiled-conditioning-vs-upstream.json` | `compiled-conditioning-vs-upstream-review/` |

Each review directory contains `index.html`, `review.csv`, four `case-*.png`
contact sheets and four `case-*.gif` previews. The GIF includes every newly
generated frame at 10 fps; the contact sheet samples frames. Preserve the entire
directory. The answer key is separate and must remain unopened during ratings.
The fixed subset uses seed 2026 and four scene/candidate/seed units chosen without
metric-based selection. It is the same subset across both comparisons, although
case IDs and A/B arrangements differ. Do not transfer a rating by row position.

Keep these supporting artifacts with each evaluation:

- `outputs/host/study-bootstrap-evaluate-<variant>-attempt-1.{json,log}` and its
  `.spec.json`, proving successful bounded execution with unchanged inputs/source.
- `outputs/study-bootstrap/qualification-v1/<variant>-evaluation-launch.json`,
  the isolated evaluator's launch record.
- `outputs/evaluation-packages/setup-receipt.json`, including its successful
  package/weight checks and actual LPIPS model-state hash.
- The original generation host receipts named in
  [compiled-bootstrap.md](compiled-bootstrap.md), plus all source run metadata
  and float arrays. Upstream's historical source differs from the current adapter;
  its completed receipt and the direct paired comparisons preserve that attribution.

The evaluator must report `perceptual.status: available` and model-state SHA
`cb2726fb57b9d7f1f7a7eb26f83bf40ef15f421885676460a0b2caa020b74d85`.
Check 32 complete matched pairs, all 32 planned request/scene identities, no missing
keys/errors/synthetic flag, and exact reference/candidate core artifact hashes.
The development manifest hash must be
`a3c6328d9f0d0b94149820f626db17e024e6fbc0e5523fe5ee19aa3316befbff`.
Every numerical, perceptual and temporal metric used for the decision must have
32 contributing pairs. Review the paired distribution, scene-cluster uncertainty,
worst cases, and recorded-ground-truth diagnostics; a single aggregate score is
insufficient. Development `gate.quality_preserving: false` is expected because
the gate requires held-out data and previously frozen limits. It is not a command
to widen the historical numerical tolerance.

The audited wrapper already supplies the isolated LPIPS environment, validates
the run/source receipts, and refuses an existing evaluation or review output.
These are its exact container commands; use the active serialized host queue and
its 1,800-second bound, and do not start duplicate jobs if they are already queued:

```bash
outputs/venvs/vista-upstream-cu118-sm90/bin/python scripts/run_bootstrap_evaluation.py \
  --hostspec configs/study_bootstrap_compiled_hostspec.json --variant compiled-default
outputs/venvs/vista-upstream-cu118-sm90/bin/python scripts/run_bootstrap_evaluation.py \
  --hostspec configs/study_bootstrap_compiled_hostspec.json --variant compiled-conditioning
```

The equivalent explicit `drivebench.evaluate` commands and environment are in
[compiled-bootstrap.md](compiled-bootstrap.md). The wrapper additionally records
and rechecks the concrete artifact hashes. Neither command submits visual votes.

## Save blinded ratings before attribution

Copy each original blank `review.csv` into a separate path, for example
`outputs/reviews/bootstrap-v1/compiled-default.rated.csv` and
`outputs/reviews/bootstrap-v1/compiled-conditioning.rated.csv`. Preserve each
case ID and the complete header:

```csv
case_id,preferred_panel_A_B_tie,acceptable_A,acceptable_B,object_disappearance,object_duplication,freezing,geometry_change,action_response_notes,reviewer_notes
```

Use `A`, `B`, `tie` or `uncertain` for preference and `yes`, `no` or `uncertain`
in both acceptance columns, as the predeclared protocol specifies. Give concrete
panel-specific observations for disappearance, duplication,
freezing, geometry and motion; do not infer trajectory adherence from appearance.
Acceptance concerns preservation of the other panel's generated behavior. Record
common model defects separately from additional defects in one panel; common
defects do not by themselves establish a regression. Preference alone does not
establish acceptance. Watch the complete motion preview before rating a case;
the prepared all-24-frame blinded atlases may help inspect every timestep.
An `uncertain` candidate vote prevents acceptance. Reinspect a lossless blinded
export for ambiguous spatial defects, preserve the initial votes, and save any
resolution in a separately versioned CSV with its rationale. The final reporter
accepts only explicit `yes` votes; an unresolved uncertainty stays pending.

Write an immutable review receipt **before opening the keys** containing the
reviewer kind, UTC completion time, index/CSV/contact-sheet/GIF SHA256s, separate
rated-CSV SHA256s, and a declaration that the answer keys were not consulted.
The final report's ratings argument has this exact structure, repeated for all
later study review indexes as well:

```json
{
  "reviewer_kind": "agent_assisted",
  "reviews": [
    {
      "index": "/workspace/outputs/study-bootstrap/qualification-v1/compiled-default-vs-upstream-review/index.html",
      "index_sha256": "ACTUAL_64_CHARACTER_SHA256",
      "csv": "/workspace/outputs/reviews/bootstrap-v1/compiled-default.rated.csv",
      "csv_sha256": "ACTUAL_64_CHARACTER_SHA256"
    }
  ]
}
```

Only after sealing those files, read the keys to associate votes with variants.
The actual candidate panel must be accepted in every reviewed case for that
comparison. A complete CSV containing a rejected candidate is not acceptance.
Do not rerun the evaluator into the same qualification path to attach votes: its
review export can rewrite review artifacts. Keep the original evaluation and join
the separately bound votes, as the standalone reporter does.

## Decide the secondary reference before materializing the plan

Review compiler-default and compiled-conditioning independently. Declare the
development quality rationale and explicit numerical, perceptual and temporal
limits; do not choose the limits solely to include every observed compiler result.
Different limits against different references require separate explanations.
If compiled-conditioning fails the metrics/review decision, choose
`conditioning-reuse` as the secondary reference and retain failed compiler points
on the development curve. Exact conditioning reuse still requires its measured
all-32 zero-error evidence; the fallback is not an assumed proof.

Save a decision JSON with `selected_optimized_reference`, `accepted` or `rejected`
for each bootstrap variant, reason, metric/evaluation/source/review hashes,
threshold-draft hash, reviewer kind, UTC decision time, and
`held_out_consulted: false`. This is a development decision receipt, not a release
quality policy or headline numerical proof.

Use this intentionally invalid-until-filled draft shape for each reference:

```yaml
schema_version: 1
policy_type: reviewed_metric_tolerances
metrics:
  max_abs_error:
    maximum: null  # Explicit finite nonnegative development decision required.
    aggregation: max
  lpips:
    maximum: null
    aggregation: mean
  temporal_derivative_mae:
    maximum: null
    aggregation: mean
performance:
  calibrated: false
```

Optional additional supported limits are `mae`, `rmse`, `temporal_change_error`
and `freeze_fraction_increase`; each uses `mean` or `max`. Record per-limit
rationales in the decision receipt, since `drivebench.freeze` preserves the draft
hash and metric values but does not copy arbitrary rationale fields. It verifies
complete real development provenance; it does **not** make the subjective
acceptance decision or require every development variant to pass the chosen limits.

Create a new approved specification from the existing 18-variant draft only after
that decision. Its current SHA is
`0875b69cc7a6fee9dfe210a92fb3a9a48fc708365825c07af1e40fe0d038c69f`.
Preserve that original draft. Record its hash, the approved specification hash,
the exact branch and the decision-receipt hash in a new proposal sidecar. The
field `draft_status` is descriptive and not enforced by the CLI.

If **compiled-conditioning is accepted**, retain all 18 variant settings and
`optimized_reference: compiled-conditioning`. The primary feature and reduced-step
families then use the accepted compiled implementation with the same 2 GiB
conditioning cache. Mark the new approved specification's reference decision as
resolved; do not rewrite the draft.

If **compiled-conditioning is rejected**, create a separately named fallback
specification only after that decision, using this pre-plan mapping:

| Approved fallback name | Exact meaning relative to the preserved draft |
|---|---|
| `upstream`, `eager`, `conditioning-reuse` | Keep these settings unchanged; select `conditioning-reuse` as the secondary reference. |
| `compiled-default`, `compiled-conditioning`, `compiled-boundary-1` | Keep these settings unchanged as descriptive compiler controls, including the rejected outcomes. |
| `steps-48`, `steps-47`, `steps-46`, `steps-45`, `steps-40`, `steps-30`, `steps-20` | Keep the named step counts; use eager execution plus the 2 GiB conditioning cache, serial batch 1, no feature reuse and `reused_conditioning` inputs. |
| `feature-2`, `feature-3`, `feature-4` | Use eager execution plus the 2 GiB conditioning cache, output block 9 reuse at the named interval, serial batch 1 and `reused_conditioning` inputs. |
| `compiled-feature-2-control` | Copy the original draft's compiled `feature-2` settings exactly into the old `eager-feature-2` slot. |
| `compiled-steps-47-control` | Copy the original draft's compiled `steps-47` settings exactly into the old `eager-steps-47` slot. |

For the rebased eager families, both compiler model selectors must be absent or
null and `compile_denoiser`/`compile_decoder` must be false. An eager feature
variant retains `feature_cache_module: output_blocks.9`; the distinct compiler
selector `compiler_feature_boundary` is null. Step variants use feature interval 1
and a null feature module. Keep every other generation setting unchanged. This
mapping keeps 18 variants without duplicating the original eager ablations. It
also keeps the `feature-*` and `steps-*` prefixes within one implementation family
for runtime matching; the two compiled controls are excluded from that automatic
matching pool. The three external bootstrap adoptions remain unchanged.

The fallback requires its own budget calculation and actual 32-scene runtime
matching. Compiler probe predictions cannot establish matches for eager variants.
Its four held-out roles can be upstream, conditioning reuse, a feature variant and
its genuinely reduced-step control. Do not infer a speed gain for the exact cache
implementation before measuring the paired timing evidence. No fallback file has
been created by this procedure.

Use the chosen new file as `configs/full_study_registered_approved_v1.yaml`, or
substitute its actual distinct approved name in the following command. Check 18
unique variant names, the chosen secondary reference, unchanged manifests/order,
the three adoption settings and the decision sidecar before planning. No command
below has been executed by this procedure:

```bash
python -m drivebench.study plan \
  --spec configs/full_study_registered_approved_v1.yaml \
  --output outputs/full-study-v1

python -m drivebench.study quality --study outputs/full-study-v1 --phase development \
  --adopt upstream=/workspace/outputs/study-bootstrap/upstream \
  --adopt-source upstream=/workspace/outputs/host/study-upstream-development-attempt-1.json \
  --adopt compiled-default=/workspace/outputs/study-bootstrap/compiled-default \
  --adopt-source compiled-default=/workspace/outputs/host/study-bootstrap-compiled-default-attempt-1.json \
  --adopt compiled-conditioning=/workspace/outputs/study-bootstrap/compiled-conditioning \
  --adopt-source compiled-conditioning=/workspace/outputs/host/study-bootstrap-compiled-conditioning-attempt-1.json

python -m drivebench.study aggregate --study outputs/full-study-v1 --phase development
```

Before adoption, match resolved generation, model selectors, optimizations,
manifest/order, candidate count 1, checksums and 32 requests/retained outputs
against each planned variant. The historical upstream run used one warmup and
source `2a8fed…`; the two compiled bootstrap runs use 3–5 warmups and source
`d8a328…`. Preserve those differences in the adopted quality evidence. The
existing adoption gate permits this warmup difference for output comparison;
these runs do not establish headline timing eligibility. Host receipts must prove exit 0 and
unchanged sources. All three external bootstrap arrays remain retained; study
retirement deliberately excludes adopted runs. The remaining 15 variants still
need their full 32 development clips. A replacement quality runtime needs the
separately verified compatible overlay; keep the original study environment bound.

The final frozen reviewed policy must bind the **main study development aggregate**,
not merely the earlier qualification JSON. Complete its fixed development reviews,
then freeze two separately attributed files when both references require nonzero
policies, replacing `SECONDARY` with the decided reference:

```bash
python -m drivebench.freeze \
  --development-evaluation outputs/full-study-v1/quality/development/aggregate/evaluation-vs-upstream.json \
  --thresholds outputs/policies/upstream-development-draft-v1.yaml \
  --output outputs/policies/reviewed-upstream-v1.yaml
python -m drivebench.freeze \
  --development-evaluation outputs/full-study-v1/quality/development/aggregate/evaluation-vs-SECONDARY.json \
  --thresholds outputs/policies/secondary-development-draft-v1.yaml \
  --output outputs/policies/reviewed-secondary-v1.yaml
```

Freeze selection with the exact conditioning anchor, accepted final roles and
those reference-specific policy arguments as documented in
[registered-study-draft.md](registered-study-draft.md). Policies and selection
must precede held-out generation. Bootstrap reviews do not automatically approve
new main-study case IDs. If acceptance fails, retain descriptive results and do
not promote a failed current-source numerical gate into a passed one.

## Prepared diagnostic controller and storage/cost checks

The reviewed queue is already materialized locally: 33 entries, comprising 18
input benchmarks, 7 saved-output strict checks, 2 full numerical gates and 6
serial candidate-set benchmarks. It is not evidence that any entry has executed.
The new host controller waits for verified completion of memory-boundary-v1:

```bash
python3 outputs/host/run-remaining-diagnostics-v1.py \
  --plan outputs/host/remaining-diagnostics-v1-plan.json
```

Run that command only from the canonical host project after copying the reviewed
files in `outputs/host/remaining-diagnostics-v1-transfer-files.txt`. This procedure
does not launch it. The controller uses existing V1 launcher
`outputs/host/run-bounded-job.py`; it preserves the allocation deadline
`2026-10-27T15:15:03.458050Z`, requires 16 GiB free before each job, and applies
V1's stricter timeout plus 240-second cleanup margin. It binds 78 static inputs,
serializes jobs through the launcher's GPU lock, checks actual batch size 1 and
50 calls per candidate, hashes all retained arrays, and stops on any failure.
Fresh specs/host logs and the controller journal are preserved. It has no automatic
retry, overwrite or output deletion. A failed/interrupted controller requires a
new reviewed continuation attempt; do not erase its journal to restart it.

Expected remaining screens are about 5.4 hours and $25.16 at the planning rate,
not an ETA. Their saved input/scaling arrays total about 12.2 GiB; full gates can
add approximately 4.6 GiB. Preserve a checksum-verified local archive before any
remote retirement. Keep the roughly 15.8 GiB of three external bootstrap arrays
available for adoption and preserve their hashes/source receipts. Main study
processing additionally needs about 15.8 GiB for three active 32-clip float runs,
its 12 GiB reserve, selected raw examples, reviews and compiler caches. These peaks
must not be added to a nearly full 200 GiB disk without verified archiving.

The conservative **accepted-compiler branch** budget is 90.32–97.32 hours,
$420.97–453.59, before
prior allocations/discovery/storage/idle costs. The upper estimate leaves only
**$46.41** for those charges under the total $500 authorization. Before reserving
the full sequence, reconcile actual accrued charges and already-completed modeled
work so it is neither omitted nor double-counted. Do not describe the plan as
within budget merely because its standalone estimate is below $500. A candidate-4
confirmatory rerun alone can add about $116 and may be unaffordable; absent that
required confirmation, leave the affected conclusion unresolved. The existing
16-hour deletion guard stays unchanged. The full candidate-4 pilot/comparison
needs a separately bounded allocation long enough for its same-GPU evidence.
A 72-hour/$340 allocation and an 84-hour/approximately $390 allocation are
unfunded, undecided proposals, not existing reservations or spending approvals
beyond the cumulative $500 limit. Reconcile the live ledger and remaining modeled
work after the current allocation's guarded cleanup before choosing a new bound.
The rejected-compiler branch needs a fresh estimate: fewer held-out roles do not
automatically compensate for slower eager headline finalists.
