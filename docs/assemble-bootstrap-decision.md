# Assemble the development reference decision

`scripts/assemble_bootstrap_decision.py` is a standalone CPU helper. It verifies
the two completed bootstrap evaluations, sealed blind ratings and explicitly
chosen metric limits before writing a new approved specification and decision
receipt. It does not choose limits, make the subjective decision, create a study
plan, freeze a release policy, or turn a historical strict failure into a pass.
The frozen package, original 18-variant draft and old outputs remain unchanged.

Run it on the canonical project filesystem after both comparisons and ratings
are complete. It streams SHA256 checks of the retained source arrays and all
other evaluator-bound inputs; it does not load video arrays or import LPIPS/Torch.
Allow disk I/O time and keep the original arrays available. All paths in evidence
must resolve: there is no permissive local-path substitution.

## Seal ratings before opening answer keys

The reviewer writes one receipt per comparison using this schema. The values
below are placeholders, not real votes or a ready-to-run receipt:

```json
{
  "schema_version": 1,
  "receipt_type": "bootstrap_blind_ratings_v1",
  "comparison": "compiled-default",
  "created_at": "UTC_TIMESTAMP",
  "reviewer_kind": "agent_assisted",
  "protocol": {
    "path": "/workspace/outputs/bootstrap-review-protocol-v1.json",
    "sha256": "c78b6e40a37e79b3c9942caee945f6d868f66ffaa8a2ef138da068033d0bbdbd"
  },
  "ratings": {"path": "ABSOLUTE_SEPARATE_RATED_CSV", "sha256": "ACTUAL_SHA256"},
  "review_index": {"path": "ABSOLUTE_ORIGINAL_INDEX_HTML", "sha256": "ACTUAL_SHA256"},
  "reviewed_case_ids": ["CASE_1", "CASE_2", "CASE_3", "CASE_4"],
  "reviewed_frames_per_case": {"CASE_1": 24, "CASE_2": 24, "CASE_3": 24, "CASE_4": 24},
  "all_panels_reviewed": true,
  "private_keys_read_before_seal": false,
  "uncertain_case_ids": [],
  "review_artifact_sha256": {"ABSOLUTE_INSPECTED_ARTIFACT": "ACTUAL_SHA256"}
}
```

The artifact map must include the original index and blank CSV, every case PNG
and GIF, plus any inspected atlases/lossless exports. Exclude the private key.
List any case with an uncertain preference or either uncertain acceptance vote
in `uncertain_case_ids`. Preserve the complete CSV header documented in
[bootstrap-decision-workflow.md](bootstrap-decision-workflow.md). Save the receipt
exclusively and send its SHA256 separately to the decision maker. The helper
requires those externally supplied hashes and checks **both seals before parsing
either key**. A reviewer declaration is evidence of the protocol followed; it is
not an independently observable guarantee about what the reviewer previously saw.

## Supply explicit decisions and limits

After reviewing the complete 32-scene diagnostics and the sealed ratings, write
a new decisions JSON. Replace every null below with the actual decision/rationale;
the template is deliberately rejected as incomplete:

```json
{
  "schema_version": 1,
  "created_at": "UTC_TIMESTAMP_AFTER_BOTH_REVIEW_SEALS",
  "reviewer_kind": "agent_assisted",
  "held_out_consulted": false,
  "selected_optimized_reference": null,
  "reference_rationale": null,
  "variants": {
    "compiled-default": {
      "status": null,
      "rationale": null,
      "policy_rationales": {
        "max_abs_error": null,
        "lpips": null,
        "temporal_derivative_mae": null
      }
    },
    "compiled-conditioning": {
      "status": null,
      "rationale": null,
      "policy_rationales": {
        "max_abs_error": null,
        "lpips": null,
        "temporal_derivative_mae": null
      }
    }
  }
}
```

Each status is `accepted` or `rejected`. The selected reference is
`compiled-conditioning` only when its explicit status is accepted, otherwise
`conditioning-reuse`. A rejected variant remains rejected even when all metrics
fit its limits; the helper never selects acceptance automatically. An accepted
variant must fit every supplied limit and have an explicit candidate-panel `yes`
in all four cases. Rejected or uncertain candidate votes prevent acceptance.

Supply unfrozen YAML quality drafts with explicit finite nonnegative numerical,
LPIPS and temporal limits, as shown in the workflow document. Each metric requires
both `maximum` and `aggregation: mean|max`; nulls/default inference are forbidden.
Each chosen metric must have a matching nonempty rationale in the decisions JSON.
Both bootstrap comparisons are against upstream. Using the same explicit policy
file for both comparisons makes that shared criterion visible; different files
remain separately attributed and do not create a common final study policy.
Performance calibration must remain false.

```bash
outputs/venvs/vista-upstream-cu118-sm90/bin/python scripts/assemble_bootstrap_decision.py \
  --decisions outputs/reviews/bootstrap-v1/explicit-decisions-v1.json \
  --ratings compiled-default=outputs/reviews/bootstrap-v1/compiled-default.seal.json \
  --ratings-sha256 compiled-default=EXTERNALLY_SAVED_DEFAULT_SEAL_SHA256 \
  --ratings compiled-conditioning=outputs/reviews/bootstrap-v1/compiled-conditioning.seal.json \
  --ratings-sha256 compiled-conditioning=EXTERNALLY_SAVED_CONDITIONING_SEAL_SHA256 \
  --policy compiled-default=outputs/policies/bootstrap-upstream-explicit-draft-v1.yaml \
  --policy compiled-conditioning=outputs/policies/bootstrap-upstream-explicit-draft-v1.yaml \
  --output outputs/study-proposals/approved-bootstrap-v1
```

The fresh output directory contains `approved-spec.yaml` and
`decision-receipt.json`. The receipt binds the preserved draft, all current package
sources, evaluator sources, completed generation/evaluation receipts, retained
arrays, metrics, review artifacts, sealed votes, explicit decisions, policy drafts
and helper itself. It records attributed candidate votes privately. These files
contain private local paths and must not be published without sanitization.

The accepted branch preserves all 18 variant settings. The rejected branch uses
the documented eager/conditioning primary step and feature families, keeps the
three compiler controls and replaces duplicate eager ablations with the original
compiled feature-2 and step-47 controls. It retains 18 variants and all three
bootstrap adoption settings. No source mutation or generation occurs.

After reviewing this concrete result, plan with its exact new path:

```bash
python -m drivebench.study plan \
  --spec outputs/study-proposals/approved-bootstrap-v1/approved-spec.yaml \
  --output outputs/full-study-v1
```

Then use the three unchanged adoption arguments in
[bootstrap-decision-workflow.md](bootstrap-decision-workflow.md). The separate
main-study aggregate, frozen policies, exact conditioning anchor, current
numerical proofs, held-out quality evidence and timing gates remain required.
Reconcile the selected branch with the live budget before generation; the helper
creates no reservation, funding or spending authorization.
