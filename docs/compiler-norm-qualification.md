# Compiler numerical qualification

This optional, versioned contract permits a development-qualified RGB error norm while retaining the original failed strict-zero proof. It does not prove bitwise or mathematical equivalence, heldout quality, candidate batching, or a speedup. The original validator, semantic sidecar, timing companion, and package source remain unchanged.

The supported candidate is the frozen `d8a32813492c0939a0a91573a945c914938972419946065e6822f6209fa470cb` source, registered default compiler, static shapes, serial candidate execution, 25 frames at 576×1024, 50 steps, FP16 autocast, and decode chunk14. Feature caching, decoder compilation, dynamic shapes, and joint candidate batches are outside this contract.

The declared RGB bound is **RMSE≤0.01 for each candidate/clip over its24 generated frames**, with RGB normalized to[0,1]. The conditioning frame is excluded. The complete32-development main comparison must also satisfy maximum per-clip LPIPS≤0.003, temporal-derivative MAE≤1/255, and freeze-fraction increase0. These are development-selected error/quality limits, not a replacement `atol` fitted to the largest pixel error. Maximum errors, locations, per-frame errors and the actual first-denoiser diagnostic remain visible.

Candidate-order and unaffected-candidate checks stay exact-zero. Enabled conditioning reuse additionally requires exact cached/uncached output agreement, invalidation keys and exercised reuse paths. An unused cache diagnostic may fail for a cache-disabled candidate; its failure remains in the result and cannot qualify a cache-enabled candidate.

## Prerequisites and order

1. Complete the main32-development comparison and its committed shard/evaluator receipts. Retain all32 original upstream and selected compiled NPY arrays, all selected-proof NPZs, model assets and provenance. `verify()` rehashes and recomputes them; a declared pass or retired array is insufficient. Adopted bootstrap references are preserved by the existing retirement policy.
2. Run the existing all-mode proof on the intended timing GPU with the selected exact configuration and a fresh output path. Preserve its strict failure:

   ```bash
   python -m drivebench.validate --config SELECTED_RUNTIME_CONFIG.yaml \
     --mode all --request-index 28 --candidates 2 --atol 0 --rtol 0 \
     --output artifacts/norm-v1/strict-all.json
   ```

   This requires a separately authorized serialized GPU window. There must be no fallback/OOM and all compiler executions must record their exact runtime implementation, positive owned graph/kernel counters and restored state. At least two serial candidates establish the selected-request semantic check; this is not all-scene or joint-batch evidence.
3. Freeze the reviewed quality policy against the **main** development aggregate before heldout generation. Bootstrap-only review is insufficient. Save this threshold YAML as `artifacts/norm-v1/declared-quality.yaml`:

   ```yaml
   metrics:
     rmse: {maximum: 0.01, aggregation: max}
     lpips: {maximum: 0.003, aggregation: max}
     temporal_derivative_mae: {maximum: 0.00392156862745098, aggregation: max}
     freeze_fraction_increase: {maximum: 0.0, aggregation: max}
   ```

   ```bash
   python -m drivebench.freeze \
     --development-evaluation STUDY/quality/development/aggregate/evaluation-vs-upstream.json \
     --thresholds artifacts/norm-v1/declared-quality.yaml \
     --output artifacts/norm-v1/reviewed-quality.yaml --minimum-scenes 32
   ```

4. Create a new norm policy JSON. Bind each input as `{"path":"ACTUAL_PATH","sha256":"ACTUAL_SHA256"}`. The policy requires these fields:

   | Field | Required value |
   | --- | --- |
   | `type` | `compiler_norm_policy_v1` |
   | `status` | `frozen`, only after all prerequisites exist |
   | `created_at`, `frozen_at` | Actual timezone-aware timestamps; never backdate |
   | `source_identity` | The full d8 source hash above |
   | `held_out_not_started_at_freeze` | `true`, supported by the declared study chronology |
   | `variant` | The exact selected main-study variant name |
   | `study` | Bound main `plan.json` file |
   | `development_evaluation` | Bound main upstream aggregate evaluation |
   | `quality_policy` | Bound newly frozen reviewed YAML |
   | `component_evidence` | Bound completed actual first-denoiser diagnostic JSON |
   | `numerical`, `review_rule` | Exact `NUMERICAL` and `REVIEW_RULE` constants from the qualifier |
   | `runtime_config` | Bound audited replacement-runtime YAML, if replacing the original GPU |

   A pending template must retain `status: pending_prerequisites_not_frozen`, null timestamps and incomplete bindings. It cannot qualify. Use actual files and hashes, not names alone. The optional runtime overlay may change only verification/lock paths and GPU UUID/PCI address; all remaining GPU/runtime settings must match. The selected proof, baseline pilot and every timing block must share the exact fresh GPU identity. The historical upstream source remains separately attributed.

## Export and independently review

Create the private case plan after the frozen policy and completed proof exist:

```bash
python scripts/qualify_compiler_numerics.py review-plan \
  --proof artifacts/norm-v1/strict-all.json --policy artifacts/norm-v1/policy.json \
  --output artifacts/norm-v1/private-review-plan.json
python scripts/export_compiler_proof_review.py \
  --proof artifacts/norm-v1/strict-all.json \
  --review-plan artifacts/norm-v1/private-review-plan.json \
  --output artifacts/norm-v1/proof-review
```

The bridge exposes only `proof-review/public/` to a blinded reviewer. Its private directory contains0600 mappings, an export receipt, and exact derived NPY copies. Source NPZs remain unchanged. Public PNG pairs retain native display pixels using `round(value×255)`; these are not lossless encodings of the original floating-point arrays. Ordered atlases use downscaled, palette-quantized previews. Both the greatest absolute-error frame and greatest frame-RMSE frame are highlighted using recomputed indices. All24 native pairs are retained, even though the required inspection is all24 ordered atlas frames plus the two selected native frames. Export leaves every review form blank.

The private plan also selects four development cases by a fixed identity hash, plus the two worst cases for each of clip RMSE and maximum pixel error. It always adds every selected-proof candidate. Export development cases from their **original source-shard evaluation**, whose arrays remain available, using the existing helper:

```bash
python scripts/export_blind_metric_cases.py \
  --evaluation STUDY/quality/development/shards/SHARD/metrics/upstream--VARIANT.json \
  --request REQUEST_ID --output artifacts/norm-v1/development-review-01
```

Each invocation supports up to four `--request` arguments. The helper retains all24 native frames; inspect the exact two frame indices recorded in the private norm review plan. The aggregate run's intentionally retired `frames_path` pointers are not source-array paths.

Review only public opaque A/B assets. Seal the independent ratings before decoding mappings. Any new disappearance, duplication, geometry defect, freezing or other localized failure vetoes qualification. Uncertainty or inability to inspect also rejects; ties are permitted. Existing bootstrap ratings are not automatically transferred.

Assemble a separate `compiler_norm_localized_review_v1` receipt with actual timestamp, policy/proof hashes, exact review rule, `held_out_consulted:false`, reviewer kind (`human` or `agent_assisted`) and every selected case. Each case must carry its exact source hashes, `native_frame_indices`, `frames_reviewed:[1,...,24]`, `ordered_atlas_frames_reviewed:true`, both `native_error_frames_reviewed` labels, observations, explicit failure-category booleans, `new_localized_failure`, `decision`, `preference`, and retained asset path/SHA256 bindings. These assertions belong to the reviewer and sealed assembly; neither exporter creates them. Software verifies declared coverage and assets, not the subjective act of viewing.

## Create and recheck

```bash
python scripts/qualify_compiler_numerics.py create \
  --proof artifacts/norm-v1/strict-all.json --policy artifacts/norm-v1/policy.json \
  --review artifacts/norm-v1/sealed-review-receipt.json \
  --output artifacts/norm-v1/qualification.json
python scripts/qualify_compiler_numerics.py check \
  --qualification artifacts/norm-v1/qualification.json
```

Creation uses a fresh path and occurs before any heldout attempt. A completed check that fails its criteria retains `qualified:false`; it cannot be consumed. Verification permits later heldout work only when its recorded start follows both policy freeze and qualification creation. The declared study directories bound this chronology; the tool cannot discover undisclosed experiments elsewhere.

The new timing companion consumes `verify(path)` and the unchanged exact upstream variability pilot. It preserves `legacy_strict_passed:false` and the core compiler benchmark's failed strict gate. Separate heldout quality, blinded review, and repeated timing evidence are still required. No timing or quality acceptance is granted by merely creating this numerical report.

CPU checks:

```bash
python -m pytest -q tests/test_compiler_norm_qualification.py \
  tests/test_compiler_proof_review_export.py tests/test_norm_timing_companion.py
```
