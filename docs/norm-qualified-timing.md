# Separate norm-qualified compiler timing

`scripts/run_norm_timing_companion.py` provides a new, explicitly named numerical contract for the frozen d8 engine. The existing schema-2 strict proof, benchmark flags, variability pilot, timing companion and reports remain unchanged. The compiler's strict-zero result remains **false** in every raw block and in the new result. This helper does not establish heldout quality or authorize an allocation.

The only supported pair is the upstream reference and one compiler finalist already present in the immutable development selection. Both use 50 denoising steps, serial candidate execution, and no feature boundary or feature reuse. The supported workloads are one future and four candidates; each needs its own upstream variability pilot on the timing GPU. There is no automatic fallback to another finalist and no extra quality generation.

## Prerequisites

Before planning, retain and verify all of the following:

- The frozen parent study, development selection, reviewed quality policies and their calibration artifacts.
- A passed **strict-zero** upstream numerical proof, with retained outputs, on the timing allocation.
- A separate compiler norm qualification created and verified by `scripts/qualify_compiler_numerics.py`. Its strict proof must remain failed, its exact candidate order/action-isolation checks must pass, and enabled conditioning caching must pass its own exact checks. Its numerical, development-quality and localized-review criteria must all pass. The qualification must bind the same 32-scene development manifest as the parent study.
- Four independently warmed 50-request upstream pilot blocks for this workload. Freeze their variability-derived policy before timing. The numerical policy and norm qualification must also precede timing; the qualifier separately enforces pre-heldout chronology.
- The same complete runtime identity, including GPU UUID, for the pilot, proofs, qualification and every timing block. An optional replacement-allocation config changes only the validated environment proof/lock paths; compatibility with old quality evidence does not permit mixing timing GPUs.

The helper revalidates retained qualification and proof artifacts before and after each block. Raw proof arrays and the qualifier's development arrays must remain available. These reads can take time and must be included in admission estimates.

## Prepare the original upstream pilot

These commands are templates. The operator supplies actual immutable artifact paths, the predeclared pilot floor, and a verified allocation/deadline. They do not choose numerical tolerances.

```sh
python scripts/run_timing_companion.py prepare-pilot \
  --study "$STUDY" --output "$PILOT_CONFIG" \
  --baseline upstream --candidate-count 4 \
  --numerical-proof "$UPSTREAM_STRICT_PROOF" \
  --runtime-config "$TIMING_RUNTIME_CONFIG"

python -m drivebench.calibrate_pilot plan \
  --config "$PILOT_CONFIG" --output "$PILOT_DIRECTORY" \
  --relative-tolerance-floor "$PREDECLARED_PILOT_FLOOR"
python -m drivebench.calibrate_pilot run --pilot "$PILOT_DIRECTORY"
python -m drivebench.calibrate_pilot freeze \
  --pilot "$PILOT_DIRECTORY" --output "$PILOT_POLICY"
```

Omit `--runtime-config` only when the original environment proof and lock are still valid for the timing GPU. Use a separate config, pilot directory and policy with candidate count 1 for the single-future workload. Pilot commands perform inference only at `run` and require the normal external GPU lock, budget and allocation guard.

## Plan and execute complete blocks

```sh
python scripts/run_norm_timing_companion.py plan \
  --study "$STUDY" --output "$NORM_TIMING_DIRECTORY" \
  --baseline upstream --candidate "$FROZEN_COMPILER_FINALIST" \
  --candidate-count 4 --pilot-policy "$PILOT_POLICY" \
  --baseline-proof "$UPSTREAM_STRICT_PROOF" \
  --qualification "$NORM_QUALIFICATION" \
  --runtime-config "$TIMING_RUNTIME_CONFIG"

python scripts/run_norm_timing_companion.py run \
  --companion "$NORM_TIMING_DIRECTORY" --max-new-blocks 1 \
  --deadline "$ABSOLUTE_ALLOCATION_DEADLINE" \
  --block-budget-seconds "$COMPLETE_BLOCK_BOUND_SECONDS" \
  --cleanup-seconds 240
```

The immutable order is upstream, compiler, compiler, upstream, compiler, upstream, upstream, compiler. Each slot is a fresh benchmark process with 3–5 stable warmups and 50 measured requests. This yields **200 measured requests per variant**, with no frame export inside timing. Request, seed and action identities are paired; blocks use the same first 50 requests of the frozen 128-clip heldout manifest. Repeated blocks provide 200 observations, not 200 independent scenes. Reports identify the actual scene count.

`--max-new-blocks 1` executes at most one new complete slot. An intentional pause returns exit code 0 with `status: paused`, `incomplete: true` and `timing_qualified: false`; an immutable pause receipt records the reason and remaining slots. An orchestrator must inspect status and must not treat exit code 0 as study completion. Invoke the same command again under the same valid allocation guard to continue. Existing completed blocks are revalidated before new work. Only all eight verified slots produce `completed.json` and `status: complete`.

Deadline admission requires both an absolute timezone-aware deadline and a positive full-block bound, with at least 240 seconds reserved for cleanup. It checks immediately before a new slot. The bound is an empirical admission estimate, not a timeout guarantee; the outer launcher must still enforce the absolute allocation deadline and cleanup. No running block is deliberately interrupted by this helper. A started block without its norm completion is preserved and causes a hard refusal to retry automatically. Investigate it before preparing a separately reviewed recovery; never delete it to conceal a failure.

Before setting the launcher's invocation timeout, measure one complete qualifier verification on the actual retained artifacts using CPU only:

```sh
/usr/bin/time -p python scripts/qualify_compiler_numerics.py check \
  --qualification "$NORM_QUALIFICATION"
```

Record the wall time, helper/artifact hashes, machine and whether files were already cached. This is a planned measurement after the qualification exists; no such runtime is assumed here. Include repeated qualifier checks, strict-proof hashing, completed-prefix replay and final reconstruction in the **whole invocation** bound. For a one-new-block invocation with `k` previously completed slots, the current implementation calls the full qualifier `k + 4` times before returning a pause (`k = 0…6`). The final slot calls it 14 times because completion is independently reconstructed and verified. It also validates every completed prefix block and rechecks all eight blocks during final reconstruction. The before-slot admission check occurs after prefix verification, so the external launcher must separately cover that work. A four-hour candidate4 estimate is provisional; set a conservative bound for the actual resume depth from measured verification cost plus complete generation/warmup and other validation costs. The allocation guard remains mandatory.

Every process must retain positive compiled graph/kernel evidence, runtime restoration, stable warmups, no measured recompilation, no fallback/OOM, complete serial denoiser work, unchanged source/configuration and a passed environment gate. The compiler's canonical numerical gate is recomputed and must remain false. A distinct `norm_execution_eligible` field describes this new contract; raw flags are never rewritten.

## Verification and interpretation

```sh
python scripts/run_norm_timing_companion.py verify \
  --companion "$NORM_TIMING_DIRECTORY"
```

`verify_completed(directory)` reconstructs results from the eight original blocks and revalidates the plan, pilot, qualification, raw artifacts, process receipts, launch configs and timing chronology. It returns `compiler_norm_timing_result_v1`. It does not manufacture legacy aggregate runs or change original observations.

The comparison reports p50/p95 latency, block medians, paired scene-cluster bootstrap uncertainty and the unchanged pilot tolerance. A regression test passing does not imply a speedup. Improvement has a separate flag requiring both the estimated reduction and its uncertainty interval to exceed the pilot tolerance. Apparent regressions return `rerun_required`. A separately collected complete companion with identical frozen configuration, qualification, pilot and GPU may confirm the same failing metric:

```sh
python scripts/run_norm_timing_companion.py confirm \
  --primary "$NORM_TIMING_DIRECTORY" --confirmation "$INDEPENDENT_CONFIRMATION_DIRECTORY"
```

This command is read-only and never schedules a rerun. It rejects copied/overlapping observations and changed identities. A confirmation is additional measured work and needs its own budget admission.

The timing result always sets `combined_headline_claim: false`. A separate supplementary report must join complete heldout quality, second-seed, both same-seed action phases and accepted blinded review evidence for the identical selected implementation before any combined quality-preserving claim. Missing evidence is a failed claim gate, not permission to infer quality from speed, norm error or development results. These checks do not establish vehicle safety, trajectory accuracy or real-time deployment suitability.
