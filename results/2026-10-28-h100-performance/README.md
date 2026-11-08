# Compact H100 performance study — norm qualification rejected

The 32-scene development comparison and selected-request semantic checks are complete. Seven blinded static review cases are sealed: six accepted with no preference and one rejected for a localized front-car geometry/detail difference at generated frame 24. **The completed machine-checked qualifier returned qualified:false.** Its numerical contract, 32-scene development limits and selected-request exact semantics passed; the unchanged localized-review gate failed. No quality-preserving acceleration or GitHub publication is claimed.

The workload remains 25 total frames (one observed and 24 generated), 576×1024, 50 steps, FP16, decoder chunk 14, with uncached single-future inputs. The original 18-variant plan and prior H100/A100 evidence remain unchanged. A100 and H100 timings are separate.

| Development comparison versus upstream | Pairs | Maximum RMSE | Maximum LPIPS | Maximum temporal-derivative MAE | Added low-change fraction |
|---|---:|---:|---:|---:|---:|
| Eager refactor | 32 | 0 | 0 | 0 | 0 |
| Default compiler | 32 | 0.007359183 | 0.001877701 | 0.002356518 | 0 |

Every eager pair was exactly equal. The compiler's development metrics satisfy the unchanged per-clip limits: RMSE≤0.01, LPIPS≤0.003, temporal-derivative MAE≤1/255 and no increase in the thresholded low-change fraction. Meeting these limits does not override localized review. That fraction is a pixel-change proxy, not a claim about correct object motion.

Descriptive median request durations were 57.3124s upstream, 57.4160s eager and 43.6498s compiled. These are separate development runs, not interleaved 200-request timing blocks; no headline speedup is established.

The compiler-boundary control completed 32 pairs but exceeded LPIPS 0.003 on one scene (maximum 0.004438987). It remains a descriptive control outside the primary norm-qualification contract.

The completed all-mode diagnostic preserves its failed strict-zero upstream comparison. Exact candidate-order, unaffected-candidate action isolation, conditioning-cache agreement, cache-key and cache-exercise checks passed for the selected request and two serial candidates. This does not establish behavior across all scenes or action adherence.

An earlier all-mode attempt ran out of GPU memory before completing isolation. Its two saved outputs and strict disagreement remain preserved. A separate [validation cleanup helper](../../research/performance-validation-v1/run_validation_cleanup.py) resets compiler process caches only between fully completed engines, then invokes the existing garbage collection and GPU-cache cleanup. The retry completed all five engines without modifying frozen inference code or within-engine checks. This is diagnostic lifecycle management, not a timing optimization or proof of the original OOM cause.

Three independent agents inspected seven ordered 24-frame atlases and ten distinct selected native pairs. The observations were sealed before model attribution. Review was static, not playback or human assessment. Common defects remain visible on accepted pairs, and one observable localized geometry difference was rejected. After sealing, attribution showed that the reviewer preferred the compiled display: the less coherent front-car shape was on the upstream reference. This is not evidence that compilation made the frame worse. The sealed rejection and unchanged conservative review gate are preserved. No images or private review mappings are included.

Reduced held-out, second-seed, action and final timing work have no completed result in this snapshot. The [qualification decision](qualification.json) is final for this frozen contract. Compute cleanup was verified: zero running instances, with the 300 GB experiment disk detached and preserved. Funding is $100; the recorded balance is $17.24872, an observed cumulative decrease of $82.75128. Retained storage continues at $0.08219178/hour. Automatic top-up remains disabled. The experiment and GitHub publication remain incomplete.

[Numeric summary](summary.json) · [All 64 development pair projections](development-pairs.json) · [File hashes](manifest.json)
