# A100 math-SDPA: four-scene 47-step screen

Evidence snapshot at **2026-10-28T04:46:09.012565+00:00**. All numerical, calibrated-quality,
matched-latency, speedup, H100, action, safety and full-study acceptance flags
remain **false**. [Machine-readable evidence](summary.json) records the exact
metrics, hashes and limits; [builder history](builder-history.json) preserves the
local preparation failure and separate reviewed repair.

## Completed generation and comparison

The 47-step eager candidate completed four fixed development smoke requests and
one warmup, with 47 denoiser evaluations per request and no failed requests,
OOMs or fallbacks. Generation finished at **2026-10-28T04:22:42.767729+00:00**; its host
completed with exit 0 and two absence checks at **2026-10-28T04:22:50.325955+00:00**.
The process-wide math-SDPA policy, frozen d8 engine and original runtime
restoration checks were preserved. Every clip contains one conditioning frame
and 24 generated future frames at 1024×576.

The CPU comparison completed at **2026-10-28T04:31:40.596956+00:00** and its host at
**2026-10-28T04:31:53.962482+00:00**, also with exit 0 and two absence checks. Source and
inputs were unchanged and the platform restored. LPIPS/alex ran on CPU without
initializing a CUDA context before or after evaluation. The reference is the
earlier four-scene math-SDPA eager50 run, **not** the 32-scene companion or recorded
future video. All four outputs paired completely; only 24 future frames per pair
were compared.

| Pair metric | Mean across four scenes | Maximum |
|---|---:|---:|
| freeze_fraction_increase | 0 | 0 |
| lpips | 0.000990859280402 | 0.00325386686503 |
| mae | 0.00105066911897 | 0.00239777758543 |
| max_abs_error | 0.241488963366 | 0.409398078918 |
| rmse | 0.00313280186588 | 0.00801896763522 |
| temporal_change_error | 8.09671640833e-05 | 0.000206246820912 |
| temporal_derivative_mae | 0.00102910134449 | 0.00302808279744 |

The saved temporal fields are whole-frame image-difference proxies. Both sides
have **[0,19,3,1]** transitions at or below the global pixel-change threshold,
out of 23 future-frame transitions per case. Thus all four have zero *additional*
thresholded transitions against eager50; this is not a finding of no low-change
transitions, literal freezing, natural motion, action adherence or safety.

Root independently recomputed all non-LPIPS pair metrics from all four retained
raw pairs with the frozen metric implementation, excluding conditioning frames;
saved results matched to an absolute and relative recheck tolerance of 1e-12.
That tolerance checks stored arithmetic and is not a quality-acceptance margin.
Aggregate means and maxima were rechecked. LPIPS itself was not recomputed
locally; it remains the archived remote CPU result. CPU/GPU LPIPS equivalence
is not claimed. No quality tolerance was calibrated or changed.

## Two independent blind static reviews

Each agent inspected all 96 ordered A/B future-frame pairs in downscaled atlases
and 13 selected native complete pairs. Reviewer 1 additionally inspected native
crops. Each declared that it had not seen source mapping, private attribution,
metrics, raw generation outputs or the other review. All 192 input-hash bindings
across the two receipts match the same 96 archived public PNGs.

Both reviewers gave **four ties**, with **two cases acceptable on both sides
and two unacceptable on both sides** under their restricted static criteria.
The shared unacceptable cases were the crossing white delivery truck
(case-2ea54225405fd97c550e554f), whose body and wheel contours deform in the
middle and later frames, and the brown delivery vehicle
(case-aa3aceef38aaf397a06db31f), whose lower body and wheel area distort as the
viewpoint advances. The other two cases received limited positive geometry
judgments; distant details remained uncertain.

These are subjective static agent observations, not human review or playback.
Only selected frames were viewed at native scale. A tie does not establish
pixel identity, statistical equivalence, noninferiority, absolute quality,
real-time smoothness, action adherence, safety or a release decision. The saved
evaluator gate remains false and is not promoted by these later observations.

## Timing and evidence integrity

The candidate's observed median request duration was **96.216807112s**
and model loading took **35.865635708s**. These are descriptive
standalone durations from this short screen. No matched latency, speedup or
pooling with historical H100 or earlier IO-confounded timings is claimed.

The full generation archive (**27 files, 708,341,875 bytes**, including four arrays)
and comparison archive (**129 files, 123,361,054 bytes**) were independently rehashed
for this publication. All 156 files match their remote inventories in exact file
set, size and SHA256, with stable local files during rehash. Transfer receipts
record matching remote inventories before and after copying; root separately
verified both archives. No raw arrays, media or private operational records are
included in this report.

## Historical source and remaining scope

The original generation builder stopped during local specification preparation
because admission contained four identical relative aliases as well as absolute
paths. It wrote no launch specification and launched no generation. Its exact
source is preserved beside the separately reviewed v2 repair, which validates
the same 263 normalized pins and accepts only the four named identical aliases;
no freshness, provenance or acceptance gate was relaxed. The recorded failure
timestamp is a receipt timestamp, not an inferred execution time.

The actual comparison used the separately reviewed comparison builder v2 and
evaluator. The omitted original comparison builder is retained by hash as an
operational dependency: the private archiver imports its utilities. It is not
described as unused. The [historical source guide](../../research/a100-pilot-v1/README.md)
documents the six new exact copies and missing private dependencies. Original
deadlines and path assumptions remain; relocated copies are not runnable.

Frozen d8 source, existing tests/configuration, historical H100 evidence, the
18-variant plan, the [32-scene A100 report](../2026-10-28-a100-development/README.md)
and [earlier A100 report](../2026-10-27-a100-pilot/README.md) are unchanged.
The separate math-SDPA strict pass still covers only one selected development
request, while the original default-runtime check failed. Held-out evaluation,
calibrated quality, matched performance and full-study completion remain open.
V5's 1,095 CPU tests and five synthetic CLI checks retain prior attribution and
were not rerun or assigned to these new helpers. This revision adds only syntax
parsing and exact-hash checks for those six source/configuration copies.
