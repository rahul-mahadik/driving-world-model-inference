# A100 math-SDPA: 32-scene baseline and recorded-future diagnostics

This report is a snapshot at **2026-10-28T04:07:13.732204+00:00**. The separate
A100 math-SDPA eager50 companion completed all **32 development requests** at
**2026-10-28 03:03:15 UTC**. The final host receipt completed successfully at
03:03:24 UTC, with unchanged source/inputs, restored policy/platform and verified
exact-container cleanup. There were no failed requests, OOMs or fallbacks.

Each scene used one recorded-action candidate, 50 denoising steps, FP16, one
observed frame and 24 future frames at 576×1024. Three warmups preceded the
32 measured requests. This was a fresh full-manifest run; four-scene smoke
timings were not pooled. All 800 observed/future image hashes were bound by the
producer, while only the 32 observed images reached inference.

The descriptive median request duration was **101.591740 seconds**
(range 101.430086–101.753237). These observations include math-policy
instrumentation and are not accepted matched latency, tail-latency, throughput
or speedup results. Warmup and model-load values remain separate in the
[machine-readable report](summary.json).

The complete generation archive is now locally verified: **all 54 files,
5,663,942,416 bytes**, including all 32 retained arrays, were independently
rehashed for this publication. File sizes, exact file set, stability during local
rehashing and all generation/report output hashes match the server inventory.
The archive receipt also records matching remote inventories before and after
transfer. This full check supersedes the earlier metadata-only backup status.
No raw arrays are included in the publication, and hashing them is not a new
local recomputation of the diagnostic metrics.

The **CPU recorded-future diagnostics completed at 03:14:56 UTC**, and the final
host exited successfully at 03:15:07 UTC with exact-container cleanup. All
**eight diagnostic archive files, 723,645 bytes**, were independently rehashed.
All 32 generated/recorded-future pairs and their 24 future-frame hashes match
the generation and frozen development manifest; the observed frame was excluded.
The evaluator used NumPy/Pillow and initialized no CUDA context before or after.
Source, inputs and platform stayed unchanged. Its 1,073 input keys normalize
to the final host's 1,069 pins through four consistent absolute/relative aliases,
with no missing or conflicting normalized pins.

| Recorded-future diagnostic | Mean across 32 scenes | Largest scene value |
|---|---:|---:|
| RGB RMSE | 0.112928 | 0.209230 |
| Temporal-derivative MAE | 0.043463 | 0.129918 |

**Eight of 32 scenes** have additional transitions at or below the `1/255`
global mean pixel-change threshold; every recorded reference has zero of 23
such transitions. Candidate counts range from zero to all 23. These are
frame-indexed pixel-change proxies, not judgments of literal freezing,
physical-time motion, action adherence or driving safety. Per-scene metrics,
counts and source-recorded bootstrap intervals remain in the JSON report.
Saved metrics were verified; they were not recomputed locally from raw frames.

No LPIPS or review export was produced, and no tolerance was calibrated. The
generic numerical, quality and headline gates remain false. The existing
strict pass covers one selected development request, not all 32 scenes.

This companion preserves the frozen d8 implementation, the original 18-variant
H100 plan and [the earlier A100 snapshot](../2026-10-27-a100-pilot/README.md).
It does not consume held-out scenes, select an optimized candidate, establish
action adherence or safety, or complete the broader study. Source inspection
copies preserve original assumptions and missing private dependencies; prior
V5 test results are not attributed to the new helpers.

[Report file manifest](manifest.json) · [Source archive](../../research/a100-pilot-v1/README.md)
