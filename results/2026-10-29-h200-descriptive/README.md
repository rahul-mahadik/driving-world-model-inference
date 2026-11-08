# Preliminary H200 compiler speed–quality trade-off

Two completed blocks were recovered from an interrupted experiment: **50 measured
requests per role, against the planned 200 per role**. All 100 completed requests
succeeded without OOMs. Each role used three warmups and the same ordered 50
distinct scenes and candidate seeds, on the same physical NVIDIA H200 and runtime.
The 16-pair held-out quality check has not run. The project remains incomplete.

| Metric | Upstream | Compiled-default |
|---|---:|---:|
| Measured requests | 50 | 50 |
| p50 request latency | 53.76568 s | 40.84276 s |
| p95 request latency | 53.78894 s | 40.93984 s |
| Throughput | 0.01860089 futures/s | 0.02448846 futures/s |
| Peak allocated GPU memory | 52.2083 GiB | 56.2175 GiB |
| Peak reserved GPU memory | 82.6836 GiB | 82.6250 GiB |

Within these two blocks, compiled median latency was **24.04% lower**, throughput
was **31.65% higher**, and peak allocated memory was **7.68% higher**. These are
preliminary observations from one block per role. The remaining six blocks and
between-block variability are incomplete; this is not the planned headline result
or a calibrated performance-regression pass. Per-block summary throughput is
completed futures divided by summed measured request latency; initialization,
compilation, warmups and exports are excluded. It is not interactive video FPS.

Settings were fixed: 25 total frames, one observed and 24 generated, 576×1024,
50 steps, FP16, guidance 2.5, cond_aug 0, decode chunk 14, one serial future,
uncached input processing. OS filesystem caching was not controlled. The default
compiler compiled the denoiser, retained native upstream attention, and reported
no recompilation during measured requests. Torch 2.0.1/CUDA 11.8 and driver 580.178.04
are recorded in the source evidence. Historical H100/A100 timings are not pooled.

## Quality remains a trade-off

The fresh H200 eager strict check passed; compiled strict pixel equality failed.
Exact candidate/action-isolation and cache checks passed in the separate semantic
diagnostic. These checks do not establish held-out video quality. `qualified:false`
and `quality_preserving:false` remain explicit.

The [historical H100 development review](../2026-10-28-h100-performance/README.md)
retains six accepted blind ties and one localized geometry/detail rejection.
After observations were sealed, the preferred display in the rejected case was
identified as compiled. This is **not evidence of compiler-induced degradation**;
the existing conservative qualification remains rejected. Historical development
metrics and that review do not substitute for the unrun 16-pair H200 quality check.

## Recovery and limits

The old allocation ended following monitoring errors; their underlying cause is
unknown. The complete evidence archive was recovered and independently hash
verified. Canonical read-only validation accepted the two completed timing blocks.
The original first-block omitted-default validation mismatch was separately
normalized without rerunning inference; all original failure evidence is retained.
The following interrupted compiler block contains 35 successful rows with no
summary/completion receipt. Those rows are preserved and excluded from this table.
No old measurements may be pooled into a cohort on different physical hardware.

[Structured summary](summary.json) · [Sanitized measured rows](measurements.json) ·
[Published-file checksums](manifest.json)

Source summary and canonical validation hashes are retained in the structured
summary. Raw images/arrays, dataset identifiers, GPU/resource identifiers, account
details, private paths and operational logs are excluded. This snapshot reports
only the recovered H200 cohort and makes no claim about later runs. This revision
adds evidence and documentation; inherited inference/tool source is unchanged.
GitHub publication has not been verified.
