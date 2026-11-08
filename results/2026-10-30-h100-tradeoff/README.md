# H100 video inference performance and memory; output quality pending

Each request generates one **24-future-frame, 576×1024 video** from a conditioning
image and trajectory with pretrained Vista, fixed 50 sampling steps and unchanged
model weights. The candidate compiles the denoiser with `torch.compile`/Inductor;
the reference uses upstream eager inference. Latency runs from input
loading/preprocessing through host-ready generated frames. Model loading,
compilation, warmups and output export are excluded.

The performance/memory comparison completed **200 measured requests per role**
on one physical NVIDIA H100 80 GB GPU and one runtime. Eight interleaved processes
measured four 50-request blocks for upstream and four for compiled-default. The
same 50 independent scenes and seeds were repeated four times per role. All 400
requests succeeded without OOMs; the independent full verification reconstructed
all eight blocks and the aggregate statistics. Timing finished on October 30, 2026.

| Metric | Upstream | Compiled-default |
|---|---:|---:|
| Measured requests | 200 | 200 |
| p50 request latency | 57.33839 s | 43.66332 s |
| p95 request latency | 57.39194 s | 43.68946 s |
| Throughput | 0.01743859 futures/s | 0.02290418 futures/s |
| Generated-frame throughput | 0.418526 frames/s | 0.549700 frames/s |
| Peak allocated GPU memory | 52.2083 GiB | 56.2175 GiB |
| Peak reserved GPU memory | 68.6211 GiB | 68.5625 GiB |

Compiled median latency was **23.85% lower (1.3132×)**, measured throughput was
**31.34% higher**, and peak allocated memory was **7.68% higher**. The 95% paired
scene-cluster interval for median latency reduction is 23.8457%–23.8580%; the p95
latency reduction is 23.8753%, with interval 23.8595%–23.8887%. These intervals use
4,000 bootstrap resamples, grouping every repeat of each scene. They do not
establish stationarity, deployment latency, or performance on other hardware.

Upstream block medians ranged from 57.32494 to 57.37082 seconds; compiled medians
ranged from 43.64800 to 43.67465 seconds. [All eight block medians](blocks.json)
and [full-precision statistics and intervals](summary.json) remain available.
There was no extra variability pilot or calibrated regression-pass decision.

Settings were fixed at 25 total frames (one observed, 24 generated), 576×1024,
50 steps, FP16, guidance 2.5, cond_aug 0, decode chunk 14, one serial future and
uncached input processing. Each process completed three to five warmups before
measurement. The default compiler targets the denoiser with native attention.
No measured-request recompilation, fallback, OOM, or missing work was admitted.
Torch 2.0.1/CUDA 11.8, xFormers 0.0.20 and driver 580.178.04 were fixed.

Throughput is completed futures divided by summed measured request latency;
initialization, compilation, warmups and synchronous export are excluded.
Generated-frame throughput multiplies by 24 and is not interactive playback FPS.
Uncached means no decoded-input or conditioning reuse; filesystem caching remains
OS-controlled. GPU memory values are CUDA allocator peaks, not total system memory.

## Quality and remaining work

The user authorized studying the compiler as a speed–quality trade-off. This
snapshot completes its **performance and memory comparison only; output quality
is still pending**. It does not establish a joint speed–quality result or an
output-preserving optimization. The fresh eager strict diagnostic passed; compiled strict pixel
equality failed, while exact candidate-order, action-isolation and conditioning
cache checks passed. `qualified:false`, `quality_preserving:false` and
`combined_headline_claim:false` remain unchanged.

The planned **16-pair held-out video-quality comparison has not run**. Historical
[32-scene development metrics and blinded review](../2026-10-28-h100-performance/README.md)
are separate evidence: six ties and one geometry/detail rejection. After sealing,
the preferred display in that rejected case was identified as compiled. That is
not evidence of compiler-induced degradation, and the old conservative rejection
has not been overridden. Seed/action sweeps remain deferred.

The [earlier H200 result](../2026-10-29-h200-descriptive/README.md) remains a separate
50-request-per-role descriptive cohort. None of those timings or partial rows
contributes to this H100 comparison.

## Evidence and publication status

This dated evidence snapshot projects completed statistics from locally hash-verified
timing and independent verification receipts. The **full new-cohort local archive
was independently verified on October 30 at 08:08:58 UTC**: 179 files totaling
4,605,682,030 bytes, including all 14 raw semantic-proof arrays (4,600,637,596 bytes).
Remote inventories matched before and after transfer; every local file was
independently rehashed. The [sanitized backup receipt](backup-verification.json)
records counts and provenance hashes. Backup integrity does not establish quality.

Publication contains compact block/aggregate summaries; individual timing rows,
raw proof arrays, images, account details, physical GPU identifiers and private
operational records are excluded. No missing rows were fabricated. The inference
and inherited tooling source are unchanged.

The project remains incomplete pending the held-out output-quality check and
verified publication. GitHub publication was unverified as of preparation.

[Structured summary](summary.json) · [Block summaries](blocks.json) ·
[Published-file checksums](manifest.json)
