# Input pipeline screen: partial completed snapshot

At 2026-10-27T04:01:46.139609+00:00, 19 of 33 diagnostic jobs were complete: 13 benchmarks and six saved-output checks. This report covers only that snapshot. The conditioning-fresh throughput job was still active.

The six saved-output checks each recorded **exact zero error on the same two clips** (atol=rtol=0). All benchmark numerical/headline flags remain false. The snapshot contains metadata only; the recorded output checks were not rerun locally.

Each completed throughput trial used four requests, one candidate per request and one warmup. Values below are descriptive; there is **no proven speedup** and no p95 or confidence interval.

| Trial | Request median, including queue (s) | Generation median (s) | Queue median (s) | Futures/s from wall time |
|---|---:|---:|---:|---:|
| pinned-async | 57.392 | 57.334 | 0.000 | 0.017424 |
| pinned | 57.477 | 57.413 | 0.000 | 0.017399 |
| pipeline | 114.707 | 57.321 | 57.314 | 0.017432 |
| prefetch | 114.959 | 57.447 | 57.439 | 0.017394 |
| uncached | 57.500 | 57.441 | 0.000 | 0.017391 |
| warm-filesystem | 57.478 | 57.419 | 0.000 | 0.017398 |

Prefetch starts input loading ahead of serial generation. Its roughly 115-second request median includes roughly 57 seconds of queue wait; generation remains about 57 seconds. Pipeline throughput uses four completed futures divided by measured wall time (about 230 seconds), so overlapping queue waits must not be added as serial work.

“Uncached” means a fresh engine context, not a guaranteed cold OS page cache. Warm-filesystem explicitly reads observed files before warmup. The conditioning-fresh correctness screen moves work into context preparation (about 1.36 seconds) and records intra-request cache hits; its two saved requests do not establish a throughput gain.

All 180 copied files and all 45 producer package source files matched their pinned hashes. Source: `d8a32813492c0939a0a91573a945c914938972419946065e6822f6209fa470cb`. Raw arrays, private paths, host identifiers and account data are excluded. [Sanitized per-request measurements and evidence hashes](summary.json).
