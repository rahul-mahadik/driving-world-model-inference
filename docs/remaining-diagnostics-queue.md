# Bounded input and serial-candidate diagnostic queue

The local queue is `outputs/execution-screen-remaining-v1/remaining-queue.json`.
It was prepared from the existing `prepare_execution_screen.py` helper and binds
the unchanged package `d8a32813492c0939a0a91573a945c914938972419946065e6822f6209fa470cb`,
the helper sources, generated configs/manifests, and active environment proof/lock.
Its status is `prepared_only_no_execution`. Preparing it launched no remote or GPU
work and did not materialize or revise the full-study draft.

The queue contains 33 ordered entries:

- 18 input benchmarks: two saved correctness requests and four export-free
  throughput requests for each of nine input/pipeline conditions, one warmup each.
- Seven strict saved-output comparisons against the matching uncached baseline.
- Two complete current-source strict numerical gates: eager and conditioning reuse
  with serial candidate execution, including candidate isolation and cache checks.
- Six complete candidate-set benchmarks: 4, 8 and 16 candidates for eager and
  conditioning reuse, two measured requests plus one warmup, one saved request.

All actual candidate batch sizes must be 1. Batch-2, larger batch, memory-capacity,
compiler and feature-reuse commands in the helper's broader plan are **not** in
this queue. The generated parent plan is not an automatic execution list.

The input variants are uncached, warm filesystem, pinned host memory, pinned plus
async transfer, background prefetch, their combined pipeline, fresh conditioning
preparation, repeated-scene uncached, and repeated-scene retained conditioning.
Four-scene input throughput runs measure continuous unrelated requests; repeated
conditioning runs deliberately use one scene. The pool has two slots; async and
double buffering are coupled in this implementation. Staging synchronizes before
inference, so CPU prefetch can overlap the next GPU generation but GPU
transfer/compute overlap must not be claimed without a trace. Checksums warm OS
pages, so uncached processing is not a cold-filesystem benchmark.

The remote controller should copy the prepared directory and two unchanged helper
scripts, then execute each queue entry's literal `argv` in the already verified
container at `/workspace`. Use the frozen Python environment, model caches and
CUDA environment. Before every GPU job:

1. Verify the queue and parent-plan hashes, every selected config, relevant
   manifests, source files and environment artifacts.
2. Require fresh output/proof paths, at least 16 GiB free, and the entry's declared
   remaining-allocation window. Keep the current budget/deletion guard unchanged.
3. Verify every dependency from its completed receipt and actual artifact. A
   passed proof requires exit 0 and JSON `passed: true`; a benchmark requires all
   expected requests successful, expected output retention and no OOM/retry.
4. Launch one GPU job at a time. Save its exact argv, UTC start/finish, exit code,
   config/environment hashes and before/after package source snapshots under the
   queue's unique host receipt/log paths. Preserve partial output on failure.

Timeouts are 600 s for saved input correctness, 900 s for input throughput,
180 s for CPU array comparison, and 1800 s for each full numerical gate.
Candidate-set limits are 1200/2100/3600 s for 4/8/16 candidates, respectively,
with 30 s termination grace. GPU entries require the timeout plus 60 s before
the existing allocation deadline. Failed dependencies stop their descendants;
other independent branches may proceed only after the failure is recorded.
Retries require new versioned output/proof paths and a newly bound queue receipt.
No old array, report or measurement file may be overwritten.

Each successful candidate request must report `50 × candidate_count` denoiser
calls and an actual-batch list of ones. Preserve all request timings, stage times,
GPU-resident timing, allocated/reserved peaks, cold start, cache hits, failures and
complete output hashes. Two candidate-set measurements do not establish p95 or a
headline speedup. Four input requests provide only exploratory throughput.
Do not use cache hit counts without comparing the corresponding outputs.

The conservative planning allowance is approximately 5.4 GPU-hours including
small full-gate validation and loading, assuming 57.3 s per serial future. It is
not an ETA. Input/scaling saved arrays can total about 12.2 GiB; the two numerical
gates can add roughly another 4.6 GiB. Preserve complete diagnostic directories
in a checksum-verified local archive before retiring remote arrays. The queue
does not delete anything. This retention is separate from the external bootstrap
arrays required for later full-study adoption.

No reduced-step controls were changed while preparing this queue. Additional
controls will be proposed only after the measured feature intervals are available.
A one-clip screen may motivate a predeclared control but cannot satisfy the
all-32-scene runtime-equivalence criterion.
