# Current-source discovery evidence

These 19 attempts used package `d8a32813492c0939a0a91573a945c914938972419946065e6822f6209fa470cb`. Every copied metadata/log artifact was rehashed against its independent remote SHA256 receipt. Raw generated arrays, profiler files and compiler binaries are excluded. All attempts retain false quality/headline eligibility.

| Attempt | Outcome | Post-warmup observation |
|---|---|---|
| batch-permutation-attempt-1 | failed | No standard warm timing sample |
| batch-permutation-attempt-2 | diagnostic_complete | No standard warm timing sample |
| integrated-compiler-default-attempt-1 | execution_complete_strict_failed | 43.595470 s; exact comparison failed |
| integrated-feature1-attempt-1 | failed | No standard warm timing sample |
| integrated-feature1-attempt-2 | execution_complete_strict_failed | 45.359485 s; exact comparison failed |
| integrated-feature2-attempt-1 | failed | No standard warm timing sample |
| integrated-feature2-attempt-2 | execution_complete_strict_failed | 42.270790 s; exact comparison failed |
| integrated-feature3-attempt-1 | failed | No standard warm timing sample |
| integrated-feature3-attempt-2 | execution_complete_strict_failed | 41.218367 s; exact comparison failed |
| integrated-feature4-attempt-1 | failed | No standard warm timing sample |
| integrated-feature4-attempt-2 | execution_complete_strict_failed | 40.728055 s; exact comparison failed |
| integrated-shape-static-attempt-1 | failed | No standard warm timing sample |
| integrated-shape-static-attempt-2 | execution_complete_strict_failed | No standard warm timing sample |
| memory-batch2-attempt-1 | failed | No standard warm timing sample |
| memory-batch2-attempt-2 | requested_batch_fits | No standard warm timing sample |
| memory-batch4-attempt-1 | failed | No standard warm timing sample |
| memory-batch4-attempt-2 | requested_batch_fits | No standard warm timing sample |
| memory-batch8-attempt-1 | failed | No standard warm timing sample |
| memory-batch8-attempt-2 | requested_batch_oom | No standard warm timing sample |

Compiler observations are one request after two warmups and exclude file decode, model load and export. Graph counts describe partial compilation, with graph breaks retained; they are not whole-model graph capture. The default and shape probes compare against upstream; feature-boundary probes compare against the same-request integrated compiled50 output, whose upstream exact comparison already failed. Feature intervals 2/3/4 change numerical output by design; interval 1 is a boundary control. These rows do not establish acceptable quality or statistical latency gains.

The initial feature/shape/memory/permutation attempts failed checkout verification before generation. The separately named second attempts are preserved independently. At this single request, eager batches 2 and 4 completed without fallback; batch 8 failed with OOM and completed cleanup. This does not establish a hardware maximum or quality-preserving batching. The one-step permutation diagnostic has no sampler rollout or video decode: candidate reversal diverged first at `input_blocks.1`, while a repeated normal ordering was exact. It localizes a numerical difference without identifying its cause or establishing video quality.

- [Sanitized per-attempt metrics, comparisons and provenance](discovery.json)
- [File checksums](checksums.json)
