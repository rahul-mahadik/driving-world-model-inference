# Bounded discovery alternatives

These attempts retain their original outcomes; no failed execution was promoted to an optimization result. All used package `2ead184559615d7c9c68620abca687431231ae9ea999a0189a39a0ab7a55206a`.

- **bf16:** Global BF16 failed at unsupported bicubic upsampling.
- **denoiser-bf16:** Denoiser-only BF16 failed at unsupported nearest upsampling; conditioning and decoding remained FP16.
- **decode-chunk6:** Decode chunk size 6 completed all three executions but failed the unchanged strict-zero comparison.
- **decoder-compile:** Decoder compilation stopped after detecting additional eager fallback frames.
- **shape-static:** Static compiler shape policy stopped after detecting additional eager fallback frames.
- **shape-dynamic:** Dynamic compiler shape policy failed a symbolic-shape assertion.
- **registered-einops-dynamic:** Registered-einops dynamic compilation failed a symbolic-shape assertion.
- **registered-einops-reduce-overhead:** Registered-einops reduce-overhead mode stopped after an OOM or retry was detected.

[Numeric evidence and provenance](summary.json). Descriptive timings exclude file decode/export and include failed or instrumented requests; they establish no comparative performance or quality claim. Transfer receipts cover metadata and host logs only.
