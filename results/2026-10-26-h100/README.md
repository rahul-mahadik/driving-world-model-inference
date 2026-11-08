# Evidence index

These are compact, sanitized projections of retained raw artifacts. Each projection records its original source path and SHA-256; its transformed bytes have a separate checksum in `checksums.json`. A source hash is an integrity reference, not a claim that the original raw artifact is included.

| File | Evidence |
|---|---|
| [provenance.json](provenance.json) | Source/checkpoint/runtime identities and one-clip scope |
| [strict-agreement.json](strict-agreement.json) | Corrected exact upstream comparison |
| [isolation.json](isolation.json) | Serial order and unrelated-candidate isolation |
| [conditioning-cache.json](conditioning-cache.json) | Exact output, actual reuse, and key checks |
| [nsys-profile.json](nsys-profile.json) | Corrected kernel families and CUDA activity |
| [torch-memory.json](torch-memory.json) | Separate allocation timeline and autocast-cache frees |
| [compiler-attempt-1.json](compiler-attempt-1.json) | Tracing failure |
| [compiler-attempt-2.json](compiler-attempt-2.json) | Partial compilation and failed exact comparison |
| [timings.jsonl](timings.jsonl) | All three selected-field timing rows |
| [timing-metadata.json](timing-metadata.json) | Two warmups, timing boundary, and failed proof gate |
| [data-readiness.json](data-readiness.json) | Counts and integrity references, without dataset files |
| [trajectory-audit.json](trajectory-audit.json) | Coordinate convention and numerical audit summary |
| [source-origins.json](source-origins.json) | Frozen source versus separately supplied environment definitions |
| [software-verification.json](software-verification.json) | 345 passing CPU software tests on this frozen source |
| [checksums.json](checksums.json) | SHA-256 of every other file in this evidence directory |

Raw tensors, images/videos, full profiler traces, compiled binaries, cloud identifiers, and credentials are excluded. Original records remain unchanged locally. The expanded study is pending; these are initial-session results only.
