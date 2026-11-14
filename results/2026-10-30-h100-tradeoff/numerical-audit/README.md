# Reproduce the historical H100 numbers from individual observations

This companion was added after the October 30 experiment. It publishes a sanitized
projection of **all 400 actual measured requests**: 200 upstream and 200
compiled-default. No request was rerun, synthesized, selected by its outcome or
replaced by an aggregate. The original report, summary and manifest in the parent
directory remain unchanged, including their dated publication-status statements.

## Run the analysis

From the repository root, Python 3.10+ is sufficient. No package installation,
NumPy, model weights, dataset, API key or GPU is needed:

```sh
python tools/analyze_h100_timing.py
```

The script verifies file hashes, validates the entire paired observation matrix,
recomputes latency quantiles, block medians, throughput, memory peaks and both
scene-cluster confidence intervals, then compares them against every corresponding
field in the [original summary](../summary.json). It exits with an error on a
mismatch. Floating-point agreement uses relative and absolute tolerances of
`1e-13`; counts, identities and flags require exact agreement. To retain the report:

```sh
python tools/analyze_h100_timing.py --output h100-reconstructed.json
```

The output path must be new. The analysis can also be run from another directory
using the script's absolute path; its default input is resolved relative to itself.

## Downloadable inputs

- [measurements.jsonl](measurements.jsonl): all 400 per-request observations, in
  their original block and request order. Numeric measurements retain their
  original JSON precision.
- [bootstrap-scene-indices.csv](bootstrap-scene-indices.csv): 4,000 rows of 50
  zero-based scene indices. These are **derived resampling instructions**, not
  additional GPU observations. They recreate the original seed-0 PCG64 schedule
  and allow a dependency-free reconstruction of the published intervals.
- [provenance.json](provenance.json): source-file digests, per-block measured
  durations, field mapping, redaction rules and the exact bootstrap recipe.
- [manifest.json](manifest.json): byte sizes and SHA-256 identities of those inputs
  and the original summary identity.

The CSV has no header; each line is one bootstrap draw with replacement. To
independently regenerate it with NumPy, use `rng = numpy.random.default_rng(0)`
and call `rng.integers(0, 50, 50)` 4,000 times. Reset the seed before the p95
analysis: the original p50 and p95 procedures used the same schedule. The exported
schedule reproduced both historical intervals; provenance records the original
and export NumPy versions.

## Observation schema and grouping

| Field | Meaning |
| --- | --- |
| `block_index` | Original chronological process block, 0–7: U,C,C,U,C,U,U,C |
| `role` | `upstream` or `compiled-default` |
| `role_block_index` | That role's repeated block, 0–3 |
| `request_index` | Original measured position within the block, 0–49 |
| `scene_index` | Position in sorted original scene IDs, 0–49; stable across both roles and all repeats |
| `repeat` | Original within-block repeat field, zero for every row; cross-block repeats are identified by `role_block_index` |
| `seed` | Original generation seed, unchanged |
| `latency_s` | Input loading/preprocessing through host-ready frames; excludes initialization, compilation, warmups and export |
| `peak_allocated_bytes`, `peak_reserved_bytes` | Original CUDA allocator peaks, not total GPU/system memory |
| `denoiser_evaluations` | Completed denoiser-call count, 50 for every admitted row |
| `status` | Original request status, `ok` for every admitted row |
| `failure_count`, `fallback_count`, `oom_attempt_count` | Counts from the original row's failure, fallback and OOM records; all zero |
| `source_line_sha256` | Original JSONL line digest before redaction, excluding the newline |

There are **50 independent scene groups**, each measured four times for each
role. A bootstrap draw selects 50 scene groups with replacement and carries all
four repetitions of each selected scene in both roles. Each statistic is the
candidate quantile divided by the reference quantile, minus one. The interval is
the 2.5th and 97.5th percentiles across 4,000 such draws. It is not a bootstrap of
400 independent requests, and it is not the median of per-request speed ratios.
Throughput uses summed measured request duration, aggregated within each block
then across the four blocks, matching the original computation.

## What this makes auditable—and what it does not

Readers can now independently recompute the published statistics from individual
numerical observations. Source-file and source-line hashes link this projection
to retained original records; without those original files, the hashes alone do
not independently prove the projection or the physical experiment. The local
export checked every row and every scene/request/action/seed pairing against the
original completed-result artifact hashes.

Scene/request identifiers, dataset paths, images, generated arrays, provider
identities and credentials are not redistributed. This companion does not assess
model outputs or supply new quality evidence. **`qualified:false`,
`quality_preserving:false` and `combined_headline_claim:false` remain unchanged.**
The planned historical 16-pair held-out quality evaluation remains unrun. The
intervals describe this one hardware/runtime cohort and do not establish temporal
stationarity, generalization to other GPUs or the quality of the newer agent task.

[Original historical report](../README.md) ·
[Analysis source](../../../tools/analyze_h100_timing.py)
