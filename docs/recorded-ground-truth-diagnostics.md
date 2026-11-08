# Recorded-future diagnostics for the upstream reference

The frozen evaluator records generated-candidate versus recorded-future proxies.
Use `scripts/export_recorded_ground_truth.py` to obtain the corresponding
upstream/reference proxies without another inference or LPIPS run. It calls the
same `_ground_truth`, `as_video` and `compare_videos` routines: center crop,
Lanczos resize, RGB normalization, and removal of conditioning frames match the
existing evaluator exactly. One clip is loaded at a time.

In a separate CPU process, before any relevant raw array is retired:

```bash
timeout --kill-after=30s 900s \
  outputs/venvs/vista-upstream-cu118-sm90/bin/python scripts/export_recorded_ground_truth.py \
  --run /workspace/outputs/study-bootstrap/upstream \
  --expected-manifest /workspace/data/manifests/development.jsonl \
  --output /workspace/outputs/study-bootstrap/qualification-v1/upstream-recorded-ground-truth-v1.json
```

The CLI hides CUDA devices and limits numeric-library threads to four. Importing
the helper does not change the caller's environment. The integration API is
`export_run(run_path, output, expected_manifest=None)`; its keyword-only manifest
argument is optional. The core package stays unchanged.

New exports require complete real manifest request coverage, matching candidate
identity/seed, exact future-image order/checksums, and all recorded-candidate raw
arrays. Whole development exports require 32 clips. The helper accepts the exact
frozen study's held-out, second-seed, action-subset and quality-shard roles. An
action run contributes only its recorded candidate; alternative actions are
explicitly excluded because recorded futures do not provide their ground truth.

The JSON stores per-pair metrics and scene-bootstrap aggregates. The separate
`.receipt.json` is written exclusively after the metrics are fsynced, and binds
the JSON, run artifacts, manifest, image hashes, source arrays, evaluator package
and exporter. Neither output contains a quality/performance pass or a safety,
feasibility, optical-flow or trajectory-adherence claim. Paths are private.

`verify_export(output, allow_retired_arrays=True)` verifies the committed metric
hash and every remaining source binding. After verified retirement, absent raw
arrays are listed explicitly in the return value; they cannot support a new
metric or recovery claim. Existing arrays are still rehashed. A missing/partial
receipt or changed metric, metadata, image, helper or available array fails closed.
Calling `export_run` again uses this verification path and does not overwrite the
original metrics. This allows a pre-retirement study wrapper to bind the exporter
and receipt SHA once per upstream shard while retaining its completed diagnostics.
