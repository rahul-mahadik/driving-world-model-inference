# Compiled development bootstrap before main study planning

These are two independent development qualification runs. No GPU work was
performed while preparing this procedure. Use the verified native CUDA 11.8/sm90
container at `/workspace`, with the active CUDA environment lock and persistent
model caches. The package remains
`d8a32813492c0939a0a91573a945c914938972419946065e6822f6209fa470cb`.

| Config | Output/adoption variant | Input condition | Conditioning cache |
|---|---|---|---:|
|`configs/study_bootstrap_compiled_default.yaml`|`compiled-default`|`uncached`|0|
|`configs/study_bootstrap_compiled_conditioning.yaml`|`compiled-conditioning`|`reused_conditioning`|2 GiB|

Both inherit the unchanged generation from the upstream 32-clip run: 25 frames,
576×1024, FP16, 50 steps, guidance 2.5, conditioning augmentation 0 and decoder
chunk 14. They use the registered compiler runtime, default/static/partial denoiser
compilation, batch 1, candidate 1 and no feature boundary. The identical 32-scene
development manifest is consumed in its existing order, with checksums enabled:
SHA256 `a3c6328d9f0d0b94149820f626db17e024e6fbc0e5523fe5ee19aa3316befbff`.
Each retains all 32 float outputs and has 3 warmups, maximum 5. Neither is a headline
benchmark or inherits a numerical acceptance proof.

The machine-readable recipe is
`configs/study_bootstrap_compiled_hostspec.json`. Before launching each job, the
host wrapper must verify its package, inherited config, manifest and environment
hashes; require a fresh output directory; require at least 18 GiB free and 3630 s
remaining before the existing allocation hard stop. Keep the existing budget
watchdog unchanged. Run one GPU job at a time, including evaluation. A conservative
planning allowance is 30–40 minutes per run, with a 60-minute timeout and 30-second
termination grace. Timing is measured by the run, not inferred from this estimate.
Float arrays alone occupy 5,662,310,400 bytes (~5.27 GiB) per run, plus NPY headers,
compiler caches and metadata. Retain both complete output directories.

For each process, record a distinct host receipt and log as named in the hostspec.
The receipt must contain `status`, UTC `started_at` and `finished_at`, actual
`exit_code`, exact Docker/timeout argv, `source_files` (every package Python file
with SHA256), `source_identity`, `source_unchanged`, and the config/manifest hashes.
Capture source files both before launch and after exit. Only a completed exit 0
receipt with equal source snapshots permits adoption. Preserve failed/terminated
attempts and logs; use a new output and receipt name for a retry. Do not reuse a
partial directory: the benchmark only rejects overwrite after measurements exist.
Also retain hashes of config/environment/model identity, environment/numerical
gates, cold starts, measurements and summary. Keep the Docker image identity and
actual environment variables in the private host receipt.

Inside the already selected container, run these separately. The host wrapper
should apply each hostspec's distinct `TORCHINDUCTOR_CACHE_DIR` and
`TRITON_CACHE_DIR`; retain them so cold compilation remains inspectable. Keep
`DRIVEBENCH_PROFILE` unset and do not enable compiler debug instrumentation.

```bash
timeout --kill-after=30s 3600s \
  outputs/venvs/vista-upstream-cu118-sm90/bin/python -m drivebench.benchmark \
  --config configs/study_bootstrap_compiled_default.yaml \
  --output outputs/study-bootstrap/compiled-default

timeout --kill-after=30s 3600s \
  outputs/venvs/vista-upstream-cu118-sm90/bin/python -m drivebench.benchmark \
  --config configs/study_bootstrap_compiled_conditioning.yaml \
  --output outputs/study-bootstrap/compiled-conditioning
```

Before qualification, inspect exit status, 32 successful requests, 32 saved arrays,
no OOM/failure, actual batch 1, a ready cold-start record, stable compiler counters,
and successful runtime restoration per request. Any measured recompile remains
visible and prevents interpreting the quality run as stable timing. Absence of a
numerical proof is expected here; headline eligibility remains false. Retain the
original strict-zero failure evidence independently.

Run LPIPS only after generation exits. Use the isolated verified package target;
do not install into or alter the inference environment. The environment below
causes the evaluator to verify the setup receipt, package files, AlexNet/learned
weights, actual loaded model-state SHA and module location before reporting scores.
The declared review subset is four fixed scene/candidate/seed units per comparison,
selected by seed 2026 independently of metric values. Both commands therefore use
the same units, with separately blinded A/B identities. Every new future frame
appears in its 10 fps GIF; the contact sheets sample frames.

```bash
env PYTHONPATH=/workspace/outputs/evaluation-packages/lpips:/workspace \
  TORCH_HOME=/workspace/outputs/evaluation-packages/torch-home \
  DRIVEBENCH_EVALUATION_SETUP_RECEIPT=/workspace/outputs/evaluation-packages/setup-receipt.json \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  timeout --kill-after=30s 1800s \
  outputs/venvs/vista-upstream-cu118-sm90/bin/python -m drivebench.evaluate \
  --runs outputs/study-bootstrap/compiled-default \
  --reference outputs/study-bootstrap/upstream \
  --tolerances configs/study_bootstrap_unfrozen_tolerances.yaml \
  --lpips --lpips-device cuda --review-limit 4 \
  --output outputs/study-bootstrap/qualification-v1/compiled-default-vs-upstream.json

env PYTHONPATH=/workspace/outputs/evaluation-packages/lpips:/workspace \
  TORCH_HOME=/workspace/outputs/evaluation-packages/torch-home \
  DRIVEBENCH_EVALUATION_SETUP_RECEIPT=/workspace/outputs/evaluation-packages/setup-receipt.json \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  timeout --kill-after=30s 1800s \
  outputs/venvs/vista-upstream-cu118-sm90/bin/python -m drivebench.evaluate \
  --runs outputs/study-bootstrap/compiled-conditioning \
  --reference outputs/study-bootstrap/upstream \
  --tolerances configs/study_bootstrap_unfrozen_tolerances.yaml \
  --lpips --lpips-device cuda --review-limit 4 \
  --output outputs/study-bootstrap/qualification-v1/compiled-conditioning-vs-upstream.json
```

Use a fresh `qualification-v1` directory. Preserve each evaluator log, exit code,
source/config hashes, setup-receipt hash, metric JSON and all review artifacts.
The two review indexes are `compiled-default-vs-upstream-review/index.html` and
`compiled-conditioning-vs-upstream-review/index.html` under that directory.
Review them without opening their private keys. Save ratings in separate CSVs,
explicitly labeled agent-assisted; record disappearance, duplication, freezing,
geometry and motion differences. Only then inspect the answer keys and join votes
to variants. Complete 32-clip numerical/LPIPS/temporal and recorded-ground-truth
diagnostics remain authoritative for coverage; four reviewed cases alone cannot
establish all-scene acceptability. No trajectory-adherence or driving-safety claim
follows from these image diagnostics.

Qualification is descriptive with unfrozen tolerances. Select and document explicit
development-only numerical/perceptual/temporal limits and review rationale; do not
rewrite the strict-zero diagnostic or select limits from held-out data. If compiler
acceptance fails, keep it on the development curve and choose the exact cache
reference. If accepted, choose the compiled-conditioning secondary reference before
materializing the main study. The final study policy must still bind its complete
development aggregate and accepted-review evidence separately.

The eventual main specification must reproduce these variant settings exactly.
With shard size 32 and candidate count 1, adoption requires no generation replay:

```bash
python -m drivebench.study quality --study outputs/full-study-v1 --phase development \
  --adopt upstream=/workspace/outputs/study-bootstrap/upstream \
  --adopt-source upstream=/workspace/outputs/host/study-upstream-development-attempt-1.json \
  --adopt compiled-default=/workspace/outputs/study-bootstrap/compiled-default \
  --adopt-source compiled-default=/workspace/outputs/host/study-bootstrap-compiled-default-attempt-1.json \
  --adopt compiled-conditioning=/workspace/outputs/study-bootstrap/compiled-conditioning \
  --adopt-source compiled-conditioning=/workspace/outputs/host/study-bootstrap-compiled-conditioning-attempt-1.json
```

The main study recomputes its paired metric receipts and never deletes these
externally adopted arrays. Variant names may be remapped only by explicitly using
the corresponding frozen plan names in the adoption arguments; runtime selectors,
generation, inputs, candidate count and retention settings must still match.
