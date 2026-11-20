# Modern Vista eager execution on A100 — November 18, 2026

**One full-workload Vista request completed on a fresh A100-SXM4-80GB using the modern PyTorch/native-SDPA adapter.** This is an execution-feasibility result, not a comparative benchmark or quality result. The workload was unchanged: one conditioning frame, 24 future frames at 576×1024, 50 denoising steps, one uncached future, and decode chunk 14.

| Observation | Recorded value | Scope |
|---|---:|---|
| Successful measured requests | 1 | One scene, one fresh process |
| Warmup requests | 3 | Same scene; excluded from the measured sample |
| Model initialization | 217.087596 s | Engine construction and synchronization after environment collection; excludes package/asset acquisition |
| Warmup latency | 99.129364, 98.183244, 98.187432 s | Not independent repetitions |
| Measured request latency | 97.994715 s | Input loading/preprocessing through host-ready frames; export excluded |
| Measured peak allocated / reserved | 46.011828 / 53.349609 GiB | PyTorch allocator; not total device memory |
| Maximum warmup peak reserved | 71.605469 GiB | First warmup; do not size hardware from the warm request alone |
| Successful request failures / OOM / fallbacks | 0 / 0 / 0 | This A100 attempt only |
| Output | Float32, 25×576×1024×3 | Conditioning frame plus 24 generated future frames |

The measured run used PyTorch 2.14.1+cu130, CUDA 13.0, Python 3.12.3, driver 595.112.07, native SDPA, FP16 UNet weights with FP32 normalization, and channels-last 2D convolutions. The VAE and conditioner retained FP32 parameters. Compilation was disabled. The GPU reported 81,920 MiB, MIG disabled; effective host RAM was 249,999,998,976 bytes. These describe the machine that ran the workload, not minimum requirements or a guarantee that another configuration fits.

The CUDA execution proof and positive Nsight Systems smoke-kernel activity trace passed. Hardware performance counters were **not requested**. Their absence does not invalidate wall-clock latency or allocator memory measurement, but this result supplies no occupancy, SM utilization or memory-bandwidth analysis. A full-model operator profile was not collected. The recorded stage timers are overlapping diagnostic scopes; do not add them into a stacked runtime attribution chart.

## Evidence and reconstruction

- [summary.json](summary.json): scope, workload, exact numerical values and explicit claim limits.
- [measurements.jsonl](measurements.jsonl): the single original numerical request row, with absolute asset/output paths removed.
- [cold-start.json](cold-start.json): initialization and all three warmup rows, separately labeled.
- [runtime.json](runtime.json): runtime versions, public source hashes and model identity, with private machine identifiers removed.
- [failed-attempts.json](failed-attempts.json): earlier setup/provider failures and the A40 first-warmup OOM.
- [source-map.json](source-map.json): field-extraction scope and original artifact/archive SHA-256 commitments.

Run the dependency-free arithmetic checker from this directory:

```sh
python verify_summary.py
```

It verifies consistency between the published rows, warmups and summary. It does **not** rerun Vista, verify the withheld raw array, independently validate the original machine, or assess quality. Full-precision values are retained in JSON; this table rounds only for display.

The original archive contains 245 files / 182,517,149 bytes. Its before/after inventories matched, and every archive member and local file hash was checked. A separate local review reconstructed the strict `read_restart` completion record and verified source/runtime/input bindings, all 50 denoiser evaluations per successful request, process exit and cleanup. That review is not replication by an external laboratory. The retained 176,947,328-byte raw output is finite, with range [0.07360514998435974, 1.0], and has SHA-256 `59ed9a9b1b45356f0e63091c2f157c2dc76f2c058f50537681bea7a34c7ab649`.

Raw arrays, input images, weights and preview media are not redistributed here. Their hash commitments do not make the withheld bytes publicly auditable. Obtain the assets under their own licenses and follow the [modern reproduction recipe](../../docs/reproduce-release.md#modern-vista-eager-execution) to generate new evidence. The original source/model/input identities are supplied for comparison; a new run must record its own GPU/runtime identity and output hashes.

## Failed cheaper-device attempt

The preceding A40 attempt passed setup, actual CUDA/runtime checks and asset verification, then ran out of GPU memory during the first warmup. It completed zero measured requests and zero futures. Its failed warmup recorded 2.974221 seconds after 51.157797 seconds of model initialization; neither is a completed-request latency. The recorded zero denoiser evaluations is an **incomplete counter**, updated only on successful return, so it does not locate the failing operation. The workload was not reduced after that failure. These separate cohorts establish an A40 failure and an A100 success for these exact attempts, not a cross-GPU speed ratio or a general 48 GB memory limit.

Earlier L40S setup failures (missing Git; a changed Git index detected during verification, with its exact origin not established) are retained. The reproduction recipe includes Git and `GIT_OPTIONAL_LOCKS=0`. Two provider HTTP 400 rejections created no GPU instance; their cause remains unknown.

## What remains unmeasured

There is no modern compiled result, repeated-process noise estimate, tail-latency estimate, compile cost/break-even, quality calibration, distributional/trajectory evaluation, live agent episode, policy comparison, or hardware-transfer finding in this cohort. The legacy H100 timing cohort is separate. The software's historical false qualification flags remain unchanged. Successful execution and finite pixels do not establish perceptual quality, action adherence or quality preservation.
