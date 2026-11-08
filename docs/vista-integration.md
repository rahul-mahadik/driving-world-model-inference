# Pinned Vista adapter

`drivebench.models.vista.VistaBackend` uses the original Vista model, conditioner,
Euler EDM sampler, guidance implementation, and overlapping temporal decoder.
`drivebench.models.upstream.UpstreamBackend` calls the original `do_sample`
function. The vendor checkout is never edited by this integration.

This implementation has CPU contract tests. **A real checkpoint and CUDA GPU
are required to establish upstream output agreement, output quality, compiler
correctness, memory limits, or speed improvements. None is claimed here.**

## Required artifacts and integrity

Use the pristine OpenDriveLab/Vista revision
`cc9821b4253ca7987c32757613d2fc2448fa9f5d`. Initialization checks both Git HEAD and
the complete working-tree status and refuses changes. The exact upstream
`configs/inference/vista.yaml` is required; its SHA-256 is recorded.

The published `vista.safetensors` at Hugging Face revision
`a92d28c7b3da931ec959d587153dc45fffe570ef` has LFS SHA-256
`1b267bb753aa5c5e4d363f6e4c864dd4e343996dedc18b5ab20af6d20be34b98`.
The configured checksum must be present and is compared against every byte
before loading. The original loader is retained; its permissive state-key
behavior is followed by a strict check of checkpoint versus model keys.
An incompatible checkpoint fails rather than leaving random initialized weights.

The original model constructor initializes OpenCLIP and may fetch its pretrained
initialization artifact before loading the Vista state. Pre-stage and record
that artifact in the GPU environment. It is not bundled in this repository.
The pinned checkpoint must match the complete model state afterward.

Use `model.backend: upstream` for the untouched reference and `vista` for the
refactor. Common fields are `upstream_dir`, `upstream_commit`, `checkpoint`,
`checkpoint_sha256`, and `inference_config`. Install the preserved upstream
environment first; use a separate environment for compiler compatibility work.

## Inference input contract

Observed input is float32 `[observed_T,3,H,W]` normalized to `[-1,1]`. Its length
must equal `generation.conditioning_frames`. The adapter supports a 25-frame,
single-round clip. Resolution is explicit and must meet the model's own spatial
shape constraints. The standard Vista resolution is 576 × 1024 pixels.

Actions are four coordinate pairs, represented as `[4,2]` or flat `[8]`.
The maintainer's external translation script was verified against 32 real
annotation clips: future LiDAR sensor origins are expressed in the initial
`LIDAR_TOP` frame, in meters, with lateral-right then longitudinal-forward axes.
The data adapter derives actual future keyframe times relative to the first
camera frame, preserving the distinct LiDAR spatial-origin timestamp. The
numeric audit is recorded in `results/2026-10-26-h100/trajectory-audit.json`.
The model adapter flattens pairs
without resampling or changing units. Dataset preparation owns timestamp and
coordinate validation. Outputs include the observed prefix and have shape
`[candidates,25,H,W,3]`, float32 in `[0,1]`, in host memory.

Only observed images enter inference. Since upstream `do_sample` encodes all
25 input frames, missing frames are padded with the last observed frame. Both
baseline and refactor receive this identical padded array. Recorded future
images must remain exclusively in the evaluator. This is an inference-safe
input adaptation to the original function, not a reproduction of the demo's
future-image file-loading procedure.

## Randomness and exact conditioning reuse

Every candidate has an isolated CPU/CUDA RNG scope. Candidate ordering and
executor batch size cannot change its random stream. Calls are process-locked
because upstream code uses global RNG and a module-level residency setting.

The adapter intentionally retains these upstream details:

1. Noise augmentation draws `randn_like(cond_img)` even when `cond_aug=0`.
2. Original `get_condition` creates conditional and unconditional embeddings.
3. The first-stage VAE draws its posterior noise on the **CPU** and transfers it
   to CUDA. Its encoder chunk order is 14 then 11 frames.
4. Diffusion initialization draws CUDA noise with `randn_like(z)` afterward.
5. The sampled latent is copied into float32 output storage before decoding.

With `conditioning_cache_bytes>0`, preparation may retain deterministic image
embeddings and deterministic VAE posterior parameters. It does not retain
sampled latents. Each candidate invokes the original regularizer on cached
parameters using the original chunk shapes and random draw order. No action
embeddings are cached. The noisy image encoder is reused only for zero
augmentation; the clean CLIP image encoder is deterministic at the pinned
configuration. The cache is built eagerly, with an explicit byte budget, so the
executor can account for its retained tensors. Preparation cost belongs in
uncached end-to-end measurements.

These choices are designed to preserve upstream computation; CPU tests only
verify the orchestration contract. Verify actual eager/cached GPU outputs
against the upstream backend before marking reuse exact in a result.

The opt-in validation command runs upstream/refactor agreement, candidate
reordering and unrelated-action isolation, cached/uncached output agreement,
and observed-content cache invalidation. Models run sequentially, with garbage
collection and CUDA allocator cleanup between backends. It saves raw output
arrays and numerical error reports. Supply tolerances frozen using development
data; no numerical tolerance is silently chosen for the user:

```bash
python -m drivebench.validate --config configs/baseline.yaml --mode all \
  --atol "$DEVELOPMENT_ATOL" --rtol "$DEVELOPMENT_RTOL" \
  --candidates 2 --output outputs/validation/numerical.json
```

This is a single-request numerical check. Repeat representative development
and held-out requests before making a broader agreement claim. It does not
substitute for perceptual or action-response evaluation.

Schema 2 reports bind the checks to the exact generation and optimization
settings, source-file fingerprint, checkpoint identity, runtime and GPU identity,
manifest and metadata checksums, verified manifest scene list, selected record,
observed image bytes, preprocessed tensors, actions and seeds. Saved output
archives have checksums too. The report records `validated_request_count: 1`
and retains this narrow scope even if every pixel comparison passes. Input
provenance never opens future evaluation images. Use the complete development
manifest and `--request-index` for evidence intended to accompany a benchmark;
recorded partition roles and scene IDs let the benchmark distinguish development
evidence from a disjoint held-out workload.

Cache agreement also requires evidence that both pinned image encoders and the
posterior-parameter path were actually reused. Equal outputs from an empty or
undersized cache cannot pass that check. By default the cache check uses the
configuration's nonzero cache budget, or 2 GiB when caching is disabled in the
configuration. An explicit `--cache-bytes` overrides this experiment budget and
is recorded in the cached run's effective settings. The context-key check can
fail when the full prepared context cannot fit in that budget.

## Independently selectable experiments

| Setting | Implemented behavior and qualification |
| --- | --- |
| `compile_denoiser` | Compiles the denoiser callable including model invocation; mode, dynamic shapes, and fullgraph are explicit. Compilation failures propagate. |
| `compile_decoder` | Compiles the decoder's `forward`, preserving its `VideoDecoder` type and the original temporal chunk wrapper. |
| `decode_chunk_size` | Changes decoding only; encoding retains the upstream chunk size and RNG sequence. Temporal boundaries can alter output quality. Sizes 4/5 are rejected because the upstream 3-frame overlap implementation cannot retain its required context there; use 1–3 or ≥6. |
| `dtype: fp16/bf16` | Changes outer autocast; upstream disables autocast in sensitive first-stage paths. BF16 is a precision experiment. Upstream already uses FP16 autocast. |
| `feature_cache_interval` | Values >1 reuse the selected middle-block feature on intermediate denoiser evaluations. Requires an explicit module. |
| `feature_cache_module` | Supports `model.diffusion_model.middle_block` (or `middle_block`) only. Select after profiling its contribution. |
| `pinned_memory` | Uses pinned host storage for H2D transfer. |
| `async_transfer` | Uses two pinned staging slots and a dedicated copy stream, with explicit event waits. Context storage is cloned before a slot can be reused. |

Approximate feature reuse is scoped to one candidate and the ordered
`[unconditional_T, conditional_T]` CFG batch. CFG features keep distinct slots;
the entire cache is destroyed on completion or failure. Returned features are
cloned to avoid later mutation of the cache. The cache saves just the selected
block's work; the remainder of the denoiser still runs. Actual denoiser calls
therefore do not decrease. Cache hit/miss counts are reported separately.

Feature reuse combined with denoiser compilation is rejected until graph
breaks and graph replay semantics are audited. Different attention backends,
CPU execution, FP32, offload, other feature modules, and multiple rollout
rounds are explicitly unsupported. Quantization and custom kernels are
separate optional experiments, not hidden settings in this adapter.

The pinned spatial attention already uses xFormers. Keeping it is baseline
behavior, not a new optimization. Model identity records parameter counts and
bytes by actual storage dtype separately from the selected autocast dtype.
When `DRIVEBENCH_PROFILE=1`, CUDA NVTX and PyTorch record-function ranges label
the stages; ordinary timing runs do not enable those extra profiling ranges.

## Candidate execution, timing, and limitations

The backend advertises `max_batch_size=1`. Requests for 4/8/16 candidates are
supported through serial candidate execution, with the fallback recorded.
True multi-candidate temporal batching is not enabled until GPU isolation tests
validate every temporal reshape and attention layout. Candidate count and
effective batch size must be reported separately.

Stage durations use CUDA events. The refactor reports conditioning, encoding,
denoising, decoding, and transfer durations. The unmodified reference reports
one aggregate sampling interval. Whole GPU inference time is an enclosing
interval; it must not be added to its component stages. Host-ready end-to-end
wall time is measured by the executor/benchmark. Compilation and initialization
can occur on the first invocation and must be excluded only after recorded
warmup. The buffer implementation ensures ownership and stream correctness;
the synchronous request API does not imply demonstrated transfer/compute
overlap or a measured speedup.

Before publishing GPU results, run the same observed inputs, trajectories,
seeds, steps, resolution, and decoder settings through both baselines. Compare
outputs and latents where available, test candidate permutation/independence,
freeze development-set numerical and quality tolerances, then repeat the
held-out workload. Check decoder chunking, BF16, compilation, cache reuse, and
feature reuse separately and in supported combinations. Reduced steps and
feature reuse need quality evaluation, not only numerical comparisons.
