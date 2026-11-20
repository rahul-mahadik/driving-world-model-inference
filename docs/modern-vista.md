# Modern Vista baseline: implementation and validation boundary

This is a new runtime candidate for the study of **verifiable rewards for lossy
inference optimization of video world models**. It does not relabel the historical
H100 measurements or establish that output quality is preserved. A full checkpoint and CUDA execution were validated for one A100 eager request
after three warmups on November 18, 2026; the [released evidence](../results/2026-11-18-a100-modern-eager-feasibility/README.md)
records the exact runtime and workload. Perceptual quality, comparative performance,
compiled execution and general hardware capacity remain unvalidated.
The previously proposed one-episode, eight-configuration, three-policy study is
historical; its completion is not a prerequisite or claimed result of this work.

## What the port actually changes

`drivebench_tuning.modern_vista.ModernVistaBackend` reuses the existing input,
conditioning, sampler, seed and output path. It requires the pristine Vista
`cc9821b4253ca7987c32757613d2fc2448fa9f5d` checkout and independently hashed
`.safetensors` checkpoint. No frozen core file or vendor source is edited.

The explicit constructor scope replaces both spatial and temporal attention
factories with `NativeSDPACrossAttention`. It preserves query/key/value/output
parameters, all trained action-control projections, optional LoRA projections,
additional-token removal and ordered cross-frame repetition. Merely changing
Vista's setting to `softmax` is incorrect: its old fallback lacks those action
projections. The replacement calls PyTorch native SDPA, with zero attention
dropout and the original default scale. Arbitrary masks remain unsupported.
Large temporal batches retain a bounded head-batch size. Native dispatch may
choose different kernels; **Flash Attention is not assumed**. See the
[SDPA API](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html).

The UNet's weights become FP16, except GroupNorm weights, which retain their
original FP32 values. Its 2D convolution weights and flattened-frame input use
channels-last strides. Logical dimensions remain NCHW; temporal Conv3d layout
is unchanged. The original FP32 video autoencoder and conditioner stay FP32;
globally calling `.half()` would violate their disabled-autocast paths.
Unsupported layout propagation may still introduce conversions, so the profiler
must measure them. See PyTorch's
[Conv2d conversion API](https://docs.pytorch.org/docs/2.14/generated/torch.nn.utils.convert_conv2d_weight_memory_format.html).

The modern eager and modern compiled roles share that same port. Their sole
difference is `compile_denoiser`; the compiled role wraps the denoiser with
Inductor, default mode, static shapes and partial-graph capture. Compilation is
lazy. A successful constructor is not evidence of compiled execution or reduced
latency. See [`torch.compile`](https://docs.pytorch.org/docs/2.14/generated/torch.compile.html)
and the [restart measurement protocol](modern-baseline-measurement.md).

## Obtainable environment, not a historical private image

The prospective CUDA target is **PyTorch 2.14.1 + CUDA 13.0**, torchvision 0.29.1,
Python 3.12 or 3.13, Linux x86_64 and an Ampere-or-newer GPU. This is a deliberately
pinned modern release, not a claim to always use the latest release. The matching
CUDA 13.0 wheel is listed by the [official installer archive](https://pytorch.org/get-started/previous-versions/).
NVIDIA documents a driver 580 minimum for CUDA 13.x minor compatibility, with
additional PTX/feature restrictions; actual driver, compiled-kernel and runtime
preflight remain required. See [NVIDIA's compatibility rules](https://docs.nvidia.com/deploy/cuda-compatibility/latest/minor-version-compatibility.html).

```bash
python3.12 -m venv .venv-modern
. .venv-modern/bin/activate
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r environments/modern-vista-cu130.in
python -m pip install . ./extensions/kernel_tuning
python -m pip check
python -m pip freeze > modern-resolved-environment.txt
python -m drivebench_tuning.modern_vista
```

Do not install the legacy `[gpu]` extra, upstream's original requirements file,
xFormers, old Triton pins or the historical compiler adapter in this environment.
The short recipe pins direct dependencies. For the tested Linux CPython 3.12
environment, use the [exact wheel/hash lock](../environments/modern-vista-cu130-linux.lock.txt),
[wheel provenance](../environments/modern-vista-cu130-linux.wheels.json) and
[modern reproduction instructions](reproduce-release.md#modern-vista-eager-execution).
Fresh CUDA/runtime evidence is required on every new host. Auxiliary OpenCLIP/timm versions
retain Vista's model architecture while PyTorch and its attention/compiler path
are modernized. The runtime rejects mismatched reviewed dependency versions.

Acquire the [Vista assets and dataset under their licenses](vista-integration.md)
before starting a timed worker. The original OpenCLIP constructor additionally
requires the cached `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` asset. Record its exact
revision and file hashes; neither it nor Vista weights/data are bundled here.
Install Git before any checkout verification and prevent Git read-only probes from
refreshing the pinned checkout index. The released wrapper
`bash scripts/with_modern_vista_env.sh COMMAND [ARGS...]` sets the following
variables **before the child imports Hugging Face or inspects the checkout**:

```bash
export GIT_OPTIONAL_LOCKS=0
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
```

Missing cached assets must fail, not download during a measured startup.
This follows the [Hub's import-time offline configuration](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables).
The implementation needs no benchmark owner's provider account or API key.
After installing the candidate on a CUDA host, run
`python -m drivebench_tuning.modern_vista --check-runtime` for compatibility
diagnostics. Passing this check does not substitute for model or profiler proof.

The unchanged historical preflight only admits an H100. New modern runs use a
separate declared-device verifier. For example, after resource admission and
with the exact GPU name/capacity fixed in the plan:

```bash
python -m drivebench_tuning.modern_verify_environment \
  --gpu-name 'NVIDIA A100-SXM4-80GB' --minimum-memory-mib 79000 \
  --minimum-host-ram-gib 64 --dedicated \
  --output outputs/modern-environment \
  --write-lock outputs/modern-environment/runtime-lock.json
```

It requires one idle device, the declared physical host/GPU memory, exact modern runtime,
actual CUDA tensor execution and a positive Nsight Systems CUDA trace of the
expected smoke kernel. The current explicit counter-free protocol does not request
Nsight Compute SM counters; it records them as not collected, rather than treating
a permission denial as a passing counter proof. It reuses the original smoke
kernel and parsers and checks
runtime/device continuity afterward. MIG-capable devices must report `Disabled`;
an explicitly declared unsupported device must report `N/A` or `[N/A]`. Unknown
states fail. The proof is labeled `modern_declared_gpu_counter_free_v2`; no legacy H100 proof
is rewritten or reclassified. The public check reads physical host RAM; effective container/cgroup limits need
the separate operator check in the [reproduction guide](reproduce-release.md#verify-hardware-and-acquire-licensed-assets).
The memory minima are declared prerequisites,
not proof that Vista fits. Existing external process-exit/cleanup verification,
full-model fit and quality checks are still required for a new workload.
This narrower activity proof supports execution and timing, not SM utilization,
occupancy or bandwidth claims. The output directory and
lock must be fresh so failed attempts cannot be overwritten.

## Interface

```python
from drivebench.engine import InferenceEngine
from drivebench.types import GenerationConfig, OptimizationConfig
from drivebench_tuning.modern_vista import ModernVistaBackend, RUNTIME

model = {
    "backend": "modern_vista",
    "upstream_dir": "vendor/Vista",
    "checkpoint": "/assets/vista.safetensors",
    "checkpoint_sha256": "<independently recorded SHA-256>",
    "modern_runtime": dict(RUNTIME),
}
generation = GenerationConfig()  # 25 total frames, 50 steps, 576 x 1024
options = OptimizationConfig(compile_denoiser=False)  # paired role toggles only this
backend = ModernVistaBackend(model, generation, options)
engine = InferenceEngine(model, generation, options, backend=backend)
```

The historical backend factory is unchanged. Use the new measurement wrapper
and paired `configs/research/modern-{eager,compiled}.yaml` configurations, which
explicitly inject this backend. Identity records the implementation hashes,
actual runtime, native/action module counts, parameter dtypes and compilation
request. It always marks GPU/quality validation as pending until external
evidence establishes it. No runtime metadata flag certifies correctness.

## What has been checked on GPU

The [November 18 cohort](../results/2026-11-18-a100-modern-eager-feasibility/README.md)
loaded the complete checkpoint and completed three warmups plus one measured
full-resolution eager request on A100-SXM4-80GB. Request latency was 97.994715 s.
Measured allocated/reserved memory was 46.01/53.35 GiB; maximum warmup reserved
memory was 71.61 GiB. All work counters and the finite raw array were independently
checked locally. No compiled request, model operator profile, quality evaluation
or external-laboratory replication was performed. The adapter's original
`modern_gpu_validated: false` and `quality_preserving: false` metadata were not
rewritten: external evidence establishes this narrow execution scope, not a
self-certified quality result.

## What has been checked locally

An isolated macOS CPU environment with Torch 2.14.1/torchvision 0.29.1 installed
successfully, and pinned Vista's original `sample_utils`, `DiffusionEngine` and
`VideoUNet` imported without xFormers. Tensor tests compare SDPA against explicit
attention math, exercise nonzero action/LoRA projections, causal attention,
additional tokens, cross-frame ordering, large temporal batches, parameter
layout and failure/restoration paths. With `VISTA_SOURCE` supplied, tests execute
the original attention forward's projection/layout logic with a CPU math oracle
in place of its CUDA-only xFormers call. This is not a GPU comparison.

A reduced instance of the actual upstream `VideoUNet` also ran on CPU using
16 native attention modules and 25 FP32 norms, producing finite FP16 output.
These CPU checks do not load the full checkpoint or demonstrate full-model fidelity.
The separate GPU execution described above does not supply fidelity validation.

```bash
python -m pip install 'pytest>=8,<9'
VISTA_SOURCE=/path/to/pristine/Vista python -m pytest extensions/kernel_tuning/tests/test_modern_vista.py
```

The historical H100 trace motivates, but does not validate, this port: copy/cast/
concatenation, attention, normalization and elementwise work all contributed
substantially to that one profiled request. That trace used a different runtime.
Fresh stage and operator profiles must establish the modern bottlenecks. Restart
measurements must separately report loading, first-use/compilation, stabilization,
steady inference and break-even request count, with raw outputs retained for
quality analysis. Until those runs finish, no modern speedup, quality reward or
agent-training result is claimed.

## Cheaper hardware starts with a measured fit check

This adapter permits Ampere-or-newer GPUs; it does not require an H100. A
48 GB RTX A6000 remains an untested candidate, not a validated target. The
November 18 A40 attempt failed during its first full-workload warmup; the separate
A100 80 GB eager attempt succeeded. This is not a universal memory boundary.
A same-device modern eager/compiled comparison is still unrun. A 48 GB RTX 6000 Ada
or L40S is another option; an 80 GB A100 provides more memory headroom. A
40 GB A100 has less margin. Only the narrow A100 eager execution above has a successful modern result;
these devices are not interchangeable hardware cohorts. See NVIDIA's
[RTX A6000 specifications](https://www.nvidia.com/content/dam/en-zz/Solutions/products/workstations/nvidia-rtx-a6000-datasheet.pdf),
[A100 specifications](https://images.nvidia.com/data-center/a100/a100-datasheet.pdf)
and [L40S specifications](https://www.nvidia.com/en-us/data-center/l40s/).

Instantiating the full-size configured UNet on PyTorch's meta device counted
1,648,079,082 parameters, including 206,080 GroupNorm parameters retained in
FP32. The stated policy reduces **persistent UNet parameter storage by
3.0694 GiB**. This is not a measured peak-memory reduction. The old 52.21 GiB
H100 peak used FP32 parameters with FP16 autocast; weight-cast caches,
activations, allocator reservation, native SDPA selection and compiler
workspaces can all change. It is therefore insufficient to guarantee a 48 GB
fit, and provides no evidence for a 24 GB fit.

The prospective paired feasibility protocol covers one fixed scene in two fresh processes, one
eager and one compiled, with externally enforced worker deadlines and complete
evidence-copy/cleanup reserves. Each process retains three to twelve cold-prefix
requests plus one measured warm request: eight to twenty-six requests in total.
This includes compiled first use and stabilization. Keep the
25 frames, 576 x 1024 resolution, 50 steps, single future and decoder chunk 14
unchanged. Record model-load and first-use time, allocated and reserved peaks,
device-reported memory, runtime identity and raw output. An OOM or timeout is
preserved as a feasibility failure; it does not trigger an automatic smaller
workload, changed decode chunk or longer allocation. A successful fit is not
a quality result or a complete latency benchmark.

The completed November 18 attempt executed only the eager role and used the
separate `drivebench_tuning.eager_completion` checker. It did not complete the
two-role plan or call the paired analyzer.

Only after both roles fit should the restart study be prospectively bound to
that one physical GPU/runtime. Compare actual cost per completed request,
including startup, compilation and storage, rather than hourly GPU price
alone. Results from a cheaper device must be labeled as its own cohort;
historical H100 speed ratios cannot be transferred.
