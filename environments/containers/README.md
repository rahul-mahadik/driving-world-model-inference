# CUDA containers for the H100 run

Keep Verda's Ubuntu 24.04 / CUDA 12.9 driver and Docker host. These
Ubuntu 22.04 user spaces supply the project's CPython 3.10, CUDA 11.7, 11.8 or 12.4
development toolkit, build tools, rsync, FFmpeg and common profiling tools.
They install no GPU driver. NVIDIA documents backward compatibility between
new drivers and older CUDA runtimes. A CUDA version shown by `nvidia-smi` is
the driver's supported maximum, not the selected compiler/runtime version.

The official `nvidia/cuda` Linux AMD64 manifests were verified through Docker
Registry v2 on 2026-10-26. The build script pins their content digests:

| Mode | Base tag | Linux AMD64 manifest SHA256 |
|---|---|---|
| upstream | `11.7.1-devel-ubuntu22.04` | `a3184e4dc6f968da5bba86df3081ff3013f8e3674a9bfce544af8be905d2f17a` |
| upstream-cu118 | `11.8.0-devel-ubuntu22.04` | `60eda04ab6790aa76d73bf0df245b361eabc6d8f7b6f6cf9846c70f399b9a1eb` |
| modern | `12.4.1-devel-ubuntu22.04` | `5645fec64549cc35930eee9d85aafd2b0006c0c3f22632be5a1d85e2604e9749` |

All modes pin `nsight-systems-2024.2.3=2024.2.3.38-242334140272v0`.
The new compatibility mode pins `nsight-compute-2025.2.1=2025.2.1.3-1`;
the preserved original recipes use `nsight-compute-2024.1.1=2024.1.1.4-1`.
Versioned executable directories come first in `PATH`. Systems 2024.2 supports
CUDA 10.0+; choosing a newer tool without checking supported CUDA versions can
lose CUDA 11.7 tracing.

On the actual H100 with driver 580.178.04, the original cu117 Torch wheel failed
at `torch.randn` with "no kernel image is available", before xFormers attention.
Compute 2024.1.1 also failed to prepare the tiny profiling kernel; the provider's
Compute 2025.2.1 collected positive SM counters on that same binary. Failed
attempts remain in `outputs/host/upstream-bootstrap-attempt-*.log` and
`outputs/environment-hardware-initial/`; successful standalone counter evidence
is under `outputs/environment-profiler-diagnostic/`. These findings motivate the
explicit `upstream-cu118` compatibility mode. It retains Vista's source commit,
Torch 2.0.1 and xFormers 0.0.20 but changes their CUDA build to 11.8. It must not
be represented as successful execution of the original cu117 environment.

The official xFormers 0.0.20 CUDA 11.8 wheel subsequently failed its FlashAttention
kernel on this H100: its recorded architecture list ends at 8.6. The separate
`upstream-cu118-sm90` Python bootstrap uses this same CUDA 11.8 container and
Torch wheels, but builds the pinned xFormers source with
`TORCH_CUDA_ARCH_LIST=9.0`. Preserve both failed environments and require the
native attention smoke to pass before using this environment for Vista. The
independent CUDA 11.8 tensor, Systems and Compute hardware checks passed in
`outputs/environment-hardware-cu118-wheel/`; those checks do not establish
xFormers or full-model correctness.

On this host, the subsequent native build passed head-40 attention, head-64
self/cross attention and the Vista dependency imports, followed by the full
hardware preflight at `2026-10-26T21:50:37Z`. Evidence is under
`outputs/environments/upstream-cu118-sm90/` and
`outputs/environment-upstream-cu118-sm90/`. The reference lock includes the
native wheel SHA256 and `cpp_lib.json` SHA256, so another xFormers build with
the same version cannot silently match it. These setup checks precede full
Vista numerical and performance validation.

The base images and profiling packages are pinned. Ubuntu system dependencies
resolve from its signed repositories at build time, so this is not yet a
byte-for-byte reproducible system-package lock. Every image records its full
installed package versions under `/opt/drivebench-container/`; retain the
built image ID and image itself for exact reuse. The Python bootstrap separately
resolves and records hashed requirements. A future build can fail if a pinned
NVIDIA package is no longer offered; do not silently substitute a version.

## Build on the GPU host

Copy the project to a persistent host directory and enter it. Confirm the
provider's Docker GPU integration and host tools first:

```bash
nvidia-smi
docker info
nvcc --version
nsys --version
ncu --version
bash environments/containers/build.sh --mode upstream-cu118
```

The host `nvcc`, `nsys` and `ncu` commands are diagnostic only; these tools are
supplied inside the containers even if absent on the host. A working NVIDIA
driver and Docker GPU integration are the host requirements.

The build context contains only this recipe directory. The script preserves
logs, image IDs, inspections and recipe copies under
`outputs/environments/containers/upstream-cu118`. It refuses an existing evidence
directory; choose a fresh `--output PATH` for a retry. Builds themselves do not
run CUDA or establish compatibility. Allow space for both images, Python
environments, source builds, checkpoint and datasets; original compressed
base layers total several GB per image. Monitor `df -h` and
`docker system df` on the 200 GiB disk.

## Install and verify the explicit CUDA 11.8 compatibility environment

From the project directory on a dedicated full H100 80 GB host:

```bash
container_image="$(cat outputs/environments/containers/upstream-cu118/image-id.txt)"
docker run --rm -it --gpus '"device=0"' --cap-add=SYS_ADMIN \
  --shm-size=8g \
  -e CUDA_MODULE_LOADING=LAZY -e CUDA_CACHE_MAXSIZE=2147483648 \
  -e CUDA_CACHE_PATH=/workspace/outputs/model-cache/cuda \
  -e HF_HOME=/workspace/outputs/model-cache/huggingface \
  -e TORCH_HOME=/workspace/outputs/model-cache/torch \
  -e XDG_CACHE_HOME=/workspace/outputs/model-cache/xdg \
  --mount "type=bind,source=$PWD,target=/workspace" --workdir /workspace \
  "$container_image" bash
```

`--gpus` requires NVIDIA Container Toolkit integration on the host. The
`SYS_ADMIN` capability enables restricted GPU counter access for this profiling
session without changing host module settings. Verda documents counter access
as enabled on its current GPU images; where the actual preflight passes without
this capability, omit it. Full `--privileged` mode, the Docker socket, host
networking and host PID namespace are not required by this recipe. Nsight runs
with `--sample=none` in the project preflight, avoiding CPU sampling permissions.

Inside the container:

```bash
cp -a /opt/drivebench-container outputs/environments/containers/upstream-cu118/runtime-packages
bash scripts/bootstrap_vista.sh --mode upstream-cu118-sm90 --python python3.10
outputs/venvs/vista-upstream-cu118-sm90/bin/python -m drivebench.verify_environment \
  --dedicated --output outputs/environment-upstream-cu118-sm90 \
  --write-lock outputs/environment-upstream-cu118-sm90/reference-lock.json
```

Use `--dedicated` only after confirming exclusive use of the allocation. The
verifier requires one full H100 with MIG disabled, at least 64 GB host RAM, no
other GPU compute processes, working CUDA PyTorch/Triton, a real Nsight Systems
kernel trace and a positive Nsight Compute SM counter. Missing permissions,
unsupported tools or unavailable counters remain failures. Keep its actual
artifacts and use their path explicitly when downloading the checkpoint:

```bash
outputs/venvs/vista-upstream-cu118-sm90/bin/python scripts/download_checkpoint.py \
  --verification outputs/environment-upstream-cu118-sm90/verification.json
```

CUDA 11.7 cannot emit native H100 `sm_90` code. The existing bootstrap builds
xFormers with `8.0+PTX`, then executes an attention smoke test on the GPU. The
source build completed on this host but the cu117 Torch runtime failed first;
that failed baseline remains preserved separately. Full Vista execution and numerical
parity remain subsequent checks even after both smoke tests pass. PTX compilation
is a cold-start cost and must stay outside steady-state timing measurements.

## Modern environment after the upstream check

Only when that experiment is authorized, exit the compatibility container and
build modern using a fresh evidence directory. The original modern recipe's
Compute 2024.1.1 needs independent verification on its host; the observed failure
with this H100's driver must be addressed before using it for profiling.

```bash
bash environments/containers/build.sh --mode modern
container_image="$(cat outputs/environments/containers/modern/image-id.txt)"
docker run --rm -it --gpus '"device=0"' --cap-add=SYS_ADMIN \
  --shm-size=8g \
  --mount "type=bind,source=$PWD,target=/workspace" --workdir /workspace \
  "$container_image" bash
```

Inside modern:

```bash
cp -a /opt/drivebench-container outputs/environments/containers/modern/runtime-packages
bash scripts/bootstrap_vista.sh --mode modern --python python3.10
outputs/venvs/vista-modern/bin/python -m drivebench.verify_environment \
  --dedicated --output outputs/environment-modern \
  --write-lock outputs/environment-modern/reference-lock.json
```

Run each mode only in its matching image, always mounting the workspace at
`/workspace`: the virtual environments record absolute paths. Stop GPU processes
before switching modes. Keep driver, physical GPU and profiler versions fixed;
record each environment lock separately. Use ordinary containers without the
extra profiling capability for unprofiled timing runs where possible.

Sources: [official CUDA image definitions](https://gitlab.com/nvidia/container-images/cuda),
[NVIDIA Ubuntu 22.04 package index](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/Packages.gz),
[CUDA driver compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/frequently-asked-questions.html),
[Hopper PTX compatibility](https://docs.nvidia.com/cuda/hopper-compatibility-guide/index.html),
[Compute 2022.3 introduces Hopper support](https://developer.nvidia.com/nsight-compute-2022_3-new-features),
[Systems 2024.2 requirements](https://archive.docs.nvidia.com/nsight-systems/2024.2/InstallationGuide/index.html),
[container setup](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
[GPU counter permissions](https://developer.nvidia.com/ERR_NVGPUCTRPERM),
[Verda profiling instructions](https://docs.verda.com/cpu-and-gpu-instances/profile-with-nsight/).
