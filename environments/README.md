# Isolated Vista inference environments

These are pinned **installation recipes**, not executed or validated Linux
environment locks. This workspace has no H100, checkpoint or nuScenes download.
The installer resolves a complete hashed dependency file on the target Linux
host, installs it into a fresh environment, then records `pip freeze --all`,
`pip inspect`, exact Python version, CUDA tools, driver, GPU identity and wheel
checksums. Preserve those generated files with every reported run.

`upstream-requirements.original.txt` is a byte-for-byte copy of Vista commit
`cc9821b4253ca7987c32757613d2fc2448fa9f5d` requirements. The original file contains
open ranges and an unpinned CLIP Git dependency, so it is evidence rather than a
reproducibility lock. `common.in` pins the dependency subset imported by Vista
inference, `sample_utils.py`, its training-module imports, and this benchmark.
Training, UI and development packages such as DeepSpeed, Streamlit, wandb, black,
and the unused standalone CLIP package are not installed. This project does not
train Vista.

| Mode | Torch | CUDA toolkit | xformers | Intended use |
|---|---|---|---|---|
| `upstream` | 2.0.1+cu117 | 11.7 | 0.0.20 source at `1dc3d7a3077ff896cd42d573db9fafeefc97b815` | Original Vista inference behavior |
| `modern` | 2.5.1+cu124 | 12.4 | 0.0.28.post3 exact cu124 wheel | Compiler and execution experiments |

Both recipes use CPython 3.10 on Linux x86_64; record its full patch version.
The modern xformers wheel requires glibc 2.28 or newer. The official
[PyTorch version matrix](https://pytorch.org/get-started/previous-versions/)
and [xformers release](https://github.com/facebookresearch/xformers/releases/tag/v0.0.28.post3)
confirm the modern package pairing. This confirms package metadata compatibility,
not execution or numerical parity of Vista.

The PyPI xformers 0.0.20 CPython3.10 wheel embeds `cpp_lib.json` identifying
CUDA11.8 and `torch2.0.1+cu118`. It is therefore not used with the original cu117
recipe. Upstream mode builds the exact source commit using CUDA11.7 and
`TORCH_CUDA_ARCH_LIST=8.0+PTX`. CUDA11.7 cannot emit native H100 sm90 code, so
forward-compatible PTX is required. The install must pass an actual CUDA
attention operation on the reference H100 before it is considered usable.
If it fails, retain the failure and environment; do not silently upgrade the
upstream baseline. Vista originally documented an A100 environment, and exact
H100 compatibility remains a target-machine validation task.

Select the matching installed toolkit through `PATH` and `CUDA_HOME`, then run
one mode explicitly:

```bash
bash scripts/bootstrap_vista.sh --mode upstream --python python3.10
bash scripts/bootstrap_vista.sh --mode modern --python python3.10
```

Each mode uses `outputs/venvs/vista-MODE` and
`outputs/environments/MODE`. Existing virtual environments are refused. Use
`--venv` and `--output` to retain a failed attempt while trying a fresh location.
The script never deletes environments or edits Vista sources. Building upstream
xformers requires a C++ compiler, Git submodules and the selected CUDA development
toolkit. `DRIVEBENCH_BUILD_JOBS` bounds compilation parallelism (default eight).

On the rented full H100, install Nsight Systems and Nsight Compute through the
provider/toolkit instructions and confirm hardware-counter permissions. Run:

```bash
outputs/venvs/vista-modern/bin/python -m drivebench.verify_environment \
  --dedicated --output outputs/environment
outputs/venvs/vista-modern/bin/python scripts/download_checkpoint.py
```

The downloader requires a passed verification with positive SM hardware counters,
both nonempty profiler artifacts, and the same current GPU UUID and driver.
It streams to an exclusive `.part` file, checks the pinned byte count and SHA256,
then publishes the verified file without overwriting an existing checkpoint.
Failures preserve `.part` for diagnosis; move it before explicitly retrying.

The exact checkpoint revision/hash/size and environment sources are recorded in
`provenance.json`. No checkpoint or GPU workload is downloaded by importing a
module, running unit tests, or inspecting these recipes. A successful environment
installation is followed by profiler verification and upstream/refactored model
parity tests; it does not establish a performance improvement.
