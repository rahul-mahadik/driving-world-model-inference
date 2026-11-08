#!/usr/bin/env bash
# Explicit on-target installation only; no downloads occur merely by importing.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/bootstrap_vista.sh --mode upstream|upstream-cu118|upstream-cu118-sm90|modern [--python python3.10] [--venv PATH] [--output PATH]

Requires Linux x86_64, CPython 3.10, the matching CUDA development toolkit,
git and a C++ build toolchain. Installs one new isolated inference environment.
Existing virtual environments and upstream checkout modifications are refused.
CUDA11.7 upstream mode builds xformers0.0.20 from pinned source. upstream-cu118
is a distinct H100 compatibility baseline using the same package versions with
verified CUDA11.8 wheels. upstream-cu118-sm90 builds pinned xFormers0.0.20
from source for native H100 sm90, after the official wheel's attention failed.
Modern mode uses an exact CUDA12.4 wheel. Full Linux dependency resolution and GPU validation
happen on this host; committed .in files are not validated Linux lockfiles.
EOF
}

mode=""
python_bin="python3.10"
venv_dir=""
output_dir=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) mode="${2:?missing mode}"; shift 2 ;;
    --python) python_bin="${2:?missing Python}"; shift 2 ;;
    --venv) venv_dir="${2:?missing venv path}"; shift 2 ;;
    --output) output_dir="${2:?missing output path}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
[[ "$mode" == upstream || "$mode" == upstream-cu118 || "$mode" == upstream-cu118-sm90 || "$mode" == modern ]] || { usage >&2; exit 2; }
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
venv_dir="${venv_dir:-$repo_root/outputs/venvs/vista-$mode}"
output_dir="${output_dir:-$repo_root/outputs/environments/$mode}"
[[ ! -e "$venv_dir" ]] || { echo "Refusing to change existing environment: $venv_dir" >&2; exit 1; }
"$python_bin" - <<'PY'
import platform,sys
assert platform.system() == "Linux" and platform.machine() == "x86_64", "Requires Linux x86_64"
assert sys.version_info[:2] == (3, 10), "Requires CPython3.10 for pinned wheels"
PY
toolkit=12.4
[[ "$mode" == upstream ]] && toolkit=11.7
[[ "$mode" == upstream-cu118 || "$mode" == upstream-cu118-sm90 ]] && toolkit=11.8
nvcc --version | grep -F "release $toolkit," >/dev/null || { echo "Select CUDA$toolkit nvcc/CUDA_HOME first" >&2; exit 1; }
vista_commit="cc9821b4253ca7987c32757613d2fc2448fa9f5d"
if [[ ! -e vendor/Vista ]]; then
  mkdir -p vendor
  git clone https://github.com/OpenDriveLab/Vista.git vendor/Vista
  git -C vendor/Vista checkout --detach "$vista_commit"
fi
[[ "$(git -C vendor/Vista rev-parse HEAD)" == "$vista_commit" ]] || { echo "Vista checkout is not pinned commit" >&2; exit 1; }
[[ -z "$(git -C vendor/Vista status --porcelain --untracked-files=no)" ]] || { echo "Vista tracked sources have local changes" >&2; exit 1; }
cmp vendor/Vista/requirements.txt environments/upstream-requirements.original.txt
mkdir -p "$output_dir"
"$python_bin" -m venv "$venv_dir"
# Build subprocesses discover Ninja and other installed tools through PATH.
# Resolve an explicitly relative --venv before pip changes working directories.
venv_dir="$(cd -- "$venv_dir" && pwd)"
export PATH="$venv_dir/bin:$PATH"
env_python="$venv_dir/bin/python"
"$env_python" -m pip install pip==24.3.1 setuptools==69.5.1 wheel==0.43.0 uv==0.8.22
"$venv_dir/bin/uv" pip compile "environments/$mode.in" --python "$env_python" --generate-hashes \
  --output-file "$output_dir/requirements.resolved.txt"
"$venv_dir/bin/uv" pip install --python "$env_python" --require-hashes -r "$output_dir/requirements.resolved.txt"
if [[ "$mode" == upstream || "$mode" == upstream-cu118-sm90 ]]; then
  xformers_source="$output_dir/xformers-source"
  [[ ! -e "$xformers_source" ]] || { echo "Choose fresh output directory; xformers source already exists" >&2; exit 1; }
  git clone https://github.com/facebookresearch/xformers.git "$xformers_source"
  git -C "$xformers_source" checkout --detach 1dc3d7a3077ff896cd42d573db9fafeefc97b815
  git -C "$xformers_source" submodule update --init --recursive
  mkdir -p "$output_dir/wheels"
  # CUDA11.7 cannot emit native sm90; preserve forward-compatible compute80 PTX.
  # Actual H100 execution below is required before this environment is usable.
  xformers_arch="8.0+PTX"
  [[ "$mode" == upstream-cu118-sm90 ]] && xformers_arch="9.0"
  BUILD_VERSION="0.0.20" TORCH_CUDA_ARCH_LIST="$xformers_arch" MAX_JOBS="${DRIVEBENCH_BUILD_JOBS:-8}" \
    "$env_python" -m pip wheel --no-deps --no-build-isolation "$xformers_source" --wheel-dir "$output_dir/wheels"
  "$env_python" -m pip install --no-deps "$output_dir"/wheels/xformers-0.0.20-*.whl
fi
"$env_python" -m pip install --no-deps --no-build-isolation -e "$repo_root"
"$env_python" -m pip check
if [[ "$mode" == upstream-cu118 || "$mode" == upstream-cu118-sm90 ]]; then
  "$env_python" - "$output_dir" <<'IMPORTS'
import importlib,json,pathlib,scipy,sys
from drivebench.models.upstream import import_reference
import_reference("vendor/Vista")
for module in ("vwm.models.diffusion", "vwm.modules.encoders.modules", "xformers.ops"):
    importlib.import_module(module)
(pathlib.Path(sys.argv[1]) / "inference-import-smoke.json").write_text(json.dumps({
    "passed": True, "scipy": scipy.__version__, "scope": "pinned Vista imports only; no model weights loaded",
    "dependency_repair": "scipy.integrate required by Vista sampling_utils"}, indent=2))
IMPORTS
fi
"$env_python" -m pip freeze --all > "$output_dir/requirements.freeze.txt"
"$env_python" -m pip inspect > "$output_dir/pip-inspect.json"
"$env_python" - "$mode" "$output_dir" <<'PY'
import hashlib,json,pathlib,platform,subprocess,sys
from drivebench.benchmarks.environment import collect_environment
mode, output = sys.argv[1],pathlib.Path(sys.argv[2])
metadata = collect_environment()
metadata.update(mode=mode, hostname=platform.node(), status="installed; model and profiler validation pending")
metadata["wheel_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (output / "wheels").glob("*.whl")}
metadata["resolved_requirements_sha256"] = hashlib.sha256((output / "requirements.resolved.txt").read_bytes()).hexdigest()
metadata["requirements_input_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (pathlib.Path("environments") / f"{mode}.in", pathlib.Path("environments/common.in"))}
metadata["direct_wheel_pins"] = {}
for line in (pathlib.Path("environments") / f"{mode}.in").read_text().splitlines():
    if " @ " in line and not line.startswith("#"):
        package, address = line.split(" @ ", 1)
        url, marker, sha256 = address.partition("#sha256=")
        if marker:
            metadata["direct_wheel_pins"][package] = {"url": url, "sha256": sha256}
metadata["dependency_repairs"] = ["scipy==1.10.1: required by Vista sampling_utils scipy.integrate import"]
if mode in {"upstream-cu118", "upstream-cu118-sm90"}:
    metadata["compatibility_baseline"] = {"original_mode": "upstream (CUDA11.7)",
        "reason": "Original Torch2.0.1+cu117 H100 randn smoke failed with no kernel image",
        "source_and_package_versions": "same pinned Vista and Torch2.0.1/xFormers0.0.20",
        "change": "official CUDA11.8 Torch wheels and matching xFormers wheel",
        "original_runtime_equivalence_established": False}
if mode == "upstream-cu118-sm90":
    source = output / "xformers-source"
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    assert commit == "1dc3d7a3077ff896cd42d573db9fafeefc97b815"
    metadata["xformers_source_build"] = {"repository": "https://github.com/facebookresearch/xformers.git",
        "commit": commit, "torch_cuda_arch_list": "9.0", "cuda_toolkit": "11.8",
        "submodule_status": subprocess.check_output(["git", "-C", str(source), "submodule", "status", "--recursive"], text=True)}
    metadata["compatibility_baseline"].update(
        reason="CUDA11.7 Torch H100 smoke failed; official CUDA11.8 xFormers wheel FlashAttention also failed with no kernel image",
        change="official CUDA11.8 Torch wheels; exact xFormers0.0.20 source compiled for native sm90")
(output / "installed_environment.json").write_text(json.dumps(metadata,indent=2))
PY
"$env_python" - "$mode" "$output_dir" <<'PY'
import json,pathlib,sys
import torch,xformers,xformers.ops
mode,output=sys.argv[1],pathlib.Path(sys.argv[2])
expected = {"upstream": ("2.0.1+cu117", "11.7", "0.0.20"),
            "upstream-cu118": ("2.0.1+cu118", "11.8", "0.0.20"),
            "upstream-cu118-sm90": ("2.0.1+cu118", "11.8", "0.0.20"),
            "modern": ("2.5.1+cu124", "12.4", "0.0.28.post3")}[mode]
assert (torch.__version__,torch.version.cuda,xformers.__version__)==expected
assert torch.cuda.is_available(), "Working NVIDIA GPU required for attention smoke test"
if mode in {"upstream-cu118", "upstream-cu118-sm90"}:
    build_metadata = json.loads(pathlib.Path(xformers.__file__).with_name("cpp_lib.json").read_text())
    assert build_metadata["version"]["cuda"] == 1108
    assert build_metadata["version"]["torch"] == "2.0.1+cu118"
    if mode == "upstream-cu118-sm90":
        assert build_metadata["env"]["TORCH_CUDA_ARCH_LIST"] == "9.0"
    (output / "xformers-wheel-build.json").write_text(json.dumps(build_metadata, indent=2))
q=torch.randn(1,256,8,40,device="cuda",dtype=torch.float16)
actual=xformers.ops.memory_efficient_attention(q,q,q)
torch.cuda.synchronize()
assert actual.shape==q.shape and torch.isfinite(actual).all()
(output / "attention-smoke.json").write_text(json.dumps({"passed":True,"torch":torch.__version__,"cuda":torch.version.cuda,
  "xformers":xformers.__version__,"device":torch.cuda.get_device_name(0),"scope":"CUDA xformers attention only; model parity and profiling remain pending"},indent=2))
PY
echo "Installed $mode in $venv_dir; preserve $output_dir as resolved environment evidence."
echo "Next run: $env_python -m drivebench.verify_environment --dedicated --output $output_dir/verification"
