# Reproduce a release

There are four different reproduction targets. Choose one before interpreting a
successful command:

| Target | Required resources | A successful result establishes |
| --- | --- | --- |
| Installed software demos | Python 3.10+, ordinary CPU, package downloads | The released packages and synthetic workflows execute outside their checkout |
| GPU operator evaluation | Dedicated NVIDIA GPU and the operator runtime | Correctness and measurements for the declared operator tasks on that runtime |
| Modern Vista eager execution | Linux CPython 3.12, dedicated compatible NVIDIA GPU, licensed assets and fresh activity/runtime proof | Full-workload execution feasibility for that one workload, without requiring hardware performance counters |
| Vista configuration-agent study | Dedicated compatible GPU, licensed assets, metric assets, protected evaluator and optional model API access | Only the outcomes actually recorded by a complete new study |

The CPU installation target is verified below. The November 18 [A100 modern
eager run](../results/2026-11-18-a100-modern-eager-feasibility/README.md) also
provides a real full-workload execution record. Historical GPU cohorts remain
separate. A clean-host run of the current portable Vista agent workflow is
**not yet verified**. There is no downloadable prebuilt DriveBench Vista image;
historical image IDs identify locally built images, not public registry assets.

## 1. Build wheels and check a clean CPU installation

The check installs no CUDA, model weights, datasets, LPIPS or LLM SDK. These
commands use a fresh temporary directory for both build output and installed
software. They do not require the authors' accounts, cloud provider or private
configuration files.

```sh
git clone https://github.com/rahul-mahadik/driving-world-model-inference.git
cd driving-world-model-inference
git rev-parse HEAD
# For a published study, check out its recorded commit before continuing.
repo_dir="$PWD"
repro_dir="$(mktemp -d)"

python3 -m venv "$repro_dir/build-env"
"$repro_dir/build-env/bin/python" -m pip wheel --no-deps \
  --wheel-dir "$repro_dir/wheels" . ./extensions/kernel_tuning

python3 -m venv "$repro_dir/installed-env"
"$repro_dir/installed-env/bin/python" -m pip install "$repro_dir"/wheels/*.whl
"$repro_dir/installed-env/bin/python" -m pip check

cd "$repro_dir"
"$repro_dir/installed-env/bin/python" -I \
  "$repo_dir/tools/reproducibility/check_install.py" \
  --output "$repro_dir/demo-results"
```

The final command uses the existing installed `drivebench-demo`,
`drivebench-search` (synthetic backend) and `drivebench-agent-demo` entry points.
It uses isolated Python imports and a minimal subprocess environment without
inherited provider/API credentials. It verifies the demos' completion receipts
and explicit synthetic labels. It is an installation check, not an operating
system sandbox or GPU benchmark. The scripted agent demo uses a numeric metric
fixture, not a real LLM or LPIPS.

Expected console summary:

```json
{"status": "passed", "synthetic": true, "gpu_validation": false, "commands_passed": 3}
```

`installation-check.json` records Python/platform and package versions, optional
package presence, the checker hash and individual receipt hashes. Each demo
retains its logs and outputs. Failed attempts remain available; the checker
refuses an existing output directory and refuses editable package installs.
Synthetic latency/reward values can vary and are not accepted GPU results.

For the independent local execution of this procedure, see the
[sanitized CPU reproduction record](../tools/reproducibility/cpu-reproduction-record.json).
The public CI's [installed-demo job](../.github/workflows/portable.yml) provides a
separate Linux/Python 3.11 check. Neither is independent external-laboratory
replication of the GPU study.

## 2. Reproduce an operator task

Use a **separate** Linux GPU environment for the operator extra:

```sh
python -m pip install './extensions/kernel_tuning[gpu]'
drivebench-search describe
drivebench-search run --backend cuda --split validation --policy random \
  --trials 4 --repeats 3 --max-seconds 900 --output ./operator-validation-01
```

The extra pins Torch 2.10.0 and Triton 3.6.0. This is not the historical Vista
environment. Run the [GPU correctness tests](../extensions/kernel_tuning/README.md)
on the actual device first; record its GPU/driver identity and the complete
resolved environment. An external process/resource timeout must bound a hung
CUDA call. The example's inner budget does not provide that protection.

Register suite hash, policy, objective, trial budget, warmups, repetitions and
measurement boundary before evaluation. Use validation for choices and preserve
failed trials. A separate fresh confirmation of the locked selected candidate is
needed for an improvement claim. The published test shapes were historically
visible; do not describe them as secret or unseen. See the
[measurement protocol](public-benchmark-protocol.md) for comparison rules.

## Modern Vista eager execution

The [tested Linux runtime recipe](#install-the-modern-linux-runtime) below supports
the single-eager execution target. It does not require the authors' provider
accounts, transport bindings or private controller. It does require assets obtained
under their own licenses and an operator-controlled process/resource timeout.
The library does not terminate a stuck CUDA process or release cloud resources.
Use a fresh output directory and preserve every failed attempt.

### Install the modern Linux runtime

Use Linux x86-64 with CPython 3.12, Git, a CUDA 13.0 compiler/toolkit, and Nsight
Systems. The observed run used Ubuntu 24.04, Python 3.12.3,
Git `1:2.43.0-1ubuntu7.4`, `nvcc` 13.0.88, Nsight Systems 2025.3.2 and NVIDIA
driver 595.112.07. Those are recorded versions, not promises that mutable system
repositories still serve them. Check the actual driver/runtime on the new host.

The [hash-locked wheel list](../environments/modern-vista-cu130-linux.lock.txt)
and [artifact provenance](../environments/modern-vista-cu130-linux.wheels.json)
record the exact third-party Python environment. The small ANTLR wheel is included
with its upstream license; weights, datasets and large CUDA wheels are not bundled.
Download dependencies before GPU measurement, then install offline:

```sh
python3.12 -m venv .venv-modern
. .venv-modern/bin/activate
python -m pip download --only-binary=:all: --require-hashes \
  --find-links third_party/wheels \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  -r environments/modern-vista-cu130-linux.lock.txt -d /fresh/wheelhouse
python -m pip install --no-index --find-links /fresh/wheelhouse --require-hashes \
  -r environments/modern-vista-cu130-linux.lock.txt
python -m pip wheel --no-deps --no-build-isolation \
  --wheel-dir /fresh/project-wheels . ./extensions/kernel_tuning
python -m pip install --no-index --no-deps /fresh/project-wheels/*.whl
python -m pip check
python -m pip freeze --all > /fresh/resolved-environment.txt
```

Here and below, replace `/fresh/...` with new caller-owned locations. Check out
Vista revision `cc9821b4253ca7987c32757613d2fc2448fa9f5d` under `vendor/Vista`
and leave it pristine. Do not install its legacy requirements into this modern
environment. Export the cache location before any runtime proof or model import:

```sh
export HF_HOME=/fresh/model-assets/hf
```

Keep this variable in the environment passed to your external bounded caller.
Prefix verification, plan, worker and terminal-check commands with
`bash scripts/with_modern_vista_env.sh`; it supplies import-time offline settings,
`PYTHONDONTWRITEBYTECODE=1` and `GIT_OPTIONAL_LOCKS=0`. The last setting prevents
read-only Git status probes from refreshing a hash-pinned checkout index. This
wrapper is not a network sandbox or a timeout.

### Verify hardware and acquire licensed assets

The successful device was an A100-SXM4-80GB with 81,920 MiB GPU memory and
232.83 GiB effective host RAM. The declared host minimum was 64 GiB; it was not
measured as the minimum necessary. **The public verifier checks physical host RAM,
not the effective container limit.** Before running, the operator must separately
check the minimum of physical RAM and every applicable cgroup memory limit, and
require at least 68,719,476,736 bytes. On cgroup v2, inspect `memory.max` for the
process cgroup and its constrained ancestors; on v1 inspect the applicable
`memory.limit_in_bytes` values. Treat `max` as unbounded at that level, not proof
that all ancestors are unbounded. A control-panel advertised value or a physical
RAM check alone is insufficient. If the effective limit cannot be established,
do not accept the host. The original bounded run separately checked its effective
limit; that private controller is not provided by the public verifier. The preceding A40 attempt ran out of GPU
memory during its first warmup. Do not infer 48 GB fit from the A100 warm-request
allocated peak: warmup reserved memory reached 71.61 GiB.

For a new dedicated A100, execute under your external deadline:

```sh
bash scripts/with_modern_vista_env.sh python -m drivebench_tuning.modern_verify_environment \
  --gpu-name NVIDIA\ A100-SXM4-80GB --minimum-memory-mib 79000 \
  --minimum-host-ram-gib 64 --dedicated \
  --output /fresh/environment --write-lock /fresh/environment/runtime-lock.json
```

MIG must be disabled. The counter-free proof requires actual CUDA execution and
a positive expected-kernel Nsight Systems trace. Nsight Compute SM counters are
not requested, so no SM/occupancy/bandwidth interpretation is available.
A hardware-only tool probe or proof copied from another machine is insufficient.

Acquire nuScenes through its access process, then prepare one real scene using
the [data workflow](full-study.md). Keep image and metadata checksums, coordinate
semantics, recorded trajectory and seed in the manifest. The released result
identifies its scene, sample, seed and trajectory hash for comparison; media are
not bundled. The manifest must contain one scene for this single-eager target.

After a passing fresh proof, the hash-checked fetch helper can obtain the declared
Vista and OpenCLIP assets. Set `PROOF_SHA256` to the SHA-256 of that new
`verification.json`, and `ASSET_DEADLINE` to your fresh absolute Unix deadline:

```sh
python scripts/fetch_modern_assets.py --asset vista --root /fresh/model-assets \
  --verification /fresh/environment/verification.json --proof-sha256 "$PROOF_SHA256" \
  --deadline-unix "$ASSET_DEADLINE" --failure-receipt /fresh/vista-fetch-failure.json
python scripts/fetch_modern_assets.py --asset openclip --root /fresh/model-assets \
  --verification /fresh/environment/verification.json --proof-sha256 "$PROOF_SHA256" \
  --deadline-unix "$ASSET_DEADLINE" --failure-receipt /fresh/openclip-fetch-failure.json
```

Fetching is a separate network-enabled preparation step. Worker initialization
uses only the prefetched cache. Preserve successful asset receipts and any partial
or failed downloads. The checkpoint and inputs retain their own licenses.

### Run and verify one eager request

Start from `configs/research/modern-eager-feasibility.yaml` and its compiled
counterpart. Create two fully resolved copies in your new workspace; set
`model.upstream_dir`, `model.checkpoint`, `data.root`, `data.manifest`,
`environment_verification` and `environment_lock` to the new verified locations.
Keep the pinned checkpoint hash and the full workload: 25 total frames, one
conditioning frame, 576×1024, 50 steps, guidance 2.5, FP16, decode chunk 14, one
uncached future, one measured request, three-to-twelve warmups, and output saving.
Only `optimizations.compile_denoiser` may differ between the role definitions.
The compiled definition is required by the existing plan format but is not executed
in this eager-only target.

```sh
bash scripts/with_modern_vista_env.sh python -m drivebench_tuning.restart_measurement plan \
  --eager-config /fresh/eager.yaml --compiled-config /fresh/compiled.yaml \
  --mode feasibility --blocks 1 --gpu-name NVIDIA\ A100-SXM4-80GB \
  --minimum-memory-mib 79000 --minimum-host-ram-gib 64 --mig-supported \
  --output /fresh/measurement-plan.json
```

Record the new plan SHA-256 as `PLAN_SHA256` and a fresh externally enforced
absolute deadline as `WORKER_DEADLINE`. Neither historical clocks nor this
document authorize a cloud allocation. Run only the eager schedule entry:

```sh
bash scripts/with_modern_vista_env.sh python -m drivebench_tuning.restart_measurement run \
  --plan /fresh/measurement-plan.json --plan-sha256 "$PLAN_SHA256" \
  --run-id 000-modern_eager --deadline-unix "$WORKER_DEADLINE" \
  --output /fresh/eager-run
bash scripts/with_modern_vista_env.sh python -m drivebench_tuning.eager_completion \
  --plan /fresh/measurement-plan.json --plan-sha256 "$PLAN_SHA256" \
  --run /fresh/eager-run --output /fresh/eager-summary.json
```

The terminal command validates existing evidence and performs no inference.
It reports single-eager feasibility and leaves quality, speedup and comparison
flags false. Do not run the paired analyzer on this incomplete two-role schedule.
Confirm parent exit and cleanup externally, hash all evidence and raw outputs,
and verify the complete backup before releasing resources. A complete paired
modern-baseline study is a separate [measurement protocol](modern-baseline-measurement.md).

## 3. Assemble a legacy Vista runtime from public recipes

The historical CUDA 11.8 H100 recipe is available as source. On a Linux x86-64
host with working NVIDIA driver and Docker GPU integration, start with:

```sh
bash environments/containers/build.sh --mode upstream-cu118 \
  --output outputs/environments/containers/new-upstream-cu118
```

Follow the [container guide](../environments/containers/README.md) to run the
built image and create a new CPython 3.10 environment using
`scripts/bootstrap_vista.sh --mode upstream-cu118-sm90`. That path uses Torch
2.0.1/CUDA 11.8 and a pinned xFormers 0.0.20 source build for native H100 support.
The recipe resolves and records actual dependencies on the target host;
committed `.in` inputs are not complete byte-for-byte environment locks.
System package repositories may also change. Preserve the resulting image,
image ID, resolved package hashes, tool versions and installation logs.

Run the actual environment verifier and retain its profiler artifacts and lock
before using the checkpoint downloader. The bundled verifier's historical
hardware contract is a dedicated full H100; it is **not** an automatic A100 or
arbitrary-GPU approval. A new device/environment contract requires an explicit,
reviewed adaptation and fresh execution evidence. A passing-looking JSON file
copied from another machine is not a verification result.

Build/runtime availability, native attention execution, profiler permissions,
memory use and model compatibility must all be checked on the new host. This
document does not claim that the recipe was freshly built on Linux during the
CPU reproduction check, or that another hardware setup reproduces historical
timings exactly.

## 4. Acquire and verify assets separately

| Asset | Acquisition and verification |
| --- | --- |
| Vista source | `scripts/bootstrap_vista.sh` obtains OpenDriveLab/Vista and checks revision `cc9821b4253ca7987c32757613d2fc2448fa9f5d` and pristine tracked sources |
| Vista checkpoint | `scripts/download_checkpoint.py` uses the pinned Hugging Face revision and validates byte count and SHA-256 after a passing actual environment proof |
| OpenCLIP initialization | Pre-stage the artifact requested by the pinned Vista/OpenCLIP constructor; record its identity and make it available without worker networking |
| Driving inputs and annotations | Obtain nuScenes through its own access process; prepare deterministic manifests with checksums using the [data workflow](full-study.md) |
| Perceptual metric | Install LPIPS 0.1.4 in its declared evaluator environment, pre-stage and hash its required network assets, and run the real metric validation |

The model integration's [artifact identities](vista-integration.md#required-artifacts-and-integrity)
and [third-party notices](../NOTICE.md) describe attribution and acquisition.
The first-party MIT license does not grant rights to redistribute third-party
images, weights or generated media. Numerical records can be published with a
clear schema and provenance while media remain subject to their own terms.

## 5. Run the declared study, not a historical machine binding

Use the public [portable launcher](../tools/host/README.md), binding the immutable
ID of **your** built image, exact interpreter path and actual environment proof.
The worker refuses automatic image pulls and runs without networking; pre-stage
dependencies and assets. Do not reuse historical image assumptions, provider
instances, deadlines or proof files as new execution authorization.

Follow the [agent study recipe](agent-study-reproduction.md) for generation,
calibration, policy selection, the final-selection barrier, final metrics and
blinded review exports. The current interface accepts bounded configuration
actions. It is not a sandbox for arbitrary agent-written Python or kernels.

Before a paid run, the operator needs a complete workload manifest; fresh source,
model, input and runtime bindings; justified output-change limits and controls;
an explicit policy/budget/ordering plan; a protected final split; an external
deadline; and capacity for every failure, warmup, raw output and verified backup.
The measurement library itself neither provisions resources nor purchases
credits. An LLM policy is optional; any API access belongs on the policy side,
outside worker files and released evidence.

## A reproduction claim must name its scope

Publish the exact source commit, task/protocol hash, hardware/runtime record,
asset acquisition steps, resolved environment, per-task numerical records,
failure counts, full policy trajectories and costs. Supply the analysis that
produces each reported statistic. Confirm the selected configuration with fresh
measurements, retain all output evidence and document backup verification.

A synthetic run verifies software integration. A real operator run validates
only that operator task. A complete small Vista episode demonstrates one
configuration-agent integration. None alone establishes broad perceptual
equivalence, driving safety, generalization or superiority over other optimizers.

For citation, use [CITATION.cff](../CITATION.cff) and include the exact Git commit
and protocol identity. There is currently no paper or DOI implied by that file.
