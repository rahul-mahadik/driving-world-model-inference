# Reproduce a release

There are three different reproduction targets. Choose one before interpreting a
successful command:

| Target | Required resources | A successful result establishes |
| --- | --- | --- |
| Installed software demos | Python 3.10+, ordinary CPU, package downloads | The released packages and synthetic workflows execute outside their checkout |
| GPU operator evaluation | Dedicated NVIDIA GPU and the operator runtime | Correctness and measurements for the declared operator tasks on that runtime |
| Vista configuration-agent study | Dedicated compatible GPU, licensed assets, metric assets, protected evaluator and optional model API access | Only the outcomes actually recorded by a complete new study |

The first target is verified below. Historical GPU cohorts remain separately
documented. A clean-host run of the current portable Vista agent workflow is
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

## 3. Assemble a new Vista runtime from public recipes

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
