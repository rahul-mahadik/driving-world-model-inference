# DriveBench: GPU Inference Benchmarks & Agent Optimization Environment

Measure inference, expose bounded optimization tools, and score a submitted configuration with independent checks. DriveBench combines a Vista video-inference benchmark, an agent tool interface for configuration optimization, and a Gymnasium/operator tier for inexpensive search experiments. Existing Triton kernels are example candidates.

**V18 preparation: agent-environment contracts are CPU validated; the full Vista LLM demonstration is not yet run.** Start with the [agent-environment guide](docs/agent-environment.md) and `drivebench-agent-demo --output ./agent-demo`. Its scripted synthetic episode is a software check, not GPU or LLM evidence. Arbitrary agent-written code is outside this configuration-only interface.

**V17 public MIT release remains the measured publication baseline.** The portable demos and search interface are validated on CPU. Their CUDA adapter reuses the previously executed operator runner, but this revised suite/environment has not been run on a GPU. Historical results stay tied to their original source/runtime. First-party project code is available under the [MIT license](LICENSE); [third-party notices](NOTICE.md) describe separately licensed dependencies, models and data.

## Start without a GPU

Clone the repository and install either or both packages in a clean Python 3.10+ environment:

```sh
git clone https://github.com/rahul-mahadik/driving-world-model-inference.git
cd driving-world-model-inference
python -m venv .venv
. .venv/bin/activate
python -m pip install . ./extensions/kernel_tuning
```

You can then leave the checkout. Both commands use installed resources and create a **new** output directory:

```sh
drivebench-demo --output ./video-demo
drivebench-search run --backend synthetic --split validation --policy random --output ./search-demo
```

The video demo exercises data preparation, generation, comparison and an HTML report. The search demo exercises the Gymnasium environment, policy evaluation and JSON evidence. Both are explicitly synthetic: their numbers are software demonstrations, not GPU performance or model-quality measurements. PyTorch, datasets and model weights are unnecessary for these demos.

## What can be benchmarked

| Track | Inputs and candidates | Measurements | Quality boundary |
| --- | --- | --- | --- |
| Vista inference | Separately obtained Vista weights and licensed driving data; eager/compiler and declared inference configurations | Request latency, sequential throughput, peak memory, failures, runtime and input identity | Numerical/development and blind-review tooling; held-out quality validation of the historical speed result is incomplete |
| Operator search | Packaged seeded tensor shapes; compiled PyTorch and eight example Triton configurations | CUDA-event latency samples, host wall time, incremental peak allocation, correctness, per-trial evidence | Reference tensor shape/dtype/finiteness, allclose and immutable inputs |
| External evaluation | Your `TaskSpec`, `CandidateSpec` and evaluator | The same cost, budget and result contract | Your evaluator must implement and document the actual reference and acceptance checks |

Whole-video settings and individual kernel configurations use the same search interface. An inference bridge connects the existing `InferenceEngine` to this contract, with a required quality check; a [runnable CPU example](extensions/kernel_tuning/examples/inference_env_quickstart.py) demonstrates the wiring. The bundled real CUDA evaluator supports GEGLU and GroupNorm+SiLU. Real Vista execution through the new bridge is unvalidated, and an arbitrary model is not automatically supported by declaring a task. The [integration guide](extensions/kernel_tuning/docs/extending.md) explains what an adapter must supply.

## Run on your GPU

The example operator runner uses a separate Linux NVIDIA environment with Torch 2.10.0 and Triton 3.6.0. It requires a physical GPU UUID and records driver/runtime identities. It performs no cloud provisioning or payments.

```sh
python -m pip install './extensions/kernel_tuning[gpu]'
drivebench-search describe
drivebench-search run --backend cuda --split validation --policy random \
  --trials 4 --repeats 3 --max-seconds 900 --output ./cuda-search
```

Use an outer process timeout and your own resource controls: an inner budget cannot interrupt a stuck compiler or CUDA call. Start with the CPU demo, inspect the suite, then run GPU correctness tests before making performance claims. See the [search guide](extensions/kernel_tuning/README.md) and [benchmark protocol](docs/public-benchmark-protocol.md). Changing runtime, objective, suite or timing settings creates a different comparison.

For actual Vista generation, follow [Vista integration](docs/vista-integration.md), [full-study data workflow](docs/full-study.md) and the [historical GPU runbook](docs/gpu-runbook.md). Historical environment recipes reproduce historical cohorts; they are not claims about the fastest current software stack.

## Use your own policy

The environment exposes task features, remaining budget, the incumbent, and per-action measurement history. `reset()` measures the reference; `step(action)` performs a fresh candidate trial. The reward is improvement in the best correctness-passing cost found. Failed trials consume budget. Repeating an unchanged good result earns no improvement reward.

The default suite gives a policy four trials among nine candidates. Fixed grid prefixes and random search without replacement are supplied as baselines. Evaluation records full trajectories and separates reference calls, candidate calls and declared training costs. Training a strong agent is the user's experiment; a working environment does not imply RL beats autotuning.

## Results and reproducibility

- [Historical H100 inference comparison](results/2026-10-30-h100-tradeoff/README.md): 200 requests per role; 23.85% lower median latency and 7.68% higher memory on the pinned setup. No quality-preservation claim.
- [Historical A100 operator and reference-agent study](results/2026-11-07-kernel-tuning/README.md): measured operator tradeoffs; no demonstrated RL advantage. The reference-agent run used two small smoke shapes, not the newly packaged suite.
- [Complete historical summary](RESULTS.md) and [previous front page](HISTORICAL_README.md): preserved cohorts, failures and limitations.

Raw per-trial samples from new operator runs are stored locally with the run. Historical public summaries do not include every raw timing row or generated array. Dataset images, generated videos and checkpoints are not redistributed. See [third-party notices](NOTICE.md).

Contributions should add adapters, workloads, policies or measurement checks through the [contribution guide](CONTRIBUTING.md), with independently measured claims and unchanged historical evidence.
