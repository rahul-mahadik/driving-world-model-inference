# DriveBench

DriveBench is a GPU benchmark and an environment for agents that tune video-model inference and GPU operators. It measures execution cost, checks candidate outputs, and gives an agent a fixed budget for trying configurations.

[Website](https://drivebench-research.rahul-a-mahadik.chatgpt.site) · [Installation](docs/reproduce-release.md) · [Agent tools](docs/agent-environment.md) · [Operator search](docs/measured-operator-search.md) · [Citation](CITATION.cff)

## Benchmark

The inference runner records request latency, throughput, GPU allocator memory, failures, and the model, input and runtime identities needed to reproduce a measurement. The [modern Vista adapter](docs/modern-vista.md) supports native PyTorch SDPA, explicit precision settings and channels-last execution. The [measurement protocol](docs/modern-baseline-measurement.md) separates cold-start costs from steady-state latency and uses fresh-process repeats.

The operator suite contains 12 workloads across GEGLU and GroupNorm+SiLU, with compiled PyTorch and eight Triton configurations. Inputs are seeded tensors; the four test workloads use selected historical Vista profile shapes. Each candidate is checked numerically against an eager reference. Operator timing excludes compilation and warmup. Successful trials also record preparation time, including compilation, input setup, correctness checks and warmup; all of it consumes the run's wall-time budget.

The [published operator dataset](results/2026-11-20-operator-search-rtx-pro-4500/README.md) contains 360 GPU measurements on an RTX PRO 4500 Blackwell, with three fresh-process repeats per task/configuration. [Collecting a measured table](docs/measured-operator-search.md) reproduces this procedure on another GPU. The table retains the CUDA-event samples, numerical checks and failed configurations. It can then be used for CPU-only search experiments without paying for another GPU measurement on each action.

The expanded [operator generalization suite](docs/operator-generalization.md) provides 60 workloads and 25 configurations, with shape-separated training and test splits, a trainable cost model, TPE search, and a procedure for evaluating frozen choices on another GPU. Its [prospective protocol](protocols/operator-generalization-v2-compile-isolation-v1.json) defines the comparison before collecting results.

## Agent and RL interfaces

Two interfaces share the benchmark's evaluator contracts:

- **`AgentSession`** exposes configuration trials, profiler feedback and final submission through bounded tools. It supports agents that inspect feedback and choose their next experiment.
- **`BudgetedTuningEnv`** implements Gymnasium's reset/step interface. An action selects a configuration; observations include the task, remaining budget and previous outcomes. Reward tracks improvement in the best valid configuration found within the episode.

The measured-table backend supplies recorded GPU observations to the same search environment. It is explicitly labeled replay, and records which original measurement supplied each observation. The supplied comparisons include fixed compiled settings, grid, random search and an optional LLM selector. Selected configurations can be [confirmed in fresh GPU processes](docs/measured-operator-search.md#confirm-selections).

These interfaces support policy evaluation and RL training; they do not require that an agent be trained with RL. The built-in actions are typed configurations. Executing arbitrary submitted code requires an additional isolation layer.

## Get started

```sh
git clone https://github.com/rahul-mahadik/driving-world-model-inference.git
cd driving-world-model-inference
python3 -m venv .venv
. .venv/bin/activate
python -m pip install . ./extensions/kernel_tuning

drivebench-search describe
```

Compare search policies on the measured dataset without a GPU:

```sh
python -m drivebench_tuning.replay_cli \
  --table results/2026-11-20-operator-search-rtx-pro-4500/table.json \
  --split test --trials 4 --repeats 20 --seed 1901 \
  --output ./policy-comparison
```

Replay returns recorded GPU measurements and preserves their source IDs. On a prepared NVIDIA host, install the pinned operator dependencies and collect fresh CUDA measurements:

```sh
python -m pip install './extensions/kernel_tuning[gpu]'
export TORCHINDUCTOR_COMPILE_THREADS=1
drivebench-search run --backend cuda --policy random \
  --split validation --trials 4 --repeats 3 --max-seconds 900 \
  --output ./gpu-search
```

Set the compiler-thread variable before launching each GPU process; the runner checks and records its effective value. Use an external job deadline and keep the operator environment separate from the Vista runtime. The package does not provision compute. See [writing an evaluator](extensions/kernel_tuning/docs/extending.md) to connect another model or kernel implementation, and [measured operator search](docs/measured-operator-search.md) for collection, replay and confirmation commands.

## Results and scope

Published measurements and their reproduction instructions are indexed in [RESULTS.md](RESULTS.md). A full-model result and an operator result are separate measurements; operator speedups do not establish an end-to-end Vista gain.

The research focus is verifiable rewards for lossy video-model optimization. The full-model [quality verifier](docs/verifier-research-design.md) still requires empirical calibration and held-out validation. Current numerical checks on operators do not establish video quality preservation. The [research protocol](docs/research-protocol.md) describes the remaining evaluation work.

First-party code is [MIT licensed](LICENSE). Model weights, datasets and third-party software retain their [separate licenses](NOTICE.md). See [CONTRIBUTING.md](CONTRIBUTING.md) for reproducible submissions.
