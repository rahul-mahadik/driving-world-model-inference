# DriveBench

DriveBench measures GPU operator latency and memory use. It provides two Gymnasium tracks: contextual-bandit search over recorded configurations, and sequential editing of a kernel schedule with live measurements.

[Dataset and results](results/2026-11-23-operator-generalization-pro-4000/README.md) · [Environment and collection](docs/operator-generalization.md) · [Website](https://rahul-mahadik.github.io/driving-world-model-inference/) · [Citation](CITATION.cff)

## Measured benchmark

The current dataset contains **60 operator workloads**, **24 Triton launch configurations**, compiled PyTorch and an eager reference. GEGLU and GroupNorm+SiLU each contribute ten shapes in float16, bfloat16 and float32. Training, validation and test sets contain 30, 12 and 18 workloads; all dtypes of a shape stay in the same split.

Three fresh processes on an **NVIDIA RTX PRO 4000 Blackwell** produced **4,680 measurements** across 1,560 workload/configuration cells. Each record contains 30 CUDA-event samples, allocator memory, preparation time and numerical checks against eager. All collected configurations passed those checks.

The policy comparison includes a compiled-query baseline, random search, Optuna TPE, a ridge cost model, PPO and two pretrained LLM selectors. Five PPO policies completed 512,000 training steps on the training split. Their mean held-out regret was 0.035556, compared with 0.031880 for TPE and 0.027931 for the compiled-query baseline. That baseline starts with eager and queries compiled, keeping the best configuration observed. The LLM comparison uses one episode per task, matched to the other methods. See the [results](RESULTS.md) and [PPO methods](docs/ppo-training.md) for definitions and analysis.

Across the 18 test workloads, mean rank correlation between RTX PRO 4000 and RTX 4090 was 0.301. The configuration with the lowest source-table cost remained optimal on 6 workloads. The [transfer dataset and plots](results/2026-11-23-operator-generalization-pro-4000/README.md#transfer-to-rtx-4090) contain all 1,404 target measurements.

## Run a comparison without a GPU

```sh
git clone https://github.com/rahul-mahadik/driving-world-model-inference.git
cd driving-world-model-inference
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e . -e './extensions/kernel_tuning[search]'

tar -xzf results/2026-11-23-operator-generalization-pro-4000/data.tar.gz -C results

python -m drivebench_tuning.generalization_cli replay \
  --suite operator-generalization-v2 \
  --table results/2026-11-23-operator-generalization-pro-4000/table.json \
  --output ./policy-comparison
```

The archive includes the measured table, sample values, policy trajectories and trained selector. Its manifest records each file's size and SHA-256. Replay identifies the recorded GPU measurement used for each observation; repeated episodes reuse this finite dataset.

## Choose a track

| Track | Action | Evaluation |
| --- | --- | --- |
| [Configuration bandit](docs/gymnasium-environment.md) | Query a finished implementation | Published GPU table, replayable on CPU; existing policy results below |
| [Sequential schedules](extensions/kernel_tuning/docs/sequential-schedules.md) | Transform the current schedule, then measure or submit it | Live CUDA measurements; new kernel paths require GPU validation before performance claims |

The bandit track changes the policy's information and remaining budget, but a query does not change the implementation queried next. Its PPO results evaluate a learned search policy on that task; they do not demonstrate a benefit from sequential program transformations. The schedule track changes the program state and available edits. It can still be solved by search or a cost model; an RL advantage is an empirical question.

## Use the configuration bandit

`DriveBench/OperatorTune-v0` is registered with Gymnasium when you import `drivebench_tuning`. The published task gives a policy **eight trials** to choose among 25 configurations; the environment accepts other trial budgets. Observations expose the workload, remaining budget and queried outcomes. Reward measures improvement in the best numerically valid configuration found. Repeated and failed trials consume the budget.

```python
import gymnasium as gym
import drivebench_tuning

env = gym.make(
    "DriveBench/OperatorTune-v0",
    table_path="results/2026-11-23-operator-generalization-pro-4000/table.json",
    split="test",
)
observation, info = env.reset(seed=2101)
env.close()
```

The [environment specification](docs/gymnasium-environment.md) lists the spaces, reward and episode behavior. The [Python example](docs/operator-generalization.md#use-the-gymnasium-interface) connects a policy to measured replay or live CUDA evaluation. The [PPO example](docs/ppo-training.md#reproduce-training-and-evaluation) trains with Stable Baselines3; the ridge selector uses supervised regression. A [live RTX 4090 comparison](results/2026-11-24-ppo-native-autotune/README.md#live-gpu-comparison) evaluates PPO seed 3101, chosen before training, against native Triton autotuning with eager and compiled references. These published results belong to the configuration-bandit track. The schedule track exposes predefined transformations; arbitrary code editing still requires a separate execution sandbox.

For new measurements, follow the [GPU collection instructions](docs/operator-generalization.md#measure-the-workloads). Set `TORCHINDUCTOR_COMPILE_THREADS=1` before launching each GPU process. The runner checks that setting, records the runtime and separates preparation from steady-state timing. The caller provisions the GPU and enforces the job deadline.

## Video-model coverage

The repository also provides a [Vista inference adapter](docs/modern-vista.md), a [full-model measurement protocol](docs/modern-baseline-measurement.md), and [agent tools](docs/agent-environment.md). Full-model speed and video quality require separate measurements; the operator dataset evaluates isolated numerical operations. Earlier Vista measurements are available in the [evidence archive](research/archive/README.md).

First-party code is [MIT licensed](LICENSE). Model weights, datasets and third-party software retain their [separate licenses](NOTICE.md). See [CONTRIBUTING.md](CONTRIBUTING.md) for reproducible submissions.
