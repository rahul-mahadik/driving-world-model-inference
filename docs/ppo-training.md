# PPO training on measured operator workloads

Five PPO policies were trained on the [RTX PRO 4000 operator dataset](../results/2026-11-23-operator-generalization-pro-4000/README.md). Its 60 workloads cover GEGLU and GroupNorm+SiLU at three precisions. Thirty workloads belong to training, twelve to validation and eighteen to test. All dtypes of an exact shape stay in one split. The test set contains six held-out shape families from the same two operators.

## Environment and training

An episode gives the policy eight configuration queries. The action space contains 24 Triton launch configurations and compiled PyTorch. The environment begins with an eager reference and retains the best observed numerically correct result. Reward is the decrease in incumbent latency cost; repeated actions consume a trial. Memory weight is zero for this experiment.

Each training evaluator contains the 2,340 recorded rows for the 30 TRAIN workloads. The network receives workload features, remaining budget, the incumbent and previously queried outcomes; task IDs, raw evidence and unqueried measurements are excluded. Hyperparameters and seeds were fixed before training, without using validation or test outcomes.

Training uses [Stable Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html), version 2.9.2, with PyTorch 2.10.0 and NumPy 2.2.0 on CPU. Each episode samples a training task and a starting recorded process repeat. Repeated queries cycle through the three recorded repeats, using the same sampling rule as the existing replay benchmark.

| Setting | Value |
| --- | --- |
| Training seeds | 3101, 3102, 3103, 3104, 3105 |
| Environment steps per seed | 102,400 |
| Completed episodes per seed | 12,800 |
| Parallel environments | 4, executed in one CPU process |
| Policy and value networks | Separate two-layer MLPs, 64 units per layer |
| Learning rate | 0.0003 |
| Rollout steps per environment | 256 |
| Minibatch size / optimization epochs | 256 / 10 |
| Discount / GAE parameter | 1.0 / 0.95 |
| PPO clipping / entropy coefficient | 0.2 / 0.01 |
| Value coefficient / gradient-norm limit | 0.5 / 0.5 |
| Torch CPU threads | 2 |

Training completed before held-out evaluation. The saved parameter hashes differ from initialization, and an offline audit reproduced all 512,000 transitions, recorded-repeat choices, rewards and incumbent updates from the source rows. Total CPU training time was about 139.5 seconds, excluding table loading, evaluation and artifact verification.

## Held-out replay results

Every trained checkpoint received twenty replay episodes on each of eighteen test workloads, for 1,800 episodes. The seed-3101 untrained initialization received the same 360-episode evaluation. Inference is deterministic, with fixed parameters across episodes and varying recorded measurements. Evaluation seed 2101 matches the TPE, random and compiled-query comparisons.

Regret is the selected configuration's median paired candidate/eager latency cost minus the best valid median cost in the table, including eager. Lower is better. The three recorded GPU process repeats define the table; additional replay episodes do not add independent GPU measurements.

| Method | Mean final table regret |
| --- | ---: |
| Compiled-query baseline | 0.027931 |
| Optuna TPE | 0.031880 |
| Random search | 0.055878 |
| PPO, seed 3101 | 0.033024 |
| PPO, seed 3102 | 0.025118 |
| PPO, seed 3103 | 0.027289 |
| PPO, seed 3104 | 0.035376 |
| PPO, seed 3105 | 0.056971 |
| **PPO, mean over all five training seeds** | **0.035556** |
| Untrained initialization, seed 3101 | 0.218278 |

The sample standard deviation of the five trained-seed means is 0.012672. Averaging all five gives higher regret than TPE and the compiled-query baseline. Seed 3101 was chosen before training for the [live comparison with native Triton autotuning on an RTX 4090](../results/2026-11-24-ppo-native-autotune/README.md#live-gpu-comparison).

The numerical gate checks operator outputs under dtype-specific tolerances. Perceptual video quality was outside the study's scope.

## Reproduce training and evaluation

From the repository root, extract the existing dataset and install the RL extra:

```sh
tar -xzf results/2026-11-23-operator-generalization-pro-4000/data.tar.gz -C results
python -m pip install -e './extensions/kernel_tuning[rl]'
python -m pip install numpy==2.2.0
mkdir ppo-run
python -m drivebench_tuning.ppo_cli freeze \
  --table results/2026-11-23-operator-generalization-pro-4000/table.json \
  --protocol ppo-run/protocol.json
python -m drivebench_tuning.ppo_cli train \
  --table results/2026-11-23-operator-generalization-pro-4000/table.json \
  --protocol ppo-run/protocol.json --output ppo-run/training
python -m drivebench_tuning.ppo_cli evaluate \
  --table results/2026-11-23-operator-generalization-pro-4000/table.json \
  --protocol ppo-run/protocol.json --training ppo-run/training \
  --output ppo-run/evaluation
```

Each run requires new output directories. Its protocol records source and data hashes before training; checkpoint manifests contain training provenance, initialization and final parameter hashes, and settings. The published table replaces private hardware identifiers, changing its file hash while preserving numerical rows and sampling. Reproduction therefore records new source and public-table bindings.

Load a trusted locally produced checkpoint for another policy experiment:

```python
from drivebench_tuning.ppo import PPOSelector
from drivebench_tuning.suite import load_suite

suite = load_suite("operator-generalization-v2")
policy = PPOSelector.load(
    "ppo-run/training/seed-3101/model.zip",
    "ppo-run/training/seed-3101/manifest.json",
    suite,
)
action = policy(observation, info)
```

The loader verifies the checkpoint, parameters, suite, candidate ordering and TRAIN-only provenance. It loads a policy with fixed parameters and no attached training environment, accepting recorded or live feedback through the same observation contract.
