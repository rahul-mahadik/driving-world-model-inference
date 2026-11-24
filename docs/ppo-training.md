# PPO training on measured operator workloads

Five PPO policies were trained on DriveBench's operator table and evaluated on held-out shapes. PPO improved substantially over its untrained initialization, but its mean final regret remained above Optuna TPE and the compiled-query baseline.

The experiment uses the [RTX PRO 4000 operator dataset](../results/2026-11-23-operator-generalization-pro-4000/README.md). Its 60 workloads cover GEGLU and GroupNorm+SiLU at three precisions. Thirty workloads belong to training, twelve to validation and eighteen to test. All dtypes of an exact shape stay in one split. The test set contains six shape families; it is not an unseen-operator test.

## Environment and training

An episode gives the policy eight configuration queries. The action space contains 24 Triton launch configurations and compiled PyTorch. The environment begins with an eager reference and retains the best observed numerically correct result. Reward is the decrease in incumbent latency cost; repeated actions consume a trial. Memory weight is zero for this experiment.

Observations contain public workload features, remaining budget, the incumbent, and results of previously queried actions. Task IDs, raw evidence and unqueried measurements are not neural-network inputs. Before training, only the 2,340 recorded rows belonging to the 30 TRAIN workloads are copied into a dedicated evaluator. The training environments do not retain the full measured table. Validation and test outcomes are not used for optimization, hyperparameter selection or checkpoint selection.

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

All five checkpoints completed before held-out evaluation. Every checkpoint's parameter hash changed from initialization. Saved training journals were independently checked against the source rows: all 512,000 transitions, recorded-repeat choices, rewards and incumbent updates matched. The five training runs used about 139.5 seconds of CPU training time in total; this timing excludes table loading, evaluation and artifact verification.

## Held-out replay results

Every trained checkpoint received twenty replay episodes on each of eighteen test workloads, for 1,800 episodes. The seed-3101 untrained initialization received the same 360-episode evaluation after all training finished. Inference is deterministic; the twenty episodes vary recorded measurements, not policy parameters. Evaluation seed 2101 matches the existing TPE, random and compiled-query comparisons.

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

The sample standard deviation of the five trained-seed means is 0.012672. This reports variation across training seeds, not a confidence interval for GPU latency. The best individual PPO seed is not selected as the headline result. Seed 3101 was fixed prospectively for a separate live comparison, before training or test results were observed.

The result supports successful RL training on this measured search environment. It does not establish PPO superiority over TPE or the compiled-query baseline. The [separate live GPU comparison](../results/2026-11-24-ppo-native-autotune/README.md#live-gpu-comparison) evaluates the prospectively fixed seed 3101 against native Triton autotuning on an RTX 4090. Numerical checks apply to operator outputs, not perceptual video quality.

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

The protocol records source and data hashes before training. Outputs use new directories and fail rather than overwrite earlier attempts. Checkpoints include hash-bound manifests, training provenance, initialization/final parameter hashes and settings. The published dataset is sanitized, so its file hash differs from the private collection file; numerical rows and sampling remain the same. A new run freezes its own source and public-table bindings.

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

The loader verifies the checkpoint, parameters, suite, candidate ordering and TRAIN-only provenance. The loaded model has no training environment and performs no fine-tuning. It can consume either recorded feedback or fresh feedback from the same public observation contract.
