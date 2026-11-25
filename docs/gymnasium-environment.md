# Configuration-bandit environment

This track is contextual-bandit search over finished implementations. Each query changes the policy's information and remaining budget, but does not transform the implementation evaluated next. For stateful kernel edits, use the separate [sequential schedule environment](../extensions/kernel_tuning/docs/sequential-schedules.md).

Importing `drivebench_tuning` registers `DriveBench/OperatorTune-v0` with
Gymnasium. Supply the extracted [operator table](../results/2026-11-23-operator-generalization-pro-4000/README.md):

```python
import gymnasium as gym
import drivebench_tuning

env = gym.make(
    "DriveBench/OperatorTune-v0",
    table_path="results/2026-11-23-operator-generalization-pro-4000/table.json",
    split="test",
)
observation, info = env.reset(seed=42)
for _ in range(8):
    observation, reward, terminated, truncated, info = env.step(0)
    if terminated or truncated:
        break
env.close()
```

Action 0 queries compiled PyTorch. Replace it with your policy's integer action.
The environment replays recorded GPU timings on CPU; it requires no Torch
installation, GPU or network access. The table must match the packaged
`operator-generalization-v2` suite and contain every workload in the requested
split. Construction validates measurements, correctness records and bindings.

## Specification

| Property | Value |
| --- | --- |
| Environment ID | `DriveBench/OperatorTune-v0` |
| Required argument | `table_path`: path to a recorded `table.json` |
| Split | `train` by default; `validation` and `test` are also available |
| Workloads per split | 30 training, 12 validation, 18 test |
| Action space | `Discrete(25)`: compiled PyTorch followed by 24 Triton configurations in suite order |
| Observation space | `Dict` with the seven entries below |
| Default query budget | `trial_budget=8`; one eager-reference lookup also occurs at reset |
| Default objective | `memory_weight=0.0`, minimizing latency relative to eager |
| Reward | Previous incumbent cost minus current incumbent cost |
| End of episode | Query-budget exhaustion returns `terminated=True`, `truncated=False` |
| Render modes | None |

| Observation key | Gymnasium space | Meaning |
| --- | --- | --- |
| `task_features` | `Box(-inf, inf, (8,), float64)` | Operator, dtype, rank, four padded log-scaled dimensions and group count |
| `budget` | `Box(0, 1, (1,), float64)` | Remaining query fraction |
| `incumbent` | `Box(0, inf, (2,), float64)` | Best observed latency and allocation ratios relative to eager |
| `incumbent_action` | `Discrete(26)` | 0 for eager; otherwise action index plus one |
| `action_counts` | `MultiDiscrete([9] * 25)` | Queries used for each action at the default budget |
| `action_status` | `MultiDiscrete([4] * 25)` | 0 untried, 1 correct, 2 incorrect, 3 error |
| `action_metrics` | `Box(0, inf, (25, 2), float64)` | Latest successful latency and allocation ratios for each action; zero for untried or failed results |

`gym.wrappers.FlattenObservation(env)` produces a 412-element vector at the
default budget. Gymnasium one-hot encodes the discrete entries, including
counts and statuses. Changing `trial_budget` changes the count space and its
flattened size.

## Episodes and recorded timings

`reset(seed=...)` seeds task selection and the starting recorded process repeat.
Pass `options={"task_index": i}` to choose a workload by its index within the
selected split. Each candidate's first query uses the same process repeat as
the eager reference; repeated queries cycle through the three recorded repeats.
The same seed and actions reproduce observations and rewards. Replay event IDs
remain unique, while each event retains its original measurement ID and table
hash in `info`. The `measured_replay` and `fresh_gpu_measurement` flags distinguish
these lookups from new GPU calls.

The eager reference starts as the incumbent at cost 1. A correct query replaces
it only if its cost is lower. Incorrect or failed outcomes consume a query,
earn zero and cannot replace the incumbent. With the default objective, cost
is candidate latency divided by reference latency. A positive `memory_weight`
changes it to `(latency_ratio + memory_weight * allocation_ratio) / (1 + memory_weight)`.
Rewards sum to the final observed improvement over eager. Regret is computed
separately from all recorded repeats after the episode.

The registration adds Gymnasium's standard order and API checkers. It has no
default `TimeLimit` wrapper because the query budget is part of the task.
Passing a shorter `max_episode_steps` to `gym.make` adds an external time limit
that returns truncation. Reset before continuing after either kind of episode
end. Close the environment when finished.

For measurements on your own GPU, supply a live evaluator to
`BudgetedTuningEnv` as described in the [evaluator guide](../extensions/kernel_tuning/docs/extending.md).
The registered table environment always uses recorded timings.
