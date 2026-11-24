# Operator generalization and hardware transfer

The `operator-generalization-v2` suite supports training a configuration selector
on measured workloads and evaluating it on different shapes. It contains 60
tasks: ten GEGLU shapes and ten GroupNorm+SiLU shapes, each in float16, bfloat16
and float32. Each task has 25 candidates: compiled PyTorch, plus Triton block
sizes 64–2048 and one, two, four or eight warps.

The split contains 30 training, 12 validation and 18 test tasks. All versions of
an exact shape stay in one split, including dtype, seed and group-count changes.
Both operators occur in every split, so the held-out unit is the shape within
an operator family.

The [compiler-isolation protocol](../protocols/operator-generalization-v2-compile-isolation-v1.json) fixes compiler-state handling and compilation concurrency for both GPU cohorts.

## Use the measured dataset

Download the [RTX PRO 4000 dataset](../results/2026-11-23-operator-generalization-pro-4000/README.md) to run the comparison without a GPU:

```sh
python -m pip install -e . -e './extensions/kernel_tuning[search]'
tar -xzf results/2026-11-23-operator-generalization-pro-4000/data.tar.gz -C results
python -m drivebench_tuning.generalization_cli replay \
  --suite operator-generalization-v2 \
  --table results/2026-11-23-operator-generalization-pro-4000/table.json \
  --output ./policy-comparison
```

`replay` records a new local protocol before training and evaluation. It binds
the published table and its original collection hashes separately from the
installed replay code. The published protocol and measurements stay unchanged.
Use a new output directory for each run.

## Use the Gymnasium interface

This example runs one eight-trial episode using recorded GPU measurements. Each
step spends one trial, including repeated or rejected actions. The evaluator
preserves the original source trial IDs in its feedback.

```python
from drivebench_tuning.replay import MeasuredTable, MeasuredReplayEvaluator
from drivebench_tuning.search import BudgetedTuningEnv

table = MeasuredTable.load(
    "results/2026-11-23-operator-generalization-pro-4000/table.json"
)
env = BudgetedTuningEnv(
    table.suite.tasks,
    table.suite.candidates,
    MeasuredReplayEvaluator(table, seed=2101),
    trial_budget=8,
    memory_weight=0.0,
    allow_replay=True,
)
try:
    env.action_space.seed(2101)
    observation, info = env.reset(
        seed=2101, options={"task_index": table.suite.splits["test"][0]}
    )
    for _ in range(8):
        action = env.action_space.sample()  # Replace with your policy.
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
finally:
    env.close()
```

The observation contains the task, remaining budget and queried outcomes. The
reward tracks improvement in the best valid normalized cost. The replay
evaluator draws observations from the supplied measurement table.

## Measure the workloads

Install the GPU extra and use three fresh processes on one device. Use a new
compiler cache and output directory for each process. These commands collect
4,680 measurements: 60 tasks × 26 configurations including eager × three repeats.

```sh
python -m pip install './extensions/kernel_tuning[gpu,search]'
export TORCHINDUCTOR_COMPILE_THREADS=1

python -m drivebench_tuning.generalization_cli freeze \
  --suite operator-generalization-v2 --output protocol.json

for repeat in 0 1 2; do
  TORCHINDUCTOR_CACHE_DIR="$(pwd)/cache-$repeat" \
  TRITON_CACHE_DIR="$(pwd)/triton-cache-$repeat" \
  python -m drivebench_tuning.table_collection collect \
    --suite operator-generalization-v2 --repeat-index "$repeat" \
    --output "repeat-$repeat"
done

python -m drivebench_tuning.table_collection merge \
  --runs repeat-0 repeat-1 repeat-2 --output table.json
```

Use an external job deadline. Collection records five warmups and 30 timed
invocations per successful measurement. Compilation is excluded from steady-state
latency and included in preparation time. Numerical failures, allocation failures
and execution errors remain recorded. The table's cell summaries distinguish
variation within a process from variation across fresh processes.

Primary collection, second-GPU collection and fresh confirmation use one
Inductor compilation thread. Set the variable before Python starts; the runner
checks the effective setting and binds it to the protocol and runtime. Compiler
state resets at each workload transition, with compilation reused within a
workload. The reset is included in preparation time, outside timed invocations,
and leaves filesystem caches intact.

## Train and evaluate selectors

The protocol frozen before collection binds the suite, source revision,
eight-turn trial budget, policy settings and training/test boundary. Keep this
file with the measurement evidence. `run` requires that original source version
and a protocol recorded before collection. Use `replay` above when evaluating an
existing dataset with a newer package.

```sh
# Run after the measured table is complete.
python -m drivebench_tuning.generalization_cli run \
  --suite operator-generalization-v2 --table table.json \
  --protocol protocol.json --output policy-comparison
```

The comparison includes:

| Selector | Information used |
| --- | --- |
| Compiled-query baseline (`fixed-compiled`) | Starts with eager, queries compiled on every turn and retains the observed incumbent |
| Random | Uniform draws without replacement |
| Optuna TPE | The episode's queried outcomes and conditional launch parameters |
| Ridge selector | A cost model fitted to training-task measurements and public task/configuration features |

The ridge model predicts log latency relative to eager. Its normalization and
weights use only training cells; regularization is fixed prospectively. Failed
cells receive a training cost of two. Ridge and TPE clip successful normalized
latency at two for fitting; reported regret uses the uncapped measurements.
Model files retain the feature schema, weights, training trial IDs and
table/runtime bindings. Ridge uses supervised cost-model training. The separate
[PPO supplement](ppo-training.md) trains five RL policies with Stable Baselines3
on the same training-task feedback.

Every selector receives eight turns on each test task, with 20 episode repeats.
TPE's repeated suggestions spend turns; rejected and failed actions do too.
Replay cycles through the three recorded process repeats. The twenty episodes
measure policy variability conditional on those timings; the number of
independent GPU measurements remains fixed. Validation outcomes are excluded
from this comparison.

An optional pretrained LLM selector uses the same task descriptions, candidate
settings, queried feedback and action budget. Declare model snapshots, episode
counts and API bounds in `--llm-plan` when freezing the protocol. If its episode
count is smaller, compare all policies on the matching first episodes and report
that panel separately from the 20-repeat CPU comparison.

The released [RTX PRO 4000 → RTX 4090 comparison](../results/2026-11-23-operator-generalization-pro-4000/README.md#transfer-to-rtx-4090) provides a complete source table and a TEST-only target table. After extracting the archive, the target table is at `results/2026-11-23-operator-generalization-pro-4000/transfer/table/table.json`.

## Confirm and transfer frozen choices

Freeze one modal final choice per selector/test task, with ties broken by the
suite's fixed order. The manifest preserves every vote and source report. It is
created before fresh confirmation or collection on another GPU.

```sh
python -m drivebench_tuning.generalization_cli selections \
  --suite operator-generalization-v2 --table table.json \
  --reports policy-comparison/fixed-compiled.json policy-comparison/random.json \
    policy-comparison/optuna-tpe.json policy-comparison/ridge-selector.json \
  --output frozen-selections.json

TORCHINDUCTOR_COMPILE_THREADS=1 python -m drivebench_tuning.confirmation \
  --suite operator-generalization-v2 --selections frozen-selections.json \
  --repeat-index 0 --output confirmation-0
```

Repeat confirmation in two additional fresh processes. Each selected pair is
remeasured alongside eager and compiled references on the original GPU/runtime.
These results do not change the frozen choices.

On a second GPU, use the identical source and suite with `--split test` in each
collection command. This smaller table needs 1,404 measurements. Keep the two
GPU cohorts separate. Then score the original choices on the target table:

```sh
python -m drivebench_tuning.generalization_cli transfer \
  --suite operator-generalization-v2 \
  --source-table table.json --target-table second-gpu-table.json \
  --selections frozen-selections.json --output transfer.json

python scripts/plot_operator_transfer.py \
  --source-table table.json --target-table second-gpu-table.json \
  --selections frozen-selections.json --output transfer-figures
```

The transfer analysis reports per-task rank correlation, regret from transferring
the source-table optimum, and regret for every frozen policy choice. Target
measurements supply a descriptive optimum while the model and selections stay
fixed. The comparison describes this pair of GPU/runtime cohorts.

As with the original [measured operator search](measured-operator-search.md),
public tables can be inspected in full. Policies receive only queried values
through the interface; running untrusted policy code requires separate process
isolation. The measurements cover isolated-operator latency and numerical
agreement.
