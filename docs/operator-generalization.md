# Operator generalization and hardware transfer

The `operator-generalization-v2` suite supports training a configuration selector
on measured workloads and evaluating it on different shapes. It contains 60
tasks: ten GEGLU shapes and ten GroupNorm+SiLU shapes, each in float16, bfloat16
and float32. Each task has 25 candidates: compiled PyTorch, plus Triton block
sizes 64–2048 and one, two, four or eight warps.

The split contains 30 training, 12 validation and 18 test tasks. All versions of
an exact shape stay in one split, including dtype, seed and group-count changes.
Both operators occur in every split. This tests generalization across selected
shapes; it does not test an unseen operator family.

## Measure the workloads

Install the GPU extra and use three fresh processes on one device. Use a new
compiler cache and output directory for each process. These commands collect
4,680 measurements: 60 tasks × 26 configurations including eager × three repeats.

```sh
python -m pip install './extensions/kernel_tuning[gpu,search]'

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

## Train and evaluate selectors

The protocol frozen **before** collection binds the suite, source revision,
eight-turn trial budget, policy settings and training/test boundary. Keep this
file with the measurement evidence.

```sh
# Run after the measured table is complete.
python -m drivebench_tuning.generalization_cli run \
  --suite operator-generalization-v2 --table table.json \
  --protocol protocol.json --output policy-comparison
```

The comparison includes:

| Selector | Information used |
| --- | --- |
| Fixed compiled | The compiled PyTorch setting on every turn |
| Random | Uniform draws without replacement |
| Optuna TPE | The episode's queried outcomes and conditional launch parameters |
| Ridge selector | A cost model fitted to training-task measurements and public task/configuration features |

The ridge model predicts log latency relative to eager. Its normalization and
weights use only training cells; regularization is fixed prospectively. Failed
cells receive a training cost of two. Ridge and TPE clip successful normalized
latency at two for fitting; reported regret uses the uncapped measurements.
Model files retain the feature schema,
weights, training trial IDs and table/runtime bindings. This is supervised
cost-model training. The Gymnasium environment also supports RL policies, but
running this comparison does not train a policy with RL.

Every selector receives eight turns on each test task, with 20 episode repeats.
TPE's repeated suggestions spend turns; rejected and failed actions do too.
Replay cycles through the three recorded process repeats. Additional episodes
measure policy randomness and response to those observations; they do not add
independent GPU samples. The unused validation outcomes remain available for a
separately declared future study.

An optional pretrained LLM selector uses the same task descriptions, candidate
settings, queried feedback and action budget. Declare model snapshots, episode
counts and API bounds in `--llm-plan` when freezing the protocol. If its episode
count is smaller, compare all policies on the matching first episodes and report
that panel separately from the 20-repeat CPU comparison.

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

python -m drivebench_tuning.confirmation \
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
measurements supply a descriptive optimum; they are not used to retrain the
model or revise selections. One pair of GPU/runtime cohorts cannot isolate a
causal hardware effect.

As with the original [measured operator search](measured-operator-search.md),
public tables can be inspected in full. The interface withholds unqueried values
from a compliant policy; it is not a security boundary for untrusted code.
These operator measurements do not establish full-model Vista quality or speed.
