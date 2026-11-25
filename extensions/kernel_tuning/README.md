# DriveBench tuning environments

DriveBench has two tuning tracks. The configuration bandit queries finished implementations; the [sequential schedule track](docs/sequential-schedules.md) edits the current kernel schedule. Both cover GEGLU and GroupNorm+SiLU, with fixed numerical checks against eager PyTorch.

`BudgetedTuningEnv` implements the configuration bandit. Supply an evaluator and candidates to compare kernel implementations or inference settings within a query budget. Queries update the search history and incumbent without changing the next candidate's implementation. The published table and policy comparisons use this track.

The [measured operator search guide](../../docs/measured-operator-search.md) covers CUDA collection, CPU replay, LLM selectors and fresh confirmation of selected configurations. The [generalization suite](../../docs/operator-generalization.md) provides measured training and held-out workloads; the [PPO study](../../docs/ppo-training.md) trains policies on that feedback and evaluates a fixed checkpoint on GPU.

The separate [agent controller](../../docs/agent-environment.md) exposes configuration proposals, profiles, trials and final submission. It includes an optional LLM selector and array-backed quality checks. Its published validation uses a CPU fixture; `drivebench-agent-demo --output ./agent-demo` runs that fixture after installing the core and extension.

## Configuration-bandit quickstart

Install the extension and extract the [recorded operator dataset](../../results/2026-11-23-operator-generalization-pro-4000/README.md), then run:

```python
import gymnasium as gym
import drivebench_tuning  # Registers DriveBench/OperatorTune-v0.

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

Action 0 queries compiled PyTorch; actions 1–24 query Triton configurations. The registered environment uses the `operator-generalization-v2` suite and recorded timings supplied by the caller. Its defaults are the training split, eight queries and a latency-only objective. It makes no GPU or network calls. See the [environment specification](../../docs/gymnasium-environment.md) for spaces, rewards and reset options.

## Install and run anywhere

```sh
python -m pip install ./extensions/kernel_tuning
drivebench-search describe
drivebench-search run --backend synthetic --policy random --split validation --output ./demo-search
```

These commands work outside the source checkout, including after installing a built wheel. The base package depends on NumPy and Gymnasium. The synthetic backend supplies deterministic fixture metrics for interface testing; use measured data for performance comparisons.

`drivebench-search export-suite --output suite.json` exports a new copy of the packaged task specification. A custom suite changes its SHA-256 identity. Existing files and run directories are never overwritten.

## A configuration-bandit episode

1. `reset(seed=..., options={"task_index": i})` requests a reference result from the evaluator.
2. A discrete action indexes a named candidate with an arbitrary JSON configuration.
3. The evaluator measures it, checks correctness and returns a `TrialResult` with matching task/candidate/input/runtime identities and a fresh trial ID.
4. A valid result replaces the incumbent only if its cost improves. Incorrect/error results consume a trial and cannot win.
5. The episode ends when its finite trial budget is spent. All attempted trials remain in the journal.

Cost is `(latency/reference_latency + w * memory/reference_memory) / (1+w)`. The default `w=0` measures latency; positive values add a memory tradeoff. Reward is the reduction in incumbent cost at this step. With undiscounted return, rewards sum to the final improvement over the reference. Measurement noise can make repeated configurations appear better, so confirm selected configurations independently.

Observations are a Gymnasium `Dict`: task features, remaining budget, incumbent latency/memory ratios and action, per-action counts, status and latest successful ratios. Use `gymnasium.wrappers.FlattenObservation` if your policy requires a flat vector.

External evaluators implement the correctness or quality checks behind `correct=True` and return their evidence. The environment validates result identities and comparison bindings. See [writing an evaluator](docs/extending.md) for that division of responsibility.

## Inference settings through the same interface

`InferenceEvaluator` connects the video `InferenceEngine` to this environment. Supply task inputs, a factory that maps configurations into engines, and a reference/candidate quality gate. The [CPU inference example](examples/inference_env_quickstart.py) demonstrates batch-setting choices with synthetic video generation:

```sh
python -m pip install . ./extensions/kernel_tuning
python extensions/kernel_tuning/examples/inference_env_quickstart.py
```

The example runs without Torch or a GPU and uses synthetic outputs and a memory proxy. The bridge measures cold requests, including first-use compilation, while recording engine construction time separately. See the [integration guide](docs/extending.md) for callback contracts, measurement boundaries and the validation required for Vista.

## Public operator suite

`operator-search-v1` contains six training, two validation and four test tasks, grouped by shape across all seeds and dtypes. Test shapes come from selected historical Vista profiles; the others are constructed within those families. Inputs are seeded random tensors, and the split is public.

The nine candidates are compiled PyTorch and eight example Triton launch configurations. The default four-trial budget is smaller than the action set. The compiled candidate is first in the published grid order; report that order when comparing grid prefixes. Policies may revisit actions, but the supplied random baseline samples without replacement. Use validation to choose policies/objectives, then freeze those choices before test runs.

The [protocol](../../docs/public-benchmark-protocol.md) defines workloads, budgets, failure handling and reporting. Results are reported for each measured hardware/runtime cohort.

The [serial comparison example](examples/compare_search_policies.py) alternates grid/random order across repeats, uses matching episode seeds and equal candidate budgets, and saves every episode before producing a comparison. It runs the existing evaluator serially in one runtime cohort:

```sh
python extensions/kernel_tuning/examples/compare_search_policies.py \
  --backend synthetic --split validation --repeats 2 --trials 4 --output ./policy-comparison
```

On a prepared NVIDIA host, run `export TORCHINDUCTOR_COMPILE_THREADS=1` before Python starts, change `--backend synthetic` to `--backend cuda`, and apply an external process timeout. This particular example has CPU coverage; its CUDA comparison remains unrun. Inspect `plan.json`, `episode-*.json`, `trials.jsonl` and `comparison.json`, which report the best results observed during search. Confirm selected configurations in a separate measurement. Replace the policy callback to evaluate another search method and record its training and inference costs separately.

## Existing GPU runner

```sh
python -m pip install './extensions/kernel_tuning[gpu,test]'
export TORCHINDUCTOR_COMPILE_THREADS=1
python -m pytest -q extensions/kernel_tuning/tests -m gpu
drivebench-search run --backend cuda --split validation --policy random \
  --trials 4 --repeats 3 --max-seconds 900 --output ./search-random
```

The CUDA extra targets Linux/NVIDIA with PyTorch 2.10 and Triton 3.6. Run on an otherwise idle dedicated device. The runner validates numerical outputs and input immutability, records physical GPU/driver/runtime identities, performs five warmups and thirty samples by default, and retains failed trials. It excludes compilation, warmups, validation, input preparation and export from operator latency, while charging them to the wall-time budget. Use an external timeout for stuck CUDA/compiler calls.

Each GPU process requires one Inductor compilation thread, checked from the effective PyTorch configuration. The evaluator resets compiler state when the workload changes and on cleanup, while reusing compilation within a workload. Resets occur outside timed samples and leave filesystem caches intact. Use one evaluator at a time in each process.

The historical Vista environment uses a different pinned Torch/CUDA/xFormers stack. Keep it separate from this operator environment; installing the operator `gpu` extra into a Vista container can replace the runtime being evaluated. Follow the [Vista integration recipe](../../docs/vista-integration.md) and bind the actual runtime for a new model cohort.

Output files: `plan.json` (suite/protocol/source identities), `trials.jsonl` (CUDA samples or synthetic fixture records), `result.json` (policy trajectories, per-task summaries and costs), `environment-journals.json`, and `completed.json` or `failed.json`. Partial evaluation reports are retained when available. The completion receipt records run status; `result.json` records policy performance. Baseline and candidate calls are counted separately. The bundled grid/random callbacks have no training phase; learned policies must supply training metadata.

The historical `KernelTuningEnv`, Q learner and `drivebench-tune benchmark` remain available with their original [workload configs](configs) and [documentation](HISTORICAL_README.md). Use `drivebench-search` for the public configuration-search protocol.

## Examples

- [CPU environment and external policy](examples/env_quickstart.py)
- [CPU inference settings and quality-gate example](examples/inference_env_quickstart.py)
- [Serial equal-budget grid/random comparison](examples/compare_search_policies.py)
- [External evaluator contract](docs/extending.md)
- [Historical measured results](../../results/2026-11-07-kernel-tuning/README.md)

First-party code is licensed under [MIT](LICENSE). Dependencies retain their respective licenses; see the repository's [third-party notices](../../NOTICE.md).
