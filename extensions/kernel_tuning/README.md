# DriveBench agent and configuration-search environments

V0.3 adds durable agent tools, an optional tool-free LLM selector, array-backed proxy checks and frozen final scoring. The [agent guide](../../docs/agent-environment.md) covers the configuration-only trust boundary and pending full Vista demonstration. Run `drivebench-agent-demo --output ./agent-demo` after installing the core and extension to exercise the complete CPU fixture. The new agent environment has no GPU/LLM result yet.

A reusable Gymnasium task for finding a good execution configuration within a measurement budget. The evaluator and candidate payloads are injectable: users can benchmark their own kernel implementations or inference configurations without changing the environment. The bundled CUDA adapter uses the existing GEGLU/GroupNorm+SiLU examples; it is not a generic model executor.

V0.2 adds a new `BudgetedTuningEnv`. The historical `KernelTuningEnv`, its per-trial reward, reference Q learner and published results are preserved. This revision's new suite and adapter wiring have CPU contract coverage; fresh GPU validation is not claimed.

## Install and run anywhere

```sh
python -m pip install ./extensions/kernel_tuning
drivebench-search describe
drivebench-search run --backend synthetic --policy random --split validation --output ./demo-search
```

These commands work outside the source checkout, including after installing a built wheel. The base package needs NumPy and Gymnasium, not Torch or CUDA. Synthetic metrics are deliberately invented to exercise the interface. Do not use them to rank systems or claim optimization gains.

`drivebench-search export-suite --output suite.json` exports a new copy of the packaged task specification. A custom suite changes its SHA-256 identity. Existing files and run directories are never overwritten.

## A search episode

1. `reset(seed=..., options={"task_index": i})` performs a fresh reference measurement.
2. A discrete action indexes a named candidate with an arbitrary JSON configuration.
3. The evaluator measures it, checks correctness and returns a `TrialResult` with matching task/candidate/input/runtime identities and a fresh trial ID.
4. A valid result replaces the incumbent only if its cost improves. Incorrect/error results consume a trial and cannot win.
5. The episode ends when its finite trial budget is spent. All attempted trials remain in the journal.

Cost is `(latency/reference_latency + w * memory/reference_memory) / (1+w)`. The default `w=0` measures latency; a memory tradeoff is an explicitly different objective. Reward is the reduction in incumbent cost at this step. With undiscounted return, rewards sum to the final improvement over the reference. A repeated configuration gets fresh measurements and can appear better due to noise; selected results need separate confirmation.

Observations are a Gymnasium `Dict`: task features, remaining budget, incumbent latency/memory ratios and action, per-action counts, status and latest successful ratios. Use `gymnasium.wrappers.FlattenObservation` if your agent requires a flat vector. The environment exposes measurement history; it does not dictate the learner's architecture.

A `correct=True` response is an evaluator assertion, not an independent proof. External evaluators must implement the documented correctness/quality gates and provide evidence. The environment checks the contract and bindings; it cannot audit an arbitrary user's GPU code. See [writing an evaluator](docs/extending.md).

## Inference settings through the same interface

`InferenceEvaluator` connects the existing video `InferenceEngine` to this environment. Supply task inputs, a factory that maps configurations into engines, and an explicit reference/candidate quality gate. The [CPU inference example](examples/inference_env_quickstart.py) demonstrates batch-setting choices using actual synthetic video generation:

```sh
python -m pip install . ./extensions/kernel_tuning
python extensions/kernel_tuning/examples/inference_env_quickstart.py
```

The example runs without Torch or a GPU. Its outputs and memory proxy are synthetic. The bridge measures cold requests, including first-use compilation when present, while recording engine construction time separately. Real Vista execution through this new bridge remains unvalidated; see the [integration guide](docs/extending.md) for the exact callback and measurement contracts.

## Public operator suite

`operator-search-v1` contains six training, two validation and four test tasks. Shapes do not cross splits, including when only seed/dtype changes. Test shapes come from selected historical Vista profile shapes; the others are constructed within those families. All values are seeded random tensors, not captured activations. This is an **open, newly specified split**, not a secret test or historically untouched data.

The nine candidates are compiled PyTorch and eight example Triton launch configurations. The default four-trial budget is smaller than the action set. The compiled candidate is first in the published grid order; report that order when comparing grid prefixes. Policies may revisit actions, but the supplied random baseline samples without replacement. Use validation to choose policies/objectives, then freeze those choices before test runs.

The [protocol](../../docs/public-benchmark-protocol.md) defines scope, budgets, failure handling, reporting and comparison rules. There is no hardware-independent leaderboard or claim of RL superiority.

## Existing GPU runner

```sh
python -m pip install './extensions/kernel_tuning[gpu,test]'
python -m pytest -q extensions/kernel_tuning/tests -m gpu
drivebench-search run --backend cuda --split validation --policy random \
  --trials 4 --repeats 3 --max-seconds 900 --output ./search-random
```

The CUDA extra is Linux/NVIDIA-specific. Historical successful execution used Torch2.10/Triton3.6 on an A100; this does not certify every GPU or this new suite. Run on an otherwise idle dedicated device. The runner validates numerical outputs and input immutability, records physical GPU/driver/runtime identities, performs five warmups and thirty samples by default, and retains failed trials. It excludes compilation, warmups, validation, input preparation and export from operator latency, while charging them to the wall-time budget. Use an external timeout for stuck CUDA/compiler calls.

Output files: `plan.json` (suite/protocol/source identities), `trials.jsonl` (raw CUDA events or explicitly synthetic records), `result.json` (policy trajectories, per-task summaries and costs), `environment-journals.json`, and `completed.json` or `failed.json`. Partial evaluation reports are retained when available. Completion means the software run ended successfully, not that a policy improved performance. Baseline and candidate calls are counted separately; the bundled policies have no training cost. Report training metadata when evaluating learned policies.

For the historical balanced exhaustive operator sweep, `drivebench-tune benchmark` is still available. Its existing [workload configs](configs) and [historical documentation](HISTORICAL_README.md) describe that separate experiment. The old `tune` command exercises the old Q-learning study and is not the recommended public search protocol.

## Examples

- [CPU environment and external policy](examples/env_quickstart.py)
- [CPU inference settings and quality-gate example](examples/inference_env_quickstart.py)
- [External evaluator contract](docs/extending.md)
- [Historical measured results](../../results/2026-11-07-kernel-tuning/README.md)

First-party code is licensed under [MIT](LICENSE). Dependencies retain their respective licenses; see the repository's [third-party notices](../../NOTICE.md). Prior sealed evidence remains unchanged.
