# Bring an evaluator or policy

The public API is in `drivebench_tuning.search` and depends on NumPy and Gymnasium. `TaskSpec` describes a workload with a stable ID, a fixed-length vector of finite numerical features and JSON metadata. `CandidateSpec` gives a candidate a stable ID and an arbitrary JSON configuration. Descriptors are copied and frozen when constructed, preserving the running task if the caller later mutates its inputs.

For example, an external inference adapter could define:

```python
from drivebench_tuning.search import TaskSpec, CandidateSpec, BudgetedTuningEnv

tasks = [TaskSpec("video-576x1024", (576., 1024., 50.),
                  {"model_revision": "your-pinned-revision", "manifest_sha256": "your-manifest-hash"})]
candidates = [
    CandidateSpec("eager", {"execution": "eager"}),
    CandidateSpec("compiled", {"execution": "compiled"}),
]
# my_evaluator must execute the declared workload and quality checks;
# use InferenceEvaluator below to bridge the existing inference engine.
# env = BudgetedTuningEnv(tasks, candidates, my_evaluator, trial_budget=1)
```

Connect these descriptors to an evaluator that executes the reference and candidates under the declared task semantics. Changes to sampling steps, dtype or resolution require explicit acceptance criteria in the quality policy.

## Connect the inference engine

`drivebench_tuning.inference.InferenceEvaluator` bridges `drivebench.engine.InferenceEngine` to the same `baseline`/`evaluate` contract used for operator search. Install both packages to use it. The [runnable inference example](../examples/inference_env_quickstart.py) uses the core synthetic backend, maps candidate batch settings into the engine, checks generated outputs and evaluates a policy through `BudgetedTuningEnv`.

Supply an immutable `InferenceRequest` from a task, an engine factory, a reference configuration and a quality gate returning `QualityDecision`. The factory must construct the requested implementation and honor its settings; an optional mapper can translate generic search choices into factory configurations. The quality gate must accept the reference and compare each candidate with its saved reference output, using application-level criteria in addition to shape and range checks.

The bridge constructs a fresh engine and measures one cold request per trial. Engine construction is timed separately; request latency includes context preparation, generation, synchronization and first-use compilation. There are no warmups or repeated samples. CUDA results use the engine's full-model peak allocated bytes; the CPU example uses returned-array size as a synthetic memory proxy. For steady-state serving comparisons, supply a repeated-measurement evaluator with the required warmups and timing boundary.

The result's comparison digest binds the request to the actual reference engine configuration, model-content identity and declared runtime evidence. The raw task/input digest is retained separately in evidence. Changing reference defaults or model contents across fresh evaluation episodes therefore invalidates the comparison, even if a caller reuses its task/cohort labels.

The example validates CPU integration. Vista use through this bridge still requires GPU execution, a measured runtime binding and validated application-quality criteria. Engines are cleaned up after each trial; external backends must supply cleanup for their own resources.

## Evaluator protocol

Implement `baseline(task)` and `evaluate(task, candidate)`, returning a `TrialResult` for each actual execution. Optionally implement `close()`. Fields are:

| Field | Contract |
| --- | --- |
| `trial_id` | New nonempty ID for every baseline/candidate call; never relabel cached measurements as fresh |
| `task_id`, `candidate_id` | Exact requested identities; reference uses `candidate_id=None` |
| `backend` | `cuda` for real GPU evaluation, `synthetic` only for explicit fixtures |
| `cohort_id` | Stable identity binding physical GPU, driver/runtime and relevant executor version |
| `input_digest` | Stable input/reference-content identity shared by reference and candidate for that task |
| `status`, `correct` | `ok`/true only after declared gates pass; otherwise `incorrect` or `error`/false |
| `latency_ms`, `peak_allocated_bytes` | Finite positive successful latency and valid memory count with documented boundaries; the reference requires positive allocated bytes even for a latency-only objective |
| `evidence` | Finite JSON metadata: raw samples, reference hashes, numerical errors, tolerances, quality scores, compile/warmup costs and failure details as appropriate |

The environment validates result identities, freshness, matching input/runtime bindings, finite metrics and correctness eligibility. The evaluator implements and documents the quality checks. Record a started event before executing a potentially failing operation and persist raw evidence as it becomes available. Return structured incorrect/error results for expected candidate failures; unexpected evaluator exceptions abort the episode and remain visible.

The built-in `OperatorEvaluator` wraps `CudaEvaluator` and maps generic candidates to its operator controls. Implement other candidate types in your adapter. `SyntheticEvaluator` supplies deterministic fixture costs for interface tests.

Before launching a process that uses `CudaEvaluator`, set `TORCHINDUCTOR_COMPILE_THREADS=1`. It checks the effective PyTorch value and owns process-wide compiler state, so use only one evaluator at a time in that process.

## Policy protocol

A policy is any callable `policy(observation, info) -> action_index`. Use `evaluate_policy(env_factory, policy, task_indices, repeats=..., seed=..., split=...)` for fixed-budget episodes with full trajectory records. The factory must return a new environment each time. Each environment calls its evaluator's `close()`. A factory may own a fresh evaluator per episode. Sharing one serial evaluator requires an idempotent, reusable cache-release `close()` (as the built-in CUDA runner provides), or a non-owning adapter with run-owned cleanup. Keep the overall wall-time deadline outside episode resets and never run overlapping measurements on one GPU.

The Gymnasium observation includes task features, per-action counts/status/measured ratios, incumbent and remaining budget. Train through `reset`/`step` directly or use a compatible RL library. The grid and random examples have no training phase. For learned policies, provide training measurements and wall time in `training_metadata` and include them in total cost. Evaluate generalization on held-out workload groups, with policy settings fixed before testing.

The [CPU example](../examples/env_quickstart.py) exercises the environment with a history-aware policy and synthetic fixture metrics.

When registering a GPU environment with Gymnasium, use `nondeterministic=True`. Seeds control task selection, policy choices and inputs; GPU timings and unique trial IDs can vary between runs. The bundled synthetic fixture tests deterministic numerical transitions separately.
