# Bring an evaluator or policy

The public API is in `drivebench_tuning.search`. It depends on NumPy and Gymnasium, not on a model or GPU library. `TaskSpec` describes a workload with a stable ID, a fixed-length vector of finite numerical features and JSON metadata. `CandidateSpec` gives a candidate a stable ID and an arbitrary JSON configuration. All descriptors are copied/frozen so caller mutation cannot change a running task.

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

This declaration alone does not run a model. Keep all comparison candidates at the declared task semantics and quality level. A changed sampling-step count, dtype or image resolution is not automatically an equivalent workload.

## Connect the inference engine

`drivebench_tuning.inference.InferenceEvaluator` bridges `drivebench.engine.InferenceEngine` to the same `baseline`/`evaluate` contract used for operator search. Install both packages to use it. The [runnable inference example](../examples/inference_env_quickstart.py) uses the core synthetic backend, maps candidate batch settings into the engine, checks generated outputs and evaluates a policy through `BudgetedTuningEnv`.

Supply an immutable `InferenceRequest` from a task, an engine factory, a reference configuration and a mandatory quality gate returning `QualityDecision`. The engine factory constructs the declared implementation; it remains responsible for honoring its configuration. A candidate configuration mapper can translate generic search choices into factory settings. The quality gate must explicitly accept the reference and compare each candidate with its saved reference output. Shape/range validation alone is insufficient to establish application-level quality.

This bridge measures one cold first request per trial: engine construction is timed separately and excluded, while context preparation, generation, synchronization and any first-use compilation are included. It adds no warmups or statistical repetitions. A fresh engine is constructed for each reference/candidate trial. CUDA results use the engine's full-model peak allocated bytes; the CPU example uses returned-array size as an explicitly synthetic memory proxy. These boundaries differ from the warm operator runner and the historical H100 study. Use a separately declared repeated-measurement evaluator when comparing steady-state serving performance.

The result's comparison digest binds the request to the actual reference engine configuration, model-content identity and declared runtime evidence. The raw task/input digest is retained separately in evidence. Changing reference defaults or model contents across fresh evaluation episodes therefore invalidates the comparison, even if a caller reuses its task/cohort labels.

The example establishes CPU integration wiring only. Real Vista execution, a new GPU/runtime binding and application quality criteria must still be independently verified before drawing GPU or video-quality conclusions. Declaring `backend="cuda"` and a cohort identity does not prove them. Engines are cleaned up after each trial; external code must provide suitable cleanup for any backend-owned resources.

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

The environment enforces result identities, freshness, matching input/runtime bindings, finite metrics and correctness eligibility. It does not inspect an external implementation to prove your quality claims. Record a started event before executing a potentially failing operation and persist raw evidence as it becomes available. Return structured incorrect/error results for expected candidate failures; unexpected evaluator exceptions abort and remain visible.

The built-in `OperatorEvaluator` wraps the existing measured `CudaEvaluator` and maps generic candidates to its example controls. Other candidate implementations belong in your own adapter. `SyntheticEvaluator` is an explicit deterministic interface fixture that invents costs; do not use it as a proxy performance model.

## Policy protocol

A policy is any callable `policy(observation, info) -> action_index`. Use `evaluate_policy(env_factory, policy, task_indices, repeats=..., seed=..., split=...)` for fixed-budget episodes with full trajectory records. The factory must return a new environment each time. Each environment calls its evaluator's `close()`. A factory may own a fresh evaluator per episode. Sharing one serial evaluator requires an idempotent, reusable cache-release `close()` (as the built-in CUDA runner provides), or a non-owning adapter with run-owned cleanup. Keep the overall wall-time deadline outside episode resets and never run overlapping measurements on one GPU.

The Gymnasium observation includes task features, per-action counts/status/measured ratios, incumbent and remaining budget. You can train through `reset`/`step` directly or use a compatible RL library. The example baselines are not trained. For learned policies, declare actual training measurements and wall time using `training_metadata`; do not count only evaluation calls when reporting total cost. Open held-out splits and generic candidate payloads are interfaces, not demonstrated generalization.

The [runnable CPU example](../examples/env_quickstart.py) shows environment use and a user-supplied history-aware policy. Its invented metrics test integration only.

If registering a real environment with Gymnasium, use `nondeterministic=True`: fresh GPU measurements and unique audit IDs need not repeat under a seed. Seeds control task/policy selection and inputs, not GPU timings or trial identity. The bundled synthetic fixture separately tests deterministic numerical transitions.
