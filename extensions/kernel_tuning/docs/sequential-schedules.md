# Sequential kernel schedules

`DriveBench/KernelSchedule-v0` is a Gymnasium environment for editing and measuring a kernel schedule. It covers GEGLU and GroupNorm+SiLU. A schedule records fusion, per-stage launch settings and, for GroupNorm, the reduction strategy. Edits persist: the next action operates on the schedule produced by the previous action.

GEGLU fusion combines erf-GELU and multiplication. GroupNorm fusion combines affine normalization and SiLU; statistics remain a separate single-stage or chunked reduction. Both preserve the reference's intermediate dtype casts.

The [configuration-bandit track](../../../docs/gymnasium-environment.md) remains available for cheap, reproducible CPU replay. Its 25 actions choose finished implementations from a measured table. The schedule track uses live CUDA measurements and does not substitute a table or learned surrogate for an unmeasured schedule.

## Scope and validation

This release supplies the schedule transitions, Gymnasium interface, CUDA lowering and measurement backend, with CPU contract tests and GPU test cases. The new kernel paths have not yet been validated on a GPU. No speedup, policy ranking, model-level coverage or cross-GPU generalization result is claimed for this track. The published leaderboard and existing datasets continue to describe configuration-bandit search.

The interface accepts predefined schedule transformations, not arbitrary Python, CUDA or Triton source. A bounded measurement subprocess limits an individual trial's runtime; it is not a security boundary for hostile native code. Operator correctness checks are numerical comparisons to eager PyTorch, not perceptual video-quality checks.

## State and actions

| Component | Meaning |
| --- | --- |
| Workload | Operator, tensor dimensions, dtype, grouping and input seed |
| Current schedule | Fusion state, main-stage block size and warp count, separate activation settings when unfused, and GroupNorm reduction settings |
| Operator features | The operation graph and its current stage structure |
| Hardware context | Device and runtime information reported by the live evaluator |
| Search state | Remaining wall time, latest measurement and trial outcomes |
| Action mask | Transformations valid for the current schedule and workload |

The action space is `Discrete(18)`. `info["action_names"]` maps integer indices to `block_down`, `block_up`, `warps_down`, `warps_up`, `unfuse`, `fuse`, four corresponding activation-stage edits, four reduction edits and two reduction-strategy switches, `measure`, and `submit`. The observation contains the current schedule and latest measurement; the journal retains the full sequence. Policies that use history must maintain their own memory.

Actions increase or decrease a schedule parameter, fuse or unfuse stages, switch the GroupNorm reduction strategy, measure the current schedule, or submit it. Launch choices for a separate activation stage are available only when that stage exists. Thus an unfuse action can enable edits that were previously invalid. A measure action leaves the schedule unchanged and requests fresh timing samples, allowing a policy to spend budget on uncertainty rather than another edit.

Unfusing copies the current main-stage launch settings to the new activation stage. Fusing removes that stage and clears its settings. For example, `block_up` then `unfuse` increases both resulting stage block sizes; `unfuse` then `block_up` increases only the main stage. These sequences produce different schedules.

The action graph has 812 reachable GEGLU schedules. GroupNorm+SiLU has up to 25,984 when both reduction strategies are supported by the workload: 812 fusion/launch combinations times 32 reduction settings. These counts enumerate structural schedules, not successful GPU compilations. Single-stage reduction is limited to 65,536 values per group; chunked reduction allows at most 4,096 chunks. Larger groups can have fewer legal schedules, and hardware limits can cause compilation or execution failures.

The initial schedule is fused with a 256-element output block and four warps. GroupNorm starts with the smallest statistics block of at least 256 elements that meets the chunk limit; this choice uses no timing feedback. Only supported launch values and lowering strategies are exposed. Boundary edits and operator-incompatible transformations are masked. Selecting a masked edit still consumes time and receives a penalty. A failed compile or incorrect measurement cannot produce a successful submission.

This creates a sequential schedule-search task, but does not establish that RL is the best optimizer. Several edits commute, and many action sequences can reach the same schedule. Random walks, local search, beam search and learned cost models remain relevant baselines. A policy comparison must use the same legal transforms and timing budget for every method.

## Measurements, time and reward

Reset measures eager and compiled PyTorch and selects the correct reference with the lower fresh median. If compilation fails or is slower, eager remains the reference; both outcomes are recorded. Each candidate query compiles or loads the current schedule, checks its output, warms it up and collects fresh CUDA-event samples. The response exposes a sampled timing and retains the raw samples in its evidence. A repeated measurement does not reuse a cached timing or a stored mean.

The episode budget is wall-clock time. Reference setup, compilation, warmup, numerical checks, failed attempts, measurement and time spent choosing actions all consume it. Each edit or illegal action costs at least 1 ms. A per-call timeout is bounded by the remaining episode time. Worker cleanup also counts; a submission that overruns its budget earns no latency score. First-call time includes compilation when needed and is recorded separately from steady-state kernel latency; it is not a pure compiler-only timer. A faster steady-state kernel can still be a poor choice under a short tuning budget.

Submit performs a fresh measurement and numerical check of the current schedule. Its latency reward is `clip(1 - submitted_latency / reference_latency, -1, 1)` only when the submission passes the checks. Elapsed time, invalid actions and failed measurements incur penalties. The journal records the submission score, total reward, outcome and elapsed time. Timing noise remains part of live evaluation; a favorable sample is not a guarantee that the selected schedule is faster. Report independent repetitions and confirmation measurements when comparing policies.

The penalties are `0.05 × newly elapsed budget fraction`, `0.01` for an invalid edit and `0.1` for a failed measurement. Edits and invalid actions cost at least 1 ms of budget. A submit action ends the episode; exhausting the time budget truncates it. Timed-out or incorrect submissions earn no latency score.

The schedule and observation do not expose every source of hardware variation or compiler cache state. Treat those as controlled runtime conditions or partial observability, and record the cache and process protocol used for a comparison.

## Start a live episode

Install the GPU extra in a compatible Linux/NVIDIA environment. The caller supplies the device; this environment does not rent GPUs or add provider credit.

```sh
python -m pip install './extensions/kernel_tuning[gpu]'
export TORCHINDUCTOR_COMPILE_THREADS=1
```

```python
import gymnasium as gym
import drivebench_tuning
from drivebench_tuning.contract import Workload

env = gym.make(
    "DriveBench/KernelSchedule-v0",
    workloads=[Workload("geglu", (64, 512), dtype="float16")],
    budget_seconds=300,
    trial_timeout_seconds=120,
    evidence_dir="./schedule-episode-001",
)
try:
    observation, info = env.reset(seed=42)
    for edit in ("unfuse", "activation_block_up", "measure", "submit"):
        action = info["action_names"].index(edit)
        assert observation["action_mask"][action]
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
finally:
    env.close()
```

The CUDA test cases can be run on a compatible GPU with `python -m pytest extensions/kernel_tuning/tests/test_schedule_kernels.py extensions/kernel_tuning/tests/test_schedule_measurement.py -m gpu`. These cases test representative paths, not every schedule.

Use a new evidence directory for each run. Closing the environment releases its worker. This fixed action sequence illustrates the interface; replace it with a policy to search schedules. CPU fixtures exercise the interface in tests and are not benchmark evidence.

## Comparisons to report

Keep the compiled reference and simple search baselines. Compare final confirmed latency against elapsed tuning time, alongside compilation time, failure rate, invalid-action cost and remeasurement frequency. Report raw trajectories and failures with each result. Since the compositional space is not fully measured, do not report exact oracle regret unless an independently measured oracle exists for the declared subspace.

The initial two-operator scope supports comparisons within those operators. Held-out shapes do not establish transfer to unseen operators. Profile-based operator coverage, whole-operator holdouts, adaptation to unseen hardware and end-to-end model latency each require their own measurements. Existing RTX PRO 4000/RTX 4090 transfer results refer to the older fixed configuration space.

## Related work

[CompilerGym](https://compilergym.com/llvm/#is-this-really-a-sequential-decision-process) exposes compiler passes as actions on the current LLVM IR, with program observations and reward signals. DriveBench uses a smaller set of explicit kernel-schedule edits and live numerical checks; it does not expose a general compiler IR.

[TVM's Ansor autoscheduler](https://tvm.apache.org/2021/03/03/intro-auto-scheduler) constructs schedule spaces from tensor expressions, searches with evolutionary methods and a learned cost model, then uses hardware measurements as feedback. It motivates retaining search and cost-model baselines here. DriveBench currently defines transformations for two operators rather than generating general tensor-program schedules.

[Adams et al.'s Halide autoscheduler](https://halide-lang.org/papers/autoscheduler2019.html) uses beam search over schedules guided by a learned runtime model. It provides another precedent for compositional schedule search without assuming RL is necessary. The DriveBench live track measures queried schedules directly; it has no implicit learned performance model.
