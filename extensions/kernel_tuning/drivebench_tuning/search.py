"""Generic, finite-budget search with explicit evaluator and evidence contracts.

This module is separate from the historical kernel environment. An action may
describe a kernel launch, compiler option, or another inference configuration;
only the external evaluator knows how to execute it. ``backend='cuda'`` and
``correct=True`` are evaluator assertions, not independent proof of GPU
execution or task quality. Adapters must supply a relevant correctness/quality
gate and retain evidence supporting those assertions.

The objective is the best admissible measured cost discovered within a budget.
Reward is the reduction in incumbent cost, so repeating an unchanged result
earns zero. All comparisons use the reset's reference measurement. The default
objective is latency alone; a nonzero memory weight explicitly trades latency
against incremental peak allocation. Outcomes are best-observed measurements,
not independently confirmed improvements. No statistical conclusion is made.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
import math
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Protocol

import gymnasium as gym
from gymnasium import spaces
import numpy as np


def _identity(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return number


def _freeze_json(value: Any, seen: set[int] | None = None) -> Any:
    """Copy finite JSON values into immutable containers; reject cycles."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return value
    if not isinstance(value, (Mapping, list, tuple)):
        raise ValueError("Descriptors and evidence must contain finite JSON values")
    seen = set() if seen is None else seen
    if id(value) in seen:
        raise ValueError("Cyclic JSON values are not supported")
    seen.add(id(value))
    try:
        if isinstance(value, Mapping):
            if any(not isinstance(key, str) for key in value):
                raise ValueError("JSON object keys must be strings")
            return MappingProxyType({key: _freeze_json(item, seen) for key, item in value.items()})
        return tuple(_freeze_json(item, seen) for item in value)
    finally:
        seen.remove(id(value))


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return _freeze_json(value)


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    features: tuple[float, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _identity(self.task_id, "task_id")
        if isinstance(self.features, (str, bytes)):
            raise ValueError("features must be a finite numeric sequence")
        try:
            features = tuple(_number(value, "feature") for value in self.features)
        except TypeError as exc:
            raise ValueError("features must be a finite numeric sequence") from exc
        if not features:
            raise ValueError("features must be nonempty")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "metadata", _object(self.metadata, "metadata"))

    def to_dict(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "features": list(self.features), "metadata": _thaw_json(self.metadata)}


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    config: Mapping[str, Any]

    def __post_init__(self) -> None:
        _identity(self.candidate_id, "candidate_id")
        object.__setattr__(self, "config", _object(self.config, "config"))

    def to_dict(self) -> dict[str, Any]:
        return {"candidate_id": self.candidate_id, "config": _thaw_json(self.config)}


@dataclass(frozen=True)
class TrialResult:
    trial_id: str
    task_id: str
    candidate_id: str | None
    backend: str
    cohort_id: str
    input_digest: str
    status: str
    correct: bool
    latency_ms: float | None
    peak_allocated_bytes: int | None
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("trial_id", "task_id", "cohort_id", "input_digest"):
            _identity(getattr(self, name), name)
        if self.candidate_id is not None:
            _identity(self.candidate_id, "candidate_id")
        if self.backend not in ("cuda", "synthetic"):
            raise ValueError("backend must be cuda or synthetic")
        if self.status not in ("ok", "incorrect", "error"):
            raise ValueError("unknown trial status")
        if type(self.correct) is not bool or self.correct != (self.status == "ok"):
            raise ValueError("correct must be true exactly when status is ok")
        if self.latency_ms is not None:
            object.__setattr__(self, "latency_ms", _number(self.latency_ms, "latency_ms", positive=True))
        if self.peak_allocated_bytes is not None:
            value = self.peak_allocated_bytes
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError("peak_allocated_bytes must be a nonnegative integer")
            object.__setattr__(self, "peak_allocated_bytes", int(value))
        if self.status == "ok" and (self.latency_ms is None or self.peak_allocated_bytes is None):
            raise ValueError("successful trials require latency and allocation metrics")
        object.__setattr__(self, "evidence", _object(self.evidence, "evidence"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id, "task_id": self.task_id,
            "candidate_id": self.candidate_id, "backend": self.backend,
            "cohort_id": self.cohort_id, "input_digest": self.input_digest,
            "status": self.status, "correct": self.correct,
            "latency_ms": self.latency_ms, "peak_allocated_bytes": self.peak_allocated_bytes,
            "evidence": _thaw_json(self.evidence),
        }


class SearchEvaluator(Protocol):
    def baseline(self, task: TaskSpec) -> TrialResult: ...

    def evaluate(self, task: TaskSpec, candidate: CandidateSpec) -> TrialResult: ...


class BudgetedTuningEnv(gym.Env[dict[str, Any], int]):
    """Reusable search interface with a correctness-constrained incumbent.

    Observation keys: ``task_features`` (F floats), ``budget`` (remaining
    fraction, shape1), ``incumbent`` (latency/allocation ratios, shape2),
    ``incumbent_action`` (0 reference, otherwise action index+1),
    ``action_counts`` (N counts), ``action_status`` (N:0untried,1ok,2incorrect,
    3error), ``action_metrics`` (N×2 latest successful latency/allocation
    ratios, zeros for untried/failed). Status distinguishes missing metrics
    from a valid zero allocation ratio. Failed trials clear their latest
    metrics; earlier outcomes remain in the journal.

    Reset accepts ``options={'task_index': i}``; absent that, task selection
    uses Gymnasium's seeded RNG. Every reset calls the reference evaluator,
    every step calls the candidate evaluator, and repeat actions are fresh
    calls. A returned incorrect/error result consumes a trial and earns zero.
    Protocol violations or evaluator exceptions preserve the attempt and
    require reset; they are never fabricated into measured GPU results.

    Budget exhaustion is a task terminal state (``terminated=True``), not an
    external time-limit truncation. No additional reward is paid at terminal:
    summed improvements telescope to 1 minus final incumbent cost. A finite
    budget does not interrupt a hung external evaluator; execution timeouts
    and resource cleanup belong to its caller.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        tasks: Sequence[TaskSpec],
        candidates: Sequence[CandidateSpec],
        evaluator: SearchEvaluator,
        trial_budget: int = 4,
        memory_weight: float = 0.0,
        allow_synthetic: bool = False,
    ) -> None:
        super().__init__()
        if not tasks or any(not isinstance(task, TaskSpec) for task in tasks):
            raise ValueError("tasks must be a nonempty sequence of TaskSpec")
        if not candidates or any(not isinstance(candidate, CandidateSpec) for candidate in candidates):
            raise ValueError("candidates must be a nonempty sequence of CandidateSpec")
        if len({task.task_id for task in tasks}) != len(tasks):
            raise ValueError("task IDs must be unique")
        if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
            raise ValueError("candidate IDs must be unique")
        if len({len(task.features) for task in tasks}) != 1:
            raise ValueError("tasks must have the same feature dimension")
        if isinstance(trial_budget, bool) or not isinstance(trial_budget, Integral) or not 1 <= trial_budget < 2**63 - 1:
            raise ValueError("trial_budget must be a positive int64-compatible integer")
        weight = _number(memory_weight, "memory_weight")
        if weight < 0:
            raise ValueError("memory_weight must be nonnegative")
        if type(allow_synthetic) is not bool:
            raise ValueError("allow_synthetic must be boolean")
        self.tasks, self.candidates = tuple(tasks), tuple(candidates)
        self.evaluator = evaluator
        self.trial_budget = int(trial_budget)
        self.memory_weight = weight
        self.allow_synthetic = allow_synthetic
        count = len(candidates)
        self.action_space = spaces.Discrete(count)
        self.observation_space = spaces.Dict({
            "task_features": spaces.Box(-np.inf, np.inf, (len(tasks[0].features),), np.float64),
            "budget": spaces.Box(0.0, 1.0, (1,), np.float64),
            "incumbent": spaces.Box(0.0, np.inf, (2,), np.float64),
            "incumbent_action": spaces.Discrete(count + 1),
            "action_counts": spaces.MultiDiscrete(np.full(count, self.trial_budget + 1, dtype=np.int64)),
            "action_status": spaces.MultiDiscrete(np.full(count, 4, dtype=np.int64)),
            "action_metrics": spaces.Box(0.0, np.inf, (count, 2), np.float64),
        })
        self._journal: list[dict[str, Any]] = []
        self._seen_trial_ids: set[str] = set()
        self._cohort_binding: tuple[str, str] | None = None
        self._task_input_digests: dict[str, str] = {}
        self._needs_reset = True
        self._closed = False
        self._cleanup_complete = False
        self._task_index: int | None = None
        self._baseline: TrialResult | None = None
        self._incumbent: TrialResult | None = None
        self._trials = 0
        self._best_cost = 1.0
        self._incumbent_action = 0
        self._counts = np.zeros(count, dtype=np.int64)
        self._statuses = np.zeros(count, dtype=np.int64)
        self._metrics = np.zeros((count, 2), dtype=np.float64)

    @property
    def journal(self) -> list[dict[str, Any]]:
        """Detached, JSON-serializable completed and partial attempt records."""
        return deepcopy(self._journal)

    @property
    def current_task(self) -> TaskSpec:
        if self._task_index is None:
            raise RuntimeError("call reset before accessing current_task")
        return self.tasks[self._task_index]

    def _validate(self, result: TrialResult, candidate_id: str | None) -> None:
        if not isinstance(result, TrialResult):
            raise ValueError("evaluator must return TrialResult")
        if result.trial_id in self._seen_trial_ids:
            raise ValueError("trial_id must identify a fresh result, including across resets")
        self._seen_trial_ids.add(result.trial_id)
        if result.task_id != self.current_task.task_id or result.candidate_id != candidate_id:
            raise ValueError("trial task/candidate identity does not match the requested evaluation")
        if result.backend == "synthetic" and not self.allow_synthetic:
            raise ValueError("synthetic results require explicit allow_synthetic=True")
        binding = (result.backend, result.cohort_id)
        if self._cohort_binding is not None and binding != self._cohort_binding:
            raise ValueError("backend and physical/runtime cohort must stay fixed")
        expected = self._task_input_digests.get(result.task_id)
        if expected is not None and result.input_digest != expected:
            raise ValueError("input_digest must stay fixed for a task")
        if candidate_id is None:
            if result.status != "ok" or not result.correct or not result.peak_allocated_bytes:
                raise ValueError("reference must be correct with positive latency and allocation")
            self._cohort_binding = binding
            self._task_input_digests[result.task_id] = result.input_digest

    def _call(self, candidate: CandidateSpec | None, action: int | None = None) -> TrialResult:
        record: dict[str, Any] = {
            "kind": "baseline" if candidate is None else "candidate",
            "task_id": self.current_task.task_id,
            "candidate_id": None if candidate is None else candidate.candidate_id,
            "accepted": False,
        }
        if candidate is not None:
            record.update(action=action, trials_used=self._trials)
        self._journal.append(record)
        try:
            result = (self.evaluator.baseline(self.current_task) if candidate is None
                      else self.evaluator.evaluate(self.current_task, candidate))
        except Exception as exc:
            record["error"] = {"type": type(exc).__name__, "message": str(exc)}
            self._needs_reset = True
            raise
        if isinstance(result, TrialResult):
            record["measurement"] = result.to_dict()
        else:
            record["returned_type"] = type(result).__name__
        try:
            self._validate(result, record["candidate_id"])
            if candidate is not None and result.status == "ok":
                self._ratios_and_cost(result)
        except Exception as exc:
            record["validation_error"] = {"type": type(exc).__name__, "message": str(exc)}
            self._needs_reset = True
            raise
        record["accepted"] = True
        return result

    def _ratios_and_cost(self, result: TrialResult) -> tuple[tuple[float, float], float]:
        assert self._baseline is not None
        assert self._baseline.latency_ms is not None and self._baseline.peak_allocated_bytes
        assert result.latency_ms is not None and result.peak_allocated_bytes is not None
        try:
            latency = result.latency_ms / self._baseline.latency_ms
            memory = result.peak_allocated_bytes / self._baseline.peak_allocated_bytes
            # Weighted mean without multiplying a very large user weight first.
            latency_weight = 1.0 / (1.0 + self.memory_weight)
            cost = latency * latency_weight + memory * (self.memory_weight * latency_weight)
        except (OverflowError, ZeroDivisionError) as exc:
            raise ValueError("normalized metrics must remain finite") from exc
        if not all(math.isfinite(value) for value in (latency, memory, cost)):
            raise ValueError("normalized metrics must remain finite")
        return (latency, memory), cost

    def _observation(self) -> dict[str, Any]:
        assert self._incumbent is not None
        ratios, _ = self._ratios_and_cost(self._incumbent)
        return {
            "task_features": np.asarray(self.current_task.features, dtype=np.float64),
            "budget": np.asarray([(self.trial_budget - self._trials) / self.trial_budget], dtype=np.float64),
            "incumbent": np.asarray(ratios, dtype=np.float64),
            "incumbent_action": self._incumbent_action,
            "action_counts": self._counts.copy(),
            "action_status": self._statuses.copy(),
            "action_metrics": self._metrics.copy(),
        }

    def _info(self, measurement: TrialResult | None = None) -> dict[str, Any]:
        assert self._baseline is not None and self._incumbent is not None
        return {
            "task_id": self.current_task.task_id, "task_index": self._task_index,
            "trials_used": self._trials, "trial_budget": self.trial_budget,
            "baseline": self._baseline.to_dict(), "incumbent": self._incumbent.to_dict(),
            "incumbent_candidate_id": self._incumbent.candidate_id,
            "incumbent_cost": self._best_cost,
            "measurement": None if measurement is None else measurement.to_dict(),
            "backend": self._baseline.backend, "cohort_id": self._baseline.cohort_id,
            "input_digest": self._baseline.input_digest,
            "synthetic": self._baseline.backend == "synthetic",
            "objective": {
                "cost": "(latency/reference_latency + memory_weight*allocation/reference_allocation)/(1+memory_weight)",
                "memory_weight": self.memory_weight,
                "reward": "previous_incumbent_cost - current_incumbent_cost",
                "selection": "best observed admissible cost including reference",
                "independent_confirmation": False,
                "quality_gate": "external evaluator assertion; inspect its supporting evidence",
            },
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._closed:
            raise RuntimeError("environment is closed")
        super().reset(seed=seed)
        self._needs_reset = True
        if options is not None and (not isinstance(options, dict) or set(options) - {"task_index"}):
            raise ValueError("reset options support only task_index")
        options = options or {}
        index = options["task_index"] if "task_index" in options else int(self.np_random.integers(len(self.tasks)))
        if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < len(self.tasks):
            raise ValueError("task_index is out of range")
        self._task_index = int(index)
        self._trials = 0
        self._best_cost = 1.0
        self._incumbent_action = 0
        self._baseline = self._incumbent = None
        self._counts.fill(0)
        self._statuses.fill(0)
        self._metrics.fill(0)
        result = self._call(None)
        self._baseline = self._incumbent = result
        self._needs_reset = False
        return self._observation(), self._info()

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self._needs_reset or self._closed:
            raise RuntimeError("call reset before step, including after episode completion")
        if isinstance(action, bool) or not isinstance(action, Integral) or not self.action_space.contains(action):
            raise ValueError("action must be a valid candidate index")
        action = int(action)
        self._trials += 1
        self._counts[action] += 1
        result = self._call(self.candidates[action], action)
        self._statuses[action] = {"ok": 1, "incorrect": 2, "error": 3}[result.status]
        self._metrics[action].fill(0)
        reward = 0.0
        if result.status == "ok":
            ratios, cost = self._ratios_and_cost(result)
            self._metrics[action] = ratios
            if cost < self._best_cost:
                reward = self._best_cost - cost
                self._best_cost = cost
                self._incumbent = result
                self._incumbent_action = action + 1
        terminated = self._trials == self.trial_budget
        self._needs_reset = terminated
        return self._observation(), float(reward), terminated, False, self._info(result)

    def close(self) -> None:
        if not self._cleanup_complete:
            self._closed = True
            self._needs_reset = True
            close = getattr(self.evaluator, "close", None)
            if callable(close):
                close()
            # A failed close leaves evaluation disabled but permits an explicit
            # cleanup retry. Only successful cleanup is idempotently skipped.
            self._cleanup_complete = True
