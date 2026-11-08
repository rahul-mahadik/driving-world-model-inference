"""A measured, finite-budget kernel configuration search environment.

This environment tunes execution configurations; it does not simulate driving.
Each step performs a new evaluator call.  No cached measurement is presented as
a fresh GPU trial.  The reward is a bounded reduction in a weighted latency /
peak-allocation cost relative to the episode's eager baseline.  Correctness is a
hard constraint: an incorrect or failed trial cannot become the incumbent.

The observation contains workload features and the search state.  Workload
features describe the operator, dtype, rank, dimensions, grouping and epsilon.
Search features contain remaining budget, the incumbent, and each action's
trial count, latest reward and outcome.  Hardware/runtime stationarity is the
evaluator's responsibility; a benchmark run must bind those in its evidence.
"""

from __future__ import annotations

from dataclasses import asdict
import math
from numbers import Integral, Real
import re
from typing import Any, Protocol, Sequence

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .contract import KernelConfig, Measurement, Workload


class Evaluator(Protocol):
    def baseline(self, workload: Workload) -> Measurement: ...

    def evaluate(self, workload: Workload, config: KernelConfig) -> Measurement: ...


class KernelTuningEnv(gym.Env[np.ndarray, int]):
    """Search a finite set of launch configurations using benchmark feedback.

    `trial_budget` counts candidate calls, including failed/incorrect calls.
    There is also one baseline call per reset.  Repeated actions are measured
    again and consume budget.  Episodes truncate at the budget; there is no
    fabricated terminal success condition.  Synthetic evaluators are allowed
    only with an explicit opt-in intended for tests and interface demos.
    """

    metadata = {"render_modes": []}
    FAILURE_REWARD = -2.0
    BASE_FEATURES = 20

    def __init__(
        self,
        workloads: Sequence[Workload],
        configs: Sequence[KernelConfig],
        evaluator: Evaluator,
        trial_budget: int = 8,
        memory_weight: float = 0.25,
        allow_synthetic: bool = False,
    ) -> None:
        super().__init__()
        if not workloads or not configs:
            raise ValueError("workloads and configs must both be nonempty")
        if isinstance(trial_budget, bool) or not isinstance(trial_budget, Integral) or trial_budget < 1:
            raise ValueError("trial_budget must be a positive integer")
        if not isinstance(memory_weight, Real) or not math.isfinite(memory_weight) or memory_weight < 0:
            raise ValueError("memory_weight must be finite and nonnegative")
        if len({(c.block_size, c.num_warps) for c in configs}) != len(configs):
            raise ValueError("duplicate action configurations are not allowed")
        if len({w.key() for w in workloads}) != len(workloads):
            raise ValueError("duplicate workloads are not allowed")
        for workload in workloads:
            if workload.operator not in {"geglu", "groupnorm_silu"}:
                raise ValueError(f"unsupported observation operator: {workload.operator}")
            if workload.dtype not in {"float16", "float32", "bfloat16"}:
                raise ValueError(f"unsupported observation dtype: {workload.dtype}")
            if not 1 <= len(workload.shape) <= 8 or any(d <= 0 for d in workload.shape):
                raise ValueError("workloads must have 1 to 8 positive dimensions")
            if workload.num_groups < 1 or not math.isfinite(workload.eps) or workload.eps <= 0:
                raise ValueError("num_groups and eps must be positive")
        self.workloads = tuple(workloads)
        self.configs = tuple(configs)
        self.evaluator = evaluator
        self.trial_budget = int(trial_budget)
        self.memory_weight = float(memory_weight)
        self.allow_synthetic = bool(allow_synthetic)
        self.action_space = spaces.Discrete(len(configs))
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.BASE_FEATURES + 4 * len(configs),), dtype=np.float32,
        )
        self._needs_reset = True
        self._closed = False
        # This journal intentionally survives reset: it retains every completed
        # baseline and candidate result, including failed/partial observations.
        self.journal: list[dict[str, Any]] = []
        self._seen_trial_ids: set[str] = set()
        self.history: list[Measurement] = []
        self.baseline_measurement: Measurement | None = None
        self.incumbent: Measurement | None = None
        self.workload_index: int | None = None

    @property
    def current_workload(self) -> Workload:
        if self.workload_index is None:
            raise RuntimeError("call reset before accessing the current workload")
        return self.workloads[self.workload_index]

    def _validate(self, result: Measurement, expected: KernelConfig | None) -> None:
        if not isinstance(result, Measurement):
            raise ValueError("evaluator must return a Measurement")
        if result.workload_key != self.current_workload.key():
            raise ValueError("measurement belongs to a different workload")
        if result.config != expected:
            raise ValueError("measurement config does not match the requested action")
        allowed = {"cuda", "synthetic"} if self.allow_synthetic else {"cuda"}
        if result.backend not in allowed:
            raise ValueError("real CUDA measurement required; synthetic fixtures require explicit opt-in")
        if result.status not in {"ok", "incorrect", "error"}:
            raise ValueError("unknown measurement status")
        if not isinstance(result.correct, bool):
            raise ValueError("correct must be a boolean")
        if result.status == "ok" and not result.correct:
            raise ValueError("ok status requires a correct result")
        if result.status != "ok" and result.correct:
            raise ValueError("failed/incorrect status cannot claim correctness")
        if result.latency_ms is not None:
            if isinstance(result.latency_ms, bool) or not isinstance(result.latency_ms, Real) or not math.isfinite(result.latency_ms) or result.latency_ms <= 0:
                raise ValueError("latency_ms must be finite and positive")
        if result.peak_allocated_bytes is not None:
            if isinstance(result.peak_allocated_bytes, bool) or not isinstance(result.peak_allocated_bytes, Integral) or result.peak_allocated_bytes < 0:
                raise ValueError("peak_allocated_bytes must be a nonnegative integer")
        if result.status == "ok" and (result.latency_ms is None or result.peak_allocated_bytes is None):
            raise ValueError("successful measurements require latency and memory metrics")
        if not isinstance(result.evidence, dict):
            raise ValueError("measurement evidence must be a dictionary")
        if result.backend == "cuda":
            cohort_id = result.evidence.get("cohort_id")
            hashes = result.evidence.get("input_sha256")
            if not isinstance(cohort_id, str) or not cohort_id:
                raise ValueError("CUDA evidence requires a nonempty cohort_id")
            if not isinstance(hashes, list) or not hashes or any(
                not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None for value in hashes
            ):
                raise ValueError("CUDA evidence requires a nonempty input_sha256 list of SHA-256 hashes")
            if not isinstance(result.evidence.get("trial_id"), str) or not result.evidence["trial_id"]:
                raise ValueError("CUDA evidence requires a unique nonempty trial_id")
        trial_id = result.evidence.get("trial_id")
        if trial_id is not None:
            if not isinstance(trial_id, str) or not trial_id or trial_id in self._seen_trial_ids:
                raise ValueError("measurement trial_id must identify a new trial")
            self._seen_trial_ids.add(trial_id)

    def _ratios(self, result: Measurement) -> tuple[float, float]:
        baseline = self.baseline_measurement
        assert baseline is not None and baseline.latency_ms is not None
        assert baseline.peak_allocated_bytes is not None and baseline.peak_allocated_bytes > 0
        assert result.latency_ms is not None and result.peak_allocated_bytes is not None
        return result.latency_ms / baseline.latency_ms, result.peak_allocated_bytes / baseline.peak_allocated_bytes

    def _cost(self, result: Measurement) -> float:
        latency_ratio, memory_ratio = self._ratios(result)
        return (latency_ratio + self.memory_weight * memory_ratio) / (1.0 + self.memory_weight)

    def _observation(self) -> np.ndarray:
        workload = self.current_workload
        assert self.incumbent is not None
        latency_ratio, memory_ratio = self._ratios(self.incumbent)
        features = [
            float(workload.operator == "geglu"), float(workload.operator == "groupnorm_silu"),
            float(workload.dtype == "float16"), float(workload.dtype == "float32"),
            float(workload.dtype == "bfloat16"), len(workload.shape) / 8.0,
        ]
        features += [min(math.log2(d) / 32.0, 1.0) for d in workload.shape]
        features += [0.0] * (8 - len(workload.shape))
        features += [
            min(math.log2(workload.num_groups) / 16.0, 1.0),
            float(np.clip(math.log10(workload.eps) / 20.0, -1.0, 1.0)),
            (self.trial_budget - self._trials) / self.trial_budget,
            latency_ratio / (1.0 + latency_ratio), memory_ratio / (1.0 + memory_ratio),
            (self._incumbent_action + 1) / len(self.configs),
        ]
        for index in range(len(self.configs)):
            features += [
                self._counts[index] / self.trial_budget,
                self._latest_rewards[index] / 2.0,
                float(self._latest_status[index] == "ok"),
                float(self._latest_status[index] in {"incorrect", "error"}),
            ]
        return np.asarray(features, dtype=np.float32)

    def _info(self, measurement: Measurement | None = None) -> dict[str, Any]:
        assert self.baseline_measurement is not None and self.incumbent is not None
        return {
            "workload_key": self.current_workload.key(),
            "workload_index": self.workload_index,
            "trials_used": self._trials,
            "trial_budget": self.trial_budget,
            "baseline_calls": 1,
            "measurement": asdict(measurement) if measurement is not None else None,
            "baseline": asdict(self.baseline_measurement),
            "incumbent": asdict(self.incumbent),
            "incumbent_action": self._incumbent_action,
            "incumbent_cost": self._best_cost,
            "memory_weight": self.memory_weight,
            "synthetic": self.baseline_measurement.backend == "synthetic",
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if self._closed:
            raise RuntimeError("environment is closed")
        super().reset(seed=seed)
        self._needs_reset = True
        options = options or {}
        if set(options) - {"workload_index"}:
            raise ValueError("only workload_index is supported in reset options")
        index = options.get("workload_index", int(self.np_random.integers(len(self.workloads))))
        if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < len(self.workloads):
            raise ValueError("workload_index is out of range")
        self.workload_index = int(index)
        self._trials = 0
        self._counts = np.zeros(len(self.configs), dtype=np.int64)
        self._latest_rewards = np.zeros(len(self.configs), dtype=np.float64)
        self._latest_status = [None] * len(self.configs)
        self.history = []
        self.baseline_measurement = None
        self.incumbent = None
        try:
            result = self.evaluator.baseline(self.current_workload)
        except Exception as exc:
            self.journal.append({"kind": "baseline", "workload_key": self.current_workload.key(),
                                 "evaluator_error": {"type": type(exc).__name__, "message": str(exc)}})
            raise
        self.journal.append({"kind": "baseline", "workload_key": self.current_workload.key(), "measurement": result})
        self._validate(result, None)
        if result.status != "ok" or not result.correct or result.peak_allocated_bytes == 0:
            raise ValueError("baseline must be correct with positive latency and peak allocated memory")
        self.baseline_measurement = self.incumbent = result
        self._best_cost = 1.0
        self._incumbent_action = -1
        self._needs_reset = False
        return self._observation(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._needs_reset or self._closed:
            raise RuntimeError("call reset before step, including after a truncated episode")
        if isinstance(action, bool) or not isinstance(action, Integral) or not self.action_space.contains(action):
            raise ValueError("action is not a valid configuration index")
        action = int(action)
        try:
            result = self.evaluator.evaluate(self.current_workload, self.configs[action])
        except Exception as exc:
            self._needs_reset = True
            self.journal.append({"kind": "candidate", "workload_key": self.current_workload.key(), "action": action,
                                 "evaluator_error": {"type": type(exc).__name__, "message": str(exc)}})
            raise
        # Retain even a rejected evaluator response.  A rejected response leaves
        # the episode unusable until reset instead of silently reusing state.
        self.journal.append({"kind": "candidate", "workload_key": self.current_workload.key(), "action": action, "measurement": result})
        try:
            self._validate(result, self.configs[action])
            assert self.baseline_measurement is not None
            if result.backend != self.baseline_measurement.backend:
                raise ValueError("baseline and candidate backends must match")
            baseline_cohort = self.baseline_measurement.evidence.get("cohort_id")
            candidate_cohort = result.evidence.get("cohort_id")
            if baseline_cohort != candidate_cohort:
                raise ValueError("baseline and candidate must belong to the same hardware/runtime cohort")
            if self.baseline_measurement.evidence.get("input_sha256") != result.evidence.get("input_sha256"):
                raise ValueError("baseline and candidate input hashes must match")
        except ValueError:
            self._needs_reset = True
            raise
        self.history.append(result)
        self._trials += 1
        self._counts[action] += 1
        self._latest_status[action] = result.status
        reward = self.FAILURE_REWARD
        if result.status == "ok":
            cost = self._cost(result)
            reward = float(np.clip(1.0 - cost, -1.0, 1.0))
            if cost < self._best_cost:
                self._best_cost = cost
                self.incumbent = result
                self._incumbent_action = action
        self._latest_rewards[action] = reward
        truncated = self._trials >= self.trial_budget
        self._needs_reset = truncated
        return self._observation(), reward, False, truncated, self._info(result)

    def close(self) -> None:
        if not self._closed:
            close = getattr(self.evaluator, "close", None)
            if callable(close):
                close()
            self._closed = True
            self._needs_reset = True
