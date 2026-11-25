"""Sequential edits of fixed CUDA operator schedules under an elapsed-time budget.

This experimental live track does not reuse the historical table as outcomes.
The evaluator is trusted code; this is not isolation for arbitrary agent code.
Only submission earns a speed score, after a new correctness-checked trial.
"""
from __future__ import annotations

from dataclasses import asdict
import copy
import json
import math
from numbers import Integral, Real
from pathlib import Path
import re
import time
import uuid

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .contract import Workload
from .schedule import ACTIONS, BLOCKS, WARPS, action_mask, initial_schedule, transform


class ScheduleTuningEnv(gym.Env):
    """Edit, measure, and submit a persistent schedule for one operator.

    Budget includes reference setup, compilation, correctness checks, all trials,
    failed calls, and time between actions. Each edit/invalid action costs at
    least 1 ms. GPU call timeouts are enforced by the owned evaluator process;
    close/reap overhead can exceed the measurement deadline and is recorded.

    Reward = terminal clip(1 - candidate_latency / baseline_latency, -1, 1),
    minus 0.05 * elapsed_fraction, 0.01 per illegal edit, and 0.1 per failed
    measurement. No score is paid on timeout or a failed submission. Timings are
    fresh sampled CUDA-event observations, so a successful check is no promise
    that a speedup exceeds the noise floor. Keep raw evidence and repeat studies.
    """
    metadata = {"render_modes": []}
    action_names = ACTIONS
    MIN_ACTION_SECONDS = 0.001

    def __init__(self, *, workloads, budget_seconds=120.0, evidence_dir=None,
                 trial_timeout_seconds=60.0, evaluator=None, allow_test_evaluator=False,
                 render_mode=None):
        super().__init__()
        if render_mode is not None:
            raise ValueError("KernelSchedule-v0 does not support rendering")
        if not workloads or any(not isinstance(w, Workload) for w in workloads):
            raise ValueError("workloads must be a nonempty sequence of Workload objects")
        if len({w.key() for w in workloads}) != len(workloads):
            raise ValueError("duplicate workloads")
        for workload in workloads:
            initial_schedule(workload)
        for value in (budget_seconds, trial_timeout_seconds):
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                raise ValueError("budgets and timeouts must be finite positive seconds")
        if evaluator is None and evidence_dir is None:
            raise ValueError("live CUDA execution requires a new evidence_dir")
        if allow_test_evaluator and evaluator is None:
            raise ValueError("test opt-in requires an explicit test evaluator")
        self.workloads = tuple(workloads)
        self.budget_seconds = float(budget_seconds)
        self.trial_timeout_seconds = float(trial_timeout_seconds)
        self._external_evaluator = evaluator
        self.evaluator = evaluator
        self.allow_test_evaluator = bool(allow_test_evaluator)
        self.evidence_dir = Path(evidence_dir) if evidence_dir is not None else None
        if self.evidence_dir is not None:
            self.evidence_dir.mkdir(parents=True, exist_ok=False)
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Dict({
            "schedule": spaces.MultiDiscrete([2, 7, 4, 7, 4, 2, 7, 4]),
            "action_mask": spaces.MultiBinary(len(ACTIONS)),
            "workload": spaces.Box(0, 1, (12,), dtype=np.float32),
            "hardware": spaces.Box(0, 1, (5,), dtype=np.float32),
            "graph_nodes": spaces.Box(0, 1, (3, 6), dtype=np.float32),
            "graph_edges": spaces.MultiBinary((3, 3)),
            "measurement": spaces.Box(0, 1, (6,), dtype=np.float32),
            "remaining_budget": spaces.Box(0, 1, (1,), dtype=np.float32),
        })
        self._closed = False
        self._needs_reset = True
        self._seen_trials = set()
        self.journal = []
        self._log_path = None
        self.render_mode = None

    def _record(self, row):
        row = copy.deepcopy(row)
        self.journal.append(row)
        if self._log_path is not None:
            with self._log_path.open("a") as stream:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")

    def _elapsed(self):
        return max(self._charged, time.monotonic() - self._started)

    def _remaining(self):
        return max(0.0, self.budget_seconds - self._elapsed())

    def _validate(self, result, schedule):
        from .schedule_measurement import ScheduleMeasurement
        if not isinstance(result, ScheduleMeasurement):
            raise ValueError("evaluator must return ScheduleMeasurement")
        if result.workload_key != self.current_workload.key() or result.schedule != schedule:
            raise ValueError("measurement workload/schedule binding mismatch")
        allowed = {"cuda", "test"} if self.allow_test_evaluator else {"cuda"}
        if result.backend not in allowed:
            raise ValueError("live CUDA measurement required; tests require explicit opt-in")
        if result.status not in ("ok", "incorrect", "error", "timeout") or type(result.correct) is not bool:
            raise ValueError("invalid measurement outcome")
        if (result.status == "ok") != result.correct:
            raise ValueError("status/correctness disagreement")
        if (isinstance(result.elapsed_seconds, bool) or not isinstance(result.elapsed_seconds, Real)
                or not math.isfinite(result.elapsed_seconds) or result.elapsed_seconds < 0):
            raise ValueError("invalid elapsed time")
        if not isinstance(result.evidence, dict):
            raise ValueError("evidence must be a dictionary")
        trial = result.evidence.get("trial_id")
        if not isinstance(trial, str) or not trial or trial in self._seen_trials:
            raise ValueError("each measurement requires a unique trial_id")
        self._seen_trials.add(trial)
        if result.status == "ok":
            samples = result.raw_samples_ms
            if not isinstance(samples, (list, tuple)) or not samples or any(
                isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x) or x <= 0 for x in samples
            ):
                raise ValueError("successful trials require finite positive raw timing samples")
            if isinstance(result.latency_ms, bool) or result.latency_ms not in samples:
                raise ValueError("latency must be one of this trial's fresh samples")
            if type(result.peak_allocated_bytes) is not int or result.peak_allocated_bytes < 0:
                raise ValueError("successful trials require nonnegative peak allocation")
            if result.backend == "cuda":
                runtime = result.evidence.get("runtime")
                hashes = result.evidence.get("input_sha256")
                if not isinstance(runtime, dict) or not runtime.get("gpu_uuid"):
                    raise ValueError("physical GPU/runtime binding required")
                if not isinstance(hashes, list) or not hashes or any(
                    not isinstance(h, str) or not re.fullmatch("[0-9a-f]{64}", h) for h in hashes
                ):
                    raise ValueError("input hash binding required")
                if not result.evidence.get("cohort_id"):
                    raise ValueError("runtime cohort binding required")
        if self.baseline is not None and result.status == "ok":
            for key in ("runtime", "cohort_id", "input_sha256"):
                if result.evidence.get(key) != self.baseline.evidence.get(key):
                    raise ValueError(f"episode {key} changed")

    def _observe(self):
        w, s = self.current_workload, self.schedule
        schedule = [int(s.fused), BLOCKS.index(s.block_size), WARPS.index(s.num_warps),
                    BLOCKS.index(s.activation_block_size), WARPS.index(s.activation_num_warps),
                    int(s.reduction == "single"), BLOCKS.index(s.reduction_block_size),
                    WARPS.index(s.reduction_num_warps)]
        dims = [min(math.log2(d) / 32, 1) for d in w.shape] + [0.0] * (4 - len(w.shape))
        workload = [float(w.operator == "geglu"), float(w.operator == "groupnorm_silu")]
        workload += [float(w.dtype == d) for d in ("float16", "bfloat16", "float32")]
        workload += dims + [len(w.shape) / 4, min(math.log2(w.num_groups) / 32, 1),
                            min(abs(math.log10(w.eps)) / 32, 1)]
        runtime = self.baseline.evidence.get("runtime", {})
        cc = runtime.get("compute_capability", [0, 0])
        hardware = [min(float(cc[0]) / 20, 1), min(float(cc[1]) / 10, 1),
                    min(float(runtime.get("total_memory", 0)) / 2**40, 1),
                    min(float(runtime.get("multiprocessor_count", 0)) / 1024, 1),
                    float(cc[0] >= 8)]
        # Fixed semantic graph, with stage assignments from the current lowering.
        # Columns: GELU, multiply, statistics, affine, SiLU, stage/2.
        nodes = np.zeros((3, 6), dtype=np.float32)
        edges = np.zeros((3, 3), dtype=np.int8)
        if w.operator == "geglu":
            nodes[0, 0] = nodes[1, 1] = 1
            nodes[1, 5] = 0 if s.fused else 0.5
            edges[0, 1] = 1
        else:
            nodes[0, 2] = nodes[1, 3] = nodes[2, 4] = 1
            nodes[1, 5] = 0.5
            nodes[2, 5] = 0.5 if s.fused else 1
            edges[0, 1] = edges[1, 2] = 1
        metrics = np.zeros(6, dtype=np.float32)
        if self.last_measurement is not None and self.last_measurement.correct:
            result = self.last_measurement
            ratio = result.latency_ms / self.baseline.latency_ms
            mean = float(np.mean(result.raw_samples_ms))
            cv = float(np.std(result.raw_samples_ms)) / mean
            memory_ratio = result.peak_allocated_bytes / max(1, self.baseline.peak_allocated_bytes)
            metrics[:] = [ratio / (1 + ratio), memory_ratio / (1 + memory_ratio), cv / (1 + cv),
                          len(result.raw_samples_ms) / (1 + len(result.raw_samples_ms)), 1, 1]
        elif self.last_measurement is not None:
            metrics[-2] = 1
        mask = action_mask(w, s) if not self._needs_reset else [0] * len(ACTIONS)
        return {"schedule": np.asarray(schedule, dtype=np.int64),
                "action_mask": np.asarray(mask, dtype=np.int8),
                "workload": np.asarray(workload, dtype=np.float32),
                "hardware": np.asarray(hardware, dtype=np.float32),
                "graph_nodes": nodes, "graph_edges": edges, "measurement": metrics,
                "remaining_budget": np.asarray([self._remaining() / self.budget_seconds], dtype=np.float32)}

    def _info(self, *, outcome=None, score=0.0):
        return {"track": "sequential-schedule-v0", "workload": asdict(self.current_workload),
                "schedule": asdict(self.schedule), "action_names": ACTIONS,
                "elapsed_seconds": self._elapsed(), "budget_seconds": self.budget_seconds,
                "measurement": asdict(self.last_measurement) if self.last_measurement else None,
                "baseline": asdict(self.baseline), "outcome": outcome, "submission_score": score,
                "reward_components": dict(self._reward_components),
                "fresh_gpu_measurement": self.baseline.backend == "cuda",
                "test_fixture": self.baseline.backend == "test",
                "evidence_directory": str(self._episode_dir) if self._episode_dir else None}

    def reset(self, *, seed=None, options=None):
        if self._closed:
            raise RuntimeError("environment is closed")
        super().reset(seed=seed)
        self._needs_reset = True
        options = options or {}
        if set(options) - {"workload_index"}:
            raise ValueError("only workload_index is supported")
        index = options.get("workload_index", int(self.np_random.integers(len(self.workloads))))
        if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < len(self.workloads):
            raise ValueError("workload_index out of range")
        self.current_workload = self.workloads[int(index)]
        self.schedule = initial_schedule(self.current_workload)
        self.baseline = self.last_measurement = None
        self._charged = self._last_reward_elapsed = 0.0
        self._reward_components = {"speed_score": 0.0, "elapsed_penalty": 0.0,
                                   "invalid_action_penalty": 0.0, "failed_measurement_penalty": 0.0}
        self._started = time.monotonic()
        self._episode_dir = None
        self._log_path = None
        if self.evidence_dir:
            self._episode_dir = self.evidence_dir / ("episode-" + uuid.uuid4().hex)
            self._episode_dir.mkdir()
            self._log_path = self._episode_dir / "environment.jsonl"
        if self._external_evaluator is None:
            if self.evaluator is not None:
                self.evaluator.close()
            from .schedule_measurement import CudaScheduleEvaluator
            self.evaluator = CudaScheduleEvaluator(evidence_directory=self._episode_dir / "evaluator",
                                                   seed=int(self.np_random.integers(2**32)))
        self._record({"event": "reset_started", "workload": asdict(self.current_workload),
                      "budget_seconds": self.budget_seconds, "initial_schedule": asdict(self.schedule)})
        try:
            start = time.monotonic()
            result = self.evaluator.baseline(self.current_workload,
                                            timeout_seconds=min(self._remaining(), self.trial_timeout_seconds))
            self._charged += max(time.monotonic() - start, result.elapsed_seconds)
            self._validate(result, None)
            self._record({"event": "baseline", "measurement": asdict(result), "elapsed_seconds": self._elapsed()})
            if not result.correct or self._remaining() <= 0:
                raise RuntimeError("reference failed or consumed the episode budget")
            self.baseline = result
        except Exception as exc:
            self._record({"event": "reset_failed", "error": type(exc).__name__ + ": " + str(exc)})
            self.evaluator.close()
            raise
        self._needs_reset = False
        return self._observe(), self._info(outcome="ready")

    def step(self, action):
        if self._closed or self._needs_reset:
            raise RuntimeError("call reset before stepping")
        if isinstance(action, bool) or not isinstance(action, Integral) or not self.action_space.contains(action):
            raise ValueError("action must be a valid integer index")
        start = time.monotonic()
        before = asdict(self.schedule)
        penalty = score = 0.0
        measured_duration = 0.0
        outcome = "edited"
        terminated = False
        if self._remaining() <= 0:
            outcome = "budget_exhausted"
        elif ACTIONS[action] in ("measure", "submit"):
            self._record({"event": "trial_started", "action": ACTIONS[action], "schedule": before,
                          "remaining_seconds": self._remaining()})
            self.last_measurement = None
            try:
                result = self.evaluator.evaluate(self.current_workload, self.schedule,
                                                timeout_seconds=min(self._remaining(), self.trial_timeout_seconds))
                # Preserve a returned attempt before accepting its bindings.
                self._record({"event": "trial_returned", "measurement": asdict(result)})
                self._validate(result, before)
                measured_duration = result.elapsed_seconds
                self.last_measurement = result
                outcome = result.status
                if not result.correct:
                    penalty = 0.1
                if result.status == "timeout" or result.evidence.get("worker_reaped"):
                    terminated = True
                elif ACTIONS[action] == "submit":
                    terminated = True
                    if result.correct and self._remaining() > 0:
                        score = float(np.clip(1 - result.latency_ms / self.baseline.latency_ms, -1, 1))
            except Exception as exc:
                outcome = "evaluator_error"
                penalty = 0.1
                self._record({"event": "trial_error", "error": type(exc).__name__ + ": " + str(exc)})
                # Binding/transport errors invalidate this episode; do not let
                # a compromised or broken evaluator continue producing reward.
                terminated = True
            finally:
                self._charged += max(time.monotonic() - start, measured_duration, self.MIN_ACTION_SECONDS)
        else:
            try:
                self.schedule = transform(self.current_workload, self.schedule, int(action))
                self.last_measurement = None
            except ValueError:
                outcome = "invalid_action"
                penalty = 0.01
            self._charged += max(time.monotonic() - start, self.MIN_ACTION_SECONDS)
        # Teardown is part of this episode's cost. A submission that only fits
        # before worker cleanup must not earn a score after its time limit.
        if terminated or self._remaining() <= 0:
            cleanup_started = time.monotonic()
            self.evaluator.close()
            self._charged += max(time.monotonic() - cleanup_started, 0)
        elapsed = self._elapsed()
        truncated = elapsed >= self.budget_seconds
        if truncated:
            score = 0.0
            terminated = False
        time_penalty = 0.05 * min(max(elapsed - self._last_reward_elapsed, 0) / self.budget_seconds, 1)
        self._last_reward_elapsed = elapsed
        reward = score - penalty - time_penalty
        self._reward_components = {"speed_score": score, "elapsed_penalty": time_penalty,
                                   "invalid_action_penalty": penalty if outcome == "invalid_action" else 0.0,
                                   "failed_measurement_penalty": penalty if outcome != "invalid_action" else 0.0}
        self._needs_reset = terminated or truncated
        self._record({"event": "step", "action": ACTIONS[action], "schedule_before": before,
                      "schedule_after": asdict(self.schedule), "measurement": asdict(self.last_measurement) if self.last_measurement else None,
                      "outcome": outcome, "reward": reward, "submission_score": score,
                      "reward_components": self._reward_components,
                      "elapsed_seconds": elapsed, "terminated": terminated, "truncated": truncated})
        return self._observe(), reward, terminated, truncated, self._info(outcome=outcome, score=score)

    def close(self):
        if self.evaluator is not None:
            self.evaluator.close()
        self._closed = self._needs_reset = True
