"""CPU search episodes backed by recorded CUDA measurements.

Replay events are not GPU measurements. The same finite collection is reused
across policy seeds, so more episodes do not add independent timing samples.
The evaluator owns the table; policies receive only outcomes of queried actions.
This API boundary is not a security boundary against untrusted Python callers.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from statistics import median
import uuid

from .search import TrialResult
from .suite import Suite, load_suite, _strict_object, _reject_constant


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class MeasuredTable:
    """Validated complete collection from one physical GPU/runtime cohort."""

    def __init__(self, document, *, suite: Suite, table_sha256=None):
        # Lazy import keeps importing the environment independent of CUDA.
        from .table_collection import validate_table
        validate_table(document)
        if document["suite_sha256"] != suite.sha256 or document["suite"] != suite.document:
            raise ValueError("table and workload suite identities differ")
        self._document = deepcopy(document)
        self.suite = suite
        self.sha256 = table_sha256 or hashlib.sha256(_canonical(document)).hexdigest()
        self.cohort_id = document["cohort_id"]
        self.repeat_indices = tuple(document["repeat_indices"])
        self._rows = {}
        for row in document["rows"]:
            m = row["measurement"]
            key = (m["task_id"], m["candidate_id"], row["repeat_index"])
            if key in self._rows:
                raise ValueError("duplicate table cell/repeat")
            self._rows[key] = deepcopy(m)
        expected = {(t.task_id, c, r) for t in suite.tasks
                    for c in (None, *(c.candidate_id for c in suite.candidates))
                    for r in self.repeat_indices}
        if set(self._rows) != expected:
            raise ValueError("table must contain every task, baseline and candidate for every repeat")

    @classmethod
    def load(cls, path, *, suite=None):
        raw = Path(path).read_bytes()
        if len(raw) > 100_000_000:
            raise ValueError("table exceeds 100 MB bound")
        document = json.loads(raw, object_pairs_hook=_strict_object, parse_constant=_reject_constant)
        return cls(document, suite=suite or load_suite(), table_sha256=hashlib.sha256(raw).hexdigest())

    def row(self, task_id, candidate_id, repeat_index):
        return deepcopy(self._rows[(task_id, candidate_id, repeat_index)])

    def cell_summary(self, task_id, candidate_id, *, memory_weight=0.0):
        """Post-episode analyst view. Never pass this method or result to policy."""
        rows = [self._rows[(task_id, candidate_id, r)] for r in self.repeat_indices]
        refs = [self._rows[(task_id, None, r)] for r in self.repeat_indices]
        successful = [m for m in rows if m["status"] == "ok"]
        all_pass = len(successful) == len(rows)
        if not all_pass:
            return {"admissible_in_all_repeats": False, "failure_rate": 1 - len(successful) / len(rows),
                    "median_cost": None}
        costs = [(m["latency_ms"] / b["latency_ms"] + memory_weight *
                  m["peak_allocated_bytes"] / b["peak_allocated_bytes"]) / (1 + memory_weight)
                 for m, b in zip(rows, refs)]
        return {"admissible_in_all_repeats": True, "failure_rate": 0.0,
                "median_cost": median(costs)}

    def oracle_cost(self, task_id, *, memory_weight=0.0):
        candidates = [self.cell_summary(task_id, c.candidate_id, memory_weight=memory_weight)
                      for c in self.suite.candidates]
        return min([1.0] + [c["median_cost"] for c in candidates if c["admissible_in_all_repeats"]])


class MeasuredReplayEvaluator:
    """Replay one repeat per episode, cycling records for repeated actions.

    Initial baseline and each first candidate query use the same recorded
    process repeat. Further queries of that candidate cycle through the other
    recorded repeats. This is deterministic replay of finite data, not an IID
    noise simulator. The seed selects a starting repeat and does not create data.
    """

    def __init__(self, table: MeasuredTable, *, seed=0, event_namespace=None):
        if not isinstance(table, MeasuredTable) or type(seed) is not int or seed < 0:
            raise ValueError("a measured table and nonnegative integer seed are required")
        self._table = table
        self.seed = seed
        self._namespace = event_namespace or uuid.uuid4().hex
        self._counts = defaultdict(int)
        self._event = 0
        self._closed = False
        self._tasks = {t.task_id: t.to_dict() for t in table.suite.tasks}
        self._candidates = {c.candidate_id: c.to_dict() for c in table.suite.candidates}

    def _run(self, task, candidate):
        if self._closed:
            raise RuntimeError("replay evaluator is closed")
        if self._tasks.get(task.task_id) != task.to_dict():
            raise ValueError("replay task differs from recorded suite")
        candidate_id = None if candidate is None else candidate.candidate_id
        if candidate is not None and self._candidates.get(candidate_id) != candidate.to_dict():
            raise ValueError("replay candidate differs from recorded suite")
        key = (task.task_id, candidate_id)
        position = (self.seed + self._counts[key]) % len(self._table.repeat_indices)
        repeat_index = self._table.repeat_indices[position]
        self._counts[key] += 1
        source = self._table.row(task.task_id, candidate_id, repeat_index)
        self._event += 1
        return TrialResult(
            trial_id=f"replay:{self._namespace}:{self._event}", task_id=task.task_id,
            candidate_id=candidate_id, backend="measured_replay", cohort_id=self._table.cohort_id,
            input_digest=source["input_digest"], status=source["status"], correct=source["correct"],
            latency_ms=source["latency_ms"], peak_allocated_bytes=source["peak_allocated_bytes"],
            evidence={"table_sha256": self._table.sha256, "source_trial_id": source["trial_id"],
                      "source_backend": "cuda", "source_repeat_index": repeat_index,
                      "fresh_gpu_measurement": False, "recorded_outcome_reused": True,
                      "sampling": "episode_start_repeat_then_per_candidate_cycle"})

    def baseline(self, task):
        return self._run(task, None)

    def evaluate(self, task, candidate):
        return self._run(task, candidate)

    def close(self):
        self._closed = True


def add_replay_analysis(report, table: MeasuredTable, *, memory_weight=0.0):
    """Add selection regret using all recorded repeats after policy execution.

    Regret is relative to candidates that passed every recorded repeat. A
    selected candidate with a recorded failure receives reference cost 1.
    Results remain table estimates requiring separate live confirmation.
    """
    result = deepcopy(report)
    result.update(backend="measured_replay", measured_replay=True, synthetic=False,
                  fresh_gpu_measurements=0, table_sha256=table.sha256,
                  recorded_process_repeats=len(table.repeat_indices),
                  independent_gpu_samples_added_by_policy_evaluation=0,
                  sampling="episode_start_repeat_then_per_candidate_cycle",
                  selected_config_confirmation_performed=False)
    for episode in result["episodes"]:
        if episode["status"] != "completed":
            continue
        task_id = episode["task_id"]
        oracle = table.oracle_cost(task_id, memory_weight=memory_weight)
        curve = []
        for step in episode["steps"]:
            # The evaluation harness records post-step info; all-cell analyst
            # access occurs only here, after the policy has completed.
            info = step["info"]
            selected = info["incumbent_candidate_id"]
            cell = table.cell_summary(task_id, selected, memory_weight=memory_weight)
            cost = cell["median_cost"] if cell["admissible_in_all_repeats"] else 1.0
            curve.append({"trials_used": info["trials_used"], "selected_candidate_id": selected,
                          "table_selection_cost": cost, "table_regret": max(0.0, cost - oracle),
                          "selection_failure_rate": cell["failure_rate"]})
        episode["table_analysis"] = {"oracle_cost": oracle, "curve": curve,
                                     "live_confirmation_performed": False}
    result["analysis_definition"] = {
        "oracle": "best median paired cost among candidates passing all recorded repeats, including eager reference",
        "failed_selection_cost": 1.0,
        "regret": "max(0, selected median paired cost - oracle cost)",
        "uncertainty": "policy seeds reuse finite measured data; not independent GPU timing repeats"}
    return result
