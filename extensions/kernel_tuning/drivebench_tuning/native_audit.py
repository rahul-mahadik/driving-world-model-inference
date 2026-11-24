"""Independent, CPU-only consistency audit of recorded native/PPO selections.

This reconstructs selection arithmetic from measurements, without loading a
policy or calling the execution environment. It checks the saved numerical-gate
receipts, not the underlying tensors; it is not a new GPU or correctness run.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics

from .contract import DTYPE_TOLERANCES


SELECTION_COST_NOTE = (
    "Instrumented harness wall time, with unequal search budgets: native uses "
    "24 configurations plus 3 reference probes; PPO uses 8 queries plus an eager "
    "reset. PPO and reference probes journal/fsync every CUDA timing sample; "
    "native tuning journals one completed row per configuration. Selectors share "
    "compilation caches within each process and alternate order. Offline PPO "
    "training and independent final confirmations are excluded. These costs are "
    "not an isolated comparison of algorithmic search overhead."
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, label, *, positive=False):
    _require(type(value) in (int, float) and math.isfinite(value)
             and (value > 0 if positive else value >= 0), f"invalid {label}")
    return value


def _same(actual, expected, label):
    """Compare recorded state structurally, allowing only roundoff in floats."""
    if type(expected) is dict:
        _require(type(actual) is dict and set(actual) == set(expected), f"{label} keys differ")
        for key in expected:
            _same(actual[key], expected[key], f"{label}.{key}")
    elif type(expected) is list:
        _require(type(actual) is list and len(actual) == len(expected), f"{label} length differs")
        for index, (a, e) in enumerate(zip(actual, expected)):
            _same(a, e, f"{label}[{index}]")
    elif type(expected) is float:
        _require(type(actual) in (int, float) and math.isfinite(actual)
                 and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), f"{label} differs")
    else:
        _require(type(actual) is type(expected) and actual == expected, f"{label} differs")


def _digest(hashes):
    return hashlib.sha256(json.dumps(hashes, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _gate(gate, dtype, *, passed):
    _require(type(gate) is dict and gate.get("passed") is passed, "numerical gate result differs")
    for key, value in DTYPE_TOLERANCES[dtype].items():
        _same(gate.get(key), value, "numerical gate tolerance")
    if passed:
        _require(gate.get("shape_dtype_match") is True and gate.get("finite") is True,
                 "successful numerical gate lacks shape/dtype/finite checks")
        _number(gate.get("max_abs_error"), "numerical max error")
        _number(gate.get("rmse"), "numerical RMSE")


def _samples(values, scalar, label):
    _require(type(values) is list and len(values) == 30, f"{label} requires 30 samples")
    for value in values:
        _number(value, f"{label} sample", positive=True)
    _number(scalar, f"{label} median", positive=True)
    _same(scalar, float(statistics.median(values)), f"{label} sample median")


def _quantiles(samples):
    ordered = sorted(samples)
    values = []
    for fraction in (.5, .2, .8):
        position = (len(ordered) - 1) * fraction
        low, high = math.floor(position), math.ceil(position)
        values.append(ordered[low] + (ordered[high] - ordered[low]) * (position - low))
    return values


def _measurement(row, *, task_id, candidate_id, cohort_id, input_digest, dtype,
                 implementation, seen):
    """Check a saved CudaEvaluator result and register its unique trial ID."""
    _require(type(row) is dict, "measurement object required")
    for key, expected in (("task_id", task_id), ("candidate_id", candidate_id),
                          ("cohort_id", cohort_id), ("input_digest", input_digest),
                          ("backend", "cuda")):
        _same(row.get(key), expected, f"measurement {key}")
    trial_id = row.get("trial_id")
    _require(type(trial_id) is str and trial_id and trial_id not in seen,
             "measurement trial ID missing or reused")
    seen.add(trial_id)
    evidence = row.get("evidence")
    _require(type(evidence) is dict, "measurement evidence required")
    for key, expected in (("trial_id", trial_id), ("cohort_id", cohort_id),
                          ("implementation", implementation), ("boundary", "isolated_operator"),
                          ("warmups", 5), ("repetitions", 30)):
        _same(evidence.get(key), expected, f"measurement evidence {key}")
    hashes = evidence.get("input_sha256")
    _require(type(hashes) is list and hashes and all(type(h) is str and len(h) == 64
             and set(h) <= set("0123456789abcdef") for h in hashes), "input hashes missing")
    _require(_digest(hashes) == input_digest, "measurement raw input hashes differ")
    status = row.get("status")
    _require(status in ("ok", "incorrect", "error") and row.get("correct") is (status == "ok"),
             "measurement status/correctness inconsistent")
    if status == "ok":
        _samples(evidence.get("cuda_event_ms"), row.get("latency_ms"), "CUDA timing")
        _gate(evidence.get("correctness"), dtype, passed=True)
        _require(evidence.get("input_integrity_verified") is True,
                 "successful measurement lacks input integrity verification")
        allocations = evidence.get("peak_allocated_delta_bytes")
        _require(type(allocations) is list and len(allocations) == 30
                 and all(type(v) is int and v >= 0 for v in allocations), "allocation samples invalid")
        _same(row.get("peak_allocated_bytes"), max(allocations), "peak allocation")
    else:
        _require(row.get("latency_ms") is None and row.get("peak_allocated_bytes") is None,
                 "failed measurement must not expose usable metrics")
        if status == "incorrect":
            _gate(evidence.get("correctness"), dtype, passed=False)
        else:
            _require(type(evidence.get("error")) is dict and evidence["error"].get("type"),
                     "failed measurement lacks its error")
    return status == "ok"


def audit_live_task(result, suite, *, cohort_id, expected_input_digest=None):
    """Audit one task's selections and all four independent confirmations.

    ``expected_input_digest`` can bind an external frozen input manifest. If it
    is omitted, this verifies internal consistency with the native input binding
    only; the receipt states that narrower scope explicitly.
    """
    task_indices = {suite.tasks[i].task_id: i for i in suite.splits["test"]}
    task_id = result.get("task_id")
    _require(task_id in task_indices, "declared held-out task required")
    task_index = task_indices[task_id]
    task = suite.tasks[task_index]
    dtype = task.to_dict()["metadata"]["workload"]["dtype"]
    candidates = list(suite.candidates)
    by_id = {c.candidate_id: c for c in candidates}
    native_ids = [c.candidate_id for c in candidates if c.config["implementation"] == "triton"]
    compiled = next(c.candidate_id for c in candidates if c.config["implementation"] == "compiled")
    _require(type(cohort_id) is str and cohort_id, "external cohort binding required")
    selectors = result.get("selectors", {})
    _require(set(selectors) == {"native", "ppo"}, "native/PPO selections required")
    native, ppo = selectors["native"], selectors["ppo"]
    digest = native.get("input_digest")
    _require(type(digest) is str and len(digest) == 64 and set(digest) <= set("0123456789abcdef"),
             "native input digest invalid")
    if expected_input_digest is not None:
        _same(digest, expected_input_digest, "external frozen input binding")
    _same(native.get("task_id"), task_id, "native task")
    _same(native.get("cohort_id"), cohort_id, "native cohort")
    seen = set()

    def measure(row, candidate_id):
        _require(candidate_id is None or candidate_id in by_id, "unknown selected candidate")
        implementation = "eager" if candidate_id is None else by_id[candidate_id].config["implementation"]
        return _measurement(row, task_id=task_id, candidate_id=candidate_id, cohort_id=cohort_id,
                            input_digest=digest, dtype=dtype, implementation=implementation, seen=seen)

    # Independently check native's chosen configuration from its saved sample medians.
    records = native.get("records", [])
    _require(len(records) == 24 and native.get("configs_attempted") == 24
             and [r.get("candidate_id") for r in records] == native_ids,
             "native grid/order must contain every declared configuration exactly once")
    admissible_native = []
    native_timings = native.get("native_configs_timings", [])
    _require(len(native_timings) == 24
             and [r.get("candidate_id") for r in native_timings] == native_ids,
             "native quantile rows must match the declared grid")
    native_quantiles = {r["candidate_id"]: r.get("quantiles_ms") for r in native_timings}
    for row in records:
        for key, expected in (("task_id", task_id), ("cohort_id", cohort_id), ("input_digest", digest)):
            _same(row.get(key), expected, f"native record {key}")
        _same(row.get("config"), by_id[row["candidate_id"]].to_dict()["config"], "native record config")
        status = row.get("status")
        _require(status in ("ok", "incorrect", "error"), "native record status invalid")
        if status == "ok":
            _samples(row.get("cuda_event_ms"), row.get("median_ms"), "native tuning")
            _gate(row.get("correctness"), dtype, passed=True)
            _require(row.get("input_integrity_verified") is True, "native record input check absent")
            _same(native_quantiles[row["candidate_id"]], _quantiles(row["cuda_event_ms"]),
                  "native quantiles")
            admissible_native.append(row)
        elif status == "incorrect":
            _gate(row.get("correctness"), dtype, passed=False)
            _same(native_quantiles[row["candidate_id"]], [None] * 3, "failed native quantiles")
        else:
            _require(type(row.get("error_type")) is str and row["error_type"], "native error missing")
            _same(native_quantiles[row["candidate_id"]], [None] * 3, "failed native quantiles")
    _require(admissible_native, "native grid has no successful configuration")
    # Triton compares the do_bench result lists lexicographically: median,
    # then p20, then p80; exact ties preserve the declared configuration order.
    selected_native = min(admissible_native,
                          key=lambda r: native_quantiles[r["candidate_id"]])["candidate_id"]
    _same(native.get("selected_candidate_id"), selected_native, "native selected candidate")
    _gate(native.get("selected_correctness"), dtype, passed=True)

    # PPO reconstruction deliberately does not invoke BudgetedTuningEnv or PPOSelector.
    _same(ppo.get("task_id"), task_id, "PPO task")
    _require(ppo.get("candidate_turns") == 8 and ppo.get("policy_updates_permitted") is False,
             "fixed-policy eight-query episode required")
    initial = ppo.get("initial_info", {})
    baseline = initial.get("baseline")
    _require(measure(baseline, None), "PPO reset must be a successful eager baseline")
    _require(baseline["peak_allocated_bytes"] > 0, "PPO reference allocation must be positive")
    incumbent, incumbent_action, best_cost = baseline, 0, 1.0
    count = len(candidates)
    counts, statuses, metrics = [0] * count, [0] * count, [[0.0, 0.0] for _ in candidates]

    def observation(used):
        return {"task_features": list(task.features), "budget": [(8 - used) / 8.0],
                "incumbent": [incumbent["latency_ms"] / baseline["latency_ms"],
                              incumbent["peak_allocated_bytes"] / baseline["peak_allocated_bytes"]],
                "incumbent_action": incumbent_action, "action_counts": list(counts),
                "action_status": list(statuses), "action_metrics": [list(m) for m in metrics]}

    def check_info(info, used, measurement):
        for key, expected in (("task_id", task_id), ("task_index", task_index), ("trials_used", used),
                              ("trial_budget", 8), ("baseline", baseline), ("incumbent", incumbent),
                              ("incumbent_candidate_id", incumbent["candidate_id"]),
                              ("incumbent_cost", float(best_cost)), ("measurement", measurement),
                              ("backend", "cuda"), ("cohort_id", cohort_id), ("input_digest", digest),
                              ("synthetic", False), ("measured_replay", False), ("fresh_gpu_measurement", True)):
            _same(info.get(key), expected, f"PPO info {key}")
        objective = info.get("objective", {})
        _same(objective.get("memory_weight"), 0.0, "PPO memory weight")
        _same(objective.get("reward"), "previous_incumbent_cost - current_incumbent_cost", "PPO reward definition")

    _same(ppo.get("initial_observation"), observation(0), "PPO initial observation")
    check_info(initial, 0, None)
    turns = ppo.get("turns", [])
    _require(len(turns) == 8, "eight PPO transitions required")
    reward_sum = 0.0
    for turn_index, row in enumerate(turns):
        _same(row.get("turn"), turn_index, "PPO turn index")
        _same(row.get("preaction_observation"), observation(turn_index), "PPO preaction observation")
        action = row.get("action")
        _require(type(action) is int and 0 <= action < count, "PPO action outside candidate grid")
        measurement = row.get("info", {}).get("measurement")
        correct = measure(measurement, candidates[action].candidate_id)
        counts[action] += 1
        statuses[action] = {"ok": 1, "incorrect": 2, "error": 3}[measurement["status"]]
        metrics[action] = [0.0, 0.0]
        reward = 0.0
        if correct:
            ratio = measurement["latency_ms"] / baseline["latency_ms"]
            metrics[action] = [ratio, measurement["peak_allocated_bytes"] / baseline["peak_allocated_bytes"]]
            if ratio < best_cost:
                reward = best_cost - ratio
                incumbent, incumbent_action, best_cost = measurement, action + 1, ratio
        _same(row.get("reward"), float(reward), "PPO reward")
        reward_sum += reward
        _number(row.get("cpu_policy_seconds"), "PPO policy inference time")
        _same(row.get("observation"), observation(turn_index + 1), "PPO postaction observation")
        check_info(row["info"], turn_index + 1, measurement)
    _same(ppo.get("final_info"), turns[-1]["info"], "PPO final info")
    _same(ppo.get("selected_candidate_id"), incumbent["candidate_id"], "PPO selected candidate")
    _same(float(reward_sum), 1.0 - best_cost, "PPO telescoping reward")

    # The executor's tie order is fixed eager/compiled/native, independent of
    # the rotated execution order or JSON dictionary insertion order.
    fallback = native.get("reference_fallback", {})
    _require(fallback.get("selected_before_confirmation") is True, "fallback selected after confirmation")
    roles = ("eager", "compiled", "native")
    _require(sorted(fallback.get("order", [])) == sorted(roles), "fallback probe execution order invalid")
    probes = fallback.get("probes", {})
    _require(set(probes) == set(roles), "all three fallback probes required")
    role_ids = {"eager": None, "compiled": compiled, "native": selected_native}
    admissible_roles = [role for role in roles if measure(probes[role], role_ids[role])]
    _require(admissible_roles, "no correct fallback probe")
    selected_role = min(admissible_roles, key=lambda role: probes[role]["latency_ms"])
    _same(fallback.get("selected_role"), selected_role, "fallback selected role")
    _same(fallback.get("selected_candidate_id"), role_ids[selected_role], "fallback selected candidate")
    for value, label in ((native.get("selection_wall_seconds"), "native selection time"),
                         (ppo.get("selection_wall_seconds"), "PPO selection time"),
                         (fallback.get("selection_wall_seconds"), "fallback selection time")):
        _number(value, label, positive=True)

    confirmations = result.get("confirmation", {})
    role_ids["ppo"] = incumbent["candidate_id"]
    _require(sorted(confirmations.get("order", [])) == sorted(role_ids), "confirmation order invalid")
    measurements = confirmations.get("measurements", {})
    _require(set(measurements) == set(role_ids), "four separate confirmation roles required")
    confirmation_passed = sum(measure(measurements[role], role_ids[role]) for role in role_ids)
    return {"kind": "independent_native_ppo_task_consistency_audit", "task_id": task_id,
            "cohort_id": cohort_id, "input_digest": digest,
            "external_input_manifest_checked": expected_input_digest is not None,
            "native_configs_checked": 24, "ppo_queries_reconstructed": 8,
            "ppo_reward_sum": reward_sum, "ppo_final_incumbent_cost": best_cost,
            "fallback_selected_role": selected_role, "unique_trial_ids_checked": len(seen),
            "confirmation_passed": confirmation_passed, "confirmation_failed": 4 - confirmation_passed,
            "fresh_gpu_measurements": 0, "policy_execution_performed": False,
            "tensor_correctness_recomputed": False, "selection_cost_note": SELECTION_COST_NOTE}
