"""Offline audit tests with explicit fixtures; none are GPU evidence."""
import hashlib
import json

import pytest

from drivebench_tuning.contract import DTYPE_TOLERANCES
from drivebench_tuning.native_audit import audit_live_task
from drivebench_tuning.search import BudgetedTuningEnv, TrialResult
from drivebench_tuning.suite import load_suite


def _json(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    return value


@pytest.fixture(scope="module")
def suite():
    return load_suite("operator-generalization-v2")


def make_recorded_task(suite, task_index=None, prefix="fixture"):
    """Capture actual env transitions around a controlled, clearly marked evaluator."""
    task_index = suite.splits["test"][0] if task_index is None else task_index
    task = suite.tasks[task_index]
    dtype = task.to_dict()["metadata"]["workload"]["dtype"]
    hashes = ["a" * 64]
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True, allow_nan=False).encode()).hexdigest()
    cohort = "fixture-only-not-a-gpu-cohort"
    candidates = list(suite.candidates)
    by_id = {c.candidate_id: c for c in candidates}
    native = [c for c in candidates if c.config["implementation"] == "triton"]
    compiled = next(c for c in candidates if c.config["implementation"] == "compiled")
    sequence = 0

    def gate(passed=True):
        return {"passed": passed, "shape_dtype_match": True, "finite": True,
                **DTYPE_TOLERANCES[dtype], "max_abs_error": 0.0 if passed else 1.0,
                "rmse": 0.0 if passed else 1.0}

    def measurement(candidate, latency=100.0, status="ok"):
        nonlocal sequence
        sequence += 1
        trial_id = f"{prefix}-trial-{sequence}"
        correct = status == "ok"
        evidence = {"test_fixture_not_gpu_evidence": True,
                    "trial_id": trial_id, "cohort_id": cohort,
                    "implementation": "eager" if candidate is None else candidate.config["implementation"],
                    "boundary": "isolated_operator", "warmups": 5, "repetitions": 30,
                    "input_sha256": hashes, "correctness": gate(correct)}
        if correct:
            evidence.update(cuda_event_ms=[latency] * 30, input_integrity_verified=True,
                            peak_allocated_delta_bytes=[100] * 30)
        elif status == "error":
            evidence["error"] = {"type": "OutOfResources", "message": "fixture"}
        return TrialResult(trial_id, task.task_id, None if candidate is None else candidate.candidate_id,
                           "cuda", cohort, digest, status, correct,
                           latency if correct else None, 100 if correct else None, evidence)

    class Controlled:
        query = 0

        def baseline(self, task):
            return measurement(None)

        def evaluate(self, task, candidate):
            latency, status = [(80., "ok"), (60., "ok"), (65., "ok"), (None, "incorrect"),
                               (55., "ok"), (80., "ok"), (50., "ok"), (52., "ok")][self.query]
            self.query += 1
            return measurement(candidate, latency, status)

    env = BudgetedTuningEnv(suite.tasks, candidates, Controlled(), trial_budget=8, memory_weight=0)
    observation, info = env.reset(seed=4101, options={"task_index": task_index})
    initial_observation, initial_info = _json(observation), _json(info)
    turns = []
    for index, action in enumerate((0, 1, 1, 2, 3, 0, 4, 4)):
        preaction = _json(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        assert terminated == (index == 7) and not truncated
        turns.append({"turn": index, "action": action, "reward": reward,
                      "preaction_observation": preaction, "cpu_policy_seconds": .001,
                      "observation": _json(observation), "info": _json(info)})
    ppo = {"task_id": task.task_id, "selected_candidate_id": info["incumbent_candidate_id"],
           "initial_info": initial_info, "initial_observation": initial_observation,
           "turns": turns, "final_info": _json(info), "candidate_turns": 8,
           "selection_wall_seconds": 1.0, "policy_updates_permitted": False}
    native_records = [{"task_id": task.task_id, "candidate_id": c.candidate_id,
                       "cohort_id": cohort, "input_digest": digest, "status": "ok",
                       "config": c.to_dict()["config"], "cuda_event_ms": [60. + i] * 30,
                       "median_ms": 60. + i, "correctness": gate(), "input_integrity_verified": True}
                      for i, c in enumerate(native)]
    # Execution order differs from the executor's fixed eager/compiled/native tie order.
    probes = {role: measurement(c, latency).to_dict() for role, c, latency in
              (("native", native[0], 75.), ("eager", None, 100.), ("compiled", compiled, 70.))}
    fallback = {"order": ["native", "eager", "compiled"], "probes": probes,
                "selected_role": "compiled", "selected_candidate_id": compiled.candidate_id,
                "selection_wall_seconds": .3, "selected_before_confirmation": True}
    n = {"task_id": task.task_id, "cohort_id": cohort, "input_digest": digest,
         "configs_attempted": 24, "records": native_records,
         "native_configs_timings": [{"candidate_id": r["candidate_id"], "quantiles_ms": [r["median_ms"]] * 3}
                                    for r in native_records],
         "selected_candidate_id": native[0].candidate_id, "selected_correctness": gate(),
         "reference_fallback": fallback, "selection_wall_seconds": 2.0}
    confirmations = {role: measurement(c, latency).to_dict() for role, c, latency in
                     (("ppo", by_id[ppo["selected_candidate_id"]], 53.), ("compiled", compiled, 65.),
                      ("eager", None, 100.), ("native", native[0], 59.))}
    result = {"task_id": task.task_id, "selector_order": ["native", "ppo"],
              "selectors": {"native": n, "ppo": ppo},
              "confirmation": {"order": list(confirmations), "measurements": confirmations}}
    return result, cohort, digest


@pytest.fixture
def recorded(suite):
    return make_recorded_task(suite)


def test_reconstructs_real_environment_state_without_running_gpu_or_policy(suite, recorded):
    row, cohort, digest = recorded
    receipt = audit_live_task(row, suite, cohort_id=cohort, expected_input_digest=digest)
    assert receipt["ppo_reward_sum"] == pytest.approx(.5)
    assert receipt["ppo_final_incumbent_cost"] == pytest.approx(.5)
    assert receipt["unique_trial_ids_checked"] == 16
    assert receipt["native_configs_checked"] == 24
    assert receipt["fallback_selected_role"] == "compiled"
    assert receipt["confirmation_passed"] == 4
    assert receipt["external_input_manifest_checked"] is True
    assert receipt["tensor_correctness_recomputed"] is False
    assert "fsync" in receipt["selection_cost_note"]
    assert "compilation caches" in receipt["selection_cost_note"]


@pytest.mark.parametrize("mutation,match", [
    ("fallback_role", "fallback selected role"),
    ("reward", "PPO reward"),
    ("selected_candidate", "PPO selected candidate"),
    ("query_candidate", "measurement candidate_id"),
    ("preaction_state", "PPO preaction observation"),
    ("postaction_state", "PPO postaction observation"),
    ("repeated_action_state", "PPO postaction observation"),
    ("changed_baseline", "PPO info baseline"),
    ("query_cohort", "measurement cohort_id"),
    ("query_input_hash", "raw input hashes"),
    ("sample_median", "sample median"),
    ("native_candidate", "native selected candidate"),
    ("query_reuse", "trial ID missing or reused"),
    ("confirmation_reuse", "trial ID missing or reused"),
    ("weakened_gate", "numerical gate tolerance"),
])
def test_rejects_material_selection_and_evidence_tampering(suite, recorded, mutation, match):
    row, cohort, digest = recorded
    ppo = row["selectors"]["ppo"]
    native = row["selectors"]["native"]
    if mutation == "fallback_role":
        native["reference_fallback"]["selected_role"] = "native"
        native["reference_fallback"]["selected_candidate_id"] = native["selected_candidate_id"]
    elif mutation == "reward":
        ppo["turns"][0]["reward"] += .01
    elif mutation == "selected_candidate":
        ppo["selected_candidate_id"] = suite.candidates[0].candidate_id
    elif mutation == "query_candidate":
        ppo["turns"][0]["info"]["measurement"]["candidate_id"] = suite.candidates[1].candidate_id
    elif mutation == "preaction_state":
        ppo["turns"][3]["preaction_observation"]["budget"] = [1.0]
    elif mutation == "postaction_state":
        ppo["turns"][1]["observation"]["incumbent"] = [.001, 1.0]
    elif mutation == "repeated_action_state":
        ppo["turns"][2]["observation"]["action_counts"][1] = 1
    elif mutation == "changed_baseline":
        ppo["turns"][0]["info"]["baseline"]["latency_ms"] = 110.
    elif mutation == "query_cohort":
        ppo["turns"][0]["info"]["measurement"]["cohort_id"] = "different-fixture"
    elif mutation == "query_input_hash":
        ppo["turns"][0]["info"]["measurement"]["evidence"]["input_sha256"] = ["b" * 64]
    elif mutation == "sample_median":
        native["reference_fallback"]["probes"]["compiled"]["latency_ms"] = 1.
    elif mutation == "native_candidate":
        native["selected_candidate_id"] = native["records"][1]["candidate_id"]
    elif mutation == "query_reuse":
        ppo["turns"][2]["info"]["measurement"]["trial_id"] = ppo["turns"][1]["info"]["measurement"]["trial_id"]
    elif mutation == "confirmation_reuse":
        row["confirmation"]["measurements"]["compiled"]["trial_id"] = native["reference_fallback"]["probes"]["compiled"]["trial_id"]
    elif mutation == "weakened_gate":
        native["reference_fallback"]["probes"]["compiled"]["evidence"]["correctness"]["atol"] = 1.
    with pytest.raises(ValueError, match=match):
        audit_live_task(row, suite, cohort_id=cohort, expected_input_digest=digest)


def test_fallback_tie_uses_fixed_role_order_not_rotated_probe_order(suite, recorded):
    row, cohort, digest = recorded
    fallback = row["selectors"]["native"]["reference_fallback"]
    for probe in fallback["probes"].values():
        probe["latency_ms"] = 70.
        probe["evidence"]["cuda_event_ms"] = [70.] * 30
    fallback["selected_role"], fallback["selected_candidate_id"] = "eager", None
    assert audit_live_task(row, suite, cohort_id=cohort)["fallback_selected_role"] == "eager"
    fallback["selected_role"] = "native"
    fallback["selected_candidate_id"] = row["selectors"]["native"]["selected_candidate_id"]
    with pytest.raises(ValueError, match="fallback selected role"):
        audit_live_task(row, suite, cohort_id=cohort)


def test_fallback_excludes_failed_probe_and_preserves_confirmation_failure(suite, recorded):
    row, cohort, digest = recorded
    fallback = row["selectors"]["native"]["reference_fallback"]
    probe = fallback["probes"]["compiled"]
    probe.update(status="incorrect", correct=False, latency_ms=None, peak_allocated_bytes=None)
    probe["evidence"]["correctness"]["passed"] = False
    fallback["selected_role"] = "native"
    fallback["selected_candidate_id"] = row["selectors"]["native"]["selected_candidate_id"]
    confirmation = row["confirmation"]["measurements"]["ppo"]
    confirmation.update(status="error", correct=False, latency_ms=None, peak_allocated_bytes=None)
    confirmation["evidence"]["error"] = {"type": "OutOfResources", "message": "fixture"}
    receipt = audit_live_task(row, suite, cohort_id=cohort)
    assert receipt["confirmation_failed"] == 1
    assert receipt["confirmation_passed"] == 3


def test_external_input_binding_is_enforced_when_supplied(suite, recorded):
    row, cohort, digest = recorded
    with pytest.raises(ValueError, match="external frozen input binding"):
        audit_live_task(row, suite, cohort_id=cohort, expected_input_digest="0" * 64)
    receipt = audit_live_task(row, suite, cohort_id=cohort)
    assert receipt["external_input_manifest_checked"] is False


def test_native_tie_uses_triton_lexicographic_quantiles(suite, recorded):
    row, cohort, digest = recorded
    native = row["selectors"]["native"]
    second = native["records"][1]
    second["cuda_event_ms"] = [50.] * 10 + [60.] * 10 + [70.] * 10
    second["median_ms"] = 60.  # Same median as first config; lower p20 wins.
    native["native_configs_timings"][1]["quantiles_ms"] = [60., 50., 70.]
    native["selected_candidate_id"] = second["candidate_id"]
    native["reference_fallback"]["probes"]["native"]["candidate_id"] = second["candidate_id"]
    row["confirmation"]["measurements"]["native"]["candidate_id"] = second["candidate_id"]
    assert audit_live_task(row, suite, cohort_id=cohort)["native_configs_checked"] == 24
    native["selected_candidate_id"] = native["records"][0]["candidate_id"]
    with pytest.raises(ValueError, match="native selected candidate"):
        audit_live_task(row, suite, cohort_id=cohort)
