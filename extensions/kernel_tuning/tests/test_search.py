"""CPU-only, explicitly synthetic contract checks; no benchmark evidence."""

from dataclasses import FrozenInstanceError, replace
import json

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np
import pytest

from drivebench_tuning.search import BudgetedTuningEnv, CandidateSpec, TaskSpec, TrialResult


TASKS = (TaskSpec("small", (0.0, 1.0), {"shape": [2, 4]}),
         TaskSpec("large", (1.0, 4.0), {"shape": [8, 4]}))
CANDIDATES = tuple(CandidateSpec(name, {"implementation": name})
                   for name in ("moderate", "fast", "slow"))


class SyntheticEvaluator:
    """Invented measurements for exercising contracts, never GPU results."""

    def __init__(self, mutate=None, baseline_mutate=None):
        self.calls = []
        self.mutate = mutate
        self.baseline_mutate = baseline_mutate
        self.close_count = 0

    def result(self, task, candidate=None):
        self.calls.append((task.task_id, None if candidate is None else candidate.candidate_id))
        latency, memory = ((10.0, 1000) if candidate is None else
                           {"moderate": (8.0, 500), "fast": (5.0, 800), "slow": (12.0, 100)}[candidate.candidate_id])
        return TrialResult(
            f"fixture-{len(self.calls)}", task.task_id,
            None if candidate is None else candidate.candidate_id,
            "synthetic", "fixture-only", f"fixture-input-{task.task_id}",
            "ok", True, latency, memory, {"fixture": True},
        )

    def baseline(self, task):
        result = self.result(task)
        return self.baseline_mutate(result) if self.baseline_mutate else result

    def evaluate(self, task, candidate):
        result = self.result(task, candidate)
        return self.mutate(result) if self.mutate else result

    def close(self):
        self.close_count += 1


def make_env(evaluator=None, **kwargs):
    return BudgetedTuningEnv(TASKS, CANDIDATES, evaluator or SyntheticEvaluator(),
                            allow_synthetic=True, **kwargs)


def test_gymnasium_contract_with_fresh_nondeterministic_audit_ids():
    # Measurements and unique audit IDs need not repeat after a seeded reset.
    # Register honestly as nondeterministic, then separately test seeded task
    # selection and synthetic numeric transitions below.
    identifier = "DrivebenchSyntheticSearchContract-v0"
    if identifier not in gym.registry:
        gym.register(identifier, entry_point=make_env, nondeterministic=True)
    env = gym.make(identifier).unwrapped
    check_env(env, skip_render_check=True)
    observation, info = env.reset(seed=11)
    assert env.observation_space.contains(observation)
    assert info["synthetic"] is True
    env.close()


def test_seed_controls_task_selection_not_reuse_of_measurements():
    env = make_env()
    first, info1 = env.reset(seed=21)
    obs1, reward1, _, _, _ = env.step(0)
    second, info2 = env.reset(seed=21)
    obs2, reward2, _, _, _ = env.step(0)
    assert info1["task_id"] == info2["task_id"]
    assert info1["baseline"]["trial_id"] != info2["baseline"]["trial_id"]
    for key in first:
        np.testing.assert_equal(first[key], second[key])
        np.testing.assert_equal(obs1[key], obs2[key])
    assert reward1 == reward2


def test_improvement_reward_telescopes_and_repeating_incumbent_does_not_pay():
    evaluator = SyntheticEvaluator()
    env = make_env(evaluator, trial_budget=4)
    env.reset(options={"task_index": 0})
    rewards = []
    for action in (0, 0, 1, 2):
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        assert env.observation_space.contains(obs)
    assert rewards == pytest.approx([0.2, 0.0, 0.3, 0.0])
    assert sum(rewards) == pytest.approx(1.0 - info["incumbent_cost"])
    assert info["incumbent_candidate_id"] == "fast"
    assert info["incumbent_cost"] == 0.5
    assert terminated is True and truncated is False
    assert len(evaluator.calls) == 5  # Reference + four fresh calls, including repeat.
    assert obs["budget"].tolist() == [0.0]
    assert obs["action_counts"].tolist() == [2, 1, 1]
    assert len({entry["measurement"]["trial_id"] for entry in env.journal}) == 5
    with pytest.raises(RuntimeError, match="reset"):
        env.step(1)


def test_default_latency_objective_rejects_memory_only_tradeoff():
    env = make_env(trial_budget=1)
    env.reset()
    obs, reward, done, _, info = env.step(2)
    assert reward == 0.0 and done
    assert info["incumbent_candidate_id"] is None
    assert info["incumbent_cost"] == 1.0
    assert obs["incumbent_action"] == 0
    assert info["objective"]["memory_weight"] == 0.0
    assert info["objective"]["independent_confirmation"] is False


def test_explicit_memory_tradeoff_has_declared_weight_and_normalized_metrics():
    env = make_env(trial_budget=1, memory_weight=1.0)
    env.reset()
    obs, reward, _, _, info = env.step(2)
    assert info["incumbent_cost"] == pytest.approx(0.65)
    assert reward == pytest.approx(0.35)
    assert info["incumbent_candidate_id"] == "slow"
    np.testing.assert_allclose(obs["incumbent"], [1.2, 0.1])
    np.testing.assert_allclose(obs["action_metrics"][2], [1.2, 0.1])
    assert info["objective"]["memory_weight"] == 1.0


@pytest.mark.parametrize("status", ["incorrect", "error"])
def test_failure_consumes_budget_without_evicting_valid_incumbent(status):
    evaluator = SyntheticEvaluator()
    env = make_env(evaluator, trial_budget=2)
    env.reset()
    env.step(0)
    evaluator.mutate = lambda result: replace(result, status=status, correct=False,
                                               latency_ms=None, peak_allocated_bytes=None)
    obs, reward, done, truncated, info = env.step(0)
    assert reward == 0.0 and done and not truncated
    assert info["trials_used"] == 2 and info["incumbent_candidate_id"] == "moderate"
    assert obs["action_counts"][0] == 2
    assert obs["action_status"][0] == {"incorrect": 2, "error": 3}[status]
    np.testing.assert_array_equal(obs["action_metrics"][0], [0.0, 0.0])
    assert env.journal[-1]["accepted"] is True  # Valid failed result, not passing correctness.
    assert env.journal[-1]["measurement"]["correct"] is False


def test_observation_history_distinguishes_searches_with_same_budget_and_incumbent():
    left, right = make_env(trial_budget=3), make_env(trial_budget=3)
    left.reset(options={"task_index": 0})
    right.reset(options={"task_index": 0})
    left.step(1)
    right.step(1)
    a, *_ = left.step(0)
    b, *_ = right.step(2)
    np.testing.assert_array_equal(a["budget"], b["budget"])
    np.testing.assert_array_equal(a["incumbent"], b["incumbent"])
    assert not np.array_equal(a["action_counts"], b["action_counts"])
    assert not np.array_equal(a["action_metrics"], b["action_metrics"])


def test_descriptors_and_evidence_are_deeply_immutable_and_serializable():
    metadata = {"shape": [2, 4], "bounds": {"enabled": True}}
    task = TaskSpec("task", [1, 2], metadata)
    candidate = CandidateSpec("candidate", metadata)
    result = TrialResult("trial", "task", "candidate", "synthetic", "cohort", "input",
                         "ok", True, 1.0, 100, metadata)
    metadata["shape"][0] = 999
    metadata["bounds"]["enabled"] = False
    assert task.metadata["shape"] == candidate.config["shape"] == result.evidence["shape"] == (2, 4)
    assert task.metadata["bounds"]["enabled"] is True
    with pytest.raises(TypeError):
        candidate.config["new"] = 1
    with pytest.raises(TypeError):
        result.evidence["bounds"]["enabled"] = False
    with pytest.raises(FrozenInstanceError):
        task.task_id = "changed"
    for item in (task, candidate, result):
        payload = item.to_dict()
        json.dumps(payload, allow_nan=False)
        field = "metadata" if item is task else "config" if item is candidate else "evidence"
        payload[field]["shape"][0] = 999
        assert item.to_dict()[field]["shape"][0] == 2


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), object(), {1: "bad-key"}, {"x": {1, 2}}])
def test_descriptor_rejects_non_json_or_nonfinite_payload(invalid):
    with pytest.raises(ValueError):
        CandidateSpec("candidate", {"nested": [invalid]})


def test_descriptor_rejects_cycles_but_accepts_shared_noncyclic_values():
    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError, match="Cyclic"):
        TaskSpec("x", (1.0,), {"cyclic": cyclic})
    shared = [1, 2]
    assert CandidateSpec("x", {"a": shared, "b": shared}).to_dict()["config"] == {"a": [1, 2], "b": [1, 2]}


@pytest.mark.parametrize("features", [(), (float("nan"),), (float("inf"),), (True,), ("1",), None, "12"])
def test_task_features_are_nonempty_finite_numeric_sequences(features):
    with pytest.raises(ValueError):
        TaskSpec("task", features)


@pytest.mark.parametrize("change", [
    {"trial_id": ""}, {"task_id": " "}, {"candidate_id": ""},
    {"backend": "cpu"}, {"cohort_id": ""}, {"input_digest": ""},
    {"status": "unknown"}, {"correct": False}, {"correct": 1},
    {"status": "incorrect", "correct": True},
    {"latency_ms": float("nan")}, {"latency_ms": float("inf")},
    {"latency_ms": 0}, {"latency_ms": -1}, {"latency_ms": True},
    {"latency_ms": None}, {"peak_allocated_bytes": None},
    {"peak_allocated_bytes": -1}, {"peak_allocated_bytes": 1.5},
    {"peak_allocated_bytes": True}, {"evidence": {"metric": float("inf")}},
])
def test_trial_contract_rejects_bad_identity_metrics_and_correctness(change):
    good = SyntheticEvaluator().result(TASKS[0])
    with pytest.raises(ValueError):
        replace(good, **change)


@pytest.mark.parametrize("change", [
    {"task_id": "wrong"}, {"candidate_id": "wrong"}, {"candidate_id": None},
    {"backend": "cuda"}, {"cohort_id": "different"}, {"input_digest": "different"},
])
def test_protocol_mismatch_is_retained_and_aborts_episode(change):
    evaluator = SyntheticEvaluator(mutate=lambda result: replace(result, **change))
    env = make_env(evaluator)
    env.reset()
    with pytest.raises(ValueError):
        env.step(0)
    assert len(env.journal) == 2
    assert env.journal[-1]["accepted"] is False
    assert "validation_error" in env.journal[-1]
    assert env.journal[-1]["trials_used"] == 1
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)


def test_synthetic_results_require_explicit_opt_in():
    env = BudgetedTuningEnv(TASKS, CANDIDATES, SyntheticEvaluator())
    with pytest.raises(ValueError, match="synthetic"):
        env.reset()
    assert env.journal[0]["accepted"] is False


@pytest.mark.parametrize("change", [{"backend": "cuda"}, {"cohort_id": "other"}, {"input_digest": "other"}])
def test_reference_binding_cannot_change_across_resets(change):
    evaluator = SyntheticEvaluator()
    env = make_env(evaluator)
    env.reset(options={"task_index": 0})
    evaluator.baseline_mutate = lambda result: replace(result, **change)
    with pytest.raises(ValueError):
        env.reset(options={"task_index": 0})
    assert len(env.journal) == 2


def test_each_task_may_have_a_different_fixed_input_digest():
    env = make_env()
    _, a = env.reset(options={"task_index": 0})
    _, b = env.reset(options={"task_index": 1})
    _, c = env.reset(options={"task_index": 0})
    assert a["input_digest"] != b["input_digest"]
    assert a["input_digest"] == c["input_digest"]


def test_duplicate_ids_are_rejected_across_episode_boundaries():
    evaluator = SyntheticEvaluator(mutate=lambda result: replace(result, trial_id="same"))
    env = make_env(evaluator, trial_budget=1)
    env.reset()
    env.step(0)
    env.reset()
    with pytest.raises(ValueError, match="fresh result"):
        env.step(0)
    assert len(env.journal) == 4


@pytest.mark.parametrize("change", [
    {"candidate_id": "unexpected"}, {"peak_allocated_bytes": 0},
    {"status": "incorrect", "correct": False},
])
def test_reference_must_be_admissible_before_search(change):
    evaluator = SyntheticEvaluator(baseline_mutate=lambda result: replace(result, **change))
    env = make_env(evaluator)
    with pytest.raises(ValueError):
        env.reset()
    with pytest.raises(RuntimeError):
        env.step(0)


def test_unexpected_exception_is_not_fabricated_into_a_gpu_trial():
    class Crashing(SyntheticEvaluator):
        def evaluate(self, task, candidate):
            raise RuntimeError("fixture failure")

    env = make_env(Crashing())
    env.reset()
    with pytest.raises(RuntimeError, match="fixture failure"):
        env.step(0)
    last = env.journal[-1]
    assert last["error"]["type"] == "RuntimeError"
    assert last["trials_used"] == 1 and last["accepted"] is False
    assert "measurement" not in last
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)


def test_wrong_return_type_is_preserved_without_becoming_valid_evidence():
    env = make_env(SyntheticEvaluator(mutate=lambda result: {"pretend": "measurement"}))
    env.reset()
    with pytest.raises(ValueError, match="TrialResult"):
        env.step(0)
    assert env.journal[-1]["returned_type"] == "dict"


def test_ratio_overflow_aborts_instead_of_exposing_infinite_observation():
    evaluator = SyntheticEvaluator(baseline_mutate=lambda result: replace(result, latency_ms=1e-300),
                                   mutate=lambda result: replace(result, latency_ms=1e300))
    env = make_env(evaluator)
    env.reset()
    with pytest.raises(ValueError, match="normalized metrics"):
        env.step(0)
    assert not env.journal[-1]["accepted"]


@pytest.mark.parametrize("action", [-1, 3, 1.5, True, "0", None])
def test_invalid_action_does_not_spend_budget_or_call_evaluator(action):
    evaluator = SyntheticEvaluator()
    env = make_env(evaluator)
    env.reset()
    with pytest.raises(ValueError, match="action"):
        env.step(action)
    assert len(evaluator.calls) == 1
    _, _, _, _, info = env.step(np.int64(0))
    assert info["trials_used"] == 1


@pytest.mark.parametrize("kwargs", [
    {"trial_budget": 0}, {"trial_budget": True}, {"trial_budget": 1.5},
    {"memory_weight": -1}, {"memory_weight": float("nan")},
    {"memory_weight": float("inf")},
])
def test_invalid_budget_or_objective_is_rejected(kwargs):
    with pytest.raises(ValueError):
        make_env(**kwargs)


def test_duplicate_descriptor_ids_and_mismatched_feature_dimensions_rejected():
    for tasks, candidates in [
        ((TASKS[0], TASKS[0]), CANDIDATES),
        (TASKS, (CANDIDATES[0], CANDIDATES[0])),
        ((TASKS[0], TaskSpec("different", (1.0,))), CANDIDATES),
    ]:
        with pytest.raises(ValueError):
            BudgetedTuningEnv(tasks, candidates, SyntheticEvaluator())


@pytest.mark.parametrize("options", [{"task_index": -1}, {"task_index": 2}, {"task_index": True}, {"extra": 0}])
def test_invalid_reset_options_call_no_evaluator(options):
    evaluator = SyntheticEvaluator()
    env = make_env(evaluator)
    with pytest.raises(ValueError):
        env.reset(options=options)
    assert evaluator.calls == []


def test_journal_and_returned_observations_cannot_mutate_internal_state():
    env = make_env()
    obs, info = env.reset()
    obs["action_counts"][0] = 1000
    info["baseline"]["latency_ms"] = 999
    log = env.journal
    log[0]["measurement"]["latency_ms"] = 999
    obs, reward, _, _, info = env.step(0)
    assert reward == pytest.approx(0.2)
    assert obs["action_counts"][0] == 1
    assert info["baseline"]["latency_ms"] == 10.0
    assert env.journal[0]["measurement"]["latency_ms"] == 10.0
    env.reset()
    assert len(env.journal) == 3


def test_close_is_idempotent_and_disallows_further_evaluation():
    evaluator = SyntheticEvaluator()
    env = make_env(evaluator)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)
    env.close()
    env.close()
    assert evaluator.close_count == 1
    with pytest.raises(RuntimeError, match="closed"):
        env.reset()
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)


def test_failed_cleanup_can_be_retried_without_reopening_evaluation():
    class FailsOnce(SyntheticEvaluator):
        def close(self):
            self.close_count += 1
            if self.close_count == 1:
                raise RuntimeError("fixture cleanup failure")

    evaluator = FailsOnce()
    env = make_env(evaluator)
    with pytest.raises(RuntimeError, match="cleanup failure"):
        env.close()
    with pytest.raises(RuntimeError, match="closed"):
        env.reset()
    env.close()
    env.close()
    assert evaluator.close_count == 2
    with pytest.raises(RuntimeError, match="closed"):
        env.reset()
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)
