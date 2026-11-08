"""Policy evaluation contracts use explicit synthetic fixtures, never GPU evidence."""
import hashlib
from itertools import count
import json

import pytest

from drivebench_tuning.evaluation import EvaluationError, evaluate_policy, grid_policy, random_policy
from drivebench_tuning.search import BudgetedTuningEnv, CandidateSpec, TaskSpec, TrialResult

TASKS = [TaskSpec(f"task-{i}", (float(i), 1.0), {"external_split": "heldout" if i == 2 else "train"})
         for i in range(3)]
CANDIDATES = [CandidateSpec(f"candidate-{i}", {"block": i + 1}) for i in range(4)]


class FixtureEvaluator:
    def __init__(self, ids, *, cohort="fixture-runtime", status="ok", crash=False, digest_suffix=""):
        self.ids, self.cohort, self.status = ids, cohort, status
        self.crash, self.digest_suffix = crash, digest_suffix
        self.calls = []

    def result(self, task, candidate=None):
        self.calls.append((task.task_id, None if candidate is None else candidate.candidate_id))
        if candidate is not None and self.crash:
            raise RuntimeError("fixture evaluator interrupted")
        status = self.status if candidate is not None else "ok"
        latency = 10.0 if candidate is None else 10.0 - candidate.config["block"]
        return TrialResult(
            str(next(self.ids)), task.task_id, None if candidate is None else candidate.candidate_id,
            "synthetic", self.cohort,
            hashlib.sha256((task.task_id + self.digest_suffix).encode()).hexdigest(),
            status, status == "ok", latency if status == "ok" else None,
            1000 if status == "ok" else None, {"fixture": True},
        )

    baseline = result

    def evaluate(self, task, candidate):
        return self.result(task, candidate)


def factory(*, budget=2, evaluator_options=None, env_options=None):
    ids = count()
    created = []

    def make():
        index = len(created)
        options = evaluator_options(index) if evaluator_options else {}
        evaluator = FixtureEvaluator(ids, **options)
        settings = env_options(index) if env_options else {}
        env = BudgetedTuningEnv(TASKS, CANDIDATES, evaluator, trial_budget=budget,
                                allow_synthetic=True, **settings)
        created.append(env)
        return env

    return make, created


def actions(report):
    return [[step["action"] for step in episode["steps"]] for episode in report["episodes"]]


def test_grid_equal_budgets_each_fresh_episode_and_external_split():
    make, created = factory()
    result = evaluate_policy(make, grid_policy, [2], repeats=3, seed=9, split="heldout")
    assert result["status"] == "completed"
    assert actions(result) == [[0, 1]] * 3
    assert result["counts"] == {"baseline_calls": 3, "candidate_calls": 6,
                                "failed_or_incorrect_trials": 0}
    assert [env.evaluator.calls for env in created] == [[("task-2", None), ("task-2", "candidate-0"),
                                                      ("task-2", "candidate-1")]] * 3
    assert len({id(env) for env in created}) == 3
    assert {episode["task_id"] for episode in result["episodes"]} == {"task-2"}
    assert result["split"] == "heldout"
    assert not result["claims"]["heldout_generalization"]
    assert result["per_task"][0]["median_observed_incumbent_cost"] == 0.8
    assert result["per_task"][0]["median_observed_incumbent_latency_ms"] == 8.0
    assert not result["selected_config_confirmation_performed"]
    assert result["confirmation_candidate_calls"] == result["confirmation_baseline_calls"] == 0
    assert not result["training"]["external_training_metadata_provided"]
    json.dumps(result, allow_nan=False)


def test_random_is_without_replacement_reproducible_and_resets_per_episode():
    policy = random_policy(3)
    make, _ = factory(budget=4)
    first = evaluate_policy(make, policy, [0, 2], repeats=2, seed=42)
    make_again, _ = factory(budget=4)
    second = evaluate_policy(make_again, policy, [0, 2], repeats=2, seed=42)
    assert actions(first) == actions(second)
    assert all(sorted(row) == [0, 1, 2, 3] for row in actions(first))
    assert len({tuple(row) for row in actions(first)}) > 1
    assert first["counts"]["candidate_calls"] == second["counts"]["candidate_calls"] == 16


def test_policy_gets_detached_state_and_full_trial_records_are_saved():
    def policy(observation, info):
        action = grid_policy(observation, info)
        observation["action_counts"].fill(99)
        info["baseline"]["latency_ms"] = 0
        return action
    make, _ = factory()
    result = evaluate_policy(make, policy, [0])
    episode = result["episodes"][0]
    assert actions(result) == [[0, 1]]
    assert episode["initial_info"]["baseline"]["latency_ms"] == 10
    assert [step["info"]["measurement"]["candidate_id"] for step in episode["steps"]] == ["candidate-0", "candidate-1"]
    assert len(episode["journal"]) == 3
    assert episode["steps"][0]["observation"]["action_counts"] == [0, 0, 0, 0]
    assert episode["steps"][1]["terminated"] is True


def test_factory_protocol_change_aborts_before_new_baseline_and_keeps_prior_evidence():
    make, created = factory(env_options=lambda index: {"memory_weight": float(index)})
    with pytest.raises(EvaluationError, match="factory changed") as exc:
        evaluate_policy(make, grid_policy, [0], repeats=2)
    result = exc.value.report
    assert result["status"] == "aborted"
    assert result["counts"]["baseline_calls"] == 1
    assert result["counts"]["candidate_calls"] == 2
    assert result["episodes"][0]["status"] == "completed"
    assert result["episodes"][1]["status"] == "failed"
    assert created[1].evaluator.calls == []
    json.dumps(result, allow_nan=False)


def test_new_factory_hardware_cohort_cannot_be_pooled():
    make, _ = factory(evaluator_options=lambda index: {"cohort": f"fixture-hardware-{index}"})
    with pytest.raises(EvaluationError, match="cohort") as exc:
        evaluate_policy(make, grid_policy, [0], repeats=2)
    result = exc.value.report
    assert result["counts"] == {"baseline_calls": 2, "candidate_calls": 2,
                                "failed_or_incorrect_trials": 0}
    assert len(result["episodes"][1]["journal"]) == 1
    assert result["episodes"][1]["initial_info"]["cohort_id"] == "fixture-hardware-1"
    assert result["per_task"][0]["episodes"] == 1


def test_same_task_input_changes_cannot_be_pooled():
    make, _ = factory(evaluator_options=lambda index: {"digest_suffix": str(index)})
    with pytest.raises(EvaluationError, match="input digest changed") as exc:
        evaluate_policy(make, grid_policy, [0], repeats=2)
    assert exc.value.report["counts"]["baseline_calls"] == 2
    assert exc.value.report["counts"]["candidate_calls"] == 2


def test_failed_results_consume_equal_budget_and_are_never_promoted():
    make, _ = factory(evaluator_options=lambda _: {"status": "incorrect"})
    result = evaluate_policy(make, grid_policy, [0], repeats=2)
    assert result["counts"] == {"baseline_calls": 2, "candidate_calls": 4,
                                "failed_or_incorrect_trials": 4}
    assert all(row["final_info"]["incumbent_candidate_id"] is None for row in result["episodes"])
    assert result["per_task"][0]["median_observed_incumbent_cost"] == 1.0
    assert all(step["reward"] == 0 for row in result["episodes"] for step in row["steps"])


def test_evaluator_exception_preserves_action_partial_journal_and_call_cost():
    make, _ = factory(evaluator_options=lambda _: {"crash": True})
    with pytest.raises(EvaluationError, match="interrupted") as exc:
        evaluate_policy(make, grid_policy, [0])
    result = exc.value.report
    assert result["counts"] == {"baseline_calls": 1, "candidate_calls": 1,
                                "failed_or_incorrect_trials": 1}
    assert result["episodes"][0]["steps"][0]["action"] == 0
    assert "reward" not in result["episodes"][0]["steps"][0]
    assert result["episodes"][0]["journal"][-1]["error"]["type"] == "RuntimeError"
    assert result["per_task"] == []
    json.dumps(result, allow_nan=False)


def test_reused_factory_environment_is_rejected_without_recounting_previous_calls():
    make, _ = factory()
    env = make()
    with pytest.raises(EvaluationError, match="fresh environment") as exc:
        evaluate_policy(lambda: env, grid_policy, [0], repeats=2)
    result = exc.value.report
    assert result["counts"]["baseline_calls"] == 1
    assert result["counts"]["candidate_calls"] == 2
    assert result["episodes"][1]["journal"] == []
    assert len(result["episodes"][1]["preexisting_journal"]) == 3


def test_trial_ids_must_be_fresh_across_environments():
    def make():
        return BudgetedTuningEnv(TASKS, CANDIDATES, FixtureEvaluator(count()),
                                 trial_budget=2, allow_synthetic=True)
    with pytest.raises(EvaluationError, match="trial ID reused") as exc:
        evaluate_policy(make, grid_policy, [0], repeats=2)
    assert exc.value.report["counts"]["candidate_calls"] == 2
    assert exc.value.report["counts"]["baseline_calls"] == 2
    assert exc.value.report["episodes"][1]["status"] == "failed"


@pytest.mark.parametrize("policy", [grid_policy, random_policy(1)])
def test_without_replacement_baselines_reject_budget_larger_than_action_space(policy):
    make, _ = factory(budget=5)
    with pytest.raises(EvaluationError, match="budget <= candidate count") as exc:
        evaluate_policy(make, policy, [0])
    assert exc.value.report["counts"]["baseline_calls"] == 1
    assert exc.value.report["counts"]["candidate_calls"] == 0


@pytest.mark.parametrize("indices,repeats,seed", [([], 1, 0), ([True], 1, 0), ([0, 0], 1, 0),
                                                  ([-1], 1, 0), ([0], 0, 0), ([0], 1, -1)])
def test_invalid_evaluation_arguments_do_not_construct_environment(indices, repeats, seed):
    make, created = factory()
    with pytest.raises(ValueError):
        evaluate_policy(make, grid_policy, indices, repeats, seed)
    assert created == []


def test_training_cost_metadata_is_external_and_never_assumed_zero():
    make, _ = factory()
    supplied = {"candidate_calls": 120, "baseline_calls": 30, "source": "caller training receipt"}
    result = evaluate_policy(make, grid_policy, [2], training_metadata=supplied)
    assert result["training"]["externally_supplied_metadata"] == supplied
    assert not result["training"]["performed_by_harness"]
    assert not result["training"]["external_training_cost_verified"]
    assert result["counts"]["candidate_calls"] == 2
