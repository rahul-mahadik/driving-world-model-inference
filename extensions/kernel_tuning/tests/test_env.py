"""CPU interface tests use explicitly synthetic measurements, never GPU evidence."""

from dataclasses import replace
import json

from gymnasium.utils.env_checker import check_env
import numpy as np
import pytest

from drivebench_tuning.contract import KernelConfig, Measurement, Workload
from drivebench_tuning.env import KernelTuningEnv
from drivebench_tuning.policy import evaluate_policies, train_q_learning


WORKLOAD = Workload("geglu", (3, 256))
CONFIGS = (KernelConfig(128, 4), KernelConfig(256, 4), KernelConfig(512, 8))


class SyntheticEvaluator:
    """Deterministic invented numbers solely for testing control flow."""

    def __init__(self, mutate=None, backend="synthetic", cohort="fixture-only"):
        self.calls = []
        self.mutate = mutate
        self.backend = backend
        self.cohort = cohort
        self.closed = False

    def baseline(self, workload):
        self.calls.append(("baseline", workload.key()))
        return Measurement(workload.key(), None, "ok", 10.0, 1000, True,
                           self.backend, {"cohort_id": self.cohort, "fixture": True})

    def evaluate(self, workload, config):
        self.calls.append(("candidate", workload.key(), config))
        index = CONFIGS.index(config)
        result = Measurement(workload.key(), config, "ok", [9.0, 5.0, 13.0][index],
                             [900, 600, 1500][index], True, self.backend,
                             {"cohort_id": self.cohort, "fixture": True})
        return self.mutate(result) if self.mutate else result

    def close(self):
        self.closed = True


def make_env(evaluator=None, **kwargs):
    return KernelTuningEnv([WORKLOAD], CONFIGS, evaluator or SyntheticEvaluator(),
                           allow_synthetic=True, **kwargs)


def test_gymnasium_contract_and_observations():
    env = make_env(trial_budget=3)
    check_env(env, skip_render_check=True)
    observation, info = env.reset(seed=7)
    assert env.observation_space.contains(observation)
    assert observation.dtype == np.float32
    assert info["synthetic"] is True


def test_reward_budget_and_remeasurement():
    evaluator = SyntheticEvaluator()
    env = make_env(evaluator, trial_budget=2, memory_weight=1.0)
    env.reset(seed=3)
    _, reward, terminated, truncated, info = env.step(1)
    assert reward == pytest.approx(0.45)  # 1 - (0.5 latency + 0.6 memory) / 2.
    assert info["incumbent"]["config"] == {"block_size": 256, "num_warps": 4}
    assert info["incumbent_cost"] == pytest.approx(0.55)
    assert not terminated and not truncated
    _, _, terminated, truncated, info = env.step(np.int64(1))
    assert truncated and not terminated and info["trials_used"] == 2
    assert len(evaluator.calls) == 3  # No action-result caching.
    assert len(env.history) == 2
    assert len(env.journal) == 3
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)
    env.reset()
    assert len(env.journal) == 4 and env.history == []


@pytest.mark.parametrize("status", ["incorrect", "error"])
def test_failed_measurements_consume_budget_and_cannot_win(status):
    evaluator = SyntheticEvaluator(lambda m: replace(m, status=status, correct=False,
                                                    latency_ms=None, peak_allocated_bytes=None))
    env = make_env(evaluator, trial_budget=1)
    env.reset()
    _, reward, _, truncated, info = env.step(0)
    assert reward == -2.0 and np.isfinite(reward)
    assert truncated and info["incumbent"]["config"] is None
    assert info["measurement"]["status"] == status


def test_slower_candidate_does_not_replace_eager_incumbent():
    env = make_env(trial_budget=1)
    env.reset()
    _, reward, _, _, info = env.step(2)
    assert reward < 0 and info["incumbent_action"] == -1


@pytest.mark.parametrize("action", [-1, 3, 0.1, True, "0", None])
def test_invalid_actions_do_not_call_evaluator(action):
    evaluator = SyntheticEvaluator()
    env = make_env(evaluator)
    env.reset()
    with pytest.raises(ValueError, match="action"):
        env.step(action)
    assert len(evaluator.calls) == 1


@pytest.mark.parametrize("backend", ["synthetic", "cpu", "mps"])
def test_non_cuda_measurements_are_denied_by_default(backend):
    env = KernelTuningEnv([WORKLOAD], CONFIGS, SyntheticEvaluator(backend=backend))
    with pytest.raises(ValueError, match="CUDA"):
        env.reset()
    assert len(env.journal) == 1


@pytest.mark.parametrize("change", [
    {"workload_key": "different"}, {"config": CONFIGS[1]},
    {"latency_ms": float("nan")}, {"latency_ms": float("inf")},
    {"latency_ms": -1}, {"latency_ms": 0}, {"latency_ms": True},
    {"peak_allocated_bytes": -1}, {"peak_allocated_bytes": 2.5},
    {"peak_allocated_bytes": None}, {"correct": False},
    {"status": "unknown"}, {"evidence": {"cohort_id": "other GPU"}},
    {"evidence": {}}, {"backend": "cuda"},
])
def test_malformed_or_out_of_cohort_feedback_is_rejected(change):
    env = make_env(SyntheticEvaluator(lambda m: replace(m, **change)))
    env.reset()
    with pytest.raises(ValueError):
        env.step(0)
    assert len(env.journal) == 2  # Rejected response is still retained.
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)


def test_baseline_must_be_correct_and_have_positive_memory():
    class InvalidBaseline(SyntheticEvaluator):
        def baseline(self, workload):
            return replace(super().baseline(workload), peak_allocated_bytes=0)

    with pytest.raises(ValueError, match="baseline"):
        make_env(InvalidBaseline()).reset()


def test_step_requires_reset_and_close_is_idempotent():
    evaluator = SyntheticEvaluator()
    env = make_env(evaluator)
    with pytest.raises(RuntimeError):
        env.step(0)
    env.close()
    env.close()
    assert evaluator.closed
    with pytest.raises(RuntimeError, match="closed"):
        env.reset()


def test_q_learning_and_equal_budget_comparison_use_fresh_measurements():
    training_evaluator = SyntheticEvaluator()
    training_env = make_env(training_evaluator, trial_budget=3)
    training = train_q_learning(training_env, episodes=8, seed=5)
    assert training.candidate_measurements == 24
    assert training.baseline_measurements == 8
    assert len(training_evaluator.calls) == 32
    assert all(record["synthetic"] for record in training.episodes)
    assert training.policy.q_values
    assert training.to_dict()["policy"]["generalization_claim"] is False
    evaluators = []

    def factory():
        evaluator = SyntheticEvaluator()
        evaluators.append(evaluator)
        return make_env(evaluator, trial_budget=3)

    comparison = evaluate_policies(factory, training.policy, episodes_per_workload=2, seed=9)
    assert len(comparison["records"]) == 6
    assert all(record["candidate_measurements"] == 3 for record in comparison["records"])
    assert all(len(evaluator.calls) == 4 and evaluator.closed for evaluator in evaluators)
    assert all(record["synthetic"] for record in comparison["records"])
    assert comparison["training_cost_included"] is False
    assert {record["policy"] for record in comparison["records"]} == {"learned", "random", "grid"}


def test_policy_comparison_rejects_different_cohorts():
    training = train_q_learning(make_env(trial_budget=1), episodes=1)
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        return make_env(SyntheticEvaluator(cohort=f"cohort-{calls}"), trial_budget=1)

    with pytest.raises(ValueError, match="cohort"):
        evaluate_policies(factory, training.policy)


def test_replayed_trial_id_is_rejected_even_after_reset():
    evaluator = SyntheticEvaluator(lambda m: replace(m, evidence={"cohort_id": "fixture-only", "trial_id": "reused"}))
    env = make_env(evaluator, trial_budget=1)
    env.reset()
    env.step(0)
    env.reset()
    with pytest.raises(ValueError, match="new trial"):
        env.step(0)


def test_unexpected_evaluator_exception_is_recorded_and_requires_reset():
    class CrashingEvaluator(SyntheticEvaluator):
        def evaluate(self, workload, config):
            raise RuntimeError("fixture failure")

    env = make_env(CrashingEvaluator())
    env.reset()
    with pytest.raises(RuntimeError, match="fixture failure"):
        env.step(0)
    assert env.journal[-1]["evaluator_error"]["type"] == "RuntimeError"
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)


def test_policy_comparison_rejects_reordered_configuration_actions():
    training = train_q_learning(make_env(trial_budget=1), episodes=1)

    def factory():
        return KernelTuningEnv([WORKLOAD], tuple(reversed(CONFIGS)), SyntheticEvaluator(),
                               trial_budget=1, allow_synthetic=True)

    with pytest.raises(ValueError, match="configurations"):
        evaluate_policies(factory, training.policy)


class CudaContractFixture(SyntheticEvaluator):
    """Fabricated contract rows for validation tests; this does not run a GPU."""

    def __init__(self, evidence_changes=None):
        super().__init__(backend="cuda")
        self.evidence_changes = evidence_changes or {}
        self.counter = 0

    def bind(self, measurement):
        self.counter += 1
        evidence = {**measurement.evidence, "trial_id": f"contract-fixture-{self.counter}",
                    "input_sha256": ["0" * 64], **self.evidence_changes}
        return replace(measurement, evidence=evidence)

    def baseline(self, workload):
        return self.bind(super().baseline(workload))

    def evaluate(self, workload, config):
        return self.bind(super().evaluate(workload, config))


@pytest.mark.parametrize("changes", [
    {"cohort_id": None}, {"cohort_id": ""}, {"cohort_id": 1},
    {"trial_id": None}, {"trial_id": ""}, {"trial_id": 1},
    {"input_sha256": None}, {"input_sha256": []}, {"input_sha256": "0" * 64},
    {"input_sha256": [""]}, {"input_sha256": ["not a hash"]},
])
def test_cuda_contract_requires_complete_provenance(changes):
    env = KernelTuningEnv([WORKLOAD], CONFIGS, CudaContractFixture(changes))
    with pytest.raises(ValueError, match="CUDA evidence"):
        env.reset()


def test_cuda_contract_fixture_requires_input_identity_and_fresh_trial_ids():
    evaluator = CudaContractFixture()
    env = KernelTuningEnv([WORKLOAD], CONFIGS, evaluator)
    env.reset()
    env.step(0)
    evaluator.evidence_changes = {"input_sha256": ["1" * 64]}
    with pytest.raises(ValueError, match="input hashes"):
        env.step(1)


def test_shared_evaluator_training_evaluation_and_json_serialization(tmp_path):
    class ReusableSyntheticEvaluator(SyntheticEvaluator):
        """Matches production close semantics: release state, keep cohort/budget/log."""

        def __init__(self):
            super().__init__()
            self.sequence = 0
            self.close_count = 0
            self.absolute_budget_token = object()

        def bind(self, measurement):
            self.sequence += 1
            return replace(measurement, evidence={**measurement.evidence,
                           "input_sha256": ["0" * 64], "trial_id": f"synthetic-{self.sequence}"})

        def baseline(self, workload):
            return self.bind(super().baseline(workload))

        def evaluate(self, workload, config):
            return self.bind(super().evaluate(workload, config))

        def close(self):
            self.close_count += 1

    evaluator = ReusableSyntheticEvaluator()
    budget_owner = evaluator.absolute_budget_token

    def factory():
        return make_env(evaluator, trial_budget=3)

    training_env = factory()
    training = train_q_learning(training_env, episodes=3, seed=16)
    training_env.close()
    comparison = evaluate_policies(factory, training.policy, episodes_per_workload=2, seed=17)
    assert evaluator.absolute_budget_token is budget_owner
    assert evaluator.close_count == 7  # Training, then six independent evaluation environments.
    assert evaluator.sequence == 36  # Nine baselines and 27 candidate measurements.
    assert len(evaluator.calls) == 36
    training_path = tmp_path / "synthetic-training-fixture.json"
    evaluation_path = tmp_path / "synthetic-evaluation-fixture.json"
    training_path.write_text(json.dumps(training.to_dict(), allow_nan=False))
    evaluation_path.write_text(json.dumps(comparison, allow_nan=False))
    training_data = json.loads(training_path.read_text())
    evaluation_data = json.loads(evaluation_path.read_text())
    rows = training_data["episodes"] + evaluation_data["records"]
    measurements = [m for row in rows for m in [row["baseline"], *row["measurements"]]]
    assert len({m["evidence"]["trial_id"] for m in measurements}) == 36
    assert all(m["backend"] == "synthetic" and m["evidence"]["fixture"] for m in measurements)
    assert all(row["synthetic"] for row in rows)
    assert training_data["candidate_measurements"] == 9
    assert {row["candidate_measurements"] for row in evaluation_data["records"]} == {3}
    assert evaluation_data["training_cost_included"] is False
    assert evaluation_data["synthetic"] is True
    assert evaluation_data["rl_improvement_validated"] is False
    assert evaluation_data["selected_config_confirmation_performed"] is False
