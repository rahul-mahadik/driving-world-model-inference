"""CPU contracts only: these fabricated timings are never benchmark evidence."""
from dataclasses import asdict, replace
from collections import deque

import gymnasium as gym
import numpy as np
import pytest

import drivebench_tuning
from drivebench_tuning.contract import Workload
from drivebench_tuning.schedule import ACTIONS, KernelSchedule, action_mask, transform
from drivebench_tuning.schedule_env import ScheduleTuningEnv
from drivebench_tuning.schedule_measurement import ScheduleMeasurement


GEGLU = Workload("geglu", (8, 128))
GN = Workload("groupnorm_silu", (1, 32, 8, 8), num_groups=8)


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


class FixtureEvaluator:
    def __init__(self, clock):
        self.clock = clock
        self.count = 0
        self.results = []
        self.calls = []
        self.closed = 0
        self.latencies = deque([0.8, 1.2, 0.7, 0.9])
        self.duration = 0.5
        self.mutate = lambda result: result

    def _result(self, workload, schedule, latency, duration):
        self.clock.value += duration
        self.count += 1
        result = ScheduleMeasurement(workload.key(), schedule, "ok", True, latency,
            [latency, latency + 0.1, latency + 0.2], 1024, duration,
            {"trial_id": f"fixture-{self.count}", "runtime": {"gpu_uuid": "TEST-ONLY", "compute_capability": [8, 9]},
             "cohort_id": "fixture", "input_sha256": ["a" * 64]}, backend="test")
        self.results.append(result)
        return result

    def baseline(self, workload, *, timeout_seconds):
        return self._result(workload, None, 1.0, 2.0)

    def evaluate(self, workload, schedule, *, timeout_seconds):
        self.calls.append((schedule, timeout_seconds))
        return self.mutate(self._result(workload, asdict(schedule), self.latencies.popleft(), self.duration))

    def close(self):
        self.closed += 1


@pytest.fixture
def fixture(monkeypatch):
    clock = Clock()
    monkeypatch.setattr("drivebench_tuning.schedule_env.time.monotonic", clock)
    evaluator = FixtureEvaluator(clock)
    return clock, evaluator


def make(fixture, **kwargs):
    return ScheduleTuningEnv(workloads=[GEGLU], evaluator=fixture[1],
                             allow_test_evaluator=True, **kwargs)


def act(env, name):
    return env.step(ACTIONS.index(name))


def test_edits_are_stateful_and_order_changes_lowering():
    initial = KernelSchedule()
    first = transform(GEGLU, transform(GEGLU, initial, "block_up"), "unfuse")
    second = transform(GEGLU, transform(GEGLU, initial, "unfuse"), "block_up")
    assert first.block_size == second.block_size == 512
    assert first.activation_block_size == 512 and second.activation_block_size == 256
    assert initial == KernelSchedule()
    assert not action_mask(GEGLU, initial)[ACTIONS.index("activation_block_up")]
    assert action_mask(GEGLU, second)[ACTIONS.index("activation_block_up")]
    fused = transform(GEGLU, first, "fuse")
    assert fused.activation_block_size == 256


@pytest.mark.parametrize("workload,expected", [(GEGLU, 812), (GN, 25984)])
def test_reachable_canonical_program_counts(workload, expected):
    seen = {KernelSchedule()}
    queue = deque(seen)
    while queue:
        current = queue.popleft()
        for action in range(len(ACTIONS) - 2):
            try:
                next_state = transform(workload, current, action)
            except ValueError:
                continue
            if next_state not in seen:
                seen.add(next_state)
                queue.append(next_state)
    assert len(seen) == expected


def test_group_size_changes_action_mask():
    large = Workload("groupnorm_silu", (1, 32, 256, 256), num_groups=1)
    # Default 256-value chunks exceed the chunk limit; construct a legal start.
    schedule = KernelSchedule(reduction_block_size=1024)
    assert not action_mask(large, schedule)[ACTIONS.index("reduction_single")]
    assert action_mask(GN, KernelSchedule())[ACTIONS.index("reduction_single")]


def test_gym_registration_spaces_and_complete_episode(fixture):
    env = gym.make("DriveBench/KernelSchedule-v0", workloads=[GEGLU],
                   evaluator=fixture[1], allow_test_evaluator=True, budget_seconds=10)
    obs, info = env.reset(seed=10)
    assert env.observation_space.contains(obs)
    assert info["test_fixture"] and not info["fresh_gpu_measurement"]
    assert info["elapsed_seconds"] == 2
    for name in ("block_up", "unfuse", "activation_warps_down", "measure"):
        obs, reward, terminated, truncated, info = act(env, name)
        assert env.observation_space.contains(obs)
        assert not terminated and not truncated and reward <= 0
    obs, reward, terminated, truncated, info = act(env, "submit")
    assert terminated and not truncated
    assert info["submission_score"] == pytest.approx(-0.2)
    assert not obs["action_mask"].any()
    assert len(fixture[1].calls) == 2
    env.close()


def test_remeasurement_returns_new_sample_without_improvement_reward(fixture):
    env = make(fixture, budget_seconds=10)
    env.reset()
    _, first_reward, _, _, a = act(env, "measure")
    _, second_reward, _, _, b = act(env, "measure")
    assert a["measurement"]["latency_ms"] == 0.8
    assert b["measurement"]["latency_ms"] == 1.2
    assert a["measurement"]["evidence"]["trial_id"] != b["measurement"]["evidence"]["trial_id"]
    assert first_reward < 0 and second_reward < 0
    _, reward, term, trunc, info = act(env, "submit")
    assert term and not trunc and reward > 0
    assert info["submission_score"] == pytest.approx(0.3)
    assert len(fixture[1].calls) == 3  # submit cannot reuse the favorable trial


def test_illegal_transform_costs_time_and_preserves_program(fixture):
    env = make(fixture, budget_seconds=10)
    env.reset()
    before = env.schedule
    _, reward, _, _, info = act(env, "activation_block_up")
    assert env.schedule == before and not fixture[1].calls
    assert info["outcome"] == "invalid_action" and reward < -0.01
    parts = info["reward_components"]
    assert reward == parts["speed_score"] - parts["elapsed_penalty"] - parts["invalid_action_penalty"] - parts["failed_measurement_penalty"]
    assert info["elapsed_seconds"] >= 2.001


def test_policy_think_time_consumes_budget_without_new_gpu_call(fixture):
    env = make(fixture, budget_seconds=10)
    env.reset()
    fixture[0].value += 8.5
    _, reward, term, trunc, info = act(env, "submit")
    assert trunc and not term and reward < 0
    assert not fixture[1].calls and info["submission_score"] == 0
    with pytest.raises(RuntimeError, match="reset"):
        act(env, "measure")


def test_late_success_cannot_earn_reward(fixture):
    env = make(fixture, budget_seconds=3)
    env.reset()
    fixture[1].duration = 2
    _, reward, term, trunc, info = act(env, "submit")
    assert fixture[1].calls[0][1] == 1
    assert trunc and not term and reward < 0 and info["submission_score"] == 0


def test_failed_compile_consumes_cost_but_keeps_schedule(fixture):
    env = make(fixture, budget_seconds=10)
    env.reset()
    act(env, "unfuse")
    before = env.schedule
    fixture[1].mutate = lambda m: replace(m, status="error", correct=False, latency_ms=None, raw_samples_ms=[])
    _, reward, term, trunc, info = act(env, "measure")
    assert env.schedule == before and reward < -0.1 and not term and not trunc
    assert info["elapsed_seconds"] >= 2.501
    fixture[1].mutate = lambda m: m
    act(env, "fuse")
    assert env.schedule.fused


@pytest.mark.parametrize("field,value", [("runtime", {"gpu_uuid": "ANOTHER"}),
    ("cohort_id", "changed"), ("input_sha256", ["b" * 64]), ("trial_id", "fixture-1")])
def test_binding_changes_and_reused_evidence_fail_closed(fixture, field, value):
    env = make(fixture, budget_seconds=10)
    env.reset()
    fixture[1].mutate = lambda m: replace(m, evidence={**m.evidence, field: value})
    _, reward, term, trunc, info = act(env, "submit")
    assert term and not trunc and reward < 0 and info["submission_score"] == 0
    assert info["outcome"] == "evaluator_error"


@pytest.mark.parametrize("change", [dict(latency_ms=123), dict(raw_samples_ms=[float("nan")]),
    dict(correct=False), dict(schedule={}), dict(elapsed_seconds=-1)])
def test_malformed_results_never_score(fixture, change):
    env = make(fixture)
    env.reset()
    fixture[1].mutate = lambda m: replace(m, **change)
    _, reward, term, _, info = act(env, "submit")
    assert term and reward < 0 and info["submission_score"] == 0


def test_no_implicit_cpu_or_table_fallback(fixture):
    with pytest.raises(ValueError, match="evidence_dir"):
        gym.make("DriveBench/KernelSchedule-v0", workloads=[GEGLU])
    env = ScheduleTuningEnv(workloads=[GEGLU], evaluator=fixture[1])
    with pytest.raises(ValueError, match="live CUDA"):
        env.reset()
    assert fixture[1].closed


def test_reference_time_is_charged_and_expired_reset_fails(fixture):
    env = make(fixture, budget_seconds=1)
    with pytest.raises(RuntimeError, match="budget"):
        env.reset()
    assert any(row["event"] == "reset_failed" for row in env.journal)


def test_info_cannot_mutate_evaluator_or_journal(fixture):
    env = make(fixture)
    _, info = env.reset()
    info["baseline"]["evidence"]["runtime"]["gpu_uuid"] = "edited"
    assert env.baseline.evidence["runtime"]["gpu_uuid"] == "TEST-ONLY"
    assert env.journal[1]["measurement"]["evidence"]["runtime"]["gpu_uuid"] == "TEST-ONLY"


def test_edits_invalidate_displayed_measurement(fixture):
    env = make(fixture)
    env.reset()
    obs, *_ = act(env, "measure")
    assert obs["measurement"][-1] == 1
    obs, *_ = act(env, "block_up")
    assert not obs["measurement"].any()


@pytest.mark.parametrize("budget", [0, -1, float("nan"), float("inf"), True])
def test_rejects_invalid_budgets(fixture, budget):
    with pytest.raises(ValueError):
        make(fixture, budget_seconds=budget)


def test_evidence_directory_never_overwrites_existing_data(fixture, tmp_path):
    env = make(fixture, evidence_dir=tmp_path / "run")
    env.reset()
    act(env, "submit")
    assert list((tmp_path / "run").glob("episode-*/environment.jsonl"))
    with pytest.raises(FileExistsError):
        make(fixture, evidence_dir=tmp_path / "run")


def test_validation_error_charges_call_once(fixture):
    env = make(fixture)
    env.reset()
    fixture[1].mutate = lambda m: replace(m, schedule={})
    _, _, _, _, info = act(env, "measure")
    assert info["elapsed_seconds"] == 2.5


def test_cleanup_overrun_cancels_submission_score(fixture):
    env = make(fixture, budget_seconds=3)
    env.reset()
    def close():
        fixture[0].value += 1
    fixture[1].close = close
    _, reward, term, trunc, info = act(env, "submit")
    assert trunc and not term and reward < 0
    assert info["submission_score"] == 0 and info["elapsed_seconds"] == 3.5


def test_reaped_worker_ends_episode_immediately(fixture):
    env = make(fixture)
    env.reset()
    fixture[1].mutate = lambda m: replace(m, status="timeout", correct=False,
        latency_ms=None, raw_samples_ms=[], evidence={**m.evidence, "worker_reaped": True})
    _, reward, term, trunc, info = act(env, "measure")
    assert term and not trunc and reward < 0 and info["outcome"] == "timeout"
    with pytest.raises(RuntimeError):
        act(env, "block_up")


def test_large_group_selects_legal_initial_stats_block(fixture):
    large = Workload("groupnorm_silu", (1, 32, 256, 256), num_groups=1)
    env = ScheduleTuningEnv(workloads=[large], evaluator=fixture[1], allow_test_evaluator=True)
    env.reset()
    assert env.schedule.reduction_block_size == 512


def test_standard_gymnasium_environment_checker(fixture):
    from gymnasium.utils.env_checker import check_env
    env = gym.make("DriveBench/KernelSchedule-v0", workloads=[GEGLU],
                   evaluator=fixture[1], allow_test_evaluator=True)
    try:
        check_env(env.unwrapped, skip_render_check=True)
    finally:
        env.close()
