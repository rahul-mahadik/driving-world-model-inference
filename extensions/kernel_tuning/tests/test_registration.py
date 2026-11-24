"""Gymnasium integration; fabricated contract rows are test fixtures only."""

import importlib

import gymnasium as gym
from gymnasium.utils.env_checker import check_env, data_equivalence
import numpy as np
import pytest

import drivebench_tuning
from drivebench_tuning import table_collection as tc
from drivebench_tuning.replay import MeasuredTable
from test_table_collection import FixtureRunner


ENV_ID = "DriveBench/OperatorTune-v0"


@pytest.fixture(scope="module")
def table_path(tmp_path_factory):
    root = tmp_path_factory.mktemp("registration-contract-fixture")
    paths = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(tc, "source_hashes", lambda: {"fixture.py": "f" * 64})
        for repeat in range(3):
            patch.setattr(tc, "_PROCESS_INSTANCE_ID", f"CPU-contract-only-{repeat}")
            path = root / f"repeat-{repeat}"
            assert tc.collect(
                path, suite="operator-generalization-v2", repeat_index=repeat,
                repetitions=3, runner_factory=FixtureRunner,
            ) == 0
            paths.append(path)
        output = root / "table.json"
        tc.merge(paths, output)
    return output


def test_registration_is_idempotent_and_requires_explicit_table():
    spec = gym.spec(ENV_ID)
    importlib.reload(drivebench_tuning)
    assert gym.spec(ENV_ID) is spec
    assert spec.max_episode_steps is None
    with pytest.raises(TypeError, match="table_path"):
        gym.make(ENV_ID)


def test_default_suite_spaces_and_standard_checker(table_path):
    env = gym.make(ENV_ID, table_path=table_path)
    try:
        assert len(env.unwrapped.tasks) == 30
        assert all(task.metadata["split"] == "train" for task in env.unwrapped.tasks)
        assert env.unwrapped.trial_budget == 8
        assert env.unwrapped.memory_weight == 0
        assert env.unwrapped.suite.document["suite_id"] == "operator-generalization-v2"
        assert env.action_space == gym.spaces.Discrete(25)
        assert env.observation_space["task_features"].shape == (8,)
        assert env.observation_space["action_metrics"].shape == (25, 2)
        assert gym.spaces.flatdim(env.observation_space) == 412
        check_env(env.unwrapped, skip_render_check=True)
    finally:
        env.close()


def test_registered_episode_wrappers_termination_and_source_rows(table_path):
    table = MeasuredTable.load(table_path)
    env = gym.make(ENV_ID, table_path=table_path, split="test")
    try:
        with pytest.raises(gym.error.ResetNeeded):
            env.step(0)
        assert isinstance(env, gym.wrappers.OrderEnforcing)
        assert isinstance(env.env, gym.wrappers.PassiveEnvChecker)
        assert len(env.unwrapped.tasks) == 18
        obs, info = env.reset(seed=9, options={"task_index": 0})
        assert env.observation_space.contains(obs)
        assert info["task_id"] == table.suite.tasks[table.suite.splits["test"][0]].task_id
        assert info["measured_replay"] and not info["fresh_gpu_measurement"]
        baseline_repeat = info["baseline"]["evidence"]["source_repeat_index"]
        event_ids = {info["baseline"]["trial_id"]}
        total_reward = 0
        for turn in range(8):
            obs, reward, terminated, truncated, info = env.step(np.int64(0))
            result = info["measurement"]
            assert env.observation_space.contains(obs)
            assert terminated is (turn == 7) and truncated is False
            assert result["evidence"]["source_repeat_index"] == (baseline_repeat + turn) % 3
            source = table.row(info["task_id"], "torch-compiled", (baseline_repeat + turn) % 3)
            assert result["latency_ms"] == source["latency_ms"]
            assert result["evidence"]["source_trial_id"] == source["trial_id"]
            assert info["trials_used"] == turn + 1
            event_ids.add(result["trial_id"])
            total_reward += reward
        assert len(event_ids) == 9
        assert total_reward == pytest.approx(1 - info["incumbent_cost"])
        with pytest.raises(RuntimeError, match="reset"):
            env.step(0)
        env.reset(seed=9)
        assert len(env.unwrapped.journal) == 10
    finally:
        env.close()
    with pytest.raises(RuntimeError, match="closed"):
        env.reset()


def test_reset_seed_controls_task_and_repeat_without_reusing_event_ids(table_path):
    env = gym.make(ENV_ID, table_path=table_path, split="validation")
    try:
        sequences = []
        for _ in range(2):
            initial, info = env.reset(seed=101)
            rows = [(initial, 0.0, info)]
            for action in (0, 3, 0):
                obs, reward, _, _, info = env.step(action)
                rows.append((obs, reward, info))
            sequences.append(rows)
        for left, right in zip(*sequences):
            assert data_equivalence(left[0], right[0], exact=True)
            assert left[1] == right[1]
            assert left[2]["task_id"] == right[2]["task_id"]
            role = "measurement" if left[2]["measurement"] else "baseline"
            assert left[2][role]["evidence"] == right[2][role]["evidence"]
            assert left[2][role]["trial_id"] != right[2][role]["trial_id"]
    finally:
        env.close()


def test_external_time_limit_and_flatten_observation(table_path):
    env = gym.wrappers.FlattenObservation(
        gym.make(ENV_ID, table_path=table_path, split="test", max_episode_steps=3)
    )
    try:
        obs, _ = env.reset(seed=2)
        assert obs.shape == (412,) and env.observation_space.contains(obs)
        for turn in range(3):
            obs, _, terminated, truncated, info = env.step(0)
            assert terminated is False and truncated is (turn == 2)
        assert info["trials_used"] == 3
        assert env.unwrapped.trial_budget == 8
    finally:
        env.close()


def test_failed_record_consumes_trial_and_cannot_replace_reference(table_path):
    env = gym.make(ENV_ID, table_path=table_path, trial_budget=1)
    try:
        env.reset(seed=0)
        action = next(i for i, candidate in enumerate(env.unwrapped.candidates)
                      if candidate.candidate_id == "triton-b128-w8")
        _, reward, terminated, truncated, info = env.step(action)
        assert terminated and not truncated and reward == 0
        assert info["incumbent_candidate_id"] is None
        assert info["measurement"]["status"] == "error"
    finally:
        env.close()


@pytest.mark.parametrize("kwargs,match", [
    ({"split": "unknown"}, "split"),
    ({"trial_budget": 0}, "trial_budget"),
    ({"memory_weight": -1}, "memory_weight"),
    ({"render_mode": "rgb_array"}, "render"),
])
def test_invalid_arguments_rejected(table_path, kwargs, match):
    with pytest.raises(ValueError, match=match):
        gym.make(ENV_ID, table_path=table_path, **kwargs)
