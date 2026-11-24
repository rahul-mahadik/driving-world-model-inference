"""PPO contracts on invented CPU fixtures; these are not research results."""
from copy import deepcopy
import json

import numpy as np
import pytest

from drivebench_tuning.ppo import (PPOSelector, PPOTrainingEnv, SETTINGS, TrainingSnapshot,
                                  train_seed)
from drivebench_tuning.ppo_cli import make_protocol, summarize_reports, validate_protocol
from drivebench_tuning.replay_policy import fixed_policy, run_replay_policy
from drivebench_tuning.replay import MeasuredReplayEvaluator
from drivebench_tuning.search import BudgetedTuningEnv
from test_measured_replay import table


def test_snapshot_reads_only_train_and_owns_no_full_table(table, monkeypatch):
    train_ids = {table.suite.tasks[i].task_id for i in table.suite.splits["train"]}
    read = table.row
    def guarded(task, candidate, repeat):
        assert task in train_ids, "held-out leakage"
        return read(task, candidate, repeat)
    monkeypatch.setattr(table, "row", guarded)
    snapshot = TrainingSnapshot(table)
    assert set(snapshot.provenance["task_ids"]) == train_ids
    assert snapshot.provenance["row_count"] == len(train_ids) * (len(table.suite.candidates)+1) * 3
    assert not any(value is table or value is table.suite for value in vars(snapshot).values())
    with pytest.raises(ValueError, match="held-out"):
        snapshot.row(table.suite.tasks[table.suite.splits["test"][0]].task_id, None, 0)
    for (task, _, _), row in table._rows.items():
        if task not in train_ids and row["status"] == "ok": row["latency_ms"] *= 1000
    assert TrainingSnapshot(table).provenance == snapshot.provenance


def test_training_environment_matches_public_budget_and_replay_contract(table):
    snapshot = TrainingSnapshot(table)
    env = PPOTrainingEnv(snapshot)
    obs, info = env.reset(seed=12)
    assert info == {}
    assert "task_id" not in obs
    task = env.inner.current_task
    task_index = next(i for i, t in enumerate(table.suite.tasks) if t.task_id == task.task_id)
    reference = BudgetedTuningEnv(table.suite.tasks, table.suite.candidates,
                                 MeasuredReplayEvaluator(table, seed=env.current["replay_seed"]),
                                 trial_budget=8, allow_replay=True)
    expected, _ = reference.reset(options={"task_index": task_index})
    for k in obs: np.testing.assert_equal(obs[k], expected[k])
    for action in [0, 0, 2, 3, 4, 5, 6, 7]:
        actual = env.step(action); expected = reference.step(action)
        assert actual[1:4] == expected[1:4]
        assert actual[4] == {}
        for k in actual[0]: np.testing.assert_equal(actual[0][k], expected[0][k])
    assert env.current["transitions"][0]["action"] == env.current["transitions"][1]["action"]
    assert env.steps_total == 8
    assert len(env.inner.journal) == 9
    env.reset()
    assert len(env.inner.journal) == 1  # No unbounded training journal in RAM.
    env.close(); reference.close()


def test_small_real_ppo_update_checkpoint_binding_and_no_info_access(table, tmp_path):
    pytest.importorskip("stable_baselines3")
    settings = {**SETTINGS, "total_timesteps": 32, "n_envs": 1, "torch_threads": 1,
                "n_steps": 8, "batch_size": 8, "n_epochs": 1, "net_arch": [8, 8]}
    snapshot = TrainingSnapshot(table)
    manifests = [train_seed(snapshot, tmp_path/f"run-{i}", 3101,
                            initial_checkpoint=True, settings=settings) for i in range(2)]
    assert manifests[0]["final_parameter_sha256"] == manifests[1]["final_parameter_sha256"]
    assert all(m["parameters_changed"] and m["parameter_delta_l2"] > 0 for m in manifests)
    assert manifests[0]["training"]["rl_training"] is True
    assert manifests[0]["actual_environment_steps"] == 32
    assert manifests[0]["completed_training_episodes"] == 4
    path = tmp_path/"run-0"
    selector = PPOSelector.load(path/"model.zip", path/"manifest.json", table.suite)
    assert selector._model.env is None
    env = BudgetedTuningEnv(table.suite.tasks, table.suite.candidates, MeasuredReplayEvaluator(table),
                           trial_budget=8, allow_replay=True)
    obs, info = env.reset(options={"task_index": table.suite.splits["test"][0]})
    class NoAccess:
        def __getitem__(self, key): raise AssertionError("info is not a network input")
    assert selector(obs, NoAccess()) == selector(obs, info)
    with (path/"model.zip").open("ab") as f: f.write(b"tampered")
    with pytest.raises(ValueError, match="binding differs"):
        PPOSelector.load(path/"model.zip", path/"manifest.json", table.suite)
    env.close()


def test_protocol_rejects_changed_plan_and_test_training(table):
    protocol = make_protocol(table)
    validate_protocol(protocol, table)
    assert protocol["evaluation"]["seed"] == 2101
    assert protocol["live_policy"]["training_seed"] == 3101
    bad = deepcopy(protocol); bad["training_seeds"] = [3105]
    with pytest.raises(ValueError, match="fixed source"):
        validate_protocol(bad, table)
    bad = deepcopy(protocol); bad["train_tasks"] = bad["test_tasks"]
    with pytest.raises(ValueError, match="fixed source"):
        validate_protocol(bad, table)


def test_summary_preserves_trained_policy_provenance_without_claiming_new_training(table):
    report = run_replay_policy(table, fixed_policy(), name="metadata-fixture", repeats=1, trials=8)
    # This test marks only an invented metadata fixture; it creates no research
    # evidence and does not imply that the fixed policy was actually trained.
    report["training_performed"] = report["rl_training_performed"] = True
    summary = summarize_reports([report])
    assert summary["policy_training_performed"] is True
    assert summary["rl_training_performed"] is True
    assert summary["training_performed_during_evaluation"] is False
    assert summary["fresh_gpu_measurements"] == 0
