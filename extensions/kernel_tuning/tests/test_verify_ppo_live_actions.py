"""CPU contracts for the public frozen-policy inference verification script."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from drivebench_tuning.search import BudgetedTuningEnv
from drivebench_tuning.suite import load_suite
from test_native_audit import make_recorded_task


@pytest.fixture(scope="module")
def verifier():
    path = Path(__file__).resolve().parents[3] / "scripts" / "verify_ppo_live_actions.py"
    spec = importlib.util.spec_from_file_location("verify_ppo_live_actions_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def controlled():
    suite = load_suite("operator-generalization-v2")
    rows = [make_recorded_task(suite, i, prefix=f"test-{i}")[0] for i in suite.splits["test"]]
    # This explicit controlled policy exercises the verifier contract, not RL
    # performance. The real checkpoint is checked separately on recorded runs.
    env = BudgetedTuningEnv(suite.tasks, suite.candidates, None, trial_budget=8)

    class Policy:
        _model = SimpleNamespace(observation_space=env.observation_space)

        def __call__(self, observation, info):
            assert info == {}
            return (0, 1, 1, 2, 3, 0, 4, 4)[int(sum(observation["action_counts"]))]

    return suite, {"results": rows}, Policy()


def test_checks_every_action_using_recorded_numeric_observations(verifier, controlled):
    suite, document, policy = controlled
    checks = verifier.verify_actions(document, suite, policy)
    assert len(checks) == 144
    assert {row["task_id"] for row in checks} == {suite.tasks[i].task_id for i in suite.splits["test"]}


def test_rejects_tampered_recorded_action(verifier, controlled):
    suite, document, policy = controlled
    document["results"][0]["selectors"]["ppo"]["turns"][3]["action"] = 5
    with pytest.raises(ValueError, match="checkpoint action mismatch"):
        verifier.verify_actions(document, suite, policy)


def test_does_not_silently_truncate_nonintegral_observation(verifier, controlled):
    suite, document, policy = controlled
    observation = deepcopy(document["results"][0]["selectors"]["ppo"]["turns"][0]["preaction_observation"])
    observation["action_counts"][0] = .5
    with pytest.raises(ValueError, match="nonintegral observation"):
        verifier.convert_observation(observation, policy._model.observation_space)


def test_refuses_incomplete_run_by_default(verifier, tmp_path):
    with pytest.raises(ValueError, match="three run directories required"):
        verifier.main(["--checkpoint", str(tmp_path / "unused.zip"), "--metadata", str(tmp_path / "unused.json"),
                       "--runs", str(tmp_path), "--output", str(tmp_path / "receipt.json")])


def test_plan_hash_checked_before_checkpoint_deserialization(verifier, tmp_path):
    checkpoint, metadata = tmp_path / "model.zip", tmp_path / "metadata.json"
    checkpoint.write_bytes(b"not a model; deserializing this would fail")
    metadata.write_text("{}")
    (tmp_path / "plan.json").write_text(json.dumps({"checkpoint_sha256": "0" * 64}))
    (tmp_path / "result.json").write_text("{}")
    with pytest.raises(ValueError, match="run plan checkpoint or manifest hash differs"):
        verifier.main(["--checkpoint", str(checkpoint), "--metadata", str(metadata),
                       "--runs", str(tmp_path), "--allow-partial", "--output", str(tmp_path / "receipt.json")])
    assert not (tmp_path / "receipt.json").exists()
