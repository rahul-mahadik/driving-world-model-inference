#!/usr/bin/env python3
"""Reproduce recorded PPO actions from a trusted frozen checkpoint, on CPU only.

Install the kernel-tuning package with its ``rl`` extra first. This verifies
policy inference on recorded observations. It constructs no tuning environment,
performs no training, and adds no GPU measurements. Run-directory plan hashes
are checked before the trusted SB3 checkpoint is deserialized.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from gymnasium import spaces


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def convert_observation(recorded, observation_space):
    """Restore JSON lists to the numeric shapes/dtypes used by SB3 inference."""
    require(type(recorded) is dict and set(recorded) == set(observation_space.spaces),
            "recorded observation keys differ from the checkpoint")
    restored = {}
    for key, space in observation_space.spaces.items():
        value = recorded[key]
        if isinstance(space, spaces.Discrete):
            require(type(value) is int and space.contains(value), f"invalid discrete observation {key}")
            restored[key] = value
            continue
        require(isinstance(space, (spaces.Box, spaces.MultiDiscrete)), f"unsupported observation space {key}")
        array = np.asarray(value)
        require(np.issubdtype(array.dtype, np.number) and np.isfinite(array).all(),
                f"observation {key} is not finite numeric data")
        if np.issubdtype(space.dtype, np.integer):
            require(np.equal(array, np.floor(array)).all(), f"nonintegral observation {key}")
        converted = array.astype(space.dtype, copy=False)
        require(converted.shape == space.shape and space.contains(converted),
                f"observation {key} is outside its declared space")
        restored[key] = converted
    return restored


def verify_actions(document, suite, policy):
    """Check every saved action without reconstructing or advancing an env."""
    expected_tasks = {suite.tasks[i].task_id for i in suite.splits["test"]}
    results = document.get("results", [])
    require(len(results) == 18 and {row.get("task_id") for row in results} == expected_tasks,
            "each completed run must contain all eighteen held-out tasks")
    observations = policy._model.observation_space
    checks = []
    for row in results:
        ppo = row.get("selectors", {}).get("ppo", {})
        turns = ppo.get("turns", [])
        require(ppo.get("policy_updates_permitted") is False and ppo.get("candidate_turns") == 8
                and len(turns) == 8, "fixed-policy eight-turn episode required")
        for index, turn in enumerate(turns):
            require(type(turn.get("turn")) is int and turn["turn"] == index, "PPO turn order differs")
            recorded_action = turn.get("action")
            require(type(recorded_action) is int and 0 <= recorded_action < len(suite.candidates),
                    "recorded action is outside the candidate grid")
            observation = convert_observation(turn.get("preaction_observation"), observations)
            predicted_action = int(policy(observation, {}))
            require(predicted_action == recorded_action,
                    f"checkpoint action mismatch: task={row['task_id']} turn={index} "
                    f"recorded={recorded_action} predicted={predicted_action}")
            checks.append({"task_id": row["task_id"], "turn": index, "action": predicted_action})
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True,
                        help="Three completed run directories, each containing plan.json and result.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true",
                        help="Explicitly audit only one/two completed processes; receipt remains partial")
    args = parser.parse_args(argv)
    require(not args.output.exists(), "refuse to replace an existing action-verification receipt")
    require(len(args.runs) in ((1, 2, 3) if args.allow_partial else (3,)),
            "three run directories required unless --allow-partial is explicit")
    checkpoint_hash, metadata_hash = sha(args.checkpoint), sha(args.metadata)
    manifest = json.loads(args.metadata.read_text())
    from drivebench_tuning.suite import load_suite
    suite = load_suite("operator-generalization-v2")
    runs = []
    for directory in args.runs:
        plan_path, result_path = directory / "plan.json", directory / "result.json"
        plan, result = json.loads(plan_path.read_text()), json.loads(result_path.read_text())
        require(plan.get("checkpoint_sha256") == checkpoint_hash
                and plan.get("ppo_manifest_sha256") == metadata_hash,
                "run plan checkpoint or manifest hash differs from the supplied frozen files")
        require(plan.get("suite_sha256") == result.get("suite_sha256") == suite.sha256,
                "run suite differs from the checkpoint suite")
        require(type(result.get("repeat_index")) is int
                and plan.get("repeat_index") == result["repeat_index"] in (0, 1, 2),
                "plan/result repeat index differs")
        require(plan.get("protocol") == result.get("protocol")
                and plan["protocol"].get("ppo_trial_budget") == 8,
                "plan/result PPO protocol differs")
        runs.append((plan, result, {"plan_sha256": sha(plan_path), "result_sha256": sha(result_path)}))
    indices = [result["repeat_index"] for _, result, _ in runs]
    require(len(set(indices)) == len(indices), "duplicate process repeat index")
    require(args.allow_partial or set(indices) == {0, 1, 2}, "complete three-process cohort required")
    require(len({result.get("cohort_id") for _, result, _ in runs}) == 1
            and all(type(result.get("cohort_id")) is str and result["cohort_id"] for _, result, _ in runs),
            "one bound physical/runtime cohort required")
    require(manifest.get("seed") == 3101 and manifest.get("training", {}).get("performed") is True,
            "prospectively selected trained seed 3101 required")

    # All run plans bind the files before deserializing this trusted checkpoint.
    import torch
    torch.set_num_threads(2)
    from drivebench_tuning.ppo import PPOSelector, parameter_hash
    policy = PPOSelector.load(args.checkpoint, args.metadata, suite)
    require(all(p.device.type == "cpu" for p in policy._model.policy.parameters()), "CPU policy required")
    before = parameter_hash(policy._model)
    run_checks = []
    for _, result, hashes in sorted(runs, key=lambda row: row[1]["repeat_index"]):
        checks = verify_actions(result, suite, policy)
        run_checks.append({"repeat_index": result["repeat_index"], **hashes,
                           "tasks_checked": 18, "actions_checked": len(checks), "actions": checks})
    after = parameter_hash(policy._model)
    require(before == after == manifest["final_parameter_sha256"], "policy parameters changed during verification")
    complete = set(indices) == {0, 1, 2}
    receipt = {"kind": "frozen_ppo_live_action_verification", "at": datetime.now(timezone.utc).isoformat(),
               "status": "complete" if complete else "partial_process_coverage",
               "all_three_processes_verified": complete, "repeat_indices": sorted(indices),
               "checkpoint_sha256": checkpoint_hash, "manifest_sha256": metadata_hash,
               "verifier_sha256": sha(__file__), "suite_sha256": suite.sha256,
               "policy_parameter_sha256_before": before, "policy_parameter_sha256_after": after,
               "training_seed": 3101, "device": "cpu", "deterministic": True,
               "actions_checked": sum(row["actions_checked"] for row in run_checks), "runs": run_checks,
               "training_performed": False, "tuning_environment_constructed": False,
               "fresh_gpu_measurements": 0, "saved_observation_reward_reconstruction_performed": False,
               "scope": "Frozen checkpoint inference on recorded preaction observations; separate native_audit reconstructs state/reward."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: value for key, value in receipt.items() if key != "runs"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
