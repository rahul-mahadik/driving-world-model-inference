"""Equal-budget evaluation of user-supplied tuning policies.

This harness records observed search outcomes, not independently confirmed
speedups. It performs no policy training, candidate confirmation, or GPU setup.
The supplied factory owns evaluator construction and durable measurement logs.
Cohort IDs and input digests are evaluator declarations: the harness checks their
consistency, rather than independently identifying hardware or validating quality.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from numbers import Integral
import random
from statistics import median
from typing import Any, Callable, Mapping, Sequence

import numpy as np


class EvaluationError(ValueError):
    """An aborted evaluation, including every completed and partial episode."""

    def __init__(self, message: str, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


def _json(value: Any) -> Any:
    """Copy public records to strict JSON, retaining malformed numeric evidence."""
    if hasattr(value, "to_dict"):
        return _json(value.to_dict())
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"invalid_nonfinite_number": repr(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"unserializable_type": type(value).__name__, "representation": repr(value)}


def _canonical(value: Any) -> str:
    return json.dumps(_json(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _action_counts(observation: Mapping[str, Any], info: Mapping[str, Any]) -> list[int]:
    counts = np.asarray(observation["action_counts"])
    if counts.ndim != 1 or len(counts) == 0:
        raise ValueError("policy requires a nonempty action_counts vector")
    if info["trial_budget"] > len(counts):
        raise ValueError("without-replacement baseline requires budget <= candidate count")
    return counts.tolist()


def grid_policy(observation: Mapping[str, Any], info: Mapping[str, Any]) -> int:
    """Visit the same candidate prefix in every episode, without replacement."""
    counts = _action_counts(observation, info)
    return next(index for index, count in enumerate(counts) if count == 0)


class _RandomPolicy:
    def __init__(self, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise ValueError("policy seed must be an integer")
        self.seed = int(seed)
        self.rng = random.Random(self.seed)

    def __call__(self, observation: Mapping[str, Any], info: Mapping[str, Any]) -> int:
        counts = _action_counts(observation, info)
        if info["trials_used"] == 0:
            # The harness supplies an episode seed; direct callers may supply it
            # too. Resetting an episode never inherits another episode's order.
            self.rng = random.Random(f"{self.seed}:{info.get('evaluation_seed', 0)}")
        available = [index for index, count in enumerate(counts) if count == 0]
        if not available:
            raise ValueError("random policy exhausted its without-replacement action set")
        return self.rng.choice(available)


def random_policy(seed: int = 0) -> Callable[[Mapping[str, Any], Mapping[str, Any]], int]:
    """Return a reproducible random policy that never repeats an episode action."""
    return _RandomPolicy(seed)


def _protocol(env: Any) -> dict[str, Any]:
    return {
        "tasks": [_json(task) for task in env.tasks],
        "candidates": [_json(candidate) for candidate in env.candidates],
        "trial_budget": int(env.trial_budget),
        "memory_weight": float(env.memory_weight),
        "allow_synthetic": bool(env.allow_synthetic),
    }


def _aggregate(episodes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for episode in episodes:
        if episode["status"] == "completed":
            groups.setdefault(episode["task_id"], []).append(episode)
    result = []
    for task_id, rows in groups.items():
        result.append({
            "task_id": task_id,
            "episodes": len(rows),
            "median_observed_incumbent_cost": median(row["final_info"]["incumbent_cost"] for row in rows),
            "median_episode_reward_sum": median(row["reward_sum"] for row in rows),
            "median_baseline_latency_ms": median(row["initial_info"]["baseline"]["latency_ms"] for row in rows),
            "median_observed_incumbent_latency_ms": median(row["final_info"]["incumbent"]["latency_ms"] for row in rows),
            "median_observed_incumbent_peak_allocated_bytes": median(row["final_info"]["incumbent"]["peak_allocated_bytes"] for row in rows),
            "candidate_calls": sum(row["candidate_calls"] for row in rows),
            "baseline_calls": sum(row["baseline_calls"] for row in rows),
            "failed_or_incorrect_trials": sum(row["failed_or_incorrect_trials"] for row in rows),
            "selected_config_confirmation_performed": False,
        })
    return result


def evaluate_policy(
    env_factory: Callable[[], Any],
    policy: Callable[[Any, dict[str, Any]], int],
    task_indices: Sequence[int],
    repeats: int = 1,
    seed: int = 0,
    *,
    split: str = "evaluation",
    training_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run equal candidate budgets on an explicitly supplied task split.

    A fresh environment is created for each task/repeat. Every episode performs
    one baseline call plus exactly ``env.trial_budget`` candidate calls. Policies
    receive ``(observation, info)`` and return an action index; ``info`` additionally
    contains ``evaluation_seed`` and ``evaluation_episode_index``. PPO and other
    policies can adapt their prediction API to this callback without changing
    the benchmark. Policy training and split selection occur outside this API.

    Factory protocol changes, mixed hardware/runtime cohorts, repeated trial IDs,
    and changing inputs for the same task abort with ``EvaluationError.report``.
    The partial receipt preserves journals and errors, rather than dropping an
    episode. Failed candidate results returned normally by the environment still
    consume budget. No confirmation or training calls are hidden after the search.
    """
    repeats = _positive_integer(repeats, "repeats")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(split, str) or not split:
        raise ValueError("split must be a nonempty external split label")
    indices = list(task_indices)
    if not indices or any(isinstance(i, bool) or not isinstance(i, Integral) or i < 0 for i in indices):
        raise ValueError("task_indices must contain nonnegative integer indices")
    if len(set(indices)) != len(indices):
        raise ValueError("task_indices must be unique; use repeats for repeated episodes")
    if not callable(env_factory) or not callable(policy):
        raise ValueError("env_factory and policy must be callable")
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "split": split,
        "split_source": "caller supplied; no train/heldout separation is inferred",
        "evidence_scope": "Cohort/input bindings and correctness are evaluator assertions; inspect supporting raw evidence.",
        "task_indices": [int(i) for i in indices],
        "repeats": repeats,
        "seed": int(seed),
        "episodes": [],
        "per_task": [],
        "training": {
            "performed_by_harness": False,
            "harness_training_candidate_calls": 0,
            "external_training_metadata_provided": training_metadata is not None,
            "external_training_cost_verified": False,
            "external_training_cost_note": "External policy training is outside the harness; absent metadata means unknown, not zero.",
            "externally_supplied_metadata": _json(training_metadata),
        },
        "confirmation_baseline_calls": 0,
        "confirmation_candidate_calls": 0,
        "selected_config_confirmation_performed": False,
        "claims": {"policy_superiority": False, "heldout_generalization": False,
                   "independently_confirmed_selection": False},
    }
    expected_protocol = None
    expected_objective = None
    expected_cohort = None
    task_inputs: dict[str, str] = {}
    trial_ids: set[str] = set()
    environments: list[Any] = []

    def check_info(info: dict[str, Any], task_index: int) -> None:
        nonlocal expected_objective, expected_cohort
        if info["task_index"] != task_index:
            raise ValueError("factory reset returned a different task index")
        objective = _canonical(info["objective"])
        if expected_objective is not None and objective != expected_objective:
            raise ValueError("objective changed across episodes")
        expected_objective = objective
        cohort = _canonical({"backend": info["backend"], "cohort_id": info["cohort_id"]})
        if expected_cohort is not None and cohort != expected_cohort:
            raise ValueError("hardware/runtime cohort or backend changed across episodes; pooling denied")
        expected_cohort = cohort
        digest = _canonical(info["input_digest"])
        if info["task_id"] in task_inputs and task_inputs[info["task_id"]] != digest:
            raise ValueError("input digest changed for the same task")
        task_inputs[info["task_id"]] = digest

    def check_trial(measurement: dict[str, Any]) -> None:
        trial_id = measurement["trial_id"]
        if trial_id in trial_ids:
            raise ValueError("measurement trial ID reused across episodes")
        trial_ids.add(trial_id)

    def finish_episode(episode: dict[str, Any], env: Any) -> None:
        all_entries = getattr(env, "journal", []) if env is not None else []
        journal = _json(all_entries[episode.get("journal_start_index", 0):])
        episode["journal"] = journal
        # Journal entries are call receipts, including evaluator exceptions.
        episode["baseline_calls"] = sum(entry.get("kind") == "baseline" for entry in journal)
        episode["candidate_calls"] = sum(entry.get("kind") == "candidate" for entry in journal)
        episode["failed_or_incorrect_trials"] = sum(
            entry.get("kind") == "candidate" and (
                bool(entry.get("error") or entry.get("validation_error"))
                or isinstance(entry.get("measurement"), dict)
                and entry["measurement"].get("status") in {"incorrect", "error"}
            ) for entry in journal
        )

    try:
        for task_index in indices:
            for repeat in range(repeats):
                episode_index = len(report["episodes"])
                episode_seed = int(seed) + episode_index
                episode: dict[str, Any] = {
                    "episode_index": episode_index, "task_index": int(task_index),
                    "repeat": repeat, "seed": episode_seed, "status": "running", "steps": [],
                    "reward_sum": 0.0,
                }
                report["episodes"].append(episode)
                env = None
                try:
                    env = env_factory()
                    prior_journal = getattr(env, "journal", [])
                    episode["journal_start_index"] = len(prior_journal)
                    if prior_journal:
                        episode["preexisting_journal"] = _json(prior_journal)
                    if any(env is prior for prior in environments):
                        raise ValueError("env_factory must return a fresh environment for every episode")
                    environments.append(env)
                    if getattr(env, "journal", []):
                        raise ValueError("fresh environment already contains measured trials")
                    protocol = _protocol(env)
                    canonical = _canonical(protocol)
                    if expected_protocol is not None and canonical != expected_protocol:
                        raise ValueError("factory changed task, candidate, objective, or budget protocol")
                    expected_protocol = canonical
                    report["protocol"] = protocol
                    report["protocol_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
                    if any(index >= len(env.tasks) for index in indices):
                        raise ValueError("task index is outside the factory's task set")
                    observation, info = env.reset(seed=episode_seed, options={"task_index": int(task_index)})
                    episode["initial_observation"] = _json(observation)
                    episode["initial_info"] = _json(info)
                    episode["task_id"] = info["task_id"]
                    check_info(info, int(task_index))
                    check_trial(info["baseline"])
                    for trial_index in range(env.trial_budget):
                        policy_info = copy.deepcopy(info)
                        policy_info.update(evaluation_seed=episode_seed, evaluation_episode_index=episode_index)
                        step: dict[str, Any] = {"trial_index": trial_index, "observation": _json(observation)}
                        episode["steps"].append(step)
                        action = policy(copy.deepcopy(observation), policy_info)
                        step["action"] = _json(action)
                        observation, reward, terminated, truncated, info = env.step(action)
                        step.update(reward=float(reward), terminated=bool(terminated), truncated=bool(truncated),
                                    observation_after=_json(observation), info=_json(info))
                        episode["reward_sum"] += float(reward)
                        check_info(info, int(task_index))
                        check_trial(info["measurement"])
                        if info["trials_used"] != trial_index + 1:
                            raise ValueError("environment trial counter disagrees with completed candidate calls")
                        if (bool(terminated) and bool(truncated)) or bool(terminated or truncated) != (trial_index + 1 == env.trial_budget):
                            raise ValueError("environment ended outside the fixed candidate budget")
                    episode["final_info"] = _json(info)
                    finish_episode(episode, env)
                    if episode["baseline_calls"] != 1 or episode["candidate_calls"] != env.trial_budget:
                        raise ValueError("journal call counts differ from the fixed episode budget")
                    episode["status"] = "completed"
                except Exception as exc:
                    episode["status"] = "failed"
                    episode["error"] = {"type": type(exc).__name__, "message": str(exc)}
                    finish_episode(episode, env)
                    raise
                finally:
                    if env is not None:
                        try:
                            env.close()
                        except Exception as close_exc:
                            episode["close_error"] = {"type": type(close_exc).__name__, "message": str(close_exc)}
                            if episode["status"] != "failed":
                                episode["status"] = "failed"
                                episode["error"] = episode["close_error"]
                                raise
    except Exception as exc:
        report["status"] = "aborted"
        report["per_task"] = _aggregate(report["episodes"])
        report["counts"] = {name: sum(episode.get(name, 0) for episode in report["episodes"])
                            for name in ("baseline_calls", "candidate_calls", "failed_or_incorrect_trials")}
        raise EvaluationError(str(exc), _json(report)) from exc
    report["status"] = "completed"
    report["per_task"] = _aggregate(report["episodes"])
    report["counts"] = {name: sum(episode[name] for episode in report["episodes"])
                        for name in ("baseline_calls", "candidate_calls", "failed_or_incorrect_trials")}
    return _json(report)
