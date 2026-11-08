"""A small reference RL learner and honest equal-budget policy comparisons.

The table learns only for workload keys seen during training.  It is a reference
agent for exercising the environment, not a claim of generalization to unseen
models, shapes or GPUs.  Training measurements are an additional, explicitly
reported cost.  Evaluation always calls a fresh environment/evaluator; neither
training measurements nor previous evaluation observations are reused.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Sequence

import numpy as np

from .env import KernelTuningEnv
from .contract import KernelConfig


def _state(info: dict[str, Any]) -> tuple[str, int]:
    # Repeated measurements are independent trials under a stationary cohort.
    # This intentionally simple reference policy uses only workload and budget
    # remaining.  The full environment exposes richer history to other agents.
    return str(info["workload_key"]), int(info["trial_budget"] - info["trials_used"])


@dataclass
class TabularQPolicy:
    action_count: int
    q_values: dict[tuple[str, int], np.ndarray] = field(default_factory=dict)
    configs: tuple[KernelConfig, ...] | None = None
    training_cohort_id: str | None = None
    training_backend: str | None = None

    def __post_init__(self) -> None:
        _positive_int(self.action_count, "action_count")
        if self.configs is not None and len(self.configs) != self.action_count:
            raise ValueError("policy configs must match action_count")

    def values(self, info: dict[str, Any]) -> np.ndarray:
        return self.q_values.setdefault(_state(info), np.zeros(self.action_count, dtype=np.float64))

    def choose(self, info: dict[str, Any], rng: np.random.Generator, epsilon: float = 0.0) -> int:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1")
        values = self.values(info)
        if rng.random() < epsilon:
            return int(rng.integers(self.action_count))
        # Random tie-breaking does not privilege config order, including unseen
        # workloads whose action values are all initially zero.
        return int(rng.choice(np.flatnonzero(values == values.max())))

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": "tabular_q_learning",
            "state": "workload_key_and_trials_remaining",
            "action_count": self.action_count,
            "configs": ([{"block_size": c.block_size, "num_warps": c.num_warps} for c in self.configs]
                        if self.configs is not None else None),
            "training_cohort_id": self.training_cohort_id,
            "training_backend": self.training_backend,
            "q_values": [
                {"workload_key": key[0], "trials_remaining": key[1], "values": values.tolist()}
                for key, values in sorted(self.q_values.items())
            ],
            "generalization_claim": False,
        }


@dataclass
class TrainingResult:
    policy: TabularQPolicy
    episodes: list[dict[str, Any]]
    candidate_measurements: int
    baseline_measurements: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.to_dict(), "episodes": self.episodes,
            "candidate_measurements": self.candidate_measurements,
            "baseline_measurements": self.baseline_measurements,
            "training_cost_excluded_from_evaluation_budget": True,
        }


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def train_q_learning(
    env: KernelTuningEnv,
    episodes: int = 12,
    *,
    epsilon: float = 0.25,
    learning_rate: float = 0.2,
    discount: float = 0.9,
    seed: int = 0,
    workload_indices: Sequence[int] | None = None,
) -> TrainingResult:
    """Train with fresh measurements, reporting all training trial costs."""
    _positive_int(episodes, "episodes")
    if not math.isfinite(epsilon) or not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be finite and between 0 and 1")
    if not math.isfinite(learning_rate) or not 0 < learning_rate <= 1:
        raise ValueError("learning_rate must be finite and in (0, 1]")
    if not math.isfinite(discount) or not 0 <= discount <= 1:
        raise ValueError("discount must be finite and between 0 and 1")
    indices = tuple(workload_indices) if workload_indices is not None else tuple(range(len(env.workloads)))
    if not indices:
        raise ValueError("workload_indices must be nonempty")
    policy = TabularQPolicy(len(env.configs), configs=tuple(env.configs))
    rng = np.random.default_rng(seed)
    records = []
    candidates = 0
    for episode in range(int(episodes)):
        index = indices[episode % len(indices)]
        _, info = env.reset(seed=seed + episode, options={"workload_index": index})
        current_cohort = info["baseline"]["evidence"].get("cohort_id")
        current_backend = info["baseline"]["backend"]
        if episode == 0:
            policy.training_cohort_id, policy.training_backend = current_cohort, current_backend
        elif (policy.training_cohort_id, policy.training_backend) != (current_cohort, current_backend):
            raise ValueError("training cannot pool different hardware/runtime cohorts or backends")
        actions: list[int] = []
        rewards: list[float] = []
        measurements: list[dict[str, Any]] = []
        while True:
            action = policy.choose(info, rng, epsilon)
            current_values = policy.values(info)
            _, reward, terminated, truncated, next_info = env.step(action)
            # A budget truncation is terminal for this finite-horizon task.
            bootstrap = 0.0 if terminated or truncated else float(policy.values(next_info).max())
            current_values[action] += learning_rate * (reward + discount * bootstrap - current_values[action])
            actions.append(action)
            rewards.append(reward)
            measurements.append(next_info["measurement"])
            candidates += 1
            info = next_info
            if terminated or truncated:
                break
        records.append({
            "episode": episode, "workload_key": info["workload_key"],
            "workload_index": info["workload_index"], "synthetic": info["synthetic"],
            "actions": actions, "rewards": rewards, "return": sum(rewards),
            "baseline": info["baseline"], "measurements": measurements,
            "incumbent": info["incumbent"], "incumbent_cost": info["incumbent_cost"],
        })
    return TrainingResult(policy, records, candidates, len(records))


def evaluate_policies(
    env_factory: Callable[[], KernelTuningEnv],
    policy: TabularQPolicy,
    *,
    workload_indices: Sequence[int] | None = None,
    episodes_per_workload: int = 1,
    seed: int = 1000,
) -> dict[str, Any]:
    """Compare learned/random/grid selection with equal fresh-trial budgets.

    Each policy gets one fresh baseline and the same candidate-call budget per
    episode.  Order rotates between episodes to limit systematic order bias.
    This is descriptive evaluation, not a statistical significance test.  A
    learned policy has separate training cost and must not be called cheaper
    than the baselines on the strength of the evaluation budget alone.
    """
    _positive_int(episodes_per_workload, "episodes_per_workload")
    names = ("learned", "random", "grid")
    records: list[dict[str, Any]] = []
    signature: tuple[Any, ...] | None = None
    cohort_id: str | None = None
    backend: str | None = None
    cohort_initialized = False
    indices: tuple[int, ...] | None = tuple(workload_indices) if workload_indices is not None else None
    if indices == ():
        raise ValueError("workload_indices must be nonempty")
    # Open one probe environment to discover the workload list.  No reset or
    # benchmark is performed by this discovery; it is also used for run one.
    first_env = env_factory()
    if indices is None:
        indices = tuple(range(len(first_env.workloads)))
    run_number = 0
    try:
        for workload_index in indices:
            for repeat in range(int(episodes_per_workload)):
                offset = (repeat + indices.index(workload_index)) % len(names)
                order = names[offset:] + names[:offset]
                for name in order:
                    env = first_env if run_number == 0 else env_factory()
                    run_number += 1
                    try:
                        current_signature = (
                            tuple(w.key() for w in env.workloads), tuple(env.configs),
                            env.trial_budget, env.memory_weight, env.allow_synthetic,
                        )
                        if signature is None:
                            signature = current_signature
                        elif signature != current_signature:
                            raise ValueError("policy comparisons must use identical workloads, actions, objective and budgets")
                        if len(env.configs) != policy.action_count:
                            raise ValueError("policy action count does not match the environment")
                        if policy.configs is not None and policy.configs != tuple(env.configs):
                            raise ValueError("policy action configurations do not match the environment")
                        episode_seed = seed + int(workload_index) * int(episodes_per_workload) + repeat
                        rng = np.random.default_rng(episode_seed)
                        _, info = env.reset(seed=episode_seed, options={"workload_index": workload_index})
                        current_cohort = info["baseline"]["evidence"].get("cohort_id")
                        if policy.training_backend is not None and (
                            policy.training_backend != info["baseline"]["backend"]
                            or policy.training_cohort_id != current_cohort
                        ):
                            raise ValueError("trained policy and evaluation must use the same hardware/runtime cohort and backend")
                        if not cohort_initialized:
                            cohort_id, cohort_initialized = current_cohort, True
                            backend = info["baseline"]["backend"]
                        elif cohort_id != current_cohort or backend != info["baseline"]["backend"]:
                            raise ValueError("policy comparisons cannot pool different hardware/runtime cohorts or backends")
                        actions, rewards, measurements = [], [], []
                        while True:
                            if name == "learned":
                                action = policy.choose(info, rng)
                            elif name == "random":
                                action = int(rng.integers(len(env.configs)))
                            else:
                                action = len(actions) % len(env.configs)
                            _, reward, terminated, truncated, info = env.step(action)
                            actions.append(action)
                            rewards.append(reward)
                            measurements.append(info["measurement"])
                            if terminated or truncated:
                                break
                        records.append({
                            "policy": name, "workload_key": info["workload_key"],
                            "workload_index": info["workload_index"], "repeat": repeat,
                            "synthetic": info["synthetic"], "candidate_measurements": len(actions),
                            "baseline_measurements": 1, "actions": actions, "rewards": rewards,
                            "return": sum(rewards), "baseline": info["baseline"],
                            "measurements": measurements, "incumbent": info["incumbent"],
                            "incumbent_cost": info["incumbent_cost"],
                        })
                    finally:
                        env.close()
    finally:
        if run_number == 0:
            first_env.close()
    return {
        "records": records, "equal_candidate_budget": True,
        "fresh_measurements": True, "training_cost_included": False,
        "cohort_id": cohort_id, "backend": backend, "synthetic": backend == "synthetic",
        "selected_config_confirmation_performed": False,
        "rl_improvement_validated": False,
        "note": "Incumbent metrics are descriptive search outcomes, not independently confirmed improvements. Training cost must be reported separately. No generalization or speedup claim is implied.",
    }
