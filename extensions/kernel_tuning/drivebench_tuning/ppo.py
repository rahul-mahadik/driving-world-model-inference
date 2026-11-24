"""PPO on measured TRAIN-only search episodes; CPU inference on public observations.

PPO training reuses finite operator timings. It does not execute a GPU or add
independent timing evidence. The learner never receives task IDs, evaluator
objects, unqueried costs, validation outcomes, or test outcomes.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import uuid

import gymnasium as gym
import numpy as np

from .search import BudgetedTuningEnv, TrialResult

SB3_VERSION = "2.9.2"
TRAINING_SEEDS = (3101, 3102, 3103, 3104, 3105)
SETTINGS = {
    "total_timesteps": 102400, "n_envs": 4, "torch_threads": 2,
    "learning_rate": 0.0003, "n_steps": 256, "batch_size": 256,
    "n_epochs": 10, "gamma": 1.0, "gae_lambda": 0.95, "clip_range": 0.2,
    "ent_coef": 0.01, "vf_coef": 0.5, "max_grad_norm": 0.5,
    "normalize_advantage": True, "net_arch": [64, 64],
    "trial_budget": 8, "memory_weight": 0.0,
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sb3():
    import stable_baselines3 as sb3
    if sb3.__version__ != SB3_VERSION:
        raise ValueError(f"stable-baselines3=={SB3_VERSION} is required")
    return sb3


class TrainingSnapshot:
    """A copied capability containing only TRAIN descriptors and measured rows.

    No reference to the source table or full suite is retained. Its row API
    rejects requests for held-out tasks even if called directly.
    """
    def __init__(self, table):
        suite = table.suite
        self.tasks = tuple(suite.tasks[i] for i in suite.splits["train"])
        if not self.tasks or any(t.metadata.get("split") != "train" for t in self.tasks):
            raise ValueError("nonempty explicit TRAIN tasks required")
        available = {t.task_id for t in table.measured_tasks}
        if not {t.task_id for t in self.tasks}.issubset(available):
            raise ValueError("table must contain the complete TRAIN split")
        self.candidates = tuple(suite.candidates)
        self.repeat_indices = tuple(table.repeat_indices)
        self.table_sha256, self.cohort_id = table.sha256, table.cohort_id
        self.suite_sha256 = suite.sha256
        self.rows = {}
        provenance = []
        fields = ("trial_id", "task_id", "candidate_id", "input_digest", "status", "correct",
                  "latency_ms", "peak_allocated_bytes")
        for task in self.tasks:
            for candidate_id in (None, *(c.candidate_id for c in self.candidates)):
                for repeat in self.repeat_indices:
                    row = table.row(task.task_id, candidate_id, repeat)
                    if candidate_id is None and (row["status"] != "ok" or not row["correct"]):
                        raise ValueError("TRAIN references must pass")
                    key = (task.task_id, candidate_id, repeat)
                    self.rows[key] = {k: deepcopy(row[k]) for k in fields}
                    provenance.append({"task_id": task.task_id, "candidate_id": candidate_id,
                                       "repeat_index": repeat, "source_trial_id": row["trial_id"],
                                       "source_measurement_sha256": digest(row)})
        self.provenance = {
            "training_split": "train", "outcome_splits_read": ["train"],
            "task_ids": [t.task_id for t in self.tasks], "row_count": len(self.rows),
            "source_table_sha256": self.table_sha256, "source_cohort_id": self.cohort_id,
            "suite_sha256": self.suite_sha256, "validation_or_test_outcomes_used": False,
            "copied_records_sha256": digest(list(self.rows.values())),
            "source_records_sha256": digest(provenance), "source_records": provenance,
        }

    def row(self, task_id, candidate_id, repeat):
        if task_id not in {t.task_id for t in self.tasks}:
            raise ValueError("TRAIN-only evaluator rejects held-out task")
        return deepcopy(self.rows[(task_id, candidate_id, repeat)])


class _TrainingReplayEvaluator:
    """Same per-candidate recorded-repeat cycle as MeasuredReplayEvaluator."""
    def __init__(self, snapshot, seed):
        self.snapshot, self.seed = snapshot, seed
        self.counts, self.event = defaultdict(int), 0
        self.namespace, self.closed = uuid.uuid4().hex, False
        self.tasks = {t.task_id: t.to_dict() for t in snapshot.tasks}
        self.candidates = {c.candidate_id: c.to_dict() for c in snapshot.candidates}

    def _run(self, task, candidate):
        if self.closed or self.tasks.get(task.task_id) != task.to_dict():
            raise ValueError("closed evaluator or task outside TRAIN snapshot")
        cid = None if candidate is None else candidate.candidate_id
        if candidate is not None and self.candidates.get(cid) != candidate.to_dict():
            raise ValueError("candidate differs from frozen suite")
        key = (task.task_id, cid)
        repeat = self.snapshot.repeat_indices[(self.seed + self.counts[key]) % len(self.snapshot.repeat_indices)]
        self.counts[key] += 1
        row = self.snapshot.row(task.task_id, cid, repeat)
        self.event += 1
        return TrialResult(trial_id=f"replay:{self.namespace}:{self.event}", task_id=task.task_id,
                           candidate_id=cid, backend="measured_replay", cohort_id=self.snapshot.cohort_id,
                           input_digest=row["input_digest"], status=row["status"], correct=row["correct"],
                           latency_ms=row["latency_ms"], peak_allocated_bytes=row["peak_allocated_bytes"],
                           evidence={"table_sha256": self.snapshot.table_sha256,
                                     "source_trial_id": row["trial_id"], "source_backend": "cuda",
                                     "source_repeat_index": repeat, "fresh_gpu_measurement": False,
                                     "recorded_outcome_reused": True,
                                     "sampling": "episode_start_repeat_then_per_candidate_cycle"})

    def baseline(self, task):
        return self._run(task, None)

    def evaluate(self, task, candidate):
        return self._run(task, candidate)

    def close(self):
        self.closed = True


class PPOTrainingEnv(gym.Env):
    """Bounded-journal wrapper around the existing eight-trial search contract."""
    metadata = {"render_modes": []}

    def __init__(self, snapshot, *, journal=None, worker=0):
        super().__init__()
        self.snapshot, self.journal, self.worker = snapshot, journal, worker
        self.episode, self.steps_total = 0, 0
        self.task_counts = defaultdict(int)
        self.inner = self._new_inner(0)
        self.action_space, self.observation_space = self.inner.action_space, self.inner.observation_space
        self.current = None

    def _new_inner(self, seed):
        return BudgetedTuningEnv(self.snapshot.tasks, self.snapshot.candidates,
                                 _TrainingReplayEvaluator(self.snapshot, seed), trial_budget=8,
                                 memory_weight=0.0, allow_replay=True)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError("training reset does not accept task override")
        self.inner.close()
        replay_seed = int(self.np_random.integers(2**31))
        self.inner = self._new_inner(replay_seed)
        task_index = int(self.np_random.integers(len(self.snapshot.tasks)))
        obs, info = self.inner.reset(options={"task_index": task_index})
        self.current = {"worker": self.worker, "episode": self.episode, "task_id": info["task_id"],
                        "replay_seed": replay_seed, "baseline_source_trial_id":
                        info["baseline"]["evidence"]["source_trial_id"], "transitions": []}
        self.task_counts[info["task_id"]] += 1
        self.episode += 1
        return obs, {}  # No task IDs or raw receipts are supplied to the learner.

    def step(self, action):
        obs, reward, terminated, truncated, info = self.inner.step(int(action))
        self.steps_total += 1
        self.current["transitions"].append({"action": int(action), "reward": reward,
                                            "source_trial_id": info["measurement"]["evidence"]["source_trial_id"],
                                            "source_repeat_index": info["measurement"]["evidence"]["source_repeat_index"],
                                            "incumbent_cost": info["incumbent_cost"]})
        if terminated:
            self.current["final_candidate_id"] = info["incumbent_candidate_id"]
            self.current["reward_sum"] = sum(x["reward"] for x in self.current["transitions"])
            if self.journal:
                self.journal.write(canonical(self.current).decode() + "\n")
        return obs, reward, terminated, truncated, {}

    def close(self):
        self.inner.close()


def parameter_hash(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.policy.state_dict().items()):
        a = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(canonical(list(a.shape))); h.update(a.tobytes())
    return h.hexdigest()


def _manifest(model, checkpoint, snapshot, seed, *, trained, initial_hash, delta_l2, steps):
    return {
        "schema_version": 1, "kind": "drivebench_sb3_ppo_checkpoint", "library": "stable-baselines3",
        "library_version": SB3_VERSION, "seed": seed, "checkpoint_sha256": file_digest(checkpoint),
        "suite_sha256": snapshot.suite_sha256,
        "candidates": [c.to_dict() for c in snapshot.candidates],
        "task_feature_count": len(snapshot.tasks[0].features), "settings": deepcopy(SETTINGS),
        "observation_keys": sorted(model.observation_space.spaces),
        "trained": trained, "training_timesteps": steps,
        "initial_parameter_sha256": initial_hash, "final_parameter_sha256": parameter_hash(model),
        "parameter_delta_l2": delta_l2, "parameters_changed": initial_hash != parameter_hash(model),
        "training": {**deepcopy(snapshot.provenance), "performed": trained, "rl_training": trained,
                     "method": "PPO clipped policy-gradient updates" if trained else "untrained initialization",
                     "hyperparameter_selection": "fixed prospectively; no validation or test selection",
                     "gpu_measurements_performed_by_training": 0},
        "inference": {"deterministic": True, "device": "cpu", "fine_tuning": False,
                      "training_seed_selected_by_results": False},
    }


def train_seed(snapshot, output, seed, *, initial_checkpoint=False, settings=None):
    """Train one fixed seed; caller must validate/freeze its protocol first.

    ``settings`` is only a bounded CPU contract-test override. The research CLI
    always passes the frozen SETTINGS verbatim and disallows overrides.
    """
    sb3 = _sb3()
    import torch
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.logger import configure
    settings = deepcopy(SETTINGS if settings is None else settings)
    if type(seed) is not int or seed < 0:
        raise ValueError("nonnegative integer training seed required")
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(settings["torch_threads"])
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    start_stamp = datetime.now(timezone.utc).isoformat()
    journal = (output / "training-episodes.jsonl").open("x")
    env = DummyVecEnv([lambda i=i: PPOTrainingEnv(snapshot, journal=journal, worker=i)
                      for i in range(settings["n_envs"])])
    try:
        kwargs = {k: settings[k] for k in ("learning_rate", "n_steps", "batch_size", "n_epochs", "gamma",
                  "gae_lambda", "clip_range", "ent_coef", "vf_coef", "max_grad_norm", "normalize_advantage")}
        model = sb3.PPO("MultiInputPolicy", env, seed=seed, device="cpu", verbose=0,
                        policy_kwargs={"net_arch": {"pi": settings["net_arch"], "vf": settings["net_arch"]}}, **kwargs)
        model.set_logger(configure(str(output), ["csv", "json"]))
        initial_hash = parameter_hash(model)
        initial = {k: v.detach().cpu().clone() for k, v in model.policy.state_dict().items()}
        if initial_checkpoint:
            checkpoint = output / "untrained.zip"; model.save(checkpoint)
            meta = _manifest(model, checkpoint, snapshot, seed, trained=False, initial_hash=initial_hash,
                             delta_l2=0.0, steps=0)
            (output / "untrained-manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
        model.learn(total_timesteps=settings["total_timesteps"], reset_num_timesteps=True, progress_bar=False)
        checkpoint = output / "model.zip"; model.save(checkpoint)
        delta_l2 = sum(float(((v.detach().cpu() - initial[k]).double() ** 2).sum())
                       for k, v in model.policy.state_dict().items()) ** .5
        manifest = _manifest(model, checkpoint, snapshot, seed, trained=True, initial_hash=initial_hash,
                             delta_l2=delta_l2, steps=int(model.num_timesteps))
        manifest["settings"] = settings
        manifest.update(started_at=start_stamp, completed_at=datetime.now(timezone.utc).isoformat(),
                        wall_seconds=time.monotonic()-started,
                        training_task_episode_counts={task.task_id: sum(e.task_counts[task.task_id] for e in env.envs)
                                                      for task in snapshot.tasks},
                        actual_environment_steps=sum(e.steps_total for e in env.envs),
                        completed_training_episodes=sum(e.steps_total // 8 for e in env.envs),
                        device=str(model.device), torch_version=torch.__version__, numpy_version=np.__version__)
        if not manifest["parameters_changed"] or delta_l2 <= 0:
            raise RuntimeError("PPO did not update its parameters")
        journal.flush()
        manifest["training_journal_sha256"] = file_digest(output / "training-episodes.jsonl")
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest
    finally:
        env.close(); journal.close()


class PPOSelector:
    """Checkpoint-only deterministic policy for replay or fresh GPU feedback."""
    def __init__(self, model, manifest, suite):
        self._model, self.manifest = model, deepcopy(manifest)
        if getattr(model, "env", None) is not None:
            raise ValueError("inference model must not retain a training environment")
        self.training_metadata = deepcopy(manifest["training"])
        self.configuration = {"method": "PPO", "library": "stable-baselines3", "version": SB3_VERSION,
                              "training_seed": manifest["seed"], "checkpoint_sha256": manifest["checkpoint_sha256"],
                              "deterministic": True, "fine_tuning": False,
                              "task_ids_in_network_observation": False}
        self._feature_count = len(suite.tasks[0].features)

    @classmethod
    def load(cls, checkpoint_path, metadata_path, suite):
        sb3 = _sb3()
        manifest = json.loads(Path(metadata_path).read_text())
        if (manifest.get("schema_version") != 1 or manifest.get("kind") != "drivebench_sb3_ppo_checkpoint"
                or manifest.get("library_version") != SB3_VERSION
                or manifest.get("suite_sha256") != suite.sha256
                or manifest.get("candidates") != [c.to_dict() for c in suite.candidates]
                or manifest.get("task_feature_count") != len(suite.tasks[0].features)
                or manifest.get("checkpoint_sha256") != file_digest(checkpoint_path)):
            raise ValueError("PPO checkpoint, suite or manifest binding differs")
        training = manifest.get("training", {})
        expected_ids = [suite.tasks[i].task_id for i in suite.splits["train"]]
        if (training.get("task_ids") != expected_ids or training.get("outcome_splits_read") != ["train"]
                or training.get("validation_or_test_outcomes_used") is not False):
            raise ValueError("PPO training provenance is not TRAIN-only")
        model = sb3.PPO.load(checkpoint_path, device="cpu", env=None)
        if parameter_hash(model) != manifest["final_parameter_sha256"]:
            raise ValueError("loaded PPO parameter hash differs")
        return cls(model, manifest, suite)

    def __call__(self, observation, info):
        # ``info`` is deliberately ignored; it can contain task IDs and raw
        # measurement receipts. Only the standard public numeric observation
        # is supplied to the network, exactly as during training.
        if set(observation) != set(self._model.observation_space.spaces):
            raise ValueError("PPO observation keys differ from training")
        action, _ = self._model.predict(observation, deterministic=True)
        return int(action)
