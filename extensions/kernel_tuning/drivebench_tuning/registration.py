"""Registered Gymnasium entry point for the measured operator benchmark."""

from __future__ import annotations

from os import PathLike

import gymnasium as gym

from .replay import MeasuredReplayEvaluator, MeasuredTable
from .search import BudgetedTuningEnv
from .suite import load_suite


class OperatorTuneEnv(BudgetedTuningEnv):
    """Replay the v2 operator suite through ``gymnasium.make``.

    ``table_path`` must name a complete, validated recorded table. Construction
    never downloads data or supplies synthetic outcomes. ``split`` restricts
    task selection; reset's ``task_index`` is relative to that split.

    Seeding reset reproduces task selection, the starting recorded repeat and
    subsequent observations/rewards for the same actions. Evidence event IDs
    remain unique across episodes. Repeated actions cycle through the recorded
    process repeats, as in ``MeasuredReplayEvaluator``.
    """

    def __init__(
        self,
        *,
        table_path: str | PathLike[str],
        split: str = "train",
        trial_budget: int = 8,
        memory_weight: float = 0.0,
        render_mode: None = None,
    ) -> None:
        if render_mode is not None:
            raise ValueError("OperatorTune-v0 has no render modes; use render_mode=None")
        suite = load_suite("operator-generalization-v2")
        if not isinstance(split, str) or split not in suite.splits:
            raise ValueError("split must be train, validation or test")
        table = MeasuredTable.load(table_path, suite=suite)
        tasks = tuple(suite.tasks[index] for index in suite.splits[split])
        available = {task.task_id for task in table.measured_tasks}
        if any(task.task_id not in available for task in tasks):
            raise ValueError(f"table does not contain the complete {split} split")
        self.render_mode = None
        self.split = split
        self.suite = suite
        self._table = table
        super().__init__(
            tasks, suite.candidates, MeasuredReplayEvaluator(table),
            trial_budget=trial_budget, memory_weight=memory_weight, allow_replay=True,
        )

    def reset(self, *, seed=None, options=None):
        if self._closed:
            raise RuntimeError("environment is closed")
        # Seed both task selection and the recorded-repeat draw. A new evaluator
        # makes every candidate's first query use this episode's baseline repeat.
        gym.Env.reset(self, seed=seed)
        replay_seed = int(self.np_random.integers(len(self._table.repeat_indices)))
        self.evaluator.close()
        self.evaluator = MeasuredReplayEvaluator(self._table, seed=replay_seed)
        return super().reset(seed=None, options=options)
