"""Run after installing the package. No CUDA, data, checkpoints or claimed speedup."""
import json
import numpy as np
from drivebench_tuning.adapters import SyntheticEvaluator
from drivebench_tuning.evaluation import evaluate_policy
from drivebench_tuning.search import BudgetedTuningEnv
from drivebench_tuning.suite import load_suite

suite = load_suite()

def make_env():
    return BudgetedTuningEnv(suite.tasks, suite.candidates, SyntheticEvaluator(),
                            trial_budget=4, allow_synthetic=True)

def my_policy(observation, info):
    # A minimal external policy that reads history. Replace with your agent.
    counts = observation['action_counts']
    return int(np.flatnonzero(counts == counts.min())[0])

if __name__ == '__main__':
    result = evaluate_policy(make_env, my_policy, suite.splits['validation'], repeats=1, seed=23)
    print(json.dumps(result, indent=2, allow_nan=False))
