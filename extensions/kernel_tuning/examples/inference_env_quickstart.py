"""Install both drivebench and its tuning extension. CPU test-double wiring only."""
import json
import numpy as np
from drivebench.engine import InferenceEngine
from drivebench.types import GenerationConfig, OptimizationConfig
from drivebench_tuning.inference import InferenceEvaluator, InferenceRequest, QualityDecision
from drivebench_tuning.search import BudgetedTuningEnv, CandidateSpec, TaskSpec
from drivebench_tuning.evaluation import evaluate_policy, grid_policy

TASK = TaskSpec('synthetic-video-batch', (8., 8., 2.), {'purpose': 'CPU interface test, not a driving model'})
CONFIG = {'generation': {'total_frames': 3, 'height': 8, 'width': 8, 'steps': 2},
          'optimizations': {'max_batch_size': 1}}
CANDIDATES = (CandidateSpec('batch-2', {'optimizations': {'max_batch_size': 2}}),
              CandidateSpec('decode-chunk-1', {'generation': {'decode_chunk_size': 1}}))


def request_loader(task):
    return InferenceRequest(np.zeros((1, 3, 8, 8), np.float32),
                            (np.zeros(8, np.float32), np.ones(8, np.float32)), (11, 12))


def engine_factory(task, config):
    return InferenceEngine(model={'backend': 'synthetic'},
                           generation=GenerationConfig(**config['generation']),
                           optimizations=OptimizationConfig(**config['optimizations']))


def quality_gate(task, candidate, result, reference):
    passed = reference is None or all(np.array_equal(result.frames[key], reference.frames[key]) for key in result.frames)
    return QualityDecision(passed, {'policy': 'exact CPU fixture output equality against fresh reference',
                                    'reference_validity_only': reference is None,
                                    'visual_quality_established': False})


def make_env():
    evaluator = InferenceEvaluator(request_loader, engine_factory, quality_gate, CONFIG,
                                   backend='synthetic', cohort_id='cpu-inference-fixture-v1')
    return BudgetedTuningEnv([TASK], CANDIDATES, evaluator, trial_budget=1, allow_synthetic=True)


if __name__ == '__main__':
    report = evaluate_policy(make_env, grid_policy, [0], repeats=1)
    report['example_scope'] = 'Actual core InferenceEngine CPU test double; no Vista, GPU performance or quality evidence'
    print(json.dumps(report, indent=2, allow_nan=False))
