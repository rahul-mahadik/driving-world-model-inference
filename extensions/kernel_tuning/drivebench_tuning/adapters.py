"""Adapters for the existing CUDA runner and an explicit CPU interface fixture.

The synthetic evaluator invents metrics to exercise the API. It never measures
GPU performance and is not a training dataset or evidence of tuning benefit.
"""
import hashlib
import json
import uuid
from .contract import KernelConfig, Workload
from .search import TrialResult


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


class OperatorEvaluator:
    """Reuse the measured CUDA runner, including its correctness/evidence gates."""
    def __init__(self, cuda_evaluator):
        self.runner = cuda_evaluator

    def _run(self, task, candidate):
        workload = Workload(**task.to_dict()['metadata']['workload'])
        config = None if candidate is None else candidate.to_dict()['config']
        if config is None:
            result = self.runner.baseline(workload)
        elif config == {'implementation': 'compiled'}:
            result = self.runner.compiled_baseline(workload)
        elif set(config) == {'implementation', 'block_size', 'num_warps'} and config['implementation'] == 'triton':
            result = self.runner.evaluate(workload, KernelConfig(config['block_size'], config['num_warps']))
        else:
            raise ValueError('Unsupported candidate for the example operator adapter')
        if result.workload_key != workload.key():
            raise ValueError('CUDA runner returned another workload')
        if result.backend != 'cuda':
            raise ValueError('OperatorEvaluator requires measured CUDA results')
        evidence = result.evidence
        if not evidence.get('input_sha256'):
            raise ValueError('CUDA trial lacks input hashes; incomplete trial preserved by runner')
        expected_config = (KernelConfig(config['block_size'], config['num_warps'])
                           if config and config['implementation'] == 'triton' else None)
        expected_impl = config['implementation'] if config else 'eager'
        if result.config != expected_config or evidence.get('implementation') != expected_impl:
            raise ValueError('CUDA runner returned another candidate implementation')
        return TrialResult(trial_id=evidence['trial_id'], task_id=task.task_id,
            candidate_id=None if candidate is None else candidate.candidate_id,
            backend='cuda', cohort_id=evidence['cohort_id'], input_digest=_digest(evidence['input_sha256']),
            status=result.status, correct=result.correct, latency_ms=result.latency_ms,
            peak_allocated_bytes=result.peak_allocated_bytes, evidence=evidence)

    def baseline(self, task):
        return self._run(task, None)

    def evaluate(self, task, candidate):
        return self._run(task, candidate)

    def close(self):
        self.runner.close()


class SyntheticEvaluator:
    """Deterministic invented costs for interface demos, always marked synthetic."""
    def __init__(self, journal=None):
        self.journal = journal

    def _run(self, task, candidate):
        signature = _digest({'task': task.to_dict(), 'candidate': None if candidate is None else candidate.to_dict()})
        ratio = 1. if candidate is None else .6 + int(signature[:4], 16) / 65535 * .8
        result = TrialResult(trial_id=str(uuid.uuid4()), task_id=task.task_id,
            candidate_id=None if candidate is None else candidate.candidate_id,
            backend='synthetic', cohort_id='synthetic-interface-v1', input_digest=_digest(task.to_dict()),
            status='ok', correct=True, latency_ms=ratio,
            peak_allocated_bytes=1000000 if candidate is None else int(1000000 * ratio),
            evidence={'invented_metrics': True, 'purpose': 'CPU interface demonstration only',
                      'correctness_policy': 'fixture only; no tensor execution or GPU measurement'})
        if self.journal:
            self.journal.append({'event': 'synthetic_trial', 'measurement': result.to_dict()})
        return result

    def baseline(self, task):
        return self._run(task, None)

    def evaluate(self, task, candidate):
        return self._run(task, candidate)

    def close(self):
        pass
