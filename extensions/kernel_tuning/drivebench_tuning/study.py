"""Bounded operator study using the evaluator, not a cloud orchestration layer."""
from dataclasses import asdict
import math
import statistics
import numpy as np

from .contract import DEFAULT_CONFIGS, KernelConfig, Workload


def load_workloads(path):
    import json
    from pathlib import Path
    document = json.loads(Path(path).read_text())
    if not isinstance(document, dict) or not isinstance(document.get('workloads'), list):
        raise ValueError('Expected a document containing a workloads list')
    workloads = tuple(Workload(**row) for row in document['workloads'])
    if not 1 <= len(workloads) <= 32 or len({w.key() for w in workloads}) != len(workloads):
        raise ValueError('Provide 1..32 unique workloads')
    return document, workloads


def comparison(baseline, candidate):
    """Descriptive comparison of two successful matching CUDA operator trials."""
    if baseline.backend != 'cuda' or candidate.backend != 'cuda':
        raise ValueError('Only actual CUDA measurements can establish a comparison')
    if baseline.workload_key != candidate.workload_key:
        raise ValueError('Cannot compare different workloads')
    if not baseline.evidence.get('cohort_id') or baseline.evidence['cohort_id'] != candidate.evidence.get('cohort_id'):
        raise ValueError('Cannot compare different/unbound hardware/runtime cohorts')
    if baseline.evidence.get('input_sha256') != candidate.evidence.get('input_sha256') or not baseline.evidence.get('input_sha256'):
        raise ValueError('Cannot compare different/unbound inputs')
    if baseline.status != 'ok' or candidate.status != 'ok' or not baseline.correct or not candidate.correct:
        return {'comparable': False, 'reason': 'correctness or execution failure'}
    for m in (baseline, candidate):
        if m.latency_ms is None or not math.isfinite(m.latency_ms) or m.latency_ms <= 0:
            raise ValueError('Invalid latency')
        if type(m.peak_allocated_bytes) is not int or m.peak_allocated_bytes <= 0:
            raise ValueError('Invalid memory measurement')
    return {'comparable': True,
            'latency_reduction_percent': 100 * (1 - candidate.latency_ms / baseline.latency_ms),
            'peak_allocation_reduction_percent': 100 * (1 - candidate.peak_allocated_bytes / baseline.peak_allocated_bytes),
            'memory_reduced': candidate.peak_allocated_bytes < baseline.peak_allocated_bytes,
            'boundary': 'isolated operator; does not establish end-to-end model benefit'}


def run_sweep(evaluator, workloads, *, configs=DEFAULT_CONFIGS, rounds=3, seed=0):
    if type(rounds) is not int or not 2 <= rounds <= 10:
        raise ValueError('rounds must be 2..10')
    rng = np.random.default_rng(seed)
    records, confirmations = [], []
    for workload in workloads:
        candidates = {config: [] for config in configs}
        for round_index in range(rounds):
            for config_index in rng.permutation(len(configs)):
                config = configs[int(config_index)]
                roles = ('eager', 'triton', 'compiled') if round_index % 2 == 0 else ('compiled', 'triton', 'eager')
                measurements = {role: evaluator.measure(workload, config if role == 'triton' else None,
                                                        implementation=role) for role in roles}
                row = {'workload_key': workload.key(), 'round': round_index, 'config': asdict(config),
                       'order': list(roles), 'measurements': {r: m.to_dict() for r, m in measurements.items()}}
                records.append(row)
                evaluator.journal.append({'event': 'sweep_pair', **row})
                candidate = measurements['triton']
                # A winner must pass in every exploratory round, including controls.
                if all(m.status == 'ok' and m.correct for m in measurements.values()):
                    candidates[config].append(candidate.latency_ms)
        eligible = [c for c, values in candidates.items() if len(values) == rounds]
        if not eligible:
            confirmations.append({'workload_key': workload.key(), 'selected_config': None,
                                  'memory_reduction_validated': False, 'reason': 'no fully valid candidate'})
            continue
        winner = min(eligible, key=lambda c: statistics.median(candidates[c]))
        # Fresh independent measurements follow selection. Never report the best
        # exploratory sample as a confirmation of improvement.
        checks = []
        for order in (('eager', 'compiled', 'triton'), ('triton', 'compiled', 'eager')):
            measured = {role: evaluator.measure(workload, winner if role == 'triton' else None,
                                                implementation=role) for role in order}
            checks.append({'order': list(order), 'measurements': {r: m.to_dict() for r, m in measured.items()},
                           'versus_eager': comparison(measured['eager'], measured['triton']),
                           'versus_compiled': comparison(measured['compiled'], measured['triton'])})
        confirmed = all(c['versus_eager'].get('memory_reduced', False) for c in checks)
        confirmations.append({'workload_key': workload.key(), 'selected_config': asdict(winner),
                              'selection': 'lowest median exploratory latency among configurations passing every round',
                              'confirmation': checks, 'memory_reduction_validated': confirmed,
                              'claim_scope': 'peak incremental allocator bytes versus eager for this operator/input/runtime only'})
    return {'schema_version': 1, 'study': 'separate operator kernel cohort', 'backend': 'cuda',
            'cohort_id': evaluator.cohort_id, 'workloads': [asdict(w) for w in workloads],
            'configs': [asdict(c) for c in configs], 'rounds': rounds, 'seed': seed,
            'records': records, 'confirmations': confirmations,
            'whole_model_kernel_benefit_validated': False, 'quality_preserving': False,
            'old_compiler_speedup_attributed_to_kernels': False}
