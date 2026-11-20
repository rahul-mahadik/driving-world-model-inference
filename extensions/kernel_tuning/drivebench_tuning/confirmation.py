"""Fresh CUDA confirmation of previously selected operator configurations.

Selections are fixed before this command starts. Confirmation never changes a
selection or supplies feedback to its policy. Run each repeat in a fresh process.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

from .adapters import OperatorEvaluator
from .benchmark import CudaEvaluator, EvidenceJournal, source_hashes, utc_now
from .suite import load_suite


def selection_pairs(document, suite):
    if document.get('suite_sha256') != suite.sha256:
        raise ValueError('Selection suite does not match the bundled workload')
    rows = document.get('selections')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Nonempty selections required')
    tasks = {t.task_id: t for t in suite.tasks}
    candidates = {c.candidate_id: c for c in suite.candidates}
    pairs = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Selection must be an object')
        task_id, candidate_id = row.get('task_id'), row.get('candidate_id')
        if task_id not in tasks or (candidate_id is not None and candidate_id not in candidates):
            raise ValueError('Unknown selected task or candidate')
        pair = (task_id, candidate_id)
        if pair not in seen:
            seen.add(pair)
            pairs.append((tasks[task_id], candidates.get(candidate_id)))
    return pairs


def selection_input_digests(document, pairs):
    """Bind fresh confirmation inputs to the originally measured task inputs."""
    digests = document.get('expected_input_digests')
    if not isinstance(digests, dict):
        raise ValueError('Original measured input digests required for confirmation')
    required = {task.task_id for task, _ in pairs}
    if any(task_id not in digests for task_id in required):
        raise ValueError('Selected task is missing its original measured input digest')
    for task_id in required:
        value = digests[task_id]
        if (not isinstance(value, str) or len(value) != 64
                or any(c not in '0123456789abcdef' for c in value)):
            raise ValueError('Expected input digest must be a SHA256 value')
    return {task_id: digests[task_id] for task_id in required}


def confirm(evaluator, pairs, *, repeat_index, seed, expected_cohort_id, expected_input_digests, journal):
    """Measure selected pairs against eager and compiled references, once each."""
    pairs = list(pairs)
    expected_input_digests = selection_input_digests(
        {'expected_input_digests': expected_input_digests}, pairs)
    random.Random(seed + repeat_index).shuffle(pairs)
    results = []
    trial_ids = set()
    digests = {}
    for pair_index, (task, candidate) in enumerate(pairs):
        # Rotate all three positions across repeats. The compiled comparator is
        # measured even when the policy selects eager or compiled itself.
        order = ['eager', 'compiled', 'selected']
        offset = (repeat_index + pair_index) % len(order)
        order = order[offset:] + order[:offset]
        measurements = {}
        for role in order:
            if role == 'eager' or (role == 'selected' and candidate is None):
                trial = evaluator.baseline(task)
            elif role == 'compiled':
                from .search import CandidateSpec
                trial = evaluator.evaluate(task, CandidateSpec('torch-compiled', {'implementation': 'compiled'}))
            else:
                trial = evaluator.evaluate(task, candidate)
            expected_candidate = ('torch-compiled' if role == 'compiled' else
                                  candidate.candidate_id if role == 'selected' and candidate else None)
            if (trial.backend != 'cuda' or trial.task_id != task.task_id or
                    trial.candidate_id != expected_candidate or trial.cohort_id != expected_cohort_id):
                raise ValueError('Confirmation hardware/runtime/task/candidate differs from selected cohort')
            if trial.trial_id in trial_ids:
                raise ValueError('Repeated confirmation trial ID')
            trial_ids.add(trial.trial_id)
            if (trial.input_digest != expected_input_digests[task.task_id]
                    or digests.setdefault(task.task_id, trial.input_digest) != trial.input_digest):
                raise ValueError('Confirmation inputs differ from original measured task')
            measurements[role] = trial.to_dict()
            journal.append({'event': 'confirmation_trial', 'repeat_index': repeat_index,
                            'selected_candidate_id': candidate.candidate_id if candidate else None,
                            'role': role, 'measurement': trial.to_dict()})
        selected = measurements['selected']
        ratios = {}
        for reference in ('eager', 'compiled'):
            base = measurements[reference]
            ratios[reference] = (base['latency_ms'] / selected['latency_ms']
                                if base['correct'] and selected['correct'] else None)
        results.append({'task_id': task.task_id,
                        'candidate_id': candidate.candidate_id if candidate else None,
                        'order': order, 'measurements': measurements,
                        'descriptive_speedup_vs': ratios})
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selections', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--repeat-index', required=True, type=int)
    p.add_argument('--max-seconds', default=600, type=int)
    p.add_argument('--seed', default=2901, type=int)
    p.add_argument('--repetitions', default=30, type=int)
    p.add_argument('--warmups', default=5, type=int)
    args = p.parse_args(argv)
    if args.repeat_index < 0 or args.seed < 0 or not 1 <= args.max_seconds <= 7200:
        p.error('Nonnegative repeat/seed and max-seconds 1..7200 required')
    raw = args.selections.read_bytes()
    selections = json.loads(raw)
    suite = load_suite()
    pairs = selection_pairs(selections, suite)
    input_digests = selection_input_digests(selections, pairs)
    cohort = selections.get('cohort_id')
    if not isinstance(cohort, str) or len(cohort) != 64:
        raise ValueError('Expected measured CUDA cohort_id required')
    journal = EvidenceJournal(args.output)
    journal.write('plan.json', {'schema_version': 1, 'created_at': utc_now(),
        'suite_sha256': suite.sha256, 'selection_sha256': hashlib.sha256(raw).hexdigest(),
        'selection_document': selections, 'source_hashes': source_hashes(),
        'repeat_index': args.repeat_index, 'seed': args.seed, 'max_seconds': args.max_seconds,
        'warmups': args.warmups, 'repetitions': args.repetitions,
        'backend': 'cuda', 'independent_of_policy_selection': True,
        'selection_updates_permitted': False, 'paired_reference_roles': ['eager', 'compiled']})
    evaluator = None
    try:
        evaluator = OperatorEvaluator(CudaEvaluator(journal, repetitions=args.repetitions,
            warmups=args.warmups, max_seconds=args.max_seconds))
        if evaluator.runner.cohort_id != cohort:
            raise ValueError('Confirmation is on a different physical GPU/runtime/source cohort')
        result = confirm(evaluator, pairs, repeat_index=args.repeat_index, seed=args.seed,
                         expected_cohort_id=cohort, expected_input_digests=input_digests, journal=journal)
        evaluator.close()
        evaluator = None
        journal.write('result.json', {'schema_version': 1, 'backend': 'cuda',
            'repeat_index': args.repeat_index, 'cohort_id': cohort, 'results': result,
            'selection_sha256': hashlib.sha256(raw).hexdigest(),
            'suite_sha256': suite.sha256, 'quality_preserving': False,
            'scope': 'Isolated operator correctness and fresh timing; no video quality or model speedup claim'})
        journal.write('completed.json', {'completed': True, 'at': utc_now(),
            'selected_pairs': len(pairs), 'cuda_trials': len(pairs) * 3,
            'selection_updated': False})
        return 0
    except Exception as exc:
        journal.write('failed.json', {'completed': False, 'at': utc_now(),
            'error_type': type(exc).__name__, 'error': str(exc)})
        return 1
    finally:
        if evaluator is not None:
            try:
                evaluator.close()
            except Exception as exc:
                journal.write('cleanup-failed.json', {'error_type': type(exc).__name__, 'error': str(exc)})


if __name__ == '__main__':
    raise SystemExit(main())
