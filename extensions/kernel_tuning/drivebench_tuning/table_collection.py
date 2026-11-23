"""Collect and merge the existing operator suite on one physical CUDA device.

Each ``collect`` invocation is one fresh-process repeat. ``merge`` accepts only
complete repeats of one protocol/runtime. Incorrect and failed candidate rows
remain in the table; complete means all attempts were recorded, not all passed.
The table is a record of measurements, not a substitute for live confirmation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time
import uuid

from .adapters import OperatorEvaluator
from .benchmark import COMPILER_STATE_POLICY, CudaEvaluator, EvidenceJournal, percentile, source_hashes, utc_now
from .search import TrialResult
from .suite import BUNDLED_SUITES, load_bound_suite, load_suite, suite_task_indices

_PROCESS_INSTANCE_ID = str(uuid.uuid4())
_ARTIFACTS = ('plan.json', 'runtime.json', 'rows.json', 'trials.jsonl')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError('Nonfinite JSON value')
    result = json.loads(Path(path).read_text(), object_pairs_hook=pairs, parse_constant=nonfinite)
    json.dumps(result, allow_nan=False)
    return result


def collection_order(suite, seed, repeat_index, task_splits=None):
    """Shuffle tasks; rotate a fixed per-task candidate permutation each repeat."""
    tasks = list(suite_task_indices(suite, task_splits))
    random.Random(seed + 100003 * repeat_index).shuffle(tasks)
    order = []
    for task_index in tasks:
        task = suite.tasks[task_index]
        candidates = list(range(len(suite.candidates)))
        random.Random(seed + 101 * task_index).shuffle(candidates)
        offset = (3 * repeat_index) % len(candidates)
        candidates = candidates[offset:] + candidates[:offset]
        order.append((task_index, None))
        order.extend((task_index, candidate_index) for candidate_index in candidates)
    return order


def _protocol(suite, *, planned_repeats, repetitions, warmups, seed, max_seconds, task_splits=None):
    if type(planned_repeats) is not int or not 2 <= planned_repeats <= 30:
        raise ValueError('planned_repeats must be 2..30')
    if type(repetitions) is not int or not 3 <= repetitions <= 1000:
        raise ValueError('repetitions must be 3..1000')
    if type(warmups) is not int or not 1 <= warmups <= 100:
        raise ValueError('warmups must be 1..100')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be uint32')
    if type(max_seconds) not in (float, int) or not math.isfinite(max_seconds) or not 0 < max_seconds <= 7200:
        raise ValueError('max_seconds must be finite in (0,7200]')
    suite_task_indices(suite, task_splits)
    protocol = {'schema_version': 1, 'kind': 'operator_table_protocol', 'backend': 'cuda',
            'suite_sha256': suite.sha256, 'source_hashes': source_hashes(),
            'compiler_state_policy': dict(COMPILER_STATE_POLICY),
            'planned_repeats': planned_repeats, 'repetitions': repetitions,
            'warmups': warmups, 'seed': seed, 'max_seconds': max_seconds,
            'order_policy': 'seeded task shuffle; baseline first; per-task candidate permutation rotated by 3 each repeat',
            'timing_boundary': suite.document['timing_boundary'],
            'memory_boundary': suite.document['memory_boundary'],
            'quality_policy': suite.document['quality_policy']}
    if task_splits is not None:
        protocol['task_splits'] = [s for s in ('train', 'validation', 'test') if s in task_splits]
        protocol['coverage_scope'] = 'All candidates and eager reference for the selected suite splits only.'
    return protocol


def _check_row(row, suite, protocol, *, cohort_id, expected_repeat, expected_index, expected_pair):
    if set(row) != {'repeat_index', 'order_index', 'measurement'}:
        raise ValueError('Unknown or missing measurement row fields')
    if (type(row['repeat_index']) is not int or row['repeat_index'] != expected_repeat
            or type(row['order_index']) is not int or row['order_index'] != expected_index):
        raise ValueError('Repeat or order index differs')
    result = TrialResult(**row['measurement'])
    task_index, candidate_index = expected_pair
    candidate_id = None if candidate_index is None else suite.candidates[candidate_index].candidate_id
    if (result.task_id != suite.tasks[task_index].task_id or result.candidate_id != candidate_id
            or result.backend != 'cuda' or result.cohort_id != cohort_id):
        raise ValueError('Measurement identity differs from collection plan')
    evidence = result.to_dict()['evidence']
    expected_impl = 'eager' if candidate_index is None else suite.candidates[candidate_index].config['implementation']
    if (evidence.get('trial_id') != result.trial_id or evidence.get('cohort_id') != cohort_id
            or evidence.get('implementation') != expected_impl):
        raise ValueError('Evidence identity differs')
    hashes = evidence.get('input_sha256')
    if (not isinstance(hashes, list) or not hashes
            or any(not isinstance(h, str) or len(h) != 64 or any(c not in '0123456789abcdef' for c in h) for h in hashes)
            or _digest(hashes) != result.input_digest):
        raise ValueError('Input identity missing or inconsistent')
    if evidence.get('repetitions') != protocol['repetitions'] or evidence.get('warmups') != protocol['warmups']:
        raise ValueError('Timing sample protocol differs')
    if result.status == 'ok':
        samples = evidence.get('cuda_event_ms', [])
        memory = evidence.get('peak_allocated_delta_bytes', [])
        host = evidence.get('host_wall_ms', [])
        if (len(samples) != protocol['repetitions'] or len(memory) != len(samples) or len(host) != len(samples)
                or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in samples + host)
                or any(type(v) is not int or v < 0 for v in memory)
                or statistics.median(samples) != result.latency_ms
                or max(memory) != result.peak_allocated_bytes):
            raise ValueError('Raw sample summary differs')
        correctness = evidence.get('correctness', {})
        if (correctness.get('passed') is not True or correctness.get('finite') is not True
                or correctness.get('shape_dtype_match') is not True
                or evidence.get('input_integrity_verified') is not True):
            raise ValueError('Successful trial lacks correctness or input-integrity evidence')
    elif result.latency_ms is not None or result.peak_allocated_bytes is not None:
        raise ValueError('Failed measurements cannot contain accepted metrics')
    if candidate_index is None and not result.correct:
        raise ValueError('Reference measurement failed; repeat cannot be complete')
    return result


def collect(directory, *, repeat_index, planned_repeats=3, repetitions=30,
            warmups=5, seed=1901, max_seconds=1200, suite=None, task_splits=None,
            runner_factory=CudaEvaluator):
    """Collect one repeat. Runner injection supports CPU contract tests only."""
    suite = load_suite(suite)
    load_bound_suite(suite.document, suite.sha256)
    protocol = _protocol(suite, planned_repeats=planned_repeats, repetitions=repetitions,
                         warmups=warmups, seed=seed, max_seconds=max_seconds, task_splits=task_splits)
    if type(repeat_index) is not int or not 0 <= repeat_index < planned_repeats:
        raise ValueError('repeat_index outside planned repeats')
    order = collection_order(suite, seed, repeat_index, protocol.get('task_splits'))
    journal = EvidenceJournal(directory)
    plan = {'schema_version': 1, 'created_at': utc_now(), 'repeat_index': repeat_index,
            'process_instance_id': _PROCESS_INSTANCE_ID, 'protocol': protocol,
            'collection_protocol_sha256': _digest(protocol), 'suite': suite.document,
            'order': [{'task_id': suite.tasks[t].task_id,
                       'candidate_id': None if c is None else suite.candidates[c].candidate_id} for t, c in order]}
    journal.write('plan.json', plan)
    rows, evaluator = [], None
    start = time.monotonic()
    try:
        evaluator = OperatorEvaluator(runner_factory(journal, repetitions=repetitions, warmups=warmups,
                                                      max_seconds=max_seconds))
        cohort_id = evaluator.runner.cohort_id
        if _digest(evaluator.runner.runtime.get('compiler_state_policy')) != _digest(protocol['compiler_state_policy']):
            raise ValueError('Evaluator compiler-state policy differs from collection plan')
        compile_threads = evaluator.runner.runtime.get('inductor_compile_threads')
        if type(compile_threads) is not int or compile_threads != protocol['compiler_state_policy']['inductor_compile_threads']:
            raise ValueError('Evaluator Inductor compile thread count differs from collection plan')
        trial_ids, input_ids = set(), {}
        for index, (task_index, candidate_index) in enumerate(order):
            task = suite.tasks[task_index]
            result = (evaluator.baseline(task) if candidate_index is None else
                      evaluator.evaluate(task, suite.candidates[candidate_index]))
            row = {'repeat_index': repeat_index, 'order_index': index, 'measurement': result.to_dict()}
            # Append before validation so malformed adapter output also survives.
            journal.append({'event': 'table_measurement', 'row': row})
            rows.append(row)
            _check_row(row, suite, protocol, cohort_id=cohort_id, expected_repeat=repeat_index,
                       expected_index=index, expected_pair=(task_index, candidate_index))
            if result.trial_id in trial_ids:
                raise ValueError('Trial identifier reused')
            trial_ids.add(result.trial_id)
            if input_ids.setdefault(task.task_id, result.input_digest) != result.input_digest:
                raise ValueError('Task inputs changed within repeat')
        evaluator.close()
        evaluator = None
        journal.write('rows.json', rows)
        journal.write('completed.json', {'completed': True, 'at': utc_now(), 'backend': 'cuda', 'synthetic': False,
                      'repeat_index': repeat_index, 'process_instance_id': _PROCESS_INSTANCE_ID,
                      'collection_protocol_sha256': _digest(protocol), 'cohort_id': cohort_id,
                      'measurement_count': len(rows), 'wall_seconds': time.monotonic() - start,
                      'artifact_sha256': {name: _file_digest(Path(directory) / name) for name in _ARTIFACTS}})
        return 0
    except Exception as exc:
        journal.write('failed.json', {'completed': False, 'at': utc_now(), 'repeat_index': repeat_index,
                      'error_type': type(exc).__name__, 'error': str(exc), 'measurement_count': len(rows)})
        return 1
    finally:
        if not (Path(directory) / 'rows.json').exists():
            journal.write('rows.json', rows)
        if evaluator is not None:
            try:
                evaluator.close()
            except Exception as exc:
                journal.write('cleanup-failed.json', {'error_type': type(exc).__name__, 'error': str(exc)})


def _cell_summaries(rows):
    groups = {}
    for row in rows:
        m = row['measurement']
        groups.setdefault((m['task_id'], m['candidate_id']), []).append(m)
    cells = []
    for (task_id, candidate_id), measurements in groups.items():
        successful = [m for m in measurements if m['status'] == 'ok']
        medians = [m['latency_ms'] for m in successful]
        cells.append({'task_id': task_id, 'candidate_id': candidate_id,
                      'attempts': len(measurements), 'status_counts': dict(Counter(m['status'] for m in measurements)),
                      'all_repeats_correct': len(successful) == len(measurements),
                      'repeat_medians_ms': medians,
                      'median_latency_ms': statistics.median(medians) if medians else None,
                      'repeat_median_min_ms': min(medians) if medians else None,
                      'repeat_median_max_ms': max(medians) if medians else None,
                      'repeat_median_stdev_ms': statistics.stdev(medians) if len(medians) > 1 else None,
                      'within_repeat_p95_minus_p05_ms': [percentile(m['evidence']['cuda_event_ms'], .95)
                                                       - percentile(m['evidence']['cuda_event_ms'], .05) for m in successful],
                      'peak_allocated_bytes': max((m['peak_allocated_bytes'] for m in successful), default=None),
                      'noise_scope': 'sample spread within a process and dispersion of fresh-process medians; not confidence intervals'})
    return cells


def validate_table(table):
    """Validate structure, full coverage, raw summaries and one runtime binding.

    This verifies internal consistency, not the authenticity of arbitrary input
    JSON. Published table bytes must also match the publisher's artifact digest.
    """
    suite = load_bound_suite(table.get('suite'), table.get('suite_sha256'))
    if (table.get('schema_version') != 1 or table.get('kind') != 'measured_operator_table'
            or table.get('backend') != 'cuda' or table.get('synthetic') is not False
            or table.get('complete') is not True or table.get('suite_sha256') != suite.sha256
            or table.get('suite') != suite.document):
        raise ValueError('Table kind, suite, completeness or backend differs')
    protocol = table['protocol']
    if (_digest(protocol) != table.get('collection_protocol_sha256')
            or protocol.get('suite_sha256') != suite.sha256 or protocol.get('backend') != 'cuda'):
        raise ValueError('Collection protocol binding differs')
    planned = protocol['planned_repeats']
    if type(planned) is not int or not 2 <= planned <= 30 or table.get('planned_repeats') != planned:
        raise ValueError('Invalid planned repeat count')
    if table.get('repeat_indices') != list(range(planned)):
        raise ValueError('Missing or repeated collection repeats')
    runtime = dict(table['runtime'])
    cohort_id = runtime.pop('cohort_id')
    if (_digest(runtime) != cohort_id or table.get('cohort_id') != cohort_id
            or runtime.get('source_hashes') != protocol.get('source_hashes') or not runtime.get('gpu_uuid')):
        raise ValueError('Runtime identity differs')
    collection_policy = protocol.get('compiler_state_policy')
    runtime_policy = runtime.get('compiler_state_policy')
    if collection_policy is not None or runtime_policy is not None:
        if (_digest(collection_policy) != _digest(COMPILER_STATE_POLICY)
                or _digest(runtime_policy) != _digest(collection_policy)):
            raise ValueError('Compiler-state policy differs between collection and runtime')
        compile_threads = runtime.get('inductor_compile_threads')
        if type(compile_threads) is not int or compile_threads != collection_policy['inductor_compile_threads']:
            raise ValueError('Runtime Inductor compile thread count differs from collection policy')
    else:
        # Historical tables remain replayable. This source revision cannot omit
        # the policy that its evaluator and generated collection plan declare.
        current_benchmark_sha = source_hashes().get('benchmark.py')
        if current_benchmark_sha is not None and protocol['source_hashes'].get('benchmark.py') == current_benchmark_sha:
            raise ValueError('Compiler-state policy missing from current-source table')
    runs = table['runs']
    if (len(runs) != planned or sorted(r['repeat_index'] for r in runs) != list(range(planned))
            or len({r['process_instance_id'] for r in runs}) != planned):
        raise ValueError('Repeats must have distinct fresh-process identities')
    rows = table['rows']
    task_splits = protocol.get('task_splits')
    width = len(suite_task_indices(suite, task_splits)) * (len(suite.candidates) + 1)
    if len(rows) != width * planned:
        raise ValueError('Table does not cover every planned cell')
    identities, input_ids = set(), {}
    for repeat_index in range(planned):
        order = collection_order(suite, protocol['seed'], repeat_index, task_splits)
        for index, pair in enumerate(order):
            result = _check_row(rows[repeat_index * width + index], suite, protocol, cohort_id=cohort_id,
                                expected_repeat=repeat_index, expected_index=index, expected_pair=pair)
            if result.trial_id in identities:
                raise ValueError('Trial identifier reused across table')
            identities.add(result.trial_id)
            if input_ids.setdefault(result.task_id, result.input_digest) != result.input_digest:
                raise ValueError('Task inputs changed between repeats')
    if table.get('cells') != _cell_summaries(rows):
        raise ValueError('Cell summaries differ from raw rows')
    return table


def merge(runs, output):
    """Merge fresh-process repeat directories; refuse incomplete or mixed runs."""
    if Path(output).exists():
        raise FileExistsError(output)
    records = []
    for directory in map(Path, runs):
        if (directory / 'failed.json').exists() or (directory / 'cleanup-failed.json').exists():
            raise ValueError('Cannot merge a failed or partial repeat')
        done = _read_json(directory / 'completed.json')
        if done.get('completed') is not True or done.get('synthetic') is not False or done.get('backend') != 'cuda':
            raise ValueError('Repeat has no completed CUDA evidence')
        if done.get('artifact_sha256') != {name: _file_digest(directory / name) for name in _ARTIFACTS}:
            raise ValueError('Repeat artifact digest differs')
        plan, rows, runtime = (_read_json(directory / name) for name in ('plan.json', 'rows.json', 'runtime.json'))
        if (done['repeat_index'] != plan['repeat_index'] or done['process_instance_id'] != plan['process_instance_id']
                or done['collection_protocol_sha256'] != plan['collection_protocol_sha256']
                or done['cohort_id'] != runtime.get('cohort_id') or done['measurement_count'] != len(rows)):
            raise ValueError('Completion record disagrees with repeat')
        journal_rows = [json.loads(line)['row'] for line in (directory / 'trials.jsonl').read_text().splitlines()
                        if json.loads(line).get('event') == 'table_measurement']
        if journal_rows != rows:
            raise ValueError('Append-only journal disagrees with rows')
        records.append((plan, rows, runtime, {**done, 'artifact_sha256': {
            **done['artifact_sha256'], 'completed.json': _file_digest(directory / 'completed.json')}}))
    if not records:
        raise ValueError('No repeat directories supplied')
    records.sort(key=lambda record: record[0]['repeat_index'])
    first_plan, _, runtime, _ = records[0]
    protocol = first_plan['protocol']
    suite = load_bound_suite(first_plan['suite'], protocol['suite_sha256'])
    if any(p['protocol'] != protocol or p['suite'] != first_plan['suite'] or r != runtime for p, _, r, _ in records):
        raise ValueError('Cannot pool different protocols or physical runtimes')
    for plan, _, _, _ in records:
        expected_order = [{'task_id': suite.tasks[t].task_id,
                           'candidate_id': None if c is None else suite.candidates[c].candidate_id}
                          for t, c in collection_order(suite, protocol['seed'], plan['repeat_index'],
                                                       protocol.get('task_splits'))]
        if plan['order'] != expected_order:
            raise ValueError('Recorded order differs from selected suite coverage')
    rows = [row for _, repeat_rows, _, _ in records for row in repeat_rows]
    table = {'schema_version': 1, 'kind': 'measured_operator_table', 'backend': 'cuda', 'synthetic': False,
             'complete': True, 'created_at': utc_now(), 'suite_sha256': protocol['suite_sha256'],
             'suite': first_plan['suite'], 'protocol': protocol,
             'collection_protocol_sha256': first_plan['collection_protocol_sha256'],
             'cohort_id': runtime['cohort_id'], 'runtime': runtime,
             'planned_repeats': protocol['planned_repeats'],
             'repeat_indices': [p['repeat_index'] for p, _, _, _ in records],
             'runs': [d for _, _, _, d in records], 'rows': rows, 'cells': _cell_summaries(rows),
             'use': 'Measured-data replay for configuration-search research; not fresh GPU execution or a live confirmation result.'}
    validate_table(table)
    with Path(output).open('x') as stream:
        json.dump(table, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('collect', help='One fresh-process CUDA repeat of all suite cells')
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--suite', choices=BUNDLED_SUITES, default=None,
                     help='Frozen bundled suite (default: operator-search-v1)')
    run.add_argument('--split', dest='task_splits', choices=('train', 'validation', 'test'),
                     action='append', help='Collect only named splits; repeat flag to select several')
    run.add_argument('--repeat-index', type=int, required=True)
    run.add_argument('--planned-repeats', type=int, default=3)
    run.add_argument('--repetitions', type=int, default=30)
    run.add_argument('--warmups', type=int, default=5)
    run.add_argument('--seed', type=int, default=1901)
    run.add_argument('--max-seconds', type=int, default=1200)
    combine = sub.add_parser('merge', help='Validate complete repeat directories and write one table')
    combine.add_argument('--runs', type=Path, nargs='+', required=True)
    combine.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'collect':
        status = collect(**{k: v for k, v in vars(args).items() if k not in ('command', 'output')}, directory=args.output)
        print(json.dumps({'completed': status == 0, 'backend': 'cuda', 'output': str(args.output)}))
        return status
    table = merge(args.runs, args.output)
    print(json.dumps({'completed': True, 'rows': len(table['rows']), 'cells': len(table['cells']),
                      'table_sha256': _file_digest(args.output), 'output': str(args.output)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
