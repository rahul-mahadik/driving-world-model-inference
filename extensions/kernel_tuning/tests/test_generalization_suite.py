"""Prospective suite/collection contracts; CPU fixtures are never measurements."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math

import pytest

from drivebench_tuning import confirmation, table_collection as tc
from drivebench_tuning.contract import DTYPE_TOLERANCES, Workload
from drivebench_tuning.kernels import MAX_GROUP_CHUNKS, SUPPORTED_BLOCK_SIZES, SUPPORTED_NUM_WARPS
from drivebench_tuning.replay import MeasuredTable, MeasuredReplayEvaluator
from drivebench_tuning.suite import load_bound_suite, load_suite, suite_task_indices
from test_table_collection import FixtureRunner


SUITE = 'operator-generalization-v2'


def test_new_suite_has_frozen_shape_splits_explicit_tolerances_and_supported_actions():
    legacy = load_suite()
    assert legacy.sha256 == '3413eeda654a677ead96547581a01e7fb8f5145f99aa02e5481b84ec0b8a3e76'
    suite = load_suite(SUITE)
    assert suite.sha256 == '5ef382542f26ca5b351805b88ae794e676c175b438ba14a1d9caf047975fda7f'
    assert len(suite.tasks) == 60 and len(suite.candidates) == 25
    assert [len(suite.splits[k]) for k in ('train', 'validation', 'test')] == [30, 12, 18]
    assert suite.document['quality_tolerances'] == DTYPE_TOLERANCES
    assert len({len(task.features) for task in suite.tasks}) == 1
    shapes = {}
    for task in suite.tasks:
        w = Workload(**task.metadata['workload'])
        key = (w.operator, w.shape)
        shapes.setdefault(key, []).append((task.metadata['split'], w.dtype, w.seed))
        assert math.prod(w.shape) * (4 if w.dtype == 'float32' else 2) <= 256 * 1024**2
        if w.operator == 'groupnorm_silu':
            assert math.ceil(math.prod(w.shape[1:]) / w.num_groups / 64) <= MAX_GROUP_CHUNKS
    assert len(shapes) == 20
    assert Counter(key[0] for key in shapes) == {'geglu': 10, 'groupnorm_silu': 10}
    for rows in shapes.values():
        assert len(rows) == 3 and len({row[0] for row in rows}) == 1
        assert {row[1] for row in rows} == set(DTYPE_TOLERANCES)
        assert len({row[2] for row in rows}) == 1
    configs = [candidate.to_dict()['config'] for candidate in suite.candidates]
    assert configs[0] == {'implementation': 'compiled'}
    assert {(c['block_size'], c['num_warps']) for c in configs[1:]} == {
        (b, n) for b in (64, 128, 256, 512, 1024, 2048) for n in (1, 2, 4, 8)}
    assert all(c['block_size'] in SUPPORTED_BLOCK_SIZES and c['num_warps'] in SUPPORTED_NUM_WARPS
               for c in configs[1:])


def test_group_count_or_dtype_does_not_create_a_heldout_shape(tmp_path):
    document = load_suite(SUITE).document
    added = deepcopy(next(t for t in document['tasks']
                          if t['split'] == 'train' and t['workload']['operator'] == 'groupnorm_silu'))
    added['id'] = 'attempted-test-leak'
    added['split'] = 'test'
    added['workload']['num_groups'] *= 2
    added['workload']['dtype'] = 'float32'
    added['workload']['seed'] += 1000
    document['tasks'].append(added)
    path = tmp_path / 'leak.json'
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='shape family overlaps'):
        load_suite(path)


def test_embedded_suite_and_tolerances_cannot_change_the_frozen_contract(tmp_path):
    suite = load_suite(SUITE)
    assert load_bound_suite(suite.document, suite.sha256).sha256 == suite.sha256
    document = suite.document
    document['tasks'][0]['workload']['seed'] += 1
    with pytest.raises(ValueError, match='document or SHA256'):
        load_bound_suite(document, suite.sha256)
    with pytest.raises(ValueError, match='document or SHA256'):
        load_bound_suite(suite.document, 'f' * 64)
    document = suite.document
    document['quality_tolerances']['float32']['atol'] = 1.0
    path = tmp_path / 'weakened.json'
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='tolerances differ'):
        load_suite(path)


@pytest.mark.parametrize('splits', [[], ['test', 'test'], ['unknown'], 'test'])
def test_invalid_subset_rejected_before_evidence_creation(tmp_path, splits):
    path = tmp_path / 'invalid'
    with pytest.raises(ValueError, match='Task splits'):
        tc.collect(path, suite=SUITE, task_splits=splits, repeat_index=0,
                   runner_factory=FixtureRunner)
    assert not path.exists()


@pytest.mark.parametrize('splits,expected_tasks', [(None, 60), (['test'], 18)])
def test_expanded_collection_merge_replay_and_subset_coverage(tmp_path, monkeypatch, splits, expected_tasks):
    monkeypatch.setattr(tc, 'source_hashes', lambda: {'fixture.py': 'f' * 64})
    paths = []
    for repeat in range(2):
        monkeypatch.setattr(tc, '_PROCESS_INSTANCE_ID', f'CPU-generalization-fixture-{repeat}')
        path = tmp_path / f'repeat-{repeat}'
        assert tc.collect(path, suite=SUITE, task_splits=splits, repeat_index=repeat,
                          planned_repeats=2, repetitions=3, runner_factory=FixtureRunner) == 0
        paths.append(path)
    output = tmp_path / 'table.json'
    document = tc.merge(paths, output)
    table = MeasuredTable.load(output)  # No explicit suite required: exact bundled hash resolves it.
    assert len(document['rows']) == expected_tasks * 26 * 2
    assert len(document['cells']) == expected_tasks * 26
    assert len(table.measured_tasks) == expected_tasks
    assert table.suite.sha256 == load_suite(SUITE).sha256
    failed = [cell for cell in document['cells'] if cell['candidate_id'] == 'triton-b128-w8']
    assert len(failed) == expected_tasks and all(c['status_counts'] == {'error': 2} for c in failed)
    assert all(c['median_latency_ms'] is None for c in failed)
    if splits:
        assert document['protocol']['task_splits'] == ['test']
        assert all(t.metadata['split'] == 'test' for t in table.measured_tasks)
        evaluator = MeasuredReplayEvaluator(table)
        assert evaluator.baseline(table.measured_tasks[0]).backend == 'measured_replay'
        with pytest.raises(ValueError, match='task differs'):
            evaluator.baseline(table.suite.tasks[0])
        tampered = deepcopy(document)
        del tampered['protocol']['task_splits']
        tampered['collection_protocol_sha256'] = tc._digest(tampered['protocol'])
        with pytest.raises(ValueError, match='every planned cell'):
            tc.validate_table(tampered)


def test_transfer_subset_order_is_complete_and_bound_to_test_family_split():
    suite = load_suite(SUITE)
    allowed = set(suite_task_indices(suite, ['test']))
    orders = [tc.collection_order(suite, 1901, r, ['test']) for r in range(3)]
    assert all(len(order) == 18 * 26 for order in orders)
    for order in orders:
        assert {task for task, _ in order} == allowed
        for task in allowed:
            cells = [candidate for t, candidate in order if t == task]
            assert cells[0] is None and set(cells[1:]) == set(range(25))


def test_confirmation_cli_binds_the_explicit_new_suite(tmp_path, monkeypatch):
    suite = load_suite(SUITE)
    task = suite.tasks[suite.splits['test'][0]]
    workload = Workload(**task.metadata['workload'])
    monkeypatch.setattr(tc, 'source_hashes', lambda: {'fixture.py': 'f' * 64})
    monkeypatch.setattr(confirmation, 'CudaEvaluator', FixtureRunner)
    document = {'suite_sha256': suite.sha256,
                'cohort_id': tc._digest({'gpu_uuid': 'CPU-CONTRACT-FIXTURE', 'source_hashes': tc.source_hashes()}),
                'expected_input_digests': {task.task_id: tc._digest([hashlib.sha256(workload.key().encode()).hexdigest()])},
                'selections': [{'task_id': task.task_id, 'candidate_id': 'triton-b2048-w1'}]}
    selections = tmp_path / 'selections.json'
    selections.write_text(json.dumps(document))
    output = tmp_path / 'confirmation'
    assert confirmation.main(['--suite', SUITE, '--selections', str(selections), '--output', str(output),
                              '--repeat-index', '0', '--repetitions', '3']) == 0
    plan = json.loads((output / 'plan.json').read_text())
    assert plan['suite_sha256'] == suite.sha256 and plan['suite_id'] == SUITE
    result = json.loads((output / 'result.json').read_text())
    assert result['results'][0]['candidate_id'] == 'triton-b2048-w1'
    assert len(result['results'][0]['measurements']) == 3
