"""CPU contract fixtures; these invented rows are never benchmark evidence."""
from copy import deepcopy
import hashlib
import json
import uuid

import pytest

from drivebench_tuning import table_collection as tc
from drivebench_tuning.contract import Measurement
from drivebench_tuning.suite import load_suite


class FixtureRunner:
    """Invented CUDA-shaped evidence strictly for protocol validation tests."""
    def __init__(self, journal, *, repetitions, warmups, max_seconds):
        self.journal, self.repetitions, self.warmups = journal, repetitions, warmups
        self.runtime = {'gpu_uuid': 'CPU-CONTRACT-FIXTURE', 'source_hashes': tc.source_hashes(),
                        'compiler_state_policy': dict(tc.COMPILER_STATE_POLICY),
                        'inductor_compile_threads': 1}
        self.cohort_id = tc._digest(self.runtime)
        journal.write('runtime.json', {**self.runtime, 'cohort_id': self.cohort_id})
        self.closed = False

    def measure(self, workload, config, implementation):
        trial_id = str(uuid.uuid4())
        samples = [1.0 + i / 10 for i in range(self.repetitions)]
        memory = [128] * self.repetitions
        # A supported error outcome remains in every completed table repeat.
        failed = config is not None and config.block_size == 128 and config.num_warps == 8
        evidence = {'trial_id': trial_id, 'cohort_id': self.cohort_id,
                    'implementation': implementation, 'repetitions': self.repetitions,
                    'warmups': self.warmups,
                    'input_sha256': [hashlib.sha256(workload.key().encode()).hexdigest()],
                    'fixture_only': True}
        if failed:
            evidence['error'] = {'type': 'OutOfResources', 'message': 'CPU test fixture'}
        else:
            evidence.update(cuda_event_ms=samples, host_wall_ms=samples,
                            peak_allocated_delta_bytes=memory, input_integrity_verified=True,
                            correctness={'passed': True, 'finite': True, 'shape_dtype_match': True})
        result = Measurement(workload.key(), config, 'error' if failed else 'ok',
                             None if failed else tc.statistics.median(samples),
                             None if failed else max(memory), not failed, 'cuda', evidence)
        self.journal.append({'event': 'failed' if failed else 'finished', 'measurement': result.to_dict()})
        return result

    def baseline(self, workload):
        return self.measure(workload, None, 'eager')

    def compiled_baseline(self, workload):
        return self.measure(workload, None, 'compiled')

    def evaluate(self, workload, config):
        return self.measure(workload, config, 'triton')

    def close(self):
        self.closed = True


@pytest.fixture
def repeat_directories(tmp_path, monkeypatch):
    monkeypatch.setattr(tc, 'source_hashes', lambda: {'fixture.py': 'f' * 64})
    paths = []
    for index in range(3):
        monkeypatch.setattr(tc, '_PROCESS_INSTANCE_ID', f'CPU-test-process-{index}')
        path = tmp_path / f'repeat-{index}'
        assert tc.collect(path, repeat_index=index, repetitions=3, runner_factory=FixtureRunner) == 0
        paths.append(path)
    return paths


def test_collection_full_coverage_preserves_failure_and_raw_samples(repeat_directories, tmp_path):
    table = tc.merge(repeat_directories[::-1], tmp_path / 'table.json')
    assert len(table['rows']) == 360
    assert len(table['cells']) == 120  # 108 candidates plus 12 eager references.
    assert table['repeat_indices'] == [0, 1, 2]
    failed = [c for c in table['cells'] if c['candidate_id'] == 'triton-b128-w8']
    assert len(failed) == 12
    assert all(c['status_counts'] == {'error': 3} and not c['all_repeats_correct'] for c in failed)
    assert all(c['median_latency_ms'] is None for c in failed)
    assert table['rows'][0]['measurement']['evidence']['cuda_event_ms'] == [1.0, 1.1, 1.2]
    assert tc.validate_table(json.loads((tmp_path / 'table.json').read_text())) == table
    with pytest.raises(FileExistsError):
        tc.merge(repeat_directories, tmp_path / 'table.json')


def test_task_and_candidate_order_complete_deterministic_and_rotated():
    suite = load_suite()
    first, second = (tc.collection_order(suite, 1901, i) for i in (0, 1))
    assert first == tc.collection_order(suite, 1901, 0)
    assert first != second
    for index in range(len(suite.tasks)):
        a = [c for t, c in first if t == index]
        b = [c for t, c in second if t == index]
        assert a[0] is None and b[0] is None
        assert set(a[1:]) == set(range(9))
        assert b[1:] == a[4:] + a[1:4]


@pytest.mark.parametrize('mutation', [
    lambda t: t['rows'].pop(),
    lambda t: t['rows'][1]['measurement'].__setitem__('backend', 'synthetic'),
    lambda t: t['rows'][1]['measurement'].__setitem__('cohort_id', 'different-runtime'),
    lambda t: t['rows'][1]['measurement']['evidence'].__setitem__('input_sha256', ['0' * 64]),
    lambda t: t['rows'][0]['measurement']['evidence']['cuda_event_ms'].__setitem__(0, 9.0),
    lambda t: t['rows'][0]['measurement']['evidence'].__setitem__('input_integrity_verified', False),
    lambda t: t['rows'][0]['measurement']['evidence']['correctness'].__setitem__('passed', False),
    lambda t: t['runtime'].__setitem__('gpu_uuid', 'another-device'),
    lambda t: t['cells'][0].__setitem__('median_latency_ms', 999.0),
    lambda t: t.__setitem__('complete', False),
    lambda t: t['runs'][1].__setitem__('process_instance_id', t['runs'][0]['process_instance_id']),
])
def test_reject_incomplete_mixed_or_tampered_table(repeat_directories, tmp_path, mutation):
    table = tc.merge(repeat_directories, tmp_path / 'original.json')
    mutation(table)
    with pytest.raises((ValueError, KeyError)):
        tc.validate_table(table)


def test_reject_artifact_change_partial_repeat_and_repeat_reuse(repeat_directories, tmp_path):
    with pytest.raises(ValueError, match='repeat|Repeats'):
        tc.merge([repeat_directories[0]] * 3, tmp_path / 'reuse.json')
    with pytest.raises(ValueError, match='repeat|Repeats'):
        tc.merge(repeat_directories[:2], tmp_path / 'partial.json')
    (repeat_directories[0] / 'rows.json').write_text('[]\n')
    with pytest.raises(ValueError, match='digest'):
        tc.merge(repeat_directories, tmp_path / 'tampered.json')
    assert not (tmp_path / 'tampered.json').exists()


def test_unexpected_execution_error_keeps_partial_rows_and_no_completion(tmp_path, monkeypatch):
    class Broken(FixtureRunner):
        def evaluate(self, workload, config):
            raise RuntimeError('invalid CUDA context fixture')
    monkeypatch.setattr(tc, 'source_hashes', lambda: {'fixture.py': 'f' * 64})
    path = tmp_path / 'failed'
    assert tc.collect(path, repeat_index=0, repetitions=3, runner_factory=Broken) == 1
    assert (path / 'failed.json').exists()
    assert not (path / 'completed.json').exists()
    assert 1 <= len(json.loads((path / 'rows.json').read_text())) <= 2
    with pytest.raises(ValueError, match='partial'):
        tc.merge([path], tmp_path / 'table.json')


def test_cleanup_failure_cannot_leave_completed_record(tmp_path, monkeypatch):
    class BadClose(FixtureRunner):
        def close(self):
            raise RuntimeError('fixture cleanup failure')
    monkeypatch.setattr(tc, 'source_hashes', lambda: {'fixture.py': 'f' * 64})
    path = tmp_path / 'bad-close'
    assert tc.collect(path, repeat_index=0, repetitions=3, runner_factory=BadClose) == 1
    assert not (path / 'completed.json').exists()
    assert (path / 'cleanup-failed.json').exists()
    assert len(json.loads((path / 'rows.json').read_text())) == 120


def test_arguments_validated_before_creating_evidence(tmp_path):
    for kwargs in ({'repeat_index': -1}, {'repeat_index': 3}, {'repeat_index': 0, 'planned_repeats': 1},
                   {'repeat_index': 0, 'max_seconds': float('inf')}, {'repeat_index': 0, 'seed': True}):
        with pytest.raises(ValueError):
            tc.collect(tmp_path / 'invalid', runner_factory=FixtureRunner, **kwargs)
    assert not (tmp_path / 'invalid').exists()


def _rebind_fixture_table(table):
    """Update invented identities so policy checks, not stale hashes, are tested."""
    runtime = {k: v for k, v in table['runtime'].items() if k != 'cohort_id'}
    cohort = tc._digest(runtime)
    table['runtime']['cohort_id'] = table['cohort_id'] = cohort
    table['collection_protocol_sha256'] = tc._digest(table['protocol'])
    for run in table['runs']:
        run['cohort_id'] = cohort
        run['collection_protocol_sha256'] = table['collection_protocol_sha256']
    for row in table['rows']:
        row['measurement']['cohort_id'] = cohort
        row['measurement']['evidence']['cohort_id'] = cohort


def test_compiler_policy_is_bound_and_mismatch_rejected(repeat_directories, tmp_path):
    table = tc.merge(repeat_directories, tmp_path / 'original.json')
    assert table['protocol']['compiler_state_policy'] == tc.COMPILER_STATE_POLICY
    assert table['runtime']['compiler_state_policy'] == tc.COMPILER_STATE_POLICY
    for location in ('runtime', 'protocol'):
        invalid = deepcopy(table)
        del invalid[location]['compiler_state_policy']
        _rebind_fixture_table(invalid)
        with pytest.raises(ValueError, match='Compiler-state policy'):
            tc.validate_table(invalid)
    invalid = deepcopy(table)
    for location in ('runtime', 'protocol'):
        invalid[location]['compiler_state_policy']['sample_timing_includes_reset'] = True
    _rebind_fixture_table(invalid)
    with pytest.raises(ValueError, match='Compiler-state policy'):
        tc.validate_table(invalid)


@pytest.mark.parametrize('actual', [None, 2, True, 1.0])
def test_runtime_compile_threads_are_checked_independently(repeat_directories, tmp_path, actual):
    table = tc.merge(repeat_directories, tmp_path / 'original.json')
    assert table['protocol']['compiler_state_policy']['inductor_compile_threads'] == 1
    if actual is None:
        table['runtime'].pop('inductor_compile_threads')
    else:
        table['runtime']['inductor_compile_threads'] = actual
    _rebind_fixture_table(table)
    with pytest.raises(ValueError, match='Inductor compile thread count'):
        tc.validate_table(table)


def test_policy_count_bool_cannot_alias_integer_one(repeat_directories, tmp_path):
    table = tc.merge(repeat_directories, tmp_path / 'original.json')
    for location in ('runtime', 'protocol'):
        table[location]['compiler_state_policy']['inductor_compile_threads'] = True
    _rebind_fixture_table(table)
    with pytest.raises(ValueError, match='Compiler-state policy'):
        tc.validate_table(table)


def test_historical_policy_absence_does_not_admit_current_source(repeat_directories, tmp_path, monkeypatch):
    table = tc.merge(repeat_directories, tmp_path / 'original.json')
    for location in ('runtime', 'protocol'):
        del table[location]['compiler_state_policy']
    _rebind_fixture_table(table)
    assert tc.validate_table(table) == table  # Legacy source fixture remains replayable.
    current = {'benchmark.py': 'b' * 64}
    monkeypatch.setattr(tc, 'source_hashes', lambda: current)
    for location in ('runtime', 'protocol'):
        table[location]['source_hashes'] = dict(current)
    _rebind_fixture_table(table)
    with pytest.raises(ValueError, match='Compiler-state policy missing'):
        tc.validate_table(table)


def test_collection_checks_compiler_policy_before_any_measurement(tmp_path):
    class WrongPolicy(FixtureRunner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.runtime.pop('compiler_state_policy')

        def measure(self, *args, **kwargs):
            raise AssertionError('No measurement may run after a policy mismatch')
    directory = tmp_path / 'bad-compiler-policy'
    assert tc.collect(directory, repeat_index=0, repetitions=3, runner_factory=WrongPolicy) == 1
    failure = json.loads((directory / 'failed.json').read_text())
    assert failure['error_type'] == 'ValueError' and failure['measurement_count'] == 0
    assert 'compiler-state policy' in failure['error']
    assert not (directory / 'completed.json').exists()


def test_collection_checks_effective_thread_count_before_measurement(tmp_path):
    class WrongThreads(FixtureRunner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.runtime['inductor_compile_threads'] = 2

        def measure(self, *args, **kwargs):
            raise AssertionError('No measurement may run with the wrong compiler thread count')
    directory = tmp_path / 'bad-compiler-threads'
    assert tc.collect(directory, repeat_index=0, repetitions=3, runner_factory=WrongThreads) == 1
    failure = json.loads((directory / 'failed.json').read_text())
    assert failure['error_type'] == 'ValueError' and failure['measurement_count'] == 0
    assert 'Inductor compile thread count' in failure['error']
    assert not (directory / 'completed.json').exists()
