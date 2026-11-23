"""CPU-only publication contracts; all timing values below are invented fixtures."""
import hashlib
import importlib.util
import json
from pathlib import Path
import uuid

import pytest

from drivebench_tuning import table_collection as tc
from drivebench_tuning.contract import Measurement

SCRIPT = Path(__file__).resolve().parents[3] / 'scripts/export_operator_table.py'
spec = importlib.util.spec_from_file_location('operator_public_export', SCRIPT)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)
GPU_ID = '12345678-1234-5678-9abc-123456789abc'
SECOND_GPU_ID = '98765432-1234-5678-9abc-123456789abc'
PRIVATE_ERROR = 'allocation failed at /workspace/private/account-xyz, host 10.1.2.3, provider pod-123456'


class FixtureRunner:
    def __init__(self, journal, *, repetitions, warmups, max_seconds):
        self.journal, self.repetitions, self.warmups = journal, repetitions, warmups
        self.runtime = {'gpu_uuid': GPU_ID, 'source_hashes': tc.source_hashes(),
                        'compiler_state_policy': dict(tc.COMPILER_STATE_POLICY),
                        'inductor_compile_threads': 1,
                        'gpu_name': 'CPU fixture only', 'torch': 'fixture',
                        'driver_inventory': ['GPU-' + GPU_ID + ', 580.178.04',
                                             'GPU-' + SECOND_GPU_ID + ', 580.178.04']}
        self.cohort_id = tc._digest(self.runtime)
        journal.write('runtime.json', {**self.runtime, 'cohort_id': self.cohort_id})

    def measure(self, workload, config, implementation):
        failed = config is not None and config.block_size == 128 and config.num_warps == 8
        samples = [1.0, 1.1, 1.2]
        evidence = {'trial_id': str(uuid.uuid4()), 'cohort_id': self.cohort_id,
                    'implementation': implementation, 'repetitions': self.repetitions,
                    'warmups': self.warmups, 'fixture_only': True,
                    'input_sha256': [hashlib.sha256(workload.key().encode()).hexdigest()]}
        if failed:
            evidence['error'] = {'type': 'OutOfResources', 'message': PRIVATE_ERROR}
        else:
            evidence.update(cuda_event_ms=samples, host_wall_ms=samples,
                            peak_allocated_delta_bytes=[128] * 3, input_integrity_verified=True,
                            correctness={'passed': True, 'finite': True, 'shape_dtype_match': True})
        m = Measurement(workload.key(), config, 'error' if failed else 'ok', None if failed else 1.1,
                        None if failed else 128, not failed, 'cuda', evidence)
        self.journal.append({'event': 'failed' if failed else 'finished', 'measurement': m.to_dict()})
        return m

    def baseline(self, workload): return self.measure(workload, None, 'eager')
    def compiled_baseline(self, workload): return self.measure(workload, None, 'compiled')
    def evaluate(self, workload, config): return self.measure(workload, config, 'triton')
    def close(self): pass


@pytest.fixture
def raw(tmp_path, monkeypatch):
    monkeypatch.setattr(tc, 'source_hashes', lambda: {'cpu-fixture.py': 'f' * 64})
    paths = []
    for index in range(2):
        monkeypatch.setattr(tc, '_PROCESS_INSTANCE_ID', f'cpu-test-process-{index}')
        path = tmp_path / f'raw-repeat-{index}'
        assert tc.collect(path, repeat_index=index, planned_repeats=2, repetitions=3,
                          runner_factory=FixtureRunner) == 0
        paths.append(path)
    table = tmp_path / 'raw-table.json'
    tc.merge(paths, table)
    return table, paths


def test_raw_and_derived_tables_validate_with_identical_measurements(raw, tmp_path):
    table, directories = raw
    before = {p: p.read_bytes() for p in [table] + [p for d in directories for p in d.iterdir()]}
    source = json.loads(table.read_text())
    tc.validate_table(source)
    output = tmp_path / 'public'
    receipt = exporter.export(table, directories, output)
    derived = json.loads((output / 'table.json').read_text())
    tc.validate_table(derived)
    assert derived['public_export']['measurements_rerun'] is False
    assert derived['public_export']['measurement_values_unchanged'] is True
    assert receipt['raw_artifact_hashes_unchanged'] is True
    assert receipt['pseudonymized_gpu_count'] == 2
    assert source['cohort_id'] != derived['cohort_id']
    assert derived['runtime']['gpu_uuid'] == exporter.gpu_pseudonym('GPU-' + GPU_ID)
    assert derived['runtime']['driver_inventory'][0] == derived['runtime']['gpu_uuid'] + ', 580.178.04'
    assert len(derived['rows']) == len(source['rows']) == 240
    assert derived['cells'] == source['cells']
    assert {p: p.read_bytes() for p in before} == before
    for entry in receipt['artifacts']:
        assert exporter.sha(output / entry['artifact']) == entry['derived_sha256']
    text = '\n'.join(p.read_text() for p in output.rglob('*') if p.is_file())
    for private in [GPU_ID, SECOND_GPU_ID, '/workspace/private', '10.1.2.3', 'pod-123456']:
        assert private not in text
    errors = [r['measurement']['evidence']['error'] for r in derived['rows']
              if r['measurement']['status'] == 'error']
    assert len(errors) == 24
    assert all(e['type'] == 'OutOfResources' and e['message_redacted'] is True for e in errors)
    assert any(count > 0 for count in receipt['error_redactions_by_artifact'].values())
    with pytest.raises(ValueError, match='new output'):
        exporter.export(table, directories, output)


def test_same_physical_gpu_pseudonym_across_representations():
    assert exporter.gpu_pseudonym(GPU_ID) == exporter.gpu_pseudonym('GPU-' + GPU_ID.upper())
    assert exporter.gpu_pseudonym(GPU_ID) == exporter.gpu_pseudonym(GPU_ID.replace('-', ''))
    assert exporter.gpu_pseudonym(GPU_ID) != exporter.gpu_pseudonym(SECOND_GPU_ID)


def test_refuses_guessed_hardware_redaction():
    with pytest.raises(ValueError, match='Unrecognized physical'):
        exporter.gpu_pseudonym('unrecognized-private-identifier')
    with pytest.raises(ValueError, match='driver inventory'):
        exporter.public_runtime({'gpu_uuid': GPU_ID, 'cohort_id': 'unused',
                                 'driver_inventory': ['GPU-' + GPU_ID + ', version/path/private']})


def test_tampered_raw_evidence_rejected_before_export(raw, tmp_path):
    table, directories = raw
    p = directories[0] / 'rows.json'
    p.write_text('[]\n')
    with pytest.raises(ValueError, match='digest'):
        exporter.export(table, directories, tmp_path / 'refused')
    assert not (tmp_path / 'refused').exists()


def test_export_cannot_add_files_inside_raw_repeat(raw):
    table, directories = raw
    with pytest.raises(ValueError, match='outside the raw'):
        exporter.export(table, directories, directories[0] / 'public')
    assert not (directories[0] / 'public').exists()
