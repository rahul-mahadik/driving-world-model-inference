"""Trusted CLI/manifest integration using actual core CPU synthetic inference."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest

from drivebench.data import write_manifest
from drivebench.data.synthetic import synthetic_records
from drivebench_tuning import vista_batch, vista_agent


def fixture(tmp_path):
    data = tmp_path / 'data'
    records = synthetic_records(data, validation_scenes=2, training_scenes=0,
                                clips_per_scene=1, frame_count=3, height=8, width=8)
    manifest = tmp_path / 'manifest.jsonl'; write_manifest(manifest, records)
    requests = []
    for index, record in enumerate(records):
        for seed, rep in [(11, 0), (23, 1)]:
            requests.append({'record_id': f'new-{index}-{seed}-{rep}', 'request_id': record['request_id'],
                'scene_id': record['scene_id'], 'seed': seed, 'replicate': rep,
                'action_sha256': vista_batch.action_sha256(record)})
    spec = {'schema_version': 1, 'cohort_id': 'cpu-manifest-contract', 'batch_id': 'reference-0',
        'synthetic': True, 'manifest': {'path': str(manifest), 'sha256': vista_agent.file_hash(manifest),
            'metadata_sha256': vista_agent.file_hash(manifest.with_suffix('.jsonl.meta.json'))},
        'data_root': str(data), 'settings': vista_agent.reference_settings(synthetic=True),
        'model': {'backend': 'synthetic'}, 'expected_runtime': vista_agent.probe_runtime(synthetic=True),
        'source_files': {str(p): vista_agent.file_hash(p) for p in vista_batch.required_source_paths()},
        'requests': requests, 'output': str(tmp_path / 'result'),
        'created_at': datetime.now(timezone.utc).isoformat(), 'deadline_unix': time.time() + 60,
        'warmup_requests': 1, 'minimum_free_bytes': 0}
    return spec, records


def save_spec(tmp_path, spec):
    path = tmp_path / 'spec.json'; path.write_text(json.dumps(spec, allow_nan=False))
    return path, vista_agent.file_hash(path)


def invoke(tmp_path, spec):
    path, sha = save_spec(tmp_path, spec)
    return vista_batch.run_spec(path, expected_sha256=sha)


def rebind_manifest(spec, records):
    path = Path(spec['manifest']['path']); write_manifest(path, records)
    spec['manifest']['sha256'] = vista_agent.file_hash(path)
    spec['manifest']['metadata_sha256'] = vista_agent.file_hash(path.with_suffix('.jsonl.meta.json'))


def test_actual_synthetic_engine_from_hashed_manifest_and_no_future_access(tmp_path):
    spec, records = fixture(tmp_path)
    original = Path(spec['manifest']['path']).read_bytes()
    for record in records:
        for future in record['future_frames']:
            (Path(spec['data_root']) / future).unlink()
    receipt = invoke(tmp_path, spec)
    assert receipt['status'] == 'ok' and receipt['synthetic']
    assert receipt['measured_outputs'] == 4 and receipt['warmup_outputs'] == 1
    assert not receipt['quality_assessed'] and not receipt['quality_preserving_claim']
    output = Path(spec['output'])
    prepared = json.loads((output / 'prepared-inputs.json').read_text())
    assert not prepared['future_images_decoded']
    assert [r['seed'] for r in prepared['records']] == [11, 23, 11, 23]
    assert prepared['records'][0]['conditioning_sha256'] == prepared['records'][1]['conditioning_sha256']
    assert prepared['records'][0]['input_sha256'] != prepared['records'][1]['input_sha256']
    completed = json.loads((output / 'batch' / 'completed.json').read_text())
    for row in completed['rows'] + completed['warmups']:
        array = output / 'batch' / row['output_file']
        assert vista_agent.file_hash(array) == row['output_sha256']
        assert np.load(array, allow_pickle=False).shape == (3, 8, 8, 3)
    assert Path(spec['manifest']['path']).read_bytes() == original
    assert (output / 'spec.json').read_bytes() == (tmp_path / 'spec.json').read_bytes()
    with pytest.raises(FileExistsError): invoke(tmp_path, spec)


def test_cli_in_fresh_process_outside_checkout_with_explicit_hash(tmp_path):
    spec, _ = fixture(tmp_path); path, sha = save_spec(tmp_path, spec)
    outside = tmp_path / 'outside'; outside.mkdir()
    import drivebench
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(Path(vista_batch.__file__).resolve().parents[1]),
                                       str(Path(drivebench.__file__).resolve().parent.parent)])
    result = subprocess.run([sys.executable, '-m', 'drivebench_tuning.vista_batch', '--spec', str(path),
        '--spec-sha256', sha], cwd=outside, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)['status'] == 'ok'


@pytest.mark.parametrize('where', ['top', 'manifest', 'request', 'settings', 'model'])
def test_unknown_fields_fail_before_output_or_engine(tmp_path, where):
    spec, _ = fixture(tmp_path)
    target = {'top': spec, 'manifest': spec['manifest'], 'request': spec['requests'][0],
              'settings': spec['settings'], 'model': spec['model']}[where]
    target['command'] = 'must never execute'
    with pytest.raises(ValueError): invoke(tmp_path, spec)
    assert not Path(spec['output']).exists()


@pytest.mark.parametrize('case', ['wrong_sha', 'duplicate_json_key', 'nonfinite_json'])
def test_spec_integrity_and_json_ambiguity_fail_closed(tmp_path, case):
    spec, _ = fixture(tmp_path); path, sha = save_spec(tmp_path, spec)
    if case == 'wrong_sha': sha = '0' * 64
    elif case == 'duplicate_json_key':
        path.write_text(path.read_text().replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1'))
        sha = vista_agent.file_hash(path)
    else:
        path.write_text(path.read_text().replace('"minimum_free_bytes": 0', '"minimum_free_bytes": NaN'))
        sha = vista_agent.file_hash(path)
    with pytest.raises(ValueError): vista_batch.run_spec(path, expected_sha256=sha)
    assert not Path(spec['output']).exists()


@pytest.mark.parametrize('case', ['duplicate_id', 'duplicate_cell', 'missing_request', 'wrong_scene', 'wrong_action', 'bool_seed'])
def test_selection_identity_rejects_invalid_cells(tmp_path, case):
    spec, _ = fixture(tmp_path)
    if case == 'duplicate_id': spec['requests'][1]['record_id'] = spec['requests'][0]['record_id']
    elif case == 'duplicate_cell':
        spec['requests'][1]['seed'] = 11; spec['requests'][1]['replicate'] = 0
    elif case == 'missing_request': spec['requests'][0]['request_id'] = 'missing'
    elif case == 'wrong_scene': spec['requests'][0]['scene_id'] = 'wrong'
    elif case == 'wrong_action': spec['requests'][0]['action_sha256'] = '0' * 64
    else: spec['requests'][0]['seed'] = True
    with pytest.raises(ValueError): invoke(tmp_path, spec)
    assert not (Path(spec['output']) / 'batch').exists()


@pytest.mark.parametrize('case', ['manifest', 'metadata', 'image', 'source', 'source_coverage', 'runtime'])
def test_bound_content_drift_never_starts_inference(tmp_path, case):
    spec, records = fixture(tmp_path)
    if case == 'manifest': Path(spec['manifest']['path']).write_text('{}\n')
    elif case == 'metadata': Path(spec['manifest']['path'] + '.meta.json').write_text('{}')
    elif case == 'image': (Path(spec['data_root']) / records[0]['observed_frames'][0]).write_bytes(b'changed')
    elif case == 'source': spec['source_files'][str(Path(vista_batch.__file__).resolve())] = '0' * 64
    elif case == 'source_coverage': spec['source_files'].pop(str(Path(vista_batch.__file__).resolve()))
    else: spec['expected_runtime']['unknown_flag'] = True
    with pytest.raises(ValueError): invoke(tmp_path, spec)
    assert not (Path(spec['output']) / 'batch').exists()


def test_image_mutation_during_preprocessing_is_detected(tmp_path, monkeypatch):
    spec, records = fixture(tmp_path)
    import drivebench.data
    real = drivebench.data.read_preprocess
    def corrupt(paths, *args):
        output = real(paths, *args); Path(paths[0]).write_bytes(b'changed after decode'); return output
    monkeypatch.setattr(drivebench.data, 'read_preprocess', corrupt)
    with pytest.raises(ValueError, match='checksum'): invoke(tmp_path, spec)
    assert (Path(spec['output']) / 'adapter-failure.json').is_file()
    assert not (Path(spec['output']) / 'batch').exists()


def test_frame_path_escape_is_rejected_by_core_resolver(tmp_path):
    spec, records = fixture(tmp_path)
    records[0]['observed_frames'] = ['../escape.png']
    records[0]['frame_checksums']['../escape.png'] = '0' * 64
    rebind_manifest(spec, records)
    with pytest.raises(ValueError, match='escapes'): invoke(tmp_path, spec)


@pytest.mark.parametrize('case', ['changed_action', 'synthetic_flag', 'frame_count'])
def test_manifest_contract_changes_cannot_be_hidden_by_rebinding_file_hash(tmp_path, case):
    spec, records = fixture(tmp_path)
    if case == 'changed_action': records[0]['trajectory'][0] += 1
    elif case == 'synthetic_flag': records[0]['synthetic'] = False
    else: records[0]['conditioning_frame_count'] = 2
    rebind_manifest(spec, records)
    with pytest.raises(ValueError): invoke(tmp_path, spec)
    assert not (Path(spec['output']) / 'batch').exists()


def test_expired_deadline_during_preparation_never_reaches_worker(tmp_path, monkeypatch):
    spec, _ = fixture(tmp_path)
    import drivebench.data
    real = drivebench.data.read_preprocess
    def expired(*args):
        output = real(*args); monkeypatch.setattr(vista_batch.time, 'time', lambda: spec['deadline_unix'] + 1); return output
    monkeypatch.setattr(drivebench.data, 'read_preprocess', expired)
    with pytest.raises(TimeoutError): invoke(tmp_path, spec)
    assert not (Path(spec['output']) / 'batch').exists()
    assert (Path(spec['output']) / 'adapter-failure.json').is_file()


def test_worker_failure_preserves_raw_evidence_and_adapter_failure(tmp_path, monkeypatch):
    spec, _ = fixture(tmp_path)
    from drivebench.engine import InferenceEngine
    real = InferenceEngine.generate
    def bad(self, *args, **kwargs):
        result = real(self, *args, **kwargs); result.denoiser_evaluations = 0; return result
    monkeypatch.setattr(InferenceEngine, 'generate', bad)
    with pytest.raises(ValueError, match='invariant'): invoke(tmp_path, spec)
    output = Path(spec['output'])
    assert (output / 'adapter-failure.json').is_file() and (output / 'batch' / 'failure.json').is_file()
    assert (output / 'batch' / '00000.npy').is_file()
    assert not (output / 'adapter-completed.json').exists()


def test_manifest_without_sidecar_requires_explicit_absence_binding(tmp_path):
    spec, _ = fixture(tmp_path)
    Path(spec['manifest']['path'] + '.meta.json').unlink()
    spec['manifest']['metadata_sha256'] = None
    assert invoke(tmp_path, spec)['status'] == 'ok'


@pytest.mark.parametrize('case', ['duplicate_json_key', 'overflowed_number'])
def test_bound_manifest_ambiguous_or_nonfinite_json_fails_before_engine(tmp_path, case):
    spec, records = fixture(tmp_path); path = Path(spec['manifest']['path'])
    content = path.read_text()
    if case == 'duplicate_json_key':
        content = content.replace('"schema_version":1', '"schema_version":1,"schema_version":1', 1)
    else:
        # Even unused metadata must remain valid finite JSON.
        content = content.replace('"schema_version":1', '"unused_metadata":1e999,"schema_version":1', 1)
    assert content != path.read_text(); path.write_text(content)
    Path(str(path) + '.meta.json').unlink()
    spec['manifest']['sha256'] = vista_agent.file_hash(path); spec['manifest']['metadata_sha256'] = None
    with pytest.raises(ValueError): invoke(tmp_path, spec)
    assert not (Path(spec['output']) / 'batch').exists()


@pytest.mark.parametrize('case', ['relative_spec', 'spec_symlink', 'relative_output', 'relative_data_root', 'expired_deadline'])
def test_path_and_deadline_contracts_fail_without_a_new_attempt(tmp_path, case, monkeypatch):
    spec, _ = fixture(tmp_path)
    if case == 'relative_output': spec['output'] = 'relative-output'
    elif case == 'relative_data_root': spec['data_root'] = 'relative-data'
    elif case == 'expired_deadline': spec['deadline_unix'] = time.time() - 1
    path, sha = save_spec(tmp_path, spec)
    if case == 'spec_symlink':
        link = tmp_path / 'link.json'; link.symlink_to(path); path = link
    elif case == 'relative_spec':
        monkeypatch.chdir(tmp_path); path = Path('spec.json')
    with pytest.raises(ValueError): vista_batch.run_spec(path, expected_sha256=sha)
    assert not (tmp_path / 'result').exists()


def real_spec_fixture(tmp_path):
    """Preflight-only dummy files; never create a real inference engine."""
    from drivebench.models.upstream import VISTA_COMMIT
    spec, _ = fixture(tmp_path)
    upstream = tmp_path / 'upstream'; config = upstream / 'configs/inference/vista.yaml'
    config.parent.mkdir(parents=True); config.write_text('model: frozen\n')
    (upstream / 'sample_utils.py').write_text('# frozen source fixture\n')
    (upstream / 'configs/shared.yaml').write_text('shared: frozen\n')
    weights = tmp_path / 'fixture.safetensors'; weights.write_bytes(b'not actual model weights')
    spec.update(synthetic=False, settings=vista_agent.reference_settings(), model={
        'checkpoint': str(weights), 'checkpoint_sha256': vista_agent.file_hash(weights),
        'upstream_dir': str(upstream), 'upstream_commit': VISTA_COMMIT, 'inference_config': str(config)})
    return spec


@pytest.mark.parametrize('missing', ['inference_config', 'upstream_python', 'upstream_yaml'])
def test_real_model_requires_all_upstream_python_and_yaml_pins_before_gpu(tmp_path, missing):
    spec = real_spec_fixture(tmp_path)
    sources = {str(p): vista_agent.file_hash(p) for p in vista_batch.required_source_paths(spec['model'])}
    filenames = {'inference_config': spec['model']['inference_config'],
                 'upstream_python': str(Path(spec['model']['upstream_dir']) / 'sample_utils.py'),
                 'upstream_yaml': str(Path(spec['model']['upstream_dir']) / 'configs/shared.yaml')}
    sources.pop(filenames[missing]); spec['source_files'] = sources
    with pytest.raises(ValueError, match='source_files must cover'):
        vista_batch._validate_spec(spec)
    assert not Path(spec['output']).exists()


def test_real_yaml_content_drift_is_rejected_before_gpu(tmp_path):
    spec = real_spec_fixture(tmp_path)
    spec['source_files'] = {str(p): vista_agent.file_hash(p) for p in vista_batch.required_source_paths(spec['model'])}
    Path(spec['model']['inference_config']).write_text('model: changed\n')
    with pytest.raises(ValueError, match='source drift'): vista_batch._validate_spec(spec)


def test_real_checkout_commit_and_untracked_source_are_independently_rejected(tmp_path, monkeypatch):
    spec = real_spec_fixture(tmp_path)
    spec['source_files'] = {str(p): vista_agent.file_hash(p) for p in vista_batch.required_source_paths(spec['model'])}
    original = spec['model']['upstream_commit']; spec['model']['upstream_commit'] = '0' * 40
    with pytest.raises(ValueError, match='audited only'): vista_batch._validate_spec(spec)
    spec['model']['upstream_commit'] = original
    from drivebench.models import upstream
    monkeypatch.setattr(upstream.subprocess, 'check_output',
                        lambda args, **kwargs: original if 'rev-parse' in args else '?? injected.py')
    with pytest.raises(ValueError, match='pristine'): vista_batch._validate_spec(spec)


def test_real_preflight_uses_core_pinned_checkout_verification_without_gpu(tmp_path, monkeypatch):
    spec = real_spec_fixture(tmp_path)
    spec['source_files'] = {str(p): vista_agent.file_hash(p) for p in vista_batch.required_source_paths(spec['model'])}
    from drivebench.models import upstream
    calls = []
    def verified(path, expected):
        calls.append((path, expected)); return {'upstream_path': path, 'upstream_commit': expected}
    monkeypatch.setattr(upstream, 'verify_checkout', verified)
    assert vista_batch._validate_spec(spec) is spec
    assert calls == [(spec['model']['upstream_dir'], upstream.VISTA_COMMIT)]
