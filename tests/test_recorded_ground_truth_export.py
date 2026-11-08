"""CPU synthetic fixtures verify exact preprocessing/alignment and durable resume."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from drivebench.config import digest, file_sha256
from drivebench.data.manifests import write_manifest
from drivebench.evaluate import _ground_truth
from drivebench.evaluation.metrics import compare_videos
from scripts import export_recorded_ground_truth as exporter


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def fixture(tmp_path, *, role='development', count=32, extra_action=False):
    dataset = tmp_path / 'dataset'; dataset.mkdir()
    for i in range(2):
        # Deliberately nonmatching aspect ratio exercises existing center crop + Lanczos.
        image = (np.arange(7 * 11 * 3).reshape(7, 11, 3) + 41 * i).astype(np.uint8)
        Image.fromarray(image).save(dataset / f'future{i}.png')
    future = [f'future{i}.png' for i in range(2)]
    checks = {p: file_sha256(dataset / p) for p in future}
    records = [{'request_id': f'r{i}', 'scene_id': f's{i}', 'future_frames': future,
                'frame_checksums': checks, 'seeds': [100 + i]} for i in range(count)]
    manifest = tmp_path / 'manifest.jsonl'
    write_manifest(manifest, records, {'partition_role': role, 'synthetic': False})
    run = tmp_path / 'upstream'; run.mkdir()
    cfg = {'model': {'backend': 'upstream'}, 'data': {'manifest': str(manifest), 'root': str(dataset),
           'manifest_sha256': file_sha256(manifest), 'verify_checksums': True},
           'generation': {'total_frames': 3, 'conditioning_frames': 1, 'height': 4, 'width': 6},
           'benchmark': {'save_outputs': count}}
    put(run / 'config.json', cfg); put(run / 'environment.json', {'fixture': True})
    put(run / 'summary.json', {'backend': 'upstream', 'status': 'ok', 'failures': 0})
    rows, expected = [], []
    for i in range(count):
        candidate = {'candidate_id': f'r{i}:c00', 'seed': 100 + i, 'action_kind': 'recorded',
                     'conditioning_frames': 1, 'frames_path': f'future-{i}.npy', 'trajectory_sha256': digest([i]),
                     'ground_truth_paths': [str(dataset / p) for p in future]}
        gt = _ground_truth(candidate, (2, 4, 6, 3))
        generated = np.clip(gt + np.float32(.01 * (i % 3)), 0, 1)
        video = np.concatenate([np.ones((1, 4, 6, 3), dtype=np.float32), generated])
        np.save(run / candidate['frames_path'], video)
        candidates = [candidate]
        if extra_action:
            other = {**candidate, 'candidate_id': f'r{i}:c01', 'action_kind': 'perturbed', 'frames_path': f'alt-{i}.npy'}
            other.pop('ground_truth_paths'); np.save(run / other['frames_path'], video); candidates.append(other)
        rows.append({'request_id': f'r{i}', 'scene_id': f's{i}', 'status': 'ok', 'repeat': 0, 'request_index': i,
                     'candidates': candidates})
        expected.append(compare_videos(gt, generated))
    (run / 'measurements.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
    return run, manifest, expected


def test_exact_existing_preprocessing_and_future_only_alignment(tmp_path):
    run, manifest, expected = fixture(tmp_path)
    output = tmp_path / 'proxies.json'
    result = exporter.export_run(run, output, expected_manifest=manifest)
    document = exporter.read(output)
    assert result['pair_count'] == 32 and result['raw_arrays_available']
    assert document['lpips_computed'] is False and document['quality_gate_changed'] is False
    assert [p['metrics'] for p in document['pairs']] == expected
    assert document['pairs'][0]['metrics']['exact_match'] is True  # Conditioning frame of ones was excluded.
    receipt = exporter.read(output.with_suffix('.receipt.json'))
    assert len(receipt['raw_array_sha256']) == 32
    assert all(str(run / name) in receipt['bound_file_sha256'] for name in
               ('config.json', 'environment.json', 'summary.json', 'measurements.jsonl'))
    original = output.read_bytes()
    assert exporter.export_run(run, output) == result
    assert output.read_bytes() == original


def test_committed_metrics_resume_after_raw_retirement_with_explicit_limitation(tmp_path):
    run, _, _ = fixture(tmp_path)
    output = tmp_path / 'proxies.json'; exporter.export_run(run, output)
    for path in run.glob('*.npy'): path.unlink()
    result = exporter.export_run(run, output)
    assert result['verified_committed_metrics'] is True
    assert len(result['missing_raw_arrays']) == 32 and result['raw_arrays_available'] is False
    with pytest.raises(FileNotFoundError): exporter.verify_export(output, allow_retired_arrays=False)
    output.with_suffix('.receipt.json').unlink()
    with pytest.raises(FileNotFoundError): exporter.export_run(run, output)


@pytest.mark.parametrize('role', sorted(exporter.ROLES - {'development'}))
def test_exact_d8_heldout_action_and_shard_roles(role, tmp_path):
    run, _, _ = fixture(tmp_path, role=role, count=2, extra_action=True)
    output = tmp_path / 'proxies.json'; exporter.export_run(run, output)
    result = exporter.read(output)
    assert result['pair_count'] == 2 and result['excluded_nonrecorded_candidates'] == 2


@pytest.mark.parametrize('mutation', ['image', 'array', 'alignment', 'seed', 'scene', 'missing', 'unsaved', 'role'])
def test_fresh_export_rejects_invalid_input_without_metric_commit(tmp_path, mutation):
    run, manifest, _ = fixture(tmp_path)
    rows = [json.loads(line) for line in (run / 'measurements.jsonl').read_text().splitlines()]
    if mutation == 'image': Image.new('RGB', (7, 11)).save(tmp_path / 'dataset/future0.png')
    if mutation == 'array': np.save(run / 'future-0.npy', np.zeros((2, 4, 6, 3), dtype=np.float32))
    if mutation == 'alignment': rows[0]['candidates'][0]['ground_truth_paths'].reverse()
    if mutation == 'seed': rows[0]['candidates'][0]['seed'] += 1
    if mutation == 'scene': rows[0]['scene_id'] = 'other'
    if mutation == 'missing': (run / 'future-0.npy').unlink()
    if mutation == 'unsaved':
        cfg = exporter.read(run / 'config.json'); cfg['benchmark']['save_outputs'] = 1; put(run / 'config.json', cfg)
    if mutation == 'role':
        meta = exporter.read(manifest.with_suffix('.jsonl.meta.json')); meta['partition_role'] = 'unknown'
        put(manifest.with_suffix('.jsonl.meta.json'), meta)
    (run / 'measurements.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
    output = tmp_path / 'proxies.json'
    with pytest.raises((ValueError, FileNotFoundError)): exporter.export_run(run, output)
    assert not output.exists()


@pytest.mark.parametrize('mutation', ['metric', 'image', 'metadata', 'array'])
def test_resume_rejects_changed_available_evidence(tmp_path, mutation):
    run, _, _ = fixture(tmp_path)
    output = tmp_path / 'proxies.json'; exporter.export_run(run, output)
    if mutation == 'metric': output.write_text('{}')
    elif mutation == 'image': (tmp_path / 'dataset/future0.png').write_bytes(b'changed')
    elif mutation == 'metadata': (run / 'summary.json').write_text('{}')
    else: (run / 'future-0.npy').write_bytes(b'changed')
    with pytest.raises(ValueError): exporter.export_run(run, output)


def test_import_has_no_gpu_or_numeric_library_side_effects():
    code = 'import os,sys; old=os.environ.get("CUDA_VISIBLE_DEVICES"); from scripts import export_recorded_ground_truth; assert "torch" not in sys.modules and "numpy" not in sys.modules; assert os.environ.get("CUDA_VISIBLE_DEVICES")==old'
    subprocess.run([sys.executable, '-c', code], check=True)
