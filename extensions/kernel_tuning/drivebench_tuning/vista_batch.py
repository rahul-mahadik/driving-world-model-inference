"""Trusted manifest-to-batch CLI; no agent code, commands, imports or provisioning.

Run this inside a fresh externally time-limited worker. The evaluator supplies
an absolute specification path and its independently pinned SHA256. That spec
is private controller input, never an agent action. Hashes detect drift, not a
hostile writer able to replace the evaluator and the caller's pinned hash.

Only observed images and the recorded trajectory enter the existing core
preprocessor. Historical manifest seeds are untouched: each new-cohort seed
and replicate is explicitly selected in the spec. The adapter never decodes
future images. Its preparation time is outside run_batch's inference timing.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from . import vista_agent


SPEC_KEYS = frozenset({
    'schema_version', 'cohort_id', 'batch_id', 'synthetic', 'manifest', 'data_root',
    'settings', 'model', 'expected_runtime', 'source_files', 'requests', 'output',
    'created_at', 'deadline_unix', 'warmup_requests', 'minimum_free_bytes',
})
REQUEST_KEYS = frozenset({'record_id', 'request_id', 'scene_id', 'seed', 'replicate', 'action_sha256'})


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('duplicate JSON object key')
        value[key] = item
    return value


def _json(text):
    value = json.loads(text, object_pairs_hook=_object,
                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON value')))
    vista_agent.canonical(value)  # Also reject overflowed JSON numbers such as1e999.
    return value


def _keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f'{label} requires exactly its declared fields')


def _hash(value, label):
    if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(f'{label} must be lowercase SHA256')
    return value


def _name(value, label):
    if type(value) is not str or not value.strip():
        raise ValueError(f'{label} must be nonempty text')


def _integer(value, label, *, minimum=0, maximum=2**63-1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f'{label} outside allowed integer range')


def _path(value, label, *, kind=None):
    if type(value) is not str or not Path(value).is_absolute():
        raise ValueError(f'{label} must be an absolute evaluator-owned path')
    path = Path(value)
    if path.is_symlink() or '..' in path.parts:
        raise ValueError(f'{label} cannot be a symlink or contain parent traversal')
    if kind == 'file' and not path.is_file():
        raise ValueError(f'{label} must be an existing file')
    if kind == 'directory' and not path.is_dir():
        raise ValueError(f'{label} must be an existing directory')
    return path


def _deadline(deadline):
    if time.time() >= deadline:
        raise TimeoutError('immutable batch deadline reached during preparation')


def _write(path, value):
    with path.open('x') as stream:
        stream.write(vista_agent.canonical(value) + '\n')


def required_source_paths(model=None):
    """Paths a trusted caller must freeze, not an authorization or auto-refresh.

    Pass the trusted real-model mapping for Vista; all upstream Python/YAML and
    the exact inference YAML are included. Checkpoint bytes remain separately
    verified against model.checkpoint_sha256 by the unchanged core constructor.
    """
    import drivebench
    core = Path(drivebench.__file__).resolve().parent
    paths = {*core.rglob('*.py'), Path(__file__).resolve(), Path(vista_agent.__file__).resolve(),
             Path(__file__).resolve().parent / '__init__.py'}
    if model is not None:
        root = _path(model['upstream_dir'], 'upstream_dir', kind='directory')
        config = _path(model['inference_config'], 'inference_config', kind='file')
        if config.resolve() != root.resolve() / 'configs/inference/vista.yaml':
            raise ValueError('inference_config must be the pinned upstream Vista YAML')
        paths.update({config, *root.rglob('*.py'), *root.rglob('*.yaml'), *root.rglob('*.yml')})
    return sorted(paths, key=str)


def action_sha256(record):
    """Bind the exact recorded float32 action and its declarative provenance."""
    action = np.asarray(record['trajectory'], dtype=np.float32)
    if action.shape != (8,) or not np.isfinite(action).all():
        raise ValueError('manifest trajectory must contain eight finite float32 values')
    return vista_agent.digest({'trajectory_float32': action.tolist(),
                               'trajectory_metadata': record['trajectory_metadata']})


def _source_check(spec):
    sources = spec['source_files']
    if not isinstance(sources, dict) or not sources:
        raise ValueError('nonempty pinned source_files mapping required')
    for path, expected in sources.items():
        _path(path, 'source file', kind='file'); _hash(expected, 'source hash')
    model = None if spec['synthetic'] else spec['model']
    if not {str(path) for path in required_source_paths(model)} <= set(sources):
        raise ValueError('source_files must cover core, adapter/worker, and real upstream Python/YAML')
    if any(vista_agent.file_hash(path) != expected for path, expected in sources.items()):
        raise ValueError('pinned evaluator source drift')
    if model is not None:
        from drivebench.models.upstream import verify_checkout
        verify_checkout(model['upstream_dir'], model['upstream_commit'])


def _validate_spec(spec):
    _keys(spec, SPEC_KEYS, 'specification')
    if type(spec['schema_version']) is not int or spec['schema_version'] != 1 or type(spec['synthetic']) is not bool:
        raise ValueError('schema_version=1 and explicit boolean synthetic required')
    for key in ('cohort_id', 'batch_id'): _name(spec[key], key)
    _keys(spec['manifest'], {'path', 'sha256', 'metadata_sha256'}, 'manifest binding')
    _path(spec['manifest']['path'], 'manifest', kind='file'); _hash(spec['manifest']['sha256'], 'manifest hash')
    if spec['manifest']['metadata_sha256'] is not None: _hash(spec['manifest']['metadata_sha256'], 'manifest metadata hash')
    _path(spec['data_root'], 'data_root', kind='directory'); output = _path(spec['output'], 'output')
    if output.exists(): raise FileExistsError('output must be a fresh directory')
    vista_agent.validate_settings(spec['settings'], synthetic=spec['synthetic'])
    vista_agent.model_for_settings(spec['model'], spec['settings'], synthetic=spec['synthetic'])
    if not spec['synthetic']:
        for key in ('checkpoint', 'inference_config'): _path(spec['model'][key], key, kind='file')
        _path(spec['model']['upstream_dir'], 'upstream_dir', kind='directory')
        _hash(spec['model']['checkpoint_sha256'], 'checkpoint hash')
    if not isinstance(spec['expected_runtime'], dict) or not spec['expected_runtime']:
        raise ValueError('explicit expected_runtime required')
    if type(spec['created_at']) is not str: raise ValueError('created_at must be timezone-aware ISO text')
    created = datetime.fromisoformat(spec['created_at'].replace('Z', '+00:00'))
    if created.tzinfo is None or created.utcoffset() is None:
        raise ValueError('created_at must include a timezone')
    deadline = spec['deadline_unix']
    if type(deadline) not in (int, float) or not math.isfinite(deadline) or not created.timestamp() <= time.time() < deadline:
        raise ValueError('created_at must precede now and the immutable future deadline')
    _integer(spec['warmup_requests'], 'warmup_requests', minimum=1, maximum=16)
    _integer(spec['minimum_free_bytes'], 'minimum_free_bytes')
    if not isinstance(spec['requests'], list) or not 1 <= len(spec['requests']) <= 512:
        raise ValueError('one to 512 explicit selected requests required')
    ids, cells, scene_requests = set(), set(), {}
    for row in spec['requests']:
        _keys(row, REQUEST_KEYS, 'selected request')
        for key in ('record_id', 'request_id', 'scene_id'): _name(row[key], key)
        _integer(row['seed'], 'seed'); _integer(row['replicate'], 'replicate')
        _hash(row['action_sha256'], 'action hash')
        cell = (row['scene_id'], row['seed'], row['replicate'])
        if row['record_id'] in ids or cell in cells:
            raise ValueError('duplicate selected measurement ID or scene/seed/replicate cell')
        ids.add(row['record_id']); cells.add(cell)
        previous = scene_requests.setdefault(row['scene_id'], row['request_id'])
        if previous != row['request_id']:
            raise ValueError('one immutable manifest request per selected scene required')
    _source_check(spec)
    return spec


def _manifest_check(spec):
    path = Path(spec['manifest']['path']); meta = path.with_suffix(path.suffix + '.meta.json')
    if vista_agent.file_hash(path) != spec['manifest']['sha256']:
        raise ValueError('manifest content hash mismatch')
    expected_meta = spec['manifest']['metadata_sha256']
    if expected_meta is None:
        if meta.exists() or meta.is_symlink(): raise ValueError('unexpected manifest metadata sidecar')
    elif meta.is_symlink() or not meta.is_file() or vista_agent.file_hash(meta) != expected_meta:
        raise ValueError('manifest metadata hash mismatch')
    return path, meta


def _prepare(spec):
    from drivebench.data import inference_request, load_manifest, read_preprocess, verify_input_checksums, PREPROCESS_VERSION
    from drivebench.data.manifests import resolve_frame
    path, meta = _manifest_check(spec)
    if path.stat().st_size > 64 * 1024 * 1024: raise ValueError('manifest exceeds 64 MiB bound')
    # The existing core loader remains authoritative; first disallow ambiguous
    # duplicate JSON keys and nonfinite constants in the bound source document.
    for line in path.read_text().splitlines():
        if line.strip() and not isinstance(_json(line), dict): raise ValueError('manifest rows must be objects')
    records = load_manifest(path)
    indexed = {r.get('request_id', r.get('id')): r for r in records}
    if len(indexed) != len(records): raise ValueError('duplicate manifest request')
    prepared, bindings, decoded = [], [], {}
    watched = {str(path): spec['manifest']['sha256']}
    if meta.is_file(): watched[str(meta)] = spec['manifest']['metadata_sha256']
    for selected in spec['requests']:
        _deadline(spec['deadline_unix'])
        record = indexed.get(selected['request_id'])
        if record is None or record.get('scene_id') != selected['scene_id']:
            raise ValueError('selected request/scene missing or mismatched in manifest')
        if record.get('synthetic', False) is not spec['synthetic']:
            raise ValueError('manifest synthetic flag differs from execution kind')
        generation = spec['settings']['generation']
        if record.get('conditioning_frame_count') != generation['conditioning_frames'] or record.get('total_frame_count') != generation['total_frames']:
            raise ValueError('manifest conditioning/total frame contract mismatch')
        if action_sha256(record) != selected['action_sha256']:
            raise ValueError('recorded action/provenance hash mismatch')
        if not spec['synthetic'] and record.get('trajectory_metadata', {}).get('synthetic', False):
            raise ValueError('real manifest cannot bypass action provenance with synthetic metadata')
        observed = record.get('observed_frames')
        if not isinstance(observed, list) or len(observed) != generation['conditioning_frames'] or any(type(v) is not str for v in observed):
            raise ValueError('exact observed-frame prefix required')
        if len(set(observed)) != len(observed): raise ValueError('duplicate observed-frame path')
        for frame in observed:
            expected = _hash(record['frame_checksums'][frame], 'observed-frame hash')
            resolved = resolve_frame(spec['data_root'], frame)
            if not resolved.is_file(): raise ValueError('observed frame missing')
            if str(resolved) in watched and watched[str(resolved)] != expected: raise ValueError('conflicting input hashes')
            watched[str(resolved)] = expected
        verify_input_checksums(record, spec['data_root'], include_future=False)
        request = inference_request(record, spec['data_root'], candidate_count=1)
        if request['action_kinds'] != ['recorded'] or len(request['trajectories']) != 1:
            raise ValueError('only one recorded action is admitted')
        if selected['request_id'] not in decoded:
            decoded[selected['request_id']] = read_preprocess(request['observed_frames'], generation['height'], generation['width'])
        frames = decoded[selected['request_id']]
        verify_input_checksums(record, spec['data_root'], include_future=False)
        item = vista_agent.BatchRequest(selected['record_id'], selected['scene_id'], selected['seed'],
                                        selected['replicate'], frames, request['trajectories'][0])
        prepared.append(item)
        bindings.append({**selected, 'input_sha256': item.input_sha256,
                         'conditioning_sha256': vista_agent.digest({'frames_sha256': hashlib.sha256(frames.tobytes()).hexdigest(),
                             'shape': list(frames.shape), 'trajectory_float32': item.trajectory.tolist(), 'scene_id': item.scene_id}),
                         'observed_frame_hashes': {frame: record['frame_checksums'][frame] for frame in observed},
                         'preprocess_version': PREPROCESS_VERSION})
    _manifest_check(spec); _source_check(spec); _deadline(spec['deadline_unix'])
    return prepared, bindings, watched


def run_spec(spec_path, *, expected_sha256):
    """Validate a separately pinned evaluator spec and preserve one fresh attempt."""
    path = _path(str(spec_path), 'spec', kind='file'); _hash(expected_sha256, 'spec hash')
    if path.stat().st_size > 4 * 1024 * 1024: raise ValueError('spec exceeds 4 MiB bound')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256: raise ValueError('spec hash mismatch')
    spec = _validate_spec(_json(raw))
    output = Path(spec['output']); output.mkdir(parents=True, exist_ok=False)
    with (output / 'spec.json').open('xb') as stream: stream.write(raw)
    _write(output / 'adapter-started.json', {'schema_version': 1, 'spec_sha256': expected_sha256,
        'cohort_id': spec['cohort_id'], 'batch_id': spec['batch_id'], 'synthetic': spec['synthetic'],
        'created_at': datetime.now(timezone.utc).isoformat(), 'deadline_unix': spec['deadline_unix']})
    try:
        if vista_agent.probe_runtime(synthetic=spec['synthetic']) != spec['expected_runtime']:
            raise ValueError('expected runtime drift before preparation')
        prepared, bindings, watched = _prepare(spec)
        _write(output / 'prepared-inputs.json', {'spec_sha256': expected_sha256, 'records': bindings,
            'future_images_decoded': False, 'image_preparation_in_inference_timing': False})
        sources = dict(spec['source_files'])
        for filename, expected in {**watched, str(path): expected_sha256}.items():
            if filename in sources and sources[filename] != expected: raise ValueError('conflicting source/input binding')
            sources[filename] = expected
        result = vista_agent.run_batch(output / 'batch', settings=spec['settings'], model=spec['model'],
            requests=prepared, cohort_id=spec['cohort_id'], expected_runtime=spec['expected_runtime'],
            source_files=sources, deadline_unix=spec['deadline_unix'], warmup_requests=spec['warmup_requests'],
            synthetic=spec['synthetic'], minimum_free_bytes=spec['minimum_free_bytes'])
        _deadline(spec['deadline_unix']); _manifest_check(spec); _source_check(spec)
        if vista_agent.file_hash(path) != expected_sha256: raise ValueError('spec changed during execution')
        receipt = {'schema_version': 1, 'status': 'ok', 'cohort_id': spec['cohort_id'], 'batch_id': spec['batch_id'],
            'spec_sha256': expected_sha256, 'prepared_inputs_sha256': vista_agent.file_hash(output / 'prepared-inputs.json'),
            'batch_completed_sha256': vista_agent.file_hash(output / 'batch' / 'completed.json'),
            'measured_outputs': len(result['rows']), 'warmup_outputs': len(result['warmups']),
            'synthetic': spec['synthetic'], 'quality_assessed': False, 'quality_preserving_claim': False,
            'external_worker_exit_verification_required': True, 'completed_at': datetime.now(timezone.utc).isoformat()}
        _write(output / 'adapter-completed.json', receipt)
        return receipt
    except BaseException as error:
        _write(output / 'adapter-failure.json', {'status': 'failed', 'error_type': type(error).__name__,
            'message': str(error), 'spec_sha256': expected_sha256, 'at': datetime.now(timezone.utc).isoformat()})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True, help='Absolute private evaluator specification path')
    parser.add_argument('--spec-sha256', required=True, help='Expected hash pinned outside the worker')
    arguments = parser.parse_args(argv)
    print(vista_agent.canonical(run_spec(arguments.spec, expected_sha256=arguments.spec_sha256)))


if __name__ == '__main__': main()
