"""Trusted Vista receipts to scoring records; no inference, scheduler or API calls.

Externally pin BOTH the adapter specification and completion-file hashes. This
checks artifacts and joins measured cells; hashes are integrity checks, not a
signature or protection from a hostile evaluator. Original manifest/observed
images must still be available at their evaluator-owned paths. No future images
are decoded. Completeness, proxy thresholds and uncertainty remain the scoring
policy's responsibility; this module never establishes quality preservation.

The trusted measure_pair callback accepts two paths plus reference_sha256,
candidate_sha256 and conditioning_frames, and returns exactly metric_names.
Bind its implementation/assets in identity_for_batches' metric definition hash.
Negative-control construction/receipts remain a separate trusted executor.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path

from . import vista_agent, vista_batch


def _time(value):
    if type(value) is not str:
        raise ValueError('timezone-aware generation timestamp required')
    stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError('timezone-aware generation timestamp required')
    return stamp.timestamp()


def _read(path, expected=None, *, limit=64 * 1024 * 1024):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError('missing, symlinked or oversized evidence file')
    raw = path.read_bytes()
    import hashlib
    actual = hashlib.sha256(raw).hexdigest()
    if expected is not None and actual != vista_batch._hash(expected, 'expected artifact hash'):
        raise ValueError('evidence file hash mismatch')
    value = vista_batch._json(raw)
    if not isinstance(value, dict):
        raise ValueError('evidence object required')
    return value, actual


def _same(actual, expected, label):
    if vista_agent.canonical(actual) != vista_agent.canonical(expected):
        raise ValueError(f'{label} mismatch')


def _positive(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{label} must be positive and finite')


def _cell(row):
    return row['scene_id'], row['seed'], row['replicate']


@dataclass(frozen=True)
class VerifiedBatch:
    """Detached immutable evidence snapshot; recheck hashes before metric use."""
    _json: str

    def to_dict(self):
        return vista_batch._json(self._json)

    def recheck(self):
        value = self.to_dict()
        for path, expected in value['files'].items():
            item = Path(path)
            if item.is_symlink() or not item.is_file() or vista_agent.file_hash(item) != expected:
                raise ValueError('evidence changed after batch validation')
        return value


def load_batch(directory, *, role, expected_spec_sha256, expected_completion_sha256):
    """Validate one completed new attempt, including all warmups and raw arrays.

    No current runtime/GPU is queried. Original source files are provenance in
    the pinned manifest, not re-executed; input preparation is independently
    reconstructed through the unchanged CPU core loader. An archived directory
    may move, but original manifest/data paths must remain available.
    """
    root = Path(directory)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError('absolute real batch evidence directory required')
    vista_batch._name(role, 'role')
    files = {}
    def read(relative, expected=None):
        path = root / relative
        value, sha = _read(path, expected)
        files[str(path)] = sha
        return value
    spec = read('spec.json', expected_spec_sha256)
    vista_batch._keys(spec, vista_batch.SPEC_KEYS, 'saved specification')
    if spec['schema_version'] != 1 or type(spec['synthetic']) is not bool:
        raise ValueError('invalid saved specification kind')
    settings = vista_agent.validate_settings(spec['settings'], synthetic=spec['synthetic'])
    model = vista_agent.model_for_settings(spec['model'], settings, synthetic=spec['synthetic'])
    complete = read('adapter-completed.json', expected_completion_sha256)
    for name in ('adapter-failure.json', 'batch/failure.json', 'batch/cleanup-failure.json'):
        if (root / name).exists(): raise ValueError('failed or partial attempt cannot be scored')
    for key, expected in {'status': 'ok', 'schema_version': 1, 'spec_sha256': expected_spec_sha256,
                          'cohort_id': spec['cohort_id'], 'batch_id': spec['batch_id'],
                          'synthetic': spec['synthetic'], 'quality_assessed': False,
                          'quality_preserving_claim': False}.items():
        _same(complete.get(key), expected, f'adapter {key}')
    prepared = read('prepared-inputs.json', complete['prepared_inputs_sha256'])
    _same(prepared.get('spec_sha256'), expected_spec_sha256, 'prepared spec')
    _same(prepared.get('future_images_decoded'), False, 'future frame boundary')
    _same(prepared.get('image_preparation_in_inference_timing'), False, 'preprocessing boundary')
    result = read('batch/completed.json', complete['batch_completed_sha256'])
    manifest = read('batch/manifest.json', result['manifest_sha256'])
    engine = read('batch/engine.json', result['engine_sha256'])
    cleanup = read('batch/cleanup.json', result['cleanup_sha256'])
    _same(cleanup.get('status'), 'completed', 'cleanup')
    _same(result.get('status'), 'ok', 'batch status')
    _same(result.get('quality_assessed'), False, 'batch quality scope')
    _same(result.get('headline_eligible'), False, 'batch headline scope')
    for key in ('cohort_id', 'synthetic'):
        _same(result.get(key), spec[key], f'batch {key}')
    for key, value in {'cohort_id': spec['cohort_id'], 'synthetic': spec['synthetic'],
                      'settings': settings, 'settings_sha256': vista_agent.digest(settings),
                      'model': model, 'runtime': spec['expected_runtime'],
                      'runtime_sha256': vista_agent.digest(spec['expected_runtime']),
                      'deadline_unix': spec['deadline_unix'], 'warmup_requests': spec['warmup_requests'],
                      'measurement_count': len(spec['requests'])}.items():
        _same(manifest.get(key), value, f'manifest {key}')
    _same(result['initialization_s'], engine['initialization_s'], 'engine initialization')
    _positive(result['initialization_s'], 'initialization_s')
    _positive(result['elapsed_s'], 'elapsed_s')
    started = _time(manifest['created_at']); deadline = spec['deadline_unix']
    if type(deadline) not in (int, float) or not math.isfinite(deadline): raise ValueError('invalid deadline')
    if not (_time(spec['created_at']) <= started <= _time(cleanup['at']) <= _time(result['completed_at'])
            <= _time(complete['completed_at']) < deadline):
        raise ValueError('batch completion chronology or deadline violated')
    # Reconstruct the exact observed-only CPU inputs; expired execution deadlines
    # do not prevent independent evidence verification after a run finishes.
    from drivebench.data import load_manifest, inference_request, read_preprocess, verify_input_checksums, PREPROCESS_VERSION
    from drivebench.data.manifests import resolve_frame
    source_manifest, meta = vista_batch._manifest_check(spec)
    files[str(source_manifest)] = spec['manifest']['sha256']
    if meta.is_file(): files[str(meta)] = spec['manifest']['metadata_sha256']
    if source_manifest.stat().st_size > 64 * 1024 * 1024: raise ValueError('manifest too large')
    for line in source_manifest.read_text().splitlines():
        if line.strip(): vista_batch._json(line)
    originals = load_manifest(source_manifest)
    indexed = {r['request_id']: r for r in originals}
    if len(indexed) != len(originals): raise ValueError('duplicate source manifest request')
    selected = spec['requests']; bindings = prepared['records']
    if not isinstance(selected, list) or not 1 <= len(selected) <= 512 or len(bindings) != len(selected):
        raise ValueError('invalid selected/prepared input counts')
    expected_inputs, decoded, cells, ids = [], {}, set(), set()
    for item, binding in zip(selected, bindings):
        vista_batch._keys(item, vista_batch.REQUEST_KEYS, 'selected request')
        if _cell(item) in cells or item['record_id'] in ids: raise ValueError('duplicate selected cell or ID')
        cells.add(_cell(item)); ids.add(item['record_id'])
        record = indexed.get(item['request_id'])
        if record is None: raise ValueError('selected request missing from manifest')
        for key in ('scene_id',): _same(record[key], item[key], 'selected scene')
        _same(record.get('synthetic', False), spec['synthetic'], 'source execution kind')
        _same(vista_batch.action_sha256(record), item['action_sha256'], 'recorded action')
        _same(record['conditioning_frame_count'], settings['generation']['conditioning_frames'], 'conditioning count')
        _same(record['total_frame_count'], settings['generation']['total_frames'], 'total frame count')
        if not spec['synthetic'] and record['trajectory_metadata'].get('synthetic', False):
            raise ValueError('real action provenance cannot be synthetic')
        verify_input_checksums(record, spec['data_root'], include_future=False)
        request = inference_request(record, spec['data_root'], candidate_count=1)
        _same(request['action_kinds'], ['recorded'], 'action kind')
        if item['request_id'] not in decoded:
            decoded[item['request_id']] = read_preprocess(request['observed_frames'], settings['generation']['height'], settings['generation']['width'])
        frames = decoded[item['request_id']]
        reconstructed = vista_agent.BatchRequest(item['record_id'], item['scene_id'], item['seed'], item['replicate'], frames, request['trajectories'][0])
        import hashlib
        expected_binding = {**item, 'input_sha256': reconstructed.input_sha256,
            'conditioning_sha256': vista_agent.digest({'frames_sha256': hashlib.sha256(frames.tobytes()).hexdigest(),
                'shape': list(frames.shape), 'trajectory_float32': reconstructed.trajectory.tolist(), 'scene_id': item['scene_id']}),
            'observed_frame_hashes': {frame: record['frame_checksums'][frame] for frame in record['observed_frames']},
            'preprocess_version': PREPROCESS_VERSION}
        _same(binding, expected_binding, 'prepared input reconstruction')
        expected_inputs.append({'record_id': item['record_id'], 'input_sha256': reconstructed.input_sha256})
        for frame in record['observed_frames']:
            files[str(resolve_frame(spec['data_root'], frame))] = record['frame_checksums'][frame]
    _same(manifest['inputs'], expected_inputs, 'manifest inputs')
    expected_sources = {**spec['source_files'], **{p: h for p, h in files.items() if p not in
                      {str(root / n) for n in ['spec.json', 'adapter-completed.json', 'prepared-inputs.json',
                         'batch/completed.json', 'batch/manifest.json', 'batch/engine.json', 'batch/cleanup.json']}}}
    for path, sha in expected_sources.items():
        _same(manifest['source_files'].get(path), sha, 'manifest source/input binding')
    original_spec_paths = set(manifest['source_files']) - set(expected_sources)
    if (len(original_spec_paths) != 1 or
            any(not Path(p).is_absolute() or manifest['source_files'][p] != expected_spec_sha256 for p in original_spec_paths)):
        raise ValueError('worker source binding must include the exact original pinned spec')
    _same(len(result['rows']), len(selected), 'measured count')
    _same(len(result['warmups']), spec['warmup_requests'], 'warmup count')
    _same(complete['measured_outputs'], len(selected), 'adapter measured count')
    _same(complete['warmup_outputs'], len(result['warmups']), 'adapter warmup count')
    all_rows = result['warmups'] + result['rows']; observed = started; outputs = set(); measured = []
    for index, row in enumerate(all_rows):
        binding = bindings[0 if index < len(result['warmups']) else index - len(result['warmups'])]
        warmup = index < len(result['warmups'])
        saved = read(f'batch/{index:05d}.json'); _same(saved, row, 'saved measurement row')
        for key in ('record_id', 'scene_id', 'seed', 'replicate', 'input_sha256'):
            _same(row[key], binding[key], f'measurement {key}')
        for key, expected in {'warmup': warmup, 'valid': True, 'synthetic': spec['synthetic'],
                              'runtime_sha256': manifest['runtime_sha256'], 'settings_sha256': manifest['settings_sha256']}.items():
            _same(row[key], expected, f'measurement {key}')
        _same(row['effective_config'], {**settings['generation'], 'candidate_count': 1, 'requested_batch_size': 1,
            'actual_batch_sizes': [1], 'optimizations': settings['optimizations']}, 'effective generation work')
        _same(row['denoiser_evaluations'], settings['generation']['steps'], 'denoiser evaluation count')
        if (row['failures'] or (not spec['synthetic'] and row['fallbacks']) or row['cache_stats'].get('context_hit') or
                row['cache_stats'].get('oom_attempts') or row['cache_stats'].get('denoiser_evaluations_complete') is not True):
            raise ValueError('failed, cached or incomplete measurement')
        delta = row['compiled_graph_delta']
        if type(delta) is not int or delta < 0 or (not warmup and delta != 0):
            raise ValueError('measured compilation or invalid compiler counter')
        for name in ('peak_allocated_bytes', 'peak_reserved_bytes'):
            vista_batch._integer(row[name], name)
        at = _time(row['observed_at'])
        if not observed <= at <= _time(cleanup['at']): raise ValueError('measurement chronology violated')
        observed = at; _positive(row['latency_ms'], 'latency_ms')
        if row['output_file'] != f'{index:05d}.npy' or row['output_file'] in outputs:
            raise ValueError('raw file traversal or reuse')
        outputs.add(row['output_file']); raw_path = root / 'batch' / row['output_file']
        sha = vista_batch._hash(row['output_sha256'], 'raw output hash')
        if raw_path.is_symlink() or not raw_path.is_file() or vista_agent.file_hash(raw_path) != sha:
            raise ValueError('raw output hash mismatch')
        files[str(raw_path)] = sha
        if not warmup:
            measured.append({**row, 'conditioning_sha256': binding['conditioning_sha256'], 'raw_path': str(raw_path),
                'record_id': vista_agent.canonical([role, spec['batch_id'], row['record_id']])})
    value = {'schema_version': 1, 'role': role, 'batch_id': spec['batch_id'], 'cohort_id': spec['cohort_id'],
             'spec': spec, 'manifest': manifest, 'completion': complete, 'rows': measured, 'files': files,
             'quality_preserving_claim': False, 'external_worker_exit_verified': False}
    verified = VerifiedBatch(vista_agent.canonical(value)); verified.recheck()
    return verified


def _collection(batches):
    if not isinstance(batches, (list, tuple)) or not batches or any(not isinstance(b, VerifiedBatch) for b in batches):
        raise ValueError('nonempty verified batch collection required')
    values = [b.recheck() for b in batches]; first = values[0]
    seen, cells = set(), set()
    for value in values:
        if value['batch_id'] in seen: raise ValueError('duplicate batch ID')
        seen.add(value['batch_id'])
        for key in ('role', 'cohort_id'): _same(value[key], first[key], f'collection {key}')
        for key in ('model', 'settings', 'expected_runtime', 'source_files', 'synthetic'):
            _same(value['spec'][key], first['spec'][key], f'collection {key}')
        for row in value['rows']:
            if _cell(row) in cells: raise ValueError('duplicate collected measurement cell')
            cells.add(_cell(row))
    return values


def identity_for_batches(reference_batches, *, metric_definition_sha256):
    """Identity to freeze prospectively in QualityPolicy before selection."""
    values = _collection(reference_batches); first = values[0]; spec = first['spec']
    if first['role'] != 'reference': raise ValueError('reference role required')
    _same(spec['settings'], vista_agent.reference_settings(synthetic=spec['synthetic']), 'canonical reference settings')
    vista_batch._hash(metric_definition_sha256, 'metric definition hash')
    return {'runtime_id': vista_agent.digest(spec['expected_runtime']), 'reference_id': vista_agent.digest(spec['settings']),
            'model_sha256': vista_agent.digest(spec['model']), 'source_files_sha256': vista_agent.digest(spec['source_files']),
            'cohort_id': spec['cohort_id'], 'metric_definition_sha256': metric_definition_sha256,
            'evidence_kind': 'synthetic' if spec['synthetic'] else 'measured'}


def _base(row, identity):
    return {key: row[key] for key in ('record_id', 'scene_id', 'seed', 'replicate', 'observed_at',
            'conditioning_sha256', 'input_sha256', 'output_sha256')} | {'identity_sha256': vista_agent.digest(identity)}


def _measure(a, b, callback, metric_names, conditioning_frames):
    if not callable(callback) or not isinstance(metric_names, (list, tuple)) or not metric_names or len(set(metric_names)) != len(metric_names):
        raise ValueError('trusted metric callback and unique metric names required')
    for row in (a, b):
        if vista_agent.file_hash(row['raw_path']) != row['output_sha256']: raise ValueError('raw output changed before metrics')
    result = callback(a['raw_path'], b['raw_path'], reference_sha256=a['output_sha256'],
                      candidate_sha256=b['output_sha256'], conditioning_frames=conditioning_frames)
    if not isinstance(result, dict) or set(result) != set(metric_names): raise ValueError('metric fields differ from frozen definition')
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in result.values()):
        raise ValueError('finite nonnegative metric values required')
    for row in (a, b):
        if vista_agent.file_hash(row['raw_path']) != row['output_sha256']: raise ValueError('raw output changed during metrics')
    return dict(result)


def calibration_rows(reference_batches, *, identity, measure_pair, metric_names):
    values = _collection(reference_batches)
    _same(identity, identity_for_batches(reference_batches, metric_definition_sha256=identity['metric_definition_sha256']), 'calibration identity')
    rows = sorted([r for b in values for r in b['rows']], key=_cell); result = []
    for row in rows:
        repeated = [r for r in rows if _cell(r)[:2] == _cell(row)[:2] and r['replicate'] != row['replicate']]
        other_seed = [r for r in rows if r['scene_id'] == row['scene_id'] and r['seed'] != row['seed'] and r['replicate'] == row['replicate']]
        if not repeated or not other_seed: raise ValueError('calibration repeat/seed anchors missing')
        repeat, seed = repeated[0], other_seed[0]
        if any(r['conditioning_sha256'] != row['conditioning_sha256'] for r in (repeat, seed)) or repeat['input_sha256'] != row['input_sha256']:
            raise ValueError('calibration conditioning/input drift')
        count = values[0]['spec']['settings']['generation']['conditioning_frames']
        result.append({**_base(row, identity), 'repeat_reference_sha256': repeat['output_sha256'],
            'seed_reference_sha256': seed['output_sha256'],
            'repeat_metrics': _measure(repeat, row, measure_pair, metric_names, count),
            'seed_metrics': _measure(seed, row, measure_pair, metric_names, count)})
    for b in reference_batches: b.recheck()
    return result


def pair_rows(reference_batches, candidate_batches, *, identity, candidate_sha256, measure_pair, metric_names,
              final_plan_path=None, expected_final_plan_sha256=None):
    """Join matching cells; final mode requires a pinned three-selection receipt.

    Returns quality_rows and timing_rows. Development calls omit final_plan_path
    and produce no timing rows. Incomplete but matching subsets are permitted;
    the scoring policy checks full required coverage. Shared reference reuse
    across separate candidate calls is explicit and statistically dependent.
    """
    refs, candidates = _collection(reference_batches), _collection(candidate_batches)
    _same(identity, identity_for_batches(reference_batches, metric_definition_sha256=identity['metric_definition_sha256']), 'reference identity')
    first = candidates[0]; ref_spec = refs[0]['spec']
    if first['role'] == 'reference': raise ValueError('candidate role required')
    for key in ('model', 'expected_runtime', 'source_files', 'synthetic', 'cohort_id'):
        _same(first['spec'][key], ref_spec[key], f'candidate/reference {key}')
    _same(vista_agent.digest(first['spec']['settings']), candidate_sha256, 'candidate settings digest')
    plan = None; barrier = None; batches_by_id = {}
    if (final_plan_path is None) != (expected_final_plan_sha256 is None): raise ValueError('both final plan and externally pinned hash required')
    if final_plan_path is not None:
        plan, plan_sha = _read(final_plan_path, expected_final_plan_sha256)
        _same(plan.get('sha256'), vista_agent.digest({k: v for k, v in plan.items() if k != 'sha256'}), 'final plan digest')
        if set(plan['selections']) != {'llm', 'random', 'grid'}: raise ValueError('all three selections required')
        _same(plan['cohort_id'], ref_spec['cohort_id'], 'final cohort')
        _same(plan['identity_sha256'], vista_agent.digest(identity), 'final policy identity')
        _same(plan['synthetic'], ref_spec['synthetic'], 'final evidence kind')
        barrier = plan['all_selections_locked_at']; _positive(barrier, 'selection barrier')
        for selected in plan['selections'].values():
            _same(selected['candidate_sha256'], vista_agent.digest(selected['config']), 'selection config')
            _positive(selected['selected_at'], 'selected_at')
            if selected['selected_at'] > barrier: raise ValueError('selection recorded after barrier')
        role = first['role'].removesuffix('_selected')
        if first['role'] != role + '_selected' or role not in plan['selections']: raise ValueError('unknown final candidate role')
        _same(plan['selections'][role]['candidate_sha256'], candidate_sha256, 'locked candidate')
        if len({b['batch_id'] for b in refs + candidates}) != len(refs + candidates):
            raise ValueError('final batch IDs must be unique across roles')
        for batch in refs + candidates:
            if _time(batch['manifest']['created_at']) <= barrier: raise ValueError('final generation began before all selections locked')
            if _time(batch['completion']['completed_at']) >= plan['deadline_unix']: raise ValueError('final evidence exceeds plan deadline')
            match = [entry for entry in plan['batches'] if entry['role'] == batch['role'] and
                {_cell(r) for r in batch['rows']} == {(s, seed, entry['replicate']) for s in entry['scenes'] for seed in entry['seeds']}]
            if len(match) != 1: raise ValueError('batch differs from frozen final cells')
            _same(match[0]['config'], batch['spec']['settings'], 'final batch config')
            batches_by_id[batch['batch_id']] = match[0]
        chronological = sorted(refs + candidates, key=lambda b: _time(b['manifest']['created_at']))
        if [batches_by_id[b['batch_id']]['batch_index'] for b in chronological] != sorted(batches_by_id[b['batch_id']]['batch_index'] for b in chronological):
            raise ValueError('final role order differs from frozen schedule')
    ref_rows = {_cell(r): (r, b) for b in refs for r in b['rows']}
    cand_rows = {_cell(r): (r, b) for b in candidates for r in b['rows']}
    if set(ref_rows) != set(cand_rows): raise ValueError('candidate/reference cells do not match')
    quality, timing = [], []
    for cell in sorted(ref_rows):
        a, ab = ref_rows[cell]; b, bb = cand_rows[cell]
        for key in ('input_sha256', 'conditioning_sha256'): _same(a[key], b[key], f'paired {key}')
        metrics = _measure(a, b, measure_pair, metric_names, ref_spec['settings']['generation']['conditioning_frames'])
        quality.append({**_base(b, identity), 'candidate_sha256': candidate_sha256,
                        'reference_output_sha256': a['output_sha256'], 'metrics': metrics})
        if plan is not None:
            if _time(ab['completion']['completed_at']) < _time(bb['manifest']['created_at']): order = 'reference-first'
            elif _time(bb['completion']['completed_at']) < _time(ab['manifest']['created_at']): order = 'candidate-first'
            else: raise ValueError('overlapping batch chronology cannot establish paired order')
            block = batches_by_id[ab['batch_id']]['block_id']
            _same(block, batches_by_id[bb['batch_id']]['block_id'], 'paired block')
            timing.append({'record_id': vista_agent.canonical(['pair', a['record_id'], b['record_id']]),
                'scene_id': cell[0], 'seed': cell[1], 'repeat': cell[2], 'block_id': block,
                'identity_sha256': vista_agent.digest(identity), 'candidate_sha256': candidate_sha256,
                'input_sha256': b['input_sha256'], 'observed_at': max([a['observed_at'], b['observed_at']], key=_time),
                'reference_measurement_id': a['record_id'], 'candidate_measurement_id': b['record_id'],
                'reference_latency_ms': a['latency_ms'], 'candidate_latency_ms': b['latency_ms'], 'order': order})
    for batch in [*reference_batches, *candidate_batches]: batch.recheck()
    if final_plan_path is not None: _read(final_plan_path, expected_final_plan_sha256)
    return {'quality_rows': quality, 'timing_rows': timing, 'quality_preserving_claim': False,
            'shared_reference_candidate_comparisons_independent': False}
