"""Actual CPU synthetic batches and tamper tests; no GPU or perceptual claims."""
from copy import deepcopy
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import time

import numpy as np
import pytest

from drivebench.data import write_manifest
from drivebench.data.synthetic import synthetic_records
from drivebench_tuning import agent_evidence as evidence, vista_agent, vista_batch


def now(): return datetime.now(timezone.utc).isoformat()

def setup(tmp_path, *, scene_count=2):
    data = tmp_path / 'data'
    records = synthetic_records(data, validation_scenes=scene_count, training_scenes=0,
                                clips_per_scene=1, frame_count=3, height=8, width=8)
    manifest = tmp_path / 'inputs.jsonl'; write_manifest(manifest, records)
    return {'records': records, 'data': data, 'manifest': manifest, 'root': tmp_path}


def run(f, name='reference', role='reference', *, reps=(0, 1), settings=None):
    spec = {'schema_version': 1, 'cohort_id': 'evidence-fixture', 'batch_id': name,
        'synthetic': True, 'manifest': {'path': str(f['manifest']), 'sha256': vista_agent.file_hash(f['manifest']),
            'metadata_sha256': vista_agent.file_hash(Path(str(f['manifest']) + '.meta.json'))},
        'data_root': str(f['data']), 'settings': settings or vista_agent.reference_settings(synthetic=True),
        'model': {'backend': 'synthetic'}, 'expected_runtime': vista_agent.probe_runtime(synthetic=True),
        'source_files': {str(p): vista_agent.file_hash(p) for p in vista_batch.required_source_paths()},
        'requests': [{'record_id': f'{i}-{seed}-{rep}', 'request_id': r['request_id'], 'scene_id': r['scene_id'],
            'seed': seed, 'replicate': rep, 'action_sha256': vista_batch.action_sha256(r)}
            for i, r in enumerate(f['records']) for seed, rep in itertools.product((11, 23), reps)],
        'output': str(f['root'] / name), 'created_at': now(), 'deadline_unix': time.time() + 120,
        'warmup_requests': 1, 'minimum_free_bytes': 0}
    path = f['root'] / f'{name}-spec.json'; path.write_text(json.dumps(spec))
    vista_batch.run_spec(path, expected_sha256=vista_agent.file_hash(path))
    pins = {'expected_spec_sha256': vista_agent.file_hash(path),
            'expected_completion_sha256': vista_agent.file_hash(Path(spec['output']) / 'adapter-completed.json')}
    return Path(spec['output']), pins, role


def load(triple):
    directory, pins, role = triple
    return evidence.load_batch(directory, role=role, **pins)


def metric(a, b, **bindings):
    assert vista_agent.file_hash(a) == bindings['reference_sha256']
    assert vista_agent.file_hash(b) == bindings['candidate_sha256']
    start = bindings['conditioning_frames']
    return {'mae': float(np.mean(np.abs(np.load(a)[start:] - np.load(b)[start:])))}


def identity(ref):
    return evidence.identity_for_batches(ref, metric_definition_sha256=vista_agent.digest('synthetic MAE only'))


def write(path, obj): path.write_text(vista_agent.canonical(obj) + '\n')

def edit(path, callback):
    obj = json.loads(path.read_text()); callback(obj); write(path, obj)


def reanchor(directory, pins):
    """Simulate an upstream producer semantic bug, not a broken hash chain."""
    completed = directory / 'batch/completed.json'
    def bind(d):
        for field, filename in [('manifest_sha256', 'manifest.json'), ('engine_sha256', 'engine.json'), ('cleanup_sha256', 'cleanup.json')]:
            d[field] = vista_agent.file_hash(directory / 'batch' / filename)
    edit(completed, bind)
    edit(directory / 'adapter-completed.json', lambda d: d.update(
        prepared_inputs_sha256=vista_agent.file_hash(directory / 'prepared-inputs.json'),
        batch_completed_sha256=vista_agent.file_hash(completed)))
    pins['expected_completion_sha256'] = vista_agent.file_hash(directory / 'adapter-completed.json')


def test_completed_chain_reconstructs_inputs_and_retains_generation_times(tmp_path):
    f = setup(tmp_path); triple = run(f); verified = load(triple); d = verified.to_dict()
    assert len(d['rows']) == 8 and len(d['files']) >= 29
    assert d['quality_preserving_claim'] is False and d['external_worker_exit_verified'] is False
    rows = evidence.calibration_rows([verified], identity=identity([verified]), measure_pair=metric, metric_names=['mae'])
    original = json.loads((triple[0] / 'batch/completed.json').read_text())['rows']
    by_cell = {(r['scene_id'], r['seed'], r['replicate']): r for r in original}
    for row in rows:
        r = by_cell[(row['scene_id'], row['seed'], row['replicate'])]
        assert row['observed_at'] == r['observed_at']
        assert json.loads(row['record_id'])[:2] == ['reference', 'reference']
        assert row['repeat_metrics']['mae'] == 0
    assert rows[0]['conditioning_sha256'] == rows[2]['conditioning_sha256']
    assert rows[0]['input_sha256'] != rows[2]['input_sha256']
    detached = d; detached['rows'][0]['latency_ms'] = -1
    assert verified.to_dict()['rows'][0]['latency_ms'] > 0


@pytest.mark.parametrize('relative', ['spec.json', 'adapter-completed.json', 'prepared-inputs.json',
    'batch/completed.json', 'batch/manifest.json', 'batch/engine.json', 'batch/cleanup.json',
    'batch/00000.json', 'batch/00001.json', 'batch/00000.npy', 'batch/00001.npy'])
def test_every_artifact_link_and_warmup_is_validated(tmp_path, relative):
    triple = run(setup(tmp_path)); target = triple[0] / relative
    target.write_bytes(target.read_bytes() + b' ')
    # Whitespace leaves JSON semantics unchanged; linked hashes must detect it.
    if relative.endswith('.json') and relative.startswith('batch/000'):
        edit(target, lambda d: d.update(valid=False))
    with pytest.raises(ValueError): load(triple)


@pytest.mark.parametrize('relative', ['adapter-failure.json', 'batch/failure.json', 'batch/cleanup-failure.json'])
def test_failed_attempt_never_becomes_success(tmp_path, relative):
    triple = run(setup(tmp_path)); (triple[0] / relative).write_text('{}')
    with pytest.raises(ValueError, match='failed or partial'): load(triple)


@pytest.mark.parametrize('field', ['conditioning_sha256', 'input_sha256', 'action_sha256'])
def test_hash_consistent_prepared_input_lies_rejected_by_reconstruction(tmp_path, field):
    triple = run(setup(tmp_path)); directory, pins, _ = triple
    edit(directory / 'prepared-inputs.json', lambda d: d['records'][0].update({field: '0' * 64}))
    reanchor(directory, pins)
    with pytest.raises(ValueError, match='reconstruction'): load(triple)


@pytest.mark.parametrize('field', ['settings_sha256', 'runtime_sha256', 'cohort_id', 'model'])
def test_hash_consistent_manifest_drift_is_rejected(tmp_path, field):
    triple = run(setup(tmp_path)); directory, pins, _ = triple
    edit(directory / 'batch/manifest.json', lambda d: d.update({field: 'wrong'}))
    reanchor(directory, pins)
    with pytest.raises(ValueError, match='manifest'): load(triple)


def test_observed_image_and_manifest_hashes_checked_without_future_images(tmp_path):
    f = setup(tmp_path); triple = run(f)
    for r in f['records']:
        for p in r['future_frames']: (f['data'] / p).unlink()
    verified = load(triple)
    path = f['data'] / f['records'][0]['observed_frames'][0]
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError): load(triple)
    with pytest.raises(ValueError): verified.recheck()


def test_raw_symlink_and_raw_change_during_metric_rejected(tmp_path):
    triple = run(setup(tmp_path)); verified = load(triple)
    def bad(a, b, **kw):
        result = metric(a, b, **kw); Path(b).write_bytes(Path(b).read_bytes() + b'x'); return result
    with pytest.raises(ValueError, match='during metrics'):
        evidence.calibration_rows([verified], identity=identity([verified]), measure_pair=bad, metric_names=['mae'])
    raw = triple[0] / 'batch/00001.npy'; backup = tmp_path / 'moved.npy'; raw.rename(backup); raw.symlink_to(backup)
    with pytest.raises(ValueError): load(triple)


@pytest.mark.parametrize('bad', [{'mae': float('nan')}, {'mae': -1}, {'mae': True}, {'wrong': 0}])
def test_invalid_metric_callback_fails_closed(tmp_path, bad):
    verified = load(run(setup(tmp_path)))
    with pytest.raises(ValueError): evidence.calibration_rows([verified], identity=identity([verified]),
        measure_pair=lambda *a, **kw: bad, metric_names=['mae'])


def test_missing_calibration_anchor_and_duplicate_collection_rejected(tmp_path):
    f = setup(tmp_path); ref = load(run(f, reps=(0,)))
    with pytest.raises(ValueError, match='anchors missing'):
        evidence.calibration_rows([ref], identity=identity([ref]), measure_pair=metric, metric_names=['mae'])
    with pytest.raises(ValueError, match='duplicate batch'):
        evidence.identity_for_batches([ref, ref], metric_definition_sha256='0' * 64)


def test_development_pairs_exact_cells_and_candidate_digest(tmp_path):
    f = setup(tmp_path); ref = load(run(f)); settings = vista_agent.reference_settings(synthetic=True)
    settings['generation']['decode_chunk_size'] = 2
    candidate = load(run(f, 'candidate', 'llm', settings=settings))
    args = dict(identity=identity([ref]), candidate_sha256=vista_agent.digest(settings), measure_pair=metric, metric_names=['mae'])
    rows = evidence.pair_rows([ref], [candidate], **args)
    assert len(rows['quality_rows']) == 8 and rows['timing_rows'] == []
    assert rows['quality_preserving_claim'] is False
    with pytest.raises(ValueError, match='settings digest'):
        evidence.pair_rows([ref], [candidate], **{**args, 'candidate_sha256': '0' * 64})
    subset = load(run(f, 'subset', 'llm', settings=settings, reps=(0,)))
    with pytest.raises(ValueError, match='cells do not match'): evidence.pair_rows([ref], [subset], **args)


def final_fixture(tmp_path):
    f = setup(tmp_path); selected_at = time.time(); barrier = time.time(); settings = vista_agent.reference_settings(synthetic=True)
    settings['generation']['decode_chunk_size'] = 2
    refs, candidates = [], []
    refs.append(load(run(f, 'ref-0', reps=(0,))))
    candidates.append(load(run(f, 'llm-0', 'llm_selected', reps=(0,), settings=settings)))
    candidates.append(load(run(f, 'llm-1', 'llm_selected', reps=(1,), settings=settings)))
    refs.append(load(run(f, 'ref-1', reps=(1,))))
    ident = identity(refs); scenes = [r['scene_id'] for r in f['records']]
    plan = {'schema_version': 1, 'cohort_id': 'evidence-fixture', 'identity_sha256': vista_agent.digest(ident),
        'synthetic': True, 'all_selections_locked_at': barrier, 'deadline_unix': time.time() + 60,
        'selections': {role: {'config': settings, 'candidate_sha256': vista_agent.digest(settings), 'selected_at': selected_at}
            for role in ('llm', 'random', 'grid')}, 'batches': []}
    for rep, roles in [(0, ['reference', 'llm_selected', 'random_selected', 'grid_selected']),
                       (1, ['grid_selected', 'random_selected', 'llm_selected', 'reference'])]:
        for role in roles:
            plan['batches'].append({'batch_index': len(plan['batches']), 'block_id': 'block-0', 'replicate': rep,
                'role': role, 'scenes': scenes, 'seeds': [11, 23],
                'config': vista_agent.reference_settings(synthetic=True) if role == 'reference' else settings})
    path = tmp_path / 'final-plan.json'
    def seal():
        plan['sha256'] = vista_agent.digest({k: v for k, v in plan.items() if k != 'sha256'}); write(path, plan)
        return vista_agent.file_hash(path)
    args = dict(identity=ident, candidate_sha256=vista_agent.digest(settings), measure_pair=metric, metric_names=['mae'],
                final_plan_path=path, expected_final_plan_sha256=seal())
    return refs, candidates, args, plan, seal


def test_final_pairs_use_actual_chronology_and_generation_timestamps(tmp_path):
    refs, cand, args, _, _ = final_fixture(tmp_path)
    result = evidence.pair_rows(refs, cand, **args)
    assert len(result['timing_rows']) == 8
    assert {r['order'] for r in result['timing_rows'] if r['repeat'] == 0} == {'reference-first'}
    assert {r['order'] for r in result['timing_rows'] if r['repeat'] == 1} == {'candidate-first'}
    saved = {r['record_id']: r['observed_at'] for b in refs + cand for r in b.to_dict()['rows']}
    for row in result['quality_rows']: assert row['observed_at'] == saved[row['record_id']]
    for row in result['timing_rows']:
        assert row['observed_at'] in (saved[row['reference_measurement_id']], saved[row['candidate_measurement_id']])
    assert result['shared_reference_candidate_comparisons_independent'] is False


@pytest.mark.parametrize('case', ['late_barrier', 'missing_selection', 'changed_config', 'wrong_identity', 'wrong_order', 'wrong_cells', 'expired_deadline'])
def test_pinned_final_admission_enforces_three_selections_and_schedule(tmp_path, case):
    refs, cand, args, plan, seal = final_fixture(tmp_path)
    if case == 'late_barrier': plan['all_selections_locked_at'] = time.time()
    elif case == 'missing_selection': del plan['selections']['grid']
    elif case == 'changed_config': plan['selections']['llm']['candidate_sha256'] = '0' * 64
    elif case == 'wrong_identity': plan['identity_sha256'] = '0' * 64
    elif case == 'wrong_order': plan['batches'][0]['batch_index'] = 20
    elif case == 'wrong_cells': plan['batches'][0]['scenes'] = ['other']
    elif case == 'expired_deadline': plan['deadline_unix'] = plan['all_selections_locked_at']
    args['expected_final_plan_sha256'] = seal()
    with pytest.raises(ValueError): evidence.pair_rows(refs, cand, **args)


@pytest.mark.parametrize('field,value', [('denoiser_evaluations', 0), ('compiled_graph_delta', 1),
    ('runtime_sha256', '0' * 64), ('input_sha256', '0' * 64), ('latency_ms', -1),
    ('output_file', '../00001.npy'), ('peak_allocated_bytes', -1)])
def test_hash_consistent_measurement_semantics_rechecked(tmp_path, field, value):
    triple = run(setup(tmp_path)); directory, pins, _ = triple
    edit(directory / 'batch/completed.json', lambda d: d['rows'][0].update({field: value}))
    edit(directory / 'batch/00001.json', lambda d: d.update({field: value}))
    reanchor(directory, pins)
    with pytest.raises(ValueError): load(triple)


def test_original_spec_is_in_worker_watched_sources(tmp_path):
    triple = run(setup(tmp_path)); directory, pins, _ = triple
    def remove(d):
        key = next(p for p, h in d['source_files'].items() if h == pins['expected_spec_sha256'])
        del d['source_files'][key]
    edit(directory / 'batch/manifest.json', remove); reanchor(directory, pins)
    with pytest.raises(ValueError, match='original pinned spec'): load(triple)


def test_actual_session_cohort_batch_scoring_integration(tmp_path):
    from drivebench_tuning.agent_session import AgentSession, SessionPolicy
    from drivebench_tuning.agent_cohort import freeze_final_plan, selection_only_executor, POLICIES, ROLES
    from drivebench_tuning.agent_scoring import freeze_quality_policy, assess_proxy_quality, score_heldout_timing
    f = setup(tmp_path, scene_count=4); dev = f['records'][:2]; final = f['records'][2:]
    planned = now(); cal = load(run({**f, 'records': dev}, 'calibration'))
    ident = identity([cal]); calrows = evidence.calibration_rows([cal], identity=ident, measure_pair=metric, metric_names=['mae'])
    controls = []
    indexed = {r['record_id']: r for r in cal.to_dict()['rows']}
    for i, row in enumerate(calrows):
        raw = indexed[row['record_id']]; arr = np.load(raw['raw_path']); arr[1:] = 0
        dest = tmp_path / f'black-{i}.npy'; np.save(dest, arr)
        sha = vista_agent.file_hash(dest)
        controls.append({**row, 'record_id': 'control-' + row['record_id'], 'control_id': 'black-future',
            'observed_at': now(), 'output_sha256': sha, 'reference_output_sha256': row['output_sha256'],
            'metrics': metric(raw['raw_path'], dest, reference_sha256=row['output_sha256'], candidate_sha256=sha, conditioning_frames=1)})
    blocks = {f'b-{i}': [r['scene_id']] for i, r in enumerate(final)}
    policy = freeze_quality_policy(identity=ident, metric_margins={'mae': .0001},
        development_scenes=[r['scene_id'] for r in dev], final_scenes=[r['scene_id'] for r in final],
        seeds=[11, 23], replicates=[0, 1], calibration_records=calrows, negative_controls=controls,
        plan_created_at=planned, frozen_at=now(), minimum_development_scenes=2, minimum_final_scenes=2,
        timing_min_blocks=2, timing_block_scenes=blocks)
    assert policy.to_dict()['calibration']['status'] == 'ready'
    metadata = {'quality_policy_sha256': policy.sha256, 'identity_sha256': policy.identity_sha256,
                'source_manifest_sha256': vista_agent.digest('source-fixture'), 'inputs_manifest_sha256': vista_agent.digest('input-fixture')}
    sessions = {}; selection_started = now()
    for name in POLICIES:
        session_policy = SessionPolicy(session_id=name, cohort_id='evidence-fixture',
            reference_config=vista_agent.reference_settings(synthetic=True), allowed_config_values=vista_agent.ALLOWED_VALUES,
            wall_seconds=60, max_tool_calls=20, max_evaluations=5, max_profiles=0, metadata=metadata)
        def trial(proposal, context):
            return {'cohort_id':'evidence-fixture', 'status':'ok', 'screening_status':'no_observed_breach',
                    'candidate_sha256': vista_agent.digest(proposal['config']), **metadata}
        directory = tmp_path / ('session-' + name)
        with AgentSession.create(directory, policy=session_policy, executor=selection_only_executor(trial)) as session:
            pid = session.call('propose', {'overrides': {'generation': {'decode_chunk_size': 2}}})['result']['proposal_id']
            session.call('trial', {'proposal_id': pid}); session.call('submit', {'proposal_id': pid})
        sessions[name] = directory, session_policy
    path = tmp_path / 'actual-final-plan.json'
    plan = freeze_final_plan(path, sessions=sessions, quality_policy=policy, block_orders={b:list(ROLES) for b in blocks},
        source_manifest_sha256=metadata['source_manifest_sha256'], inputs_manifest_sha256=metadata['inputs_manifest_sha256'],
        deadline_unix=time.time()+60, synthetic=True)
    assert all(type(v['selected_at']) in (int, float) for v in plan['selections'].values())
    loaded = {role: [] for role in ROLES}
    for entry in plan['batches']:
        request_fixture = {**f, 'records':[r for r in final if r['scene_id'] in entry['scenes']]}
        loaded[entry['role']].append(load(run(request_fixture, f'final-{entry["batch_index"]}', entry['role'],
            reps=(entry['replicate'],), settings=entry['config'])))
    for method in POLICIES:
        output = evidence.pair_rows(loaded['reference'], loaded[method+'_selected'], identity=ident,
            candidate_sha256=plan['selections'][method]['candidate_sha256'], measure_pair=metric, metric_names=['mae'],
            final_plan_path=path, expected_final_plan_sha256=vista_agent.file_hash(path))
        assessed = assess_proxy_quality(policy, output['quality_rows'], split='final', selection_started_at=selection_started)
        scored = score_heldout_timing(policy, output['timing_rows'], quality_assessment=assessed, selection_started_at=selection_started)
        assert assessed['status'] == scored['status'] == 'pass'
        assert scored['confidence_claim'] is False and assessed['quality_preserving_claim'] is False
        assert len(output['quality_rows']) == len(output['timing_rows']) == 8
