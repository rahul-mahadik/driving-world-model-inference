"""Synthetic evidence exercises assembly; no real ratings, videos or acceptance."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil

import pytest
import yaml

from drivebench.config import digest, load_config, merge, validate_run_config
from scripts import assemble_bootstrap_decision as module


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))
    return path


def reseal(path, update):
    document = module.read(path)
    update(document)
    put(path, document)


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    root = tmp_path / 'project'; root.mkdir()
    for name in ('drivebench', 'configs'):
        shutil.copytree(module.ROOT / name, root / name, ignore=shutil.ignore_patterns('__pycache__'))
    (root / 'scripts').mkdir()
    for name in ('run_bootstrap_evaluation.py', 'run_compiled_bootstrap.py'):
        shutil.copy(module.ROOT / 'scripts' / name, root / 'scripts' / name)
    protocol = put(root / 'outputs/bootstrap-review-protocol-v1.json', {'created_at': '2020-01-01T00:00:00Z'})
    monkeypatch.setattr(module, 'PROTOCOL_SHA', module.evaluator.sha(protocol))
    old_files = {'historical.py': 'a' * 64}
    monkeypatch.setattr(module.evaluator, 'UPSTREAM_SOURCE', digest(old_files))
    current_files = {str(p.relative_to(root / 'drivebench')): module.evaluator.sha(p)
                     for p in sorted((root / 'drivebench').rglob('*.py'))}
    scenes = [f'scene-{i:02}' for i in range(32)]
    runs, bindings, output_hashes = {}, {}, {}
    for name in ('upstream', *module.VARIANTS):
        run = root / 'outputs/study-bootstrap' / name; run.mkdir(parents=True)
        config = {'model': {'backend': 'upstream' if name == 'upstream' else 'vista'},
                  'data': {'manifest_sha256': module.MANIFEST_SHA,
                           'manifest_metadata': {'partition_role': 'development', 'manifest_sha256': module.MANIFEST_SHA,
                                                 'scene_ids': scenes, 'scene_ids_sha256': digest(scenes), 'clip_count': 32}},
                  'benchmark': {'save_outputs': 32, 'warmup_requests': 1 if name == 'upstream' else 3}}
        put(run / 'config.json', config); put(run / 'environment.json', {'fixture': True})
        put(run / 'summary.json', {'backend': config['model']['backend'], 'failures': 0})
        rows = []
        for i in range(32):
            filename = f'future-{i}.npy'; (run / filename).write_bytes(f'synthetic-fixture-{name}-{i}'.encode())
            output_hashes[name, i] = module.evaluator.sha(run / filename)
            rows.append({'request_id': f'r{i}', 'scene_id': scenes[i], 'status': 'ok', 'repeat': 0,
                         'candidates': [{'candidate_id': f'r{i}:c00', 'seed': i, 'frames_path': filename}]})
        (run / 'measurements.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
        host = root / 'outputs/host' / ('study-upstream-development-attempt-1.json' if name == 'upstream'
                                       else f'study-bootstrap-{name}-attempt-1.json')
        args = ['python', '-m', 'drivebench.benchmark', '--config', 'configs/study_upstream_development.yaml']
        if name != 'upstream':
            args = ['python', 'scripts/run_compiled_bootstrap.py', '--variant', name,
                    '--hostspec', 'configs/study_bootstrap_compiled_hostspec.json']
        put(host, {'status': 'completed', 'exit_code': 0, 'finished_at': '2020-01-01T00:01:00Z',
                   'source_unchanged': True, 'inputs_unchanged': True, 'argv': args,
                   'source_files': old_files if name == 'upstream' else current_files,
                   'source_identity': module.evaluator.UPSTREAM_SOURCE if name == 'upstream' else module.evaluator.SOURCE})
        bindings[str(host)] = module.evaluator.sha(host)
        bindings.update({str(p): module.evaluator.sha(p) for p in run.iterdir()})
        runs[name] = run
    for path in [*list((root / 'scripts').iterdir()), root / 'configs/study_bootstrap_compiled_hostspec.json']:
        bindings[str(path)] = module.evaluator.sha(path)
    qualification = root / 'outputs/study-bootstrap/qualification-v1'; qualification.mkdir()
    receipts, receipt_hashes, policies = {}, {}, {}
    metrics = {'max_abs_error': .1, 'lpips': .01, 'temporal_derivative_mae': .02}
    limits = {'max_abs_error': .2, 'lpips': .02, 'temporal_derivative_mae': .03}
    policy = {'schema_version': 1, 'policy_type': 'reviewed_metric_tolerances', 'performance': {'calibrated': False},
              'metrics': {k: {'maximum': v, 'aggregation': 'max' if k == 'max_abs_error' else 'mean'} for k, v in limits.items()}}
    for name in module.VARIANTS:
        pairs = [{'request_id': f'r{i}', 'scene_id': scenes[i], 'repeat': 0, 'candidate_id': f'r{i}:c00', 'seed': i,
                  'action_kind': 'recorded', 'trajectory_sha256': digest([i]), 'metrics': metrics,
                  'reference_output_sha256': output_hashes['upstream', i],
                  'candidate_output_sha256': output_hashes[name, i], 'recorded_ground_truth': {'mae': .3}} for i in range(32)]
        selected = sorted(pairs, key=lambda p: hashlib.sha256(f"2026:{p['scene_id']}:{p['candidate_id']}:{p['seed']}".encode()).digest())[:4]
        review = qualification / f'{name}-vs-upstream-review'; review.mkdir()
        (review / 'index.html').write_text('synthetic review fixture')
        cases = []
        for pair in selected:
            case = {k: v for k, v in pair.items() if k not in {'metrics', 'recorded_ground_truth'}}
            case.update(run=name, reference_run='upstream', conditioning_frames=1,
                        reference_sha256=pair['reference_output_sha256'], candidate_sha256=pair['candidate_output_sha256'])
            identity = {k: case.get(k) for k in ('request_id', 'scene_id', 'repeat', 'candidate_id', 'seed', 'run',
                        'reference_run', 'action_kind', 'trajectory_sha256', 'conditioning_frames')}
            identity.update(reference_sha256=case['reference_sha256'], candidate_sha256=case['candidate_sha256'], review_seed=2026)
            case_id = 'case-' + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
            swap = bool(hashlib.sha256(f'2026:{case_id}'.encode()).digest()[0] & 1)
            case.update(case_id=case_id, A=name if swap else 'upstream', B='upstream' if swap else name)
            cases.append(case)
            for ext in ('.png', '.gif'): (review / (case_id + ext)).write_bytes(b'synthetic preview fixture')
        with (review / 'review.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=module.FIELDS); writer.writeheader()
            writer.writerows({key: case['case_id'] if key == 'case_id' else '' for key in module.FIELDS} for case in cases)
        put(review / 'private_key.json', {'seed': 2026, 'cases': cases})
        aggregate = {k: {'pairs': 32, 'mean': v, 'max': v} for k, v in metrics.items()}
        artifacts = lambda run: {n: module.evaluator.sha(run / n) for n in module.ARTIFACTS}
        document = {'reference': str(runs['upstream']), 'reference_artifact_sha256': artifacts(runs['upstream']),
                    'perceptual': {'status': 'available', 'provenance': {'device': 'cuda',
                        'setup_receipt_sha256': module.evaluator.SETUP_SHA, 'state_dict_sha256': module.STATE_SHA}},
                    'evaluator_source_sha256': {str(p.relative_to(root / 'drivebench')): module.evaluator.sha(p)
                        for p in [root / 'drivebench/evaluate.py', *sorted((root / 'drivebench/evaluation').glob('*.py'))]},
                    'tolerances': {'frozen': False}, 'runs': [{'run': name, 'run_path': str(runs[name]), 'synthetic': False,
                        'complete_pairs': True, 'matched_pairs': 32, 'errors': [], 'gate': {'quality_preserving': False},
                        'artifact_sha256': artifacts(runs[name]), 'pairs': pairs, 'aggregates': aggregate}],
                    'review': {'cases': 4, 'index': str(review / 'index.html'), 'csv': str(review / 'review.csv'),
                               'private_key': str(review / 'private_key.json')}}
        metric_path = put(qualification / f'{name}-vs-upstream.json', document)
        put(qualification / f'{name}-evaluation-launch.json', {'status': 'completed', 'exit_code': 0,
            'inputs_unchanged': True, 'variant': name, 'source_identity': module.evaluator.SOURCE,
            'upstream_generation_source_identity': module.evaluator.UPSTREAM_SOURCE,
            'quality_acceptance_claim': False, 'descriptive_only': True, 'finished_at': '2020-01-01T00:02:00Z',
            'output_sha256': module.evaluator.sha(metric_path), 'bound_file_sha256': bindings,
            'review_artifact_sha256': {str(p): module.evaluator.sha(p) for p in review.iterdir()},
            'argv': ['drivebench.evaluate', '--runs', str(runs[name]), '--reference', str(runs['upstream']),
                     '--tolerances', str(root / 'configs/study_bootstrap_unfrozen_tolerances.yaml'),
                     '--lpips', '--lpips-device', 'cuda', '--review-limit', '4', '--output', str(metric_path)]})
        votes = root / 'ratings' / (name + '.csv'); votes.parent.mkdir(exist_ok=True)
        with votes.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=module.FIELDS); writer.writeheader()
            writer.writerows({'case_id': case['case_id'], 'preferred_panel_A_B_tie': 'tie', 'acceptable_A': 'yes',
                              'acceptable_B': 'yes', 'reviewer_notes': 'Synthetic fixture only.'} for case in cases)
        seal = put(root / 'ratings' / (name + '.seal.json'), {'schema_version': 1,
            'receipt_type': 'bootstrap_blind_ratings_v1', 'comparison': name, 'created_at': '2020-01-01T00:03:00Z',
            'reviewer_kind': 'agent_assisted', 'protocol': {'path': str(protocol), 'sha256': module.PROTOCOL_SHA},
            'ratings': {'path': str(votes), 'sha256': module.evaluator.sha(votes)},
            'review_index': {'path': str(review / 'index.html'), 'sha256': module.evaluator.sha(review / 'index.html')},
            'reviewed_case_ids': [c['case_id'] for c in cases],
            'reviewed_frames_per_case': {c['case_id']: 24 for c in cases}, 'all_panels_reviewed': True,
            'private_keys_read_before_seal': False, 'uncertain_case_ids': [],
            'review_artifact_sha256': {str(p): module.evaluator.sha(p) for p in review.iterdir() if p.name != 'private_key.json'}})
        receipts[name], receipt_hashes[name] = str(seal), module.evaluator.sha(seal)
        policy_path = root / 'ratings' / (name + '.policy.yaml'); policy_path.write_text(yaml.safe_dump(policy))
        policies[name] = str(policy_path)
    decision = put(root / 'explicit-decisions.json', {'schema_version': 1, 'held_out_consulted': False,
        'reviewer_kind': 'agent_assisted', 'created_at': '2020-01-01T00:04:00Z',
        'selected_optimized_reference': 'compiled-conditioning', 'reference_rationale': 'Synthetic explicit choice.',
        'variants': {name: {'status': 'accepted', 'rationale': 'Synthetic explicit assessment.',
                          'policy_rationales': {key: 'Synthetic predeclared criterion.' for key in limits}} for name in module.VARIANTS}})
    return {'project': root, 'draft': root / 'configs/full_study_registered_draft.yaml', 'qualification': qualification,
            'decisions': decision, 'rating_receipts': receipts, 'rating_hashes': receipt_hashes,
            'policies': policies, 'output': root / 'approved-v1'}


def test_whole_assembly_preserves_draft_and_requires_fresh_destination(evidence):
    original = Path(evidence['draft']).read_bytes()
    result = module.assemble(**evidence)
    assert result['status'] == 'approved_proposal_only'
    assert result['headline_claim'] is False and result['release_policy_frozen'] is False
    assert result['selected_optimized_reference'] == 'compiled-conditioning'
    assert len([p for p in result['bound_file_sha256'] if p.endswith('.npy')]) == 96
    assert all(str(p) in result['bound_file_sha256'] for p in (evidence['project'] / 'drivebench').rglob('*.py'))
    assert result['approved_spec_sha256'] == module.evaluator.sha(result['approved_spec'])
    approved = yaml.safe_load(Path(result['approved_spec']).read_text())
    assert approved['variants'] == yaml.safe_load(original)['variants']
    assert Path(evidence['draft']).read_bytes() == original
    with pytest.raises(FileExistsError): module.assemble(**evidence)


def test_explicit_rejection_rebases_only_primary_families_and_keeps_adoption(evidence):
    def update(value):
        value['selected_optimized_reference'] = 'conditioning-reuse'
        value['variants']['compiled-conditioning']['status'] = 'rejected'
    reseal(evidence['decisions'], update)
    result = module.assemble(**evidence)
    approved = yaml.safe_load(Path(result['approved_spec']).read_text())
    draft = yaml.safe_load(Path(evidence['draft']).read_text())
    assert len(approved['variants']) == 18
    for name in ('upstream', 'eager', 'conditioning-reuse', 'compiled-default', 'compiled-conditioning', 'compiled-boundary-1'):
        assert approved['variants'][name] == draft['variants'][name]
    assert approved['variants']['compiled-feature-2-control'] == draft['variants']['feature-2']
    assert approved['variants']['compiled-steps-47-control'] == draft['variants']['steps-47']
    for name, variant in approved['variants'].items():
        if name.startswith(('steps-', 'feature-')):
            config = merge(load_config(approved['base_config']), variant)
            _, options = validate_run_config(config)
            assert not options.compile_denoiser and options.conditioning_cache_bytes == 2 * 1024**3
            assert config['model']['compiler_runtime'] is None and config['model']['compiler_feature_boundary'] is None


@pytest.mark.parametrize('mutation', ['ratings_hash', 'missing_key', 'modified_votes', 'missing_rationale', 'null_limit',
    'strict_too_small', 'aggregate_drift', 'raw_output_drift', 'partial_pairs', 'lpips_weights', 'host_source',
    'missing_source_binding', 'early_decision', 'heldout', 'contradictory_reference', 'review_not_complete'])
def test_adversarial_evidence_never_writes_approved_output(evidence, mutation):
    name = 'compiled-default'; seal = Path(evidence['rating_receipts'][name])
    review = evidence['qualification'] / f'{name}-vs-upstream-review'
    metric = evidence['qualification'] / f'{name}-vs-upstream.json'
    launch = evidence['qualification'] / f'{name}-evaluation-launch.json'
    if mutation == 'ratings_hash': evidence['rating_hashes'][name] = '0' * 64
    elif mutation == 'missing_key': (review / 'private_key.json').unlink()
    elif mutation == 'modified_votes': Path(module.read(seal)['ratings']['path']).write_text('changed')
    elif mutation == 'missing_rationale': reseal(evidence['decisions'], lambda d: d['variants'][name].pop('policy_rationales'))
    elif mutation in ('null_limit', 'strict_too_small'):
        policy = Path(evidence['policies'][name]); value = yaml.safe_load(policy.read_text())
        value['metrics']['lpips']['maximum'] = None if mutation == 'null_limit' else .001
        policy.write_text(yaml.safe_dump(value))
    elif mutation in ('aggregate_drift', 'partial_pairs', 'lpips_weights'):
        def edit(value):
            if mutation == 'aggregate_drift': value['runs'][0]['aggregates']['lpips']['mean'] = .001
            elif mutation == 'partial_pairs': value['runs'][0]['pairs'].pop()
            else: value['perceptual']['provenance']['state_dict_sha256'] = '0' * 64
        reseal(metric, edit)
        reseal(launch, lambda d: d.update(output_sha256=module.evaluator.sha(metric)))
    elif mutation == 'raw_output_drift': (evidence['project'] / 'outputs/study-bootstrap/upstream/future-0.npy').write_bytes(b'changed')
    elif mutation == 'host_source':
        host = evidence['project'] / 'outputs/host/study-bootstrap-compiled-default-attempt-1.json'
        reseal(host, lambda d: d.update(source_unchanged=False))
        for variant in module.VARIANTS:
            reseal(evidence['qualification'] / f'{variant}-evaluation-launch.json',
                   lambda d: d['bound_file_sha256'].update({str(host): module.evaluator.sha(host)}))
    elif mutation == 'missing_source_binding':
        reseal(launch, lambda d: d['bound_file_sha256'].pop(str(evidence['project'] / 'scripts/run_bootstrap_evaluation.py')))
    elif mutation == 'early_decision': reseal(evidence['decisions'], lambda d: d.update(created_at='2020-01-01T00:01:00Z'))
    elif mutation == 'heldout': reseal(evidence['decisions'], lambda d: d.update(held_out_consulted=True))
    elif mutation == 'contradictory_reference': reseal(evidence['decisions'], lambda d: d.update(selected_optimized_reference='conditioning-reuse'))
    elif mutation == 'review_not_complete':
        reseal(seal, lambda d: d['reviewed_frames_per_case'].update({d['reviewed_case_ids'][0]: 23}))
        evidence['rating_hashes'][name] = module.evaluator.sha(seal)
    with pytest.raises((ValueError, FileNotFoundError)):
        module.assemble(**evidence)
    assert not evidence['output'].exists()


def test_both_seals_are_verified_before_any_key_is_parsed(evidence, monkeypatch):
    evidence['rating_hashes']['compiled-conditioning'] = '0' * 64
    original = module.read
    def spy(path):
        assert Path(path).name != 'private_key.json'
        return original(path)
    monkeypatch.setattr(module, 'read', spy)
    with pytest.raises(ValueError): module.assemble(**evidence)


@pytest.mark.parametrize('vote', ['no', 'uncertain'])
def test_candidate_panel_must_explicitly_accept_even_with_permissive_metrics(evidence, vote):
    name = 'compiled-conditioning'; seal_path = Path(evidence['rating_receipts'][name]); seal = module.read(seal_path)
    review = evidence['qualification'] / f'{name}-vs-upstream-review'
    case = module.read(review / 'private_key.json')['cases'][0]  # Synthetic fixture key only.
    panel = 'A' if case['A'] == name else 'B'
    votes_path = Path(seal['ratings']['path'])
    rows = list(csv.DictReader(votes_path.read_text().splitlines()))
    for row in rows:
        if row['case_id'] == case['case_id']: row['acceptable_' + panel] = vote
    with votes_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=module.FIELDS); writer.writeheader(); writer.writerows(rows)
    seal['ratings']['sha256'] = module.evaluator.sha(votes_path)
    seal['uncertain_case_ids'] = [case['case_id']] if vote == 'uncertain' else []
    put(seal_path, seal); evidence['rating_hashes'][name] = module.evaluator.sha(seal_path)
    with pytest.raises(ValueError, match='candidate votes'): module.assemble(**evidence)
    reseal(evidence['decisions'], lambda d: (d.update(selected_optimized_reference='conditioning-reuse'),
                                            d['variants'][name].update(status='rejected')))
    result = module.assemble(**evidence)
    assert result['comparison_evidence'][name]['review']['candidate_accepted_in_all_cases'] is False


def test_duplicate_assignments_and_missing_external_hash_reject():
    with pytest.raises(ValueError): module.assignments(['compiled-default=a', 'compiled-default=b'])
    with pytest.raises(ValueError): module.assignments(['unknown=a'])
    with pytest.raises(ValueError): module.Ledger().bind_map({'missing': None})
