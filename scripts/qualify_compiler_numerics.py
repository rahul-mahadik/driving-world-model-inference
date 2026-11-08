#!/usr/bin/env python3
"""CPU-only, explicitly norm-qualified compiler evidence; never rewrites strict proofs.

Creation is pre-held-out only. Verification permits later held-out attempts only
when their recorded starts follow the unchanged policy AND qualification receipt.
The policy's study directory is the declared experiment boundary, not a claim
that this process can discover undisclosed experiments elsewhere.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import hashlib
import math
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from drivebench.benchmarks.environment import reference_identity, source_identity
from drivebench.config import digest, file_sha256, merge, validate_run_config
from drivebench.data.manifest import inference_request, load_manifest, verify_input_checksums
from drivebench.data.preprocess import read_preprocess
from drivebench.evaluation.io import artifact_sha256, candidates, load_run
from drivebench.evaluation.metrics import compare_videos
from drivebench.evaluation.study_policy import reviewed_policy_binding
from drivebench.study import generation_sources, quality_runtime_identity, quality_runtime_overlay, _verify_metric_receipt
from drivebench.validate import compare_outputs, cache_coverage, input_provenance

SOURCE = 'd8a32813492c0939a0a91573a945c914938972419946065e6822f6209fa470cb'
UPSTREAM_DEVELOPMENT_SOURCE = '2a8fed43b4f2d13e94f9310e9078cfada379f19d8f2f74107fe1066991b7ff00'
TYPE = 'compiler_norm_qualification_v1'
POLICY_TYPE = 'compiler_norm_policy_v1'
PRIMARY = {'total_frames': 25, 'conditioning_frames': 1, 'height': 576, 'width': 1024,
           'steps': 50, 'dtype': 'fp16', 'decode_chunk_size': 14, 'cond_aug': 0.0}
NUMERICAL = {'metric': 'generated_rgb_rmse', 'maximum': 0.01,
             'aggregation': 'max_per_candidate_clip', 'normalization': 'RGB_[0,1]',
             'conditioning_frames': 1, 'total_frames': 25}
QUALITY = {'rmse': .01, 'lpips': .003, 'temporal_derivative_mae': 1 / 255,
           'freeze_fraction_increase': 0.0}
REVIEW_RULE = {'version': 1, 'fixed_cases': 4, 'fixed_selection': 'sha256(request_id+candidate_id)',
              'worst_cases_per_metric': 2, 'worst_metrics': ['max_abs_error', 'rmse'],
              'selected_proof_candidates': 'all', 'ordered_generated_frames': 'all24_in_blinded_atlases',
              'native_error_frames': ['maximum_absolute_error', 'maximum_frame_rmse'],
              'failure_categories': ['disappearance','duplication','geometry_change','freezing','other_localized_failure'], 'any_new_localized_failure': 'reject',
              'uncertain': 'reject', 'rule_independent_of_error_threshold': True}
IDENTITY_KEYS = ('upstream_commit', 'checkpoint_sha256', 'inference_config_sha256',
                 'backend', 'compiler_runtime', 'compiler_feature_boundary')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stamp(value):
    result = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    require(result.tzinfo is not None, 'Timestamps must be timezone-aware')
    return result


def sha256_string(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def read(path):
    value = json.loads(Path(path).read_text())
    require(isinstance(value, dict), 'Expected JSON object')
    return value


class Ledger:
    def __init__(self):
        self.hashes = {}

    def bind(self, path, expected=None):
        path = Path(path).resolve(strict=True)
        require(path.is_file(), 'Evidence must be a regular file')
        actual = file_sha256(path)
        require(expected is None or actual == expected, f'Evidence checksum mismatch: {path}')
        require(str(path) not in self.hashes or self.hashes[str(path)] == actual, 'Evidence changed during validation')
        self.hashes[str(path)] = actual
        return path

    def record(self, value):
        require(isinstance(value, dict) and set(value) == {'path', 'sha256'}, 'Require explicit path and SHA256 binding')
        return self.bind(value['path'], value['sha256'])

    def unchanged(self):
        require(all(file_sha256(p) == h for p, h in self.hashes.items()), 'Evidence changed during validation')


def arrays(path, ids, generation):
    shape = (generation['total_frames'], generation['height'], generation['width'], 3)
    with np.load(path, allow_pickle=False) as source:
        require(len(source.files) == len(set(source.files)) and set(source.files) == set(ids), 'Saved candidate IDs differ')
        result = {key: source[key] for key in ids}
    for value in result.values():
        require(value.dtype == np.float32 and value.shape == shape, 'Require declared float32 THWC shape')
        require(np.isfinite(value).all() and value.min() >= 0 and value.max() <= 1, 'Require finite RGB in [0,1]')
    return result


def rgb_errors(left, right, conditioning_frames):
    """Frame-sized float64 scratch; exclude observed frames from the RMS budget."""
    require(left.shape == right.shape and 0 < conditioning_frames < len(left), 'Invalid comparison shape/conditioning count')
    frames, total, squared, maximum, location = [], 0, 0., -1., None
    edges = np.array([0., 1 / 255, .01, .1, 1.])
    histogram = np.zeros(4, dtype=np.int64)
    for index in range(conditioning_frames, len(left)):
        error = np.abs(left[index].astype(np.float64) - right[index].astype(np.float64))
        require(np.isfinite(error).all(), 'Nonfinite error')
        squares = float(np.square(error).sum()); peak = float(error.max())
        position = [index, *map(int, np.unravel_index(error.argmax(), error.shape))]
        if peak > maximum:
            maximum, location = peak, position
        total += error.size; squared += squares
        histogram += np.histogram(error, edges)[0]
        frames.append({'frame': index, 'rmse': float(np.sqrt(squares / error.size)),
                       'max_abs_error': peak, 'max_abs_location_THWC': position,
                       'p50_p95_p99_abs_error': np.quantile(error, [.5, .95, .99]).tolist()})
    rmse = float(np.sqrt(squared / total))
    return {'rmse': rmse, 'passed': rmse <= NUMERICAL['maximum'], 'max_abs_error': maximum,
            'max_abs_location_THWC': location, 'values': total, 'per_frame': frames,
            'absolute_error_histogram': {'edges': edges.tolist(), 'counts': histogram.tolist(),
                                        'intervals': 'left-closed/right-open; final includes1'},
            'conditioning_frames_excluded': conditioning_frames}


def _compiler_execution(runtime, compiler_identity):
    require(runtime.get('passed') is True and runtime.get('restoration_complete') is True
            and runtime.get('require_graphs') is True
            and runtime.get('runtime_implementation_sha256') == compiler_identity['implementation_sha256']
            and all(type(runtime.get(k)) is int and runtime[k] > 0
                    for k in ('graphs_created_by_runtime','kernels_created_by_runtime')),
            'Compiler execution/codegen/restoration identity failed')
    for label in ('before','after'):
        counters = runtime.get(label, {})
        require(counters.get('unimplemented') == {} and all(type(counters.get(k)) is int and counters[k] >= 0
                for k in ('unique_graphs','generated_kernels')), 'Compiler fallback/counter evidence missing')
    require(all(runtime['after'][k] >= runtime['before'][k] for k in ('unique_graphs','generated_kernels')),
            'Compiler counters reset during execution')


def _execution(details, options, generation, count, *, compiler_identity=None):
    expected = [1] * count
    effective = details.get('effective_config', {})
    require(effective.get('optimizations') == options and all(effective.get(k) == v for k, v in generation.items()), 'Execution configuration differs')
    require(effective.get('candidate_count') == count and effective.get('actual_batch_sizes') == expected
            and effective.get('requested_batch_size') == 1, 'Require actual serial candidate execution')
    require(details.get('fallbacks') == [] and details.get('denoiser_evaluations') == generation['steps'] * count, 'Fallback or incomplete denoiser work')
    stats = details.get('cache_stats', {})
    require(stats.get('oom_attempts') == [] and stats.get('denoiser_evaluations_complete') is True, 'OOM/completion evidence missing')
    if compiler_identity is not None:
        _compiler_execution(stats.get('compiler_runtime', {}), compiler_identity)


def _proof(proof, ledger):
    require(proof.get('schema_version') == 2 and proof.get('mode') == 'all', 'Require complete legacy all-mode schema2 evidence')
    require(type(proof.get('passed')) is bool and proof.get('atol') == 0 and proof.get('rtol') == 0, 'Original strict proof must retain explicit0/0 result')
    require(proof.get('implementation_sha256') == SOURCE == source_identity(), 'Frozen d8 source differs')
    config = proof['config']; generation, options = validate_run_config(config)
    generation, options = generation.to_dict(), options.to_dict()
    require(proof.get('generation') == generation and proof.get('optimizations') == options, 'Proof config differs')
    require(all(generation.get(k) == v for k, v in PRIMARY.items()), 'Primary generation settings required')
    require(options['compile_denoiser'] and not options['compile_decoder'] and options['compile_mode'] == 'default'
            and not options['dynamic_shapes'] and not options['fullgraph'] and options['max_batch_size'] == 1
            and options['feature_cache_interval'] == 1 and options['feature_cache_module'] is None,
            'Only static serial denoiser compiler without feature reuse is qualified')
    require(config['model'].get('backend') == 'vista' and config['model'].get('compiler_runtime') == 'registered_einops_attention_v1'
            and config['model'].get('compiler_feature_boundary') is None, 'Compiler selector differs')
    require(stamp(proof.get('created_at')) <= datetime.now(timezone.utc), 'Proof timestamp is missing or in the future')
    runtime = proof['runtime_identity']
    require(runtime == reference_identity(proof['environment']) and runtime.get('gpu', {}).get('uuid'), 'Full runtime identity differs')
    environment = read(ledger.bind(config['environment_verification']))
    require(environment.get('passed') is True and reference_identity(environment['environment']) == runtime
            and read(ledger.bind(config['environment_lock'])) == runtime, 'Environment proof/lock differs')
    provenance = proof['input_provenance']; manifest = ledger.bind(provenance['manifest_path'], provenance['manifest_sha256'])
    require(Path(config['data']['manifest']).resolve() == manifest, 'Proof config and input manifest differ')
    metadata = read(ledger.bind(manifest.with_suffix(manifest.suffix + '.meta.json'), provenance['manifest_metadata_sha256']))
    records = load_manifest(manifest); record = records[provenance['request_index']]
    require(metadata.get('partition_role') == 'development' and len(records) == len({r['scene_id'] for r in records}) == 32
            and provenance.get('synthetic') is False and not metadata.get('synthetic', False)
            and all(r.get('dataset') == 'nuscenes' and not r.get('trajectory_metadata', {}).get('synthetic', False) for r in records), 'Require complete32 real development scenes')
    require(config['data'].get('verify_checksums', True) is True, 'Observed checksums must be verified')
    verify_input_checksums(record, config['data']['root'], include_future=False)
    ids = provenance['candidate_ids']; require(len(ids) >= 2 and len(ids) == len(set(ids)), 'Two or more distinct candidate IDs required')
    request = inference_request(record, config['data']['root'], candidate_count=len(ids))
    frames = read_preprocess(request['observed_frames'], generation['height'], generation['width'])
    actual = input_provenance(manifest, records, record, request, frames, request_index=provenance['request_index'])
    require(actual == provenance and digest(provenance) == proof['input_provenance_sha256'] and proof.get('validated_request_count') == 1, 'Observed/action/seed provenance differs')
    for path in request['observed_frames']:
        ledger.bind(path)
    runs = proof['runs']; require({'upstream', 'refactored', 'isolation', 'uncached', 'cached'} <= set(runs), 'Incomplete all-mode runs')
    identity = runs['refactored']['model_identity']
    require(identity.get('backend') == 'vista', 'Selected proof model must identify Vista')
    compiler = identity.get('compiler_runtime', {})
    require(compiler.get('implementation_sha256') == digest({k:v for k,v in compiler.items() if k != 'implementation_sha256'})
            and compiler.get('source_sha256', {}).get('drivebench/runtime/compiler.py') == file_sha256(ROOT/'drivebench/runtime/compiler.py')
            and compiler.get('suppress_errors') is False and compiler.get('cache_size_limit') == 512
            and compiler.get('native_operator_unchanged') is True, 'Compiler implementation declaration differs')
    require(identity.get('compiler_runtime', {}).get('name') == config['model']['compiler_runtime']
            and identity.get('compiler_feature_boundary') is None, 'Active compiler identity differs')
    for key in ('upstream_commit', 'checkpoint_sha256'):
        require(identity.get(key) == config['model'].get(key) and identity.get(key), 'Configured model identity differs')
    require(ledger.bind(config['model']['checkpoint'], identity['checkpoint_sha256'])
            and ledger.bind(config['model']['inference_config'], identity['inference_config_sha256']), 'Model assets differ')
    for name, details in runs.items():
        require(details.get('candidate_count') == len(ids), 'Proof run candidate count differs')
        require(set(details.get('outputs', {})) == set(details.get('outputs_sha256', {})) and details['outputs'], 'Missing raw output/hash coverage')
        for kind, path in details['outputs'].items():
            ledger.bind(path, details['outputs_sha256'][kind])
        if name in ('refactored', 'isolation', 'uncached', 'cached'):
            require(all(details['model_identity'].get(k) == identity.get(k) for k in IDENTITY_KEYS), 'Selected model identity differs between proof runs')
            expected_options = dict(options)
            if name == 'uncached': expected_options['conditioning_cache_bytes'] = 0
            if name == 'cached' and not options['conditioning_cache_bytes']:
                budget = details.get('effective_config', {}).get('optimizations', {}).get('conditioning_cache_bytes')
                require(type(budget) is int and budget > 0, 'Optional cache diagnostic budget missing')
                expected_options['conditioning_cache_bytes'] = budget
            _execution(details, expected_options, generation, len(ids), compiler_identity=compiler)
            executions = details.get('executions', {})
            required = ('normal', 'reverse', 'action_change') if name == 'isolation' else ('normal',)
            for label in required:
                require(executions.get(label, {}).get('status') == 'ok', 'Required execution failed')
                _execution(executions[label], expected_options, generation, len(ids), compiler_identity=compiler)
    upstream = runs['upstream']; upid = upstream['model_identity']
    require(upid.get('backend') == 'upstream' and not upid.get('compiler_runtime') and not upid.get('compiler_feature_boundary')
            and all(upid.get(k) == identity.get(k) for k in ('checkpoint_sha256', 'upstream_commit', 'inference_config_sha256')), 'Reference model differs or is compiled')
    from drivebench.types import OptimizationConfig
    _execution(upstream, OptimizationConfig().to_dict(), generation, len(ids))
    require(upstream.get('executions', {}).get('normal', {}).get('status') == 'ok', 'Reference execution failed')
    _execution(upstream['executions']['normal'], OptimizationConfig().to_dict(), generation, len(ids))
    return generation, options, identity, ids, manifest, records


def _raw_checks(proof, generation, options, ids):
    runs = proof['runs']; shape = generation
    require(set(proof['checks']) == {'upstream_agreement','candidate_order','unrelated_candidate_action_change',
            'conditioning_cache_agreement','conditioning_cache_keys','conditioning_cache_exercised'}, 'Legacy all-mode check coverage differs')
    def values(run, kind='normal', keys=ids):
        return arrays(runs[run]['outputs'][kind], keys, shape)
    left, right = values('upstream'), values('refactored')
    strict = compare_outputs(left, right, atol=0, rtol=0)
    require(strict == proof['checks']['upstream_agreement'], 'Legacy upstream check differs from raw outputs')
    numeric = {key: rgb_errors(left[key], right[key], generation['conditioning_frames']) for key in ids}
    del left, right
    original = values('isolation')
    semantic = {'candidate_order': compare_outputs(original, values('isolation', 'reverse'), atol=0, rtol=0),
                'unrelated_candidate_action_change': compare_outputs({k:original[k] for k in ids[1:]},
                    values('isolation', 'other_candidates_after_action_change', ids[1:]), atol=0, rtol=0)}
    del original
    for key, value in semantic.items():
        require(value == proof['checks'][key], 'Legacy semantic comparison differs from saved arrays')
    cache = {'required': bool(options['conditioning_cache_bytes'])}
    cache['agreement'] = compare_outputs(values('uncached'), values('cached'), atol=0, rtol=0)
    require(cache['agreement'] == proof['checks']['conditioning_cache_agreement'], 'Legacy cache check differs from raw arrays')
    keys = runs['cached'].get('cache_key_checks', {})
    require(set(keys) == {'unchanged_input_hit', 'unchanged_key_equal', 'changed_input_missed', 'changed_key_differs'}
            and all(type(v) is bool for v in keys.values()), 'Cache invalidation check coverage differs')
    cache['keys'] = {**keys, 'passed': all(keys.values())}
    cache['coverage'] = cache_coverage(runs['cached'])
    require(cache['keys'] == proof['checks']['conditioning_cache_keys']
            and cache['coverage'] == proof['checks']['conditioning_cache_exercised'], 'Legacy cache key/coverage evidence differs')
    if cache['required']:
        semantic.update(conditioning_cache_agreement=cache['agreement'], conditioning_cache_keys=cache['keys'], conditioning_cache_exercised=cache['coverage'])
    require(proof['passed'] == all(x.get('passed') is True for x in proof['checks'].values()), 'Legacy overall pass contradicts its own checks')
    return {'passed': all(x['passed'] for x in numeric.values()), 'candidates': numeric, 'contract': NUMERICAL}, {
        'passed': all(x['passed'] for x in semantic.values()), 'checks': semantic, 'atol': 0, 'rtol': 0}, cache


def _run_bindings(run, ledger):
    for name, expected in artifact_sha256(run).items():
        ledger.bind(run.path / name, expected)


def _metric_bindings(study, plan, document, entry, ledger):
    """Reconstruct the aggregate from committed evaluator artifacts, including LPIPS."""
    aggregate = read(ledger.bind(study/'quality/development/aggregation-receipt.json'))
    expected_evaluation = str((study/'quality/development/aggregate/evaluation-vs-upstream.json').resolve())
    require(aggregate.get('results', {}).get('upstream') == expected_evaluation, 'Main aggregation receipt differs')
    require(expected_evaluation in aggregate.get('artifact_sha256', {}), 'Aggregate evaluation hash missing')
    for path, checksum in aggregate['artifact_sha256'].items():
        ledger.bind(path, checksum)
    artifacts = entry.get('metric_artifacts', [])
    require(artifacts, 'Committed development evaluator artifacts missing')
    pairs, source_receipts = [], []
    for artifact in artifacts:
        metric = ledger.record(artifact)
        require(metric.name == f"upstream--{entry['run']}.json"
                and metric.parent.name == 'metrics' and metric.parent.parent.parent == study/'quality/development/shards', 'Unexpected development metric location')
        complete = read(ledger.bind(metric.parent.parent/'complete.json'))
        receipt_path = metric.with_suffix('.receipt.json')
        receipt = read(ledger.bind(receipt_path))
        require(complete.get('metric_artifacts', {}).get(str(metric)) == artifact['sha256']
                and complete['metric_artifacts'].get(str(receipt_path)) == file_sha256(receipt_path), 'Committed metric receipt/hash differs')
        reference, candidate = complete['runs']['upstream'], complete['runs'][entry['run']]
        _verify_metric_receipt(metric, receipt, reference, candidate, plan)
        shard = read(metric)
        require(shard.get('perceptual') == document.get('perceptual')
                and shard.get('evaluator_source_sha256') == document.get('evaluator_source_sha256')
                and len(shard.get('runs', [])) == 1 and shard['runs'][0].get('complete_pairs') is True
                and not shard['runs'][0].get('errors') and not shard['runs'][0].get('synthetic'), 'Evaluator provenance or completeness differs')
        require(shard.get('perceptual', {}).get('name') == 'LPIPS/alex'
                and shard['perceptual'].get('status') == 'available', 'Measured pinned LPIPS evidence required')
        require(shard.get('evaluator_source_sha256'), 'Evaluator source hashes missing')
        for path, checksum in shard['evaluator_source_sha256'].items():
            ledger.bind(ROOT/'drivebench'/path, checksum)
        ledger.bind(shard['tolerances_path'], receipt['tolerances_sha256'])
        for path, checksum in receipt['review_artifact_sha256'].items():
            ledger.bind(path, checksum)
        pairs.extend(shard['runs'][0]['pairs']); source_receipts.append((reference,candidate))
    require(pairs == entry['pairs'], 'Main aggregate pairs differ from committed evaluator output')
    return source_receipts


def _development(policy, proof, ledger, generation, options, identity):
    study_path = ledger.record(policy['study']); require(study_path.name == 'plan.json', 'Bind main study plan.json')
    study = study_path.parent; plan = read(study_path)
    require(plan.get('generation_sources_at_plan') == generation_sources(), 'Main study generation source differs')
    old_environment = read(ledger.bind(plan['base_config']['environment_verification']))
    require(old_environment.get('passed') is True, 'Original study environment proof failed')
    if reference_identity(old_environment['environment']) != proof['runtime_identity']:
        runtime_path = ledger.record(policy['runtime_config'])
        overlay = quality_runtime_overlay(plan, runtime_path)
        require(overlay['environment_verification'] == proof['config']['environment_verification']
                and overlay['environment_lock'] == proof['config']['environment_lock'], 'Proof differs from audited replacement runtime')
    evaluation_path = ledger.record(policy['development_evaluation'])
    require(evaluation_path == (study / 'quality/development/aggregate/evaluation-vs-upstream.json').resolve()
            and plan.get('reference') == 'upstream', 'Require main upstream development aggregate')
    quality_path = ledger.record(policy['quality_policy'])
    reviewed = reviewed_policy_binding(study, plan, 'upstream', quality_path)
    frozen = yaml.safe_load(quality_path.read_text())
    require(stamp(frozen['frozen_at']) <= stamp(policy['frozen_at']), 'Norm policy predates its frozen quality policy')
    require(policy['variant'] in reviewed['calibrated_variants'], 'Variant absent from reviewed development policy')
    for metric, cap in QUALITY.items():
        setting = frozen['metrics'].get(metric, {})
        require(setting.get('aggregation') == 'max' and type(setting.get('maximum')) in (int, float)
                and 0 <= setting['maximum'] <= cap, 'Reviewed policy exceeds declared per-clip limits')
    document = read(evaluation_path); entry = next((x for x in document['runs'] if x['run'] == policy['variant']), None)
    require(entry and entry['complete_pairs'] and not entry['errors'] and not entry['synthetic']
            and entry['matched_pairs'] == len(entry['pairs']) == 32, 'Require all32 complete development pairs')
    metric_sources = _metric_bindings(study,plan,document,entry,ledger)
    ref, run = load_run(document['reference']), load_run(entry['run_path'])
    require(artifact_sha256(ref) == document['reference_artifact_sha256'] and artifact_sha256(run) == entry['artifact_sha256'], 'Development run artifacts differ')
    for item in (ref, run):
        _run_bindings(item, ledger)
    cfg = run.config
    require(cfg.get('generation') == generation and cfg.get('optimizations') == options and cfg.get('model') == proof['config']['model'], 'Development candidate implementation differs')
    require(ref.config['generation'] == generation and ref.config['model'].get('backend') == 'upstream', 'Development reference differs')
    index, reference = candidates(run), candidates(ref)
    require(set(index) == set(reference) and len(index) == 32 and len({p['scene_id'] for p in entry['pairs']}) == 32, 'Development identities incomplete')
    manifest_records = load_manifest(proof['input_provenance']['manifest_path'])
    expected_requests = {}
    for record in manifest_records:
        verify_input_checksums(record, proof['config']['data']['root'], include_future=False)
        request = inference_request(record, proof['config']['data']['root'], 1)
        key = (request['request_id'],0,request['candidate_ids'][0],request['seeds'][0])
        require(key not in expected_requests, 'Duplicate manifest request identity')
        expected_requests[key] = (record['scene_id'],digest(np.asarray(request['trajectories'][0]).tolist()))
        for path in request['observed_frames']:
            ledger.bind(path)
    require(set(index) == set(expected_requests), 'Development candidate IDs/seeds differ from manifest')
    seen = set(); passed = True
    for pair in entry['pairs']:
        key = (pair['request_id'], pair['repeat'], pair['candidate_id'], pair['seed'])
        require(key in index and key not in seen, 'Repeated or unknown development pair'); seen.add(key)
        row, candidate = index[key]; rr, rc = reference[key]
        require(pair['scene_id'] == row['scene_id'] == rr['scene_id']
                and pair['trajectory_sha256'] == candidate['trajectory_sha256'] == rc['trajectory_sha256']
                and expected_requests[key] == (pair['scene_id'],pair['trajectory_sha256'])
                , 'Development input/output hashes differ')
        for metric, cap in QUALITY.items():
            value = pair['metrics'].get(metric)
            require(type(value) in (int, float) and math.isfinite(value) and value >= 0, 'Missing/nonfinite development metric')
            passed &= value <= frozen['metrics'][metric]['maximum']
        require(math.isfinite(pair['metrics']['max_abs_error']), 'Missing localized-error diagnostic')
    require(seen == set(index), 'Missing development pairs')
    # Bind the actual source run identity rather than only aggregate metadata.
    raw_maps = {}; reference_sources = []
    for item, expected_backend in ((run, 'vista'), (ref, 'upstream')):
        sources = item.config.get('study_aggregation', {}).get('source_runs', [])
        require(sources, 'Main aggregate must identify its real source runs')
        require(sources == [pair[0 if expected_backend == 'upstream' else 1] for pair in metric_sources], 'Metric and generation source receipts differ')
        requests = []; raw_map = {}
        for receipt in sources:
            source = load_run(receipt['path']); _run_bindings(source, ledger)
            require(artifact_sha256(source) == receipt['artifact_sha256'], 'Source-run hashes changed')
            require(all(source.config.get(k) == item.config.get(k) for k in ('generation', 'optimizations', 'model')), 'Source-run implementation differs')
            actual_identity = read(ledger.bind(source.path / 'model_identity.json'))
            for key in ('upstream_commit', 'checkpoint_sha256', 'inference_config_sha256'):
                require(actual_identity.get(key) == identity.get(key), 'Source model/checkpoint differs')
            require(actual_identity.get('backend') == expected_backend, 'Source backend differs')
            if expected_backend == 'vista':
                require(all(actual_identity.get(k) == identity.get(k) for k in IDENTITY_KEYS), 'Source compiler identity differs')
            if receipt.get('adopted'):
                bound = receipt['generation_source_evidence']
                evidence = read(ledger.bind(bound['path'], bound['sha256']))
                allowed = {SOURCE} if expected_backend == 'vista' else {SOURCE, UPSTREAM_DEVELOPMENT_SOURCE}
                require(evidence.get('source_unchanged') is True and evidence.get('exit_code') == 0
                        and evidence.get('finished_at') and evidence.get('source_identity') in allowed
                        and digest(evidence.get('source_files')) == evidence['source_identity'], 'Adopted source evidence differs')
                if expected_backend == 'vista':
                    require(all(evidence['source_files'].get(k) == v for k,v in generation_sources().items()), 'Compiled generation source differs')
                else:
                    reference_sources.append(evidence['source_identity'])
            else:
                require(source.config.get('study_provenance', {}).get('generation_sources') == generation_sources(), 'Source provenance differs')
            # Quality may span verified replacements; full proof GPU remains exact below.
            require(quality_runtime_identity(source.environment) == quality_runtime_identity(proof['environment']), 'Development runtime is incompatible')
            for output in receipt.get('outputs', []):
                key = (output['request_id'], output['candidate_id'])
                require(key not in raw_map, 'Duplicate retained development output')
                raw_map[key] = (ledger.bind(output['path'], output['sha256']), output['sha256'])
            requests.extend(r['request_id'] for r in source.measurements)
        require(len(requests) == len(set(requests)) == 32 and set(requests) == {r['request_id'] for r in item.measurements}, 'Source runs do not cover32clips exactly')
        raw_maps[expected_backend] = raw_map
    numerical_rows = {}
    for pair in entry['pairs']:
        key = (pair['request_id'], pair['candidate_id'])
        lp, lh = raw_maps['upstream'][key]; rp, rh = raw_maps['vista'][key]
        require(lh == pair['reference_output_sha256'] and rh == pair['candidate_output_sha256'], 'Development saved output hash differs from evaluation')
        left, right = np.load(lp, allow_pickle=False), np.load(rp, allow_pickle=False)
        expected_shape = (generation['total_frames'], generation['height'], generation['width'], 3)
        for value in (left,right):
            require(value.dtype == np.float32 and value.shape == expected_shape and np.isfinite(value).all()
                    and value.min() >= 0 and value.max() <= 1, 'Development output dtype/shape/finite/range failed')
        count = generation['conditioning_frames']
        actual_metrics = compare_videos(left[count:],right[count:])
        require(all(actual_metrics[m] == pair['metrics'][m] for m in ('rmse','max_abs_error','temporal_derivative_mae','freeze_fraction_increase')), 'Development saved metrics differ from raw arrays')
        errors = rgb_errors(left,right,count); passed &= errors['passed']
        numerical_rows[pair['request_id']] = errors
    manifest = plan['manifests']['development']
    require(manifest['sha256'] == proof['input_provenance']['manifest_sha256'], 'Proof and development manifest differ')
    return {'passed': bool(passed), 'clips': 32, 'variant': policy['variant'], 'quality_policy': reviewed,
            'metrics_maximum': {m:max(p['metrics'][m] for p in entry['pairs']) for m in QUALITY},
            'raw_arrays_recomputed': True, 'numerical_by_request': numerical_rows,
            'reference_source_identities': sorted(set(reference_sources)),
            'basis': 'All32 retained outputs recomputed for RGB/temporal metrics; measured LPIPS remains bound to main evaluator artifacts'}, entry['pairs'], study


def _component(policy, proof, ledger, identity):
    path = ledger.record(policy['component_evidence']); value = read(path)
    component_generation, component_options = validate_run_config(value['config'])
    expected_options = dict(proof['optimizations']); expected_options['conditioning_cache_bytes'] = 0
    require(component_generation.to_dict() == proof['generation'] and component_options.to_dict() == expected_options,
            'Component must use the same compiler/generation with uncached conditioning')
    require(value.get('implementation_sha256') == SOURCE and value.get('source_unchanged') is True
            and value.get('bound_files_unchanged') is True, 'Component evidence source changed')
    require(all(value.get('model_identity', {}).get(k) == identity.get(k) for k in IDENTITY_KEYS), 'Component model/compiler differs')
    component_environment = read(ledger.bind(value['config']['environment_verification']))
    component_actual = read(ledger.bind(path.parent/'environment.json'))
    require(component_environment.get('passed') is True
            and reference_identity(component_environment['environment']) == reference_identity(component_actual)
            and quality_runtime_identity(component_actual) == quality_runtime_identity(proof['environment']), 'Component runtime differs')
    require(value.get('environment_gate', {}).get('passed') is True
            and value['environment_gate'].get('verification_sha256') == file_sha256(value['config']['environment_verification']), 'Component environment proof differs')
    require(all(value['input_provenance'].get(k) == proof['input_provenance'].get(k) for k in ('manifest_sha256','manifest_metadata_sha256','manifest_scene_ids','partition_role')), 'Component development input manifest differs')
    diagnostic = value['diagnostic']; executions = diagnostic.get('executions', [])
    require(value.get('status') in ('diagnostic_complete_strict_failed','diagnostic_complete_exact') and value.get('exit_code') in (0,2), 'Component diagnostic did not complete')
    require(diagnostic.get('capture_count') == 1 and diagnostic.get('input_tensors_reused_without_resampling') is True
            and diagnostic.get('same_backend_and_ema_scope') is True and diagnostic.get('callable_restored_after_capture') is True
            and len(executions) == 2 and {x.get('label') for x in executions} == {'eager','compiled'}
            and diagnostic.get('candidate_count') == 1 and diagnostic.get('denoiser_evaluations') == 2
            and diagnostic.get('sampler_rollouts_completed') == 0 and diagnostic.get('video_decoder_calls') == 0
            and diagnostic.get('autocast_policy') == 'fp16; actual input/output dtypes recorded'
            and diagnostic.get('cfg_order') == '25 unconditional frames followed by 25 conditional frames', 'Component execution contract missing')
    generation = component_generation.to_dict()
    latent_shape = [2*generation['total_frames'],4,generation['height']//8,generation['width']//8]
    shapes = {'x':latent_shape,'condition.concat':latent_shape,'sigma':[50],'mask':[50],
              'condition.crossattn':[50,1,3456],'condition.vector':[50,768]}
    before = executions[0]['input_hashes_before']
    require(set(before) == {'x','sigma','mask','condition.concat','condition.crossattn','condition.vector'}
            and all(sha256_string(v.get('sha256')) and v.get('shape') == shapes[k] and v.get('dtype') == 'torch.float32'
                    and v.get('device_type') == 'cuda' and len(v.get('stride',[])) == len(shapes[k])
                    and all(type(s) is int and s >= 0 for s in v['stride']) for k,v in before.items()), 'Actual component input hash/shape/dtype coverage missing')
    require(all(x.get('input_hashes_before') == before == x.get('input_hashes_after') and x.get('rng_state_unchanged') is True for x in executions), 'Component input/RNG changed')
    require(all(x.get('shape') == latent_shape and x.get('dtype') == 'float32' and sha256_string(x.get('output_sha256'))
                and set(x.get('rng_state_before',{})) == {'cpu','cuda'} and x.get('rng_state_before') == x.get('rng_state_after')
                and all(sha256_string(v) for v in x['rng_state_before'].values()) for x in executions)
            and executions[0]['rng_state_before'] == executions[1]['rng_state_before'], 'Component actual output/RNG hash coverage differs')
    comparison = diagnostic['comparison']
    require(all(type(comparison.get(k)) in (int,float) and math.isfinite(comparison[k]) and comparison[k] >= 0 for k in ('rmse','max_absolute_error')), 'Component numerical diagnostics missing')
    _compiler_execution(diagnostic['compiled_runtime'],identity['compiler_runtime'])
    require(value.get('bound_file_sha256'), 'Component dependency bindings missing')
    for name, checksum in value.get('bound_file_sha256', {}).items():
        ledger.bind(name, checksum)
    return {'path': str(path), 'sha256': ledger.hashes[str(path)], 'comparison': comparison,
            'runtime_identity': reference_identity(component_actual),
            'input_provenance': value['input_provenance'],
            'scope': 'One actual first-denoiser input. Latent units are not RGB; no latent threshold or operator equivalence claimed.'}


def review_cases(pairs, proof, numeric, development_numeric):
    """A threshold-independent selection; tie breaks fixed by identity."""
    key = lambda p: p['request_id'] + p['candidate_id']
    fixed = sorted(pairs, key=lambda p:hashlib.sha256(key(p).encode()).hexdigest())[:4]
    selected = {key(p):p for p in fixed}
    for metric in REVIEW_RULE['worst_metrics']:
        for pair in sorted(pairs, key=lambda p:(-p['metrics'][metric],key(p)))[:2]:
            selected[key(pair)] = pair
    cases = {}
    def error_frames(errors):
        return {'maximum_absolute_error':errors['max_abs_location_THWC'][0],
                'maximum_frame_rmse':max(errors['per_frame'],key=lambda x:x['rmse'])['frame']}
    for pair in selected.values():
        case = 'development:' + digest(key(pair))
        cases[case] = {'reference_output_sha256':pair['reference_output_sha256'], 'candidate_output_sha256':pair['candidate_output_sha256'],
                       'native_frame_indices':error_frames(development_numeric[pair['request_id']])}
    for candidate in numeric['candidates']:
        case = 'selected-proof:' + candidate
        cases[case] = {'reference_output_sha256':proof['runs']['upstream']['outputs_sha256']['normal'],
                      'candidate_output_sha256':proof['runs']['refactored']['outputs_sha256']['normal'],
                      'native_frame_indices':error_frames(numeric['candidates'][candidate])}
    return cases


def _review(path, policy_hash, proof_hash, cases, ledger, freeze):
    receipt = read(ledger.bind(path))
    require(receipt.get('type') == 'compiler_norm_localized_review_v1' and receipt.get('policy_sha256') == policy_hash
            and receipt.get('proof_sha256') == proof_hash and receipt.get('rule') == REVIEW_RULE, 'Review contract/bindings differ')
    require(freeze <= stamp(receipt.get('created_at')) <= datetime.now(timezone.utc) and receipt.get('held_out_consulted') is False
            and receipt.get('reviewer_kind') in ('human','agent_assisted'), 'Review chronology/declaration invalid')
    rows = receipt.get('cases', [])
    require(len(rows) == len(cases) and {x.get('case_id') for x in rows} == set(cases), 'Fixed/worst/proof review coverage incomplete')
    accepted = True
    for row in rows:
        require(all(row.get(k) == v for k,v in cases[row['case_id']].items()), 'Review outputs differ')
        require(row.get('frames_reviewed') == list(range(1,25)) and row.get('native_error_frames_reviewed') == REVIEW_RULE['native_error_frames']
                and row.get('ordered_atlas_frames_reviewed') is True
                and isinstance(row.get('observations'), str) and row['observations'].strip(), 'Review lacks ordered/native inspection evidence')
        require(type(row.get('new_localized_failure')) is bool and row.get('decision') in ('accept','reject','uncertain'), 'Review decision missing')
        assets = row.get('assets', []); require(assets, 'Review assets must be retained')
        for asset in assets:
            ledger.record(asset)
        flags = row.get('failure_categories', {})
        require(set(flags) == set(REVIEW_RULE['failure_categories']) and all(type(v) is bool for v in flags.values()), 'Missing explicit localized-failure categories')
        require(row.get('preference') in ('A','B','tie','uncertain'), 'Review preference missing')
        accepted &= row['new_localized_failure'] is False and not any(flags.values()) and row['decision'] == 'accept' and row['preference'] != 'uncertain'
    return {'passed': bool(accepted), 'cases': len(rows), 'path':str(Path(path).resolve()),
            'reviewer_kind':receipt['reviewer_kind'], 'scope':'Verified declared coverage and immutable review assets; visual judgment itself remains reviewer evidence.'}


def _chronology(study, policy, *, qualification_created=None):
    frozen = stamp(policy['frozen_at']); created = stamp(policy['created_at'])
    require(created <= frozen <= datetime.now(timezone.utc), 'Policy creation/freeze timestamp invalid')
    for phase in ('evaluation','evaluation_second_seed','evaluation_actions'):
        for config in (study/'quality'/phase).rglob('config.json'):
            started = stamp(read(config).get('started_at'))
            require(qualification_created is not None and started > stamp(qualification_created) and started > frozen,
                    'Held-out work started before qualification/policy; creation must precede all held-out attempts')


def inspect(proof_path, policy_path, review_path, *, qualification_created=None):
    ledger = Ledger(); proof_path = ledger.bind(proof_path); policy_path = ledger.bind(policy_path)
    proof, policy = read(proof_path), read(policy_path)
    require(policy.get('type') == POLICY_TYPE and policy.get('status') == 'frozen'
            and policy.get('source_identity') == SOURCE and policy.get('numerical') == NUMERICAL
            and policy.get('review_rule') == REVIEW_RULE, 'Explicit frozen norm policy required')
    require(policy.get('held_out_not_started_at_freeze') is True, 'Policy must declare held-out not started')
    generation, options, identity, ids, manifest, records = _proof(proof, ledger)
    numerical, semantics, optional_cache = _raw_checks(proof,generation,options,ids)
    development, pairs, study = _development(policy,proof,ledger,generation,options,identity)
    _chronology(study,policy,qualification_created=qualification_created)
    component = _component(policy,proof,ledger,identity)
    review = _review(review_path,ledger.hashes[str(policy_path)],ledger.hashes[str(proof_path)],
                     review_cases(pairs,proof,numerical,development['numerical_by_request']),ledger,max(stamp(policy['frozen_at']),stamp(proof['created_at'])))
    if qualification_created is not None:
        require(stamp(qualification_created) >= max(stamp(policy['frozen_at']),stamp(proof['created_at']),stamp(read(review_path)['created_at'])), 'Qualification predates its prerequisites')
    ledger.bind(__file__); ledger.unchanged()
    require(source_identity() == SOURCE, 'Package changed during qualification')
    return {'type':TYPE, 'qualified':all(x['passed'] for x in (numerical,semantics,development,review)),
            'legacy_strict_passed':proof['passed'], 'legacy_proof':{'path':str(proof_path),'sha256':ledger.hashes[str(proof_path)]},
            'policy':{'path':str(policy_path),'sha256':ledger.hashes[str(policy_path)],'frozen_at':policy['frozen_at']},
            'implementation_sha256':SOURCE, 'runtime_identity':proof['runtime_identity'], 'generation':generation,
            'optimizations':options, 'model_identity':identity,
            'configuration':{k:proof['config'][k] for k in ('model','generation','optimizations')},
            'development_manifest':{'path':str(manifest),'sha256':file_sha256(manifest),'scene_ids':sorted({r['scene_id'] for r in records})},
            'outputs_sha256':{p:h for p,h in ledger.hashes.items() if p.endswith('.npz')},
            'exact_semantics':semantics, 'numerical':numerical, 'optional_cache_diagnostic':optional_cache,
            'quality_development':development, 'component_diagnostic':component, 'localized_review':review,
            'artifact_sha256':ledger.hashes, 'qualifier_sha256':file_sha256(__file__),
            'claims':{'bitwise_upstream_equivalence':proof['checks']['upstream_agreement']['passed'],
                      'mathematical_equivalence_proven':False,'held_out_quality_validated':False,'headline_speedup_validated':False},
            'scope':'Development-selected RGB error-norm qualification plus exact selected-request candidate/cache semantics. Separate held-out quality/review and repeated same-GPU timing are mandatory.'}


def create(proof_path, policy_path, review_path, output):
    path = Path(output).resolve(); require(not path.exists(),'Qualifications cannot be overwritten')
    value = inspect(proof_path,policy_path,review_path)
    value['created_at'] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False); stream.write('\n')
    return value


def plan_review(proof_path, policy_path, output):
    """Write an unrated private case plan; this cannot create a qualification."""
    path=Path(output).resolve(); require(not path.exists(),'Review plans cannot be overwritten')
    ledger=Ledger(); proof_path=ledger.bind(proof_path); policy_path=ledger.bind(policy_path)
    proof,policy=read(proof_path),read(policy_path)
    require(policy.get('type') == POLICY_TYPE and policy.get('status') == 'frozen'
            and policy.get('source_identity') == SOURCE and policy.get('numerical') == NUMERICAL
            and policy.get('review_rule') == REVIEW_RULE and policy.get('held_out_not_started_at_freeze') is True,
            'Explicit frozen norm policy required')
    generation,options,identity,ids,_,_=_proof(proof,ledger)
    numerical,_,_=_raw_checks(proof,generation,options,ids)
    development,pairs,study=_development(policy,proof,ledger,generation,options,identity)
    _chronology(study,policy); _component(policy,proof,ledger,identity)
    cases=review_cases(pairs,proof,numerical,development['numerical_by_request'])
    # Explicit identities are private export instructions, never a blinded UI.
    identity_map={'development:'+digest(p['request_id']+p['candidate_id']):
                  {'request_id':p['request_id'],'candidate_id':p['candidate_id']} for p in pairs}
    identity_map.update({'selected-proof:'+key:{'request_id':proof['input_provenance']['request_id'],'candidate_id':key}
                         for key in ids})
    ledger.bind(__file__); ledger.unchanged()
    value={'type':'compiler_norm_localized_review_plan_v1','created_at':datetime.now(timezone.utc).isoformat(),
           'policy_sha256':ledger.hashes[str(policy_path)],'proof_sha256':ledger.hashes[str(proof_path)],
           'rule':REVIEW_RULE,'cases':{key:{**item,**identity_map[key]} for key,item in cases.items()},
           'artifact_sha256':ledger.hashes,'qualified':False,'status':'unrated_private_export_plan'}
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    path.chmod(0o600)
    return value


def verify(report_path):
    path=Path(report_path).resolve(); checksum=file_sha256(path); stored=read(path)
    created=stored.pop('created_at',None); require(stamp(created)<=datetime.now(timezone.utc),'Invalid qualification timestamp')
    actual=inspect(stored['legacy_proof']['path'],stored['policy']['path'],stored['localized_review']['path'],qualification_created=created)
    require(actual==stored and actual['qualified'] is True and file_sha256(path)==checksum,'Qualification failed, changed, or differs from recomputed evidence')
    return {**actual,'created_at':created,'report_path':str(path),'report_sha256':checksum}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='command',required=True)
    make=sub.add_parser('create')
    for name in ('proof','policy','review','output'):make.add_argument('--'+name,required=True)
    check=sub.add_parser('check');check.add_argument('--qualification',required=True)
    review=sub.add_parser('review-plan')
    for name in ('proof','policy','output'):review.add_argument('--'+name,required=True)
    args=parser.parse_args(argv)
    if args.command=='review-plan':
        value=plan_review(args.proof,args.policy,args.output)
        print(json.dumps({'type':value['type'],'status':value['status'],'case_count':len(value['cases']),'qualified':False}))
        return 0
    value=create(args.proof,args.policy,args.review,args.output) if args.command=='create' else verify(args.qualification)
    print(json.dumps({'type':value['type'],'qualified':value['qualified'],'legacy_strict_passed':value['legacy_strict_passed']},sort_keys=True))
    return 0 if value['qualified'] else 2


if __name__=='__main__':
    raise SystemExit(main())
