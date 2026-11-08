#!/usr/bin/env python3
"""Separately admitted, generation-only A100 math-SDPA development smoke."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import signal
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
REL = 'outputs/host/a100-math-quality-v1'
RESULT = 'outputs/a100-math-quality-v1'
RUNTIME = 'outputs/host/a100-math-sdpa-v1/runtime.py'
RUNTIME_SHA = 'b9246c90ec52ba2e8c83ab238b9797fb76365f3c387fd1b8ffe1cd05ada53b94'
STRICT_SPEC = 'outputs/host/a100-math-sdpa-v1/validate-spec.json'
STRICT_SPEC_SHA = 'b74b2cf6a5058ee345073f4805190fc9796fedf16321a4e144e86226a154c8c9'
LAUNCHER = 'outputs/host/run-bounded-a100-job-v1.py'
LAUNCHER_SHA = '8acf1240512bc75ef69ab41888794cb3fadd8ce83835b68c67121e98618d99a3'
SOURCE = 'd8a32813492c0939a0a91573a945c914938972419946065e6822f6209fa470cb'
DEADLINE = 1793168129.682277
VARIANTS = ('eager50', 'eager40', 'feature2')
MANIFEST_PINS = {
    'data/manifests/smoke.jsonl': 'f19b73e956c0dd5868124f35abae2a92b96886279d73819b6679477e1b2eb2e1',
    'data/manifests/smoke.jsonl.meta.json': '97e6ef729dfb5d0509ded876f350555380cc95a40961dc47d8df12c43f9a87cf',
    'data/manifests/development.jsonl': 'a3c6328d9f0d0b94149820f626db17e024e6fbc0e5523fe5ee19aa3316befbff',
    'data/manifests/development.jsonl.meta.json': 'f07de92cf445a4e69850a36f206f7a1651798b997ff47c8f3794d2afcde9bd32',
}
SHAPE = (25, 576, 1024, 3)
ARRAY_BYTES = math.prod(SHAPE) * 4
RESERVE_SECONDS = 1800
MAX_JOB_SECONDS = 3600
RATE_USD_PER_HOUR = 2.182 + .0547945205479452
SCOPE = ('Four fixed development scenes under a separately declared A100 math-SDPA runtime. '
         'Generation and descriptive timings only; no H100 adoption, quality acceptance, '
         'matched-latency, speedup, tail-latency, threshold calibration or headline claim.')


def read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError('Expected a JSON object')
    return value


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def project_file(value):
    p = Path(value)
    if p.is_absolute() and str(p).startswith('/workspace/'):
        p = ROOT / p.relative_to('/workspace')
    elif not p.is_absolute():
        p = ROOT / p
    if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Missing, redirected or non-project artifact: ' + str(value))
    return p


def bindings(paths):
    return {str(project_file(p)): sha(project_file(p)) for p in paths}


def check_files(values):
    for name, expected in values.items():
        if sha(project_file(name)) != expected:
            raise ValueError('Bound artifact changed: ' + str(name))
    from drivebench.benchmarks.environment import source_identity
    if source_identity() != SOURCE:
        raise ValueError('Frozen engine source changed')


def positive(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('Positive finite duration required')
    return float(value)


def epoch(value):
    result = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('Timezone-aware evidence timestamps required')
    return result.timestamp()


def elapsed(value):
    return positive(epoch(value['finished_at']) - epoch(value['started_at']))


def dependencies(expected):
    if sha(__file__) != expected or sha(project_file(RUNTIME)) != RUNTIME_SHA:
        raise ValueError('Reviewed quality/runtime helper hash differs')
    spec = importlib.util.spec_from_file_location('quality_frozen_math_runtime', project_file(RUNTIME))
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    ab, encoder, base, verifier = runtime.dependencies()
    original = runtime.config_checked(runtime.CONFIG_PATH, base, verifier)
    return SimpleNamespace(runtime=runtime, ab=ab, encoder=encoder, base=base,
                           verifier=verifier, original=original)


def expected_config(original, variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown fixed smoke variant')
    config = copy.deepcopy(original)
    config['a100_pilot']['scope'] = ('Inherited platform and selected-request strict provenance only; '
        'this auxiliary generates four fixed development scenes under separately declared math SDPA. '
        'No H100 adoption or headline claim.')
    config['name'] = 'a100-math-quality-v1-' + variant
    config['output'] = '/workspace/' + RESULT + '/runs/' + variant
    config['data']['manifest'] = '/workspace/data/manifests/smoke.jsonl'
    config['numerical_validation'] = '/workspace/outputs/a100-math-sdpa-v1/strict-all.json'
    config['benchmark'].update(requests=4, candidate_count=1, workload='single_future',
        warmup_requests=1, max_warmup_requests=1, save_outputs=4, headline=False, input_condition='uncached')
    if variant == 'eager40':
        config['generation']['steps'] = 40
    if variant == 'feature2':
        config['optimizations'].update(feature_cache_interval=2, feature_cache_module='output_blocks.9')
    config['a100_quality_smoke'] = {'variant': variant, 'scope': SCOPE,
        'strict_coverage': 'Selected original development request at50 steps, not exact equivalence of approximations or all smoke scenes.',
        'retire_outputs': False, 'evaluation_pending': True}
    return config


def config_checked(context, variant):
    from drivebench.config import load_config, validate_run_config
    path = project_file(REL + '/configs/' + variant + '.json')
    config = load_config(path)
    if config != expected_config(context.original, variant):
        raise ValueError('Only the fixed generation-only smoke configuration is allowed')
    validate_run_config(config)
    return path, config


def smoke_inputs(config):
    from drivebench.config import digest
    from drivebench.data.manifest import load_manifest, inference_request, verify_input_checksums
    for name, expected in MANIFEST_PINS.items():
        if sha(project_file(name)) != expected:
            raise ValueError('Fixed smoke/development manifest or metadata changed')
    manifest = project_file(config['data']['manifest'])
    records = load_manifest(manifest)
    meta = read(manifest.with_suffix('.jsonl.meta.json'))
    parent = load_manifest(project_file('data/manifests/development.jsonl'))
    if (len(records) != 4 or len({r['scene_id'] for r in records}) != 4 or records != parent[:4]
            or meta.get('partition_role') != 'development_subset' or meta.get('parent_manifest_role') != 'development'
            or meta.get('parent_manifest_sha256') != MANIFEST_PINS['data/manifests/development.jsonl']
            or meta.get('scene_ids_sha256') != digest(sorted({r['scene_id'] for r in records}))):
        raise ValueError('Fixed four-scene development-only provenance differs')
    bound = bindings(MANIFEST_PINS)
    expected = []
    root = ROOT / 'data/nuscenes'
    for record in records:
        # Future frames are evaluator inputs, never read for generation admission.
        verify_input_checksums(record, root, include_future=False)
        request = inference_request(record, root, candidate_count=1)
        if len(request['observed_frames']) != 1 or request['action_kinds'] != ['recorded']:
            raise ValueError('Require one observed frame and the recorded trajectory')
        bound.update(bindings(request['observed_frames']))
        expected.append({'request_id': request['request_id'], 'scene_id': request['scene_id'],
            'candidate_id': request['candidate_ids'][0], 'seed': request['seeds'][0],
            'trajectory_sha256': digest(request['trajectories'][0].tolist())})
    return records, expected, bound


def lifecycle(execution, current_flags):
    active = copy.deepcopy(current_flags)
    active['sdp'] = {'flash_sdp_enabled': False, 'mem_efficient_sdp_enabled': False, 'math_sdp_enabled': True}
    count = execution.get('sdpa_call_count')
    geometry = execution.get('sdpa_calls', {})
    if (any(execution.get(key) is not True for key in ('flags_restored', 'active_flags_unchanged', 'callable_restored'))
            or execution.get('flags_before') != current_flags or execution.get('flags_after') != current_flags
            or execution.get('flags_active') != active or type(count) is not int or count <= 0
            or not isinstance(geometry, dict) or not geometry
            or any(type(n) is not int or n <= 0 for n in geometry.values()) or sum(geometry.values()) != count):
        raise ValueError('Complete active math-policy lifetime/restoration evidence required')


def cleanup_complete(rows):
    if not rows or any(row.get('volumes_deleted') is not False for row in rows):
        return False
    return (rows[-1].get('status') == 'removed' or
        len(rows) >= 2 and [row.get('status') for row in rows[-2:]] == ['absent', 'absent'])


def host_strict(context, admitted):
    from drivebench.config import digest
    spec_path = project_file(STRICT_SPEC)
    if sha(spec_path) != STRICT_SPEC_SHA or sha(project_file(LAUNCHER)) != LAUNCHER_SHA:
        raise ValueError('Frozen strict host launch specification changed')
    spec = read(spec_path)
    path = project_file(spec['metadata'])
    host = read(path)
    cleanup = host.get('cleanup', [])
    if (host.get('status') != 'completed' or host.get('exit_code') != 0
            or host.get('source_unchanged') is not True or host.get('inputs_unchanged') is not True
            or host.get('specification_sha256') != STRICT_SPEC_SHA or host.get('launcher_sha256') != LAUNCHER_SHA
            or host.get('runtime_identity') != admitted['runtime_identity']
            or host.get('allocation_deadline_epoch') != DEADLINE or host.get('client_stop_error') is not None
            or host.get('source_identity') != SOURCE or digest(host.get('source_files')) != SOURCE
            or host.get('input_file_hashes') != spec['expected_file_hashes']
            or not cleanup_complete(cleanup)):
        raise ValueError('Successful strict host completion and exact-container cleanup required')
    check_files(spec['expected_file_hashes'])
    bound = bindings([spec_path, path, LAUNCHER])
    bound.update({str(project_file(p)): h for p, h in spec['expected_file_hashes'].items()})
    return host, bound


def strict_prerequisite(context, admitted):
    import torch
    rt, base = context.runtime, context.base
    policy = rt.policy_gate(context.original, admitted, context.ab, context.encoder, base)
    path = rt.VALIDATE_DIR / 'strict-all.json'
    completion_path = rt.VALIDATE_DIR / 'completed.json'
    completion = read(completion_path)
    result = rt.strict_result(path, context.original, admitted, base)
    rationale = rt.ab_evidence(context.ab, base)
    required = rt.input_bindings(rt.CONFIG_PATH, context.original, context.ab, context.encoder, base)
    required.update(rationale['artifact_sha256'])
    required.update(rt.file_map([context.original['algorithm_policy'][k] for k in ('verification', 'lock')], base))
    if (completion.get('type') != 'a100_math_sdpa_validate_attempt_v1' or completion.get('status') != 'completed'
            or completion.get('source_identity') != SOURCE or completion.get('producer_sha256') != RUNTIME_SHA
            or completion.get('raw_validator_exit_code') != 0
            or completion.get('validator_identity_function_restored') is not True
            or completion.get('inputs_and_source_unchanged') is not True
            or completion.get('platform_unchanged_after_restoration') is not True
            or completion.get('original_default_runtime_strict_pass_claimed') is not False
            or completion.get('headline_eligible') is not False or completion.get('quality_validated') is not False
            or completion.get('platform_gate') != admitted or completion.get('policy_gate') != policy
            or completion.get('algorithm_policy') != rt.declaration(base)
            or completion.get('allocation_deadline_epoch') != DEADLINE
            or completion.get('development_rationale') != rationale
            or completion.get('strict_result') != result or completion.get('input_sha256') != required
            or result.get('raw_report_passed') is not True or result.get('all_six_checks_passed') is not True):
        raise ValueError('Passed complete math-runtime strict proof and its exact input inventory required')
    lifecycle(completion.get('policy_execution', {}), context.ab.snapshot(torch, base))
    rt.recheck(required, base)
    report = read(path)
    if (report.get('request_id') != context.verifier.SCENE_REQUEST or report.get('passed') is not True
            or set(report.get('runs', {})) != {'upstream', 'refactored', 'isolation', 'uncached', 'cached'}
            or not epoch(completion['started_at']) <= epoch(report['created_at']) <= epoch(completion['finished_at'])):
        raise ValueError('Strict selected-request identity or chronology differs')
    for label, run in report['runs'].items():
        names = {'normal', 'reverse', 'action_change'} if label == 'isolation' else {'normal'}
        if set(run.get('executions', {})) != names:
            raise ValueError('Missing strict execution evidence')
        for row in run['executions'].values():
            stats = row.get('cache_stats', {})
            if (row.get('status') != 'ok' or row.get('fallbacks') or row.get('denoiser_evaluations') != 100
                    or row.get('effective_config', {}).get('actual_batch_sizes') != [1, 1]
                    or stats.get('denoiser_evaluations_complete') is not True or stats.get('oom_attempts') != []):
                raise ValueError('Strict execution was incomplete, nonserial, or used fallback')
    bound = dict(required)
    for name, expected in completion.get('artifact_sha256', {}).items():
        p = Path(name)
        if p.is_absolute() or '..' in p.parts:
            raise ValueError('Unsafe strict artifact path')
        bound[str(project_file(rt.VALIDATE_DIR / p))] = expected
    bound.update(result['artifact_sha256'])
    bound.update(bindings([path, completion_path]))
    host, host_bindings = host_strict(context, admitted)
    if not epoch(host['started_at']) <= epoch(completion['started_at']) < epoch(completion['finished_at']) <= epoch(host['finished_at']):
        raise ValueError('Strict host and payload chronology differ')
    bound.update(host_bindings)
    check_files(bound)
    return {'passed': True, 'source_identity': SOURCE, 'runtime_identity': policy['runtime_identity'],
        'policy_gate': policy, 'strict_report': str(path), 'strict_report_sha256': sha(path),
        'strict_completion_sha256': sha(completion_path), 'strict_elapsed_seconds': elapsed(completion),
        'strict_host_elapsed_seconds': elapsed(host), 'strict_futures': 14,
        'artifact_sha256': bound, 'coverage': 'Selected original development request at50 steps; not smoke-scene or approximation equivalence.'}


def row_checked(row, cfg, expected, *, measured):
    from drivebench.config import validate_run_config
    generation, options = validate_run_config(cfg)
    stats, effective = row.get('cache_stats', {}), row.get('effective_config', {})
    if (row.get('status') != 'ok' or row.get('candidate_count') != 1 or row.get('batch_size') != 1
            or row.get('input_condition') != 'uncached' or row.get('fallbacks') or row.get('failures')
            or row.get('denoiser_evaluations') != generation.steps
            or effective.get('actual_batch_sizes') != [1] or effective.get('requested_batch_size') != 1
            or effective.get('optimizations') != options.to_dict()
            or any(effective.get(k) != v for k, v in generation.to_dict().items())
            or stats.get('oom_attempts') != [] or stats.get('denoiser_evaluations_complete') is not True
            or stats.get('context_hit') is not False
            or any(stats.get(k, 0) for k in ('conditioning_embedding_hits', 'posterior_parameter_hits'))):
        raise ValueError('Failed, incomplete or changed generation workload')
    if row.get('request_id') != expected['request_id'] or row.get('scene_id') != expected['scene_id']:
        raise ValueError('Observed scene/request order differs')
    positive(row.get('latency_s'))
    if cfg['a100_quality_smoke']['variant'] == 'feature2':
        if (stats.get('feature_hits') != 25 or stats.get('feature_misses') != 25
                or stats.get('feature_cache_module') != 'output_blocks.9'
                or stats.get('feature_cache_configuration') != {'module': 'output_blocks.9', 'interval': 2,
                    'candidate_scope': 'single_candidate', 'cfg_layout': 'unconditional_T_then_conditional_T'}):
            raise ValueError('Configured approximate whole-block cache was not exercised')
    elif stats.get('feature_hits') != 0:
        raise ValueError('Eager run reused denoising features')
    if measured:
        candidates = row.get('candidates', [])
        if row.get('repeat') != 0 or len(candidates) != 1:
            raise ValueError('Require exactly one retained candidate per smoke scene')
        c = candidates[0]
        if (c.get('candidate_id') != expected['candidate_id'] or c.get('seed') != expected['seed']
                or c.get('trajectory_sha256') != expected['trajectory_sha256'] or c.get('action_kind') != 'recorded'
                or c.get('conditioning_frames') != 1 or not c.get('frames_path')):
            raise ValueError('Candidate action, seed or output identity differs')


def run_artifacts(output, cfg, expected, admitted):
    import numpy as np
    from drivebench.benchmarks.environment import reference_identity
    from drivebench.config import validate_run_config
    from drivebench.evaluation.io import load_run, output_path
    run = load_run(output)
    proof = read(project_file(cfg['numerical_validation']))
    if read(output / 'model_identity.json') != proof['runs']['refactored']['model_identity']:
        raise ValueError('Actual model identity differs from the separately proved eager baseline')
    expected_data = {**cfg['data'], 'manifest_sha256': MANIFEST_PINS['data/manifests/smoke.jsonl'],
        'manifest_metadata': read(project_file('data/manifests/smoke.jsonl.meta.json'))}
    generation, options = validate_run_config(cfg)
    if (len(run.measurements) != 4 or run.summary.get('backend') != 'vista' or run.summary.get('failures') != 0 or run.summary.get('run_failure')
            or run.summary.get('headline_eligible') is not False or run.summary.get('profiled') is not False
            or run.summary.get('recompiled_during_measurement') is not False
            or run.summary.get('completed_futures') != 4 or run.summary.get('successful_requests') != 4
            or run.summary.get('numerical_validation', {}).get('passed') is not False
            or read(output / 'numerical_gate.json') != run.summary['numerical_validation']
            or run.config.get('generation') != generation.to_dict() or run.config.get('optimizations') != options.to_dict()
            or any(run.config.get(k) != cfg.get(k) for k in ('model', 'benchmark', 'algorithm_policy', 'a100_quality_smoke', 'numerical_validation'))
            or run.config.get('data') != expected_data
            or reference_identity(run.environment) != admitted['runtime_identity']):
        raise ValueError('Completed generation artifacts or visible legacy gates differ')
    cold = read(output / 'cold_start.json')
    if cold.get('status') != 'ready' or len(cold.get('warmups', [])) != 1:
        raise ValueError('Exactly one complete eager warmup required')
    row_checked(cold['warmups'][0], cfg, expected[0], measured=False)
    positive(cold['model_load_s'])
    for index, (row, wanted) in enumerate(zip(run.measurements, expected)):
        if row.get('request_index') != index:
            raise ValueError('Measured request index differs')
        row_checked(row, cfg, wanted, measured=True)
        p = output_path(run, row['candidates'][0]['frames_path'])
        if p.is_symlink():
            raise ValueError('Retained output cannot be a symlink')
        array = np.load(p, mmap_mode='r', allow_pickle=False)
        if array.shape != SHAPE or array.dtype != np.float32:
            raise ValueError('Retained video shape/dtype differs')
        for frame in array:
            if not np.isfinite(frame).all() or frame.min() < 0 or frame.max() > 1:
                raise ValueError('Retained video is nonfinite or outside the model output range')
    files = {str(p.relative_to(output)): sha(p) for p in output.rglob('*') if p.is_file() and p.name != 'completed.json'}
    return {'artifact_sha256': files, 'saved_outputs': 4, 'new_frames_each': 24,
        'model_load_s': cold['model_load_s'],
        'max_observed_request_s': max(r['latency_s'] for r in [*cold['warmups'], *run.measurements]),
        'legacy_numerical_passed': False, 'headline_eligible': False, 'quality_preserving': False}


def completed_baseline(context, admitted, strict, cfg, expected):
    import torch
    output = ROOT / RESULT / 'runs/eager50'
    done_path = project_file(output / 'completed.json')
    done = read(done_path)
    actual = run_artifacts(output, cfg, expected, admitted)
    if (done.get('type') != 'a100_math_quality_generation_v1' or done.get('status') != 'completed'
            or done.get('variant') != 'eager50' or done.get('producer_sha256') != sha(__file__)
            or done.get('source_identity') != SOURCE or done.get('strict_prerequisite') != strict
            or done.get('platform_gate') != admitted or done.get('result') != actual
            or done.get('inputs_and_source_unchanged') is not True or done.get('platform_restored') is not True
            or done.get('quality_preserving') is not False or done.get('headline_eligible') is not False):
        raise ValueError('Complete unchanged eager50 quality generation is required before approximations')
    lifecycle(done.get('policy_execution', {}), context.ab.snapshot(torch, context.base))
    _, _, observed = smoke_inputs(cfg)
    required = {**strict['artifact_sha256'], **observed,
        **bindings([__file__, REL + '/configs/eager50.json'])}
    admission_path = project_file(done['admission_path'])
    old_admission = read(admission_path)
    if (done.get('admission_sha256') != sha(admission_path) or old_admission.get('input_sha256') != required
            or old_admission.get('variant') != 'eager50' or old_admission.get('producer_sha256') != sha(__file__)
            or old_admission.get('strict_prerequisite') != strict or old_admission.get('time_bound') != done.get('time_bound')
            or done.get('algorithm_policy') != context.runtime.declaration(context.base)
            or done.get('allocation_deadline_epoch') != DEADLINE or done.get('legacy_numerical_passed') is not False
            or done.get('variant_exact_agreement_claimed') is not False or done.get('h100_reference_eligible') is not False):
        raise ValueError('Baseline admission, policy or false-claim provenance changed')
    required.update(bindings([admission_path]))
    if done.get('input_sha256') != required:
        raise ValueError('Baseline generation input inventory is incomplete or changed')
    check_files(done['input_sha256'])
    bound = bindings([done_path])
    bound.update({str(output / p): h for p, h in actual['artifact_sha256'].items()})
    return {'result': actual, 'elapsed_seconds': elapsed(done), 'artifact_sha256': bound}


def estimate(strict_seconds, baseline=None):
    if baseline is None:
        # Strict time includes14 futures, five model loads and evidence export.
        workload = 5 * positive(strict_seconds) / 14
        method = '1.5 times5/14 of actual completed strict elapsed; no measured per-request claim'
    else:
        result = baseline['result']
        workload = 5 * max(positive(result['max_observed_request_s']), positive(baseline['elapsed_seconds']) / 5)
        workload += positive(result['model_load_s'])
        method = '1.5 times five worst observed eager50 requests plus startup; no assumed approximation speedup'
    seconds = math.ceil(600 + 1.5 * workload + 300)
    if not 900 <= seconds <= MAX_JOB_SECONDS:
        raise ValueError('Conservative complete-variant estimate exceeds the one-hour job ceiling')
    return {'outer_timeout_seconds': seconds, 'internal_timeout_seconds': seconds - 60,
        'prechecks_reserved_seconds': 600, 'postchecks_reserved_seconds': 300,
        'archive_evaluation_cleanup_reserve_seconds': RESERVE_SECONDS,
        'method': method, 'empirical_bound_guaranteed': False,
        'rate_usd_per_hour': RATE_USD_PER_HOUR,
        'reserved_compute_and_disk_usd': (seconds + RESERVE_SECONDS) / 3600 * RATE_USD_PER_HOUR}


def admission_limits(bound, *, now=None, free_bytes=None):
    now = time.time() if now is None else now
    if now + bound['outer_timeout_seconds'] + RESERVE_SECONDS >= DEADLINE:
        raise ValueError('Complete variant and archive/evaluation/cleanup reserve do not fit the allocation deadline')
    needed = 4 * (ARRAY_BYTES + 1024) + 10 * 1024**3
    if free_bytes is None:
        free_bytes = shutil.disk_usage(ROOT / 'outputs').free
    if free_bytes < needed:
        raise ValueError('Insufficient disk reserve for all four arrays, evaluation exports and8GiB free')
    return {'required_free_bytes': needed, 'observed_free_bytes': free_bytes}


def prepare(variant, expected_sha):
    context = dependencies(expected_sha)
    path, config = config_checked(context, variant)
    records, wanted, inputs = smoke_inputs(config)
    admitted = context.verifier.gate(context.original, DEADLINE)
    strict = strict_prerequisite(context, admitted)
    inputs.update(strict['artifact_sha256'])
    inputs.update(bindings([__file__, path]))
    baseline = None
    if variant != 'eager50':
        _, baseline_config = config_checked(context, 'eager50')
        baseline = completed_baseline(context, admitted, strict, baseline_config, wanted)
        inputs.update(baseline['artifact_sha256'])
    bound = estimate(strict['strict_elapsed_seconds'], baseline)
    return SimpleNamespace(context=context, path=path, config=config, records=records, wanted=wanted,
        admitted=admitted, strict=strict, inputs=inputs, baseline=baseline, bound=bound)


def admit(variant, output, expected_sha):
    output = Path(output)
    if output.exists() or not output.resolve().is_relative_to((ROOT / REL).resolve()):
        raise ValueError('Write a fresh admission only inside the new auxiliary directory')
    started = time.monotonic()
    plan = prepare(variant, expected_sha)
    disk = admission_limits(plan.bound)
    value = {'type': 'a100_math_quality_admission_v1', 'created_at': plan.context.base.stamp(),
        'variant': variant, 'producer_sha256': expected_sha, 'source_identity': SOURCE,
        'allocation_deadline_epoch': DEADLINE, 'platform_gate': plan.admitted,
        'strict_prerequisite': plan.strict, 'input_sha256': plan.inputs, 'time_bound': plan.bound,
        'disk': disk, 'admission_precheck_elapsed_seconds': time.monotonic() - started,
        'automatic_launch_authorized': False, 'scope': SCOPE}
    check_files(plan.inputs)
    plan.context.base.save(output, value)
    print(json.dumps({'admission': str(output), 'sha256': sha(output), 'variant': variant,
                      'time_bound': plan.bound, 'automatic_launch_authorized': False}))
    return 0


def run(variant, admission_path, output, expected_sha):
    admission_path = project_file(admission_path)
    admission = read(admission_path)
    if (admission.get('type') != 'a100_math_quality_admission_v1' or admission.get('variant') != variant
            or admission.get('producer_sha256') != expected_sha or admission.get('source_identity') != SOURCE
            or admission.get('allocation_deadline_epoch') != DEADLINE
            or not 0 <= time.time() - epoch(admission.get('created_at')) <= 300):
        raise ValueError('Fresh unchanged same-variant admission required')
    outer = admission['time_bound']['outer_timeout_seconds']
    if type(outer) is not int or not 900 <= outer <= MAX_JOB_SECONDS:
        raise ValueError('Invalid admitted job bound')
    output = Path(output)
    if output.resolve() != (ROOT / RESULT / 'runs' / variant).resolve() or output.exists():
        raise ValueError('Use the exact fresh variant output directory')
    def timeout(_signum, _frame):
        raise TimeoutError('Complete-variant generation time bound expired')
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(outer - 60)
    value = None
    try:
        plan = prepare(variant, expected_sha)
        if (admission.get('platform_gate') != plan.admitted or admission.get('strict_prerequisite') != plan.strict
                or admission.get('input_sha256') != plan.inputs or admission.get('time_bound') != plan.bound):
            raise ValueError('Current proof/input/runtime or admission estimate changed')
        admission_limits(plan.bound)
        output.mkdir(parents=True)
        inputs = {**plan.inputs, **bindings([admission_path])}
        value = {'type': 'a100_math_quality_generation_v1', 'variant': variant, 'status': 'running',
            'started_at': plan.context.base.stamp(), 'producer_sha256': expected_sha, 'source_identity': SOURCE,
            'input_sha256': inputs, 'platform_gate': plan.admitted, 'strict_prerequisite': plan.strict,
            'algorithm_policy': plan.context.runtime.declaration(plan.context.base),
            'admission_path': str(admission_path), 'admission_sha256': sha(admission_path), 'time_bound': plan.bound,
            'allocation_deadline_epoch': DEADLINE, 'scope': SCOPE,
            'legacy_numerical_passed': False, 'headline_eligible': False, 'quality_preserving': False,
            'h100_reference_eligible': False, 'evaluation_pending': True,
            'variant_exact_agreement_claimed': False, 'policy_execution': {}}
        plan.context.base.save(output / 'attempt.json', value)
        import torch
        from drivebench.benchmark import run_benchmark
        from drivebench.benchmarks.environment import collect_environment, reference_identity
        with plan.context.runtime.activate(torch, plan.context.ab, plan.context.base, value['policy_execution']):
            destination, summary = run_benchmark(plan.config, output, profiled=False)
            if Path(destination).resolve() != output.resolve():
                raise ValueError('Benchmark wrote an unexpected destination')
        lifecycle(value['policy_execution'], plan.context.ab.snapshot(torch, plan.context.base))
        if reference_identity(collect_environment()) != plan.admitted['runtime_identity']:
            raise ValueError('Original platform identity changed after generation')
        value['result'] = run_artifacts(output, plan.config, plan.wanted, plan.admitted)
        if strict_prerequisite(plan.context, plan.admitted) != plan.strict:
            raise ValueError('Strict provenance changed during generation')
        check_files(inputs)
        value.update(status='completed', inputs_and_source_unchanged=True, platform_restored=True)
    except BaseException as error:
        if value is None:
            raise
        value.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
    finally:
        if value is not None:
            value['finished_at'] = plan.context.base.stamp()
            value['partial_artifact_sha256'] = {str(p.relative_to(output)): sha(p) for p in output.rglob('*') if p.is_file()}
            plan.context.base.save(output / 'completed.json', value)
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    print(json.dumps({'status': value['status'], 'variant': variant, 'output': str(output), 'headline_eligible': False}))
    return 0 if value['status'] == 'completed' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['admit', 'run'])
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--admission')
    args = parser.parse_args()
    if args.command == 'run' and not args.admission:
        parser.error('run requires --admission')
    if args.command == 'admit':
        raise SystemExit(admit(args.variant, Path(args.output), args.expected_sha256))
    raise SystemExit(run(args.variant, args.admission, args.output, args.expected_sha256))
