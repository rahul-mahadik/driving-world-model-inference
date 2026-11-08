#!/usr/bin/env python3
"""Bounded, separately declared A100 math-runtime full-development baseline only."""
from __future__ import annotations
import argparse
import copy
import importlib.util
import math
from pathlib import Path
import shutil
import signal
import sys
import time
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
REL = 'outputs/host/a100-math-development-v1'
RESULT = 'outputs/a100-math-development-v1/runs/eager50'
QUALITY = 'outputs/host/a100-math-quality-v1/quality.py'
QUALITY_SHA = '233b18a2d0e64f937978de9b65db5a03c86f175fa04fa29e0ecc6c0885422724'
MANIFEST = 'data/manifests/development.jsonl'
MANIFEST_SHA = 'a3c6328d9f0d0b94149820f626db17e024e6fbc0e5523fe5ee19aa3316befbff'
PINS = {
    MANIFEST: MANIFEST_SHA,
    MANIFEST + '.meta.json': 'f07de92cf445a4e69850a36f206f7a1651798b997ff47c8f3794d2afcde9bd32',
    'outputs/full-study-v1/manifests/development.jsonl': MANIFEST_SHA,
    'outputs/full-study-v1/plan.json': 'a15bac95d6cd0d4376aa9bbf07cefb392d3cf91ffce9ee2ee7ef74d8ee121c1e',
}
COUNT, WARMUPS = 32, 3
OUTER, INNER, RESERVE = 4800, 4680, 1200
FREE_FLOOR, WORKING_MARGIN = 12 * 1024**3, 2 * 1024**3
ARRAY_FILE_BYTES = 176947328
BASELINE_COMPLETION = 'outputs/a100-math-quality-v1/runs/eager50/completed.json'
BASELINE_COMPLETION_SHA = 'a911682a6403f63cb9f097d7b61fab83382cbf18864acfb9faee028bd56fcd55'
SCOPE = ('Separate A100 math-SDPA companion:32 fixed development scenes, recorded actions, '
         'one candidate,50 steps,3 warmups,32 retained arrays. Generation evidence only. '
         'Original18-variant H100 study unchanged; no full-scene numerical agreement, '
         'quality acceptance, threshold calibration, speedup, matched latency or headline claim.')


class DevelopmentTimeout(BaseException):
    """Escape benchmark request-level Exception handlers; host bound remains outer guard."""


def quality():
    import hashlib
    path = ROOT / QUALITY
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != QUALITY_SHA:
        raise ValueError('Frozen quality primitive changed')
    spec = importlib.util.spec_from_file_location('a100_development_frozen_quality', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected_config(original, q):
    config = q.expected_config(original, 'eager50')
    config['name'] = 'a100-math-development-v1-eager50'
    config['output'] = '/workspace/' + RESULT
    config['data']['manifest'] = '/workspace/' + MANIFEST
    config['benchmark'].update(requests=COUNT, warmup_requests=WARMUPS,
        max_warmup_requests=WARMUPS, save_outputs=COUNT)
    config['a100_pilot']['scope'] = SCOPE
    # Keep the fixed eager50 discriminator consumed by the frozen row validator;
    # this metadata explicitly replaces its old four-scene scope.
    config['a100_quality_smoke'].update(scope=SCOPE,
        strict_coverage='Original selected development request only; not all32 scenes.',
        retained_compatibility_field=True)
    config['a100_development_companion'] = {'scope': SCOPE,
        'parent_plan_sha256': PINS['outputs/full-study-v1/plan.json'],
        'completed_smoke_outputs_preserved': True, 'full_manifest_fresh_run': True,
        'reuse_smoke_timings': False, 'retire_outputs': False, 'evaluation_pending': True}
    return config


def config_checked(q, context):
    from drivebench.config import load_config, validate_run_config
    path = q.project_file(REL + '/configs/eager50.json')
    config = load_config(path)
    if config != expected_config(context.original, q):
        raise ValueError('Exact32-scene eager50 companion configuration required')
    validate_run_config(config)
    return path, config


def development_inputs(q, config):
    from drivebench.config import digest
    from drivebench.data.manifest import load_manifest, inference_request, verify_input_checksums
    q.check_files(PINS)
    records = load_manifest(q.project_file(MANIFEST))
    meta = q.read(q.project_file(MANIFEST + '.meta.json'))
    if (len(records) != COUNT or len({r['scene_id'] for r in records}) != COUNT
            or len({r['request_id'] for r in records}) != COUNT
            or meta.get('partition_role') != 'development' or meta.get('split') != 'development'
            or meta.get('clip_count') != COUNT or meta.get('scene_count') != COUNT
            or meta.get('manifest_sha256') != MANIFEST_SHA
            or meta.get('scene_ids_sha256') != digest(sorted({r['scene_id'] for r in records}))
            or config['data']['manifest'] != '/workspace/' + MANIFEST):
        raise ValueError('Frozen32-scene development-only provenance differs')
    root = ROOT / 'data/nuscenes'
    bound, wanted = q.bindings(PINS), []
    for record in records:
        # Hash all25 images for provenance; only observed_frames reaches inference.
        verify_input_checksums(record, root, include_future=True)
        if len(record['observed_frames']) != 1 or len(record['future_frames']) != 24:
            raise ValueError('Fixed one-observation24-target-frame geometry required')
        bound.update(q.bindings(root / p for p in record['observed_frames'] + record['future_frames']))
        request = inference_request(record, root, candidate_count=1)
        if len(request['observed_frames']) != 1 or request['action_kinds'] != ['recorded']:
            raise ValueError('Recorded single-candidate trajectory required')
        wanted.append({'request_id': request['request_id'], 'scene_id': request['scene_id'],
            'candidate_id': request['candidate_ids'][0], 'seed': request['seeds'][0],
            'trajectory_sha256': digest(request['trajectories'][0].tolist())})
    return records, wanted, bound


def estimate(q, baseline):
    result = baseline['result']
    maximum = q.positive(result['max_observed_request_s'])
    model_load = q.positive(result['model_load_s'])
    empirical = (COUNT + WARMUPS) * maximum + model_load + 600 + 300
    if not math.isfinite(empirical) or empirical >= INNER:
        raise ValueError('Observed baseline cannot support the fixed inner time bound')
    return {'outer_timeout_seconds': OUTER, 'internal_timeout_seconds': INNER,
        'archive_cleanup_reserve_seconds': RESERVE,
        'observed_max_request_including_warmup_s': maximum,
        'observed_model_load_s': model_load, 'empirical_estimate_seconds': empirical,
        'prechecks_reserved_seconds': 600, 'postchecks_reserved_seconds': 300,
        'method': '35 times completed smoke max request including warmup plus model load and900s checks',
        'empirical_bound_guaranteed': False,
        'rate_usd_per_hour_planning_only': q.RATE_USD_PER_HOUR,
        'reserved_compute_and_disk_usd_planning_only': (OUTER + RESERVE) / 3600 * q.RATE_USD_PER_HOUR,
        'funding_guard': 'Fresh provider quote, balance and active unchanged allocation guard checked by host operator before launch.'}


def admission_limits(q, *, now=None, free_bytes=None):
    now = time.time() if now is None else now
    if not math.isfinite(now) or now + OUTER + RESERVE >= q.DEADLINE:
        raise ValueError('Fixed job and archive/cleanup reserve exceed allocation deadline')
    needed = COUNT * ARRAY_FILE_BYTES + FREE_FLOOR + WORKING_MARGIN
    free_bytes = shutil.disk_usage(ROOT / 'outputs').free if free_bytes is None else free_bytes
    if type(free_bytes) is not int or free_bytes < needed:
        raise ValueError('Insufficient disk for32 arrays plus12GiB floor and2GiB working margin')
    return {'required_free_bytes': needed, 'observed_free_bytes': free_bytes,
        'array_file_bytes': ARRAY_FILE_BYTES, 'array_count': COUNT,
        'preserved_free_floor_bytes': FREE_FLOOR, 'working_margin_bytes': WORKING_MARGIN}


def prepare(expected_sha):
    q = quality()
    if q.sha(__file__) != expected_sha:
        raise ValueError('Reviewed companion helper hash differs')
    context = q.dependencies(QUALITY_SHA)
    path, config = config_checked(q, context)
    records, wanted, inputs = development_inputs(q, config)
    admitted = context.verifier.gate(context.original, q.DEADLINE)
    strict = q.strict_prerequisite(context, admitted)
    _, smoke_config = q.config_checked(context, 'eager50')
    _, smoke_wanted, _ = q.smoke_inputs(smoke_config)
    if q.sha(q.project_file(BASELINE_COMPLETION)) != BASELINE_COMPLETION_SHA:
        raise ValueError('Measured smoke timing evidence changed')
    baseline = q.completed_baseline(context, admitted, strict, smoke_config, smoke_wanted)
    inputs.update(strict['artifact_sha256'])
    inputs.update(baseline['artifact_sha256'])
    inputs.update(q.bindings([__file__, QUALITY, path]))
    q.check_files(inputs)
    return NS(q=q, context=context, path=path, config=config, records=records, wanted=wanted,
        admitted=admitted, strict=strict, baseline=baseline, inputs=inputs, bound=estimate(q, baseline))


def run_artifacts(plan, output):
    import numpy as np
    from drivebench.benchmarks.environment import reference_identity
    from drivebench.config import validate_run_config
    from drivebench.evaluation.io import load_run, output_path
    q, cfg = plan.q, plan.config
    run = load_run(output)
    proof = q.read(q.project_file(cfg['numerical_validation']))
    if q.read(output / 'model_identity.json') != proof['runs']['refactored']['model_identity']:
        raise ValueError('Model identity differs from proved eager baseline')
    expected_data = {**cfg['data'], 'manifest_sha256': MANIFEST_SHA,
        'manifest_metadata': q.read(q.project_file(MANIFEST + '.meta.json'))}
    generation, options = validate_run_config(cfg)
    if (len(run.measurements) != COUNT or run.summary.get('backend') != 'vista'
            or run.summary.get('failures') != 0 or run.summary.get('run_failure')
            or run.summary.get('headline_eligible') is not False or run.summary.get('profiled') is not False
            or run.summary.get('recompiled_during_measurement') is not False
            or run.summary.get('completed_futures') != COUNT or run.summary.get('successful_requests') != COUNT
            or run.summary.get('numerical_validation', {}).get('passed') is not False
            or q.read(output / 'numerical_gate.json') != run.summary['numerical_validation']
            or run.config.get('generation') != generation.to_dict() or run.config.get('optimizations') != options.to_dict()
            or any(run.config.get(k) != cfg.get(k) for k in ('model', 'benchmark', 'algorithm_policy',
                'a100_quality_smoke', 'a100_development_companion', 'numerical_validation'))
            or run.config.get('data') != expected_data
            or reference_identity(run.environment) != plan.admitted['runtime_identity']):
        raise ValueError('Completed32-scene artifacts or visible legacy gates differ')
    cold = q.read(output / 'cold_start.json')
    if cold.get('status') != 'ready' or len(cold.get('warmups', [])) != WARMUPS:
        raise ValueError('Exactly3 complete eager warmups required')
    for index, row in enumerate(cold['warmups']):
        q.row_checked(row, cfg, plan.wanted[index], measured=False)
    q.positive(cold['model_load_s'])
    saved = set()
    for index, (row, wanted) in enumerate(zip(run.measurements, plan.wanted)):
        if row.get('request_index') != index:
            raise ValueError('Development request order differs')
        q.row_checked(row, cfg, wanted, measured=True)
        p = output_path(run, row['candidates'][0]['frames_path'])
        if p.is_symlink() or not p.resolve().is_relative_to(output.resolve()) or p.resolve() in saved:
            raise ValueError('Require32 distinct project-owned retained arrays')
        saved.add(p.resolve())
        array = np.load(p, mmap_mode='r', allow_pickle=False)
        if array.shape != q.SHAPE or array.dtype != np.float32 or p.stat().st_size != ARRAY_FILE_BYTES:
            raise ValueError('Retained video shape/dtype/file size differs')
        for frame in array:
            if not np.isfinite(frame).all() or frame.min() < 0 or frame.max() > 1:
                raise ValueError('Retained video is nonfinite or outside model output range')
    if {p.resolve() for p in output.rglob('*.npy')} != saved:
        raise ValueError('Unexpected extra or missing saved arrays')
    files = {str(p.relative_to(output)): q.sha(p) for p in output.rglob('*') if p.is_file() and p.name != 'completed.json'}
    return {'artifact_sha256': files, 'saved_outputs': COUNT, 'new_frames_each': 24,
        'legacy_numerical_passed': False, 'headline_eligible': False, 'quality_preserving': False,
        'scope': SCOPE}


def admit(output, expected_sha):
    output = Path(output)
    if output.exists() or output.is_symlink() or output.resolve() != (ROOT / REL / 'admission-01.json').resolve():
        raise ValueError('Use the exact fresh development admission path')
    started = time.monotonic()
    plan = prepare(expected_sha)
    value = {'type': 'a100_math_development_admission_v1', 'created_at': plan.context.base.stamp(),
        'producer_sha256': expected_sha, 'source_identity': plan.q.SOURCE,
        'allocation_deadline_epoch': plan.q.DEADLINE, 'platform_gate': plan.admitted,
        'strict_prerequisite': plan.strict, 'smoke_baseline': plan.baseline,
        'input_sha256': plan.inputs, 'time_bound': plan.bound, 'disk': admission_limits(plan.q),
        'admission_precheck_elapsed_seconds': time.monotonic() - started,
        'automatic_launch_authorized': False, 'scope': SCOPE}
    plan.q.check_files(plan.inputs)
    plan.context.base.save(output, value)
    print({'admission': str(output), 'sha256': plan.q.sha(output), 'automatic_launch_authorized': False})
    return 0


def run(admission_path, output, expected_sha):
    q = quality()
    admission_path = q.project_file(admission_path)
    admission = q.read(admission_path)
    if (admission_path.resolve() != (ROOT / REL / 'admission-01.json').resolve()
            or admission.get('type') != 'a100_math_development_admission_v1'
            or admission.get('producer_sha256') != expected_sha or admission.get('source_identity') != q.SOURCE
            or admission.get('allocation_deadline_epoch') != q.DEADLINE
            or not 0 <= time.time() - q.epoch(admission.get('created_at')) <= 300):
        raise ValueError('Fresh unchanged development admission required')
    output = Path(output)
    if output.resolve() != (ROOT / RESULT).resolve() or output.exists() or output.is_symlink():
        raise ValueError('Use exact fresh development output')
    def timeout(_signum, _frame):
        raise DevelopmentTimeout('Fixed development inner time bound expired')
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(INNER)
    value = None
    try:
        plan = prepare(expected_sha)
        if any(admission.get(k) != v for k, v in {
            'platform_gate': plan.admitted, 'strict_prerequisite': plan.strict,
            'smoke_baseline': plan.baseline, 'input_sha256': plan.inputs, 'time_bound': plan.bound}.items()):
            raise ValueError('Current proof/input/runtime or admission estimate changed')
        disk = admission_limits(q)
        output.mkdir(parents=True)
        inputs = {**plan.inputs, **q.bindings([admission_path])}
        value = {'type': 'a100_math_development_generation_v1', 'status': 'running',
            'started_at': plan.context.base.stamp(), 'producer_sha256': expected_sha, 'source_identity': q.SOURCE,
            'input_sha256': inputs, 'platform_gate': plan.admitted, 'strict_prerequisite': plan.strict,
            'smoke_baseline': plan.baseline, 'algorithm_policy': plan.context.runtime.declaration(plan.context.base),
            'admission_path': str(admission_path), 'admission_sha256': q.sha(admission_path), 'time_bound': plan.bound,
            'allocation_deadline_epoch': q.DEADLINE, 'scope': SCOPE, 'disk_at_start': disk,
            'legacy_numerical_passed': False, 'headline_eligible': False, 'quality_preserving': False,
            'h100_reference_eligible': False, 'evaluation_pending': True,
            'all_development_scenes_exact_agreement_claimed': False, 'policy_execution': {}}
        plan.context.base.save(output / 'attempt.json', value)
        import torch
        from drivebench.benchmark import run_benchmark
        from drivebench.benchmarks.environment import collect_environment, reference_identity
        with plan.context.runtime.activate(torch, plan.context.ab, plan.context.base, value['policy_execution']):
            destination, _ = run_benchmark(plan.config, output, profiled=False)
            if Path(destination).resolve() != output.resolve():
                raise ValueError('Benchmark wrote an unexpected destination')
        q.lifecycle(value['policy_execution'], plan.context.ab.snapshot(torch, plan.context.base))
        if reference_identity(collect_environment()) != plan.admitted['runtime_identity']:
            raise ValueError('Original platform identity changed after generation')
        value['result'] = run_artifacts(plan, output)
        if q.strict_prerequisite(plan.context, plan.admitted) != plan.strict:
            raise ValueError('Strict provenance changed during generation')
        q.check_files(inputs)
        if shutil.disk_usage(ROOT / 'outputs').free < FREE_FLOOR:
            raise ValueError('Original12GiB disk floor not preserved')
        value.update(status='completed', inputs_and_source_unchanged=True, platform_restored=True)
    except BaseException as error:
        if value is None:
            raise
        value.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
    finally:
        if value is not None:
            value['finished_at'] = plan.context.base.stamp()
            value['partial_artifact_sha256'] = {str(p.relative_to(output)): q.sha(p) for p in output.rglob('*') if p.is_file()}
            plan.context.base.save(output / 'completed.json', value)
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    print({'status': value['status'], 'output': str(output), 'headline_eligible': False})
    return 0 if value['status'] == 'completed' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['admit', 'run'])
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--admission')
    args = parser.parse_args()
    if args.command == 'run' and not args.admission:
        parser.error('run requires --admission')
    raise SystemExit(admit(args.output, args.expected_sha256) if args.command == 'admit'
        else run(args.admission, args.output, args.expected_sha256))
