#!/usr/bin/env python3
"""Bounded CPU recorded-future diagnostics for the completed 32-scene A100 baseline."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
REL = 'outputs/host/a100-math-development-eval-v1'
DEST = 'outputs/a100-math-development-eval-v1/eager50-recorded-future'
DEVELOPMENT = 'outputs/host/a100-math-development-v1/development.py'
DEVELOPMENT_SHA = 'b87a909c988384f41e33fef555e00d8266241645cd09570a5e8f9369731bd8d1'
GENERATION_SPEC = 'outputs/host/a100-math-development-v1/run-01-spec.json'
GENERATION_SPEC_SHA = 'caa01d0b5bc004867f63925087e7c0f57023b98bf5505cfb3956123a7af469d6'
ADMISSION = 'outputs/host/a100-math-development-v1/admission-01.json'
ADMISSION_SHA = 'a2a6e3bb12bbeda15795e922d22a027a33e6a9cf2e4ed9402311562c9debc7cb'
SOURCE = 'd8a32813492c0939a0a91573a945c914938972419946065e6822f6209fa470cb'
DEADLINE = 1793168129.682277
COUNT = 32
OUTER, INNER, RESERVE = 1800, 1680, 4200
FREE_FLOOR, WORKING_MARGIN = 12 * 1024**3, 256 * 1024**2
METRICS = ('rmse', 'temporal_change_error', 'freeze_fraction_increase', 'temporal_derivative_mae')
SCOPE = ('Descriptive recorded-future diagnostics for all32 fixed development scenes from the separate '
         'A100 math-SDPA eager50 companion. CPU NumPy/Pillow calculations; observed frame excluded. '
         'Temporal differences and thresholded global-pixel-change freezes are frame-indexed proxies, '
         'not time-normalized motion, literal freeze, action adherence or safety. No LPIPS, review export, '
         'quality acceptance, tolerance calibration, timing, speedup, H100 or headline claim.')


class EvaluationTimeout(BaseException):
    """Escape any nested per-item Exception handlers; the host has an outer timeout."""


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError('Expected a JSON object')
    return value


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def development(expected):
    if (sha(__file__) != expected or (ROOT / DEVELOPMENT).is_symlink()
            or sha(ROOT / DEVELOPMENT) != DEVELOPMENT_SHA
            or sha(ROOT / GENERATION_SPEC) != GENERATION_SPEC_SHA
            or sha(ROOT / ADMISSION) != ADMISSION_SHA):
        raise ValueError('Reviewed evaluator, development helper, generation spec or admission changed')
    spec = importlib.util.spec_from_file_location('a100_development_eval_frozen', ROOT / DEVELOPMENT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def limits(*, now=None, free=None):
    now = time.time() if now is None else now
    if type(now) not in (int, float) or not math.isfinite(now) or now + OUTER + RESERVE >= DEADLINE:
        raise ValueError('Fixed evaluation and backup/cleanup reserve exceed allocation deadline')
    free = shutil.disk_usage(ROOT / 'outputs').free if free is None else free
    if type(free) is not int or free < FREE_FLOOR + WORKING_MARGIN:
        raise ValueError('Require12GiB free floor plus256MiB diagnostic working margin')
    return {'outer_timeout_seconds': OUTER, 'internal_timeout_seconds': INNER,
        'separate_archive_cleanup_reserve_seconds': RESERVE, 'free_bytes': free,
        'preserved_free_floor_bytes': FREE_FLOOR, 'working_margin_bytes': WORKING_MARGIN,
        'empirical_bound_guaranteed': False,
        'funding_admission': 'Host operator must freshly verify provider quote, balance and unchanged armed guard before launch.'}


def cpu_only(torch):
    if torch.cuda.is_initialized() is not False:
        raise ValueError('CPU diagnostics must not initialize a CUDA context')


def generation_host(d, plan, done):
    from drivebench.config import digest
    q = plan.q
    path = q.project_file(GENERATION_SPEC)
    if sha(path) != GENERATION_SPEC_SHA:
        raise ValueError('Frozen generation host specification changed')
    spec = read(path)
    host_path = q.project_file(spec['metadata']); host = read(host_path)
    expected_args = [DEVELOPMENT, 'run', '--expected-sha256', DEVELOPMENT_SHA,
        '--admission', '/workspace/' + ADMISSION, '--output', '/workspace/' + d.RESULT]
    if (spec.get('python_arguments') != expected_args or spec.get('fresh_outputs') != [d.RESULT]
            or spec.get('allocation_deadline_epoch') != DEADLINE or spec.get('expected_source_identity') != SOURCE
            or spec.get('timeout_seconds') != d.OUTER or host.get('status') != 'completed'
            or host.get('exit_code') != 0 or host.get('specification_sha256') != GENERATION_SPEC_SHA
            or host.get('launcher_sha256') != q.LAUNCHER_SHA or host.get('source_identity') != SOURCE
            or digest(host.get('source_files')) != SOURCE or host.get('runtime_identity') != plan.admitted['runtime_identity']
            or host.get('input_file_hashes') != spec.get('expected_file_hashes')
            or host.get('source_unchanged') is not True or host.get('inputs_unchanged') is not True
            or host.get('client_stop_error') is not None or not q.cleanup_complete(host.get('cleanup', []))
            or host.get('allocation_deadline_epoch') != DEADLINE
            or not q.epoch(host['started_at']) <= q.epoch(done['started_at']) < q.epoch(done['finished_at']) <= q.epoch(host['finished_at'])):
        raise ValueError('Complete generation host provenance and exact-container cleanup required')
    pins = spec.get('expected_file_hashes', {})
    for required in (DEVELOPMENT, d.REL + '/configs/eager50.json', ADMISSION, q.LAUNCHER):
        if pins.get(required) != sha(q.project_file(required)):
            raise ValueError('Generation host lacks a required frozen input pin')
    q.check_files(pins)
    bound = q.bindings([path, host_path])
    bound.update({str(q.project_file(p)): h for p, h in pins.items()})
    return bound


def generation(d, plan):
    import torch
    q = plan.q
    output = ROOT / d.RESULT
    done_path = q.project_file(output / 'completed.json'); done = read(done_path)
    actual = d.run_artifacts(plan, output)
    admission_path = q.project_file(ADMISSION); admission = read(admission_path)
    expected_admission = {'type': 'a100_math_development_admission_v1',
        'producer_sha256': DEVELOPMENT_SHA, 'source_identity': SOURCE,
        'allocation_deadline_epoch': DEADLINE, 'platform_gate': plan.admitted,
        'strict_prerequisite': plan.strict, 'smoke_baseline': plan.baseline,
        'input_sha256': plan.inputs, 'time_bound': plan.bound,
        'automatic_launch_authorized': False, 'scope': d.SCOPE}
    required = {**plan.inputs, **q.bindings([admission_path])}
    if (sha(admission_path) != ADMISSION_SHA or any(admission.get(k) != v for k, v in expected_admission.items())
            or done.get('type') != 'a100_math_development_generation_v1' or done.get('status') != 'completed'
            or done.get('producer_sha256') != DEVELOPMENT_SHA or done.get('source_identity') != SOURCE
            or done.get('input_sha256') != required or done.get('platform_gate') != plan.admitted
            or done.get('strict_prerequisite') != plan.strict or done.get('smoke_baseline') != plan.baseline
            or done.get('algorithm_policy') != plan.context.runtime.declaration(plan.context.base)
            or done.get('admission_path') != '/workspace/' + ADMISSION or done.get('admission_sha256') != ADMISSION_SHA
            or done.get('time_bound') != plan.bound or done.get('allocation_deadline_epoch') != DEADLINE
            or done.get('scope') != d.SCOPE or done.get('result') != actual
            or done.get('partial_artifact_sha256') != actual['artifact_sha256']
            or done.get('inputs_and_source_unchanged') is not True or done.get('platform_restored') is not True
            or done.get('evaluation_pending') is not True
            or any(done.get(k) is not False for k in ('legacy_numerical_passed', 'headline_eligible',
                'quality_preserving', 'h100_reference_eligible', 'all_development_scenes_exact_agreement_claimed'))):
        raise ValueError('Complete generation, unchanged admission, full inventory and false acceptance gates required')
    if not 0 <= q.epoch(done['started_at']) - q.epoch(admission['created_at']) <= 300:
        raise ValueError('Generation did not use a fresh admission')
    for value, stamp in ((admission, q.epoch(admission['created_at'])), (done, q.epoch(done['started_at']))):
        disk = value['disk'] if value is admission else value['disk_at_start']
        if disk != d.admission_limits(q, now=stamp, free_bytes=disk.get('observed_free_bytes')):
            raise ValueError('Generation disk/deadline admission differs')
    q.lifecycle(done.get('policy_execution', {}), plan.context.ab.snapshot(torch, plan.context.base))
    artifacts = {str(q.project_file(output / p)): h for p, h in actual['artifact_sha256'].items()}
    bindings = {**required, **artifacts, **q.bindings([done_path]), **generation_host(d, plan, done)}
    q.check_files(bindings)
    from drivebench.evaluation.io import load_run
    return {'run': load_run(output), 'records': plan.records, 'wanted': plan.wanted,
        'bindings': bindings, 'completion': done}


def future_inputs(q, item):
    from drivebench.data.manifest import verify_input_checksums
    from drivebench.data.manifests import resolve_frame
    from drivebench.evaluation.io import output_path
    if len(item['records']) != COUNT or len(item['run'].measurements) != COUNT:
        raise ValueError('Exactly32 complete manifest records and generated requests required')
    bindings, pairs = {}, {}
    for record, row in zip(item['records'], item['run'].measurements):
        verify_input_checksums(record, ROOT / 'data/nuscenes', include_future=True)
        if row['request_id'] != record['request_id'] or row['scene_id'] != record['scene_id']:
            raise ValueError('Recorded-future request order differs')
        candidate = row['candidates'][0]
        future = [str(resolve_frame(ROOT / 'data/nuscenes', p)) for p in record['future_frames']]
        if (len(record['observed_frames']) != 1 or len(future) != 24
                or candidate.get('ground_truth_paths') != future or candidate.get('conditioning_frames') != 1
                or candidate.get('action_kind') != 'recorded'):
            raise ValueError('Recorded candidate, one observation and exact24 future paths/order required')
        for relative in record['observed_frames'] + record['future_frames']:
            path = q.project_file(resolve_frame(ROOT / 'data/nuscenes', relative))
            bindings[str(path)] = record['frame_checksums'][relative]
        path = q.project_file(output_path(item['run'], candidate['frames_path']))
        key = (row['request_id'], row['repeat'], candidate['candidate_id'], candidate['seed'])
        if key in pairs:
            raise ValueError('Duplicate recorded-future pair')
        pairs[key] = sha(path)
    if len(pairs) != COUNT:
        raise ValueError('Exactly32 distinct recorded-future pairs required')
    q.check_files(bindings)
    return bindings, pairs


def prepare(d):
    plan = d.prepare(DEVELOPMENT_SHA)
    item = generation(d, plan)
    future, pairs = future_inputs(plan.q, item)
    bindings = {**item['bindings'], **future, **plan.q.bindings([__file__])}
    return {'development_plan': plan, 'item': item, 'bindings': bindings, 'pairs': pairs}


def diagnostics(plan):
    import numpy as np
    from drivebench.evaluate import _ground_truth, aggregate_pairs
    from drivebench.evaluation.metrics import as_video, compare_videos
    from drivebench.evaluation.io import output_path
    from drivebench.data.preprocessing import PREPROCESS_VERSION
    item = plan['item']; pairs = []
    for row in item['run'].measurements:
        candidate = row['candidates'][0]
        if candidate.get('conditioning_frames') != 1 or candidate.get('action_kind') != 'recorded':
            raise ValueError('One observed frame and a recorded candidate required')
        path = output_path(item['run'], candidate['frames_path'])
        key = (row['request_id'], row['repeat'], candidate['candidate_id'], candidate['seed'])
        if sha(path) != plan['pairs'].get(key):
            raise ValueError('Retained array changed before its metric calculation')
        future = as_video(np.load(path, mmap_mode='r', allow_pickle=False))[1:]
        truth = _ground_truth(candidate, future.shape)
        if truth is None or len(future) != 24:
            raise ValueError('Complete24-frame recorded future required')
        metrics = compare_videos(truth, future)
        metrics = {k: metrics[k] for k in (*METRICS, 'reference_temporal', 'candidate_temporal')}
        pairs.append({'request_id': row['request_id'], 'scene_id': row['scene_id'], 'repeat': row['repeat'],
            'candidate_id': candidate['candidate_id'], 'seed': candidate['seed'], 'action_kind': 'recorded',
            'trajectory_sha256': candidate['trajectory_sha256'], 'conditioning_frames': 1,
            'output_sha256': plan['pairs'][key],
            'ground_truth_sha256': [sha(p) for p in candidate['ground_truth_paths']], 'metrics': metrics})
        del future, truth
    if len(pairs) != COUNT or len({p['scene_id'] for p in pairs}) != COUNT:
        raise ValueError('All32 distinct development scenes must have complete diagnostics')
    aggregates = aggregate_pairs(pairs)
    return {'type': 'a100_32_scene_baseline_recorded_future_v1', 'source_identity': SOURCE,
        'manifest_role': 'development', 'manifest_sha256': item['run'].config['data']['manifest_sha256'],
        'pair_count': COUNT, 'scene_count': COUNT, 'future_frames_per_pair': 24,
        'preprocessing': PREPROCESS_VERSION, 'computation_device': 'cpu_numpy_pillow',
        'computation': 'Unchanged _ground_truth and compare_videos; observed frame excluded; selected descriptive fields only',
        'pairs': pairs, 'aggregates': {k: aggregates[k] for k in METRICS},
        'freeze_threshold_global_mean_pixel_change': 1 / 255,
        'lpips_computed': False, 'review_exported': False, 'quality_preserving': False,
        'headline_eligible': False, 'h100_reference_eligible': False, 'action_response_evaluated': False,
        'tolerances_calibrated': False, 'scope': SCOPE}


def execute(output, expected):
    output = Path(output)
    if output.resolve() != (ROOT / DEST).resolve() or output.exists() or output.is_symlink():
        raise ValueError('Use the exact fresh32-scene diagnostic destination')
    bound = limits()
    output.mkdir(parents=True)
    receipt = {'type': 'a100_math_development_evaluation_v1', 'status': 'running',
        'started_at': datetime.now(timezone.utc).isoformat(), 'producer_sha256': expected,
        'source_identity': SOURCE, 'scope': SCOPE, 'bounds': bound, 'allocation_deadline_epoch': DEADLINE,
        'legacy_numerical_passed': False, 'quality_preserving': False, 'headline_eligible': False,
        'h100_reference_eligible': False, 'action_response_evaluated': False, 'lpips_computed': False,
        'review_exported': False, 'tolerances_calibrated': False, 'computation_device': 'cpu_numpy_pillow'}
    save(output / 'attempt.json', receipt)
    def timeout(*_):
        raise EvaluationTimeout('Fixed CPU diagnostic time bound expired')
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(INNER)
    try:
        import torch
        cpu_only(torch); receipt['cuda_initialized_before'] = False
        d = development(expected)
        plan = prepare(d)
        p = plan['development_plan']; q = p.q
        cpu_only(torch)
        receipt.update(platform_gate=p.admitted, strict_prerequisite=p.strict,
            generation_policy=p.context.runtime.declaration(p.context.base), input_sha256=plan['bindings'])
        save(output / 'recorded-future.json', diagnostics(plan))
        cpu_only(torch)
        current = p.context.verifier.gate(p.context.original, DEADLINE)
        if current != p.admitted or q.strict_prerequisite(p.context, current) != p.strict:
            raise ValueError('Original platform/strict evidence changed during CPU diagnostics')
        q.check_files(plan['bindings']); cpu_only(torch)
        if shutil.disk_usage(ROOT / 'outputs').free < FREE_FLOOR:
            raise ValueError('Original12GiB free disk floor was not preserved')
        receipt.update(status='completed', inputs_and_source_unchanged=True, platform_restored=True,
            cuda_initialized_after=False)
    except BaseException as error:
        receipt.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous)
        receipt['finished_at'] = datetime.now(timezone.utc).isoformat()
        receipt['artifact_sha256'] = {str(p.relative_to(output)): sha(p) for p in output.rglob('*') if p.is_file()}
        save(output / 'completed.json', receipt)
    print(json.dumps({'status': receipt['status'], 'output': str(output), 'quality_preserving': False, 'headline_eligible': False}))
    return 0 if receipt['status'] == 'completed' else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '4'
    return execute(args.output, args.expected_sha256)


if __name__ == '__main__':
    raise SystemExit(main())
