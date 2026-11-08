#!/usr/bin/env python3
"""Prepared-only CPU comparison of actual four-scene eager47 with eager50."""
from __future__ import annotations
import argparse
import csv
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
REL = 'outputs/host/a100-math-step47-eval-v1'
DEST = 'outputs/a100-math-step47-eval-v1/eager47-vs-eager50'
STEP = 'outputs/host/a100-math-step47-v1/step47.py'
STEP_SHA = 'a42fde8397fce3d98387f36a959e78d9e8ba0e61c9974be4c10439f505575d75'
EVAL = 'outputs/host/a100-math-quality-eval-v1/evaluate.py'
EVAL_SHA = 'fd493adc303cd72f5742dac23f3285930c58563ac70cae2d185a66177e006b07'
SETUP_SHA = 'b158ef6962d310dc94ff74401dde8ade4709c35905450a93b7106987c5703d99'
STEP_REL = 'outputs/host/a100-math-step47-v1'
STEP_RESULT = 'outputs/a100-math-step47-v1/runs/eager47'
GEN_SPEC = STEP_REL + '/run-01-spec.json'
GEN_HOST = STEP_REL + '/run-01-host.json'
ADMISSION = STEP_REL + '/admission-01.json'
OPERATOR = REL + '/operator-prerequisites-01.json'
SOURCE = 'd8a32813492c0939a0a91573a945c914938972419946065e6822f6209fa470cb'
DEADLINE = 1793168129.682277
OUTER, INNER, RESERVE = 1800, 1680, 1200
FREE_FLOOR, EXPORT_MARGIN = 12 * 1024**3, 1024**3
FALSE_FLAGS = ('legacy_numerical_passed', 'quality_preserving', 'headline_eligible',
    'h100_reference_eligible', 'matched_latency_established', 'optimized_baseline_claimed',
    'action_response_evaluated', 'private_attribution_printed', 'tolerances_calibrated')
SCOPE = ('Four fixed development-subset scenes; actual eager47 versus preserved eager50 with identical '
    'recorded actions/seeds. Future-only CPU LPIPS/alex and unchanged RGB/temporal diagnostics plus '
    'unrated blind native exports. Descriptive evidence only; no quality, matched-latency, optimized-baseline, '
    'action-response, safety, H100, speedup or headline acceptance. Temporal proxies are frame-indexed, '
    'not physical-time normalized. Prior generation timings may include archive IO and are not pooled.')


class ComparisonTimeout(BaseException):
    pass


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''): h.update(block)
    return h.hexdigest()


def read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict): raise ValueError('JSON object required')
    return value


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def load(relative, expected, name):
    path = ROOT / relative
    if path.is_symlink() or sha(path) != expected: raise ValueError('Frozen dependency changed: ' + relative)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def limits(*, now=None, free=None):
    now = time.time() if now is None else now
    free = shutil.disk_usage(ROOT / 'outputs').free if free is None else free
    if type(now) not in (int, float) or not math.isfinite(now) or now + OUTER + RESERVE >= DEADLINE:
        raise ValueError('Fixed comparison and archive reserve exceed allocation deadline')
    if type(free) is not int or free < FREE_FLOOR + EXPORT_MARGIN:
        raise ValueError('Require12GiB floor plus1GiB native export margin')
    return {'outer_seconds': OUTER, 'inner_seconds': INNER,
        'separate_archive_cleanup_reserve_seconds': RESERVE, 'free_bytes': free,
        'preserved_free_floor_bytes': FREE_FLOOR, 'export_margin_bytes': EXPORT_MARGIN,
        'empirical_bound_guaranteed': False}


def cpu_only(torch):
    if torch.cuda.is_initialized() is not False: raise ValueError('CPU comparison must not initialize CUDA')


def generation_documents(s, q, plan, admission, done, spec, host, *, admission_sha, spec_sha, actual):
    """Validate the actual step47 format without changing its frozen implementation."""
    from drivebench.config import digest
    fields = s.admission_fields(plan)
    required = {**plan.inputs, '/workspace/' + ADMISSION: admission_sha}
    false = ('legacy_numerical_passed', 'headline_eligible', 'quality_preserving',
        'h100_reference_eligible', 'matched_latency_claimed', 'optimized_baseline_claimed',
        'variant_exact_agreement_claimed')
    if (admission.get('type') != 'a100_math_step47_admission_v1'
            or admission.get('producer_sha256') != STEP_SHA or admission.get('source_identity') != SOURCE
            or admission.get('allocation_deadline_epoch') != DEADLINE
            or any(admission.get(k) != v for k, v in fields.items())
            or done.get('type') != 'a100_math_step47_generation_v1' or done.get('status') != 'completed'
            or done.get('variant') != 'eager47' or done.get('producer_sha256') != STEP_SHA
            or done.get('source_identity') != SOURCE or done.get('allocation_deadline_epoch') != DEADLINE
            or any(done.get(k) != v for k, v in fields.items() if k != 'input_sha256')
            or done.get('input_sha256') != required or done.get('admission_path') != '/workspace/' + ADMISSION
            or done.get('admission_sha256') != admission_sha or done.get('result') != actual
            or done.get('partial_artifact_sha256') != actual.get('artifact_sha256')
            or done.get('algorithm_policy') != plan.context.runtime.declaration(plan.context.base)
            or done.get('inputs_and_source_unchanged') is not True or done.get('platform_restored') is not True
            or done.get('evaluation_pending') is not True or any(done.get(k) is not False for k in false)):
        raise ValueError('Actual completed47 admission, generation, inventory and false gates required')
    stamps = [q.epoch(host['started_at']), q.epoch(done['started_at']),
              q.epoch(done['finished_at']), q.epoch(host['finished_at'])]
    admitted_at = q.epoch(admission['created_at'])
    if (stamps != sorted(stamps) or stamps[1] >= stamps[2]
            or not 0 <= stamps[1] - admitted_at <= 300 or stamps[-1] >= DEADLINE
            or stamps[2] - stamps[1] >= s.INNER or stamps[3] - stamps[0] >= s.OUTER):
        raise ValueError('Fresh47 admission and bounded generation chronology required')
    for value, key, when in ((admission, 'disk', admitted_at), (done, 'disk_at_start', stamps[1])):
        disk = value.get(key, {})
        if disk != s.admission_limits(now=when, free_bytes=disk.get('observed_free_bytes')):
            raise ValueError('Actual47 remote disk/deadline admission differs')
    args = [STEP, 'run', '--expected-sha256', STEP_SHA, '--admission', '/workspace/' + ADMISSION,
            '--output', '/workspace/' + STEP_RESULT]
    if (spec.get('python_arguments') != args or spec.get('fresh_outputs') != [STEP_RESULT]
            or spec.get('metadata') != GEN_HOST or spec.get('container_name') != 'drivebench-a100-math-step47-run-01'
            or spec.get('allocation_deadline_epoch') != DEADLINE or spec.get('expected_source_identity') != SOURCE
            or spec.get('timeout_seconds') != s.OUTER or host.get('status') != 'completed' or host.get('exit_code') != 0
            or host.get('specification_sha256') != spec_sha or host.get('launcher_sha256') != q.LAUNCHER_SHA
            or host.get('source_identity') != SOURCE or digest(host.get('source_files')) != SOURCE
            or host.get('runtime_identity') != plan.admitted['runtime_identity']
            or host.get('input_file_hashes') != spec.get('expected_file_hashes')
            or host.get('source_unchanged') is not True or host.get('inputs_unchanged') is not True
            or host.get('client_stop_error') is not None or not q.cleanup_complete(host.get('cleanup', []))
            or host.get('allocation_deadline_epoch') != DEADLINE):
        raise ValueError('Actual47 host specification, identity and cleanup differ')
    return required


def candidate(s, e, plan):
    import torch
    q = plan.q; output = ROOT / STEP_RESULT
    if output.is_symlink() or not output.is_dir() or not output.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Existing project-owned47 result directory required')
    paths = [q.project_file(p) for p in (ADMISSION, GEN_SPEC, GEN_HOST, STEP_RESULT + '/completed.json')]
    admission, spec, host, done = (read(p) for p in paths)
    actual = s.run_artifacts(plan, output)
    bindings = generation_documents(s, q, plan, admission, done, spec, host,
        admission_sha=sha(paths[0]), spec_sha=sha(paths[1]), actual=actual)
    q.lifecycle(done.get('policy_execution', {}), plan.context.ab.snapshot(torch, plan.context.base))
    bindings.update(q.bindings(paths))
    bindings.update({str(q.project_file(p)): h for p, h in spec['expected_file_hashes'].items()})
    bindings.update({str(q.project_file(output / p)): h for p, h in actual['artifact_sha256'].items()})
    q.check_files(bindings)
    from drivebench.evaluation.io import load_run
    return {'run': load_run(output), 'records': plan.records, 'wanted': plan.wanted,
        'bindings': bindings, 'completion': done}


def operator_proof(q):
    proof = read(q.project_file(OPERATOR))
    if (proof.get('type') != 'a100_step47_comparison_operator_prerequisites_v1'
            or proof.get('source_identity') != SOURCE or proof.get('allocation_deadline_epoch') != DEADLINE
            or proof.get('baseline32_full_backup_verified') is not True
            or proof.get('baseline32_gt_completed') is not True or proof.get('automatic_launch_authorized') is not False
            or proof.get('quality_acceptance_claimed') is not False or not proof.get('remote_artifact_sha256')):
        raise ValueError('Verified full32-baseline backup and completed GT prerequisites required')
    bindings = dict(proof['remote_artifact_sha256']); q.check_files(bindings)
    bindings.update(q.bindings([OPERATOR]))
    return proof, bindings


def prepare(expected):
    if sha(__file__) != expected: raise ValueError('Reviewed comparison helper changed')
    s = load(STEP, STEP_SHA, 'comparison_frozen_step47')
    e = load(EVAL, EVAL_SHA, 'comparison_frozen_metrics')
    plan = s.prepare(STEP_SHA); q = plan.q
    baseline = e.generation(q, plan.context, plan.admitted, plan.strict, 'eager50',
        q.REL + '/eager50-run-01-spec.json')
    items = {'eager50': baseline, 'eager47': candidate(s, e, plan)}
    bindings, pairs = {}, {}
    for name, item in items.items():
        future, pair_hashes = e.future_inputs(q, item)
        bindings.update(item['bindings']); bindings.update(future); pairs[name] = pair_hashes
    if set(pairs['eager50']) != set(pairs['eager47']): raise ValueError('Four scene/action/seed pair identities differ')
    proof, operator_bindings = operator_proof(q); bindings.update(operator_bindings)
    if sha(ROOT / e.NATIVE) != e.NATIVE_SHA or sha(ROOT / e.TOLERANCES) != e.TOLERANCES_SHA:
        raise ValueError('Frozen native exporter or uncalibrated tolerances changed')
    bindings.update(q.bindings([__file__, EVAL, STEP, e.NATIVE, e.TOLERANCES]))
    return s, e, plan, {'items': items, 'bindings': bindings, 'pairs': pairs,
        'admitted': plan.admitted, 'strict': plan.strict, 'variant': 'eager47'}, proof


def blank_review(path, ids):
    fields = ['case_id', 'preferred_panel_A_B_tie', 'acceptable_A', 'acceptable_B',
        'object_disappearance', 'object_duplication', 'freezing', 'geometry_change',
        'action_response_notes', 'reviewer_notes']
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream); rows = list(reader)
        if reader.fieldnames != fields or len(rows) != 4 or {r.get('case_id') for r in rows} != ids:
            raise ValueError('Four complete blind review rows required')
    if any(v != '' for r in rows for k, v in r.items() if k != 'case_id'):
        raise ValueError('Review must remain unrated')


def compare(e, q, plan, output):
    import torch
    cpu_only(torch)
    with e.isolated_lpips() as env:
        _, lpips = e.perceptual_setup(q, SETUP_SHA); plan['bindings'].update(lpips)
        from drivebench.evaluate import evaluate_runs
        metric = output / 'evaluation.json'
        result = evaluate_runs(plan['items']['eager47']['run'].path,
            reference=plan['items']['eager50']['run'].path, tolerances_path=ROOT / e.TOLERANCES,
            lpips=True, lpips_device='cpu', action_response=False, review_limit=4, output=metric)
        e.validate_result(plan, result, SETUP_SHA, 'cpu'); cpu_only(torch)
        review = Path(result['review']['index']).parent
        metric_receipt = output / 'metric-receipt.json'
        save(metric_receipt, {'evaluation_sha256': sha(metric),
            'review_artifact_sha256': {str(p): sha(p) for p in review.rglob('*') if p.is_file()}})
        registry = {str(p): h for item in plan['items'].values() for p, h in item['bindings'].items() if p.endswith('.npy')}
        native = e.load_module('step47_native_export', ROOT / e.NATIVE)
        bundle = native.export_comparison(metric, metric_receipt, registry, output / 'native-review', minimum_free_bytes=FREE_FLOOR)
        manifest = read(bundle / 'public/manifest.json'); cases = manifest.get('cases', [])
        ids = {c.get('case_id') for c in cases}
        if len(cases) != 4 or len(ids) != 4 or any(len(c.get('frames', [])) != 24 for c in cases):
            raise ValueError('Four complete future-only native review cases required')
        blank_review(result['review']['csv'], ids); blank_review(bundle / 'public/review.csv', ids)
        cpu_only(torch)
    return {'evaluation_environment': env, 'setup_receipt_sha256': SETUP_SHA,
        'native_review_public_index': str(bundle / 'public/index.html'), 'review_cases': 4,
        'lpips_device': 'cpu', 'lpips_device_equivalence_claimed': False}


def execute(output, expected):
    output = Path(output)
    if output.resolve() != (ROOT / DEST).resolve() or output.exists() or output.is_symlink():
        raise ValueError('Exact fresh comparison destination required')
    bound = limits(); output.mkdir(parents=True)
    receipt = {'type': 'a100_math_step47_comparison_v1', 'status': 'running', 'variant': 'eager47',
        'started_at': datetime.now(timezone.utc).isoformat(), 'producer_sha256': expected,
        'source_identity': SOURCE, 'scope': SCOPE, 'bounds': bound, 'allocation_deadline_epoch': DEADLINE,
        **{k: False for k in FALSE_FLAGS}}
    save(output / 'attempt.json', receipt)
    def timeout(*_): raise ComparisonTimeout('Fixed CPU comparison bound expired')
    previous = signal.signal(signal.SIGALRM, timeout); signal.alarm(INNER)
    try:
        import torch
        cpu_only(torch); receipt['cuda_initialized_before'] = False
        s, e, context, plan, proof = prepare(expected); q = context.q
        receipt.update(platform_gate=plan['admitted'], strict_prerequisite=plan['strict'],
            generation_policy=context.context.runtime.declaration(context.context.base),
            operator_prerequisites_sha256=sha(ROOT / OPERATOR), input_sha256=plan['bindings'])
        cpu_only(torch); receipt.update(compare(e, q, plan, output)); cpu_only(torch)
        current = context.context.verifier.gate(context.context.original, DEADLINE)
        if current != plan['admitted'] or q.strict_prerequisite(context.context, current) != plan['strict']:
            raise ValueError('Fresh idle platform and strict provenance changed during comparison')
        q.check_files(plan['bindings']); cpu_only(torch)
        if shutil.disk_usage(ROOT / 'outputs').free < FREE_FLOOR: raise ValueError('Original12GiB floor not preserved')
        receipt.update(status='completed', inputs_and_source_unchanged=True, platform_restored=True,
            cuda_initialized_after=False)
    except BaseException as error:
        receipt.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
    finally:
        try:
            receipt['finished_at'] = datetime.now(timezone.utc).isoformat()
            receipt['artifact_sha256'] = {str(p.relative_to(output)): sha(p) for p in output.rglob('*') if p.is_file()}
            save(output / 'completed.json', receipt)
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, previous)
    print(json.dumps({'status': receipt['status'], 'output': str(output), 'quality_preserving': False, 'headline_eligible': False}))
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-sha256', required=True); p.add_argument('--output', required=True)
    args = p.parse_args()
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'): os.environ[key] = '4'
    raise SystemExit(execute(args.output, args.expected_sha256))
