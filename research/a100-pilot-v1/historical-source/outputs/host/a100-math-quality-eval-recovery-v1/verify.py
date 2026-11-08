#!/usr/bin/env python3
"""Verify retained metric artifacts from one known failed evaluation; never rerun metrics."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
REL = 'outputs/host/a100-math-quality-eval-recovery-v1'
RESULT = 'outputs/a100-math-quality-eval-v1/eager40-vs-eager50'
DEST = 'outputs/a100-math-quality-eval-recovery-v1/eager40-attempt-1'
EVALUATOR = 'outputs/host/a100-math-quality-eval-v1/evaluate.py'
EVALUATOR_SHA = 'fd493adc303cd72f5742dac23f3285930c58563ac70cae2d185a66177e006b07'
HOST_SPEC = 'outputs/host/a100-math-quality-eval-v1/eager40-compare-01-spec.json'
HOST = 'outputs/host/a100-math-quality-eval-v1/eager40-compare-01-host.json'
POSTSTOP = 'outputs/host/a100-math-quality-eval-v1/eager40-compare-01-host.cleanup-43aca53fdd284949ac80c5224c86d0e7.json'
SETUP_SHA = 'b158ef6962d310dc94ff74401dde8ade4709c35905450a93b7106987c5703d99'
SOURCE = 'd8a32813492c0939a0a91573a945c914938972419946065e6822f6209fa470cb'
DEADLINE = 1793168129.682277
OUTER_SECONDS, INNER_SECONDS, RESERVE_SECONDS = 900, 840, 600
EXPECTED_ERROR = {'type': 'ValueError', 'message': 'GPU must be idle; compute-process inspection must succeed'}
FALSE_FLAGS = ('legacy_numerical_passed', 'quality_preserving', 'headline_eligible',
               'h100_reference_eligible', 'action_response_evaluated', 'private_attribution_printed')
PINS = {
    EVALUATOR: EVALUATOR_SHA,
    HOST_SPEC: 'aba8d49e2a8a047b280a6db337847599d8552513b9db9288ac215c790d6dc7b8',
    HOST: '119d5a8e3ef802ea5d18b68894bbaaeba64cd3e43dbee3dc7705e3c309b31dd7',
    POSTSTOP: '7a07fae69ddfe4070c5188bc10bce3f3338f495e3f91589c8c96cf38f575cd57',
    RESULT + '/completed.json': '96a3ccca0c58bafaa10fe7f879aa13fa5e965d420daad48f33ad4bad215285f1',
    'outputs/evaluation-packages/a100-setup-receipt.json': SETUP_SHA,
    'scripts/prepare_perceptual_evaluation.py': 'd0ca8bed525f27c334d0f91e30d0b2f3c8621bfc29b7db385b02ccbed149685c',
    'outputs/host/a100-math-quality-v1/quality.py': '233b18a2d0e64f937978de9b65db5a03c86f175fa04fa29e0ecc6c0885422724',
    'outputs/host/run-bounded-a100-job-v1.py': '8acf1240512bc75ef69ab41888794cb3fadd8ce83835b68c67121e98618d99a3',
    'scripts/run_quality_bounded.py': '225edfb799f86ab59b9c2c1424063bda7782c2c0438f4fd3ca31e843e396eca1',
    'configs/tolerances.yaml': 'acf035bbbce67528a46d40ae0a16eab7df79f2757c628c3413f96875b8702683',
}
SPECS = {name: 'outputs/host/a100-math-quality-v1/' + name + '-run-01-spec.json'
         for name in ('eager50', 'eager40')}
RATING_FIELDS = ['case_id', 'preferred_panel_A_B_tie', 'acceptable_A', 'acceptable_B',
    'object_disappearance', 'object_duplication', 'freezing', 'geometry_change',
    'action_response_notes', 'reviewer_notes']
SCOPE = ('Verification of existing measured artifacts from the exact known failed eager40 comparison. '
    'The original job remains failed; this fresh process verifies its completed metric/native-export files '
    'and current idle runtime/provenance. No recomputation, future generation, inferred human ratings, '
    'numerical/quality promotion, action-response, safety, H100, timing, speedup or headline acceptance. '
    'The original final process roster was not saved; the exact idle/inspection error does not prove '
    'an exclusive process cause or dedicated conditions during the metric computation.')


class RecoveryTimeout(BaseException):
    pass


def sha(path):
    import hashlib
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''): value.update(block)
    return value.hexdigest()


def read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict): raise ValueError('JSON object required')
    return value


def stamp(): return datetime.now(timezone.utc).isoformat()


def save(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def safe_file(value):
    path = Path(value)
    path = ROOT / path if not path.is_absolute() else path
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Missing, redirected or non-project recovery input')
    return path


def frozen_inputs(expected):
    if sha(__file__) != expected: raise ValueError('Recovery verifier source differs')
    for name, digest in PINS.items():
        if sha(safe_file(name)) != digest: raise ValueError('Known failed attempt/dependency changed: ' + name)
    return {str(safe_file(name)): digest for name, digest in PINS.items()}


def load_evaluator():
    spec = importlib.util.spec_from_file_location('a100_frozen_evaluation_recovery', safe_file(EVALUATOR))
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def limits(*, now=None, free=None):
    import shutil
    now = time.time() if now is None else now
    if not math.isfinite(now) or now + OUTER_SECONDS + RESERVE_SECONDS >= DEADLINE:
        raise ValueError('Recovery bound and archive reserve exceed immutable allocation deadline')
    free = shutil.disk_usage(ROOT / 'outputs').free if free is None else free
    if free < 8 * 1024**3: raise ValueError('Require 8 GiB retained free-space reserve')
    return {'outer_seconds': OUTER_SECONDS, 'inner_seconds': INNER_SECONDS,
        'archive_cleanup_reserve_seconds': RESERVE_SECONDS, 'free_bytes': free}


def no_cuda_context(torch):
    if torch.cuda.is_initialized(): raise ValueError('Recovery process must never initialize a CUDA context')


def artifact_inventory(failed):
    output = ROOT / RESULT
    if output.is_symlink() or not output.is_dir(): raise ValueError('Original result directory missing or redirected')
    inventory = failed.get('artifact_sha256')
    if not isinstance(inventory, dict) or len(inventory) != 122:
        raise ValueError('Exact original 122-artifact inventory required')
    for name, digest in inventory.items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError('Unsafe original artifact inventory')
    paths = list(output.rglob('*'))
    if any(p.is_symlink() for p in paths): raise ValueError('Symlink in original result tree')
    actual = {str(p.relative_to(output)): sha(p) for p in paths if p.is_file() and p != output / 'completed.json'}
    if actual != inventory: raise ValueError('Original result inventory or bytes changed')
    return {str(output / name): digest for name, digest in inventory.items()}


def failed_attempt(failed, plan, context, evaluation_env, setup_bindings):
    required = {**plan['bindings'], **setup_bindings}
    bounds = failed.get('bounds', {})
    if (failed.get('type') != 'a100_math_quality_evaluation_v1' or failed.get('status') != 'failed'
            or failed.get('command') != 'compare' or failed.get('variant') != 'eager40'
            or failed.get('error') != EXPECTED_ERROR or failed.get('producer_sha256') != EVALUATOR_SHA
            or failed.get('source_identity') != SOURCE or failed.get('allocation_deadline_epoch') != DEADLINE
            or failed.get('platform_gate') != plan['admitted'] or failed.get('strict_prerequisite') != plan['strict']
            or failed.get('generation_policy') != context.runtime.declaration(context.base)
            or failed.get('input_sha256') != required or failed.get('evaluation_environment') != evaluation_env
            or failed.get('setup_receipt_sha256') != SETUP_SHA or failed.get('review_cases') != 4
            or any(failed.get(k) is not False for k in FALSE_FLAGS)
            or 'inputs_and_source_unchanged' in failed or 'platform_restored' in failed
            or bounds.get('outer_seconds') != 1800 or bounds.get('inner_seconds') != 1680
            or bounds.get('separate_archive_cleanup_reserve_seconds') != 600):
        raise ValueError('Only the exact known late idle-check failure is recoverable')
    return required


def host_failure(q, failed, admitted, spec, host, post):
    from drivebench.config import digest
    expected_args = [EVALUATOR, 'compare', '--variant', 'eager40', '--expected-sha256', EVALUATOR_SHA,
        '--setup-sha256', SETUP_SHA, '--lpips-device', 'cuda',
        '--generation-spec', 'eager50=' + SPECS['eager50'], '--generation-spec', 'eager40=' + SPECS['eager40'],
        '--output', str(ROOT / RESULT)]
    if (spec.get('python_arguments') != expected_args or spec.get('fresh_outputs') != [RESULT]
            or spec.get('metadata') != HOST or spec.get('allocation_deadline_epoch') != DEADLINE
            or spec.get('timeout_seconds') != 1800 or spec.get('expected_source_identity') != SOURCE
            or host.get('status') != 'failed' or host.get('exit_code') != 1 or 'error' in host or 'drift_error' in host
            or host.get('source_unchanged') is not True or host.get('inputs_unchanged') is not True
            or host.get('source_identity') != SOURCE or digest(host.get('source_files')) != SOURCE
            or host.get('specification_sha256') != PINS[HOST_SPEC] or host.get('launcher_sha256') != q.LAUNCHER_SHA
            or host.get('runtime_identity') != admitted['runtime_identity']
            or host.get('input_file_hashes') != spec.get('expected_file_hashes')
            or host.get('client_stop_error') is not None or not q.cleanup_complete(host.get('cleanup', []))
            or host.get('allocation_deadline_epoch') != DEADLINE or host.get('timeout_seconds') != 1800
            or host.get('container_name') != spec.get('container_name')
            or host.get('container_image') != 'sha256:d2ab425962b505ae4bf21175451d889a66c12bf394ce4763479be9e3f7776022'
            or not re.fullmatch(r'[0-9a-f]{64}', host.get('container_id', ''))
            or not re.fullmatch(r'[0-9a-f]{32}', host.get('container_nonce', ''))):
        raise ValueError('Known failed host completion/provenance/cleanup differs')
    if (post.get('type') != 'a100_diagnostic_poststop_cleanup_v1' or post.get('status') != 'completed'
            or post.get('exit_code') != 0 or post.get('volumes_deleted') is not False
            or not q.cleanup_complete(post.get('cleanup', []))
            or post.get('specification_sha256') != PINS[HOST_SPEC] or post.get('metadata_sha256') != PINS[HOST]
            or post.get('launcher_sha256') != q.LAUNCHER_SHA
            or post.get('container_name') != host['container_name'] or post.get('container_nonce') != host['container_nonce']
            or post.get('expected_container_id') != host['container_id']
            or post.get('observed_container_id', host['container_id']) != host['container_id']):
        raise ValueError('Mandatory exact-container poststop cleanup evidence differs')
    chronology = [q.epoch(host['started_at']), q.epoch(failed['started_at']), q.epoch(failed['finished_at']),
                  q.epoch(host['finished_at']), q.epoch(post['started_at']), q.epoch(post['finished_at'])]
    if (chronology != sorted(chronology) or chronology[1] >= chronology[2]
            or chronology[-1] >= DEADLINE or chronology[3] - chronology[0] >= 1800
            or chronology[2] - chronology[1] >= 1680):
        raise ValueError('Failure/cleanup chronology or non-timeout duration differs')
    q.check_files(spec['expected_file_hashes'])
    return {str(q.project_file(p)): h for p, h in spec['expected_file_hashes'].items()}


def blank_review(path, case_ids):
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != RATING_FIELDS: raise ValueError('Original blank review columns differ')
        rows = list(reader)
    if (len(rows) != 4 or {r.get('case_id') for r in rows} != case_ids
            or any(v != '' for r in rows for k, v in r.items() if k != 'case_id')):
        raise ValueError('Review is incomplete, changed or contains ratings')


def saved_artifacts(ev, q, plan, failed):
    from drivebench.config import digest
    output = ROOT / RESULT
    metric_path, metric_receipt = output / 'evaluation.json', output / 'metric-receipt.json'
    result = read(metric_path)
    ev.validate_result(plan, result, SETUP_SHA, 'cuda')
    for key, name in {'index': 'index.html', 'csv': 'review.csv', 'private_key': 'private_key.json'}.items():
        if result['review'].get(key) != str(output / 'review' / name): raise ValueError('Original review paths differ')
    proof = read(metric_receipt)
    review_dir = output / 'review'
    review_files = {str(p): sha(p) for p in review_dir.rglob('*') if p.is_file()}
    if proof != {'evaluation_sha256': sha(metric_path), 'review_artifact_sha256': review_files}:
        raise ValueError('Metric receipt does not bind the complete original review')
    registry = {str(path): checksum for item in plan['items'].values()
                for path, checksum in item['bindings'].items() if path.endswith('.npy')}
    q.check_files(registry)  # No retirement exception: all original arrays remain mandatory.
    native = ev.load_module('a100_recovery_frozen_native', ROOT / ev.NATIVE)
    cases, binding, review = native._bundle_inputs(metric_path, metric_receipt, registry)
    token = digest({'metric': binding['metric_sha256'], 'wrapper': binding['wrapper_sha256']})[:24]
    bundle = output / 'native-review' / token
    if failed.get('native_review_public_index') != str(bundle / 'public/index.html'):
        raise ValueError('Recorded native export path differs')
    native_receipt = native.validate_export(bundle, binding)
    manifest = read(bundle / 'public/manifest.json')
    expected_ids = {row['case_id'] for row in cases}
    public_cases = manifest.get('cases', [])
    if (len(cases) != 4 or len(expected_ids) != 4 or len(public_cases) != 4
            or {c.get('case_id') for c in public_cases} != expected_ids
            or manifest.get('receipt_type') != native.EXPORT
            or any(c.get('future_frames') != 24 or c.get('panel_width') != 1024 or c.get('panel_height') != 576
                or len(c.get('frames', [])) != 24 or len(c.get('pages', [])) != 2 for c in public_cases)):
        raise ValueError('Four complete native 24-frame cases required')
    frame_names = [name for case in public_cases for name in case['frames']]
    if (len(set(frame_names)) != 96 or set(manifest.get('frame_sha256', {})) != set(frame_names)
            or any(native_receipt['public_artifact_sha256'].get(n) != h for n, h in manifest['frame_sha256'].items())):
        raise ValueError('Native frame inventory differs')
    blank_review(review['csv'], expected_ids)
    blank_review(bundle / 'public/review.csv', expected_ids)
    if sha(review['csv']) != sha(bundle / 'public/review.csv'): raise ValueError('Original/native blank ratings differ')
    return {'metric_sha256': sha(metric_path), 'metric_receipt_sha256': sha(metric_receipt),
        'native_receipt_sha256': sha(bundle / 'private-receipt.json'), 'native_public_index': str(bundle / 'public/index.html'),
        'complete_pairs': 4, 'native_cases': 4, 'native_future_frames_per_case': 24,
        'lpips_device_of_original_measurements': 'cuda', 'review_ratings_present': False,
        'metrics_recomputed': False, 'native_export_regenerated': False,
        'original_generic_quality_preserving': False}


def execute(output, expected):
    output = Path(output)
    if output.resolve() != (ROOT / DEST).resolve() or output.exists() or output.is_symlink():
        raise ValueError('Exact fresh recovery output required')
    bound = limits(); frozen_inputs(expected)
    ev = load_evaluator(); q, context = ev.dependencies(EVALUATOR_SHA)
    import torch
    no_cuda_context(torch)
    output.mkdir(parents=True)
    receipt = {'type': 'a100_math_evaluation_artifact_recovery_v1', 'status': 'running',
        'started_at': stamp(), 'producer_sha256': expected, 'source_identity': SOURCE,
        'bounds': bound, 'allocation_deadline_epoch': DEADLINE, 'scope': SCOPE,
        'original_job_succeeded': False, 'original_failure_preserved': True,
        'metrics_recomputed': False, 'native_export_regenerated': False,
        'original_final_process_roster_available': False,
        'original_exclusive_process_cause_verified': False,
        'original_dedicated_conditions_during_metrics_verified': False,
        **{key: False for key in FALSE_FLAGS}}
    save(output / 'attempt.json', receipt)
    previous = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(RecoveryTimeout('Recovery verification bound expired')))
    signal.alarm(INNER_SECONDS)
    try:
        original = read(ROOT / RESULT / 'completed.json')
        inventory = artifact_inventory(original)
        plan = ev.prepare(q, context, 'eager40', SPECS)
        no_cuda_context(torch)
        with ev.isolated_lpips() as env:
            setup, setup_bindings = ev.perceptual_setup(q, SETUP_SHA)
            no_cuda_context(torch)
        required = failed_attempt(original, plan, context, env, setup_bindings)
        host_inputs = host_failure(q, original, plan['admitted'], read(ROOT / HOST_SPEC), read(ROOT / HOST), read(ROOT / POSTSTOP))
        bindings = {**required, **host_inputs, **inventory, **frozen_inputs(expected), str(Path(__file__)): expected}
        q.check_files(bindings)
        result = saved_artifacts(ev, q, plan, original)
        no_cuda_context(torch)
        current = context.verifier.gate(context.original, DEADLINE)
        if current != plan['admitted'] or q.strict_prerequisite(context, current) != plan['strict']:
            raise ValueError('Fresh idle platform/strict identity differs')
        no_cuda_context(torch)
        if artifact_inventory(original) != inventory: raise ValueError('Original output inventory changed during verification')
        q.check_files(bindings)
        frozen_inputs(expected)
        receipt.update(status='completed', existing_measured_artifacts_verified=True,
            original_attempt_sha256=PINS[RESULT + '/completed.json'], original_error=EXPECTED_ERROR,
            original_host_sha256=PINS[HOST], mandatory_poststop_cleanup_sha256=PINS[POSTSTOP],
            original_status='failed', current_idle_platform_gate=current,
            strict_prerequisite=plan['strict'], input_sha256=bindings, result=result,
            inputs_outputs_source_unchanged=True, recovery_cuda_context_initialized=False)
    except BaseException as error:
        receipt.update(status='failed', existing_measured_artifacts_verified=False,
            error={'type': type(error).__name__, 'message': str(error)})
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous)
        receipt['finished_at'] = stamp()
        save(output / 'verification.json', receipt)
    print(json.dumps({'status': receipt['status'], 'existing_measured_artifacts_verified': receipt.get('existing_measured_artifacts_verified', False),
        'original_job_succeeded': False, 'quality_preserving': False, 'headline_eligible': False}))
    return 0 if receipt['status'] == 'completed' else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'): os.environ[key] = '4'
    return execute(args.output, args.expected_sha256)


if __name__ == '__main__': raise SystemExit(main())
