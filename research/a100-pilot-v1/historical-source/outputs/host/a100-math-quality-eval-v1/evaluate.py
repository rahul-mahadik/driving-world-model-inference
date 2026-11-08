#!/usr/bin/env python3
"""Bounded descriptive evaluation of the fixed A100 math-SDPA smoke; no acceptance."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
QUALITY = 'outputs/host/a100-math-quality-v1/quality.py'
QUALITY_SHA = '233b18a2d0e64f937978de9b65db5a03c86f175fa04fa29e0ecc6c0885422724'
NATIVE = 'scripts/run_quality_bounded.py'
NATIVE_SHA = '225edfb799f86ab59b9c2c1424063bda7782c2c0438f4fd3ca31e843e396eca1'
SETUP_HELPER = 'scripts/prepare_perceptual_evaluation.py'
SETUP_HELPER_SHA = 'd0ca8bed525f27c334d0f91e30d0b2f3c8621bfc29b7db385b02ccbed149685c'
TOLERANCES = 'configs/tolerances.yaml'
TOLERANCES_SHA = 'acf035bbbce67528a46d40ae0a16eab7df79f2757c628c3413f96875b8702683'
SETUP = 'outputs/evaluation-packages/a100-setup-receipt.json'
DEST = 'outputs/a100-math-quality-eval-v1'
SOURCE = 'd8a32813492c0939a0a91573a945c914938972419946065e6822f6209fa470cb'
DEADLINE = 1793168129.682277
OUTER_SECONDS = 1800
RESERVE_SECONDS = 600
SCOPE = ('Four fixed development-subset scenes only. Uncalibrated descriptive RGB/LPIPS and recorded-future '
         'diagnostics plus unrated blind exports; no quality, action-response, safety, H100, timing or headline acceptance. '
         'Temporal differences are frame-indexed proxies: irregular source intervals are not normalized to physical time; '
         '10 fps GIFs are previews only. Request timings include policy instrumentation and concurrent backup IO; whole-job elapsed also includes provenance checks.')


class EvaluationTimeout(BaseException):
    """Cannot be swallowed by evaluator per-pair OSError/Exception handlers."""


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def sha(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''): h.update(block)
    return h.hexdigest()


def read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict): raise ValueError('Expected JSON object')
    return value


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())


def dependencies(expected):
    if sha(__file__) != expected or sha(ROOT / QUALITY) != QUALITY_SHA or sha(ROOT / NATIVE) != NATIVE_SHA:
        raise ValueError('Reviewed evaluator/generator/native-export helper differs')
    q = load_module('a100_quality_frozen', ROOT / QUALITY)
    context = q.dependencies(QUALITY_SHA)
    return q, context


def limits(*, now=None, free=None):
    import shutil
    if (time.time() if now is None else now) + OUTER_SECONDS + RESERVE_SECONDS >= DEADLINE:
        raise ValueError('Evaluation bound and archive/cleanup reserve exceed allocation deadline')
    free = shutil.disk_usage(ROOT / 'outputs').free if free is None else free
    # Four native A/B sets consume at most about350MiB before PNG compression.
    if free < 10 * 1024**3: raise ValueError('Require10GiB free for export plus8GiB reserve')
    return {'outer_seconds': OUTER_SECONDS, 'inner_seconds': OUTER_SECONDS - 120,
            'separate_archive_cleanup_reserve_seconds': RESERVE_SECONDS,
            'free_bytes': free, 'empirical_bound_guaranteed': False}


def generation_host(q, variant, spec_path, done, admitted):
    from drivebench.config import digest
    path = q.project_file(spec_path); spec = read(path)
    host_path = q.project_file(spec['metadata']); host = read(host_path)
    output = '/workspace/' + q.RESULT + '/runs/' + variant
    expected_args = [QUALITY, 'run', '--variant', variant, '--expected-sha256', QUALITY_SHA,
        '--admission', done['admission_path'], '--output', output]
    if (spec.get('python_arguments') != expected_args or spec.get('fresh_outputs') != [q.RESULT + '/runs/' + variant]
            or spec.get('allocation_deadline_epoch') != DEADLINE
            or spec.get('expected_source_identity') != SOURCE
            or spec.get('timeout_seconds') != done['time_bound']['outer_timeout_seconds']
            or host.get('status') != 'completed' or host.get('exit_code') != 0
            or host.get('specification_sha256') != sha(path) or host.get('launcher_sha256') != q.LAUNCHER_SHA
            or host.get('source_identity') != SOURCE or digest(host.get('source_files')) != SOURCE
            or host.get('runtime_identity') != admitted['runtime_identity']
            or host.get('input_file_hashes') != spec.get('expected_file_hashes')
            or host.get('source_unchanged') is not True or host.get('inputs_unchanged') is not True
            or host.get('client_stop_error') is not None or not q.cleanup_complete(host.get('cleanup', []))
            or host.get('allocation_deadline_epoch') != DEADLINE
            or not q.epoch(host['started_at']) <= q.epoch(done['started_at']) < q.epoch(done['finished_at']) <= q.epoch(host['finished_at'])):
        raise ValueError('Generation host command, completion, provenance or cleanup differs')
    pins = spec.get('expected_file_hashes', {})
    for required in (QUALITY, q.REL + '/configs/' + variant + '.json', q.LAUNCHER):
        if pins.get(required) != sha(q.project_file(required)):
            raise ValueError('Generation host missing required code/config pin')
    q.check_files(pins)
    bound = q.bindings([path, host_path]); bound.update({str(q.project_file(p)): h for p, h in pins.items()})
    return bound


def generation(q, context, admitted, strict, variant, host_spec, baseline=None):
    import torch
    config_path, cfg = q.config_checked(context, variant)
    records, wanted, observed = q.smoke_inputs(cfg)
    output = ROOT / q.RESULT / 'runs' / variant
    done_path = q.project_file(output / 'completed.json'); done = read(done_path)
    actual = q.run_artifacts(output, cfg, wanted, admitted)
    required = {**strict['artifact_sha256'], **observed, **q.bindings([ROOT / QUALITY, config_path])}
    if variant != 'eager50':
        if baseline is None: raise ValueError('Baseline evidence required')
        required.update(baseline['baseline_artifact_sha256'])
    admission_path = q.project_file(done['admission_path']); admission = read(admission_path)
    bound = q.estimate(strict['strict_elapsed_seconds'], None if variant == 'eager50' else baseline['baseline'])
    if (done.get('type') != 'a100_math_quality_generation_v1' or done.get('status') != 'completed'
            or done.get('variant') != variant or done.get('producer_sha256') != QUALITY_SHA
            or done.get('source_identity') != SOURCE or done.get('strict_prerequisite') != strict
            or done.get('platform_gate') != admitted or done.get('result') != actual
            or done.get('algorithm_policy') != context.runtime.declaration(context.base)
            or done.get('allocation_deadline_epoch') != DEADLINE or done.get('time_bound') != bound
            or done.get('inputs_and_source_unchanged') is not True or done.get('platform_restored') is not True
            or any(done.get(k) is not False for k in ('quality_preserving', 'headline_eligible', 'legacy_numerical_passed',
                'variant_exact_agreement_claimed', 'h100_reference_eligible'))
            or done.get('admission_sha256') != sha(admission_path)
            or admission.get('type') != 'a100_math_quality_admission_v1' or admission.get('variant') != variant
            or admission.get('producer_sha256') != QUALITY_SHA or admission.get('source_identity') != SOURCE
            or admission.get('strict_prerequisite') != strict or admission.get('platform_gate') != admitted
            or admission.get('allocation_deadline_epoch') != DEADLINE or admission.get('input_sha256') != required
            or admission.get('time_bound') != bound):
        raise ValueError('Complete generation sidecar/admission/policy evidence differs')
    q.lifecycle(done.get('policy_execution', {}), context.ab.snapshot(torch, context.base))
    required.update(q.bindings([admission_path]))
    if done.get('input_sha256') != required: raise ValueError('Generation input inventory differs')
    q.check_files(required)
    artifacts = q.bindings([done_path])
    artifacts.update({str(output / p): h for p, h in actual['artifact_sha256'].items()})
    # Preserve exact baseline inventory used by the frozen approximation admission.
    base = {'result': actual, 'elapsed_seconds': q.elapsed(done), 'artifact_sha256': artifacts}
    hosts = generation_host(q, variant, host_spec, done, admitted)
    from drivebench.evaluation.io import load_run
    return {'run': load_run(output), 'records': records, 'wanted': wanted,
        'bindings': {**required, **artifacts, **hosts}, 'baseline': base,
        'baseline_artifact_sha256': artifacts, 'completion': done}


def future_inputs(q, item):
    from drivebench.data.manifests import verify_input_checksums, resolve_frame
    from drivebench.evaluation.io import output_path
    records, run = item['records'], item['run']
    bound, pairs = {}, {}
    for record, row in zip(records, run.measurements):
        verify_input_checksums(record, ROOT / 'data/nuscenes', include_future=True)
        candidate = row['candidates'][0]
        future = [str(resolve_frame(ROOT / 'data/nuscenes', p)) for p in record['future_frames']]
        if len(future) != 24 or candidate.get('ground_truth_paths') != future:
            raise ValueError('All24 recorded future paths/order must match the fixed manifest')
        for relative in record['observed_frames'] + record['future_frames']:
            path = resolve_frame(ROOT / 'data/nuscenes', relative)
            bound[str(q.project_file(path))] = record['frame_checksums'][relative]
        path = output_path(run, candidate['frames_path'])
        key = (row['request_id'], row['repeat'], candidate['candidate_id'], candidate['seed'])
        if key in pairs: raise ValueError('Duplicate evaluation pair')
        pairs[key] = sha(path)
    if len(pairs) != 4: raise ValueError('Exactly four retained pairs required')
    q.check_files(bound)
    return bound, pairs


def prepare(q, context, variant, specs):
    selected = ['eager50'] + ([variant] if variant else [])
    if set(specs) != set(selected): raise ValueError('Provide exactly the required generation host specifications')
    admitted = context.verifier.gate(context.original, DEADLINE)
    strict = q.strict_prerequisite(context, admitted)
    items, bindings, pair_hashes = {}, {}, {}
    for name in selected:
        item = generation(q, context, admitted, strict, name, specs[name], items.get('eager50'))
        future, pairs = future_inputs(q, item)
        items[name] = item; bindings.update(item['bindings']); bindings.update(future); pair_hashes[name] = pairs
    if variant and set(pair_hashes[variant]) != set(pair_hashes['eager50']): raise ValueError('Pair identities differ')
    bindings.update(q.bindings([__file__, ROOT / NATIVE, ROOT / TOLERANCES]))
    if sha(ROOT / TOLERANCES) != TOLERANCES_SHA: raise ValueError('Uncalibrated default policy changed')
    return {'items': items, 'bindings': bindings, 'pairs': pair_hashes,
        'admitted': admitted, 'strict': strict, 'variant': variant}


@contextmanager
def isolated_lpips():
    target = str(ROOT / 'outputs/evaluation-packages/lpips')
    values = {'PYTHONPATH': os.pathsep.join((target, str(ROOT))),
        'TORCH_HOME': str(ROOT / 'outputs/evaluation-packages/torch-home'),
        'DRIVEBENCH_EVALUATION_SETUP_RECEIPT': str(ROOT / SETUP)}
    prior = {key: os.environ.get(key) for key in values}; old_path = sys.path[:]
    try:
        os.environ.update(values); sys.path[:] = [target, str(ROOT), *[p for p in old_path if p not in (target, str(ROOT))]]
        yield values
    finally:
        sys.path[:] = old_path
        for key, value in prior.items():
            if value is None: os.environ.pop(key, None)
            else: os.environ[key] = value


def perceptual_setup(q, expected):
    from drivebench.evaluation.metrics import verify_perceptual_setup
    path = q.project_file(SETUP)
    if sha(path) != expected or sha(ROOT / SETUP_HELPER) != SETUP_HELPER_SHA:
        raise ValueError('Fresh A100 LPIPS setup or producer pin differs')
    setup = verify_perceptual_setup(path)
    cpu = setup['cpu_verification']
    if (setup.get('python_executable') != sys.executable
            or setup.get('package', {}).get('target') != str(ROOT / 'outputs/evaluation-packages/lpips')
            or setup.get('script', {}).get('sha256') != SETUP_HELPER_SHA
            or setup.get('no_dependency_installation') is not True
            or setup.get('frozen_inference_environment_mutated') is not False
            or cpu.get('cuda_initialized_after') is not False or cpu.get('cuda_initialized_before') is not False
            or cpu.get('network_blocked_during_model_construction_and_forward') is not True
            or cpu.get('model_state_dict_sha256') != 'cb2726fb57b9d7f1f7a7eb26f83bf40ef15f421885676460a0b2caa020b74d85'):
        raise ValueError('LPIPS receipt must describe current A100 interpreter and pinned isolated assets')
    import importlib.metadata
    if any(importlib.metadata.version(k) != v for k, v in cpu['versions'].items()):
        raise ValueError('Current evaluation package versions differ from fresh setup')
    bound = q.bindings([path, ROOT / SETUP_HELPER])
    for name, data in setup['package']['files'].items(): bound[str(Path(setup['package']['target']) / name)] = data['sha256']
    for key in ('wheel', 'alexnet_cached_weight', 'lpips_learned_weight'):
        bound[setup[key]['path']] = setup[key]['sha256']
    for data in cpu['source_files'].values(): bound[data['path']] = data['sha256']
    q.check_files(bound)
    return setup, bound


def baseline_ground_truth(plan):
    import numpy as np
    from drivebench.evaluate import _ground_truth, aggregate_pairs
    from drivebench.evaluation.metrics import as_video, compare_videos
    from drivebench.evaluation.io import output_path
    from drivebench.data.preprocessing import PREPROCESS_VERSION
    item = plan['items']['eager50']; pairs = []
    for row in item['run'].measurements:
        candidate = row['candidates'][0]; path = output_path(item['run'], candidate['frames_path'])
        if candidate.get('conditioning_frames') != 1 or candidate.get('action_kind') != 'recorded':
            raise ValueError('Recorded candidate and one conditioning frame required')
        future = as_video(np.load(path, mmap_mode='r', allow_pickle=False))[1:]
        truth = _ground_truth(candidate, future.shape)
        if truth is None or len(future) != 24: raise ValueError('Complete recorded future required')
        metrics = compare_videos(truth, future)
        pairs.append({'request_id': row['request_id'], 'scene_id': row['scene_id'], 'repeat': row['repeat'],
            'candidate_id': candidate['candidate_id'], 'seed': candidate['seed'], 'action_kind': 'recorded',
            'trajectory_sha256': candidate['trajectory_sha256'], 'conditioning_frames': 1,
            'output_sha256': sha(path), 'ground_truth_sha256': [sha(p) for p in candidate['ground_truth_paths']], 'metrics': metrics})
        del future, truth
    return {'type': 'a100_four_scene_baseline_recorded_future_v1', 'source_identity': SOURCE,
        'manifest_role': 'development_subset', 'manifest_sha256': item['run'].config['data']['manifest_sha256'],
        'pair_count': len(pairs), 'future_frames_per_pair': 24, 'preprocessing': PREPROCESS_VERSION,
        'computation': 'Unchanged _ground_truth and compare_videos; observed frame excluded',
        'pairs': pairs, 'aggregates': aggregate_pairs(pairs), 'lpips_computed': False,
        'quality_preserving': False, 'headline_eligible': False, 'scope': SCOPE}


def validate_result(plan, result, setup_sha, device):
    from drivebench.evaluation.io import artifact_sha256
    variant = plan['variant']; metric = result.get('perceptual', {})
    if (metric.get('status') != 'available' or metric.get('provenance', {}).get('device') != device
            or metric['provenance'].get('setup_receipt_sha256') != setup_sha
            or metric['provenance'].get('state_dict_sha256') != 'cb2726fb57b9d7f1f7a7eb26f83bf40ef15f421885676460a0b2caa020b74d85'
            or len(result.get('runs', [])) != 1 or result.get('review', {}).get('cases') != 4
            or any(result.get('tolerances', {}).get(k) is not False for k in ('frozen', 'calibrated'))):
        raise ValueError('Complete isolated LPIPS/unrated four-case evidence required')
    entry = result['runs'][0]
    if (entry.get('run') != variant or entry.get('reference') != 'eager50'
            or entry.get('complete_pairs') is not True or entry.get('matched_pairs') != 4
            or len(entry.get('pairs', [])) != 4 or entry.get('errors') or entry.get('action_response') != []
            or entry.get('gate', {}).get('quality_preserving') is not False
            or entry.get('artifact_sha256') != artifact_sha256(plan['items'][variant]['run'])
            or result.get('reference_artifact_sha256') != artifact_sha256(plan['items']['eager50']['run'])):
        raise ValueError('Comparison coverage, identities, visible false gate or artifact bindings differ')
    seen = set()
    for pair in entry['pairs']:
        key = tuple(pair[k] for k in ('request_id', 'repeat', 'candidate_id', 'seed'))
        lpips = pair.get('metrics', {}).get('lpips')
        if (key in seen or key not in plan['pairs']['eager50']
                or pair.get('reference_output_sha256') != plan['pairs']['eager50'][key]
                or pair.get('candidate_output_sha256') != plan['pairs'][variant].get(key)
                or type(lpips) not in (int, float) or not math.isfinite(lpips) or lpips < 0
                or not isinstance(pair.get('recorded_ground_truth'), dict)):
            raise ValueError('Pair metrics, raw array hashes or recorded-future coverage differ')
        seen.add(key)
    return result


def execute(command, variant, specs, output, expected, setup_sha=None, device='cuda'):
    output = Path(output)
    name = variant + '-vs-eager50' if command == 'compare' else 'eager50-recorded-future'
    if output.resolve() != (ROOT / DEST / name).resolve() or output.exists():
        raise ValueError('Use exact fresh evaluation destination')
    if command == 'compare' and (variant not in ('eager40', 'feature2') or not setup_sha):
        raise ValueError('Fixed candidate and pinned fresh A100 setup receipt required')
    if command == 'baseline-gt' and variant is not None: raise ValueError('Baseline diagnostic has no candidate variant')
    bound = limits(); q, context = dependencies(expected)
    output.mkdir(parents=True)
    receipt = {'type': 'a100_math_quality_evaluation_v1', 'status': 'running', 'command': command,
        'variant': variant, 'started_at': datetime.now(timezone.utc).isoformat(), 'producer_sha256': expected,
        'source_identity': SOURCE, 'scope': SCOPE, 'bounds': bound, 'allocation_deadline_epoch': DEADLINE,
        'legacy_numerical_passed': False, 'quality_preserving': False, 'headline_eligible': False,
        'h100_reference_eligible': False, 'action_response_evaluated': False, 'private_attribution_printed': False,
        'evaluation_policy': 'Isolated LPIPS/alex; inference SDPA policy is generation provenance, not an LPIPS equivalence claim.'}
    save(output / 'attempt.json', receipt)
    previous = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(EvaluationTimeout('Evaluation time bound expired')))
    signal.alarm(OUTER_SECONDS - 120)
    try:
        plan = prepare(q, context, variant, specs)
        receipt.update(platform_gate=plan['admitted'], strict_prerequisite=plan['strict'],
            generation_policy=context.runtime.declaration(context.base), input_sha256=plan['bindings'])
        if command == 'baseline-gt':
            save(output / 'recorded-future.json', baseline_ground_truth(plan))
        else:
            with isolated_lpips() as env:
                setup, lpips_bound = perceptual_setup(q, setup_sha)
                plan['bindings'].update(lpips_bound)
                receipt.update(evaluation_environment=env, setup_receipt_sha256=setup_sha)
                from drivebench.evaluate import evaluate_runs
                metric_path = output / 'evaluation.json'
                result = evaluate_runs(plan['items'][variant]['run'].path,
                    reference=plan['items']['eager50']['run'].path, tolerances_path=ROOT / TOLERANCES,
                    lpips=True, lpips_device=device, action_response=False, review_limit=4, output=metric_path)
                validate_result(plan, result, setup_sha, device)
                review = Path(result['review']['index']).parent
                review_files = {str(p): sha(p) for p in review.rglob('*') if p.is_file()}
                metric_receipt = output / 'metric-receipt.json'
                save(metric_receipt, {'evaluation_sha256': sha(metric_path), 'review_artifact_sha256': review_files})
                registry = {str(path): checksum for item in plan['items'].values()
                    for path, checksum in item['bindings'].items() if path.endswith('.npy')}
                native = load_module('a100_quality_native_export', ROOT / NATIVE)
                bundle = native.export_comparison(metric_path, metric_receipt, registry, output / 'native-review',
                    minimum_free_bytes=8 * 1024**3)
                receipt['native_review_public_index'] = str(bundle / 'public/index.html')
                receipt['review_cases'] = 4
        # Isolation restored before full platform comparison: LPIPS package presence must not be
        # silently merged into the inference identity. GPU kernels here do not generate futures.
        current = context.verifier.gate(context.original, DEADLINE)
        if current != plan['admitted'] or q.strict_prerequisite(context, current) != plan['strict']:
            raise ValueError('Platform/strict evidence changed during evaluation')
        q.check_files(plan['bindings'])
        receipt.update(status='completed', inputs_and_source_unchanged=True, platform_restored=True)
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['baseline-gt', 'compare'])
    p.add_argument('--variant', choices=['eager40', 'feature2'])
    p.add_argument('--generation-spec', action='append', default=[], metavar='VARIANT=PATH')
    p.add_argument('--setup-sha256'); p.add_argument('--lpips-device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--expected-sha256', required=True); p.add_argument('--output', required=True)
    args = p.parse_args(argv)
    specs = {}
    for value in args.generation_spec:
        key, sep, path = value.partition('=')
        if not sep or key in specs or key not in ('eager50', 'eager40', 'feature2'): p.error('Unique VARIANT=PATH bindings required')
        specs[key] = path
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'): os.environ[key] = '4'
    return execute(args.command, args.variant, specs, args.output, args.expected_sha256, args.setup_sha256, args.lpips_device)


if __name__ == '__main__': raise SystemExit(main())
