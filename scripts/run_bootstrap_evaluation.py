#!/usr/bin/env python3
"""Run one frozen, descriptive 32-clip LPIPS comparison; queue/timeout stay external."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys

SOURCE = 'd8a32813492c0939a0a91573a945c914938972419946065e6822f6209fa470cb'
UPSTREAM_SOURCE = '2a8fed43b4f2d13e94f9310e9078cfada379f19d8f2f74107fe1066991b7ff00'
RECIPE_SHA = 'd60c904ab65f80120b00ba07e287d2cb8e0b758f829ef378337190bbee9e2d91'
SETUP_SHA = '3327e3d117d02a88f83e6665c9a222f9de9acd322e52140fcd7f3d3990755d4f'
TOLERANCES_SHA = '1fd3a6789b7a49a81723f9a67aa841683c9ac65b30293e6aded3c3575fa8309b'
VARIANTS = ('compiled-default', 'compiled-conditioning')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


@contextmanager
def isolated_environment(root):
    """Set import/thread/cache policy before importing any numerical libraries."""
    if any(name in sys.modules for name in ('numpy', 'torch', 'lpips', 'drivebench.evaluate')):
        raise RuntimeError('Evaluation requires a fresh process before numerical-library imports')
    target = root / 'outputs/evaluation-packages/lpips'
    environment = {
        'PYTHONPATH': os.pathsep.join((str(target), str(root))),
        'TORCH_HOME': str(root / 'outputs/evaluation-packages/torch-home'),
        'DRIVEBENCH_EVALUATION_SETUP_RECEIPT': str(root / 'outputs/evaluation-packages/setup-receipt.json'),
        **{name: '4' for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')},
    }
    old = {key: os.environ.get(key) for key in environment}
    old_path = sys.path[:]
    try:
        os.environ.update(environment)
        sys.path[:] = [str(target), str(root), *[p for p in old_path if p and p not in (str(target), str(root))]]
        yield environment
    finally:
        sys.path[:] = old_path
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def validate_host_receipt(path, expected_source, output):
    from drivebench.config import digest
    receipt = read(path)
    if (receipt.get('status') != 'completed' or receipt.get('exit_code') != 0
            or receipt.get('source_unchanged') is not True or not receipt.get('finished_at')
            or receipt.get('source_identity') != expected_source
            or not receipt.get('source_files') or digest(receipt['source_files']) != expected_source
            or receipt.get('inputs_unchanged', True) is not True):
        raise ValueError('Generation host receipt is incomplete or has the wrong source/output: ' + str(path))
    args = receipt.get('argv', [])
    name = Path(output).name
    if name == 'upstream':
        allowed = ('configs/study_upstream_development.yaml', '/workspace/configs/study_upstream_development.yaml')
        valid_command = ('drivebench.benchmark' in args and '--config' in args
                         and args[args.index('--config') + 1] in allowed)
    else:
        valid_command = (name in VARIANTS and receipt.get('inputs_unchanged') is True
                         and any(Path(value).name == 'run_compiled_bootstrap.py' for value in args)
                         and '--variant' in args and args[args.index('--variant') + 1] == name
                         and '--hostspec' in args and Path(args[args.index('--hostspec') + 1]).name
                         == 'study_bootstrap_compiled_hostspec.json')
    if not valid_command:
        raise ValueError('Generation host command differs from the declared run')
    return receipt


def run_inputs(run, records, root, *, compiled):
    """Check primary workload and retain a content binding for every input artifact."""
    import numpy as np
    from drivebench.config import digest
    from drivebench.data.manifests import inference_request, verify_input_checksums
    from drivebench.evaluation.io import output_path
    summary = run.summary
    if (any(summary.get(key) != 32 for key in ('measured_requests', 'successful_requests', 'completed_futures'))
            or summary.get('oom_attempts') != 0 or summary.get('profiled') is not False
            or summary.get('recompiled_during_measurement') is not False
            or summary.get('headline_eligible') is not False
            or read(run.path / 'cold_start.json').get('status') != 'ready'):
        raise ValueError('Require complete, unprofiled, stable 32-request generation')
    bindings = {str(path): sha(path) for path in sorted(run.path.iterdir()) if path.is_file()}
    pairs = {}
    for record, row in zip(records, run.measurements):
        request = inference_request(record, root / 'data/nuscenes', candidate_count=1)
        if (row.get('scene_id') != request['scene_id'] or row.get('repeat') != 0
                or row.get('candidate_count') != 1 or row.get('batch_size') != 1
                or row.get('effective_config', {}).get('actual_batch_sizes') != [1]
                or row.get('denoiser_evaluations') != 50 or row.get('fallbacks') or row.get('failures')
                or row.get('cache_stats', {}).get('oom_attempts') or len(row.get('candidates', [])) != 1):
            raise ValueError('Generation does not match the single-candidate primary workload')
        if compiled:
            runtime = row.get('cache_stats', {}).get('compiler_runtime') or {}
            if (runtime.get('passed') is not True or runtime.get('restoration_complete') is not True
                    or runtime.get('graphs_created_by_runtime', 0) < 1 or runtime.get('kernels_created_by_runtime', 0) < 1
                    or runtime.get('after', {}).get('unimplemented')):
                raise ValueError('Missing successful compiler/runtime restoration evidence')
        candidate = row['candidates'][0]
        expected = {'candidate_id': request['candidate_ids'][0], 'seed': request['seeds'][0],
                    'action_kind': request['action_kinds'][0], 'conditioning_frames': 1,
                    'trajectory_sha256': digest(np.asarray(request['trajectories'][0]).tolist())}
        if any(candidate.get(key) != value for key, value in expected.items()):
            raise ValueError('Saved candidate identity differs from the frozen manifest')
        future = [str(root / 'data/nuscenes' / path) for path in record['future_frames']]
        if candidate.get('ground_truth_paths') != future or len(future) != 24:
            raise ValueError('Recorded future paths differ from the frozen manifest')
        verify_input_checksums(record, root / 'data/nuscenes', include_future=True)
        for relative in record['observed_frames'] + record['future_frames']:
            bindings[str(root / 'data/nuscenes' / relative)] = record['frame_checksums'][relative]
        path = output_path(run, candidate['frames_path'])
        video = np.load(path, mmap_mode='r', allow_pickle=False)
        if video.shape != (25, 576, 1024, 3) or video.dtype != np.float32:
            raise ValueError('Saved primary float output shape/dtype differs')
        del video
        checksum = sha(path)
        bindings[str(path)] = checksum
        key = (row['request_id'], 0, candidate['candidate_id'], candidate['seed'])
        if key in pairs:
            raise ValueError('Duplicate generated pair')
        pairs[key] = checksum
    if len(pairs) != 32:
        raise ValueError('Require all 32 retained arrays')
    return bindings, pairs


def validate_inputs(hostspec, variant, root):
    # Imports deliberately occur only after isolated_environment has set policy.
    from scripts.run_compiled_bootstrap import config_bindings, project_path
    from drivebench.benchmarks.environment import source_identity
    from drivebench.config import digest, load_config
    from drivebench.data.manifests import load_manifest
    from drivebench.evaluation.metrics import verify_perceptual_setup
    from drivebench.study import _validate_completed_run
    hostspec = hostspec.resolve()
    if (variant not in VARIANTS or not hostspec.is_relative_to(root)
            or sha(hostspec) != RECIPE_SHA or source_identity() != SOURCE):
        raise ValueError('Frozen recipe/source/variant differs')
    recipe = read(hostspec)
    selected = next(item for item in recipe['runs'] if item['adoption_variant'] == variant)
    config = load_config(root / selected['config'])
    if (config_bindings(root, root / selected['config']) != selected['inherited_config_sha256']
            or digest(config) != selected['resolved_config_digest']):
        raise ValueError('Inherited or resolved configuration changed')
    bindings = {str(root / path): value for path, value in selected['inherited_config_sha256'].items()}
    bindings.update({str(root / path): value for path, value in recipe['expected_environment_artifact_sha256'].items()})
    manifest = project_path(root, config['data']['manifest'])
    bindings.update({str(hostspec): RECIPE_SHA, str(manifest): recipe['expected_manifest_sha256'],
                    str(manifest) + '.meta.json': recipe['expected_manifest_metadata_sha256']})
    tolerances = root / 'configs/study_bootstrap_unfrozen_tolerances.yaml'
    setup_path = root / 'outputs/evaluation-packages/setup-receipt.json'
    bindings.update({str(tolerances): TOLERANCES_SHA, str(setup_path): SETUP_SHA})
    check_bindings(bindings)
    setup = verify_perceptual_setup(setup_path)
    if (setup.get('python_executable') != sys.executable
            or setup['package']['target'] != str(root / 'outputs/evaluation-packages/lpips')
            or setup.get('no_dependency_installation') is not True
            or setup.get('frozen_inference_environment_mutated') is not False):
        raise ValueError('Unexpected isolated LPIPS installation/interpreter')
    for relative, item in setup['package']['files'].items():
        bindings[str(Path(setup['package']['target']) / relative)] = item['sha256']
    for key in ('wheel', 'alexnet_cached_weight', 'lpips_learned_weight'):
        bindings[setup[key]['path']] = setup[key]['sha256']
    records = load_manifest(manifest)
    metadata = read(str(manifest) + '.meta.json')
    if len(records) != 32 or metadata.get('partition_role') != 'development' or metadata.get('synthetic'):
        raise ValueError('Require the original 32 real development records')
    runs, pair_hashes = {}, {}
    for name, cfg, source, receipt_path in [
        (variant, config, SOURCE, root / selected['host_receipt']),
        ('upstream', load_config(root / 'configs/study_upstream_development.yaml'), UPSTREAM_SOURCE,
         root / 'outputs/host/study-upstream-development-attempt-1.json'),
    ]:
        directory = root / 'outputs/study-bootstrap' / name
        validate_host_receipt(receipt_path, source, '/workspace/' + str(directory.relative_to(root)))
        bindings[str(receipt_path)] = sha(receipt_path)
        run = _validate_completed_run(directory, cfg, records)
        run_bindings, pairs = run_inputs(run, records, root, compiled=name != 'upstream')
        bindings.update(run_bindings)
        runs[name], pair_hashes[name] = run, pairs
    if set(pair_hashes[variant]) != set(pair_hashes['upstream']):
        raise ValueError('Generated pair identities differ')
    bindings[str(Path(__file__).resolve())] = sha(__file__)
    helper = root / 'scripts/run_compiled_bootstrap.py'
    bindings[str(helper)] = sha(helper)
    qualification = root / 'outputs/study-bootstrap/qualification-v1'
    output = qualification / f'{variant}-vs-upstream.json'
    review = qualification / f'{variant}-vs-upstream-review'
    receipt_path = qualification / f'{variant}-evaluation-launch.json'
    for path in (qualification, output, review, receipt_path):
        if not path.resolve().is_relative_to(root) or path.is_symlink():
            raise ValueError('Qualification path escapes the project')
    for path in (output, review, receipt_path):
        if path.exists():
            raise FileExistsError('Evaluation output/review/receipt must be fresh: ' + str(path))
    argv = ['drivebench.evaluate', '--runs', str(runs[variant].path), '--reference', str(runs['upstream'].path),
            '--tolerances', str(tolerances), '--lpips', '--lpips-device', 'cuda', '--review-limit', '4', '--output', str(output)]
    return dict(root=root, variant=variant, output=output, review=review, receipt=receipt_path,
                bindings=bindings, pair_hashes=pair_hashes, argv=argv, runs=runs)


def check_bindings(bindings):
    for path, expected in bindings.items():
        if sha(path) != expected:
            raise ValueError('Bound input changed: ' + path)


def validate_result(plan):
    from drivebench.evaluation.io import artifact_sha256
    result = read(plan['output'])
    metric = result.get('perceptual', {})
    if (metric.get('status') != 'available' or metric.get('provenance', {}).get('device') != 'cuda'
            or metric['provenance'].get('setup_receipt_sha256') != SETUP_SHA):
        raise ValueError('CUDA LPIPS was not available from the verified setup')
    if len(result.get('runs', [])) != 1:
        raise ValueError('Expected exactly one comparison')
    entry = result['runs'][0]
    if (entry.get('run') != plan['variant'] or entry.get('complete_pairs') is not True
            or entry.get('matched_pairs') != 32 or len(entry.get('pairs', [])) != 32 or entry.get('errors')
            or entry.get('gate', {}).get('quality_preserving') is not False
            or result.get('tolerances', {}).get('frozen') is not False
            or result.get('review', {}).get('cases') != 4
            or Path(result['review']['index']) != plan['review'] / 'index.html'):
        raise ValueError('Descriptive evaluation/review coverage is incomplete or incorrectly accepted')
    if (result.get('reference_artifact_sha256') != artifact_sha256(plan['runs']['upstream'])
            or entry.get('artifact_sha256') != artifact_sha256(plan['runs'][plan['variant']])):
        raise ValueError('Evaluator metadata bindings differ')
    seen = set()
    for pair in entry['pairs']:
        key = (pair['request_id'], pair['repeat'], pair['candidate_id'], pair['seed'])
        if (key in seen or key not in plan['pair_hashes']['upstream']
                or pair.get('reference_output_sha256') != plan['pair_hashes']['upstream'][key]
                or pair.get('candidate_output_sha256') != plan['pair_hashes'][plan['variant']][key]
                or not isinstance(pair.get('metrics', {}).get('lpips'), (int, float))
                or not math.isfinite(pair['metrics']['lpips']) or pair.get('recorded_ground_truth') is None):
            raise ValueError('Evaluated pair hash/LPIPS/recorded-ground-truth coverage differs')
        seen.add(key)
    return result


def execute(plan, environment, *, evaluator_main=None):
    plan['receipt'].parent.mkdir(parents=True, exist_ok=True)
    receipt = dict(schema_version=1, status='starting', started_at=datetime.now(timezone.utc).isoformat(),
                   variant=plan['variant'], source_identity=SOURCE, upstream_generation_source_identity=UPSTREAM_SOURCE,
                   environment=environment, argv=plan['argv'], bound_file_sha256=plan['bindings'],
                   descriptive_only=True, quality_acceptance_claim=False, gpu_queue_and_timeout='external; 1800s')
    # Exclusive creation also prevents two wrappers from racing on one comparison.
    with plan['receipt'].open('x') as stream:
        json.dump(receipt, stream, indent=2)
    old_argv = sys.argv
    try:
        check_bindings(plan['bindings'])
        for path in (plan['output'], plan['review']):
            if path.exists() or path.is_symlink():
                raise FileExistsError('Evaluation destination appeared after validation')
        sys.argv = plan['argv']
        main = evaluator_main or importlib.import_module('drivebench.evaluate').main
        code = main()
        if code not in (None, 0):
            raise RuntimeError('Evaluator returned a failure exit code')
        validate_result(plan)
        check_bindings(plan['bindings'])
        receipt.update(status='completed', exit_code=0, inputs_unchanged=True,
                       output_sha256=sha(plan['output']),
                       review_artifact_sha256={str(p): sha(p) for p in sorted(plan['review'].rglob('*')) if p.is_file()})
        # Hashing private_key.json binds the review artifact; its contents are not read/decoded or displayed.
        return 0
    except BaseException as error:
        receipt.update(status='failed', exit_code=1, error={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        sys.argv = old_argv
        receipt['finished_at'] = datetime.now(timezone.utc).isoformat()
        temporary = plan['receipt'].with_suffix('.json.tmp')
        temporary.write_text(json.dumps(receipt, indent=2) + '\n')
        temporary.replace(plan['receipt'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hostspec', type=Path, default=Path('configs/study_bootstrap_compiled_hostspec.json'))
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    with isolated_environment(root) as environment:
        return execute(validate_inputs(args.hostspec, args.variant, root), environment)


if __name__ == '__main__':
    raise SystemExit(main())
