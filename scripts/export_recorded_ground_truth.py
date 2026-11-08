#!/usr/bin/env python3
"""CPU-only recorded-future diagnostics, committed before raw-video retirement."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SOURCE = 'd8a32813492c0939a0a91573a945c914938972419946065e6822f6209fa470cb'
ROLES = {'development', 'held_out_evaluation', 'held_out_evaluation_second_seed',
         'development_action_subset', 'held_out_evaluation_action_subset'}
ROLES |= {role + '_quality_shard' for role in tuple(ROLES)}


def read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError('Expected JSON object')
    return value


def write_exclusive(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def verify_export(output, *, allow_retired_arrays=True):
    """Verify committed metrics; missing retired arrays remain explicitly unavailable."""
    from drivebench.config import file_sha256
    output = Path(output).resolve()
    receipt = read(output.with_suffix('.receipt.json'))
    if (receipt.get('receipt_type') != 'recorded_ground_truth_export_v1'
            or receipt.get('status') != 'completed' or receipt.get('output') != str(output)
            or receipt.get('output_sha256') != file_sha256(output)):
        raise ValueError('Missing or changed committed ground-truth export')
    document = read(output)
    if (document.get('schema_version') != 1 or document.get('diagnostic_only') is not True
            or document.get('headline_claim') is not False or document.get('lpips_computed') is not False
            or not document.get('pairs') or document.get('pair_count') != len(document['pairs'])
            or document.get('source_identity') != SOURCE):
        raise ValueError('Invalid recorded-future diagnostic document')
    for path, expected in receipt['bound_file_sha256'].items():
        if file_sha256(path) != expected:
            raise ValueError('Bound diagnostic input/source changed: ' + path)
    required = {str(Path(document['run']) / name): value for name, value in document['run_artifact_sha256'].items()}
    required[str(Path(__file__).resolve())] = file_sha256(__file__)
    required[document['manifest']] = document['manifest_sha256']
    if any(receipt['bound_file_sha256'].get(path) != expected for path, expected in required.items()):
        raise ValueError('Committed diagnostic is missing its run/manifest/helper bindings')
    if receipt.get('raw_array_sha256') != {p['output_path']: p['output_sha256'] for p in document['pairs']}:
        raise ValueError('Raw-array receipt does not cover the committed metric pairs')
    missing = []
    for path, expected in receipt['raw_array_sha256'].items():
        if not Path(path).exists():
            if not allow_retired_arrays:
                raise FileNotFoundError('Raw source array is no longer available: ' + path)
            missing.append(path)
        elif file_sha256(path) != expected:
            raise ValueError('Available raw source array changed: ' + path)
    return {'output': str(output), 'output_sha256': receipt['output_sha256'],
            'receipt': str(output.with_suffix('.receipt.json')),
            'receipt_sha256': file_sha256(output.with_suffix('.receipt.json')),
            'pair_count': document['pair_count'], 'missing_raw_arrays': missing,
            'raw_arrays_available': not missing, 'verified_committed_metrics': True,
            'retirement_scope': 'Missing arrays were hashed before this metric commit; no recovery or new metric claim.'}


def export_run(run_path, output, *, expected_manifest=None):
    import numpy as np
    from drivebench.benchmarks.environment import source_identity
    from drivebench.config import digest, file_sha256
    from drivebench.data.manifests import load_manifest, resolve_frame
    from drivebench.data.preprocessing import PREPROCESS_VERSION
    from drivebench.evaluate import _ground_truth, aggregate_pairs
    from drivebench.evaluation.io import artifact_sha256, candidates, load_run, output_path
    from drivebench.evaluation.metrics import as_video, compare_videos

    output, run_path = Path(output).resolve(), Path(run_path).resolve()
    receipt_path = output.with_suffix('.receipt.json')
    if output == receipt_path:
        raise ValueError('Metric output must have a distinct receipt path')
    if output.exists() or receipt_path.exists():
        result = verify_export(output)
        saved = read(output)
        if Path(saved['run']).resolve() != run_path:
            raise ValueError('Existing diagnostic belongs to a different run')
        if expected_manifest is not None and file_sha256(expected_manifest) != saved['manifest_sha256']:
            raise ValueError('Existing diagnostic belongs to a different manifest')
        return result
    if source_identity() != SOURCE:
        raise ValueError('Recorded-future exporter requires the frozen evaluator package')
    run = load_run(run_path)
    if (run.synthetic or run.summary.get('failures') or run.summary.get('run_failure')
            or run.summary.get('status', 'ok') not in {'ok', 'success'}
            or not run.measurements or any(row.get('status') != 'ok' for row in run.measurements)):
        raise ValueError('Require a complete successful real run')
    manifest = Path(run.config['data']['manifest']).resolve()
    manifest_meta = manifest.with_suffix(manifest.suffix + '.meta.json')
    metadata = read(manifest_meta)
    manifest_hash = file_sha256(manifest)
    if (manifest_hash != run.config['data'].get('manifest_sha256')
            or metadata.get('manifest_sha256') != manifest_hash
            or metadata.get('synthetic') is True
            or metadata.get('partition_role') not in ROLES
            or run.config['data'].get('verify_checksums') is not True):
        raise ValueError('Manifest checksum/real partition provenance differs')
    if expected_manifest is not None and file_sha256(expected_manifest) != manifest_hash:
        raise ValueError('Expected manifest differs')
    records = load_manifest(manifest)
    if (len(records) != len(run.measurements) or [r['request_id'] for r in records] != [r['request_id'] for r in run.measurements]
            or metadata.get('clip_count') != len(records)
            or metadata.get('scene_ids_sha256') != digest(sorted({r['scene_id'] for r in records}))):
        raise ValueError('Export must cover every declared request once and in manifest order')
    if metadata['partition_role'] == 'development' and len(records) != 32:
        raise ValueError('Whole bootstrap development export requires all 32 clips')
    root = Path(run.config['data']['root']).resolve()
    indexed = candidates(run)
    if len(indexed) != sum(len(r.get('candidates', [])) for r in run.measurements):
        raise ValueError('All measured candidates must have been retained')
    bindings = {str(run.path / name): checksum for name, checksum in artifact_sha256(run).items()}
    bindings.update({str(manifest): manifest_hash, str(manifest_meta): file_sha256(manifest_meta),
                     str(Path(__file__).resolve()): file_sha256(__file__)})
    package = ROOT / 'drivebench'
    sources = {str(p.relative_to(package)): file_sha256(p) for p in sorted(package.rglob('*.py'))}
    if digest(sources) != SOURCE:
        raise ValueError('Evaluator package source inventory changed')
    bindings.update({str(package / name): value for name, value in sources.items()})
    pairs, arrays, excluded = [], {}, 0
    generation = run.config['generation']
    for record, row in zip(records, run.measurements):
        if row['scene_id'] != record['scene_id'] or row.get('repeat', 0) != 0:
            raise ValueError('Scene/repeat identity differs from the manifest')
        recorded = [c for c in row['candidates'] if c.get('action_kind') == 'recorded']
        if len(recorded) != 1:
            raise ValueError('Exactly one recorded-action candidate per request is required')
        excluded += len(row['candidates']) - 1
        candidate = recorded[0]
        from drivebench.data.trajectories import candidate_seed
        wanted_seed = (record.get('seeds') or [candidate_seed(record['request_id'], 0, record.get('base_seed', 23))])[0]
        if candidate.get('candidate_id') != record['request_id'] + ':c00' or candidate.get('seed') != wanted_seed:
            raise ValueError('Recorded candidate identity/seed differs from the manifest')
        count = candidate.get('conditioning_frames', generation.get('conditioning_frames', 1))
        if type(count) is not int or count < 1 or count != generation.get('conditioning_frames', 1):
            raise ValueError('Conditioning-frame alignment differs')
        truth = [resolve_frame(root, p) for p in record['future_frames']]
        if candidate.get('ground_truth_paths') != [str(p) for p in truth]:
            raise ValueError('Recorded future order/path differs from the manifest')
        for relative, path in zip(record['future_frames'], truth):
            expected = record.get('frame_checksums', {}).get(relative)
            if not expected or file_sha256(path) != expected:
                raise ValueError('Recorded future image checksum differs')
            bindings[str(path)] = expected
        path = output_path(run, candidate['frames_path'])
        before = file_sha256(path)
        video = as_video(np.load(path, allow_pickle=False))
        wanted = (generation['total_frames'], generation['height'], generation['width'], 3)
        if video.shape != wanted or len(truth) != wanted[0] - count:
            raise ValueError('Saved array or future-frame count differs; no truncation allowed')
        future = video[count:]
        gt = _ground_truth(candidate, future.shape)
        if gt is None:
            raise ValueError('Recorded ground-truth diagnostic is unavailable')
        metrics = compare_videos(gt, future)
        if file_sha256(path) != before:
            raise ValueError('Source array changed during metric computation')
        arrays[str(path)] = before
        pairs.append({'request_id': row['request_id'], 'scene_id': row['scene_id'], 'repeat': 0,
                      'candidate_id': candidate['candidate_id'], 'seed': candidate['seed'], 'action_kind': 'recorded',
                      'trajectory_sha256': candidate.get('trajectory_sha256'), 'conditioning_frames': count,
                      'output_path': str(path), 'output_sha256': before,
                      'ground_truth_paths': [str(p) for p in truth],
                      'ground_truth_sha256': [bindings[str(p)] for p in truth], 'metrics': metrics})
        del gt, future, video
    if len(pairs) != len(records):
        raise ValueError('Recorded-future coverage is incomplete')
    for path, expected in bindings.items():
        if file_sha256(path) != expected:
            raise ValueError('Input/source changed before diagnostic commit')
    document = {'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
                'run': str(run_path), 'run_artifact_sha256': artifact_sha256(run), 'source_identity': SOURCE,
                'manifest': str(manifest), 'manifest_sha256': manifest_hash, 'manifest_metadata': metadata,
                'preprocessing': PREPROCESS_VERSION, 'computation': 'CPU NumPy/Pillow; existing _ground_truth and compare_videos',
                'scope': 'Recorded future versus generated future after removing conditioning frames; no counterfactual truth.',
                'pair_count': len(pairs), 'excluded_nonrecorded_candidates': excluded, 'pairs': pairs,
                'aggregates': aggregate_pairs(pairs), 'diagnostic_only': True, 'lpips_computed': False,
                'headline_claim': False, 'quality_gate_changed': False, 'generation_source_note':
                    'Run/config artifacts retain generation provenance; source_identity above identifies this evaluator.'}
    write_exclusive(output, document)
    write_exclusive(receipt_path, {'schema_version': 1, 'receipt_type': 'recorded_ground_truth_export_v1',
        'status': 'completed', 'completed_at': datetime.now(timezone.utc).isoformat(),
        'output': str(output), 'output_sha256': file_sha256(output),
        'bound_file_sha256': bindings, 'raw_array_sha256': arrays,
        'raw_retirement_policy': 'Committed diagnostic may be reused after missing raw arrays; available arrays and all other bindings are rechecked.'})
    return verify_export(output, allow_retired_arrays=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--expected-manifest')
    args = parser.parse_args(argv)
    # CLI is a separate CPU subprocess; importing this helper never changes a caller's GPU visibility.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '4'
    try:
        result = export_run(args.run, args.output, expected_manifest=args.expected_manifest)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f'recorded ground truth: {error}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
