#!/usr/bin/env python3
"""Summarize paired recorded-future proxies without reading model arrays or review keys."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from drivebench.evaluation.metrics import bootstrap_mean

METRICS = ('mae', 'rmse', 'max_abs_error', 'temporal_derivative_mae',
           'temporal_change_error', 'freeze_fraction_increase')
INPUT_SHA = {
    'upstream-recorded-ground-truth-v1.json': '2215041529f43310266146b92296bbf96dc2ee510c5a15fe464cbc9842a7d0ed',
    'upstream-recorded-ground-truth-v1.receipt.json': 'ab93373026f3c0b3dfc0d2cbc4ccce5ebad093953af1265c03a2bf633ef7019a',
    'compiled-default-vs-upstream.json': '0ec9e5ed974d8f7533008d5f65ae4795300c91e2aab82e4908df07fe4f03d4f3',
    'compiled-conditioning-vs-upstream.json': '2c0cd6ff32c2873bdc3eaa31db642d2c465e8237e60fd3efa267d05f0922cf46',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(gt, comparisons, *, expected_count=32):
    pairs = gt['pairs']
    if (gt.get('pair_count') != expected_count or len(pairs) != expected_count
            or gt.get('diagnostic_only') is not True or gt.get('lpips_computed') is not False
            or len({p['scene_id'] for p in pairs}) != expected_count
            or len({p['candidate_id'] for p in pairs}) != expected_count):
        raise ValueError('Require complete distinct development scenes and candidate identities')
    indexed = {p['candidate_id']: p for p in pairs}
    if set(comparisons) != {'compiled-default', 'compiled-conditioning'}:
        raise ValueError('Both declared compiler comparisons are required')
    summary, details = {}, {}
    for name, document in comparisons.items():
        if (len(document['runs']) != 1 or document['reference_artifact_sha256'] != gt['run_artifact_sha256']):
            raise ValueError('Reference run provenance differs')
        run = document['runs'][0]
        if (run.get('synthetic') is not False or run.get('complete_pairs') is not True
                or run.get('errors') or len(run['pairs']) != expected_count
                or len({p['candidate_id'] for p in run['pairs']}) != expected_count
                or run['identity']['manifest_sha256'] != gt['manifest_sha256']):
            raise ValueError('Require complete real matched development comparisons')
        joined = []
        for pair in run['pairs']:
            base = indexed.get(pair['candidate_id'])
            if base is None or any(base[k] != pair[k] for k in
                    ('request_id', 'scene_id', 'repeat', 'candidate_id', 'seed', 'action_kind', 'trajectory_sha256')):
                raise ValueError('Pair identity differs')
            candidate = pair['recorded_ground_truth']
            if (base['output_sha256'] != pair['reference_output_sha256']
                    or base['metrics']['reference_temporal'] != candidate['reference_temporal']
                    or base['metrics']['candidate_temporal'] != pair['metrics']['reference_temporal']):
                raise ValueError('Source output or ground-truth/reference temporal alignment differs')
            delta = {}
            for metric in METRICS:
                values = (base['metrics'][metric], candidate[metric])
                if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
                    raise ValueError('Missing/nonfinite metric')
                delta[metric] = values[1] - values[0]
            joined.append({'candidate_id': pair['candidate_id'], 'request_id': pair['request_id'],
                           'scene_id': pair['scene_id'], 'reference_output_sha256': base['output_sha256'],
                           'candidate_output_sha256': pair['candidate_output_sha256'],
                           'upstream_recorded_ground_truth': base['metrics'],
                           'candidate_recorded_ground_truth': candidate, 'candidate_minus_upstream': delta})
        measures = {}
        for metric in METRICS:
            delta = [p['candidate_minus_upstream'][metric] for p in joined]
            measures[metric] = {'upstream_mean': sum(p['upstream_recorded_ground_truth'][metric] for p in joined) / expected_count,
                'candidate_mean': sum(p['candidate_recorded_ground_truth'][metric] for p in joined) / expected_count,
                'paired_change': {**bootstrap_mean(delta), 'minimum': min(delta), 'maximum': max(delta),
                                  'positive_clips': sum(d > 0 for d in delta),
                                  'negative_clips': sum(d < 0 for d in delta), 'zero_clips': sum(d == 0 for d in delta)}}
        measures['freeze_case_counts'] = {
            'upstream_increase_vs_recorded': sum(p['upstream_recorded_ground_truth']['freeze_fraction_increase'] > 0 for p in joined),
            'candidate_increase_vs_recorded': sum(p['candidate_recorded_ground_truth']['freeze_fraction_increase'] > 0 for p in joined),
            'same_fraction_per_clip': all(p['candidate_minus_upstream']['freeze_fraction_increase'] == 0 for p in joined)}
        summary[name], details[name] = measures, joined
    return summary, details


def write(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())


def run(qualification, output):
    qualification, output = Path(qualification).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Fresh audit output required')
    documents = {}
    for name, expected in INPUT_SHA.items():
        path = qualification / name
        if sha(path) != expected:
            raise ValueError('Input differs from independently collected metadata: ' + name)
        documents[name] = json.loads(path.read_text())
    gt = documents['upstream-recorded-ground-truth-v1.json']
    receipt = documents['upstream-recorded-ground-truth-v1.receipt.json']
    if (receipt.get('status') != 'completed' or receipt.get('output_sha256') != INPUT_SHA['upstream-recorded-ground-truth-v1.json']
            or receipt.get('raw_array_sha256') != {p['output_path']: p['output_sha256'] for p in gt['pairs']}):
        raise ValueError('Committed GT export does not bind all source outputs')
    summary, details = analyze(gt, {name: documents[name + '-vs-upstream.json']
                                  for name in ('compiled-default', 'compiled-conditioning')})
    public = {'schema_version': 1, 'scope': 'Development recorded-future numeric/temporal proxies; diagnostic only',
        'clips': 32, 'distinct_scenes': 32, 'direction': 'candidate minus upstream; positive means greater proxy error',
        'uncertainty': '2000 paired scene bootstrap resamples, seed 0; one clip per scene',
        'preprocessing': 'Existing evaluator center crop, Lanczos, RGB normalization; 24 future frames after conditioning frame',
        'comparisons': summary, 'source_metadata_sha256': INPUT_SHA, 'lpips_vs_recorded_computed': False,
        'quality_acceptance': False, 'headline_claim': False,
        'limitations': ['This analysis rechecks collected metadata, pair identities and source-output hashes, not the remote arrays or images.',
            'Error proxies are not trajectory adherence, safety, or object-motion correctness.',
            'The same seven global-freeze cases occur upstream and in both compiled variants; no added global-freeze case is observed.',
            'Mean MAE, RMSE and temporal-derivative changes have intervals spanning zero; no broad improvement claim.',
            'The small temporal-change-error reduction is descriptive; these multiple proxy tests do not establish perceptual improvement.']}
    output.mkdir(parents=True, exist_ok=False); (output / 'public').mkdir()
    write(output / 'public/summary.json', public)
    write(output / 'paired-analysis-private.json', {'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
        'metadata_files_sha256': {str(qualification / n): s for n, s in INPUT_SHA.items()},
        'analysis_source_sha256': sha(__file__), 'source_raw_arrays_rehashed_here': False,
        'private_keys_or_ratings_read': False, 'summary': public, 'pairs': details})
    for name, expected in INPUT_SHA.items():
        if sha(qualification / name) != expected:
            raise ValueError('Input changed during audit')
    write(output / 'receipt.json', {'schema_version': 1, 'status': 'completed_metadata_audit',
        'analysis_source_sha256': sha(__file__), 'input_sha256': INPUT_SHA,
        'output_sha256': {str(p.relative_to(output)): sha(p) for p in output.rglob('*.json')}})
    for directory in (output / 'public', output):
        fd = os.open(directory, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    return public


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualification', default='outputs/study-bootstrap/qualification-v1')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = run(args.qualification, args.output)
    print(json.dumps({'clips': result['clips'], 'output': args.output, 'headline_claim': False}))
