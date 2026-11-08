"""Describe frozen manifest coverage without changing inference inputs or selection."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import ijson


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def rows(path):
    with Path(path).open('rb') as stream:
        yield from ijson.items(stream, 'item', use_float=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, default=Path('data/nuscenes/v1.0-trainval'))
    parser.add_argument('--manifests', type=Path, default=Path('data/manifests'))
    parser.add_argument('--output', type=Path, default=Path('reports/scene-coverage.json'))
    args = parser.parse_args()
    metadata = args.metadata
    sources = ['scene', 'sample_data', 'ego_pose', 'instance', 'category', 'sample_annotation']
    manifests = {name: [json.loads(line) for line in (args.manifests / (name + '.jsonl')).read_text().splitlines()]
                 for name in ['development', 'evaluation', 'evaluation_second_seed']}
    all_rows = [row for values in manifests.values() for row in values]
    scenes = {row['token']: row for row in rows(metadata / 'scene.json')}
    lidar_tokens = {row['trajectory_metadata']['spatial_reference_lidar_token'] for row in all_rows}
    data = {row['token']: row for row in rows(metadata / 'sample_data.json') if row['token'] in lidar_tokens}
    if set(data) != lidar_tokens:
        raise ValueError('Initial LiDAR metadata join is incomplete')
    poses_needed = {row['ego_pose_token'] for row in data.values()}
    poses = {row['token']: row for row in rows(metadata / 'ego_pose.json') if row['token'] in poses_needed}
    if set(poses) != poses_needed:
        raise ValueError('Initial ego-pose metadata join is incomplete')
    for record in all_rows:
        details = record['trajectory_metadata']
        if details['units'] != 'meters' or details['axis_order'] != ['lateral', 'longitudinal']:
            raise ValueError('Unexpected trajectory coordinate convention')
        if data[details['spatial_reference_lidar_token']]['sample_token'] != details['spatial_reference_sample_token']:
            raise ValueError('Initial LiDAR/sample token mismatch')
        if scenes[record['scene_id']]['name'] != record['scene_name']:
            raise ValueError('Scene metadata mismatch')
    sample_origins = {row['sample_token']: poses[row['ego_pose_token']]['translation'] for row in data.values()}
    categories = {row['token']: row['name'] for row in rows(metadata / 'category.json')}
    instances = {row['token']: categories[row['category_token']] for row in rows(metadata / 'instance.json')}
    nearby = defaultdict(Counter)
    for row in rows(metadata / 'sample_annotation.json'):
        origin = sample_origins.get(row['sample_token'])
        if origin is None:
            continue
        dx, dy = row['translation'][0] - origin[0], row['translation'][1] - origin[1]
        if math.hypot(dx, dy) <= 30:
            nearby[row['sample_token']][instances[row['instance_token']]] += 1
    result = {'schema_version': 1, 'selection_changed': False,
              'scope': 'Offline metadata coverage; not conditioning, quality, safety or front-camera visibility evidence.',
              'rules': {'description': 'Case-insensitive literal scene-description keywords; absence means not stated. Scene-wide mentions do not establish lighting/weather in each selected clip; these sparse strata are not balanced.',
                        'object_distance': 'Annotation-center to ego-pose horizontal distance <=30 m at initial LiDAR reference timestamp; all directions, not camera visibility or distance to the sensor origin.',
                        'motion': 'Recorded trajectory endpoint displacement: stationary/slow <1 m, otherwise moving; lateral change if absolute final lateral offset >=1 m. These are workload descriptors, not a validated turning classifier.'},
              'metadata_sha256': {name: sha(metadata / (name + '.json')) for name in sources}, 'partitions': {}}
    for name, records in manifests.items():
        entries, counts = [], Counter()
        for record in records:
            desc = scenes[record['scene_id']]['description']
            low = desc.lower()
            trajectory = record['trajectory']
            endpoint = trajectory[-1] if isinstance(trajectory[-1], list) else trajectory[-2:]
            offset = math.hypot(*endpoint)
            labels = ['description:night-mentioned' if 'night' in low else 'description:night-not-stated',
                      'description:rain-mentioned' if 'rain' in low else 'description:rain-not-stated',
                      'motion:slow' if offset < 1 else 'motion:moving']
            if abs(endpoint[0]) >= 1:
                labels.append('motion:lateral-offset-ge1m')
            objects = nearby[record['trajectory_metadata']['spatial_reference_sample_token']]
            if any(key.startswith('human.pedestrian') for key in objects):
                labels.append('metadata:pedestrian-within30m')
            if any(key.startswith('vehicle.') for key in objects):
                labels.append('metadata:vehicle-within30m')
            counts.update(labels)
            entries.append({'request_id': record['request_id'], 'scene_id': record['scene_id'], 'scene_name': record['scene_name'],
                            'scene_description': desc, 'labels': labels, 'endpoint_displacement_m': offset,
                            'endpoint_lateral_m': endpoint[0], 'nearby_annotation_counts': dict(objects)})
        result['partitions'][name] = {'manifest_sha256': sha(args.manifests / (name + '.jsonl')),
                                       'clips': len(records), 'scenes': len({r['scene_id'] for r in records}),
                                       'label_clip_counts': dict(counts), 'records': entries}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({name: {key: value for key, value in data.items() if key != 'records'}
                      for name, data in result['partitions'].items()}, indent=2))


if __name__ == '__main__':
    main()
