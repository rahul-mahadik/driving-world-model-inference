import copy

import pytest

from scripts.analyze_recorded_ground_truth import METRICS, analyze


def fixture():
    gt = {'pair_count': 2, 'diagnostic_only': True, 'lpips_computed': False,
          'run_artifact_sha256': {'config.json': 'fixture'}, 'manifest_sha256': 'manifest', 'pairs': []}
    rows = []
    for i in range(2):
        identity = dict(candidate_id=f'c{i}', scene_id=f's{i}', request_id=f'r{i}', repeat=0,
                        seed=i, action_kind='recorded', trajectory_sha256=f'a{i}')
        metrics = {k: .1 for k in METRICS}
        metrics.update(reference_temporal={'fixture': 'recorded'}, candidate_temporal={'fixture': 'upstream'})
        gt['pairs'].append({**identity, 'output_sha256': f'output{i}', 'metrics': metrics})
        changed = {**metrics, 'rmse': .1 + (-.01 if i else .02)}
        rows.append({**identity, 'reference_output_sha256': f'output{i}', 'candidate_output_sha256': f'changed{i}',
                     'recorded_ground_truth': changed, 'metrics': {'reference_temporal': metrics['candidate_temporal']}})
    comparison = {'reference_artifact_sha256': gt['run_artifact_sha256'], 'runs': [{
        'synthetic': False, 'complete_pairs': True, 'errors': [], 'pairs': rows[::-1],
        'identity': {'manifest_sha256': gt['manifest_sha256']}}]}
    return gt, {name: copy.deepcopy(comparison) for name in ('compiled-default', 'compiled-conditioning')}


def test_pairs_are_joined_by_identity_not_array_order():
    gt, comparisons = fixture(); summary, details = analyze(gt, comparisons, expected_count=2)
    for value in summary.values():
        assert value['rmse']['paired_change']['mean'] == pytest.approx(.005)
        assert value['rmse']['paired_change']['positive_clips'] == 1
        assert value['rmse']['paired_change']['negative_clips'] == 1
        assert value['freeze_case_counts']['same_fraction_per_clip']
    assert details['compiled-default'][0]['candidate_id'] == 'c1'


@pytest.mark.parametrize('mutation', ['seed', 'hash', 'temporal', 'source', 'duplicate', 'nan', 'manifest'])
def test_mismatched_inputs_fail_closed(mutation):
    gt, comparisons = fixture(); doc = comparisons['compiled-default']; run = doc['runs'][0]; pair = run['pairs'][0]
    if mutation == 'seed': pair['seed'] = 9
    if mutation == 'hash': pair['reference_output_sha256'] = 'changed'
    if mutation == 'temporal': pair['recorded_ground_truth']['reference_temporal'] = {'changed': True}
    if mutation == 'source': doc['reference_artifact_sha256'] = {}
    if mutation == 'duplicate': run['pairs'][1] = copy.deepcopy(pair)
    if mutation == 'nan': pair['recorded_ground_truth']['rmse'] = float('nan')
    if mutation == 'manifest': run['identity']['manifest_sha256'] = 'changed'
    with pytest.raises(ValueError): analyze(gt, comparisons, expected_count=2)
