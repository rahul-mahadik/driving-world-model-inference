"""Synthetic evidence fixtures only: no actual calibration or GPU measurements."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import hashlib
import itertools
import json

import pytest

from drivebench_tuning.agent_scoring import (
    QualityPolicy, assess_proxy_quality, freeze_quality_policy, score_heldout_timing,
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


IDENTITY = {'runtime_id': 'synthetic-runtime', 'reference_id': 'fixture-reference',
            'metric_definition_sha256': digest('fixture-metrics'), 'evidence_kind': 'synthetic'}
MARGINS = {'rmse': 0.02, 'lpips': 0.05, 'temporal': 0.02}
DEV, FINAL, SEEDS, REPS = ['dev-a', 'dev-b'], ['final-a', 'final-b'], [7, 8], [0, 1]
PLAN, OBSERVED, FROZEN, SELECTED, EVALUATED, TIMED = [f'2026-01-01T0{i}:00:00+00:00' for i in range(6)]
CANDIDATE = digest('fixture-candidate')


def rows(scenes, kind):
    result = []
    for scene, seed, rep in itertools.product(scenes, SEEDS, REPS):
        row = {'record_id': f'{kind}-{scene}-{seed}-{rep}', 'scene_id': scene, 'seed': seed,
               'replicate': rep, 'identity_sha256': digest(IDENTITY),
               'observed_at': EVALUATED if kind == 'candidate' else OBSERVED,
               'conditioning_sha256': digest(['conditioning', scene]),
               'input_sha256': digest([scene, seed]), 'output_sha256': digest([kind, scene, seed, rep])}
        if kind == 'calibration':
            row.update(repeat_reference_sha256=digest([kind, scene, seed, 1 - rep]),
                       seed_reference_sha256=digest([kind, scene, 15 - seed, rep]),
                       repeat_metrics={name: 0.001 for name in MARGINS},
                       seed_metrics={name: 10.0 for name in MARGINS})
        else:
            row['metrics'] = {name: 0.5 if kind == 'control' else 0.005 for name in MARGINS}
            if kind == 'control':
                row['control_id'] = 'visible-degradation-fixture'
                row['reference_output_sha256'] = digest(['calibration', scene, seed, rep])
            else: row.update(candidate_sha256=CANDIDATE, reference_output_sha256=digest(['reference', scene, seed, rep]))
        result.append(row)
    return result


def args():
    return {'identity': deepcopy(IDENTITY), 'metric_margins': dict(MARGINS),
            'development_scenes': list(DEV), 'final_scenes': list(FINAL), 'seeds': list(SEEDS),
            'replicates': list(REPS), 'calibration_records': rows(DEV, 'calibration'),
            'negative_controls': rows(DEV, 'control'), 'plan_created_at': PLAN, 'frozen_at': FROZEN,
            'minimum_development_scenes': 2, 'minimum_final_scenes': 2,
            'timing_noise_floor': 0.025, 'timing_min_blocks': 2, 'timing_min_repeats': 2}


def policy(**overrides):
    options = args(); options.update(overrides)
    return freeze_quality_policy(**options)


def quality(p=None, records=None):
    p = p or policy()
    return assess_proxy_quality(p, rows(FINAL, 'candidate') if records is None else records,
                                split='final', selection_started_at=SELECTED)


def timings(p=None, candidate=CANDIDATE):
    p = p or policy(); result = []
    for b, (block, scenes) in enumerate(p.to_dict()['plan']['timing']['block_scenes'].items()):
        for scene, seed, repeat in itertools.product(scenes, SEEDS, REPS):
            row_id = f'{block}-{scene}-{seed}-{repeat}'
            result.append({'record_id': row_id, 'block_id': block, 'scene_id': scene, 'seed': seed,
                           'repeat': repeat, 'identity_sha256': digest(IDENTITY), 'candidate_sha256': candidate,
                           'input_sha256': digest([scene, seed]), 'observed_at': TIMED,
                           'reference_measurement_id': 'r-' + row_id,
                           'candidate_measurement_id': 'c-' + candidate + row_id,
                           'reference_latency_ms': 100.0, 'candidate_latency_ms': 80.0 + 5 * b,
                           'order': 'reference-first' if repeat == 0 else 'candidate-first'})
    return result


def test_policy_is_immutable_detached_json_and_exactly_reconstructs():
    options = args(); p = freeze_quality_policy(**options)
    options['metric_margins']['lpips'] = 99
    detached = p.to_dict(); detached['plan']['metric_margins']['lpips'] = 50
    assert p.to_dict()['plan']['metric_margins'] == MARGINS
    with pytest.raises(FrozenInstanceError): p._canonical_json = '{}'
    restored = QualityPolicy.from_dict(json.loads(json.dumps(p.to_dict())))
    assert restored == p and restored.sha256 == p.sha256
    assert p.to_dict()['calibration']['real_calibration_established'] is False


def test_huge_reference_seed_spread_never_relaxes_margins_or_claims_equivalence():
    p = policy(); assert p.to_dict()['calibration']['reference_seed_spread']['lpips']['maximum'] == 10
    good = quality(p); assert good['status'] == 'pass' and not good['quality_preserving_claim']
    bad_rows = rows(FINAL, 'candidate'); bad_rows[0]['metrics']['lpips'] = 0.051
    bad = quality(p, bad_rows)
    assert bad['status'] == 'fail' and bad['breaches'][0]['maximum'] == 0.05
    assert not bad['quality_preserving_claim']


@pytest.mark.parametrize('change', ['missing_reference', 'missing_control', 'undetected_control', 'repeat_noise', 'sample_floor'])
def test_insufficient_calibration_is_inconclusive(change):
    options = args()
    if change == 'missing_reference': options['calibration_records'].pop()
    elif change == 'missing_control': options['negative_controls'] = []
    elif change == 'undetected_control': options['negative_controls'][0]['metrics'] = {m: 0 for m in MARGINS}
    elif change == 'repeat_noise': options['calibration_records'][0]['repeat_metrics']['rmse'] = 0.03
    else: options['minimum_development_scenes'] = 32
    p = freeze_quality_policy(**options)
    assert p.to_dict()['calibration']['status'] == 'inconclusive'
    assert quality(p)['status'] == 'inconclusive'
    result = score_heldout_timing(p, timings(p), quality_assessment=quality(p), selection_started_at=SELECTED)
    assert result['status'] == 'inconclusive' and result['score'] is None


def test_each_control_requires_complete_development_matrix():
    options = args(); second = deepcopy(options['negative_controls'])
    for row in second: row['record_id'] += '-second'; row['control_id'] = 'second-control'
    second.pop(); options['negative_controls'] += second
    p = freeze_quality_policy(**options)
    assert p.to_dict()['calibration']['status'] == 'inconclusive'
    assert not p.to_dict()['calibration']['negative_controls']['second-control']['complete_matrix']


@pytest.mark.parametrize('field,row_field', [('seeds', 'seed'), ('replicates', 'replicate')])
def test_single_seed_or_replicate_is_inconclusive_not_an_anchor_exception(field, row_field):
    options = args(); selected = options[field][0]; options[field] = [selected]
    for kind in ('calibration_records', 'negative_controls'):
        options[kind] = [row for row in options[kind] if row[row_field] == selected]
    p = freeze_quality_policy(**options)
    assert p.to_dict()['calibration']['status'] == 'inconclusive'


@pytest.mark.parametrize('field,value', [('identity_sha256', digest('drift')), ('observed_at', PLAN),
                                       ('observed_at', FROZEN), ('observed_at', '2026-01-01T01:00:00'),
                                       ('output_sha256', 'not-a-hash'), ('seed', True)])
def test_calibration_rejects_invalid_identity_timing_or_hash(field, value):
    options = args(); options['calibration_records'][0][field] = value
    with pytest.raises(ValueError): freeze_quality_policy(**options)


def test_calibration_rejects_duplicate_or_nonfinite_evidence_and_overlap():
    options = args(); options['calibration_records'].append(deepcopy(options['calibration_records'][0]))
    with pytest.raises(ValueError, match='distinct|duplicate'): freeze_quality_policy(**options)
    options = args(); options['calibration_records'][0]['repeat_metrics']['rmse'] = float('nan')
    with pytest.raises(ValueError): freeze_quality_policy(**options)
    options = args(); options['final_scenes'] = list(DEV)
    with pytest.raises(ValueError, match='overlap'): freeze_quality_policy(**options)


@pytest.mark.parametrize('field', ['repeat_reference_sha256', 'seed_reference_sha256', 'conditioning_sha256'])
def test_reference_variability_requires_real_matrix_anchors_and_fixed_conditioning(field):
    options = args(); options['calibration_records'][0][field] = digest('unrelated-output')
    with pytest.raises(ValueError): freeze_quality_policy(**options)


def test_control_must_derive_from_declared_calibration_output():
    options = args(); options['negative_controls'][0]['reference_output_sha256'] = digest('other-reference')
    with pytest.raises(ValueError, match='corresponding'): freeze_quality_policy(**options)


@pytest.mark.parametrize('modification', ['digest', 'recomputed_false_status'])
def test_policy_roundtrip_detects_tampered_or_fabricated_derived_calibration(modification):
    options = args(); options['negative_controls'] = []
    value = freeze_quality_policy(**options).to_dict()
    if modification == 'digest': value['plan']['metric_margins']['lpips'] = 999
    else:
        value['calibration']['status'] = 'ready'; value['calibration']['reasons'] = []
        value['policy_digest'] = digest({k: v for k, v in value.items() if k != 'policy_digest'})
    with pytest.raises(ValueError): QualityPolicy.from_dict(value)


def test_candidate_full_matrix_chronology_and_input_identity_are_required():
    p = policy(); incomplete = rows(FINAL, 'candidate')[:-1]
    assert quality(p, incomplete)['status'] == 'inconclusive'
    with pytest.raises(ValueError, match='frozen'):
        assess_proxy_quality(p, rows(FINAL, 'candidate'), split='final', selection_started_at=FROZEN)
    wrong = rows(FINAL, 'candidate'); wrong[0]['observed_at'] = SELECTED
    with pytest.raises(ValueError, match='chronology'): quality(p, wrong)
    wrong = rows(FINAL, 'candidate'); wrong[0]['input_sha256'] = digest('different')
    with pytest.raises(ValueError, match='input'): quality(p, wrong)
    wrong = rows(FINAL, 'candidate'); wrong[0]['candidate_sha256'] = digest('different')
    with pytest.raises(ValueError, match='one selected'): quality(p, wrong)
    with pytest.raises(ValueError): assess_proxy_quality(p, rows(DEV, 'candidate'), split='final', selection_started_at=SELECTED)


def test_partitioned_timing_scores_worst_block_plus_noise_without_confidence_claim():
    p = policy(); r = score_heldout_timing(p, timings(p), quality_assessment=quality(p), selection_started_at=SELECTED)
    assert r['status'] == 'pass' and r['score'] == pytest.approx(0.125)
    assert r['worst_block_median_ratio'] == 0.85
    assert r['paired_samples'] == len(FINAL) * len(SEEDS) * len(REPS) == 8
    assert r['measurement_calls'] == 16
    assert not r['confidence_claim'] and not r['quality_preserving_claim']
    assert r['evidence_kind'] == 'synthetic'


@pytest.mark.parametrize('change', ['duplicate_row', 'duplicate_measurement', 'runtime', 'input', 'candidate', 'zero', 'nan', 'chronology', 'partition', 'extra_repeat'])
def test_timing_rejects_bad_freshness_runtime_inputs_measurements_or_partition(change):
    p = policy(); values = timings(p)
    if change == 'duplicate_row': values.append(deepcopy(values[0]))
    elif change == 'duplicate_measurement': values[1]['reference_measurement_id'] = values[0]['reference_measurement_id']
    elif change == 'runtime': values[0]['identity_sha256'] = digest('other-runtime')
    elif change == 'input': values[0]['input_sha256'] = digest('other-input')
    elif change == 'candidate': values[0]['candidate_sha256'] = digest('other-candidate')
    elif change == 'zero': values[0]['reference_latency_ms'] = 0
    elif change == 'nan': values[0]['candidate_latency_ms'] = float('nan')
    elif change == 'chronology': values[0]['observed_at'] = SELECTED
    elif change == 'partition': values[0]['block_id'] = 'block-1'
    else: values[0]['repeat'] = 2
    with pytest.raises(ValueError): score_heldout_timing(p, values, quality_assessment=quality(p), selection_started_at=SELECTED)


@pytest.mark.parametrize('change', ['missing_cell', 'missing_block', 'unbalanced'])
def test_incomplete_or_unbalanced_timing_is_inconclusive(change):
    p = policy(); values = timings(p)
    if change == 'missing_cell': values.pop()
    elif change == 'missing_block': values = [r for r in values if r['block_id'] == 'block-0']
    else:
        for row in values: row['order'] = 'reference-first'
    report = score_heldout_timing(p, values, quality_assessment=quality(p), selection_started_at=SELECTED)
    assert report['status'] == 'inconclusive' and report['score'] is None


def test_quality_failure_and_tampered_quality_report_cannot_earn_timing_reward():
    p = policy(); candidate_rows = rows(FINAL, 'candidate'); candidate_rows[0]['metrics']['lpips'] = 1
    failed = quality(p, candidate_rows)
    report = score_heldout_timing(p, timings(p), quality_assessment=failed, selection_started_at=SELECTED)
    assert report['status'] == 'fail' and report['score'] is None
    failed['status'] = 'pass'
    with pytest.raises(ValueError, match='changed'):
        score_heldout_timing(p, timings(p), quality_assessment=failed, selection_started_at=SELECTED)
    dev = assess_proxy_quality(p, rows(DEV, 'candidate'), split='development', selection_started_at=SELECTED)
    with pytest.raises(ValueError): score_heldout_timing(p, timings(p), quality_assessment=dev, selection_started_at=SELECTED)


def test_shared_reference_across_candidates_is_allowed_and_declared_dependent():
    p = policy(); first = score_heldout_timing(p, timings(p), quality_assessment=quality(p), selection_started_at=SELECTED)
    selected = digest('second-candidate'); second_rows = rows(FINAL, 'candidate')
    for row in second_rows: row['candidate_sha256'] = selected
    second = score_heldout_timing(p, timings(p, selected), quality_assessment=quality(p, second_rows), selection_started_at=SELECTED)
    assert first['status'] == second['status'] == 'pass'
    assert second['shared_reference_across_candidate_assessments_permitted']
    assert second['shared_reference_candidate_comparisons_independent'] is False


def test_noise_floor_prevents_a_small_apparent_improvement_from_scoring():
    p = policy(); values = timings(p)
    for row in values: row['candidate_latency_ms'] = 99
    result = score_heldout_timing(p, values, quality_assessment=quality(p), selection_started_at=SELECTED)
    assert result['status'] == 'pass' and result['score'] == 0


def test_invalid_frozen_block_partitions_rejected():
    with pytest.raises(ValueError, match='exactly once'):
        policy(timing_block_scenes={'a': list(FINAL), 'b': [FINAL[0]]})
    with pytest.raises(ValueError, match='exactly once'):
        policy(timing_block_scenes={'a': [FINAL[0]]})
