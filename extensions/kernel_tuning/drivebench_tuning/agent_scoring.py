"""Frozen proxy-quality policy and conservative heldout timing bookkeeping.

This module reads supplied numeric evidence; it runs no models or GPU code and
does not authenticate a caller's hardware, timestamps, hashes or quality labels.
Seed spread describes the reference and NEVER selects or widens error margins.
Passing finite proxy limits is not perceptual equivalence or driving safety.

Common quality row fields: record_id, scene_id, seed, replicate,
identity_sha256, observed_at (timezone-aware ISO), conditioning_sha256,
input_sha256, output_sha256. Conditioning binds image/action inputs excluding
the seed and must stay constant within a scene. Calibration
rows additionally contain repeat_metrics and seed_metrics: nonnegative metric
distances to explicitly hashed repeat_reference_sha256 / seed_reference_sha256
outputs. Negative-control rows contain control_id, reference_output_sha256 and
metrics; they bind the corresponding unmodified calibration output. Candidate rows
contain candidate_sha256, reference_output_sha256 and metrics. Metric mappings
must exactly cover the policy's explicitly supplied metric_margins. Every
declared scene x seed x replicate cell is required; control groups each require
that same complete development matrix. Distances and hashes are supplied by a
separate, version-bound metric executor, not recomputed from arrays here.

Timing rows contain record_id, block_id, repeat, scene_id, seed,
identity_sha256, candidate_sha256, input_sha256, observed_at,
reference_measurement_id, candidate_measurement_id, reference_latency_ms,
candidate_latency_ms, and order ('reference-first' or 'candidate-first'). Each
block covers its frozen scene partition x seed x repeat; partitions cover each
final scene exactly once. Both orders are required for every scene/seed, with
counts differing by at most one. Separate measurement IDs and
fresh rows are mandatory. Worst observed block-median ratio plus a prespecified
noise floor gives a conservative descriptive score, NOT a confidence bound.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import itertools
import json
import math
from typing import Mapping

import numpy as np


def _json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError('evidence must be finite JSON') from exc


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _hash(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(f'{label} must be a lowercase SHA256')
    return value


def _name(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{label} must be a nonempty string')
    return value


def _number(value, label, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f'{label} must be finite and {"positive" if positive else "nonnegative"}')
    return float(value)


def _integer(value, label, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{label} must be an integer >= {minimum}')
    return value


def _time(value):
    if not isinstance(value, str):
        raise ValueError('timestamps must be timezone-aware ISO strings')
    try:
        timestamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as exc:
        raise ValueError('timestamps must be timezone-aware ISO strings') from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError('timestamps must include a timezone')
    return timestamp


def _unique(values, label, validate):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f'{label} must be a nonempty list')
    result = [validate(value, label) for value in values]
    if len(set(result)) != len(result):
        raise ValueError(f'{label} contains duplicates')
    return sorted(result)


def _metrics(value, margins, label):
    if not isinstance(value, dict) or set(value) != set(margins):
        raise ValueError(f'{label} must exactly cover the declared metrics')
    return {key: _number(value[key], f'{label}.{key}') for key in margins}


def _rows(records):
    if not isinstance(records, (list, tuple)):
        raise ValueError('records must be a list')
    result = json.loads(_json(records))
    if any(not isinstance(row, dict) for row in result):
        raise ValueError('each evidence record must be an object')
    return result


def _quality_rows(records, plan, scenes, *, kind, after, before=None):
    rows = _rows(records)
    ids, cells, inputs, conditioning = set(), set(), {}, {}
    expected = set(itertools.product(scenes, plan['seeds'], plan['replicates']))
    for row in rows:
        rid = _name(row.get('record_id'), 'record_id')
        if rid in ids:
            raise ValueError('duplicate record_id')
        ids.add(rid)
        cell = (_name(row.get('scene_id'), 'scene_id'),
                _integer(row.get('seed'), 'seed'), _integer(row.get('replicate'), 'replicate'))
        if cell in cells or cell not in expected:
            raise ValueError('duplicate or undeclared scene/seed/replicate cell')
        cells.add(cell)
        if row.get('identity_sha256') != plan['identity_sha256']:
            raise ValueError('runtime/reference/metric identity drift')
        observed = _time(row.get('observed_at'))
        if observed <= after or (before is not None and observed >= before):
            raise ValueError('evidence violates the frozen chronology')
        input_hash = _hash(row.get('input_sha256'), 'input_sha256')
        input_key = cell[:2]
        if input_key in inputs and inputs[input_key] != input_hash:
            raise ValueError('input identity changed across quality replicates')
        inputs[input_key] = input_hash
        condition_hash = _hash(row.get('conditioning_sha256'), 'conditioning_sha256')
        if cell[0] in conditioning and conditioning[cell[0]] != condition_hash:
            raise ValueError('conditioning/action inputs changed within a scene')
        conditioning[cell[0]] = condition_hash
        _hash(row.get('output_sha256'), 'output_sha256')
        if kind == 'calibration':
            _hash(row.get('repeat_reference_sha256'), 'repeat_reference_sha256')
            _hash(row.get('seed_reference_sha256'), 'seed_reference_sha256')
            _metrics(row.get('repeat_metrics'), plan['metric_margins'], 'repeat_metrics')
            _metrics(row.get('seed_metrics'), plan['metric_margins'], 'seed_metrics')
        else:
            _metrics(row.get('metrics'), plan['metric_margins'], 'metrics')
            _hash(row.get('reference_output_sha256'), 'reference_output_sha256')
        if kind == 'candidate':
            _hash(row.get('candidate_sha256'), 'candidate_sha256')
    return rows, expected - cells


def _distribution(rows, field, margins):
    result = {}
    for metric in margins:
        by_scene = {}
        for row in rows:
            by_scene.setdefault(row['scene_id'], []).append(row[field][metric])
        means = [float(np.mean(values)) for values in by_scene.values()]
        values = [row[field][metric] for row in rows]
        result[metric] = {'scene_mean': float(np.mean(means)) if means else None,
                          'maximum': max(values) if values else None,
                          'scene_count': len(means), 'record_count': len(values),
                          'independent_unit': 'scene; seeds/replicates remain grouped',
                          'confidence_claim': False}
    return result


@dataclass(frozen=True)
class QualityPolicy:
    """Immutable canonical JSON; to_dict returns a detached mutable copy.

    Construct with freeze_quality_policy or from_dict. Hash integrity is not an
    authenticity signature; consumers must retain the originally frozen hash.
    """
    _canonical_json: str

    @property
    def sha256(self):
        return hashlib.sha256(self._canonical_json.encode()).hexdigest()

    @property
    def identity_sha256(self):
        return self.to_dict()['plan']['identity_sha256']

    def to_dict(self):
        return json.loads(self._canonical_json)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {'schema_version', 'plan', 'calibration', 'policy_digest'}:
            raise ValueError('invalid quality-policy schema')
        unsigned = {key: value[key] for key in value if key != 'policy_digest'}
        if value.get('schema_version') != 1 or value.get('policy_digest') != _digest(unsigned):
            raise ValueError('quality policy digest mismatch')
        plan = value['plan']
        required = {'identity', 'identity_sha256', 'metric_margins', 'aggregation', 'development_scenes',
                    'final_scenes', 'seeds', 'replicates', 'plan_created_at', 'frozen_at', 'requirements',
                    'timing', 'quality_preserving_claim', 'threshold_selection'}
        if not isinstance(plan, dict) or set(plan) != required or plan['identity_sha256'] != _digest(plan['identity']):
            raise ValueError('invalid policy plan identity')
        if plan['quality_preserving_claim'] is not False or plan['aggregation'] != 'maximum_per_output':
            raise ValueError('policy cannot establish visual quality preservation')
        calibration = value['calibration']
        if not isinstance(calibration, dict) or calibration.get('status') not in ('ready', 'inconclusive'):
            raise ValueError('invalid calibration status')
        requirements = plan['requirements']; timing = plan['timing']
        try:
            replay = freeze_quality_policy(
                identity=plan['identity'], metric_margins=plan['metric_margins'],
                development_scenes=plan['development_scenes'], final_scenes=plan['final_scenes'],
                seeds=plan['seeds'], replicates=plan['replicates'],
                calibration_records=calibration['reference_records'], negative_controls=calibration['control_records'],
                plan_created_at=plan['plan_created_at'], frozen_at=plan['frozen_at'],
                **requirements, timing_noise_floor=timing['noise_floor'],
                timing_min_blocks=timing['minimum_blocks'], timing_min_repeats=timing['minimum_repeats'],
                timing_block_scenes=timing['block_scenes'])
        except (KeyError, TypeError) as exc:
            raise ValueError('invalid quality-policy contents') from exc
        if replay._canonical_json != _json(value):
            raise ValueError('quality policy calibration does not reconstruct exactly')
        return replay


def freeze_quality_policy(*, identity, metric_margins, development_scenes, final_scenes,
                          seeds, replicates, calibration_records, negative_controls,
                          plan_created_at, frozen_at, minimum_development_scenes=32,
                          minimum_final_scenes=16, minimum_seeds=2, minimum_replicates=2,
                          negative_control_min_detection_rate=1.0, timing_noise_floor=0.0,
                          timing_min_blocks=4, timing_min_repeats=2, timing_block_scenes=None):
    """Freeze explicit maxima; incomplete/undetected calibration is inconclusive.

    identity requires runtime_id, reference_id, metric_definition_sha256 and
    evidence_kind ('measured' or 'synthetic'); include model/config/asset hashes.
    Times must satisfy plan < calibration/control evidence < freeze < selection.
    Requirements are declared sample floors, not a statistical power guarantee.
    """
    identity = json.loads(_json(identity))
    if not isinstance(identity, dict):
        raise ValueError('identity must be an object')
    _name(identity.get('runtime_id'), 'runtime_id'); _name(identity.get('reference_id'), 'reference_id')
    _hash(identity.get('metric_definition_sha256'), 'metric_definition_sha256')
    if identity.get('evidence_kind') not in ('measured', 'synthetic'):
        raise ValueError('identity must declare measured or synthetic evidence_kind')
    if not isinstance(metric_margins, Mapping) or not metric_margins:
        raise ValueError('explicit metric maximums are required')
    margins = {_name(name, 'metric name'): _number(value, 'metric maximum') for name, value in metric_margins.items()}
    dev = _unique(development_scenes, 'development_scenes', _name)
    final = _unique(final_scenes, 'final_scenes', _name)
    if set(dev) & set(final):
        raise ValueError('development and final scenes overlap')
    seeds = _unique(seeds, 'seeds', _integer); reps = _unique(replicates, 'replicates', _integer)
    requirements = {'minimum_development_scenes': _integer(minimum_development_scenes, 'minimum_development_scenes', minimum=2),
                    'minimum_final_scenes': _integer(minimum_final_scenes, 'minimum_final_scenes', minimum=2),
                    'minimum_seeds': _integer(minimum_seeds, 'minimum_seeds', minimum=2),
                    'minimum_replicates': _integer(minimum_replicates, 'minimum_replicates', minimum=2),
                    'negative_control_min_detection_rate': _number(negative_control_min_detection_rate, 'negative_control_min_detection_rate', positive=True)}
    if requirements['negative_control_min_detection_rate'] > 1:
        raise ValueError('negative-control detection rate must be <= 1')
    planned, frozen = _time(plan_created_at), _time(frozen_at)
    if frozen <= planned:
        raise ValueError('freeze must follow the prespecified plan')
    minimum_blocks = _integer(timing_min_blocks, 'timing_min_blocks', minimum=2)
    if timing_block_scenes is None:
        count = min(minimum_blocks, len(final))
        partitions = {f'block-{index}': final[index::count] for index in range(count)}
    else:
        if not isinstance(timing_block_scenes, Mapping) or not timing_block_scenes:
            raise ValueError('timing_block_scenes must be a nonempty mapping')
        partitions = {_name(block, 'block_id'): _unique(scenes, 'block scenes', _name)
                      for block, scenes in timing_block_scenes.items()}
    assigned = [scene for scenes in partitions.values() for scene in scenes]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(final):
        raise ValueError('timing partitions must cover each final scene exactly once')
    plan = {'identity': identity, 'identity_sha256': _digest(identity), 'metric_margins': margins,
            'aggregation': 'maximum_per_output', 'development_scenes': dev, 'final_scenes': final,
            'seeds': seeds, 'replicates': reps, 'plan_created_at': plan_created_at, 'frozen_at': frozen_at,
            'requirements': requirements, 'timing': {'noise_floor': _number(timing_noise_floor, 'timing_noise_floor'),
            'minimum_blocks': minimum_blocks, 'block_scenes': partitions,
            'minimum_repeats': _integer(timing_min_repeats, 'timing_min_repeats', minimum=2),
            'score_rule': 'max(0, 1 - max(block_median_candidate_reference_ratio) - noise_floor)',
            'confidence_claim': False}, 'quality_preserving_claim': False,
            'threshold_selection': 'explicit prespecified maxima; reference seed spread never widens margins'}
    rows, missing = _quality_rows(calibration_records, plan, dev, kind='calibration', after=planned, before=frozen)
    controls = _rows(negative_controls)
    if len({row.get('record_id') for row in rows + controls}) != len(rows + controls):
        raise ValueError('calibration and controls must have distinct record IDs')
    grouped = {}
    for row in controls:
        grouped.setdefault(_name(row.get('control_id'), 'control_id'), []).append(row)
    reasons = []
    for name, actual, minimum in [('development scenes', len(dev), minimum_development_scenes),
                                  ('final scenes', len(final), minimum_final_scenes),
                                  ('seeds', len(seeds), minimum_seeds), ('replicates', len(reps), minimum_replicates)]:
        if actual < minimum:
            reasons.append(f'insufficient declared {name}: {actual} < {minimum}')
    if missing: reasons.append('incomplete reference calibration matrix')
    anchors_missing = False
    for row in rows:
        repeated = {other['output_sha256'] for other in rows
                    if (other['scene_id'], other['seed']) == (row['scene_id'], row['seed'])
                    and other['replicate'] != row['replicate']}
        other_seeds = {other['output_sha256'] for other in rows
                       if other['scene_id'] == row['scene_id'] and other['seed'] != row['seed']}
        if row['repeat_reference_sha256'] not in repeated or row['seed_reference_sha256'] not in other_seeds:
            anchors_missing = True
    if anchors_missing:
        if not missing and len(seeds) >= 2 and len(reps) >= 2:
            raise ValueError('repeat/seed metric anchors must reference actual matching calibration outputs')
        reasons.append('calibration anchor outputs are missing')
    if len(partitions) < minimum_blocks: reasons.append('insufficient declared timing blocks')
    if not controls: reasons.append('negative controls are required')
    detections = {}
    reference_cells = {(row['scene_id'], row['seed'], row['replicate']): row for row in rows}
    for control_id, group in grouped.items():
        checked, absent = _quality_rows(group, plan, dev, kind='control', after=planned, before=frozen)
        for row in checked:
            original = reference_cells.get((row['scene_id'], row['seed'], row['replicate']))
            if original is not None and (row['reference_output_sha256'] != original['output_sha256']
                                        or row['input_sha256'] != original['input_sha256']
                                        or row['conditioning_sha256'] != original['conditioning_sha256']):
                raise ValueError('negative control must bind its corresponding calibration output and inputs')
        detected = sum(any(row['metrics'][name] > maximum for name, maximum in margins.items()) for row in checked)
        rate = detected / len(checked)
        detections[control_id] = {'records': len(checked), 'detected': detected, 'detection_rate': rate,
                                  'complete_matrix': not absent}
        if absent: reasons.append(f'incomplete negative-control matrix: {control_id}')
        if rate < negative_control_min_detection_rate: reasons.append(f'negative control not reliably detected: {control_id}')
    repeat_noise = _distribution(rows, 'repeat_metrics', margins)
    if any(row['repeat_metrics'][name] > maximum for row in rows for name, maximum in margins.items()):
        reasons.append('reference repeat noise exceeds a prespecified proxy maximum')
    calibration = {'status': 'inconclusive' if reasons else 'ready', 'reasons': reasons,
                   'reference_repeat_noise': repeat_noise, 'reference_seed_spread': _distribution(rows, 'seed_metrics', margins),
                   'negative_controls': detections, 'record_count': len(rows), 'control_record_count': len(controls),
                   'evidence_sha256': _digest({'reference': rows, 'negative_controls': controls}),
                   'reference_records': rows, 'control_records': controls,
                   'real_calibration_established': not reasons and identity['evidence_kind'] == 'measured',
                   'limitation': 'Caller-supplied metric provenance; seed spread and proxy bounds do not establish perceptual equivalence.'}
    unsigned = {'schema_version': 1, 'plan': plan, 'calibration': calibration}
    return QualityPolicy(_json({**unsigned, 'policy_digest': _digest(unsigned)}))


def _policy(policy):
    if not isinstance(policy, QualityPolicy):
        raise ValueError('use a frozen QualityPolicy')
    return QualityPolicy.from_dict(policy.to_dict()).to_dict()


def assess_proxy_quality(policy, records, *, split, selection_started_at):
    """Assess every declared output. A proxy pass always leaves quality claim false.

    Invalid evidence raises ValueError; structurally valid but incomplete or
    insufficient evidence is inconclusive. A measured threshold breach fails.
    """
    value = _policy(policy); plan = value['plan']
    if split not in ('development', 'final'):
        raise ValueError('split must be development or final')
    selection = _time(selection_started_at)
    if selection <= _time(plan['frozen_at']):
        raise ValueError('policy must be frozen before candidate selection starts')
    scenes = plan['development_scenes'] if split == 'development' else plan['final_scenes']
    rows, missing = _quality_rows(records, plan, scenes, kind='candidate', after=selection)
    candidates = {row['candidate_sha256'] for row in rows}
    if len(candidates) > 1:
        raise ValueError('quality assessment must bind one selected candidate')
    reasons = list(value['calibration']['reasons'])
    if missing: reasons.append('incomplete candidate output matrix')
    breaches = [{'record_id': row['record_id'], 'metric': name, 'actual': row['metrics'][name], 'maximum': maximum}
                for row in rows for name, maximum in plan['metric_margins'].items() if row['metrics'][name] > maximum]
    status = 'fail' if breaches else ('inconclusive' if reasons else 'pass')
    return {'schema_version': 1, 'status': status, 'reasons': reasons, 'breaches': breaches,
            'split': split, 'selection_started_at': selection_started_at, 'policy_sha256': policy.sha256,
            'identity_sha256': plan['identity_sha256'], 'candidate_sha256': next(iter(candidates)) if candidates else None,
            'record_count': len(rows), 'records': rows, 'records_sha256': _digest(rows),
            'metrics': _distribution(rows, 'metrics', plan['metric_margins']),
            'evidence_kind': plan['identity']['evidence_kind'], 'quality_preserving_claim': False,
            'claim_scope': 'Complete supplied outputs within prespecified proxy maxima only; no visual equivalence or safety claim.'}


def score_heldout_timing(policy, paired_rows, *, quality_assessment, selection_started_at):
    """Score fresh heldout pairs only after a replay-verified final proxy gate.

    No probability coverage is claimed for the worst-block statistic. The noise
    floor is a separately prespecified dimensionless margin, not inferred from
    these candidate measurements. All measurement calls remain chargeable.
    """
    value = _policy(policy); plan = value['plan']; selection = _time(selection_started_at)
    if not isinstance(quality_assessment, dict):
        raise ValueError('quality_assessment must be an assessment report')
    expected_quality = assess_proxy_quality(policy, quality_assessment.get('records'), split='final',
                                            selection_started_at=selection_started_at)
    if _json(expected_quality) != _json(quality_assessment):
        raise ValueError('final quality assessment changed, has wrong split, or uses another policy/selection')
    rows = _rows(paired_rows)
    quality_inputs = {(row['scene_id'], row['seed']): row['input_sha256'] for row in expected_quality['records']}
    ids, measurements, cells, blocks, inputs = set(), set(), set(), {}, {}
    for row in rows:
        rid = _name(row.get('record_id'), 'record_id')
        if rid in ids: raise ValueError('duplicate timing record_id')
        ids.add(rid)
        block = _name(row.get('block_id'), 'block_id'); scene = _name(row.get('scene_id'), 'scene_id')
        seed = _integer(row.get('seed'), 'seed'); repeat = _integer(row.get('repeat'), 'repeat')
        if scene not in plan['final_scenes'] or seed not in plan['seeds']:
            raise ValueError('timing must use declared final scenes and seeds')
        if block not in plan['timing']['block_scenes'] or scene not in plan['timing']['block_scenes'][block]:
            raise ValueError('timing row violates frozen block-scene partition')
        if repeat >= plan['timing']['minimum_repeats']:
            raise ValueError('timing repeat exceeds the frozen repeat budget')
        cell = (block, scene, seed, repeat)
        if cell in cells: raise ValueError('duplicate timing block/scene/seed/repeat cell')
        cells.add(cell)
        if row.get('identity_sha256') != plan['identity_sha256']:
            raise ValueError('timing runtime/reference/metric identity drift')
        if row.get('candidate_sha256') != expected_quality['candidate_sha256']:
            raise ValueError('timing candidate differs from final quality candidate')
        _hash(row.get('candidate_sha256'), 'candidate_sha256'); inp = _hash(row.get('input_sha256'), 'input_sha256')
        key = (scene, seed)
        if quality_inputs.get(key) != inp: raise ValueError('timing input differs from the quality-scored input')
        if key in inputs and inputs[key] != inp: raise ValueError('timing input identity changed across repeats')
        inputs[key] = inp
        if _time(row.get('observed_at')) <= selection:
            raise ValueError('confirmation measurements must follow selection')
        for field in ('reference_measurement_id', 'candidate_measurement_id'):
            mid = _name(row.get(field), field)
            if mid in measurements: raise ValueError('measurement ID reused; confirmation must be fresh')
            measurements.add(mid)
        ref = _number(row.get('reference_latency_ms'), 'reference_latency_ms', positive=True)
        candidate = _number(row.get('candidate_latency_ms'), 'candidate_latency_ms', positive=True)
        if row.get('order') not in ('reference-first', 'candidate-first'):
            raise ValueError('declare paired execution order')
        ratio = candidate / ref
        if not math.isfinite(ratio): raise ValueError('nonfinite timing ratio')
        blocks.setdefault(block, []).append((row, ratio))
    reasons = []
    if expected_quality['status'] != 'pass': reasons.append('final proxy quality did not pass')
    if set(blocks) != set(plan['timing']['block_scenes']): reasons.append('incomplete declared timing blocks')
    summaries = {}
    for block, values in sorted(blocks.items()):
        repeats = {row['repeat'] for row, _ in values}
        if len(repeats) < plan['timing']['minimum_repeats'] or repeats != set(range(len(repeats))):
            reasons.append(f'insufficient or noncontiguous timing repeats: {block}')
        block_scenes = plan['timing']['block_scenes'][block]
        expected = set(itertools.product(block_scenes, plan['seeds'], range(plan['timing']['minimum_repeats'])))
        actual = {(row['scene_id'], row['seed'], row['repeat']) for row, _ in values}
        if actual != expected: reasons.append(f'incomplete final timing matrix: {block}')
        for scene, seed in itertools.product(block_scenes, plan['seeds']):
            orders = [row['order'] for row, _ in values if (row['scene_id'], row['seed']) == (scene, seed)]
            nref = orders.count('reference-first'); ncandidate = orders.count('candidate-first')
            if not min(nref, ncandidate) or abs(nref - ncandidate) > 1:
                reasons.append(f'unbalanced timing order: {block}/{scene}/{seed}')
        ratios = [ratio for _, ratio in values]
        summaries[block] = {'paired_samples': len(values), 'median_ratio': float(np.median(ratios)),
                            'minimum_ratio': min(ratios), 'maximum_ratio': max(ratios)}
    bound = max((x['median_ratio'] for x in summaries.values()), default=None)
    conservative = None if bound is None else bound + plan['timing']['noise_floor']
    score = max(0.0, 1.0 - conservative) if not reasons else None
    status = 'fail' if expected_quality['status'] == 'fail' else ('inconclusive' if reasons else 'pass')
    return {'schema_version': 1, 'status': status, 'reasons': reasons, 'score': score,
            'policy_sha256': policy.sha256, 'quality_assessment_sha256': _digest(quality_assessment),
            'candidate_sha256': expected_quality['candidate_sha256'], 'identity_sha256': plan['identity_sha256'],
            'records_sha256': _digest(rows), 'paired_samples': len(rows), 'measurement_calls': len(measurements),
            'blocks': summaries, 'worst_block_median_ratio': bound, 'declared_noise_floor': plan['timing']['noise_floor'],
            'conservative_ratio': conservative, 'confidence_claim': False,
            'shared_reference_across_candidate_assessments_permitted': True,
            'shared_reference_candidate_comparisons_independent': False,
            'evidence_kind': plan['identity']['evidence_kind'], 'quality_preserving_claim': False,
            'scope': 'Descriptive independent final scoring of supplied paired measurements; no population-confidence or agent-superiority claim.'}
