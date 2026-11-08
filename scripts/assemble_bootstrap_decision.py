#!/usr/bin/env python3
"""Verify sealed development evidence and assemble a fresh approved study proposal.

This CPU-only command never chooses thresholds, submits votes, creates a study
plan, or declares held-out quality/numerical/performance acceptance.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drivebench.benchmarks.environment import source_identity
from drivebench.config import digest, load_config, merge, validate_run_config
from drivebench.evaluation.io import candidates, output_path
from drivebench.freeze import ARTIFACTS, METRICS, _checked_run
from scripts import run_bootstrap_evaluation as evaluator

DRAFT_SHA = '0875b69cc7a6fee9dfe210a92fb3a9a48fc708365825c07af1e40fe0d038c69f'
PROTOCOL_SHA = 'c78b6e40a37e79b3c9942caee945f6d868f66ffaa8a2ef138da068033d0bbdbd'
MANIFEST_SHA = 'a3c6328d9f0d0b94149820f626db17e024e6fbc0e5523fe5ee19aa3316befbff'
STATE_SHA = 'cb2726fb57b9d7f1f7a7eb26f83bf40ef15f421885676460a0b2caa020b74d85'
VARIANTS = evaluator.VARIANTS
FIELDS = ['case_id', 'preferred_panel_A_B_tie', 'acceptable_A', 'acceptable_B',
          'object_disappearance', 'object_duplication', 'freezing', 'geometry_change',
          'action_response_notes', 'reviewer_notes']


def read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError('Expected a JSON object')
    return value


def stamp(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('UTC-aware timestamps are required')
    return result.astimezone(timezone.utc)


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


class Ledger:
    """Stream hashes, retain all bindings, and reject files changing during validation."""
    def __init__(self):
        self.hashes, self.signatures = {}, {}

    @staticmethod
    def signature(path):
        value = path.stat()
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns

    def bind(self, path, expected=None):
        path = Path(path).resolve(strict=True)
        if not path.is_file():
            raise ValueError('Expected a regular evidence file')
        name, before = str(path), self.signature(path)
        if name in self.hashes:
            if before != self.signatures[name]:
                raise ValueError('Bound evidence changed during assembly')
            actual = self.hashes[name]
        else:
            actual = evaluator.sha(path)
            if self.signature(path) != before:
                raise ValueError('Evidence changed while hashing')
            self.hashes[name], self.signatures[name] = actual, before
        if expected is not None and (not isinstance(expected, str) or actual != expected):
            raise ValueError('Evidence SHA256 mismatch: ' + str(path))
        return path

    def bind_map(self, values):
        if not isinstance(values, dict) or not values:
            raise ValueError('Missing artifact hash map')
        for path, checksum in values.items():
            if not isinstance(checksum, str) or not re.fullmatch('[0-9a-f]{64}', checksum):
                raise ValueError('Artifact SHA256 must be explicit')
            self.bind(path, checksum)

    def require_bound(self, path, values):
        path = str(Path(path).resolve())
        if path not in values:
            raise ValueError('Required source artifact is absent from the original receipt')
        return self.bind(path, values[path])

    def unchanged(self):
        if any(self.signature(Path(path)) != signature for path, signature in self.signatures.items()):
            raise ValueError('Evidence changed before committing the proposal')


def read_votes(path):
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError('Ratings CSV must preserve the complete exported header')
        rows = list(reader)
    if len(rows) != 4 or len({r['case_id'] for r in rows}) != 4:
        raise ValueError('Exactly four distinct rated cases are required')
    for row in rows:
        if (row.get('preferred_panel_A_B_tie') not in {'A', 'B', 'tie', 'uncertain'}
                or any(row.get(key) not in {'yes', 'no', 'uncertain'} for key in ('acceptable_A', 'acceptable_B'))
                or not any(nonempty(row.get(key)) for key in FIELDS[4:])):
            raise ValueError('Ratings are incomplete or lack observations')
    return {row['case_id']: row for row in rows}


def verify_seal(path, expected_sha, variant, ledger, protocol):
    """Validate externally pinned votes before any answer key is parsed."""
    if not isinstance(expected_sha, str) or not re.fullmatch('[0-9a-f]{64}', expected_sha):
        raise ValueError('An external SHA256 for each sealed rating receipt is required')
    path = ledger.bind(path, expected_sha)
    sealed = read(path)
    if (sealed.get('schema_version') != 1 or sealed.get('receipt_type') != 'bootstrap_blind_ratings_v1'
            or sealed.get('comparison') != variant or sealed.get('reviewer_kind') != 'agent_assisted'
            or sealed.get('private_keys_read_before_seal') is not False
            or sealed.get('all_panels_reviewed') is not True):
        raise ValueError('Missing explicit sealed blinded-review declarations')
    if (Path(sealed['protocol']['path']).resolve() != protocol
            or sealed['protocol']['sha256'] != PROTOCOL_SHA):
        raise ValueError('Blinded protocol differs')
    votes_path = ledger.bind(sealed['ratings']['path'], sealed['ratings']['sha256'])
    index = ledger.bind(sealed['review_index']['path'], sealed['review_index']['sha256'])
    if votes_path == index.parent / 'review.csv':
        raise ValueError('Ratings must be separate from the original blank export')
    ledger.bind_map(sealed['review_artifact_sha256'])
    if any(Path(p).name == 'private_key.json' for p in sealed['review_artifact_sha256']):
        raise ValueError('The blind-review artifact map must exclude answer keys')
    votes = read_votes(votes_path)
    if (len(sealed.get('reviewed_case_ids', [])) != 4 or set(sealed['reviewed_case_ids']) != set(votes)
            or sealed.get('reviewed_frames_per_case') != {key: 24 for key in votes}):
        raise ValueError('All four cases and all 24 future frames must be attested')
    uncertain = sorted(key for key, row in votes.items() if 'uncertain' in
                       (row['preferred_panel_A_B_tie'], row['acceptable_A'], row['acceptable_B']))
    if sorted(sealed.get('uncertain_case_ids', [])) != uncertain:
        raise ValueError('Uncertainty declaration differs from the saved votes')
    for name in ['index.html', 'review.csv', *[key + ext for key in votes for ext in ('.png', '.gif')]]:
        ledger.require_bound(index.parent / name, sealed['review_artifact_sha256'])
    created = stamp(sealed['created_at'])
    if created < stamp(read(protocol)['created_at']) or created > datetime.now(timezone.utc):
        raise ValueError('Review seal timestamp does not follow the predeclared protocol')
    return {'path': str(path), 'sealed': sealed, 'votes': votes, 'index': index, 'created': created}


def checked_policy(path, rationale, ledger):
    path = ledger.bind(path)
    policy = yaml.safe_load(path.read_text())
    if (not isinstance(policy, dict) or policy.get('policy_type') != 'reviewed_metric_tolerances'
            or policy.get('frozen') or policy.get('calibrated') or policy.get('performance', {}).get('calibrated')):
        raise ValueError('Supply an explicit, unfrozen development quality draft')
    metrics = policy.get('metrics', {})
    if (not isinstance(metrics, dict) or not ({'mae', 'rmse', 'max_abs_error'} & metrics.keys())
            or 'lpips' not in metrics or not ({'temporal_change_error', 'freeze_fraction_increase', 'temporal_derivative_mae'} & metrics.keys())):
        raise ValueError('Explicit numerical, LPIPS and temporal limits are required')
    if not isinstance(rationale, dict) or set(rationale) != set(metrics) or not all(map(nonempty, rationale.values())):
        raise ValueError('Every chosen metric limit needs an explicit rationale')
    for key, item in metrics.items():
        if (key not in METRICS or not isinstance(item, dict) or set(item) != {'maximum', 'aggregation'}
                or item['aggregation'] not in {'max', 'mean'} or type(item['maximum']) not in (int, float)
                or not math.isfinite(item['maximum']) or item['maximum'] < 0):
            raise ValueError('Invalid explicit metric limit; no defaults are selected')
    return path, policy


def verify_evaluation(qualification, variant, seal, ledger, root):
    path = ledger.bind(qualification / f'{variant}-vs-upstream.json')
    launch_path = ledger.bind(qualification / f'{variant}-evaluation-launch.json')
    launch, document = read(launch_path), read(path)
    if (launch.get('status') != 'completed' or launch.get('exit_code') != 0
            or launch.get('inputs_unchanged') is not True or launch.get('variant') != variant
            or launch.get('source_identity') != evaluator.SOURCE
            or launch.get('upstream_generation_source_identity') != evaluator.UPSTREAM_SOURCE
            or launch.get('quality_acceptance_claim') is not False or launch.get('descriptive_only') is not True):
        raise ValueError('Completed descriptive evaluation provenance is required')
    ledger.bind_map({str(path): launch.get('output_sha256')})
    bound = launch['bound_file_sha256']
    ledger.bind_map(bound)
    ledger.bind_map(launch['review_artifact_sha256'])
    for helper in ('run_bootstrap_evaluation.py', 'run_compiled_bootstrap.py'):
        ledger.require_bound(root / 'scripts' / helper, bound)
    recipe = ledger.require_bound(root / 'configs/study_bootstrap_compiled_hostspec.json', bound)
    ledger.bind(recipe, evaluator.RECIPE_SHA)
    if seal['created'] < stamp(launch['finished_at']):
        raise ValueError('Ratings must be sealed after evaluation/export completes')
    metric = document.get('perceptual', {})
    if (metric.get('status') != 'available' or metric.get('provenance', {}).get('device') != 'cuda'
            or metric['provenance'].get('setup_receipt_sha256') != evaluator.SETUP_SHA
            or metric['provenance'].get('state_dict_sha256') != STATE_SHA):
        raise ValueError('Verified LPIPS model/weights are required')
    for relative, checksum in document['evaluator_source_sha256'].items():
        source = (root / 'drivebench' / relative).resolve()
        if not source.is_relative_to(root / 'drivebench'):
            raise ValueError('Evaluator source path escapes the package')
        ledger.bind(source, checksum)
    expected_sources = {str(p.relative_to(root / 'drivebench')) for p in
                        [root / 'drivebench/evaluate.py', *sorted((root / 'drivebench/evaluation').glob('*.py'))]}
    if set(document['evaluator_source_sha256']) != expected_sources:
        raise ValueError('Evaluator source inventory is incomplete')
    if len(document.get('runs', [])) != 1:
        raise ValueError('Expected one bootstrap comparison')
    entry = document['runs'][0]
    if (entry.get('run') != variant or entry.get('synthetic') or entry.get('complete_pairs') is not True
            or entry.get('matched_pairs') != 32 or entry.get('errors')
            or entry.get('gate', {}).get('quality_preserving') is not False
            or document.get('tolerances', {}).get('frozen') is not False):
        raise ValueError('Require all 32 descriptive real pairs with the historical gate unchanged')
    reference, manifest, scenes, _ = _checked_run(document['reference'], document['reference_artifact_sha256'], 32)
    candidate, other_manifest, other_scenes, _ = _checked_run(entry['run_path'], entry['artifact_sha256'], 32)
    if manifest != MANIFEST_SHA or other_manifest != manifest or scenes != other_scenes or len(scenes) != 32:
        raise ValueError('Bootstrap development manifest differs')
    runs = {'upstream': reference, variant: candidate}
    for name, run in runs.items():
        host_name = ('study-upstream-development-attempt-1.json' if name == 'upstream'
                     else f'study-bootstrap-{name}-attempt-1.json')
        host_path = ledger.require_bound(root / 'outputs/host' / host_name, bound)
        evaluator.validate_host_receipt(host_path, evaluator.UPSTREAM_SOURCE if name == 'upstream' else evaluator.SOURCE, str(run.path))
        for filename in ARTIFACTS:
            ledger.require_bound(run.path / filename, bound)
    indexed = {name: candidates(run) for name, run in runs.items()}
    pairs = entry.get('pairs', [])
    keys = [(p['request_id'], p['repeat'], p['candidate_id'], p['seed']) for p in pairs]
    if (len(keys) != 32 or len(set(keys)) != 32 or set(keys) != set(indexed['upstream'])
            or set(keys) != set(indexed[variant]) or {p['scene_id'] for p in pairs} != set(scenes)):
        raise ValueError('Pair identities do not cover exactly the measured development clips')
    for pair, key in zip(pairs, keys):
        for name, prefix in (('upstream', 'reference'), (variant, 'candidate')):
            row, item = indexed[name][key]
            if row['scene_id'] != pair['scene_id']:
                raise ValueError('Pair scene differs from its measurement')
            array = ledger.require_bound(output_path(runs[name], item['frames_path']), bound)
            ledger.bind(array, pair[prefix + '_output_sha256'])
        if not isinstance(pair.get('recorded_ground_truth'), dict):
            raise ValueError('Recorded-ground-truth diagnostics are missing')
    review = document['review']
    if review.get('cases') != 4 or Path(review['index']).resolve() != seal['index']:
        raise ValueError('Review index/case coverage differs')
    for value in review['index'], review['csv'], review['private_key']:
        ledger.require_bound(value, launch['review_artifact_sha256'])
    expected_argv = ['drivebench.evaluate', '--runs', str(candidate.path), '--reference', str(reference.path),
                     '--tolerances', str(root / 'configs/study_bootstrap_unfrozen_tolerances.yaml'),
                     '--lpips', '--lpips-device', 'cuda', '--review-limit', '4', '--output', str(path)]
    if launch.get('argv') != expected_argv:
        raise ValueError('Evaluation argv differs from the audited bootstrap comparison')
    return {'path': str(path), 'launch': str(launch_path), 'document': document, 'entry': entry, 'pairs': pairs}


def attribute_votes(evidence, seal, ledger, variant):
    """Called only after both externally pinned seals have passed validation."""
    private = ledger.bind(evidence['document']['review']['private_key'])
    key = read(private)
    cases = key.get('cases', [])
    votes = seal['votes']
    if key.get('seed') != 2026 or len(cases) != 4 or {c['case_id'] for c in cases} != set(votes):
        raise ValueError('Private key does not match the sealed four cases')
    pairs = {(p['request_id'], p['repeat'], p['candidate_id'], p['seed']): p for p in evidence['pairs']}
    selected = sorted(evidence['pairs'], key=lambda p: hashlib.sha256(
        f"2026:{p['scene_id']}:{p['candidate_id']}:{p['seed']}".encode()).digest())[:4]
    if {(p['request_id'], p['repeat'], p['candidate_id'], p['seed']) for p in cases} != {
            (p['request_id'], p['repeat'], p['candidate_id'], p['seed']) for p in selected}:
        raise ValueError('Review key does not use the fixed metric-independent subset')
    accepted, observations = True, []
    for case in cases:
        pair = pairs[(case['request_id'], case['repeat'], case['candidate_id'], case['seed'])]
        if (case.get('run') != variant or case.get('reference_run') != 'upstream'
                or {case.get('A'), case.get('B')} != {'upstream', variant}
                or any(case.get(field) != pair.get(field) for field in ('scene_id', 'action_kind', 'trajectory_sha256'))
                or case.get('conditioning_frames') != 1
                or case.get('reference_sha256') != pair['reference_output_sha256']
                or case.get('candidate_sha256') != pair['candidate_output_sha256']):
            raise ValueError('Answer key identity/output hashes differ from the paired metrics')
        identity = {key: case.get(key) for key in ('request_id', 'scene_id', 'repeat', 'candidate_id', 'seed',
                    'run', 'reference_run', 'action_kind', 'trajectory_sha256', 'conditioning_frames')}
        identity.update(reference_sha256=case['reference_sha256'], candidate_sha256=case['candidate_sha256'], review_seed=2026)
        expected_id = 'case-' + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
        swapped = bool(hashlib.sha256(f'2026:{expected_id}'.encode()).digest()[0] & 1)
        if case['case_id'] != expected_id or case['A'] != (variant if swapped else 'upstream'):
            raise ValueError('Case ID or blinded panel mapping differs')
        panel = 'A' if case['A'] == variant else 'B'
        vote = votes[case['case_id']]['acceptable_' + panel]
        accepted = accepted and vote == 'yes'
        observations.append({'case_id': case['case_id'], 'candidate_panel': panel, 'candidate_vote': vote})
    return {'candidate_accepted_in_all_cases': accepted, 'cases': observations,
            'blindness_basis': 'reviewer declaration in externally SHA-pinned pre-key receipt; not independently observable'}


def build_spec(draft, selected):
    spec = copy.deepcopy(draft)
    spec['optimized_reference'] = selected
    if selected == 'conditioning-reuse':
        variants = spec['variants']
        feature_control, step_control = copy.deepcopy(variants['feature-2']), copy.deepcopy(variants['steps-47'])
        for name in list(variants):
            if name.startswith(('steps-', 'feature-')):
                # YAML aliases can share dictionaries; detach every variant before mutation.
                variants[name] = copy.deepcopy(variants[name])
                variants[name]['model'] = {'compiler_runtime': None, 'compiler_feature_boundary': None}
                variants[name]['optimizations'].update(compile_denoiser=False, compile_decoder=False)
        variants = {('compiled-feature-2-control' if name == 'eager-feature-2' else
                     'compiled-steps-47-control' if name == 'eager-steps-47' else name):
                    (feature_control if name == 'eager-feature-2' else step_control if name == 'eager-steps-47' else value)
                    for name, value in variants.items()}
        spec['variants'] = variants
    spec['draft_status'] = {**spec['draft_status'], 'materialized': False,
                            'reference_decision': 'explicit_development_decision_assembled',
                            'selected_optimized_reference': selected,
                            'notice': 'Approved proposal only; no study plan, held-out acceptance or numerical proof is created.'}
    return spec


def assemble(*, project, draft, qualification, decisions, rating_receipts, rating_hashes, policies, output):
    root, output = Path(project).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a new output directory; approved evidence cannot be overwritten')
    if source_identity() != evaluator.SOURCE:
        raise ValueError('Frozen package source changed')
    ledger = Ledger()
    draft_path = ledger.bind(draft, DRAFT_SHA)
    protocol = ledger.bind(root / 'outputs/bootstrap-review-protocol-v1.json', PROTOCOL_SHA)
    decision_path = ledger.bind(decisions)
    choice = read(decision_path)
    if (choice.get('schema_version') != 1 or choice.get('held_out_consulted') is not False
            or choice.get('reviewer_kind') != 'agent_assisted'
            or choice.get('selected_optimized_reference') not in {'compiled-conditioning', 'conditioning-reuse'}
            or set(choice.get('variants', {})) != set(VARIANTS)
            or not nonempty(choice.get('reference_rationale'))):
        raise ValueError('Explicit reference and both variant decisions with development-only scope are required')
    if any(set(values) != set(VARIANTS) for values in (rating_receipts, rating_hashes, policies)):
        raise ValueError('Both comparisons need explicit sealed ratings, external hashes and policy drafts')
    seals = {name: verify_seal(rating_receipts[name], rating_hashes[name], name, ledger, protocol) for name in VARIANTS}
    decided_at = stamp(choice['created_at'])
    if decided_at < max(seal['created'] for seal in seals.values()) or decided_at > datetime.now(timezone.utc):
        raise ValueError('Explicit decisions must follow both sealed reviews')
    qualified = {name: verify_evaluation(Path(qualification).resolve(), name, seals[name], ledger, root) for name in VARIANTS}
    outcomes = {}
    for name in VARIANTS:
        explicit = choice['variants'][name]
        if explicit.get('status') not in {'accepted', 'rejected'} or not nonempty(explicit.get('rationale')):
            raise ValueError('Every variant needs an explicit accepted/rejected decision and rationale')
        policy_path, policy = checked_policy(policies[name], explicit.get('policy_rationales'), ledger)
        attribution = attribute_votes(qualified[name], seals[name], ledger, name)
        observed, within = {}, True
        for metric, setting in policy['metrics'].items():
            values = [p['metrics'].get(metric) for p in qualified[name]['pairs']]
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
                raise ValueError('Chosen metric has missing or nonfinite pairs')
            actual = max(values) if setting['aggregation'] == 'max' else sum(values) / len(values)
            declared = qualified[name]['entry']['aggregates'].get(metric, {})
            if (declared.get('pairs') != 32 or type(declared.get(setting['aggregation'])) not in (int, float)
                    or not math.isclose(declared[setting['aggregation']], actual, rel_tol=1e-12, abs_tol=1e-12)):
                raise ValueError('Metric aggregate differs from the 32 raw pairs')
            observed[metric] = actual
            within = within and actual <= setting['maximum']
        if explicit['status'] == 'accepted' and (not within or not attribution['candidate_accepted_in_all_cases']):
            raise ValueError('Explicit acceptance conflicts with chosen limits or actual candidate votes')
        outcomes[name] = {'explicit_decision': explicit, 'evaluation': qualified[name]['path'],
                          'evaluation_launch': qualified[name]['launch'], 'rating_seal': seals[name]['path'],
                          'threshold_draft': str(policy_path), 'thresholds': policy['metrics'],
                          'observed_metrics': observed, 'within_explicit_limits': within, 'review': attribution}
    selected = choice['selected_optimized_reference']
    if (selected == 'compiled-conditioning') != (choice['variants']['compiled-conditioning']['status'] == 'accepted'):
        raise ValueError('Selected secondary reference contradicts the explicit compiler decision')
    approved = build_spec(yaml.safe_load(draft_path.read_text()), selected)
    base = load_config(approved['base_config'])
    if len(approved['variants']) != 18:
        raise ValueError('Approved proposal must retain 18 distinct variants')
    for changes in approved['variants'].values():
        validate_run_config(merge(base, changes))
    source_files = {str(p.relative_to(root / 'drivebench')): evaluator.sha(p)
                    for p in sorted((root / 'drivebench').rglob('*.py'))}
    if digest(source_files) != evaluator.SOURCE:
        raise ValueError('Project package source inventory differs from the executing frozen package')
    for path in sorted((root / 'drivebench').rglob('*.py')):
        ledger.bind(path)
    ledger.bind(__file__)
    # Bind inherited proposal inputs too; their relative paths retain project-root semantics.
    from scripts.run_compiled_bootstrap import config_bindings
    ledger.bind_map({str(root / name): value for name, value in config_bindings(root, root / approved['base_config']).items()})
    ledger.unchanged()
    output.mkdir(parents=True, exist_ok=False)
    spec_path = output / 'approved-spec.yaml'
    with spec_path.open('x') as stream:
        yaml.safe_dump(approved, stream, sort_keys=False)
        stream.flush(); os.fsync(stream.fileno())
    receipt = {'schema_version': 1, 'receipt_type': 'bootstrap_development_decision_v1',
               'created_at': datetime.now(timezone.utc).isoformat(), 'source_identity': evaluator.SOURCE,
               'status': 'approved_proposal_only', 'selected_optimized_reference': selected,
               'decision': choice, 'comparison_evidence': outcomes,
               'bound_file_sha256': ledger.hashes,
               'approved_spec': str(spec_path), 'approved_spec_sha256': evaluator.sha(spec_path),
               'held_out_consulted': False, 'study_materialized': False, 'release_policy_frozen': False,
               'historical_numerical_gate_changed': False, 'headline_claim': False,
               'remaining_requirements': ['Main development aggregate and separately frozen reference-specific policies',
                   'All-32 exact conditioning anchor; current scoped numerical proofs for any headline timing',
                   'Held-out metrics and accepted blinded reviews; workload-specific variability and timing evidence',
                   'Branch-specific live budget reconciliation; no allocation or funding created by assembly']}
    ledger.unchanged()
    with (output / 'decision-receipt.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    descriptor = os.open(output, os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)
    return receipt


def assignments(values):
    result = {}
    for item in values:
        name, value = item.split('=', 1)
        if name in result or name not in VARIANTS or not value:
            raise ValueError('Assignments must name each bootstrap variant exactly once')
        result[name] = value
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draft', default='configs/full_study_registered_draft.yaml')
    parser.add_argument('--qualification', default='outputs/study-bootstrap/qualification-v1')
    parser.add_argument('--decisions', required=True)
    parser.add_argument('--ratings', action='append', required=True, metavar='VARIANT=RECEIPT')
    parser.add_argument('--ratings-sha256', action='append', required=True, metavar='VARIANT=EXTERNALLY_SEALED_SHA256')
    parser.add_argument('--policy', action='append', required=True, metavar='VARIANT=EXPLICIT_DRAFT')
    parser.add_argument('--output', required=True, help='new directory for approved-spec.yaml and decision-receipt.json')
    args = parser.parse_args(argv)
    try:
        result = assemble(project=ROOT, draft=args.draft, qualification=args.qualification, decisions=args.decisions,
                          rating_receipts=assignments(args.ratings), rating_hashes=assignments(args.ratings_sha256),
                          policies=assignments(args.policy), output=args.output)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(1, f'assemble bootstrap decision: {error}\n')
    print(json.dumps({'status': result['status'], 'selected_optimized_reference': result['selected_optimized_reference'],
                      'approved_spec_sha256': result['approved_spec_sha256'], 'study_materialized': False}))


if __name__ == '__main__':
    main()
