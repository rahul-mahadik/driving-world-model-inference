#!/usr/bin/env python3
"""Export allowlisted v2 trajectories after the separate statistical export.

This does not run a GPU or calculate policy aggregates. Every parent environment
and evaluator journal event is retained, in its original stream order. Streams
lack a common timestamp, so no cross-stream ordering is invented.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics

_SPEC = importlib.util.spec_from_file_location('schedule_analysis_v2', Path(__file__).with_name('analyze_schedule_study_v2.py'))
analysis = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(analysis)
PROTOCOL_SHA256 = 'cee48654a1055da038e1101c49d6678c4e7bf057d14e1ff1adfd5edb52aa30cc'
IDENTITY = ('episode_id', 'task_id', 'seed', 'policy', 'order_position')
ACTIONS = ('block_down', 'block_up', 'warps_down', 'warps_up', 'unfuse', 'fuse',
    'activation_block_down', 'activation_block_up', 'activation_warps_down', 'activation_warps_up',
    'reduction_single', 'reduction_chunked', 'reduction_block_down', 'reduction_block_up',
    'reduction_warps_down', 'reduction_warps_up', 'measure', 'submit')
OUTCOMES = {'edited', 'budget_exhausted', 'invalid_action', 'evaluator_error', 'ok', 'incorrect', 'error', 'timeout'}
ERROR_TYPES = {'TimeoutError', 'ValueError', 'TypeError', 'RuntimeError', 'AssertionError', 'OSError',
    'BrokenPipeError', 'EOFError', 'KeyboardInterrupt', 'OutOfResources', 'CompilationError',
    'CompileTimeAssertionFailure', 'OutOfMemoryError', 'CalledProcessError', 'FileNotFoundError',
    'PermissionError', 'ConnectionError', 'MemoryError'}
COMPONENTS = ('speed_score', 'elapsed_penalty', 'invalid_action_penalty', 'failed_measurement_penalty')


def fail(message):
    raise ValueError(message)


def boolean(value):
    if type(value) is not bool:
        fail('Expected boolean')
    return value


def nonnegative(value):
    return analysis.number(value, nonnegative=True)


def error_type(value):
    if value is None:
        return None
    name = value.get('type') if isinstance(value, dict) else str(value).partition(':')[0]
    return name if name in ERROR_TYPES else 'OtherError'


def input_hashes(value):
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        fail('Input hashes must be a nonempty list')
    return [analysis.digest_string(item) for item in value]


def correctness(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        fail('Correctness evidence must be an object')
    result = {}
    for key in ('passed', 'finite', 'shape_dtype_match'):
        if key in value:
            result[key] = boolean(value[key])
    for key in ('atol', 'rtol', 'max_abs_error', 'rmse'):
        if key in value:
            result[key] = None if value[key] is None else nonnegative(value[key])
    return result


def schedule(value):
    result = analysis.public_schedule(value)
    if result is not None:
        for key in ('block_size', 'activation_block_size', 'reduction_block_size'):
            if result[key] not in (64, 128, 256, 512, 1024, 2048, 4096):
                fail('Unsupported schedule block size')
        for key in ('num_warps', 'activation_num_warps', 'reduction_num_warps'):
            if result[key] not in (1, 2, 4, 8):
                fail('Unsupported schedule warp count')
    return result


def _read(inputs, relative, *, json_lines=False, external=None):
    if relative not in inputs.files:
        return inputs.read(relative, external=external, json_lines=json_lines)
    path, sha, size = inputs.files[relative]
    raw = path.read_bytes()
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != sha:
        fail('Source evidence changed during trajectory export')
    if json_lines:
        return [analysis.parse_json(line) for line in raw.decode().splitlines() if line.strip()]
    return analysis.parse_json(raw.decode())


def _role(value, workload_key):
    """Legacy reference-role measurements use their own (non-sampled) schema."""
    if value.get('workload_key') != workload_key or value.get('config') is not None:
        fail('Reference role workload/config differs')
    status = value.get('status')
    if status not in ('ok', 'incorrect', 'error') or value.get('correct') is not (status == 'ok'):
        fail('Unknown reference role outcome')
    evidence = value.get('evidence', {})
    samples = evidence.get('cuda_event_ms', [])
    if not isinstance(samples, list) or any(analysis.number(x) <= 0 for x in samples):
        fail('Invalid reference role samples')
    if status == 'ok' and len(samples) != 7:
        fail('Successful reference role requires seven samples')
    return dict(status=status, correct=value['correct'], raw_samples_ms=samples,
        median_ms=statistics.median(samples) if samples else None,
        peak_allocated_bytes=None if value.get('peak_allocated_bytes') is None else analysis.count(value['peak_allocated_bytes']),
        preparation_seconds=None if evidence.get('preparation_seconds') is None else nonnegative(evidence['preparation_seconds']),
        correctness=correctness(evidence.get('correctness')), input_sha256=input_hashes(evidence.get('input_sha256')),
        error_type=error_type(evidence.get('error')))


def measurement(value, *, fixture_only):
    result = analysis.public_measurement(value, fixture_only=fixture_only)
    evidence = value.get('evidence', {})
    sampled_index = evidence.get('sampled_index')
    if sampled_index is not None and (type(sampled_index) is not int or not 0 <= sampled_index < 7):
        fail('Sampled index must be null or an integer in the seven-sample contract')
    result.update(trial_id=analysis.identifier(evidence['trial_id']),
        workload_key=analysis.digest_string(value['workload_key']), schedule=schedule(value['schedule']),
        sampled_latency_ms=value['latency_ms'], sampled_index=sampled_index,
        input_sha256=input_hashes(evidence.get('input_sha256')), correctness=correctness(evidence.get('correctness')),
        error_type=error_type(evidence.get('error')))
    for key in ('worker_reaped', 'input_integrity_verified', 'compiled_fallback'):
        if key in evidence:
            result[key] = boolean(evidence[key])
    for key in ('timeout_seconds', 'preparation_seconds'):
        if key in evidence:
            result[key] = nonnegative(evidence[key])
    for key in ('selected_reference', 'fixed_reference_confirmation'):
        if key in evidence:
            if evidence[key] not in ('eager', 'compiled'):
                fail('Unknown reference implementation')
            result[key] = evidence[key]
    if 'baseline_roles' in evidence:
        if set(evidence['baseline_roles']) != {'eager', 'compiled'}:
            fail('Reference setup must preserve both roles')
        result['baseline_roles'] = {role: _role(raw, value['workload_key'])
            for role, raw in evidence['baseline_roles'].items()}
    return result


def episode_records(inputs, spec, workload, source_hashes, *, fixture_only=False, seen_trials=None):
    """Sanitize one episode, also usable for non-publishing development inspection."""
    seen_trials = set() if seen_trials is None else seen_trials
    identity = {key: spec[key] for key in IDENTITY}
    episode = analysis.identifier(identity['episode_id'])
    prefix = 'episodes/' + episode + '/'
    workload_key = analysis.canonical_hash(workload)
    result = _read(inputs, prefix + 'result.json')
    if any(result.get(key) != identity[key] for key in IDENTITY):
        fail('Trajectory episode identity differs from protocol')
    query_path = prefix + 'evaluator/queries.jsonl'
    query_missing = not (inputs.root / query_path).exists()
    failed_before_baseline = (result.get('status') in ('failed', 'interrupted', 'cleanup_failed')
        and all(result.get(key) is None for key in ('baseline', 'paired_final_reference', 'confirmation')))
    if query_missing and not failed_before_baseline:
        fail('Started episode lacks its evaluator journal')
    query_events = [] if query_missing else _read(inputs, query_path, json_lines=True)
    starts, finishes, sanitized, records = {}, {}, {}, []
    runtime_identity = None
    created = closed = False
    for index, event in enumerate(query_events):
        kind = event.get('event')
        row = dict(identity, stream='evaluator', event_index=index, event=kind)
        if kind == 'created':
            if created or index != 0:
                fail('Duplicate or misplaced evaluator creation')
            created = True
            if event.get('repetitions') != 7 or event.get('warmups') != 2:
                fail('Evaluator sample contract differs')
            if event.get('sampling_seed') != identity['seed']:
                fail('Evaluator sampling seed differs from protocol episode')
            row.update(sampling_seed=analysis.count(event['sampling_seed']), repetitions=7, warmups=2)
        elif kind == 'query_started':
            if not created or closed or any(key not in finishes for key in starts):
                fail('Evaluator query order is inconsistent')
            trial = analysis.identifier(event['trial_id'])
            if trial in seen_trials:
                fail('Repeated evaluator trial identity')
            seen_trials.add(trial)
            if event.get('workload') != workload or event.get('repetitions') != 7 or event.get('warmups') != 2:
                fail('Evaluator query workload/sample contract differs')
            role = event.get('reference_role')
            candidate = schedule(event.get('schedule'))
            if role not in (None, 'compiled', 'eager') or (role is not None and candidate is not None):
                fail('Evaluator query role/schedule differs')
            starts[trial] = event
            row.update(trial_id=trial, workload_key=workload_key, schedule=candidate,
                       reference_role=role, timeout_seconds=nonnegative(event['timeout_seconds']), repetitions=7, warmups=2)
        elif kind == 'query_finished':
            value = event['measurement']
            trial = analysis.identifier(value.get('evidence', {}).get('trial_id'))
            if trial not in starts or trial in finishes or closed:
                fail('Unmatched or repeated evaluator completion')
            request = starts[trial]
            if value.get('workload_key') != workload_key or value.get('schedule') != request.get('schedule'):
                fail('Evaluator completion workload/schedule differs')
            evidence = value.get('evidence', {})
            if value.get('correct') and request.get('reference_role') is not None:
                if evidence.get('fixed_reference_confirmation') != request['reference_role']:
                    fail('Fixed reference confirmation role differs')
            if value.get('correct'):
                analysis.verify_runtime_identity(evidence, source_hashes, fixture_only=fixture_only)
                bound = (evidence.get('runtime'), evidence.get('cohort_id'), evidence.get('input_sha256'))
                if runtime_identity is not None and bound != runtime_identity:
                    fail('Intermediate query runtime/input identity differs')
                runtime_identity = bound
            finishes[trial] = value
            sanitized[trial] = measurement(value, fixture_only=fixture_only)
            row['measurement'] = sanitized[trial]
        elif kind == 'closed':
            if not created or closed:
                fail('Duplicate or misplaced evaluator cleanup')
            closed = True
            row['cleanup_seconds'] = nonnegative(event['cleanup_seconds'])
        else:
            fail('Unknown evaluator journal event')
        records.append(row)
    if not created and not query_missing:
        fail('Evaluator creation record is missing')
    if len(starts) != len(finishes) and result.get('status') not in ('failed', 'interrupted', 'cleanup_failed'):
        fail('Incomplete query is not bound to a failed/interrupted episode')

    env_dir = inputs.root / prefix / 'environment'
    paths = sorted(env_dir.glob('*/environment.jsonl')) if env_dir.exists() else []
    if len(paths) != 1 and not (not paths and failed_before_baseline):
        fail('Expected exactly one environment journal per started episode')
    env_path = paths[0].relative_to(inputs.root).as_posix() if paths else None
    events = _read(inputs, env_path, json_lines=True) if paths else []
    current = None
    pending = None
    returned = None
    trial_failed = False
    baseline_seen = False
    last_elapsed = 0.
    last_step_elapsed = 0.
    terminal = False
    env_trials = set()
    rewards = []

    def bound_measurement(value):
        trial = analysis.identifier(value.get('evidence', {}).get('trial_id'))
        if finishes.get(trial) != value:
            fail('Environment measurement differs from evaluator journal')
        return sanitized[trial]

    for index, event in enumerate(events):
        kind = event.get('event')
        row = dict(identity, stream='environment', event_index=index, event=kind)
        if kind == 'reset_started':
            if index != 0 or current is not None or event.get('workload') != workload or event.get('budget_seconds') != 90:
                fail('Environment reset identity/budget differs')
            current = schedule(event['initial_schedule'])
            if current is None:
                fail('Initial schedule is missing')
            row.update(workload_key=workload_key, budget_seconds=90, initial_schedule=current)
        elif kind == 'baseline':
            if current is None or index != 1 or event['measurement'].get('schedule') is not None:
                fail('Environment baseline order/schedule differs')
            value = event['measurement']
            trial = value.get('evidence', {}).get('trial_id')
            if (not starts or trial != next(iter(starts)) or starts[trial].get('reference_role') is not None
                    or starts[trial].get('schedule') is not None):
                fail('Environment baseline is not the initial reference-selection query')
            if result.get('baseline') is not None and value != result['baseline']:
                fail('Environment baseline differs from retained initial baseline')
            baseline_seen = True
            row['measurement'] = bound_measurement(value)
        elif kind == 'trial_started':
            if pending is not None or terminal or event.get('action') not in ('measure', 'submit') or event.get('schedule') != current:
                fail('Environment trial state/action differs')
            pending = event
            trial_failed = False
            row.update(action=event['action'], schedule=schedule(event['schedule']), remaining_seconds=nonnegative(event['remaining_seconds']))
        elif kind == 'trial_returned':
            if pending is None or returned is not None or trial_failed or event['measurement'].get('schedule') != current:
                fail('Environment returned trial is unmatched')
            public = bound_measurement(event['measurement'])
            if public['trial_id'] in env_trials:
                fail('Repeated environment measurement identity')
            env_trials.add(public['trial_id'])
            returned = event['measurement']
            row['measurement'] = public
        elif kind in ('reset_failed', 'trial_error'):
            if kind == 'trial_error':
                if pending is None or trial_failed:
                    fail('Unmatched environment trial error')
                trial_failed = True
            row['error_type'] = error_type(event.get('error'))
        elif kind == 'step':
            action = event.get('action')
            if terminal or action not in ACTIONS or event.get('outcome') not in OUTCOMES or event.get('schedule_before') != current:
                fail('Environment step state/action differs')
            if action in ('measure', 'submit') and event['outcome'] != 'budget_exhausted':
                if pending is None or pending['action'] != action:
                    fail('Environment step does not match its trial')
                if trial_failed:
                    if event['outcome'] != 'evaluator_error' or event.get('measurement') is not None:
                        fail('Rejected trial requires an evaluator_error step without an accepted measurement')
                elif returned is None or event.get('measurement') != returned or event['outcome'] == 'evaluator_error':
                    fail('Environment step does not match its trial')
            elif pending is not None:
                fail('Environment edit overlaps a measurement')
            after = schedule(event['schedule_after'])
            if after is None or (action in ('measure', 'submit') or event['outcome'] in ('invalid_action', 'budget_exhausted')) and after != current:
                fail('Non-edit action changed its schedule')
            row.update(action=action, schedule_before=schedule(current), schedule_after=after,
                outcome=event['outcome'], reward=analysis.number(event['reward']), submission_score=analysis.number(event['submission_score']),
                terminated=boolean(event['terminated']), truncated=boolean(event['truncated']),
                measurement=bound_measurement(event['measurement']) if event.get('measurement') is not None else None)
            components = event.get('reward_components')
            if not isinstance(components, dict) or set(components) != set(COMPONENTS):
                fail('Unknown reward component schema')
            row['reward_components'] = {key: analysis.number(components[key]) for key in COMPONENTS}
            expected = components['speed_score'] - sum(components[key] for key in COMPONENTS[1:])
            if not analysis.equal_number(expected, row['reward']) or not analysis.equal_number(components['speed_score'], row['submission_score']):
                fail('Recorded reward components differ from reward')
            elapsed = nonnegative(event['elapsed_seconds'])
            row['elapsed_since_previous_step_seconds'] = elapsed - last_step_elapsed
            last_step_elapsed = elapsed
            rewards.append(row['reward'])
            terminal = row['terminated'] or row['truncated']
            current, pending, returned, trial_failed = after, None, None, False
        else:
            fail('Unknown environment journal event')
        if 'elapsed_seconds' in event:
            elapsed = nonnegative(event['elapsed_seconds'])
            if elapsed < last_elapsed:
                fail('Environment elapsed time moved backwards')
            last_elapsed = elapsed
            row['elapsed_seconds'] = elapsed
        records.append(row)
    if current is None and paths:
        fail('Environment reset record is missing')
    if result.get('baseline') is not None and not baseline_seen:
        fail('Successfully reset episode is missing its environment baseline event')
    completed_candidates = {trial for trial, request in starts.items() if request.get('schedule') is not None and trial in finishes}
    if completed_candidates != env_trials:
        fail('Completed candidate queries are missing from the environment journal')
    if last_elapsed > nonnegative(result['elapsed_seconds']):
        fail('Environment elapsed time exceeds retained episode duration')
    if result.get('environment_return') is not None and not analysis.equal_number(sum(rewards), result['environment_return']):
        fail('Environment rewards differ from retained episode return')
    counts = Counter(row['stream'] for row in records)
    return records, dict(identity, evaluator_events=counts['evaluator'], environment_events=counts['environment'],
        queries_started=len(starts), queries_finished=len(finishes), queries_without_recorded_completion=len(starts)-len(finishes),
        failed_queries=sum(not value['correct'] for value in finishes.values()),
        missing_evaluator_journal=query_missing, missing_environment_journal=not paths,
        source_journals=([query_path] if not query_missing else []) + ([env_path] if env_path else []))


def source_receipts(inputs):
    """Use logical source identities, never local paths or environment UUIDs."""
    result = []
    fixed = {'protocol.json': 'protocol', 'summary.json': 'study_summary', 'completed.json': 'completion',
        'episodes.jsonl': 'study_episodes', 'source-manifest.json': 'source_manifest',
        'frozen-protocol.json': 'frozen_protocol', 'analysis-plan.json': 'analysis_plan',
        'validated-public-episodes.json': 'validated_public_episodes',
        'validated-analysis-provenance.json': 'validated_analysis_provenance'}
    for item in inputs.manifest():
        name = item['file']
        row = {'sha256': item['sha256'], 'bytes': item['bytes']}
        if name in fixed:
            row['source'] = fixed[name]
        elif name.startswith('scientific-code/') and name.split('/')[-1] in analysis.SOURCE_MODULES:
            row.update(source='scientific_code', module=name.split('/')[-1])
        else:
            match = re.fullmatch(r'episodes/([0-9]{3})/(result.json|evaluator/queries.jsonl|environment/[^/]+/environment.jsonl)', name)
            if match is None:
                fail('Unknown source receipt identity')
            channel = ('episode_result' if match[2] == 'result.json' else
                       'evaluator' if match[2] == 'evaluator/queries.jsonl' else 'environment')
            row.update(source=channel, episode_id=match[1])
        result.append(row)
    return result


def export(study, public_analysis, output, source_manifest, frozen_source, analysis_plan, *, fixture_only=False):
    inputs = analysis.Inputs(study)
    output = Path(output)
    if output.exists() or output.is_symlink() or output.resolve().is_relative_to(inputs.root):
        fail('Trajectory export must be a new directory outside source evidence')
    protocol, _, public_rows, _, _ = analysis.validate_and_sanitize(inputs, source_manifest, frozen_source, fixture_only=fixture_only)
    if not fixture_only and analysis.canonical_hash(protocol) != PROTOCOL_SHA256:
        fail('Trajectory export requires the exact registered v2 protocol')
    plan = _read(inputs, 'analysis-plan.json', external=analysis_plan)
    if plan.get('study_id') != 'sequential-schedules-live-v2' or plan.get('protocol_sha256') != analysis.canonical_hash(protocol) or plan.get('registered_before_gpu_measurement') is not True:
        fail('Analysis plan does not bind this prospective study')
    if not fixture_only and inputs.files['analysis-plan.json'][1] != analysis.REGISTERED_ANALYSIS_PLAN_SHA256:
        fail('Analysis plan differs from the registered bytes')
    public_analysis = Path(public_analysis)
    exported = _read(inputs, 'validated-public-episodes.json', external=public_analysis/'episodes.json')
    expected_kind = 'synthetic-software-test-fixture' if fixture_only else 'measured-cuda-study'
    if exported != dict(evidence_kind=expected_kind, scientific_results=not fixture_only, episodes=public_rows):
        fail('Public episodes differ from independently validated episode identities/results')
    provenance = _read(inputs, 'validated-analysis-provenance.json', external=public_analysis/'provenance.json')
    if provenance.get('protocol_content_sha256') != analysis.canonical_hash(protocol) or provenance.get('registered_analysis_plan_sha256') != inputs.files['analysis-plan.json'][1]:
        fail('Public analysis provenance binding differs')
    if provenance.get('analysis_code', {}).get('sha256') != analysis.file_hash(Path(__file__).with_name('analyze_schedule_study_v2.py')):
        fail('Public analysis was not produced by the registered v2 analyzer')
    workloads = {task['task_id']: task['metadata']['workload'] for task in protocol['tasks']}
    records, counts, seen = [], [], set()
    for spec in protocol['episodes']:
        rows, count = episode_records(inputs, spec, workloads[spec['task_id']], protocol['source_sha256'], fixture_only=fixture_only, seen_trials=seen)
        records.extend(rows)
        # Source paths go only into the hashed manifest, never into trajectory rows.
        count.pop('source_journals')
        counts.append(count)
    if len(counts) != 54:
        fail('Expected all 54 registered episode trajectories')
    inputs.verify_unchanged()
    payload = ''.join(json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n' for row in records).encode()
    compressed = gzip.compress(payload, mtime=0)
    output.mkdir(parents=True, exist_ok=False)
    (output/'trajectories.jsonl.gz').write_bytes(compressed)
    receipt = dict(schema_version=1, study_id='sequential-schedules-live-v2', evidence_kind=expected_kind,
        scientific_results=not fixture_only, statistical_aggregates=False,
        protocol_sha256=analysis.canonical_hash(protocol), episodes=counts, events=len(records),
        streams=dict(Counter(row['stream'] for row in records)),
        ordering='Event indices preserve each parent journal independently. No cross-stream interleaving is inferred.',
        cost_definition='Environment step deltas include elapsed reset/policy/measurement time since the preceding step. Evaluator elapsed times overlap those costs; do not sum the two streams.',
        error_policy='Only recognized exception class names are retained; other classes become OtherError. Messages and stderr are omitted.',
        source_evidence_modified=False, consumed_sources=source_receipts(inputs),
        seed_evidence='Evaluator sampling seed is checked against the protocol episode seed. Environment reset records the fixed workload/input seed; it does not separately record a policy seed.',
        exporter_sha256=analysis.file_hash(__file__),
        archive=dict(file='trajectories.jsonl.gz', sha256=hashlib.sha256(compressed).hexdigest(), bytes=len(compressed),
                     uncompressed_sha256=hashlib.sha256(payload).hexdigest(), uncompressed_bytes=len(payload)))
    analysis.write_json(output/'provenance.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'public-analysis', 'output', 'source-manifest', 'frozen-source', 'analysis-plan'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--fixture-only', action='store_true', help='Synthetic software tests only; never marks results scientific')
    args = parser.parse_args()
    receipt = export(args.study, args.public_analysis, args.output, args.source_manifest, args.frozen_source, args.analysis_plan, fixture_only=args.fixture_only)
    print(json.dumps({'events': receipt['events'], 'episodes': len(receipt['episodes']), 'archive': receipt['archive']}, sort_keys=True))


if __name__ == '__main__':
    main()
