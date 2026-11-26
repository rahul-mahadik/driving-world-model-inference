#!/usr/bin/env python3
"""Verify and export one recovered sequential-schedule study, without GPU work.

Source evidence is read only. Public output is an explicit field allowlist; it
contains no local paths, provider identifiers, GPU UUIDs, or raw exception text.
Bootstrap draws pair policies by task family and seed, not 54 independent rows.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys

POLICIES = ('compiled_eager_no_edit', 'random_walk', 'greedy_local')
SOURCE_MODULES = ('schedule_study.py', 'schedule.py', 'schedule_env.py', 'schedule_kernels.py',
                  '_schedule_triton.py', 'schedule_measurement.py', 'benchmark.py',
                  'contract.py', 'kernels.py', '_triton_kernels.py')
SCHEDULE_FIELDS = ('fused', 'block_size', 'num_warps', 'activation_block_size',
                   'activation_num_warps', 'reduction', 'reduction_block_size', 'reduction_num_warps')
STATUSES = {'started', 'finished', 'failed', 'interrupted', 'cleanup_failed'}
STOP_REASONS = {'fixed_reference_confirmed', 'reference_confirmation_failed', 'confirmation_reserve',
    'ended_during_edit', 'worker_failure_or_budget', 'query_safety_cap', 'ended_restoring_selection',
    'no_confirmation_time', 'unconfirmed_before_deadline', 'episode_exception', 'interrupted', 'cleanup_failed'}
MEASUREMENT_STATUSES = {'ok', 'incorrect', 'error', 'timeout'}
REGISTERED_ANALYSIS_PLAN_SHA256 = '5c4211df3a87fcca7bf8a908b0bfeb388c611b69b93f607ad46a7455a8c98cbd'
SCORE_DEFINITION = ('Confirmed episodes: clip(1 - median(selected confirmation samples) / '
                    'median(paired final reference samples), -1, 1). Failed or unconfirmed episodes: -1. '
                    'The reference role is selected during reset and fixed before final measurement. '
                    'This score differs from the environment return, which uses its initial reference.')


def fail(message):
    raise ValueError(message)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        fail('Expected a finite number')
    if nonnegative and value < 0:
        fail('Expected a nonnegative number')
    return value


def count(value):
    if type(value) is not int or value < 0:
        fail('Expected a nonnegative integer')
    return value


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[a-zA-Z0-9_.-]{1,100}', value):
        fail('Invalid portable identifier')
    return value


def digest_string(value):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value):
        fail('Expected SHA256 digest')
    return value


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            fail('Duplicate JSON key')
        result[key] = value
    return result


def parse_json(text):
    value = json.loads(text, object_pairs_hook=reject_duplicates,
                       parse_constant=lambda _: fail('Nonfinite JSON constant'))
    json.dumps(value, allow_nan=False)
    return value


class Inputs:
    """Hash every consumed file before and after analysis; never mutate inputs."""
    def __init__(self, root):
        original = Path(root)
        if original.is_symlink() or not original.is_dir():
            fail('Recovered study must be a real directory')
        self.root = original.resolve()
        self.files = {}

    def read(self, relative, *, external=None, json_lines=False):
        raw=self.read_raw(relative,external=external)
        if json_lines:
            return [parse_json(line) for line in raw.decode().splitlines() if line.strip()]
        return parse_json(raw.decode())

    def read_raw(self, relative, *, external=None):
        path = Path(external) if external is not None else self.root / relative
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 64*1024**2:
            fail('Input must be a bounded regular file')
        resolved = path.resolve()
        if external is None and not resolved.is_relative_to(self.root):
            fail('Study input escapes recovered directory')
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if relative in self.files:
            fail('Input consumed twice under one identity')
        self.files[relative] = (resolved, sha, len(raw))
        return raw

    def verify_unchanged(self):
        for path, sha, size in self.files.values():
            if path.is_symlink() or path.stat().st_size != size or file_hash(path) != sha:
                fail('Source evidence changed during analysis')

    def manifest(self):
        return [{'file': key, 'sha256': value[1], 'bytes': value[2]} for key, value in sorted(self.files.items())]


def measurement_median(value):
    if not value or value.get('correct') is not True or value.get('status') != 'ok':
        return None
    samples = value.get('raw_samples_ms')
    if not isinstance(samples, list) or not samples:
        fail('Successful measurement has no raw timing samples')
    for sample in samples:
        if number(sample) <= 0:
            fail('Timing samples must be positive')
    return float(statistics.median(samples))


def equal_number(actual, expected):
    if expected is None:
        return actual is None
    return (not isinstance(actual, bool) and isinstance(actual, (int, float))
            and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12))


def public_measurement(value, *, fixture_only):
    if value is None:
        return None
    if value.get('backend') != ('fixture' if fixture_only else 'cuda'):
        fail('Measurement backend is incompatible with export mode')
    if value.get('status') not in MEASUREMENT_STATUSES or type(value.get('correct')) is not bool:
        fail('Unknown measurement status')
    if value['correct'] != (value['status'] == 'ok'):
        fail('Measurement status and correctness disagree')
    samples = value.get('raw_samples_ms')
    if not isinstance(samples, list):
        fail('Timing samples must be a list')
    for sample in samples:
        if number(sample) <= 0:
            fail('Timing samples must be positive')
    median = measurement_median(value)
    if value['correct']:
        if len(samples)!=7 or value.get('latency_ms') not in samples:
            fail('Correct query requires seven fresh samples and a sampled observation')
        index=value.get('evidence',{}).get('sampled_index')
        if type(index) is not int or not 0<=index<len(samples) or value['latency_ms']!=samples[index]:
            fail('Sampled observation binding differs')
    elif value.get('latency_ms') is not None:
        fail('Failed query cannot carry a successful observation')
    elapsed = number(value['elapsed_seconds'], nonnegative=True)
    allocated = value.get('peak_allocated_bytes')
    if allocated is not None:
        count(allocated)
    compile_seconds = value.get('evidence', {}).get('compile_and_first_call_seconds')
    if compile_seconds is not None:
        number(compile_seconds, nonnegative=True)
    return dict(status=value['status'], correct=value['correct'], raw_samples_ms=samples,
                median_ms=median, elapsed_seconds=elapsed, peak_allocated_bytes=allocated,
                compile_and_first_call_seconds=compile_seconds)


def public_schedule(value):
    if value is None:
        return None
    if set(value) != set(SCHEDULE_FIELDS) or type(value['fused']) is not bool:
        fail('Unexpected selected schedule fields')
    if value['reduction'] not in ('single', 'chunked'):
        fail('Unknown reduction strategy')
    for key in SCHEDULE_FIELDS:
        if key not in ('fused', 'reduction'):
            count(value[key])
    return {key: value[key] for key in SCHEDULE_FIELDS}


def public_runtime(value, *, fixture_only):
    if not isinstance(value, dict):
        fail('Runtime identity must be an object')
    if fixture_only:
        return {'gpu_name': 'synthetic software fixture', 'measured_gpu': False}
    name=value.get('gpu_name')
    if not isinstance(name,str) or not re.fullmatch(r'(?:NVIDIA |Tesla |GeForce )[A-Za-z0-9 .()+_-]{1,80}',name):
        fail('Unexpected GPU model label; review before public export')
    result={'gpu_name':name, 'measured_gpu':True}
    for key in ('torch','triton','cuda','python'):
        version=value.get(key)
        if not isinstance(version,str) or not re.fullmatch(r'[0-9][0-9A-Za-z.+_-]{0,63}',version):
            fail('Unexpected runtime version label')
        result[key]=version
    capability=value.get('compute_capability')
    if not isinstance(capability,list) or len(capability)!=2 or any(type(x) is not int or x<0 for x in capability):
        fail('Invalid device capability')
    result.update(compute_capability=capability,total_memory_bytes=count(value['total_memory']))
    return result


def verify_runtime_identity(identity, source_hashes, *, fixture_only):
    if fixture_only:
        return
    runtime=identity.get('runtime')
    if not isinstance(runtime,dict) or not isinstance(runtime.get('gpu_uuid'),str) or not runtime['gpu_uuid'].strip():
        fail('Production measurement lacks its physical GPU identity')
    expected=hashlib.sha256(json.dumps(runtime,sort_keys=True).encode()).hexdigest()
    if identity.get('cohort_id')!=expected:
        fail('Runtime cohort digest differs from the measured runtime')
    recorded_sources=runtime.get('source_hashes',{})
    if any(recorded_sources.get(name)!=sha for name,sha in source_hashes.items()):
        fail('Measured runtime scientific code differs from the frozen source')


def verify_measurement_bindings(inputs, episode_id, row, workload, seen_trial_ids):
    """Cross-check independent evaluator journals and distinct fresh queries."""
    expected_key=canonical_hash(workload)
    if row.get('workload_key')!=expected_key:
        fail('Episode workload key differs from the fixed task')
    journal_path='episodes/'+episode_id+'/evaluator/queries.jsonl'
    if not (inputs.root/journal_path).exists():
        if any(row.get(key) is not None for key in ('baseline','paired_final_reference','confirmation')):
            fail('Recorded measurements require the evaluator query journal')
        return
    events=inputs.read(journal_path,json_lines=True)
    started={};finished={}
    for event in events:
        if event.get('event')=='query_started':
            trial=identifier(event['trial_id'])
            if trial in started or trial in seen_trial_ids:
                fail('Repeated query identity cannot establish fresh measurements')
            if event.get('workload')!=workload or event.get('repetitions')!=7:
                fail('Evaluator query differs from the fixed workload or sample count')
            started[trial]=event;seen_trial_ids.add(trial)
        elif event.get('event')=='query_finished':
            value=event['measurement'];trial=identifier(value['evidence']['trial_id'])
            if trial not in started or trial in finished:
                fail('Unmatched or duplicate evaluator query completion')
            finished[trial]=value
    represented=[]
    for key in ('baseline','paired_final_reference','confirmation'):
        value=row.get(key)
        if value is None:continue
        trial=identifier(value.get('evidence',{}).get('trial_id'))
        if trial in represented or finished.get(trial)!=value:
            fail('Episode measurement is duplicated or differs from the evaluator journal')
        represented.append(trial)
        if value.get('workload_key')!=expected_key:
            fail('Measurement workload key differs')
        reference=(key=='paired_final_reference' or (key=='confirmation' and row['policy']==POLICIES[0]))
        schedule=None if key=='baseline' or reference else row.get('selected_schedule')
        if value.get('schedule')!=schedule or started[trial].get('schedule')!=schedule:
            fail('Confirmation schedule differs from selected schedule')
        role=row.get('selected_reference') if reference else None
        if started[trial].get('reference_role')!=role:
            fail('Evaluator reference role differs from the frozen selection')
        if value.get('correct'):
            if reference and value['evidence'].get('fixed_reference_confirmation')!=role:
                fail('Paired reference role binding differs')
            if key=='baseline' and value['evidence'].get('selected_reference')!=row.get('selected_reference'):
                fail('Initial reference selection differs')
            hashes=value['evidence'].get('input_sha256')
            if not isinstance(hashes,list) or not hashes:
                fail('Fresh measurement lacks input hashes')
            for item in hashes:digest_string(item)


def validate_and_sanitize(inputs, source_manifest, frozen_source, *, fixture_only):
    protocol = inputs.read('protocol.json')
    summary = inputs.read('summary.json')
    completion = inputs.read('completed.json')
    rows = inputs.read('episodes.jsonl', json_lines=True)
    manifest = inputs.read('source-manifest.json', external=source_manifest)
    if protocol.get('study_id') != 'sequential-schedules-live-v1' or protocol.get('schema_version') != 1:
        fail('Unknown study protocol')
    if fixture_only != (protocol.get('fixture_only') is True):
        fail('Synthetic fixtures require explicit fixture-only mode; real evidence cannot use that mode')
    if protocol.get('policies') != list(POLICIES) or len(protocol.get('seeds', [])) != 3:
        fail('Expected three policies and three nested seeds')
    seeds = protocol['seeds']
    if len(set(seeds)) != 3 or any(type(seed) is not int or seed < 0 for seed in seeds):
        fail('Expected three unique nonnegative seeds')
    if protocol.get('episode_seconds') != 90 or len(protocol.get('tasks', [])) != 6:
        fail('Expected six task families and fixed 90-second episode ceilings')
    tasks = {}
    families = set()
    for task in protocol['tasks']:
        key = identifier(task['task_id'])
        workload = task['metadata']['workload']
        if key in tasks or task['metadata'].get('split') != 'test':
            fail('Duplicate or non-test task')
        if workload['operator'] not in ('geglu', 'groupnorm_silu') or workload['dtype'] != 'float16':
            fail('Unexpected task operator or dtype')
        shape = workload['shape']
        if not isinstance(shape, list) or not shape or any(type(x) is not int or x <= 0 for x in shape):
            fail('Invalid workload shape')
        families.add((workload['operator'], tuple(shape)))
        tasks[key] = dict(task_id=key, operator=workload['operator'], shape=shape,
                          dtype=workload['dtype'], num_groups=count(workload['num_groups']),
                          eps=number(workload['eps'], nonnegative=True), input_seed=count(workload['seed']))
    if len(families) != 6 or sum(t['operator'] == 'geglu' for t in tasks.values()) != 3:
        fail('Expected six distinct shape families, three per operator')
    expected = protocol.get('episodes', [])
    if len(expected) != 54 or len(rows) != 54:
        fail('Complete study must retain all 54 attempts')
    design = {(task, seed, policy) for task in tasks for seed in seeds for policy in POLICIES}
    actual_design = {(r['task_id'], r['seed'], r['policy']) for r in expected}
    if actual_design != design or len({r['episode_id'] for r in expected}) != 54:
        fail('Incomplete or duplicate factorial design')
    sha = canonical_hash(protocol)
    if (completion.get('status') != 'complete' or completion.get('schema_version') != 1
        or completion.get('failure') is not None or completion.get('expected_episodes') != 54
        or completion.get('attempted_episodes') != 54 or completion.get('all_attempts_preserved') is not True
        or completion.get('gpu_validation_claim') is not False
        or completion.get('protocol_sha256') != sha or completion.get('summary_sha256') != canonical_hash(summary)):
        fail('Invalid completion hashes or accounting')
    if (summary.get('expected_episodes') != 54 or summary.get('attempted_episodes') != 54
        or summary.get('missing_episode_ids') != [] or summary.get('failure') is not None
        or summary.get('protocol_sha256') != sha or summary.get('all_attempts_preserved') is not True):
        fail('Invalid source summary accounting')
    source_hashes = protocol.get('source_sha256', {})
    if set(source_hashes) != set(SOURCE_MODULES):
        fail('Expected exact scientific code manifest')
    manifest_files = {r['path']: r for r in manifest['files']}
    if len(manifest_files) != len(manifest['files']):
        fail('Duplicate source manifest paths')
    frozen_source=Path(frozen_source)
    if frozen_source.is_symlink() or not frozen_source.is_dir():
        fail('Explicit frozen source directory required')
    frozen_source=frozen_source.resolve()
    for name, source_sha in source_hashes.items():
        digest_string(source_sha)
        entry = manifest_files.get('extensions/kernel_tuning/drivebench_tuning/'+name)
        if entry is None or entry['sha256'] != source_sha:
            fail('Scientific code hash differs from frozen source manifest')
        source_file=frozen_source/'extensions/kernel_tuning/drivebench_tuning'/name
        if not source_file.resolve().is_relative_to(frozen_source):
            fail('Frozen source file escapes payload')
        inputs.read_raw('scientific-code/'+name,external=source_file)
        if inputs.files['scientific-code/'+name][1]!=source_sha:
            fail('Actual frozen scientific code differs from the protocol')
    protocol_entry = manifest_files.get('runtime-inputs/study-protocol.json')
    if protocol_entry is None or protocol_entry['sha256'] != inputs.files['protocol.json'][1]:
        fail('Recovered protocol differs from the prospectively frozen bytes')
    inputs.read_raw('frozen-protocol.json',external=frozen_source/'runtime-inputs/study-protocol.json')
    if inputs.files['frozen-protocol.json'][1]!=inputs.files['protocol.json'][1]:
        fail('Recovered protocol differs from the actual frozen payload')
    public = []
    cohort_ids = set()
    runtime_hashes = set()
    runtime_public = None
    seen_trial_ids=set()
    for row, spec in zip(rows, expected):
        episode_id = identifier(spec['episode_id'])
        if episode_id != f'{len(public):03d}':
            fail('Expected canonical sequential episode identities')
        if any(row.get(key) != value for key, value in spec.items()):
            fail('Episode identity or order differs from protocol')
        retained = inputs.read('episodes/'+episode_id+'/result.json')
        if retained != row:
            fail('Per-episode evidence differs from append-only study record')
        if row.get('status') not in STATUSES or row.get('stop_reason') not in STOP_REASONS | {None}:
            fail('Unknown episode status or stop reason')
        workload=next(t['metadata']['workload'] for t in protocol['tasks'] if t['task_id']==row['task_id'])
        verify_measurement_bindings(inputs,episode_id,row,workload,seen_trial_ids)
        measurements = {key: public_measurement(row.get(key), fixture_only=fixture_only)
                        for key in ('baseline', 'paired_final_reference', 'confirmation')}
        originals = [row.get(key) for key in measurements if row.get(key) is not None]
        identities = [m.get('evidence', {}) for m in originals if m.get('correct')]
        if identities:
            for identity in identities:
                verify_runtime_identity(identity,source_hashes,fixture_only=fixture_only)
                cohort = identity.get('cohort_id')
                if not isinstance(cohort, str) or not cohort:
                    fail('Correct measurement lacks a runtime cohort binding')
                cohort_ids.add(cohort)
                runtime_hashes.add(canonical_hash(identity.get('runtime')))
                runtime_public=public_runtime(identity.get('runtime'),fixture_only=fixture_only)
                if (identity.get('runtime') != identities[0].get('runtime')
                    or identity.get('input_sha256') != identities[0].get('input_sha256')):
                    fail('Within-episode runtime or input identity changed')
        initial = measurements['baseline']['median_ms'] if measurements['baseline'] else None
        paired = measurements['paired_final_reference']['median_ms'] if measurements['paired_final_reference'] else None
        confirmed = measurements['confirmation']['median_ms'] if measurements['confirmation'] else None
        elapsed = number(row['elapsed_seconds'], nonnegative=True)
        success = (row['status'] == 'finished' and all(v is not None for v in (initial, paired, confirmed))
                   and elapsed < 90 and (row['policy'] == POLICIES[0] or row.get('environment_completed') is True))
        ratio = confirmed / paired if success else None
        score = max(-1., min(1., 1-ratio)) if success else -1.
        if row.get('successful') is not success or not equal_number(row.get('confirmed_latency_ratio'), ratio):
            fail('Recorded episode outcome differs from fresh-reference accounting')
        if not equal_number(row.get('failure_adjusted_speed_score'), score):
            fail('Recorded primary score differs from recomputation')
        for key, value in [('baseline_median_ms',initial), ('paired_final_reference_median_ms',paired), ('confirmed_median_ms',confirmed)]:
            if not equal_number(row.get(key), value):
                fail('Recorded measurement median differs from raw samples')
        selected_reference = row.get('selected_reference')
        if selected_reference not in (None, 'eager', 'compiled'):
            fail('Unknown reference role')
        env_return = row.get('environment_return')
        if env_return is not None:
            number(env_return)
        result = dict(episode_id=episode_id, task_id=row['task_id'], policy=row['policy'], seed=row['seed'],
            order_position=count(row['order_position']), status=row['status'], stop_reason=row.get('stop_reason'),
            successful=success, episode_budget_seconds=90, elapsed_seconds=elapsed,
            primary_score=score, confirmed_latency_ratio=ratio,
            confirmed_speedup=(1/ratio if success else None), environment_return=env_return,
            selected_reference=selected_reference, selected_schedule=public_schedule(row.get('selected_schedule')),
            query_safety_cap_hit=row.get('stop_reason') == 'query_safety_cap',
            exploration_queries=count(row['exploration_queries']), failed_queries=count(row['failed_queries']),
            invalid_actions=count(row['invalid_actions']), restore_actions=count(row['restore_actions']),
            measurements=measurements)
        public.append(result)
    if len(cohort_ids)>1 or len(runtime_hashes)>1:
        fail('Physical runtime cohorts cannot be pooled')
    if not cohort_ids:
        fail('No verified runtime cohort; preserve failed attempts without a complete measured-study aggregate')
    if completion.get('all_episodes_successful') is not all(r['successful'] for r in public):
        fail('Completion success flag differs from episode outcomes')
    return protocol, tasks, public, runtime_public, len(cohort_ids)


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered)-1)*fraction
    low = int(position)
    high = min(low+1, len(ordered)-1)
    return ordered[low]+(ordered[high]-ordered[low])*(position-low)


def aggregate(rows, tasks, seeds, *, replicates, bootstrap_seed):
    if type(replicates) is not int or not 100 <= replicates <= 100000:
        fail('Bootstrap replicates must be 100..100000')
    if type(bootstrap_seed) is not int or bootstrap_seed < 0:
        fail('Bootstrap seed must be a nonnegative integer')
    lookup = {(r['task_id'], r['seed'], r['policy']): r for r in rows}
    task_ids = sorted(tasks)
    distributions = {p: [] for p in POLICIES}
    rng = random.Random(bootstrap_seed)
    for _ in range(replicates):
        indices = [(task, [rng.choice(seeds) for _ in seeds])
                   for task in [rng.choice(task_ids) for _ in task_ids]]
        for policy in POLICIES:
            # The same task and nested seed draws are used for every policy.
            distributions[policy].append(statistics.mean(
                statistics.mean(lookup[(task, seed, policy)]['primary_score'] for seed in selected)
                for task, selected in indices))
    policies = {}
    task_rows = []
    for policy in POLICIES:
        selected = [r for r in rows if r['policy'] == policy]
        by_task = [[lookup[(task, seed, policy)] for seed in seeds] for task in task_ids]
        score = statistics.mean(statistics.mean(r['primary_score'] for r in group) for group in by_task)
        successful = [r for r in selected if r['successful']]
        policies[policy] = dict(attempts=len(selected), confirmed=len(successful),
            failed_or_unconfirmed=len(selected)-len(successful), success_ratio=len(successful)/len(selected),
            mean_primary_score=score, primary_score_ci95=[percentile(distributions[policy], .025), percentile(distributions[policy], .975)],
            total_episode_seconds=sum(r['elapsed_seconds'] for r in selected),
            median_episode_seconds=statistics.median(r['elapsed_seconds'] for r in selected),
            min_episode_seconds=min(r['elapsed_seconds'] for r in selected),
            max_episode_seconds=max(r['elapsed_seconds'] for r in selected),
            wall_time_ceiling_overruns=sum(r['elapsed_seconds'] >= 90 for r in selected),
            query_safety_cap_hits=sum(r['query_safety_cap_hit'] for r in selected),
            exploration_queries=sum(r['exploration_queries'] for r in selected),
            failed_queries=sum(r['failed_queries'] for r in selected),
            invalid_actions=sum(r['invalid_actions'] for r in selected),
            median_confirmed_latency_ratio_successes_only=(statistics.median(r['confirmed_latency_ratio'] for r in successful) if successful else None))
        for task, group in zip(task_ids, by_task):
            task_rows.append(dict(task_id=task, policy=policy, attempts=len(group),
                confirmed=sum(r['successful'] for r in group),
                mean_primary_score=statistics.mean(r['primary_score'] for r in group),
                total_episode_seconds=sum(r['elapsed_seconds'] for r in group)))
    comparisons = []
    for first, second in itertools.combinations(POLICIES, 2):
        differences = [a-b for a,b in zip(distributions[first], distributions[second])]
        comparisons.append(dict(first=first, second=second,
            mean_score_difference=policies[first]['mean_primary_score']-policies[second]['mean_primary_score'],
            paired_difference_ci95=[percentile(differences,.025),percentile(differences,.975)]))
    return dict(policies=policies, task_policy_means=task_rows, paired_comparisons=comparisons,
        total_episode_seconds=sum(r['elapsed_seconds'] for r in rows),
        uncertainty=dict(method='paired hierarchical percentile bootstrap', confidence_level=.95,
            task_family_clusters=6, seeds_per_family=3, bootstrap_replicates=replicates,
            bootstrap_seed=bootstrap_seed, paired_policy_resampling=True,
            interpretation='Exploratory intervals from six observed task families and three nested seeds; not 54 independent observations or evidence of unseen-operator generalization.'))


def write_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def write_csv(path, rows, fields):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def export(study, output, source_manifest, frozen_source, analysis_plan, *, fixture_only=False, replicates=10000, bootstrap_seed=20261125):
    inputs = Inputs(study)
    output = Path(output)
    if output.exists() or output.is_symlink() or output.resolve().is_relative_to(inputs.root):
        fail('Export must be a new directory outside source evidence')
    protocol, tasks, episodes, runtime, cohorts = validate_and_sanitize(inputs, source_manifest, frozen_source, fixture_only=fixture_only)
    plan=inputs.read('analysis-plan.json',external=analysis_plan)
    if (plan.get('study_id')!='sequential-schedules-live-v1' or plan.get('schema_version')!=1
            or plan.get('protocol_sha256')!=canonical_hash(protocol)
            or plan.get('registered_before_gpu_measurement') is not True):
        fail('Analysis plan does not bind this prospective study')
    if not fixture_only:
        if inputs.files['analysis-plan.json'][1]!=REGISTERED_ANALYSIS_PLAN_SHA256:
            fail('Analysis plan differs from the registered bytes')
        if replicates!=10000 or bootstrap_seed!=20261125:
            fail('Measured exports must use the registered 10,000 draws and analysis seed')
    totals = aggregate(episodes, tasks, protocol['seeds'], replicates=replicates, bootstrap_seed=bootstrap_seed)
    evidence_kind = 'synthetic-software-test-fixture' if fixture_only else 'measured-cuda-study'
    public = dict(schema_version=1, study_id='sequential-schedules-live-v1', evidence_kind=evidence_kind,
        scientific_results=not fixture_only, exploratory=True, trained_rl_policy=False, oracle_measured=False,
        whole_operator_generalization_claim=False, whole_model_speedup_claim=False,
        cuda_validation_gate_verified_by_this_export=False,
        primary_score_definition=SCORE_DEFINITION,
        wall_time_definition='Actual elapsed time per episode includes reference setup, compilation, policy work, failures, confirmation and teardown; sums exclude gaps between episodes and outer host setup/validation.',
        scope=dict(task_families=6, operators=2, seeds=protocol['seeds'], policies=list(POLICIES),
                   attempts=54, episode_ceiling_seconds=90, verified_hardware_runtime_cohorts=cohorts),
        runtime=runtime,
        tasks=list(tasks.values()), **totals)
    provenance = dict(schema_version=1, evidence_kind=evidence_kind, scientific_results=not fixture_only,
        consumed_files=inputs.manifest(), protocol_content_sha256=canonical_hash(protocol),
        registered_analysis_plan_sha256=inputs.files['analysis-plan.json'][1],
        scientific_code_sha256_verified_against_frozen_payload=protocol['source_sha256'],
        analysis_code={'filename':Path(__file__).name,'sha256':file_hash(__file__)},
        source_evidence_modified=False,
        redaction='Explicit allowlist; excludes source paths, provider identifiers, physical GPU UUIDs, raw exception messages and stderr. Original evidence is unchanged.')
    inputs.verify_unchanged()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'summary.json', public)
    write_json(output/'episodes.json', dict(evidence_kind=evidence_kind, scientific_results=not fixture_only, episodes=episodes))
    write_json(output/'provenance.json', provenance)
    csv_rows=[]
    for row in episodes:
        record={k:v for k,v in row.items() if k not in ('measurements','selected_schedule')}
        record['evidence_kind']=evidence_kind
        for key,value in row['measurements'].items():
            record[key+'_median_ms']=value['median_ms'] if value else None
        csv_rows.append(record)
    fields=['evidence_kind','episode_id','task_id','policy','seed','order_position','status','stop_reason',
        'successful','primary_score','confirmed_latency_ratio','confirmed_speedup','episode_budget_seconds',
        'elapsed_seconds','query_safety_cap_hit','exploration_queries','failed_queries','invalid_actions',
        'restore_actions','selected_reference','baseline_median_ms','paired_final_reference_median_ms',
        'confirmation_median_ms','environment_return']
    write_csv(output/'episodes.csv',csv_rows,fields)
    write_csv(output/'task-policy-means.csv',[dict(evidence_kind=evidence_kind,**r) for r in totals['task_policy_means']],
              ['evidence_kind','task_id','policy','attempts','confirmed','mean_primary_score','total_episode_seconds'])
    title='Synthetic software test — not scientific results' if fixture_only else 'Sequential schedule comparison'
    lines=['# '+title,'',
        'A small exploratory comparison of a fixed reference, random schedule edits, and greedy local search on six task families, with three seeds per family.', '',
        '| Policy | Primary score (95% interval) | Confirmed | Actual episode time | Query-cap hits |',
        '|---|---:|---:|---:|---:|']
    for policy, row in public['policies'].items():
        ci=row['primary_score_ci95']
        lines.append(f"| {policy} | {row['mean_primary_score']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}] | {row['confirmed']}/{row['attempts']} | {row['total_episode_seconds']:.1f} s | {row['query_safety_cap_hits']} |")
    lines += ['',SCORE_DEFINITION,'',
        'Intervals resample six task families, then the three seeds within each family, using identical draws for all policies. This is a small exploratory study. It does not evaluate a trained RL policy, measure an oracle, or establish transfer to unseen operators.', '',
        public['wall_time_definition'],'',
        'All 54 attempts, including failed and unconfirmed episodes, appear in the JSON and CSV exports. Successful-only latency ratios are secondary; the primary score includes failures.', '']
    (output/'SUMMARY.md').write_text('\n'.join(lines))
    inputs.verify_unchanged()
    write_json(output/'export-manifest.json',dict(schema_version=1,evidence_kind=evidence_kind,
        files=[dict(file=p.name,bytes=p.stat().st_size,sha256=file_hash(p)) for p in sorted(output.iterdir()) if p.is_file()]))
    return public


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-manifest',type=Path,required=True)
    parser.add_argument('--frozen-source',type=Path,required=True)
    parser.add_argument('--analysis-plan',type=Path,required=True)
    parser.add_argument('--bootstrap-replicates',type=int,default=10000)
    parser.add_argument('--bootstrap-seed',type=int,default=20261125)
    parser.add_argument('--fixture-only',action='store_true')
    args=parser.parse_args(argv)
    result=export(args.study,args.output,args.source_manifest,args.frozen_source,args.analysis_plan,fixture_only=args.fixture_only,
                  replicates=args.bootstrap_replicates,bootstrap_seed=args.bootstrap_seed)
    print(json.dumps(dict(evidence_kind=result['evidence_kind'],attempts=result['scope']['attempts'],
                         scientific_results=result['scientific_results'],source_evidence_modified=False)))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
