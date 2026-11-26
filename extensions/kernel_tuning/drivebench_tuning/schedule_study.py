"""Prospective, wall-clock-bounded comparison of live schedule search baselines.

This runner neither trains an RL policy nor reads the old timing lookup table.
All selected schedules are confirmed by a new measurement. The reference-only
comparator is reported as a comparator, not as an environment return.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import signal
import statistics
import time
import uuid

from .contract import Workload
from .schedule import ACTIONS, BLOCKS, WARPS, KernelSchedule, action_mask, check_schedule, transform
from .suite import load_suite

POLICIES = ('compiled_eager_no_edit', 'random_walk', 'greedy_local')
SEEDS = (7101, 7102, 7103)
EPISODE_SECONDS = 90.0
SETUP_TIMEOUT = 45.0
QUERY_TIMEOUT = 15.0
FINAL_RESERVE = 20.0
MAX_QUERIES = 128
MAX_SECONDS = 5700.0
SOURCE_FILES = ('schedule_study.py', 'schedule.py', 'schedule_env.py', 'schedule_kernels.py',
                '_schedule_triton.py', 'schedule_measurement.py', 'benchmark.py',
                'contract.py', 'kernels.py', '_triton_kernels.py')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def append(path, value):
    import os
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def protocol():
    suite = load_suite('operator-generalization-v2')
    tasks = [task.to_dict() for i, task in enumerate(suite.tasks)
             if i in suite.splits['test'] and task.metadata['workload']['dtype'] == 'float16']
    if len(tasks) != 6 or {t['metadata']['workload']['operator'] for t in tasks} != {'geglu', 'groupnorm_silu'}:
        raise ValueError('Expected six fixed float16 TEST shape-family tasks')
    episodes = []
    for seed_index, seed in enumerate(SEEDS):
        for task_index, task in enumerate(tasks):
            offset = (seed_index + task_index) % len(POLICIES)
            order = POLICIES[offset:] + POLICIES[:offset]
            for position, policy in enumerate(order):
                episodes.append({'episode_id': f'{len(episodes):03d}', 'task_id': task['task_id'],
                                 'seed': seed, 'policy': policy, 'order_position': position})
    return {'schema_version': 1, 'study_id': 'sequential-schedules-live-v2',
        'parent_protocol_sha256': '428153e29b8644c12d26c578bd8b8f06dda6e1280a476fb09411168a4929e803',
        'scope': 'small live baseline comparison; no trained RL or generalization claim',
        'suite_id': 'operator-generalization-v2', 'suite_sha256': suite.sha256,
        'tasks': tasks, 'seeds': list(SEEDS), 'policies': list(POLICIES), 'episodes': episodes,
        'episode_seconds': EPISODE_SECONDS, 'setup_timeout_seconds': SETUP_TIMEOUT,
        'exploratory_query_timeout_seconds': QUERY_TIMEOUT, 'final_reserve_seconds': FINAL_RESERVE,
        'exploration_admission': 'require more than the full 15-second query window plus 20-second confirmation reserve before and after navigation; each admitted query receives 15 seconds',
        'max_exploration_queries': MAX_QUERIES, 'repetitions': 7, 'warmups': 2,
        'budget_definition': 'equal 90-second ceilings including reference setup, compilation, edits, failures, fresh confirmation and worker teardown; actual time reported; reference may finish early',
        'cache_policy': 'fresh private TorchInductor and Triton filesystem caches per episode, retained within episode; bulk caches are scratch, not scientific evidence',
        'selection': 'minimum observed sampled latency among correct explored schedules; restore by charged legal transformations; fresh fixed-reference then selected-schedule confirmation within final reserve; final observations never change selection',
        'greedy': 'seed-shuffled legal one-edit neighbors of current best; failed/nonimproving candidates return by legal edits; repeat incumbent if all neighbors visited',
        'random': 'one uniformly sampled legal edit from current schedule between each new measurement; best observed schedule restored before final submit',
        'reference': 'lower-median correct eager/compiled role chosen during reset; freshly remeasure that fixed role; no environment-return claim for comparator',
        'reporting': 'all attempts and failures included; failure-adjusted speed score assigns -1 to unconfirmed episodes, otherwise clip(1-confirmed_median/paired_final_reference_median,-1,1); environment return uses its original initial-reference normalization and is separate; no oracle/regret',
        'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in SOURCE_FILES}}


def validate_protocol(document):
    if document != protocol():
        raise ValueError('Protocol does not exactly match the current frozen study design and source')
    return digest(document)


def navigation_actions(workload, current, target):
    """Construct a legal edit path; never teleport to a previously seen state."""
    check_schedule(workload, current); check_schedule(workload, target)
    state, path = current, []
    def apply(name):
        nonlocal state
        state = transform(workload, state, name)
        path.append(ACTIONS.index(name))
    def tune(field, prefix, choices, desired):
        while getattr(state, field) != desired:
            suffix = 'up' if choices.index(getattr(state, field)) < choices.index(desired) else 'down'
            apply(prefix + '_' + suffix)
    if state == target:
        return path
    # Fusing canonicalizes only the inactive activation fields. Restoration
    # rebuilds them through explicit unfuse and launch edits when needed.
    if not state.fused:
        apply('fuse')
    tune('block_size', 'block', BLOCKS, target.block_size)
    tune('num_warps', 'warps', WARPS, target.num_warps)
    if workload.operator == 'groupnorm_silu':
        if state.reduction != target.reduction:
            apply('reduction_' + target.reduction)
        if target.reduction == 'chunked':
            tune('reduction_block_size', 'reduction_block', BLOCKS, target.reduction_block_size)
        tune('reduction_num_warps', 'reduction_warps', WARPS, target.reduction_num_warps)
    if not target.fused:
        apply('unfuse')
        tune('activation_block_size', 'activation_block', BLOCKS, target.activation_block_size)
        tune('activation_num_warps', 'activation_warps', WARPS, target.activation_num_warps)
    if state != target:
        raise RuntimeError('Navigation did not restore the selected schedule')
    return path


def legal_edits(workload, schedule):
    return [i for i, allowed in enumerate(action_mask(workload, schedule))
            if allowed and ACTIONS[i] not in ('measure', 'submit')]


def _safe_median(measurement):
    if not measurement or not measurement.get('correct') or measurement.get('status') != 'ok':
        return None
    samples = measurement.get('raw_samples_ms')
    if not samples or any(not math.isfinite(x) or x <= 0 for x in samples):
        return None
    return float(statistics.median(samples))


def _run_episode(spec, task, directory, cache_directory, *, env_factory=None, evaluator_factory=None):
    from .schedule_env import ScheduleTuningEnv
    from .schedule_measurement import CudaScheduleEvaluator
    env_factory = env_factory or ScheduleTuningEnv
    evaluator_factory = evaluator_factory or CudaScheduleEvaluator
    directory.mkdir(parents=True, exist_ok=False)
    workload = Workload(**task['metadata']['workload'])
    started = time.monotonic()
    row = {**spec, 'workload_key': workload.key(), 'status': 'started', 'successful': False,
           'exploration_queries': 0, 'failed_queries': 0, 'invalid_actions': 0,
           'restore_actions': 0, 'environment_return': None, 'stop_reason': None,
           'episode_budget_seconds': EPISODE_SECONDS, 'baseline': None, 'confirmation': None,
           'paired_final_reference': None,
           'selected_schedule': None, 'selected_reference': None,
           'cache_directory': str(cache_directory), 'evidence_directory': str(directory)}
    write_new(directory / 'started.json', row)
    env = evaluator = None
    final_info = None
    total_reward = 0.0
    def remaining_seconds(info):
        # Recompute after policy/navigation work; info is a previous observation.
        return max(0., min(EPISODE_SECONDS - info['elapsed_seconds'],
                           EPISODE_SECONDS - (time.monotonic() - started)))
    try:
        evaluator = evaluator_factory(evidence_directory=directory / 'evaluator',
            repetitions=7, warmups=2, seed=spec['seed'], cache_directory=cache_directory)
        env = env_factory(workloads=[workload], budget_seconds=EPISODE_SECONDS,
            trial_timeout_seconds=SETUP_TIMEOUT, evidence_dir=directory / 'environment', evaluator=evaluator)
        _, info = env.reset(seed=spec['seed'])
        row['baseline'] = info['baseline']
        row['selected_reference'] = info['baseline']['evidence']['selected_reference']
        reference_median = _safe_median(info['baseline'])
        if reference_median is None:
            raise ValueError('Baseline has no valid fresh samples')
        if spec['policy'] == 'compiled_eager_no_edit':
            remaining = remaining_seconds(info)
            if remaining <= 0:
                raise TimeoutError('Reference setup exhausted its episode')
            reference = evaluator.measure_reference(workload, row['selected_reference'],
                                                     timeout_seconds=min(FINAL_RESERVE / 2, remaining))
            row['paired_final_reference'] = asdict(reference)
            if not reference.correct:
                raise RuntimeError('Paired final reference failed')
            remaining = remaining_seconds(info)
            confirmed = evaluator.measure_reference(workload, row['selected_reference'],
                                                     timeout_seconds=min(FINAL_RESERVE, remaining))
            row['confirmation'] = asdict(confirmed)
            row['stop_reason'] = 'fixed_reference_confirmed' if confirmed.correct else 'reference_confirmation_failed'
        else:
            rng = random.Random(spec['seed'])
            best, best_latency = env.schedule, math.inf
            visited = set()
            ended = False
            while row['exploration_queries'] < MAX_QUERIES:
                remaining = remaining_seconds(info)
                # A shrinking timeout can kill the evaluator while leaving the
                # final reserve unusable. Admit only a complete query window.
                if remaining <= FINAL_RESERVE + QUERY_TIMEOUT:
                    row['stop_reason'] = 'confirmation_reserve'
                    break
                if row['exploration_queries']:
                    if spec['policy'] == 'random_walk':
                        edits = [rng.choice(legal_edits(workload, env.schedule))]
                    else:
                        edits = navigation_actions(workload, env.schedule, best)
                        choices = legal_edits(workload, best)
                        rng.shuffle(choices)
                        unseen = [a for a in choices if (digest(asdict(best)), a) not in visited]
                        if unseen:
                            chosen = unseen[0]
                            visited.add((digest(asdict(best)), chosen))
                            edits.append(chosen)
                    for action in edits:
                        _, reward, terminated, truncated, info = env.step(action)
                        total_reward += reward
                        row['invalid_actions'] += int(info['outcome'] == 'invalid_action')
                        if terminated or truncated:
                            ended = True
                            row['stop_reason'] = 'ended_during_edit'
                            break
                    if ended:
                        break
                # Recompute time after navigation; edits and policy work cost time.
                remaining = remaining_seconds(info)
                if remaining <= FINAL_RESERVE + QUERY_TIMEOUT:
                    row['stop_reason'] = 'confirmation_reserve'
                    break
                env.trial_timeout_seconds = QUERY_TIMEOUT
                _, reward, terminated, truncated, info = env.step(ACTIONS.index('measure'))
                total_reward += reward
                row['exploration_queries'] += 1
                measured = info['measurement']
                if measured and measured['correct']:
                    if measured['latency_ms'] < best_latency:
                        best, best_latency = env.schedule, measured['latency_ms']
                else:
                    row['failed_queries'] += 1
                if terminated or truncated:
                    ended = True
                    row['stop_reason'] = 'worker_failure_or_budget'
                    break
            if row['exploration_queries'] >= MAX_QUERIES:
                row['stop_reason'] = 'query_safety_cap'
            row['selected_schedule'] = asdict(best)
            row['best_exploration_sample_ms'] = best_latency if math.isfinite(best_latency) else None
            if not ended:
                for action in navigation_actions(workload, env.schedule, best):
                    _, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    row['restore_actions'] += 1
                    if terminated or truncated:
                        ended = True
                        row['stop_reason'] = 'ended_restoring_selection'
                        break
            if not ended:
                remaining = remaining_seconds(info)
                if remaining <= 0:
                    row['stop_reason'] = 'no_confirmation_time'
                else:
                    reference = evaluator.measure_reference(workload, row['selected_reference'],
                        timeout_seconds=min(FINAL_RESERVE / 2, remaining))
                    row['paired_final_reference'] = asdict(reference)
                    if not reference.correct:
                        raise RuntimeError('Paired final reference failed')
                    remaining = remaining_seconds(info)
                    if remaining <= 0:
                        raise TimeoutError('Paired reference exhausted remaining confirmation time')
                    env.trial_timeout_seconds = min(FINAL_RESERVE, remaining)
                    _, reward, terminated, truncated, final_info = env.step(ACTIONS.index('submit'))
                    total_reward += reward
                    row['confirmation'] = final_info['measurement']
                    row['environment_submission_score'] = final_info['submission_score']
                    row['environment_reward_components'] = final_info['reward_components']
                    if truncated or not terminated:
                        row['stop_reason'] = 'unconfirmed_before_deadline'
                    row['environment_completed'] = terminated and not truncated
            row['environment_return'] = total_reward
        row['status'] = 'finished'
    except Exception as exc:
        row.update(status='failed', stop_reason='episode_exception',
                   error={'type': type(exc).__name__, 'message': str(exc)})
    except BaseException as exc:
        # Signals stop the campaign after this failed attempt is durably saved.
        row.update(status='interrupted', stop_reason='interrupted', interrupted=True,
                   error={'type': type(exc).__name__, 'message': str(exc)})
    finally:
        try:
            if env is not None:
                env.close()
            elif evaluator is not None:
                evaluator.close()
        except BaseException as exc:
            row.update(status='cleanup_failed', stop_reason='cleanup_failed', cleanup_failed=True,
                       cleanup_error={'type': type(exc).__name__, 'message': str(exc)})
        row['elapsed_seconds'] = time.monotonic() - started
        if spec['policy'] != 'compiled_eager_no_edit' and row['baseline'] is not None:
            row['environment_return'] = total_reward
    baseline_median, confirmed_median = _safe_median(row['baseline']), _safe_median(row['confirmation'])
    paired_reference_median = _safe_median(row['paired_final_reference'])
    row['successful'] = (row['status'] == 'finished' and baseline_median is not None
        and paired_reference_median is not None and confirmed_median is not None and row['elapsed_seconds'] < EPISODE_SECONDS
        and (spec['policy'] == 'compiled_eager_no_edit' or row.get('environment_completed', False)))
    row['baseline_median_ms'], row['confirmed_median_ms'] = baseline_median, confirmed_median
    row['paired_final_reference_median_ms'] = paired_reference_median
    row['confirmed_latency_ratio'] = confirmed_median / paired_reference_median if row['successful'] else None
    row['failure_adjusted_speed_score'] = (max(-1., min(1., 1-row['confirmed_latency_ratio']))
                                           if row['successful'] else -1.)
    write_new(directory / 'result.json', row)
    return row


def summarize(rows, document):
    expected = document['episodes']
    by_id = {spec['episode_id']: spec for spec in expected}
    seen = {r['episode_id'] for r in rows}
    if len(seen) != len(rows):
        raise ValueError('Duplicate episode results')
    if any(r['episode_id'] not in {e['episode_id'] for e in expected} for r in rows):
        raise ValueError('Unexpected episode result')
    for row in rows:
        if any(row.get(key) != value for key, value in by_id[row['episode_id']].items()):
            raise ValueError('Episode task, seed, policy or order differs from the frozen design')
    summary = {'schema_version': 1, 'expected_episodes': len(expected), 'attempted_episodes': len(rows),
        'missing_episode_ids': [e['episode_id'] for e in expected if e['episode_id'] not in seen],
        'protocol_sha256': digest(document), 'all_attempts_preserved': True,
        'trained_rl_policy': False, 'oracle_measured': False, 'whole_model_claim': False,
        'score_definition': document['reporting'], 'policies': {}}
    for policy in POLICIES:
        group = [r for r in rows if r['policy'] == policy]
        summary['policies'][policy] = {'attempted': len(group),
            'confirmed': sum(r['successful'] for r in group),
            'failed_or_unconfirmed': sum(not r['successful'] for r in group),
            'query_safety_cap_hits': sum(r['stop_reason'] == 'query_safety_cap' for r in group),
            'mean_failure_adjusted_speed_score': statistics.mean(r['failure_adjusted_speed_score'] for r in group) if group else None,
            'median_elapsed_seconds': statistics.median(r['elapsed_seconds'] for r in group) if group else None,
            'exploration_queries': sum(r['exploration_queries'] for r in group),
            'failed_queries': sum(r['failed_queries'] for r in group),
            'invalid_actions': sum(r['invalid_actions'] for r in group),
            'episodes': [r['episode_id'] for r in group]}
    return summary


def run_study(output, cache_root, *, max_seconds=MAX_SECONDS, deadline_unix=None,
              document=None, expected_protocol_sha256=None):
    if (isinstance(max_seconds, bool) or not isinstance(max_seconds, (float, int)) or
            not math.isfinite(max_seconds) or max_seconds <= 0 or max_seconds > MAX_SECONDS):
        raise ValueError('max_seconds must be positive and no greater than 5700')
    output, cache_root = Path(output).resolve(), Path(cache_root).resolve()
    if output == cache_root or output in cache_root.parents or cache_root in output.parents:
        raise ValueError('Compiler cache scratch and scientific evidence must be separate directories')
    document = document or protocol()
    sha = validate_protocol(document)
    if expected_protocol_sha256 is not None and sha != expected_protocol_sha256:
        raise ValueError('Prospective protocol hash mismatch')
    output.mkdir(parents=True, exist_ok=False)
    cache_root.mkdir(parents=True, exist_ok=False)
    write_new(output / 'protocol.json', document)
    write_new(output / 'admission.json', {'protocol_sha256': sha, 'max_seconds': max_seconds,
        'deadline_unix': deadline_unix, 'cache_root': str(cache_root), 'cache_archive_included': False})
    deadline = time.monotonic() + max_seconds
    if deadline_unix is not None:
        if not isinstance(deadline_unix, (float, int)) or not math.isfinite(deadline_unix):
            raise ValueError('deadline_unix must be finite')
        deadline = min(deadline, time.monotonic() + deadline_unix-time.time())
    tasks = {t['task_id']: t for t in document['tasks']}
    rows, cohort, failure = [], None, None
    try:
        for spec in document['episodes']:
            if time.monotonic() + EPISODE_SECONDS + 5 > deadline:
                failure = 'whole episode and teardown do not fit the remaining study budget'
                break
            row = _run_episode(spec, tasks[spec['task_id']], output / 'episodes' / spec['episode_id'],
                               cache_root / spec['episode_id'])
            rows.append(row)
            append(output / 'episodes.jsonl', row)
            if row.get('interrupted'):
                failure = 'study interrupted; the current failed attempt is preserved'
                break
            if row.get('cleanup_failed'):
                failure = 'worker cleanup failed; no further episode admitted'
                break
            baseline = row.get('baseline')
            if baseline is not None:
                observed = baseline['evidence']['cohort_id']
                if cohort is None:
                    cohort = observed
                elif cohort != observed:
                    failure = 'GPU/runtime cohort changed; study stopped without pooling'
                    break
    except BaseException as exc:
        failure = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        summary = summarize(rows, document)
        summary.update(cohort_id=cohort, failure=failure)
        if failure == 'GPU/runtime cohort changed; study stopped without pooling':
            summary['policies'] = {}
            summary['comparison_valid'] = False
            summary['aggregation_suppressed'] = 'hardware/runtime cohorts differ; inspect individual preserved episodes'
        else:
            summary['comparison_valid'] = failure is None and len(rows) == len(document['episodes'])
        write_new(output / 'summary.json', summary)
        complete = len(rows) == len(document['episodes']) and failure is None
        write_new(output / ('completed.json' if complete else 'partial.json'), {
            'schema_version': 1, 'status': 'complete' if complete else 'partial',
            'expected_episodes': len(document['episodes']), 'attempted_episodes': len(rows),
            'all_attempts_preserved': True, 'protocol_sha256': sha, 'summary_sha256': digest(summary),
            'gpu_validation_claim': False, 'all_episodes_successful': all(r['successful'] for r in rows) and complete,
            'failure': failure})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cache-root', type=Path)
    parser.add_argument('--max-seconds', type=float, default=MAX_SECONDS)
    parser.add_argument('--deadline-unix', type=float)
    parser.add_argument('--write-protocol', type=Path)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--expected-protocol-sha256')
    args = parser.parse_args(argv)
    if args.write_protocol:
        doc = protocol()
        write_new(args.write_protocol, doc)
        print(json.dumps({'protocol_sha256': digest(doc), 'episodes': len(doc['episodes'])}))
        return 0
    if args.output is None or args.cache_root is None:
        parser.error('--output and --cache-root are required for live execution')
    if bool(args.protocol) != bool(args.expected_protocol_sha256):
        parser.error('--protocol and --expected-protocol-sha256 must be supplied together')
    document = json.loads(args.protocol.read_text()) if args.protocol else None
    previous = signal.getsignal(signal.SIGTERM)
    def terminate(signum, frame):
        raise KeyboardInterrupt('Study terminated; closing owned CUDA worker')
    signal.signal(signal.SIGTERM, terminate)
    try:
        result = run_study(args.output, args.cache_root, max_seconds=args.max_seconds,
            deadline_unix=args.deadline_unix, document=document,
            expected_protocol_sha256=args.expected_protocol_sha256)
        print(json.dumps({'attempted_episodes': result['attempted_episodes'], 'failure': result['failure']}))
        return 0 if not result['missing_episode_ids'] and not result['failure'] else 1
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    raise SystemExit(main())
