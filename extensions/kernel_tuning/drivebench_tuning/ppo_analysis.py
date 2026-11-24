"""Summaries for PPO replay and a separate native-autotuning GPU supplement.

Replay regret and fresh confirmation latency are different quantities. Native
selection samples are not an independent oracle, so this module never reports
live regret or converts a latency difference into a regret-reduction claim.
"""
from collections import defaultdict
import math
import statistics

import numpy as np


def _families(suite):
    result = {}
    for index in suite.splits['test']:
        task = suite.tasks[index]
        w = task.to_dict()['metadata']['workload']
        result[task.task_id] = (w['operator'], tuple(w['shape']), w['num_groups'], w['eps'])
    return result


def family_interval(task_values, suite, *, geometric=False, seed=2311):
    """Percentile cluster bootstrap over the six held-out shape families.

    All dtypes of a sampled family travel together. Input values already
    aggregate process or replay repeats; this conditional interval does not
    treat reused timings as new samples or estimate population GPU variation.
    """
    families = _families(suite)
    grouped = defaultdict(list)
    for task_id, value in sorted(task_values.items()):
        if task_id not in families or not math.isfinite(value) or (geometric and value <= 0):
            raise ValueError('Finite, declared held-out task values required')
        grouped[families[task_id]].append(math.log(value) if geometric else value)
    if not grouped:
        return None
    groups = [grouped[key] for key in sorted(grouped)]
    rng = np.random.default_rng(seed)
    samples = []
    for draw in rng.integers(0, len(groups), size=(10000, len(groups))):
        samples.append(statistics.mean(v for i in draw for v in groups[int(i)]))
    mean = statistics.mean(v for group in groups for v in group)
    low, high = np.quantile(samples, [.025, .975])
    transform = math.exp if geometric else float
    return {'estimate': transform(mean), 'lower_95': transform(low), 'upper_95': transform(high),
            'bootstrap_draws': 10000, 'seed': seed, 'shape_families': len(groups),
            'tasks': len(task_values), 'method': 'shape-family cluster percentile bootstrap',
            'scope': 'conditional on measured cohorts, fixed policy checkpoints and aggregated repeats'}


def replay_comparison(reports, suite):
    """Summarize every supplied checkpoint/policy; never select a winning seed."""
    expected = set(_families(suite))
    rows, names, bindings = [], set(), set()
    for report in reports:
        name = report['policy_name']
        if name in names or report.get('status') != 'completed' or report.get('split') != 'test':
            raise ValueError('Unique completed TEST policy reports required')
        names.add(name)
        if (report.get('suite_sha256') != suite.sha256 or report.get('backend') != 'measured_replay'
                or report.get('fresh_gpu_measurements') != 0 or report.get('repeats') != 20
                or report.get('seed') != 2101 or report.get('protocol', {}).get('trial_budget') != 8):
            raise ValueError('Replay protocol, seed or measurement boundary differs')
        bindings.add(report['table_sha256'])
        by_task = defaultdict(list)
        for episode in report['episodes']:
            if episode.get('status') != 'completed' or len(episode.get('steps', [])) != 8:
                raise ValueError('Complete eight-turn trajectories required')
            by_task[episode['task_id']].append(episode)
        if set(by_task) != expected or any({e['repeat'] for e in es} != set(range(20)) or len(es) != 20
                                         for es in by_task.values()):
            raise ValueError('Every TEST task requires exactly twenty replay draws')
        curve = []
        for turn in range(8):
            values = {task: statistics.mean(e['table_analysis']['curve'][turn]['table_regret'] for e in es)
                      for task, es in by_task.items()}
            curve.append({'trials': turn + 1, 'mean_table_regret': statistics.mean(values.values())})
        final = {task: statistics.mean(e['table_analysis']['curve'][-1]['table_regret'] for e in es)
                 for task, es in by_task.items()}
        rows.append({'policy': name, 'rl_training_performed': report.get('rl_training_performed', False),
                     'episodes': len(report['episodes']), 'curve': curve, 'per_task_final_regret': final,
                     'mean_final_regret': statistics.mean(final.values()),
                     'family_bootstrap': family_interval(final, suite),
                     'repeated_action_fraction': statistics.mean(
                         e['budget_analysis']['repeated_action_turns'] / 8 for e in report['episodes']),
                     'failed_or_rejected_fraction': statistics.mean(
                         e['budget_analysis']['failed_or_rejected_turns'] / 8 for e in report['episodes'])})
    if len(bindings) != 1:
        raise ValueError('Replay comparisons must use one exact measured table')
    trained = [r for r in rows if r['policy'] in {f'ppo-seed-{seed}' for seed in range(3101, 3106)}]
    if len(trained) != 5:
        raise ValueError('All five prospectively declared PPO seeds required')
    mean_tasks = {task: statistics.mean(r['per_task_final_regret'][task] for r in trained) for task in sorted(expected)}
    comparisons = []
    for baseline in rows:
        if baseline in trained:
            continue
        differences = {task: mean_tasks[task] - baseline['per_task_final_regret'][task] for task in sorted(expected)}
        comparisons.append({'baseline': baseline['policy'], 'ppo_minus_baseline_regret': family_interval(differences, suite)})
    return {'kind': 'ppo_measured_replay_supplement', 'suite_sha256': suite.sha256,
            'table_sha256': next(iter(bindings)), 'policies': rows, 'ppo_training_seeds': 5,
            'ppo_mean_across_seeds': family_interval(mean_tasks, suite), 'paired_comparisons': comparisons,
            'fresh_gpu_measurements': 0, 'selection_rule': 'all five seeds; no best-seed selection',
            'interval_limit': 'six shape families; training-seed spread reported separately; exploratory descriptive intervals'}


def _valid_measurement(row):
    if row.get('status') != 'ok' or row.get('correct') is not True:
        return False
    latency = row.get('latency_ms')
    samples = row.get('evidence', {}).get('cuda_event_ms', [])
    if (not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency <= 0
            or len(samples) != 30 or any(not math.isfinite(v) or v <= 0 for v in samples)
            or not math.isclose(statistics.median(samples), latency, rel_tol=1e-12, abs_tol=1e-12)
            or row.get('evidence', {}).get('correctness', {}).get('passed') is not True
            or row.get('evidence', {}).get('input_integrity_verified') is not True):
        raise ValueError('Successful confirmation lacks exact raw samples/correctness evidence')
    return True


def live_comparison(documents, suite):
    """Validate bindings and report paired fresh latency, not native search regret."""
    from .native_audit import audit_live_task, SELECTION_COST_NOTE

    expected = set(_families(suite))
    if len(documents) != 3 or {d.get('repeat_index') for d in documents} != {0, 1, 2}:
        raise ValueError('Exactly three distinct fresh-process reports required')
    cohort = {d.get('cohort_id') for d in documents}
    if len(cohort) != 1 or None in cohort:
        raise ValueError('One physical GPU/runtime cohort required')
    protocols = [d.get('protocol') for d in documents]
    if any(p != protocols[0] for p in protocols) or protocols[0].get('fresh_oracle_measured') is not False:
        raise ValueError('One declared protocol without a fabricated live oracle required')
    by_task, seen_trials, counts = defaultdict(list), set(), defaultdict(int)
    selection_audits = []
    native_ids = {c.candidate_id for c in suite.candidates if c.config['implementation'] == 'triton'}
    compiled = next(c.candidate_id for c in suite.candidates if c.config['implementation'] == 'compiled')
    for doc in documents:
        if doc.get('suite_sha256') != suite.sha256 or len(doc['results']) != len(expected) or {
                r['task_id'] for r in doc['results']} != expected:
            raise ValueError('Incomplete or unexpected TEST cohort')
        for result in doc['results']:
            selection_audits.append({'repeat_index': doc['repeat_index'],
                                     **audit_live_task(result, suite, cohort_id=doc['cohort_id'])})
            native, ppo = (result['selectors'][key] for key in ('native', 'ppo'))
            tuning = native['records']
            if len(tuning) != 24 or {r['candidate_id'] for r in tuning} != native_ids or native['configs_attempted'] != 24:
                raise ValueError('Native autotuner must retain all twenty-four attempted configs')
            if len(ppo['turns']) != 8 or ppo['candidate_turns'] != 8 or ppo.get('policy_updates_permitted') is not False:
                raise ValueError('Eight fixed-policy live queries required')
            fallback = native['reference_fallback']
            if fallback.get('selected_before_confirmation') is not True or fallback['selected_role'] not in ('eager', 'compiled', 'native'):
                raise ValueError('Native reference fallback must be frozen before confirmation')
            measurements = result['confirmation']['measurements']
            if set(measurements) != {'eager', 'compiled', 'native', 'ppo'}:
                raise ValueError('All four independent confirmation roles required')
            ids = {'eager': None, 'compiled': compiled, 'native': native['selected_candidate_id'], 'ppo': ppo['selected_candidate_id']}
            selection_trials = {ppo['initial_info']['baseline']['trial_id']}
            selection_trials |= {r['info']['measurement']['trial_id'] for r in ppo['turns']}
            selection_trials |= {r['trial_id'] for r in fallback['probes'].values()}
            valid = {}
            for role, measurement in measurements.items():
                tid = measurement['trial_id']
                if (measurement['backend'] != 'cuda' or measurement['task_id'] != result['task_id']
                        or measurement['cohort_id'] != doc['cohort_id'] or measurement['candidate_id'] != ids[role]
                        or measurement['input_digest'] != native['input_digest'] or tid in selection_trials or tid in seen_trials):
                    raise ValueError('Confirmation reused a search trial or changed task/input/cohort/candidate')
                seen_trials.add(tid)
                valid[role] = _valid_measurement(measurement)
                counts['confirmation_' + ('passed' if valid[role] else 'failed')] += 1
            counts['native_config_trials'] += 24
            counts['native_failed_configs'] += sum(r['status'] != 'ok' for r in tuning)
            counts['ppo_candidate_queries'] += 8
            counts['ppo_failed_queries'] += sum(r['info']['measurement']['status'] != 'ok' for r in ppo['turns'])
            ratios = {}
            for name, base in [('native', 'native'), ('compiled', 'compiled'), ('native_with_fallback', fallback['selected_role'])]:
                ratios[name] = measurements[base]['latency_ms'] / measurements['ppo']['latency_ms'] if valid[base] and valid['ppo'] else None
            by_task[result['task_id']].append({'repeat_index': doc['repeat_index'], 'ratios': ratios,
                'native_tuning_seconds': native['selection_wall_seconds'], 'ppo_search_seconds': ppo['selection_wall_seconds'],
                'native_fallback_seconds': fallback['selection_wall_seconds'],
                'confirmation_latency_ms': {k: v['latency_ms'] if valid[k] else None for k, v in measurements.items()}})
    task_rows = []
    for task, repeats in sorted(by_task.items()):
        ratios = {key: statistics.median(r['ratios'][key] for r in repeats)
                  if all(r['ratios'][key] is not None for r in repeats) else None
                  for key in ('native', 'compiled', 'native_with_fallback')}
        task_rows.append({'task_id': task, 'median_speedup_ratios': ratios, 'repeats': repeats})
    summaries = {key: family_interval({r['task_id']: r['median_speedup_ratios'][key] for r in task_rows
                                     if r['median_speedup_ratios'][key] is not None}, suite, geometric=True)
                 for key in ('native', 'compiled', 'native_with_fallback')}
    return {'kind': 'fresh_ppo_native_autotune_confirmation', 'suite_sha256': suite.sha256,
            'cohort_id': next(iter(cohort)), 'counts': dict(counts), 'tasks': task_rows,
            'ppo_speedup_ratios': summaries, 'ratio_definition': 'comparator_latency / PPO_latency; greater than one favors PPO',
            'aggregation': 'median of three within-process paired ratios per task, then geometric mean over tasks',
            'search_budgets': {'native_configs': 24, 'native_fallback_probes': 3, 'ppo_queries': 8},
            'native_adapter': protocols[0]['native_adapter'], 'live_regret_reported': False,
            'independent_selection_audits': selection_audits,
            'selection_cost_note': SELECTION_COST_NOTE,
            'equal_search_budget': False, 'offline_ppo_training_cost_excluded_from_search_time': True,
            'interval_limit': 'six shape families on one allocation; descriptive bootstrap, not cross-GPU population inference'}


def main(argv=None):
    """Recompute summaries from recorded reports without running inference."""
    import argparse
    import json
    from pathlib import Path
    from .suite import load_suite

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('replay', 'live'))
    parser.add_argument('--suite', default='operator-generalization-v2')
    parser.add_argument('--reports', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    documents = [json.loads(path.read_text()) for path in args.reports]
    summarize = replay_comparison if args.kind == 'replay' else live_comparison
    result = summarize(documents, load_suite(args.suite))
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
