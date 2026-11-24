"""Freeze modal policy selections and describe their cross-GPU transfer."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone

import numpy as np

from .generalization import digest
from .replay_cli import selection_manifest
from .replay import MeasuredReplayEvaluator
from .search import BudgetedTuningEnv
from .evaluation import _json


def validate_replay_report(table, report):
    """Reconstruct recorded actions on CPU; never rerun a policy or GPU trial.

    A report's identity strings alone are insufficient. Every source measurement,
    reward, incumbent transition and budget receipt must match the bound table.
    This does not authenticate who generated a valid policy action sequence.
    """
    protocol = {'tasks': [t.to_dict() for t in table.suite.tasks],
                'candidates': [c.to_dict() for c in table.suite.candidates],
                'trial_budget': 8, 'memory_weight': 0.0, 'allow_synthetic': False, 'allow_replay': True}
    if (report.get('status') != 'completed' or report.get('backend') != 'measured_replay'
            or report.get('measured_replay') is not True or report.get('synthetic') is not False
            or report.get('fresh_gpu_measurements') != 0 or report.get('split') != 'test'
            or report.get('table_sha256') != table.sha256 or report.get('suite_sha256') != table.suite.sha256
            or report.get('protocol') != protocol or report.get('protocol_sha256') != digest(protocol)):
        raise ValueError('report must be complete eight-turn, latency-only measured replay on this table')
    repeats, seed = report.get('repeats'), report.get('seed')
    if type(repeats) is not int or not 1 <= repeats <= 20 or type(seed) is not int or seed < 0:
        raise ValueError('bounded report repeat count and seed required')
    expected = [(i, repeat) for i in table.suite.splits['test'] for repeat in range(repeats)]
    episodes = report.get('episodes', [])
    if len(episodes) != len(expected) or report.get('task_indices') != list(table.suite.splits['test']):
        raise ValueError('missing or unexpected test-task episodes')
    totals = {k: 0 for k in ('baseline_calls', 'candidate_calls', 'failed_or_incorrect_trials', 'rejected_policy_actions')}
    seen_ids = set()
    for index, (episode, (task_index, repeat)) in enumerate(zip(episodes, expected)):
        if (episode.get('status') != 'completed' or episode.get('episode_index') != index
                or episode.get('task_index') != task_index or episode.get('repeat') != repeat
                or episode.get('seed') != seed + index or episode.get('journal_start_index') != 0):
            raise ValueError('missing, reordered or incomplete policy repeats')
        trial_id = episode.get('initial_info', {}).get('baseline', {}).get('trial_id', '')
        if not isinstance(trial_id, str) or not trial_id.startswith('replay:') or not trial_id.endswith(':1'):
            raise ValueError('initial replay event identity missing')
        evaluator = MeasuredReplayEvaluator(table, seed=seed + 100003 * task_index + repeat,
                                            event_namespace=trial_id[7:-2])
        env = BudgetedTuningEnv(table.suite.tasks, table.suite.candidates, evaluator,
                               trial_budget=8, memory_weight=0., allow_replay=True)
        try:
            observation, info = env.reset(seed=seed + index, options={'task_index': task_index})
            if (episode.get('initial_info') != _json(info) or episode.get('initial_observation') != _json(observation)
                    or episode.get('task_id') != info['task_id'] or len(episode.get('steps', [])) != 8):
                raise ValueError('initial source evidence or complete recorded trajectories differ')
            reward_sum = 0.
            for turn, step in enumerate(episode['steps']):
                if step.get('trial_index') != turn or step.get('observation') != _json(observation):
                    raise ValueError('pre-action observation differs from recorded trajectory')
                if step.get('action') is None:
                    observation, reward, terminated, truncated, info = env.reject_action(step.get('policy_rejection'))
                else:
                    if 'policy_rejection' in step:
                        raise ValueError('one turn cannot contain both action and rejection')
                    observation, reward, terminated, truncated, info = env.step(step['action'])
                expected_step = {'observation_after': _json(observation), 'info': _json(info),
                                 'reward': reward, 'terminated': terminated, 'truncated': truncated}
                if any(step.get(key) != value for key, value in expected_step.items()):
                    raise ValueError('source measurement, reward or incumbent differs from measured replay')
                reward_sum += reward
            journal = env.journal
            if (episode.get('final_info') != _json(info) or episode.get('reward_sum') != reward_sum
                    or episode.get('journal') != journal):
                raise ValueError('final trajectory or call journal differs from measured replay')
            counts = {'baseline_calls': 1,
                      'candidate_calls': sum(j['kind'] == 'candidate' for j in journal),
                      'rejected_policy_actions': sum(j['kind'] == 'policy_rejection' for j in journal),
                      'failed_or_incorrect_trials': sum(j['kind'] == 'candidate' and
                          j['measurement']['status'] != 'ok' for j in journal)}
            if any(episode.get(k) != v for k, v in counts.items()):
                raise ValueError('episode budget counts differ from replayed actions')
            for entry in journal:
                if 'measurement' in entry:
                    identifier = entry['measurement']['trial_id']
                    if identifier in seen_ids:
                        raise ValueError('duplicate replay event across episodes')
                    seen_ids.add(identifier)
            for key, value in counts.items():
                totals[key] += value
        finally:
            env.close()
    if report.get('counts') != totals:
        raise ValueError('report counts differ from reconstructed episodes')
    return True


def freeze_modal_selections(table, reports):
    """One modal final selection per policy/test task; ties follow suite order.

    All votes remain in the decision record. A representative source episode
    binds each modal choice to an actual policy output; no episode is rewritten.
    """
    order = {None: 0, **{c.candidate_id: i + 1 for i, c in enumerate(table.suite.candidates)}}
    test_ids = [table.suite.tasks[i].task_id for i in table.suite.splits['test']]
    representatives, decisions, names = [], [], set()
    for report in reports:
        if report['policy_name'] in names or report.get('status') != 'completed' or report.get('split') != 'test':
            raise ValueError('completed test reports with unique policy names required')
        names.add(report['policy_name'])
        validate_replay_report(table, report)
        chosen = []
        if {e['task_id'] for e in report['episodes']} != set(test_ids):
            raise ValueError('modal confirmation requires every test task')
        for task_id in test_ids:
            episodes = [e for e in report['episodes'] if e['task_id'] == task_id]
            if (len(episodes) != report['repeats'] or any(e['status'] != 'completed' for e in episodes)
                    or {e['repeat'] for e in episodes} != set(range(report['repeats']))):
                raise ValueError('missing, duplicate or incomplete policy repeats')
            for episode in episodes:
                steps = episode.get('steps', [])
                if (len(steps) != report['protocol']['trial_budget']
                        or [s['info']['trials_used'] for s in steps] != list(range(1, len(steps) + 1))
                        or episode['final_info'] != steps[-1]['info']):
                    raise ValueError('modal votes must match complete recorded episode trajectories')
            votes = Counter(e['final_info']['incumbent_candidate_id'] for e in episodes)
            if any(candidate not in order for candidate in votes):
                raise ValueError('unknown candidate in modal votes')
            selected = min(votes, key=lambda candidate: (-votes[candidate], order[candidate]))
            representative = next(e for e in episodes if e['final_info']['incumbent_candidate_id'] == selected)
            chosen.append(representative)
            decisions.append({'policy': report['policy_name'], 'task_id': task_id, 'candidate_id': selected,
                'episode_count': len(episodes), 'votes': [{'candidate_id': c, 'count': votes[c]}
                     for c in sorted(votes, key=lambda c: order[c])],
                'source_episode_indices': [e['episode_index'] for e in episodes],
                'representative_episode_index': representative['episode_index'],
                'source_report_sha256': digest(report)})
        r = deepcopy(report); r['episodes'] = chosen; representatives.append(r)
    if not reports:
        raise ValueError('nonempty policy reports required')
    result = selection_manifest(table, representatives)
    result.update(selection_rule='modal final candidate per policy/test task; ties eager then suite candidate order',
                  modal_decisions=decisions, frozen=True,
                  frozen_at=datetime.now(timezone.utc).isoformat(),
                  selection_updates_permitted=False,
                  source_policy_count=len(reports),
                  policy_episode_count=sum(len(r['episodes']) for r in reports))
    return result


def _ranks(values):
    """Average ranks for exact ties; no approximate-equality rank inflation."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind='stable')
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2 + 1
        i = j
    return ranks


def _spearman(a, b):
    if len(a) < 2:
        return None
    a, b = _ranks(a), _ranks(b)
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def transfer_analysis(source, target, frozen):
    """Score already-frozen source selections using the second GPU's table.

    The second table supplies the descriptive oracle and ranks only. It does
    not retrain, rerank policy actions, or alter a primary selection.
    """
    if source.suite.sha256 != target.suite.sha256:
        raise ValueError('transfer tables must use the exact same suite')
    if source.cohort_id == target.cohort_id:
        raise ValueError('transfer requires a separate hardware/runtime cohort')
    names = [t._document['runtime'].get('gpu_name') for t in (source, target)]
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError('both transfer tables require recorded GPU model names')
    if names[0] == names[1]:
        raise ValueError('hardware transfer requires a second GPU model')
    if source._document['protocol']['source_hashes'] != target._document['protocol']['source_hashes']:
        raise ValueError('transfer must retain the same benchmark implementation')
    if (frozen.get('frozen') is not True or frozen.get('selection_updates_permitted') is not False
            or frozen.get('table_sha256') != source.sha256 or frozen.get('cohort_id') != source.cohort_id
            or frozen.get('suite_sha256') != source.suite.sha256 or not frozen.get('modal_decisions')):
        raise ValueError('frozen primary selections are not bound to the source table')
    tasks = [source.suite.tasks[i] for i in source.suite.splits['test']]
    candidates = [c.candidate_id for c in source.suite.candidates]
    selected_pairs = {(s['task_id'], s['candidate_id']) for s in frozen['selections']}
    if any((d['task_id'], d['candidate_id']) not in selected_pairs for d in frozen['modal_decisions']):
        raise ValueError('modal decisions differ from frozen confirmation selections')
    records = []
    for task in tasks:
        source_input = source.row(task.task_id, None, source.repeat_indices[0])['input_digest']
        target_input = target.row(task.task_id, None, target.repeat_indices[0])['input_digest']
        if (source_input != target_input
                or frozen.get('expected_input_digests', {}).get(task.task_id) != source_input):
            raise ValueError('source/target/selection inputs differ for transfer task')
        src = {c: source.cell_summary(task.task_id, c) for c in [None] + candidates}
        dst = {c: target.cell_summary(task.task_id, c) for c in [None] + candidates}
        valid = [c for c in candidates if src[c]['admissible_in_all_repeats'] and dst[c]['admissible_in_all_repeats']]
        source_best = min([c for c in [None] + candidates if src[c]['admissible_in_all_repeats']],
                          key=lambda c: src[c]['median_cost'])
        target_best = min([c for c in [None] + candidates if dst[c]['admissible_in_all_repeats']],
                          key=lambda c: dst[c]['median_cost'])
        oracle = dst[target_best]['median_cost']
        def score(candidate):
            cell = dst[candidate]
            cost = cell['median_cost'] if cell['admissible_in_all_repeats'] else 1.0
            return {'candidate_id': candidate, 'target_cost': cost,
                    'target_failure_rate': cell['failure_rate'], 'target_regret': max(0., cost - oracle)}
        records.append({'task_id': task.task_id, 'rank_correlation': _spearman(
                           [src[c]['median_cost'] for c in valid], [dst[c]['median_cost'] for c in valid]),
                        'ranked_candidates_passing_on_both': len(valid),
                        'source_failed_candidate_count': sum(not src[c]['admissible_in_all_repeats'] for c in candidates),
                        'target_failed_candidate_count': sum(not dst[c]['admissible_in_all_repeats'] for c in candidates),
                        'source_optimal_transfer': score(source_best), 'target_oracle_candidate_id': target_best,
                        'target_oracle_cost': oracle,
                        'frozen_policy_selections': [{'policy': d['policy'], **score(d['candidate_id'])}
                             for d in frozen['modal_decisions'] if d['task_id'] == task.task_id]})
    policy_names = sorted({d['policy'] for d in frozen['modal_decisions']})
    keys = [(d['policy'], d['task_id']) for d in frozen['modal_decisions']]
    if len(keys) != len(set(keys)) or set(keys) != {(name, t.task_id) for name in policy_names for t in tasks}:
        raise ValueError('each frozen policy must have exactly one modal selection per test task')
    return {'schema_version': 1, 'kind': 'frozen_selection_cross_gpu_transfer',
            'source_table_sha256': source.sha256, 'target_table_sha256': target.sha256,
            'selection_sha256': digest(frozen), 'source_gpu': source._document['runtime']['gpu_name'],
            'target_gpu': target._document['runtime']['gpu_name'], 'suite_sha256': source.suite.sha256,
            'records': records, 'training_on_target': False, 'selection_updated_on_target': False,
            'target_table_use': 'descriptive ranks, oracle and frozen-selection scoring only',
            'summary': {'tasks': len(records),
                'mean_source_optimal_transfer_regret': float(np.mean([
                    r['source_optimal_transfer']['target_regret'] for r in records])),
                'policies': [{'policy': name, 'mean_target_regret': float(np.mean([
                    s['target_regret'] for r in records for s in r['frozen_policy_selections'] if s['policy'] == name]))}
                    for name in policy_names]},
            'rank_metric': 'Spearman average ranks of candidates passing every repeat on both GPUs',
            'scope': 'one source and one target hardware/runtime cohort; no causal hardware-only attribution'}
