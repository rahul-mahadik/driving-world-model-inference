"""Freeze and execute an equal-budget, TRAIN-only operator search protocol."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path

from .benchmark import COMPILER_STATE_POLICY, source_hashes
from .evaluation import random_policy
from .generalization import (OPTUNA_VERSION, RIDGE_ALPHA, FAILURE_COST, FEATURE_VERSION,
                             RidgeSelector, TPESelector, digest, train_cost_model, structured_tpe_space)
from .generalization_analysis import freeze_modal_selections, transfer_analysis
from .replay import MeasuredTable
from .replay_cli import comparison_summary, write_json
from .replay_policy import fixed_policy, run_replay_policy
from .suite import load_suite, _strict_object, _reject_constant


def read_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=_strict_object, parse_constant=_reject_constant)


def make_protocol(suite, *, seed=2101, llm_plan=None):
    if type(seed) is not int or not 0 <= seed < 2**32 or len(suite.candidates) <= 8:
        raise ValueError('bounded seed and more than eight configurations required')
    if llm_plan is not None:
        test_ids = [suite.tasks[i].task_id for i in suite.splits['test']]
        if (llm_plan.get('kind') != 'pretrained_llm_supplement' or llm_plan.get('test_task_ids') != test_ids
                or llm_plan.get('repeats') != 1 or llm_plan.get('trials') != 8
                or llm_plan.get('max_calls_per_model') != len(test_ids) * 8
                or llm_plan.get('max_output_tokens') != 1024 or llm_plan.get('max_wall_seconds') != 600
                or llm_plan.get('retain_history') is not True or llm_plan.get('training_performed') is not False
                or llm_plan.get('automatic_retries') != 0 or llm_plan.get('reservation_refunds') is not False):
            raise ValueError('LLM supplement differs from the bounded one-episode protocol')
        models = llm_plan.get('models', [])
        if len(models) != 2 or len({m.get('model') for m in models}) != 2:
            raise ValueError('two distinct explicit model snapshots required')
        for model in models:
            if (not isinstance(model.get('model'), str) or not model['model']
                    or any(not Decimal(model[k]).is_finite() or Decimal(model[k]) <= 0
                           for k in ('input_usd_per_million', 'output_usd_per_million', 'cap_usd'))):
                raise ValueError('explicit positive model prices and caps required')
        if (sum(Decimal(m['cap_usd']) for m in models) != Decimal(llm_plan['reserved_total_cap_usd'])
                or Decimal(llm_plan['reserved_total_cap_usd']) > Decimal('3.50')):
            raise ValueError('LLM reservations must fit the declared $3.50 total cap')
    return {'schema_version': 1, 'kind': 'operator_generalization_preregistered_protocol',
        'suite_sha256': suite.sha256, 'source_hashes': source_hashes(),
        'compiler_state_policy': dict(COMPILER_STATE_POLICY),
        'training_tasks': [suite.tasks[i].task_id for i in suite.splits['train']],
        'validation_tasks': [suite.tasks[i].task_id for i in suite.splits['validation']],
        'test_tasks': [suite.tasks[i].task_id for i in suite.splits['test']],
        'split_boundary': 'shape families; all dtypes/seeds/group counts stay together; not operator-family holdout',
        'cpu_policies': ['fixed-compiled', 'random', 'optuna-tpe', 'ridge-selector'],
        'trials': 8, 'repeats': 20, 'seed': seed, 'split': 'test', 'memory_weight': 0.0,
        'learned': {'training': 'supervised ridge regression on TRAIN only', 'feature_version': FEATURE_VERSION,
                    'ridge_alpha': RIDGE_ALPHA, 'failure_cost': FAILURE_COST,
                    'hyperparameter_selection': 'none; fixed before collection', 'rl_training': False},
        'tpe': {'library': 'optuna', 'version': OPTUNA_VERSION, 'startup_trials': 3,
                'n_ei_candidates': 24, 'multivariate': False, 'constant_liar': False,
                'search_space': structured_tpe_space(suite), 'failed_cost': FAILURE_COST,
                'successful_cost': 'min(observed paired latency ratio, 2)',
                'repeated_suggestions_consume_turns': True, 'resampling': False},
        'llm_supplement': deepcopy(llm_plan),
        'primary': 'all four CPU policies, 20 episodes/task; equal task weighting',
        'comparison': 'per-turn table regret, failed/rejected turns, repeated actions and zero-reward turns',
        'confirmation': 'modal final candidate per policy/test task; ties eager then suite order; freeze before fresh confirmation and target collection',
        'transfer': 'second GPU test-only table; no training or selection update on target; descriptive per-task rank correlation and frozen-selection regret',
        'validation_use': 'unread outcomes; no hyperparameter selection',
        'claims': {'policy_superiority': False, 'rl_training': False, 'video_quality_preservation': False}}


def validate_protocol(document, suite):
    if set(document) - {'frozen_at'} != set(make_protocol(suite)):
        raise ValueError('unexpected or missing frozen protocol fields')
    expected = make_protocol(suite, seed=document.get('seed'), llm_plan=document.get('llm_supplement'))
    if ({k: v for k, v in document.items() if k != 'frozen_at'} != expected
            or digest(document['compiler_state_policy']) != digest(expected['compiler_state_policy'])):
        raise ValueError('frozen protocol differs from installed source/suite or fixed evaluation settings')
    stamp = datetime.fromisoformat(document.get('frozen_at', ''))
    if stamp.tzinfo is None:
        raise ValueError('timezone-aware protocol freeze time required')
    return deepcopy(document)


def add_budget_metrics(report):
    result = deepcopy(report)
    for episode in result['episodes']:
        seen, repeated, zero, invalid = set(), 0, 0, 0
        cumulative = []
        for step in episode['steps']:
            action = step['action']
            repeated += action is not None and action in seen
            if action is not None:
                seen.add(action)
            measurement = step['info']['measurement']
            invalid += bool(step.get('policy_rejection')) or bool(measurement and measurement['status'] != 'ok')
            zero += step['reward'] == 0.0
            cumulative.append({'trials_used': step['info']['trials_used'], 'repeated_action_turns': repeated,
                               'failed_or_rejected_turns': invalid, 'zero_reward_turns': zero,
                               'distinct_candidates_queried': len(seen)})
        episode['budget_analysis'] = {'curve': cumulative, 'repeated_action_turns': repeated,
                                     'zero_reward_turns': zero, 'failed_or_rejected_turns': invalid}
    return result


def run_comparison(table, protocol, output):
    protocol = validate_protocol(protocol, table.suite)
    if table._document['protocol']['source_hashes'] != protocol['source_hashes']:
        raise ValueError('table was measured with a different source revision than frozen protocol')
    if digest(table._document['protocol'].get('compiler_state_policy')) != digest(protocol['compiler_state_policy']):
        raise ValueError('table compiler-state policy differs from frozen study protocol')
    if any(datetime.fromisoformat(r['at']) <= datetime.fromisoformat(protocol['frozen_at']) for r in table._document['runs']):
        raise ValueError('protocol must be frozen before collection completes')
    # The original freeze receipt/host manifest, not this timestamp comparison,
    # establishes prospective ordering. Source hashes bind the implementation.
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    write_json(output/'protocol.json', protocol)
    model = train_cost_model(table)
    write_json(output/'trained-model.json', model)
    compiled = next(i for i, c in enumerate(table.suite.candidates)
                    if c.to_dict()['config'] == {'implementation': 'compiled'})
    policies = [('fixed-compiled', fixed_policy(compiled)), ('random', random_policy(protocol['seed'])),
                ('optuna-tpe', TPESelector(suite=table.suite, seed=protocol['seed'], startup_trials=3)),
                ('ridge-selector', RidgeSelector(model, suite=table.suite,
                    expected_table_sha256=table.sha256, expected_cohort_id=table.cohort_id))]
    reports = []
    for name, policy in policies:
        report = run_replay_policy(table, policy, name=name, split='test', repeats=20,
                                  seed=protocol['seed'], trials=8, memory_weight=0.)
        report = add_budget_metrics(report)
        report['frozen_study_protocol_sha256'] = digest(protocol)
        write_json(output/f'{name}.json', report); reports.append(report)
        if report['status'] != 'completed':
            write_json(output/'failed.json', {'completed': False, 'policy': name})
            return False
    summary = comparison_summary(reports)
    summary.update(policy_training_performed=True, training_kind='supervised cost model', rl_training_performed=False)
    summary['trained_model_sha256'] = digest(model)
    summary['frozen_study_protocol_sha256'] = digest(protocol)
    for policy, report in zip(summary['policies'], reports):
        policy['report_file'] = report['policy_name'] + '.json'
        policy['report_sha256'] = digest(report)
        for point in policy['budget_curve']:
            rows = [e['budget_analysis']['curve'][point['trials_used'] - 1] for e in report['episodes']]
            for metric in ('repeated_action_turns', 'failed_or_rejected_turns',
                           'zero_reward_turns', 'distinct_candidates_queried'):
                point['mean_' + metric] = sum(r[metric] for r in rows) / len(rows)
    write_json(output/'comparison.json', summary)
    write_json(output/'confirmation-request-cpu.json', freeze_modal_selections(table, reports))
    write_json(output/'completed.json', {'completed': True, 'backend': 'measured_replay',
                                         'fresh_gpu_measurements': 0, 'rl_training_performed': False})
    return True


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    freeze = sub.add_parser('freeze'); freeze.add_argument('--suite', required=True)
    freeze.add_argument('--output', required=True, type=Path); freeze.add_argument('--seed', type=int, default=2101)
    freeze.add_argument('--llm-plan', type=Path)
    run = sub.add_parser('run'); run.add_argument('--suite', required=True)
    run.add_argument('--table', type=Path, required=True); run.add_argument('--protocol', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True)
    select = sub.add_parser('selections'); select.add_argument('--suite', required=True)
    select.add_argument('--table', type=Path, required=True); select.add_argument('--reports', type=Path, nargs='+', required=True)
    select.add_argument('--output', type=Path, required=True)
    transfer = sub.add_parser('transfer'); transfer.add_argument('--suite', required=True)
    transfer.add_argument('--source-table', type=Path, required=True); transfer.add_argument('--target-table', type=Path, required=True)
    transfer.add_argument('--selections', type=Path, required=True); transfer.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv); suite = load_suite(args.suite)
    if args.command == 'freeze':
        doc = make_protocol(suite, seed=args.seed, llm_plan=read_json(args.llm_plan) if args.llm_plan else None)
        doc['frozen_at'] = datetime.now(timezone.utc).isoformat(); write_json(args.output, doc)
    elif args.command == 'run':
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        return 0 if run_comparison(MeasuredTable.load(args.table, suite=suite), read_json(args.protocol), args.output) else 1
    elif args.command == 'selections':
        write_json(args.output, freeze_modal_selections(MeasuredTable.load(args.table, suite=suite),
                                                       [read_json(path) for path in args.reports]))
    else:
        write_json(args.output, transfer_analysis(MeasuredTable.load(args.source_table, suite=suite),
            MeasuredTable.load(args.target_table, suite=suite), read_json(args.selections)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
