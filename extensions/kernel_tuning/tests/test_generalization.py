"""CPU-only selector contracts on invented fixtures, never research evidence."""
from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest

from drivebench_tuning import generalization_cli as cli
from drivebench_tuning.generalization import (RidgeSelector, TPESelector, digest, load_cost_model,
                                             train_cost_model, validate_cost_model)
from drivebench_tuning.generalization_analysis import freeze_modal_selections, transfer_analysis
from drivebench_tuning.replay_policy import fixed_policy, run_replay_policy
from drivebench_tuning.search import BudgetedTuningEnv
from drivebench_tuning.replay import MeasuredReplayEvaluator, MeasuredTable
from drivebench_tuning.suite import load_suite
from test_measured_replay import table


def test_training_reads_train_only_and_is_reproducible(table, monkeypatch):
    train_ids = {table.suite.tasks[i].task_id for i in table.suite.splits['train']}
    read = table.row
    accessed = []
    def guarded(task_id, candidate, repeat):
        assert task_id in train_ids, 'held-out outcome leakage'
        accessed.append(task_id)
        return read(task_id, candidate, repeat)
    monkeypatch.setattr(table, 'row', guarded)
    a, b = train_cost_model(table), train_cost_model(table)
    assert a == b
    assert set(accessed) == train_ids
    assert a['training']['outcome_splits_read'] == ['train']
    assert a['training']['rl_training'] is False
    assert a['training']['cell_count'] == len(train_ids) * len(table.suite.candidates)
    assert all(r['task_id'] in train_ids for r in a['training']['records'])


def test_training_coefficients_ignore_heldout_values(table):
    a = train_cost_model(table)
    for (task, candidate, repeat), row in table._rows.items():
        if task not in a['training']['task_ids'] and row['status'] == 'ok':
            row['latency_ms'] *= 10000
    b = train_cost_model(table)
    assert a == b


def test_saved_model_has_strict_identity_and_numeric_validation(table, tmp_path):
    model = train_cost_model(table)
    path = tmp_path/'model.json'; path.write_text(json.dumps(model))
    assert load_cost_model(path, table.suite, expected_table_sha256=table.sha256,
                           expected_cohort_id=table.cohort_id) == model
    with pytest.raises(ValueError, match='another table'):
        load_cost_model(path, table.suite, expected_table_sha256='0'*64)
    with pytest.raises(ValueError, match='another GPU'):
        load_cost_model(path, table.suite, expected_cohort_id='0'*64)
    with pytest.raises(ValueError, match='identities differ'):
        validate_cost_model(model, replace(table.suite, sha256='0'*64))
    for field, value in [('coefficients', [float('nan')]*model['feature_count']),
                         ('scale', [0.]*model['feature_count']), ('feature_version', 'unreviewed')]:
        invalid = deepcopy(model); invalid[field] = value
        with pytest.raises(ValueError): validate_cost_model(invalid, table.suite)
    invalid = deepcopy(model); invalid['training']['records'][0]['task_id'] = table.suite.tasks[-1].task_id
    invalid['training']['training_records_sha256'] = digest(invalid['training']['records'])
    with pytest.raises(ValueError, match='TRAIN'):
        validate_cost_model(invalid, table.suite)


def test_learned_selector_owns_no_table_and_uses_no_unqueried_outcomes(table):
    model = train_cost_model(table)
    selector = RidgeSelector(model, suite=table.suite)
    env = BudgetedTuningEnv(table.suite.tasks, table.suite.candidates, MeasuredReplayEvaluator(table),
                            trial_budget=8, allow_replay=True)
    obs, info = env.reset(options={'task_index': table.suite.splits['test'][0]})
    first = selector(obs, info)
    obs['action_metrics'][:] = 1e20; info['unqueried_table'] = 'must be ignored'
    assert selector(obs, info) == first
    assert not hasattr(selector, '_table')
    model['coefficients'][:] = [100]*model['feature_count']
    assert selector(obs, info) == first  # constructor deep-copies learned weights.
    env.close()
    report = run_replay_policy(table, selector, name='ridge-selector', trials=8, repeats=1)
    assert report['status'] == 'completed' and report['training_performed'] is True
    assert report['rl_training_performed'] is False
    assert report['training']['externally_supplied_metadata']['outcome_splits_read'] == ['train']
    assert all(len({s['action'] for s in e['steps']}) == 8 for e in report['episodes'])


def test_real_tpe_is_deterministic_equal_budget_and_observes_last_turn(table):
    optuna = pytest.importorskip('optuna'); optuna.logging.set_verbosity(optuna.logging.WARNING)
    policies = [TPESelector(suite=table.suite, seed=42) for _ in range(2)]
    reports = [run_replay_policy(table, p, name='optuna-tpe', trials=8, repeats=2, seed=7) for p in policies]
    assert all(r['status'] == 'completed' for r in reports)
    assert [[s['action'] for e in r['episodes'] for s in e['steps']] for r in reports][0] == [
        s['action'] for e in reports[1]['episodes'] for s in e['steps']]
    for policy, report in zip(policies, reports):
        assert isinstance(policy._study.sampler, optuna.samplers.TPESampler)
        assert len(policy.search_receipts) == report['counts']['candidate_calls'] == 64
        assert policy._pending is None
        assert len(policy._study.trials) == 8
        assert all(t.state == optuna.trial.TrialState.COMPLETE for t in policy._study.trials)
        assert all(r['status'] == 'observed' for r in policy.search_receipts)
        assert sum(r['sampler_phase'] == 'startup' for r in policy.search_receipts) == 24
        assert any(r['repeated_action'] for r in policy.search_receipts)
        assert all(r['objective_value'] == 2.0 for r in policy.search_receipts if r['observed_status'] != 'ok')
        assert report['counts']['rejected_policy_actions'] == 0
        assert report['fresh_gpu_measurements'] == 0


def test_protocol_fixes_budget_training_splits_and_source(table, monkeypatch):
    monkeypatch.setattr(cli, 'source_hashes', lambda: {'fixture.py': 'f'*64})
    p = cli.make_protocol(table.suite)
    p['frozen_at'] = '2000-01-01T00:00:00+00:00'
    assert cli.validate_protocol(p, table.suite) == p
    assert p['trials'] == 8 and p['repeats'] == 20 and p['tpe']['startup_trials'] == 3
    for key, value in [('trials', 7), ('training_tasks', p['test_tasks']), ('source_hashes', {})]:
        bad = deepcopy(p); bad[key] = value
        with pytest.raises(ValueError): cli.validate_protocol(bad, table.suite)


def test_modal_ties_follow_suite_order_and_preserve_votes(table):
    # Alter only invented in-memory fixture costs, then produce real replay
    # transitions. No fabricated policy record is accepted by the validator.
    choices = (table.suite.candidates[0].candidate_id, table.suite.candidates[-1].candidate_id)
    for (_task, candidate, _repeat), row in table._rows.items():
        if candidate in choices:
            row['latency_ms'] *= .5
    def alternating(_obs, info):
        return len(table.suite.candidates)-1 if info['evaluation_episode_index'] % 2 == 0 else 0
    report = run_replay_policy(table, alternating, name='fixture-alternating', trials=8, repeats=2)
    frozen = freeze_modal_selections(table, [report])
    assert all(s['candidate_id'] == choices[0] for s in frozen['selections'])
    assert all(d['episode_count'] == 2 and sum(v['count'] for v in d['votes']) == 2 for d in frozen['modal_decisions'])
    assert frozen['policy_episode_count'] == 8 and frozen['frozen']
    with pytest.raises(ValueError, match='unique'):
        freeze_modal_selections(table, [report, report])
    bad = deepcopy(report); bad['episodes'].pop()
    with pytest.raises(ValueError, match='missing'):
        freeze_modal_selections(table, [bad])


def test_transfer_uses_frozen_selections_and_never_trains(table, monkeypatch):
    report = run_replay_policy(table, fixed_policy(), name='fixed-compiled', trials=8, repeats=1)
    frozen = freeze_modal_selections(table, [report])
    target = MeasuredTable(table._document, suite=table.suite)
    table._document['runtime']['gpu_name'] = 'PRIMARY-CPU-FIXTURE-NOT-A-GPU-RESULT'
    target.cohort_id = 'e'*64; target.sha256 = 'd'*64
    target._document['runtime']['gpu_name'] = 'SECOND-CPU-FIXTURE-NOT-A-GPU-RESULT'
    a = transfer_analysis(table, target, frozen)
    assert len(a['records']) == 4 and a['training_on_target'] is False and a['selection_updated_on_target'] is False
    assert all(r['source_optimal_transfer']['target_regret'] == 0 for r in a['records'])
    assert all(r['rank_correlation'] is None for r in a['records'])  # all invented fixture costs tie.
    with pytest.raises(ValueError, match='separate'):
        transfer_analysis(table, table, frozen)
    bad = deepcopy(frozen); bad['table_sha256'] = '1'*64
    with pytest.raises(ValueError, match='bound'):
        transfer_analysis(table, target, bad)


def test_complete_cpu_comparison_records_supervised_training_and_equal_budgets(table, tmp_path, monkeypatch):
    pytest.importorskip('optuna')
    monkeypatch.setattr(cli, 'source_hashes', lambda: {'fixture.py': 'f'*64})
    p = cli.make_protocol(table.suite); p['frozen_at'] = '2000-01-01T00:00:00+00:00'
    output = tmp_path/'cpu-contract-comparison'
    assert cli.run_comparison(table, p, output)
    reports = [json.loads((output/f'{name}.json').read_text()) for name in p['cpu_policies']]
    assert all(len(r['episodes']) == 80 and r['counts']['candidate_calls'] == 640 for r in reports)
    assert all(len(e['budget_analysis']['curve']) == 8 for r in reports for e in r['episodes'])
    assert [r['training_performed'] for r in reports] == [False, False, False, True]
    assert all(r['rl_training_performed'] is False for r in reports)
    request = json.loads((output/'confirmation-request-cpu.json').read_text())
    assert request['policy_episode_count'] == 320 and len(request['modal_decisions']) == 16


def test_v2_train_test_coverage_and_transfer_table_cannot_refit(tmp_path, monkeypatch):
    from drivebench_tuning import table_collection as tc
    from test_table_collection import FixtureRunner
    monkeypatch.setattr(tc, 'source_hashes', lambda: {'fixture.py': 'f'*64})
    suite = load_suite('operator-generalization-v2')
    tables = []
    for label, splits in [('full', None), ('test-only', ['test'])]:
        paths = []
        for repeat in range(3):
            monkeypatch.setattr(tc, '_PROCESS_INSTANCE_ID', f'CPU-fixture-{label}-{repeat}')
            path = tmp_path/f'{label}-{repeat}'
            assert tc.collect(path, repeat_index=repeat, repetitions=3, suite='operator-generalization-v2',
                              task_splits=splits, runner_factory=FixtureRunner) == 0
            paths.append(path)
        path = tmp_path/f'{label}.json'; tc.merge(paths, path)
        tables.append(MeasuredTable.load(path, suite=suite))
    primary, target = tables
    model = train_cost_model(primary)
    assert len(model['training']['task_ids']) == 30 and model['training']['cell_count'] == 750
    assert len(target.measured_tasks) == 18
    with pytest.raises(ValueError, match='complete training split'):
        train_cost_model(target)
    selector = RidgeSelector(model, suite=suite, expected_table_sha256=primary.sha256,
                             expected_cohort_id=primary.cohort_id)
    report = run_replay_policy(primary, selector, name='ridge-selector', trials=8, repeats=1)
    assert report['status'] == 'completed' and report['counts']['candidate_calls'] == 144
    assert all(e['task_id'] not in model['training']['task_ids'] for e in report['episodes'])
    tpe = TPESelector(suite=suite)
    assert tpe.configuration['search_space'] == {'implementation': ['compiled', 'triton'],
        'triton_block_exponent': [6, 11], 'triton_warp_exponent': [0, 3]}


def test_modal_rejects_missing_or_rewritten_trajectory(table):
    report = run_replay_policy(table, fixed_policy(), name='fixed', trials=8, repeats=1)
    bad = deepcopy(report); bad['episodes'][0]['steps'].pop()
    with pytest.raises(ValueError, match='trajectory|trajectories'):
        freeze_modal_selections(table, [bad])
    bad = deepcopy(report); bad['episodes'][0]['final_info']['incumbent_cost'] = 0.123456
    with pytest.raises(ValueError, match='trajectory|trajectories'):
        freeze_modal_selections(table, [bad])


def test_tpe_feedback_cannot_be_swapped_with_unqueried_action(table):
    env = BudgetedTuningEnv(table.suite.tasks, table.suite.candidates, MeasuredReplayEvaluator(table),
                            trial_budget=8, allow_replay=True)
    obs, info = env.reset(options={'task_index': table.suite.splits['test'][0]})
    policy = TPESelector(suite=table.suite)
    action = policy(obs, info)
    obs, _, _, _, feedback = env.step(action)
    feedback['measurement']['candidate_id'] = 'unqueried-candidate'
    with pytest.raises(ValueError, match='feedback differs'):
        policy.observe(obs, feedback)
    env.close()


@pytest.mark.parametrize('change', ['synthetic', 'backend', 'objective', 'source_id', 'reward'])
def test_modal_rejects_forged_report_evidence(table, change):
    report = run_replay_policy(table, fixed_policy(), name='fixed', trials=8, repeats=1)
    if change == 'synthetic': report['synthetic'] = True
    elif change == 'backend': report['backend'] = 'cuda'
    elif change == 'objective':
        report['protocol']['memory_weight'] = 0.5
        report['protocol_sha256'] = digest(report['protocol'])
    elif change == 'source_id':
        report['episodes'][0]['steps'][0]['info']['measurement']['evidence']['source_trial_id'] = 'fabricated'
    else: report['episodes'][0]['steps'][0]['reward'] = 99.
    with pytest.raises(ValueError):
        freeze_modal_selections(table, [report])


def test_transfer_rejects_changed_source_and_inputs(table):
    report = run_replay_policy(table, fixed_policy(), name='fixed', trials=8, repeats=1)
    frozen = freeze_modal_selections(table, [report])
    target = MeasuredTable(table._document, suite=table.suite)
    table._document['runtime']['gpu_name'] = 'PRIMARY-FIXTURE'
    target._document['runtime']['gpu_name'] = 'TARGET-FIXTURE'
    target.cohort_id = 'e'*64
    target._document['protocol']['source_hashes'] = {'other.py': '0'*64}
    with pytest.raises(ValueError, match='same benchmark implementation'):
        transfer_analysis(table, target, frozen)
    target._document['protocol']['source_hashes'] = deepcopy(table._document['protocol']['source_hashes'])
    task = table.suite.tasks[table.suite.splits['test'][0]].task_id
    target._rows[(task, None, target.repeat_indices[0])]['input_digest'] = '0'*64
    with pytest.raises(ValueError, match='inputs differ'):
        transfer_analysis(table, target, frozen)


def test_protocol_rejects_unbounded_or_mismatched_llm_supplement():
    suite = load_suite('operator-generalization-v2')
    plan = {'kind':'pretrained_llm_supplement','test_task_ids':[suite.tasks[i].task_id for i in suite.splits['test']],
        'repeats':1,'trials':8,'max_calls_per_model':144,'max_output_tokens':1024,'max_wall_seconds':600,
        'retain_history':True,'training_performed':False,'automatic_retries':0,'reservation_refunds':False,
        'reserved_total_cap_usd':'3.50','models':[
            {'model':'fixture-mini','input_usd_per_million':'0.75','output_usd_per_million':'4.50','cap_usd':'2.50'},
            {'model':'fixture-nano','input_usd_per_million':'0.20','output_usd_per_million':'1.25','cap_usd':'1.00'}]}
    assert cli.make_protocol(suite, llm_plan=plan)['llm_supplement'] == plan
    for field, value in [('repeats',3),('trials',9),('test_task_ids',[]),('automatic_retries',1),('reserved_total_cap_usd','3.51')]:
        bad=deepcopy(plan);bad[field]=value
        with pytest.raises(ValueError):cli.make_protocol(suite,llm_plan=bad)
