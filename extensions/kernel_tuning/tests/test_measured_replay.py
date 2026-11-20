"""Replay/selector contract checks with invented CPU fixtures, never results."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from drivebench_tuning import table_collection as tc
from drivebench_tuning.agent_budget import APICostLedger, APICostPolicy, BudgetedResponsesClient
from drivebench_tuning.evaluation import PolicyActionRejected, grid_policy, random_policy
from drivebench_tuning.replay import MeasuredTable, MeasuredReplayEvaluator
from drivebench_tuning.replay_cli import comparison_summary, selection_manifest
from drivebench_tuning.replay_policy import (ResponsesIndexPolicy, fixed_policy,
                                            public_policy_observation, run_replay_policy)
from drivebench_tuning.search import BudgetedTuningEnv, CandidateSpec
from drivebench_tuning.suite import load_suite
from test_table_collection import FixtureRunner


@pytest.fixture
def table(tmp_path, monkeypatch):
    monkeypatch.setattr(tc, 'source_hashes', lambda: {'fixture.py': 'f' * 64})
    directories = []
    for index in range(3):
        monkeypatch.setattr(tc, '_PROCESS_INSTANCE_ID', f'CPU-contract-only-{index}')
        path = tmp_path / f'repeat-{index}'
        assert tc.collect(path, repeat_index=index, repetitions=3, runner_factory=FixtureRunner) == 0
        directories.append(path)
    path = tmp_path / 'table.json'
    tc.merge(directories, path)
    return MeasuredTable.load(path)


def make_env(table, *, seed=0, allow_replay=True):
    return BudgetedTuningEnv(table.suite.tasks, table.suite.candidates,
                            MeasuredReplayEvaluator(table, seed=seed), trial_budget=4,
                            allow_replay=allow_replay)


def test_replay_opt_in_and_original_source_identity(table):
    env = make_env(table, allow_replay=False)
    with pytest.raises(ValueError, match='allow_replay'):
        env.reset(options={'task_index': 0})
    env.close()
    env = make_env(table, seed=2)
    _, info = env.reset(options={'task_index': 0})
    assert info['measured_replay'] and not info['fresh_gpu_measurement'] and not info['synthetic']
    sources = []
    ids = []
    for _ in range(4):
        _, _, _, _, info = env.step(0)
        measurement = info['measurement']
        assert measurement['backend'] == 'measured_replay'
        source = table.row(table.suite.tasks[0].task_id, table.suite.candidates[0].candidate_id,
                           measurement['evidence']['source_repeat_index'])
        assert measurement['evidence']['source_trial_id'] == source['trial_id']
        assert measurement['evidence']['table_sha256'] == table.sha256
        assert measurement['latency_ms'] == source['latency_ms']
        sources.append(measurement['evidence']['source_repeat_index'])
        ids.append(measurement['trial_id'])
    assert sources == [2, 0, 1, 2]
    assert len(set(ids)) == 4  # replay event IDs, not four independent GPU trials.


def test_changed_config_and_missing_cell_are_rejected(table):
    evaluator = MeasuredReplayEvaluator(table)
    task = table.suite.tasks[0]
    with pytest.raises(ValueError, match='candidate differs'):
        evaluator.evaluate(task, CandidateSpec('torch-compiled', {'implementation': 'eager'}))
    document = deepcopy(table._document)
    document['rows'].pop()
    with pytest.raises(ValueError):
        MeasuredTable(document, suite=table.suite)


def test_measured_failure_consumes_budget_without_reward(table):
    env = make_env(table)
    env.reset(options={'task_index': 0})
    _, reward, terminated, _, info = env.step(2)  # deliberately failed CPU fixture.
    assert reward == 0 and not terminated and info['trials_used'] == 1
    assert info['measurement']['status'] == 'error'
    assert info['incumbent_candidate_id'] is None


def test_public_observation_does_not_disclose_unqueried_table(table):
    env = make_env(table)
    observation, info = env.reset(options={'task_index': 0})
    public = public_policy_observation(table.suite, observation, info)
    assert public['queried_actions'] == []
    assert len(public['candidates']) == 9
    raw = json.dumps(public)
    for forbidden in ('rows', 'cells', 'table_sha256', 'source_trial_id', 'oracle', 'cohort_id', 'input_digest'):
        assert forbidden not in public
    assert 'CPU-CONTRACT-FIXTURE' not in raw
    observation, _, _, _, info = env.step(1)
    public = public_policy_observation(table.suite, observation, info)
    assert [r['action'] for r in public['queried_actions']] == [1]


def test_replay_comparison_counts_and_confirmation_selection(table):
    reports = [run_replay_policy(table, p, name=n, repeats=2)
               for n, p in [('fixed', fixed_policy()), ('grid', grid_policy), ('random', random_policy(9))]]
    for report in reports:
        assert report['status'] == 'completed'
        assert report['counts']['candidate_calls'] == 4 * 4 * 2
        assert report['fresh_gpu_measurements'] == 0
        assert report['independent_gpu_samples_added_by_policy_evaluation'] == 0
        assert len(report['episodes']) == 8
        assert all(len(e['table_analysis']['curve']) == 4 for e in report['episodes'])
    selections = selection_manifest(table, reports)
    assert not selections['confirmed']
    assert len(selections['selections']) == 4  # equal invented costs select the reference.
    assert all(len(r['selected_by']) == 6 for r in selections['selections'])
    assert set(selections['expected_input_digests']) == {t.task_id for t in table.suite.tasks if t.metadata['split'] == 'test'}
    from drivebench_tuning.confirmation import selection_pairs
    assert len(selection_pairs(selections, table.suite)) == 4
    summary = comparison_summary(reports)
    assert len(summary['policies']) == 3
    assert all(p['budget_curve'][0]['mean_regret'] == 0 for p in summary['policies'])


def test_rejected_policy_turns_are_not_fabricated_measurements(table):
    def refused(observation, info):
        raise PolicyActionRejected('refused')
    report = run_replay_policy(table, refused, name='fixture-refusal', repeats=1)
    assert report['status'] == 'completed'
    assert report['counts']['candidate_calls'] == 0
    assert report['counts']['rejected_policy_actions'] == 16
    assert all(e['reward_sum'] == 0 for e in report['episodes'])
    assert all(s['info']['measurement'] is None for e in report['episodes'] for s in e['steps'])
    assert all(e['final_info']['trials_used'] == 4 for e in report['episodes'])


def test_common_recorded_repeats_when_policy_repeat_counts_differ(table):
    short = run_replay_policy(table, grid_policy, name='three-repeats', repeats=3)
    long = run_replay_policy(table, grid_policy, name='five-repeats', repeats=5)
    matching = {(e['task_id'], e['repeat']): e for e in long['episodes']}
    for episode in short['episodes']:
        other = matching[(episode['task_id'], episode['repeat'])]
        sources = lambda e: [s['info']['measurement']['evidence']['source_trial_id'] for s in e['steps']]
        assert sources(episode) == sources(other)


class FakeClient:
    def __init__(self, response_text='{"action": 0}'):
        self.requests = []
        self.response_text = response_text
        self.responses = SimpleNamespace(create=self.create)

    def with_options(self, **options):
        assert options.get('max_retries', 0) == 0
        return self

    def close(self):
        self.closed = True

    def create(self, **request):
        self.requests.append(request)
        return {'id': 'fixture-response', 'status': 'completed', 'model': 'fixture-model',
                'service_tier': 'default', 'usage': {'input_tokens': 10, 'output_tokens': 5},
                'output': [{'type': 'message', 'role': 'assistant', 'status': 'completed',
                            'content': [{'type': 'output_text', 'text': self.response_text}]}]}


def bounded_fake(tmp_path, *, response_text='{"action": 0}'):
    client = FakeClient(response_text)
    policy = APICostPolicy(budget_id='fixture', model='fixture-model', input_usd_per_million_tokens='1',
                          output_usd_per_million_tokens='1', hard_usd_cap='1', max_output_tokens=1024,
                          input_bound_kind='explicit_upper_bound', input_bound_id='fixture-bytes-plus-2048')
    ledger = APICostLedger.create(tmp_path / 'ledger', policy=policy,
                                 input_token_bound=lambda r: len(json.dumps(r).encode()) + 2048)
    return client, ledger, BudgetedResponsesClient(client, ledger)


def test_bounded_llm_selects_only_indices_and_retains_usage(table, tmp_path):
    client, ledger, bounded = bounded_fake(tmp_path)
    try:
        policy = ResponsesIndexPolicy(bounded, suite=table.suite, model='fixture-model', max_calls=16)
        report = run_replay_policy(table, policy, name='fixture-llm', repeats=1)
        assert report['status'] == 'completed' and len(client.requests) == 16
        assert len(report['policy_api_receipts']) == 16
        assert all(r['status'] == 'completed' and r['usage']['input_tokens'] == 10 for r in policy.receipts)
        assert all(r['tools'] == [] and r['store'] is False for r in client.requests)
        assert all(r['text']['format']['schema']['properties']['action']['enum'] == list(range(9)) for r in client.requests)
        assert ledger.state['attempts'] == 16
    finally:
        ledger.close()


@pytest.mark.parametrize('text', ['{"action":99}', '{"action":true}', '{"action":0,"action":1}',
                                  '{"action":0,"code":"print(1)"}', 'not json'])
def test_invalid_llm_output_charged_no_retry_no_fallback(table, tmp_path, text):
    client, ledger, bounded = bounded_fake(tmp_path, response_text=text)
    try:
        policy = ResponsesIndexPolicy(bounded, suite=table.suite, model='fixture-model', max_calls=16)
        report = run_replay_policy(table, policy, name='invalid-fixture-llm', repeats=1)
        assert len(client.requests) == ledger.state['attempts'] == 16
        assert report['counts']['candidate_calls'] == 0
        assert report['counts']['rejected_policy_actions'] == 16
        assert all(r['status'] == 'failed' for r in policy.receipts)
    finally:
        ledger.close()


def test_example_driver_cost_ledger_and_public_output(table, tmp_path, monkeypatch):
    example = Path(__file__).parents[1] / 'examples' / 'compare_llm_replay.py'
    spec = importlib.util.spec_from_file_location('compare_llm_replay_fixture', example)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / 'measured-fixture-table.json'
    path.write_text(json.dumps(table._document))
    key = tmp_path / 'fixture-secret'
    key.write_text('fixture-only-not-a-key')
    key.chmod(0o600)
    output = tmp_path / 'experiment'
    client = FakeClient()
    monkeypatch.setitem(sys.modules, 'openai', SimpleNamespace(OpenAI=lambda **kwargs: client))
    monkeypatch.setattr(sys, 'argv', [str(example), '--table', str(path), '--output', str(output),
        '--key-file', str(key), '--private-ledger', str(tmp_path / 'private-ledger'), '--model', 'fixture-model',
        '--input-usd-per-million', '0.75', '--output-usd-per-million', '4.5', '--cap-usd', '0.65', '--repeats', '1'])
    assert module.main() == 0
    result = json.loads((output / 'result.json').read_text())
    assert result['api_cost']['attempts'] == len(client.requests) == 16
    assert 0 < float(result['api_cost']['reported_usage_estimated_usd']) < float(result['api_cost']['reserved_usd']) < .65
    assert result['counts']['candidate_calls'] == 16 and result['counts']['rejected_policy_actions'] == 0
    assert client.closed
    assert 'fixture-only-not-a-key' not in ''.join(p.read_text() for p in output.iterdir())
