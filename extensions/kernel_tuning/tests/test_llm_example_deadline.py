"""Bounded real-client wrapper behavior; all policy data here are CPU fixtures."""
import importlib.util
import json
from pathlib import Path
import signal
import time
from types import SimpleNamespace

import pytest

from drivebench_tuning.replay_policy import ResponsesIndexPolicy, run_replay_policy
from drivebench_tuning.agent_budget import BudgetedResponsesClient
from test_measured_replay import table, FakeClient, bounded_fake, make_env


def load_example():
    path=Path(__file__).parents[1]/'examples/compare_llm_replay.py'
    spec=importlib.util.spec_from_file_location('bounded_llm_example',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class RecordingClient(FakeClient):
    def __init__(self, wait=0):
        super().__init__();self.options=[];self.wait=wait
    def with_options(self, **options):
        self.options.append(options);return self
    def create(self, **request):
        if self.wait:time.sleep(self.wait)
        return super().create(**request)


def test_expired_transport_never_dispatches_or_changes_alarm():
    m=load_example();client=RecordingClient();prior=signal.getsignal(signal.SIGALRM)
    transport=m.DeadlineTransport(client,9,clock=lambda:10)
    with pytest.raises(TimeoutError):transport.responses.create(model='fixture-model')
    assert not client.requests and not client.options
    assert signal.getsignal(signal.SIGALRM)==prior


def test_transport_timeout_clipped_to_remaining_stage_time():
    m=load_example();client=RecordingClient()
    transport=m.DeadlineTransport(client,13,clock=lambda:10).with_options(timeout=60,max_retries=0)
    response=transport.responses.create(model='fixture-model')
    assert response['status']=='completed'
    assert client.options==[{'max_retries':0,'timeout':3.0}]
    assert signal.getitimer(signal.ITIMER_REAL)[0]==0


def test_absolute_timer_stops_blocking_call_and_restores_handler():
    m=load_example();client=RecordingClient(wait=.15);prior=signal.getsignal(signal.SIGALRM)
    start=time.monotonic();transport=m.DeadlineTransport(client,start+.03)
    with pytest.raises(TimeoutError):transport.responses.create(model='fixture-model')
    assert not client.requests
    assert signal.getitimer(signal.ITIMER_REAL)[0]==0
    assert signal.getsignal(signal.SIGALRM)==prior


def test_existing_alarm_is_never_replaced():
    m=load_example();client=RecordingClient();signal.setitimer(signal.ITIMER_REAL,10)
    try:
        with pytest.raises(RuntimeError,match='cannot be replaced'):
            m.DeadlineTransport(client,time.monotonic()+1).responses.create(model='fixture-model')
        assert signal.getitimer(signal.ITIMER_REAL)[0]>9 and not client.requests
    finally:signal.setitimer(signal.ITIMER_REAL,0)


def test_expiry_consumes_every_remaining_turn_without_api_or_measurement(table,tmp_path):
    m=load_example();client,ledger,bounded=bounded_fake(tmp_path)
    selector=ResponsesIndexPolicy(bounded,suite=table.suite,model='fixture-model',max_calls=16)
    journal=m.DurableTurns(tmp_path/'turns.jsonl')
    policy=m.DurableDeadlinePolicy(selector,table.suite,journal,99,max_wall_seconds=600,clock=lambda:100)
    try:
        report=run_replay_policy(table,policy,name='expired-fixture',repeats=1)
        assert report['status']=='completed'
        assert report['counts']['rejected_policy_actions']==16
        assert report['counts']['candidate_calls']==0 and ledger.state['attempts']==0
        assert not client.requests
        assert all(e['reward_sum']==0 for e in report['episodes'])
        # Read before closing: every start/end was already flushed and fsynced.
        rows=[json.loads(x) for x in (tmp_path/'turns.jsonl').read_text().splitlines()]
        assert len(rows)==32 and all(r['api_receipt'] is None for r in rows if r['event']=='finished')
        assert all('public_observation' in r for r in rows if r['event']=='started')
    finally:journal.close();ledger.close()


def test_successful_action_and_usage_are_durable(table,tmp_path):
    m=load_example();client,ledger,bounded=bounded_fake(tmp_path)
    selector=ResponsesIndexPolicy(bounded,suite=table.suite,model='fixture-model',max_calls=16)
    journal=m.DurableTurns(tmp_path/'turns.jsonl')
    policy=m.DurableDeadlinePolicy(selector,table.suite,journal,time.monotonic()+10,max_wall_seconds=10)
    env=make_env(table)
    try:
        observation,info=env.reset(options={'task_index':0})
        assert policy(observation,info)==0
        rows=[json.loads(x) for x in (tmp_path/'turns.jsonl').read_text().splitlines()]
        assert rows[0]['event']=='started' and rows[1]['action']==0
        assert rows[1]['api_receipt']['usage']=={'input_tokens':10,'output_tokens':5}
        assert len(client.requests)==1 and ledger.state['attempts']==1
    finally:env.close();journal.close();ledger.close()


def test_interruption_keeps_started_record_and_sanitizes_exception(table,tmp_path):
    m=load_example();journal=m.DurableTurns(tmp_path/'turns.jsonl')
    class Interrupted:
        receipts=[];configuration={}
        def __call__(self,*args):raise KeyboardInterrupt('SENSITIVE_EXCEPTION_CONTENT')
    policy=m.DurableDeadlinePolicy(Interrupted(),table.suite,journal,time.monotonic()+10,max_wall_seconds=10)
    env=make_env(table)
    try:
        observation,info=env.reset(options={'task_index':0})
        with pytest.raises(KeyboardInterrupt):policy(observation,info)
        text=(tmp_path/'turns.jsonl').read_text();rows=[json.loads(x) for x in text.splitlines()]
        assert rows[0]['event']=='started' and rows[1]['error_type']=='KeyboardInterrupt'
        assert 'SENSITIVE_EXCEPTION_CONTENT' not in text
    finally:env.close();journal.close()
