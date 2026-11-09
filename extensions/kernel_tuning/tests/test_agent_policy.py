"""Mocked provider transport and CPU-only durable action-loop contracts."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from drivebench_tuning.agent_policy import ActionError, ResponsesActionPolicy, run_agent, validate_action
from drivebench_tuning.agent_session import AgentSession, SessionError, SessionPolicy


def model_response(action=None, **updates):
    action = {'tool': 'inspect', 'arguments_json': '{}'} if action is None else action
    response = {'id': 'resp_mock', 'model': 'exact-test-model-snapshot', 'status': 'completed',
                'error': None, 'incomplete_details': None,
                'usage': {'input_tokens': 11, 'output_tokens': 7, 'total_tokens': 18},
                'output': [{'type': 'message', 'role': 'assistant', 'status': 'completed',
                            'content': [{'type': 'output_text', 'text': json.dumps(action)}]}]}
    response.update(updates)
    return response


class Client:
    def __init__(self, responses):
        self.outputs, self.requests, self.options = list(responses), [], []
        self.responses = SimpleNamespace(create=self.create)

    def with_options(self, **kwargs):
        self.options.append(kwargs)
        return self

    def create(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        value = self.outputs.pop(0)
        if isinstance(value, BaseException):
            raise value
        return deepcopy(value)


def selector(outputs, **kwargs):
    client = Client(outputs)
    return ResponsesActionPolicy(client, model=kwargs.pop('model', 'exact-test-model'),
                                 max_calls=kwargs.pop('max_calls', 4), **kwargs), client


def make_session(tmp_path, executor=None, clock=None):
    policy = SessionPolicy(session_id='cpu-policy-fixture', cohort_id='cpu-fixture',
        reference_config={'optimizations': {'compile_denoiser': False}},
        allowed_config_values={'optimizations.compile_denoiser': [False, True]},
        wall_seconds=100, max_tool_calls=12, max_evaluations=3, max_profiles=1)
    executor = executor or (lambda kind, proposal, context: {'cohort_id': 'cpu-fixture', 'status': 'ok', 'synthetic': True})
    return AgentSession.create(tmp_path / 'session', policy=policy, executor=executor, clock=clock or (lambda: 1000.))


def public(state, outcome):
    # Deliberately explicit projection: does not expose raw executor output.
    return {'proposal_id': state['latest_proposal_id'], 'remaining_tools': state['tool_calls_remaining'],
            'last_tool_ok': None if outcome is None else outcome['ok']}


def test_responses_request_has_exact_model_strict_json_and_no_host_or_provider_tools():
    policy, client = selector([model_response()], max_output_tokens=500, request_timeout_seconds=12)
    assert client.requests == []
    assert policy({'public': 'fixture'}) == {'tool': 'inspect', 'arguments': {}}
    request = client.requests[0]
    assert request['model'] == 'exact-test-model'
    assert request['tools'] == [] and request['store'] is False
    assert request['max_output_tokens'] == 500
    assert request['text']['format']['type'] == 'json_schema'
    assert request['text']['format']['strict'] is True
    assert request['text']['format']['schema']['additionalProperties'] is False
    assert client.options == [{'max_retries': 0, 'timeout': 12}] * 2
    assert json.loads(request['input'][0]['content']) == {'public': 'fixture'}
    assert set(request) == {'model', 'instructions', 'input', 'tools', 'store', 'max_output_tokens', 'text', 'service_tier'}
    assert request['service_tier'] == policy.configuration['service_tier'] == 'default'
    assert policy.last_receipt['usage']['total_tokens'] == 18
    assert policy.last_receipt['requested_model'] == 'exact-test-model'
    assert policy.last_receipt['returned_model'] == 'exact-test-model-snapshot'


def test_sdk_model_dump_is_supported_without_importing_the_sdk():
    class Response:
        def model_dump(self, *, mode):
            assert mode == 'json'
            return model_response()

    policy, _ = selector([Response()])
    assert policy({})['tool'] == 'inspect'


@pytest.mark.parametrize('response', [
    model_response(status='incomplete', incomplete_details={'reason': 'max_output_tokens'}),
    model_response(status='failed', error={'message': 'mock error'}),
    model_response(output=[{'type': 'function_call', 'name': 'shell', 'arguments': '{}'}]),
    model_response(output=[{'type': 'message', 'role': 'assistant', 'content': [{'type': 'refusal', 'refusal': 'No'}]}]),
    model_response(output=[{'type': 'message', 'role': 'user', 'content': [{'type': 'output_text', 'text': '{}'}]}]),
    model_response(output=[]),
    model_response(output=model_response()['output'] * 2),
    model_response(output=[{'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'not JSON'}]}]),
    model_response({'tool': 'shell', 'arguments_json': '{}'}),
    model_response({'tool': 'inspect', 'arguments_json': '[]'}),
    model_response({'tool': 'inspect', 'arguments_json': '{"x":NaN}'}),
    model_response({'tool': 'inspect', 'arguments_json': '{"x":1,"x":2}'}),
    model_response({'tool': 'inspect', 'arguments_json': '{}', 'extra': 'forbidden'}),
])
def test_noncompleted_refusal_tools_and_malformed_outputs_rejected_but_charged(response):
    policy, client = selector([response], max_calls=1)
    with pytest.raises(ActionError):
        policy({})
    assert policy.calls_used == 1
    assert policy.last_receipt['status'] == 'failed'
    with pytest.raises(ActionError, match='call budget'):
        policy({})
    assert policy.last_receipt['status'] == 'not_called'
    assert policy.last_receipt['usage'] is None
    assert len(client.requests) == 1


def test_reasoning_items_are_not_parsed_or_retained_as_actions():
    response = model_response()
    response['output'].insert(0, {'type': 'reasoning', 'summary': [{'text': 'untrusted non-action'}]})
    policy, _ = selector([response])
    assert policy({}) == {'tool': 'inspect', 'arguments': {}}
    assert policy.last_receipt['output_types'] == ['reasoning', 'message']
    assert 'untrusted non-action' not in json.dumps(policy.last_receipt)


def test_transport_failure_consumes_a_call_without_implicit_retry():
    policy, client = selector([RuntimeError('mock transport failure')], max_calls=1)
    with pytest.raises(RuntimeError, match='transport'):
        policy({})
    assert policy.calls_used == 1
    assert policy.last_receipt['error']['message'] == 'mock transport failure'
    assert len(client.requests) == 1


@pytest.mark.parametrize('kwargs', [{'model': ''}, {'max_calls': 0}, {'max_output_tokens': True}, {'request_timeout_seconds': 0}])
def test_selector_requires_explicit_model_and_bounded_parameters(kwargs):
    with pytest.raises(ValueError):
        selector([], **kwargs)


@pytest.mark.parametrize('action', [None, {'tool': 'shell', 'arguments': {}}, {'tool': {}, 'arguments': {}},
                                     {'tool': 'inspect', 'arguments': [], 'score': 1},
                                     {'tool': 'inspect', 'arguments': {}, 'code': 'forbidden'}])
def test_provider_neutral_action_validation(action):
    with pytest.raises(ActionError):
        validate_action(action)


def test_action_loop_retains_actions_usage_and_stops_after_final_without_sending_private_result(tmp_path):
    def executor(kind, proposal, context):
        return {'cohort_id': 'cpu-fixture', 'status': 'ok', 'private_final_evidence': 'not sent to model'}

    session = make_session(tmp_path, executor)
    proposal_id = session.state['reference_proposal_id']
    policy, client = selector([model_response(), model_response({'tool': 'submit', 'arguments_json': json.dumps({'proposal_id': proposal_id})})])
    report = run_agent(session, policy, observation_builder=public, max_turns=4)
    assert report['status'] == 'session_terminal' and report['state']['status'] == 'completed'
    assert report['turns_used'] == 2
    assert len(client.requests) == 2
    assert 'private_final_evidence' not in json.dumps(client.requests)
    assert report['turns'][1]['action']['tool'] == 'submit'
    assert report['turns'][1]['selector_receipt']['usage']['total_tokens'] == 18
    assert report['turns'][1]['outcome']['response']['result']['private_final_evidence'] == 'not sent to model'
    session.close()


def test_errors_consume_persistent_turn_budget_and_preserve_raw_invalid_action(tmp_path):
    session = make_session(tmp_path)
    actions = iter([{'tool': 'shell', 'arguments': {}}, {'tool': 'propose', 'arguments': {'overrides': {'generation': {'steps': 1}}}}])
    report = run_agent(session, lambda observation: next(actions), observation_builder=public, max_turns=2)
    assert report['status'] == 'turn_budget_exhausted'
    assert [turn['outcome']['ok'] for turn in report['turns']] == [False, False]
    assert report['turns'][0]['raw_action']['tool'] == 'shell'
    assert report['state']['tool_calls_used'] == 1  # Bad shape/tool never reaches session.
    called = []
    resumed = run_agent(session, lambda observation: called.append(1), observation_builder=public, max_turns=2)
    assert resumed['turns_used'] == 2 and called == []
    with pytest.raises(SessionError, match='cannot be reset'):
        run_agent(session, lambda observation: None, observation_builder=public, max_turns=3)
    session.close()


def test_interrupted_model_turn_is_not_retried_on_restart(tmp_path):
    session = make_session(tmp_path)

    def interrupted(observation):
        raise KeyboardInterrupt('model outcome unknown')

    with pytest.raises(KeyboardInterrupt):
        run_agent(session, interrupted, observation_builder=public, max_turns=2)
    with pytest.raises(SessionError, match='interrupted'):
        run_agent(session, lambda obs: {'tool': 'inspect', 'arguments': {}}, observation_builder=public, max_turns=2)
    session.close()


def test_tampered_turn_receipt_and_changed_selector_cannot_resume(tmp_path):
    session = make_session(tmp_path)
    run_agent(session, lambda obs: {'tool': 'inspect', 'arguments': {}}, observation_builder=public,
              max_turns=1, selector_metadata={'model': 'pinned'})
    with pytest.raises(SessionError, match='policy changed'):
        run_agent(session, lambda obs: None, observation_builder=public, max_turns=1, selector_metadata={'model': 'changed'})
    receipt = session.directory / 'agent-actions' / '00000001-finished.json'
    doc = json.loads(receipt.read_text()); doc['action']['tool'] = 'submit'
    receipt.chmod(0o600); receipt.write_text(json.dumps(doc))
    with pytest.raises(SessionError, match='hash mismatch'):
        run_agent(session, lambda obs: None, observation_builder=public, max_turns=1, selector_metadata={'model': 'pinned'})
    session.close()


def test_observation_projection_is_explicit_and_model_only_receives_a_copy(tmp_path):
    session = make_session(tmp_path)
    seen = []

    def next_action(observation):
        seen.append(deepcopy(observation))
        observation['nested']['safe'] = False
        return {'tool': 'inspect', 'arguments': {}}

    data = {'nested': {'safe': True}}
    report = run_agent(session, next_action, observation_builder=lambda state, outcome: data, max_turns=1)
    assert seen == [{'nested': {'safe': True}}]
    assert data['nested']['safe'] is True
    started = json.loads((session.directory / 'agent-actions' / '00000001-started.json').read_text())
    assert started['public_observation'] == data
    assert report['gpu_validation_claim'] is False
    session.close()


def test_expired_reopened_session_never_invokes_the_selector(tmp_path):
    clock = SimpleNamespace(now=1000.)
    session = make_session(tmp_path, clock=lambda: clock.now)
    pinned = session.policy
    session.close()
    clock.now = 1101.
    calls = []
    with AgentSession.open(tmp_path / 'session', executor=lambda *args: None,
                           expected_policy=pinned, clock=lambda: clock.now) as resumed:
        report = run_agent(resumed, lambda obs: calls.append(obs), observation_builder=public, max_turns=2)
        assert report['status'] == 'session_terminal'
        assert report['state']['status'] == 'expired'
        assert report['state']['deadline_unix'] == 1100.
        assert report['turns_used'] == report['state']['tool_calls_used'] == 0
        assert calls == []


def test_slow_observation_builder_cannot_admit_a_post_deadline_model_call(tmp_path):
    clock = SimpleNamespace(now=1000.)
    session = make_session(tmp_path, clock=lambda: clock.now)
    calls = []

    def slow_projection(state, outcome):
        clock.now = 1100.
        return public(state, outcome)

    report = run_agent(session, lambda obs: calls.append(obs), observation_builder=slow_projection, max_turns=1)
    assert report['state']['status'] == 'expired'
    assert report['turns_used'] == 0 and calls == []
    session.close()


def test_responses_timeout_is_capped_by_controller_remaining_time_not_model_observation(tmp_path):
    clock = SimpleNamespace(now=1000.)
    session = make_session(tmp_path, clock=lambda: clock.now)
    clock.now = 1098.5
    policy, client = selector([model_response()], request_timeout_seconds=60)
    report = run_agent(session, policy, observation_builder=lambda *args: {'remaining_seconds': 999999}, max_turns=1)
    assert client.options[-1] == {'max_retries': 0, 'timeout': 1.5}
    assert report['turns'][0]['selector_receipt']['request_timeout_seconds'] == 1.5
    assert report['state']['deadline_unix'] == 1100.
    session.close()


def test_max_calls_rejection_does_not_duplicate_prior_token_usage_in_a_later_turn(tmp_path):
    session = make_session(tmp_path)
    policy, client = selector([model_response()], max_calls=1)
    report = run_agent(session, policy, observation_builder=public, max_turns=2)
    assert len(client.requests) == 1
    first, second = [turn['selector_receipt'] for turn in report['turns']]
    assert first['usage']['total_tokens'] == 18
    assert second['status'] == 'not_called' and second['usage'] is None
    assert 'call_index' not in second
    assert report['turns'][1]['outcome']['ok'] is False
    session.close()


def test_fresh_selector_after_completed_turn_cannot_refund_model_call_budget(tmp_path, monkeypatch):
    import drivebench_tuning.agent_policy as module
    session = make_session(tmp_path)
    policy, client = selector([model_response()], max_calls=1)
    original_write = module._write_new

    def interrupted_after_durable_completion(path, value):
        original_write(path, value)
        if path.name == '00000001-finished.json':
            raise KeyboardInterrupt('controller stopped between completed turns')

    monkeypatch.setattr(module, '_write_new', interrupted_after_durable_completion)
    with pytest.raises(KeyboardInterrupt):
        run_agent(session, policy, observation_builder=public, max_turns=2)
    assert len(client.requests) == 1
    session.close()
    monkeypatch.setattr(module, '_write_new', original_write)
    fresh_policy, fresh_client = selector([], max_calls=1)
    with AgentSession.open(tmp_path / 'session', executor=lambda *args: None, clock=lambda: 1000.) as resumed:
        report = run_agent(resumed, fresh_policy, observation_builder=public, max_turns=2)
        assert report['turns_used'] == 2 and fresh_policy.calls_used == 1
        assert fresh_client.requests == []
        assert report['turns'][1]['selector_receipt']['status'] == 'not_called'
        assert report['turns'][1]['selector_receipt']['usage'] is None


def test_provider_attempts_outside_the_durable_loop_cannot_be_erased(tmp_path):
    session = make_session(tmp_path)
    policy, client = selector([model_response()])
    policy({})
    with pytest.raises(SessionError, match='durable receipts'):
        run_agent(session, policy, observation_builder=public, max_turns=2)
    assert len(client.requests) == 1 and policy.calls_used == 1
    session.close()


def test_clock_rollback_after_failed_model_only_turn_blocks_another_selector_and_reopen(tmp_path, monkeypatch):
    import drivebench_tuning.agent_policy as module
    clock = SimpleNamespace(now=1000.)
    session = make_session(tmp_path, clock=lambda: clock.now)
    calls = []
    original_write = module._write_new

    def failed_selector(observation):
        calls.append(observation)
        clock.now = 1002.
        raise RuntimeError('mock failure before any session tool')

    def rollback_after_receipt(path, value):
        original_write(path, value)
        if path.name == '00000001-finished.json':
            clock.now = 1001.

    monkeypatch.setattr(module, '_write_new', rollback_after_receipt)
    with pytest.raises(SessionError, match='clock moved backwards'):
        run_agent(session, failed_selector, observation_builder=public, max_turns=2)
    assert len(calls) == 1 and session.state['tool_calls_used'] == 0
    assert session.state['last_timestamp'] == 1000.
    session.close()
    with AgentSession.open(tmp_path / 'session', executor=lambda *args: None, clock=lambda: clock.now) as resumed:
        with pytest.raises(SessionError, match='clock moved backwards'):
            run_agent(resumed, failed_selector, observation_builder=public, max_turns=2)
        assert len(calls) == 1


def test_invalid_failed_selector_completion_clock_leaves_an_interrupted_turn(tmp_path):
    clock = SimpleNamespace(now=1000.)
    session = make_session(tmp_path, clock=lambda: clock.now)

    def rollback(observation):
        clock.now = 999.
        raise RuntimeError('mock failure')

    with pytest.raises(SessionError, match='clock moved backwards'):
        run_agent(session, rollback, observation_builder=public, max_turns=2)
    assert not (session.directory / 'agent-actions' / '00000001-finished.json').exists()
    clock.now = 1001.
    with pytest.raises(SessionError, match='interrupted'):
        run_agent(session, lambda obs: None, observation_builder=public, max_turns=2)
    session.close()


def test_replay_rejects_nonmonotonic_action_time_even_with_recomputed_local_hashes(tmp_path):
    from drivebench_tuning.agent_session import _hash
    session = make_session(tmp_path)
    run_agent(session, lambda obs: {'tool': 'inspect', 'arguments': {}}, observation_builder=public, max_turns=1)
    started_path = session.directory / 'agent-actions' / '00000001-started.json'
    finished_path = session.directory / 'agent-actions' / '00000001-finished.json'
    started, finished = json.loads(started_path.read_text()), json.loads(finished_path.read_text())
    started['at'] = 1002.
    started['sha256'] = _hash({k: v for k, v in started.items() if k != 'sha256'})
    finished['started_sha256'] = started['sha256']
    finished['sha256'] = _hash({k: v for k, v in finished.items() if k != 'sha256'})
    for path, doc in ((started_path, started), (finished_path, finished)):
        path.chmod(0o600); path.write_text(json.dumps(doc))
    with pytest.raises(SessionError, match='clock moved backwards'):
        run_agent(session, lambda obs: None, observation_builder=public, max_turns=1)
    session.close()


@pytest.mark.parametrize('remaining', [0, -1, float('inf'), float('nan'), True])
def test_invalid_controller_timeout_never_calls_the_provider(remaining):
    policy, client = selector([])
    with pytest.raises(ActionError, match='remaining wall time'):
        policy.select_action({}, remaining_seconds=remaining)
    assert client.requests == []
    assert policy.calls_used == 0
    assert policy.last_receipt['status'] == 'not_called'
