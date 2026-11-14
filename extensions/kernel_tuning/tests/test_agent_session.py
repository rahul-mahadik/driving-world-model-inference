"""CPU controller contracts only; no agent model, provider or GPU execution."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from drivebench_tuning.agent_session import AgentSession, SessionError, SessionPolicy


class Clock:
    def __init__(self):
        self.now = 1000.

    def __call__(self):
        return self.now


class Executor:
    def __init__(self):
        self.calls = []
        self.fail = False

    def __call__(self, kind, proposal, context):
        self.calls.append((kind, proposal, context))
        if self.fail:
            raise RuntimeError('deliberate trusted fixture failure')
        return {'cohort_id': context['cohort_id'], 'status': 'ok', 'synthetic': True,
                'proposal_id': proposal['proposal_id'], 'quality_established': False}


def policy(**overrides):
    values = dict(session_id='test-session', cohort_id='synthetic-contract-only',
                  reference_config={'generation': {'steps': 50, 'decode_chunk_size': 14},
                                    'optimizations': {'compile_denoiser': False, 'max_batch_size': 1}},
                  allowed_config_values={'optimizations.compile_denoiser': [False, True],
                                         'optimizations.max_batch_size': [1, 2],
                                         'generation.decode_chunk_size': [1, 14]},
                  wall_seconds=100, max_tool_calls=20, max_evaluations=4, max_profiles=2,
                  final_evaluations=2, metadata={'fixture': True}, final_reserve_seconds=20)
    values.update(overrides)
    return SessionPolicy(**values)


def create(tmp_path, **kwargs):
    clock, executor = kwargs.pop('clock', Clock()), kwargs.pop('executor', Executor())
    return AgentSession.create(tmp_path / 'session', policy=policy(**kwargs), executor=executor, clock=clock), executor, clock


def events(directory):
    return [json.loads(path.read_text()) for path in sorted((directory / 'events').glob('*.json'))]


def test_config_tools_execute_only_trusted_callback_and_final_is_irreversible(tmp_path):
    session, executor, clock = create(tmp_path)
    assert session.call('inspect')['result']['policy']['reference_config']['generation']['steps'] == 50
    proposed = session.call('propose', {'overrides': {'optimizations': {'compile_denoiser': True}}})['result']
    assert proposed['config']['optimizations']['compile_denoiser'] is True
    assert proposed['config']['generation']['steps'] == 50
    session.call('profile', {'proposal_id': proposed['proposal_id']})
    session.call('trial', {'proposal_id': proposed['proposal_id']})
    final = session.call('submit', {'proposal_id': proposed['proposal_id']})
    assert final['state']['status'] == 'completed'
    assert final['state']['selected_proposal_id'] == proposed['proposal_id']
    assert final['state']['evaluations_used'] == 3  # One dev call + two reserved final units.
    assert [call[0] for call in executor.calls] == ['profile', 'trial', 'final']
    assert [call[2]['evaluation_units'] for call in executor.calls] == [0, 1, 2]
    for tool in ('inspect', 'propose', 'trial', 'submit'):
        with pytest.raises(SessionError, match='terminal'):
            session.call(tool)
    saved = session.state
    session.close()
    resumed = AgentSession.open(session.directory, executor=executor, expected_policy=policy(), clock=clock)
    assert resumed.state == saved
    with pytest.raises(SessionError, match='terminal'):
        resumed.call('submit', {'proposal_id': proposed['proposal_id']})
    resumed.close()


def test_restart_replays_exact_budget_deadline_and_latest_config_without_refund(tmp_path):
    session, executor, clock = create(tmp_path)
    first = session.call('propose', {'overrides': {'optimizations': {'max_batch_size': 2}}})['result']
    second = session.call('propose', {'overrides': {'generation': {'decode_chunk_size': 1}}})['result']
    assert second['parent_id'] == first['proposal_id']
    assert second['config']['optimizations']['max_batch_size'] == 2
    session.call('trial', {'proposal_id': second['proposal_id']})
    saved = session.state
    session.close()
    clock.now += 30
    with AgentSession.open(tmp_path / 'session', executor=executor, expected_policy=policy(), clock=clock) as resumed:
        assert resumed.state == saved
        assert resumed.call('inspect')['result']['latest_proposal'] == second
        assert resumed.state['deadline_unix'] == 1100
        assert resumed.state['evaluations_used'] == 1
        assert resumed.state['tool_calls_used'] == 4


@pytest.mark.parametrize('tool,args', [
    ('shell', {'command': 'echo forbidden'}),
    ('propose', {'code': 'print(1)', 'overrides': {}}),
    ('propose', {'overrides': {'files': {'kernel.py': 'code'}}}),
    ('propose', {'overrides': {'generation': {'steps': 1}}}),
    ('propose', {'overrides': {'optimizations': {'compile_denoiser': 1}}}),
    ('propose', {'overrides': {}, 'path': '../../private'}),
    ('trial', {'proposal_id': '../../private'}),
    ('inspect', {'path': '/private'}),
])
def test_code_paths_unlisted_config_and_type_coercion_rejected_and_charged(tmp_path, tool, args):
    session, executor, _ = create(tmp_path)
    with pytest.raises((ValueError, SessionError)):
        session.call(tool, args)
    assert session.state['tool_calls_used'] == 1
    assert session.state['evaluations_used'] == 0
    assert not executor.calls
    assert events(session.directory)[-1]['type'] == 'tool_failed'
    session.close()


def test_agent_cannot_supply_its_own_measurement_score_or_executor(tmp_path):
    session, executor, _ = create(tmp_path)
    for forged in ({'score': 999}, {'result': {'cohort_id': policy().cohort_id}}, {'executor': 'arbitrary_module'}):
        with pytest.raises(ValueError, match='measurements cannot be supplied'):
            session.call('trial', {'proposal_id': session.state['reference_proposal_id'], **forged})
    assert len(executor.calls) == 0
    assert session.state['tool_calls_used'] == 3
    session.close()


def test_failed_execution_charges_units_and_survives_restart(tmp_path):
    session, executor, clock = create(tmp_path)
    executor.fail = True
    with pytest.raises(RuntimeError, match='fixture failure'):
        session.call('trial', {'proposal_id': session.state['reference_proposal_id']})
    assert session.state['evaluations_used'] == 1
    session.close()
    with AgentSession.open(tmp_path / 'session', executor=executor, expected_policy=policy(), clock=clock) as resumed:
        assert resumed.state['evaluations_used'] == 1
        assert resumed.state['tool_calls_used'] == 1
        assert events(resumed.directory)[-1]['payload']['error']['type'] == 'RuntimeError'


def test_development_cannot_spend_reserved_final_units_or_tool_slot(tmp_path):
    session, executor, _ = create(tmp_path, max_tool_calls=4)
    args = {'proposal_id': session.state['reference_proposal_id']}
    session.call('trial', args)
    session.call('trial', args)
    with pytest.raises(SessionError, match='final evaluation reserved'):
        session.call('trial', args)
    assert session.state['tool_calls_used'] == 3
    assert session.state['evaluations_used'] == 2
    with pytest.raises(SessionError, match='submission slot reserved'):
        session.call('inspect')
    assert session.call('submit', args)['state']['evaluations_used'] == 4
    assert len(executor.calls) == 3
    session.close()


def test_profiles_have_an_independent_fixed_budget(tmp_path):
    session, executor, _ = create(tmp_path, max_profiles=1)
    args = {'proposal_id': session.state['reference_proposal_id']}
    session.call('profile', args)
    with pytest.raises(SessionError, match='profile budget exhausted'):
        session.call('profile', args)
    assert session.state['profiles_used'] == 1
    assert session.state['evaluations_used'] == 0
    assert session.state['tool_calls_used'] == 2
    assert len(executor.calls) == 1
    session.close()


def test_development_wall_window_is_reserved_for_final(tmp_path):
    session, executor, clock = create(tmp_path)
    args = {'proposal_id': session.state['reference_proposal_id']}
    clock.now = 1080.
    with pytest.raises(SessionError, match='final window reserved'):
        session.call('trial', args)
    result = session.call('submit', args)
    assert result['state']['status'] == 'completed'
    assert executor.calls[0][2]['remaining_seconds'] == 20
    session.close()


def test_late_executor_receipt_is_preserved_but_not_accepted(tmp_path):
    clock = Clock()

    def late(kind, proposal, context):
        assert context['execution_deadline_unix'] == 1080
        clock.now = 1081
        return {'cohort_id': context['cohort_id'], 'status': 'ok', 'fixture': 'too late'}

    session, _, _ = create(tmp_path, executor=late, clock=clock)
    with pytest.raises(SessionError, match='immutable wall deadline'):
        session.call('trial', {'proposal_id': session.state['reference_proposal_id']})
    assert session.state['evaluations_used'] == 1
    assert events(session.directory)[-1]['payload']['executor_result']['fixture'] == 'too late'
    session.close()


@pytest.mark.parametrize('tool', ['trial', 'profile'])
def test_recorded_completion_timestamp_enforces_development_cutoff_on_replay(tmp_path, tool):
    session, executor, _ = create(tmp_path)
    # Admission, execution start, executor context and pre-return check are all
    # before the cutoff. The authoritative completion-event timestamp is not.
    ticks = iter([1079., 1079., 1079., 1079., 1080., 1080.])
    session.clock = lambda: next(ticks)
    with pytest.raises(SessionError, match='immutable wall deadline'):
        session.call(tool, {'proposal_id': session.state['reference_proposal_id']})
    assert events(session.directory)[-1]['type'] == 'tool_failed'
    assert events(session.directory)[-1]['payload']['executor_result']['status'] == 'ok'
    session.close()
    with AgentSession.open(tmp_path / 'session', executor=executor, expected_policy=policy(), clock=lambda: 1080.) as resumed:
        assert resumed.state['last_result']['successful_tool'] is False
        assert resumed.state['tool_calls_used'] == 1
        assert resumed.state['evaluations_used'] == int(tool == 'trial')


@pytest.mark.parametrize('tool, cutoff', [('trial', 1080.), ('profile', 1080.), ('submit', 1100.)])
def test_cutoff_crossed_before_executor_dispatch_is_charged_without_execution(tmp_path, tool, cutoff):
    session, executor, _ = create(tmp_path)
    ticks = iter([cutoff - 1, cutoff - 1, cutoff, cutoff])
    session.clock = lambda: next(ticks)
    with pytest.raises(SessionError, match='execution dispatch'):
        session.call(tool, {'proposal_id': session.state['reference_proposal_id']})
    assert executor.calls == []
    assert session.state['tool_calls_used'] == 1
    assert session.state['evaluations_used'] == (2 if tool == 'submit' else int(tool == 'trial'))
    assert session.state['profiles_used'] == int(tool == 'profile')
    assert events(session.directory)[-1]['type'] == 'tool_failed'
    session.close()


@pytest.mark.parametrize('result_status, expected', [('ok', 'completed'), ('pass', 'completed'),
    ('fail', 'final_failed'), ('inconclusive', 'final_failed'), ('error', 'final_failed'),
    ('incorrect', 'final_failed'), ('failed', 'final_failed')])
def test_explicit_final_score_status_controls_acceptance_and_stays_locked(tmp_path, result_status, expected):
    session, executor, clock = create(tmp_path, executor=lambda kind, proposal, context:
        {'cohort_id': context['cohort_id'], 'status': result_status})
    result = session.call('submit', {'proposal_id': session.state['reference_proposal_id']})
    assert result['state']['status'] == expected
    session.close()
    with AgentSession.open(tmp_path / 'session', executor=executor, clock=clock) as resumed:
        assert resumed.state['status'] == expected
        with pytest.raises(SessionError, match='terminal'):
            resumed.call('submit', {'proposal_id': resumed.state['reference_proposal_id']})


@pytest.mark.parametrize('result_status', [None, 'unknown'])
def test_missing_or_unrecognized_final_score_is_rejected_with_raw_receipt(tmp_path, result_status):
    receipt = {'cohort_id': policy().cohort_id}
    if result_status is not None:
        receipt['status'] = result_status
    session, _, _ = create(tmp_path, executor=lambda *args: receipt)
    with pytest.raises(SessionError, match='recognized status'):
        session.call('submit', {'proposal_id': session.state['reference_proposal_id']})
    assert session.state['status'] == 'final_failed'
    assert session.state['evaluations_used'] == policy().final_evaluations
    assert events(session.directory)[-1]['payload']['executor_result'] == receipt
    session.close()


def test_expiry_and_clock_rollback_fail_closed_without_refreshing_deadline(tmp_path):
    session, executor, clock = create(tmp_path)
    clock.now = 999.
    with pytest.raises(SessionError, match='backwards'):
        session.call('inspect')
    clock.now = 1100.
    with pytest.raises(SessionError, match='deadline exhausted'):
        session.call('inspect')
    assert session.state['status'] == 'expired'
    session.close()
    with AgentSession.open(tmp_path / 'session', executor=executor, clock=clock) as resumed:
        assert resumed.state['status'] == 'expired'
        assert resumed.state['deadline_unix'] == 1100


@pytest.mark.parametrize('final', [False, True])
def test_interrupted_execution_is_irrecoverable_and_never_retried(tmp_path, final):
    def interrupted(*args):
        raise KeyboardInterrupt('simulated process interruption')

    session, _, clock = create(tmp_path, executor=interrupted)
    selected = session.state['reference_proposal_id']
    with pytest.raises(KeyboardInterrupt):
        session.call('submit' if final else 'trial', {'proposal_id': selected})
    consumed = session.state['evaluations_used']
    assert session.state['pending'] is not None
    session.close()
    executor = Executor()
    with AgentSession.open(tmp_path / 'session', executor=executor, expected_policy=policy(), clock=clock) as resumed:
        assert resumed.state['status'] == 'aborted'
        assert resumed.state['evaluations_used'] == consumed == (2 if final else 1)
        assert resumed.state['selected_proposal_id'] == (selected if final else None)
        with pytest.raises(SessionError, match='terminal'):
            resumed.call('submit', {'proposal_id': selected})
        assert executor.calls == []
        assert events(resumed.directory)[-1]['type'] == 'recovery_aborted'


def test_submission_is_durably_locked_before_executor_and_failure_does_not_reopen(tmp_path):
    observed = []
    session = None

    def final_failure(kind, proposal, context):
        observed.append(session.state)
        assert session.state['status'] == 'submitted'
        assert events(session.directory)[-1]['type'] == 'execution_started'
        raise RuntimeError('final failed')

    session, _, clock = create(tmp_path, executor=final_failure)
    args = {'proposal_id': session.state['reference_proposal_id']}
    with pytest.raises(RuntimeError, match='final failed'):
        session.call('submit', args)
    assert session.state['status'] == 'final_failed'
    assert session.state['evaluations_used'] == 2
    with pytest.raises(SessionError, match='terminal'):
        session.call('submit', args)
    session.close()


@pytest.mark.parametrize('value', [None, {'cohort_id': 'forged'}, {'cohort_id': 'synthetic-contract-only', 'score': float('nan')}])
def test_invalid_trusted_executor_response_is_charged_and_not_a_success(tmp_path, value):
    session, _, _ = create(tmp_path, executor=lambda *args: value)
    with pytest.raises(ValueError):
        session.call('trial', {'proposal_id': session.state['reference_proposal_id']})
    assert session.state['evaluations_used'] == 1
    assert session.state['last_result']['successful_tool'] is False
    session.close()


@pytest.mark.parametrize('target', ['event', 'snapshot', 'policy', 'missing_event', 'orphan_proposal'])
def test_disk_tampering_and_partial_artifacts_prevent_reopening(tmp_path, target):
    session, executor, clock = create(tmp_path)
    proposal_id = session.call('propose', {'overrides': {}})['result']['proposal_id']
    directory = session.directory
    session.close()
    if target == 'event':
        path = directory / 'events' / '00000001.json'
        doc = json.loads(path.read_text()); doc['state']['evaluations_used'] = 0xFFFF
    elif target == 'snapshot':
        path = directory / 'proposals' / (proposal_id + '.json')
        doc = json.loads(path.read_text()); doc['config']['generation']['steps'] = 1
    elif target == 'policy':
        path = directory / 'policy.json'
        doc = json.loads(path.read_text()); doc['deadline_unix'] += 500
    elif target == 'missing_event':
        (directory / 'events' / '00000002.json').unlink()
    else:
        (directory / 'proposals' / ('a' * 64 + '.json')).write_text('{}')
    if target in ('event', 'snapshot', 'policy'):
        path.chmod(0o600); path.write_text(json.dumps(doc))
    with pytest.raises(SessionError):
        AgentSession.open(directory, executor=executor, expected_policy=policy(), clock=clock)


def test_external_policy_pin_rejects_a_new_budget_or_cohort(tmp_path):
    session, executor, clock = create(tmp_path)
    session.close()
    for changed in (replace(policy(), max_evaluations=5), replace(policy(), cohort_id='different')):
        with pytest.raises(SessionError, match='caller-pinned policy'):
            AgentSession.open(tmp_path / 'session', executor=executor, expected_policy=changed, clock=clock)


def test_concurrent_controller_and_overwriting_session_are_refused(tmp_path):
    session, executor, clock = create(tmp_path)
    with pytest.raises(SessionError, match='another controller'):
        AgentSession.open(session.directory, executor=executor, clock=clock)
    with pytest.raises(FileExistsError):
        AgentSession.create(session.directory, policy=policy(), executor=executor, clock=clock)
    session.close()


def test_policy_state_proposals_and_executor_inputs_cannot_mutate_controller_state(tmp_path):
    original = policy()
    with pytest.raises(TypeError):
        original.reference_config['generation']['steps'] = 1

    def mutating_executor(kind, proposal, context):
        proposal['config']['generation']['steps'] = 1
        context['metadata']['fixture'] = False
        return {'cohort_id': context['cohort_id'], 'status': 'ok'}

    session, _, _ = create(tmp_path, executor=mutating_executor)
    snapshot = session.call('inspect')['result']['latest_proposal']
    snapshot['config']['generation']['steps'] = 1
    session.call('trial', {'proposal_id': snapshot['proposal_id']})
    assert session.call('inspect')['result']['latest_proposal']['config']['generation']['steps'] == 50
    state = session.state; state['evaluations_used'] = 0
    assert session.state['evaluations_used'] == 1
    session.close()
