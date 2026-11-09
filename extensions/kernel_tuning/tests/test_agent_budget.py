"""CPU-only cost-reservation contracts; prices/model/transport are test fixtures."""
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
import json
from types import SimpleNamespace

import pytest

from drivebench_tuning.agent_budget import APICostLedger, APICostPolicy, BudgetedResponsesClient, CostBudgetError
from drivebench_tuning.agent_policy import ResponsesActionPolicy, run_agent
from drivebench_tuning.agent_session import AgentSession, SessionError, SessionPolicy, _bytes


def policy(**overrides):
    values = dict(budget_id='cpu-cost-fixture', model='fixture-model-not-selected-for-use',
        input_usd_per_million_tokens='2', output_usd_per_million_tokens='10', hard_usd_cap='0.01',
        max_output_tokens=100, input_bound_kind='explicit_upper_bound',
        input_bound_id='caller-asserted fixture bound: full transport bytes plus 100 framing tokens')
    values.update(overrides)
    return APICostPolicy(**values)


def byte_bound(request):
    return len(_bytes(request)) + 100


def request(**overrides):
    values = dict(model=policy().model, instructions='Fixed fixture instructions',
        input=[{'role': 'user', 'content': '{"public":"fixture"}'}], tools=[], store=False, service_tier='default', max_output_tokens=100,
        text={'format': {'type': 'json_schema', 'name': 'drivebench_action', 'strict': True,
                         'schema': {'type': 'object', 'properties': {}}}})
    values.update(overrides)
    return values


def response(usage=None):
    return {'id': 'response_fixture', 'model': policy().model, 'status': 'completed',
        'usage': usage if usage is not None else {'input_tokens': 20, 'output_tokens': 10, 'total_tokens': 30,
                                                'output_tokens_details': {'reasoning_tokens': 7}},
        'output': [{'type': 'message', 'role': 'assistant', 'status': 'completed',
                    'content': [{'type': 'output_text', 'text': '{"tool":"inspect","arguments_json":"{}"}'}]}]}


class Client:
    def __init__(self, outcomes):
        self.outcomes, self.requests, self.options = list(outcomes), [], []
        self.responses = SimpleNamespace(create=self.create)

    def with_options(self, **options):
        self.options.append(options)
        return self

    def create(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        result = self.outcomes.pop(0)
        if isinstance(result, BaseException):
            raise result
        return deepcopy(result)


def make(tmp_path, outcomes=None, *, cost_policy=None, bound=byte_bound):
    ledger = APICostLedger.create(tmp_path / 'cost', policy=cost_policy or policy(), input_token_bound=bound)
    client = Client([response()] if outcomes is None else outcomes)
    return ledger, client, BudgetedResponsesClient(client, ledger)


def test_exact_transport_including_instructions_schema_and_unicode_is_reserved_before_send(tmp_path):
    captured = []
    ledger = None

    def bound(req):
        captured.append(deepcopy(req))
        req['instructions'] = 'callback mutation must not affect request'
        return byte_bound(captured[-1])

    ledger, client, wrapped = make(tmp_path, bound=bound)
    original = client.responses.create

    def checked(**kwargs):
        assert ledger.state['pending']['request_sha256']
        assert ledger.state['attempts'] == 1
        assert len(list((ledger.directory / 'events').glob('*.json'))) == 1
        return original(**kwargs)

    client.responses.create = checked
    sent = request(input=[{'role': 'user', 'content': '{"unicode":"🚗雪"}'}])
    wrapped.responses.create(**sent)
    expected = policy().reservation_usd(byte_bound(sent), 100)
    assert captured == client.requests == [sent]
    assert ledger.state['charged_usd'] == expected
    assert ledger.state['last_receipt']['usage_status'] == 'reported'
    # Input caching/short visible output do not refund either token reservation.
    assert Decimal(expected) > Decimal(policy().reservation_usd(20, 10))
    assert ledger.state['charges_are_full_reservations'] is True
    assert ledger.state['provider_invoice_verified'] is False
    ledger.close()


def test_hard_cap_rejects_next_request_and_survives_fresh_client_restart(tmp_path):
    req = request()
    one = policy().reservation_usd(byte_bound(req), 100)
    frozen = policy(hard_usd_cap=one)
    ledger, client, wrapped = make(tmp_path, cost_policy=frozen)
    wrapped.responses.create(**req)
    assert ledger.state['remaining_usd'] == '0'
    with pytest.raises(CostBudgetError, match='cap would be exceeded'):
        wrapped.responses.create(**req)
    assert len(client.requests) == 1
    ledger.close()
    with APICostLedger.open(tmp_path / 'cost', expected_policy=frozen, input_token_bound=byte_bound) as resumed:
        fresh = Client([])
        with pytest.raises(CostBudgetError, match='cap would be exceeded'):
            BudgetedResponsesClient(fresh, resumed).responses.create(**req)
        assert fresh.requests == [] and resumed.state['attempts'] == 1


@pytest.mark.parametrize('outcome', [RuntimeError('fixture network error'), response({}),
    response({'input_tokens': 20, 'output_tokens': 10}),
    response({'input_tokens': True, 'output_tokens': 1, 'total_tokens': 2}),
    response({'input_tokens': 20, 'output_tokens': 10, 'total_tokens': 999}),
    {'id': 'missing_usage', 'status': 'failed'}, {'status': 'incomplete', 'usage': None}])
def test_failed_and_unknown_usage_attempts_keep_full_charge(tmp_path, outcome):
    ledger, client, wrapped = make(tmp_path, [outcome])
    if isinstance(outcome, Exception):
        with pytest.raises(RuntimeError, match='network error'):
            wrapped.responses.create(**request())
    else:
        wrapped.responses.create(**request())
    assert ledger.state['charged_usd'] == policy().reservation_usd(byte_bound(request()), 100)
    assert ledger.state['attempts'] == 1 and ledger.state['pending'] is None
    assert ledger.state['last_receipt']['usage_status'] == 'unknown'
    assert len(client.requests) == 1
    ledger.close()


def test_interrupted_transport_is_fully_charged_and_halts_on_open(tmp_path):
    ledger, client, wrapped = make(tmp_path, [KeyboardInterrupt('unknown provider outcome')])
    with pytest.raises(KeyboardInterrupt):
        wrapped.responses.create(**request())
    saved = ledger.state['charged_usd']
    assert ledger.state['pending'] is not None
    ledger.close()
    with APICostLedger.open(tmp_path / 'cost', expected_policy=policy(), input_token_bound=byte_bound) as resumed:
        assert resumed.state['charged_usd'] == saved and resumed.state['halted'] is True
        assert resumed.state['last_receipt']['outcome'] == 'interrupted'
        with pytest.raises(CostBudgetError, match='halted'):
            BudgetedResponsesClient(Client([]), resumed).responses.create(**request())


@pytest.mark.parametrize('usage', [
    {'input_tokens': 999999, 'output_tokens': 1, 'total_tokens': 1000000},
    {'input_tokens': 1, 'output_tokens': 101, 'total_tokens': 102,
     'output_tokens_details': {'reasoning_tokens': 100}},
    {'input_tokens': 999999},  # Partial usage cannot hide a known bound violation.
])
def test_reported_token_bound_violation_halts_future_calls_without_refund(tmp_path, usage):
    ledger, client, wrapped = make(tmp_path, [response(usage)])
    with pytest.raises(CostBudgetError, match='exceeded the reserved bounds'):
        wrapped.responses.create(**request())
    assert ledger.state['halted'] and ledger.state['last_receipt']['bound_violation']
    assert ledger.state['charged_usd'] == policy().reservation_usd(byte_bound(request()), 100)
    with pytest.raises(CostBudgetError, match='halted'):
        wrapped.responses.create(**request())
    assert len(client.requests) == 1
    ledger.close()


@pytest.mark.parametrize('bound', [None, lambda request: 1, lambda request: True, lambda request: float('inf')])
def test_no_implicit_token_estimate_or_undercounted_byte_floor(tmp_path, bound):
    if bound is None:
        with pytest.raises(ValueError, match='trusted input-token bound'):
            make(tmp_path, bound=None)
        return
    ledger, client, wrapped = make(tmp_path, bound=bound)
    with pytest.raises((ValueError, CostBudgetError)):
        wrapped.responses.create(**request())
    assert client.requests == [] and ledger.state['charged_usd'] == '0'
    ledger.close()


def test_caller_verified_tokenizer_contract_can_supply_tighter_bound(tmp_path):
    frozen = policy(input_bound_kind='verified_tokenizer', input_bound_id='offline verified counter fixture')
    ledger, client, wrapped = make(tmp_path, cost_policy=frozen, bound=lambda request: 20)
    wrapped.responses.create(**request())
    assert ledger.state['charged_usd'] == frozen.reservation_usd(20, 100)
    ledger.close()


def test_full_verified_context_bound_covers_framing_and_all_reasoning_output_without_tokenizer(tmp_path):
    frozen = policy(input_bound_kind='verified_context_limit', input_token_limit=400000,
        input_bound_id='caller verified fixed context-limit fixture', max_output_tokens=1024,
        input_usd_per_million_tokens='0.75', output_usd_per_million_tokens='4.50', hard_usd_cap='5')
    ledger = APICostLedger.create(tmp_path / 'cost', policy=frozen)
    client = Client([response()] * 16)
    wrapped = BudgetedResponsesClient(client, ledger)
    # UTF-8 text can be larger than the model's token context limit; the
    # caller-verified provider context limit, not a text-length guess, is used.
    req = request(max_output_tokens=1024, input=[{'role': 'user', 'content': 'x' * 400001}])
    for _ in range(16):
        wrapped.responses.create(**req)
    assert frozen.reservation_usd(400000, 1024) == '0.304608'
    assert ledger.state['charged_usd'] == '4.873728'
    assert ledger.state['remaining_usd'] == '0.126272'
    with pytest.raises(CostBudgetError, match='cap would be exceeded'):
        wrapped.responses.create(**req)
    assert len(client.requests) == 16
    ledger.close()


@pytest.mark.parametrize('field, value', [('service_tier', 'priority'), ('model', 'different-snapshot')])
def test_returned_model_or_tier_mismatch_halts_further_requests_and_keeps_full_charge(tmp_path, field, value):
    result = response(); result[field] = value
    ledger, client, wrapped = make(tmp_path, [result])
    with pytest.raises(CostBudgetError, match='further API calls halted'):
        wrapped.responses.create(**request())
    assert ledger.state['halted'] is True
    assert ledger.state['last_receipt']['returned_' + field] == value
    assert ledger.state['charged_usd'] == policy().reservation_usd(byte_bound(request()), 100)
    ledger.close()
    with APICostLedger.open(tmp_path / 'cost', expected_policy=policy(), input_token_bound=byte_bound) as reopened:
        assert reopened.state['halted'] is True


@pytest.mark.parametrize('custom_metadata', [None, {'experiment': 'ordinary caller label'},
                                           {'cost_binding': {'ledger_id': 'forged replacement'}}])
def test_restarting_agent_loop_with_fresh_cost_ledger_is_rejected_before_transport(tmp_path, custom_metadata):
    ledger, client, wrapped = make(tmp_path)
    selector = ResponsesActionPolicy(wrapped, model=policy().model, max_calls=1, max_output_tokens=100)
    session_policy = SessionPolicy(session_id='cost-loop-binding', cohort_id='cpu',
        reference_config={'optimizations': {'compile_denoiser': False}},
        allowed_config_values={'optimizations.compile_denoiser': [False, True]},
        wall_seconds=100, max_tool_calls=3, max_evaluations=1, max_profiles=0)
    with AgentSession.create(tmp_path / 'session', policy=session_policy,
                             executor=lambda *args: None, clock=lambda: 1000.) as session:
        report = run_agent(session, selector, observation_builder=lambda *args: {}, max_turns=1,
                           selector_metadata=custom_metadata)
        assert report['turns_used'] == 1 and len(client.requests) == 1
    old_binding = selector.configuration['cost_binding']
    ledger.close()
    replacement = APICostLedger.create(tmp_path / 'fresh-cost', policy=policy(), input_token_bound=byte_bound)
    fresh_client = Client([])
    fresh_selector = ResponsesActionPolicy(BudgetedResponsesClient(fresh_client, replacement),
        model=policy().model, max_calls=1, max_output_tokens=100)
    assert fresh_selector.configuration['cost_binding']['ledger_id'] != old_binding['ledger_id']
    with AgentSession.open(tmp_path / 'session', executor=lambda *args: None, clock=lambda: 1000.) as session:
        with pytest.raises(SessionError, match='loop policy changed'):
            run_agent(session, fresh_selector, observation_builder=lambda *args: {}, max_turns=1,
                      selector_metadata=custom_metadata)
    assert fresh_client.requests == []
    replacement.close()


@pytest.mark.parametrize('override', [
    {'model': 'changed-model'}, {'tools': [{'type': 'web_search'}]}, {'store': True}, {'service_tier': 'priority'},
    {'previous_response_id': 'hidden-context'}, {'stream': True}, {'max_output_tokens': 101},
    {'input': [{'role': 'user', 'content': [{'type': 'input_image', 'image_url': 'hidden'}]}]},
    {'instructions': None}, {'text': {'format': {'type': 'text'}}},
    {'input': [{'role': 'user', 'content': 'x' * 1048577}]},
])
def test_hidden_context_tools_or_unbounded_transport_rejected_before_provider(tmp_path, override):
    ledger, client, wrapped = make(tmp_path)
    with pytest.raises((ValueError, CostBudgetError)):
        wrapped.responses.create(**request(**override))
    assert client.requests == [] and ledger.state['attempts'] == 0
    ledger.close()


@pytest.mark.parametrize('change', [{'hard_usd_cap': '10'}, {'input_usd_per_million_tokens': '1'},
                                   {'output_usd_per_million_tokens': '1'}, {'input_bound_id': 'changed'}])
def test_restart_cannot_change_frozen_prices_cap_or_bound_contract(tmp_path, change):
    ledger, _, wrapped = make(tmp_path)
    wrapped.responses.create(**request())
    ledger.close()
    with pytest.raises(CostBudgetError, match='caller-pinned policy'):
        APICostLedger.open(tmp_path / 'cost', expected_policy=policy(**change), input_token_bound=byte_bound)


def test_tampered_ledger_and_recreated_directory_cannot_refund_charge(tmp_path):
    ledger, _, wrapped = make(tmp_path)
    wrapped.responses.create(**request())
    path = ledger.directory / 'events' / '00000001.json'
    event = json.loads(path.read_text()); event['payload']['reservation_usd'] = '0'
    path.chmod(0o600); path.write_text(json.dumps(event))
    with pytest.raises(CostBudgetError, match='hash chain'):
        wrapped.responses.create(**request())
    ledger.close()
    with pytest.raises(FileExistsError):
        APICostLedger.create(tmp_path / 'cost', policy=policy(), input_token_bound=byte_bound)


def test_exclusive_ownership_and_transport_retries_cannot_duplicate_spending(tmp_path):
    ledger, client, wrapped = make(tmp_path)
    with pytest.raises(CostBudgetError, match='another controller'):
        APICostLedger.open(tmp_path / 'cost', expected_policy=policy(), input_token_bound=byte_bound)
    with pytest.raises(CostBudgetError, match='retries'):
        wrapped.with_options(max_retries=1)
    with pytest.raises(CostBudgetError, match='overrides'):
        wrapped.with_options(base_url='other-provider')
    clone = wrapped.with_options(max_retries=0, timeout=1)
    assert clone.ledger is ledger
    clone.responses.create(**request())
    assert ledger.state['attempts'] == 1 and all(item['max_retries'] == 0 for item in client.options)
    ledger.close()


def test_decimal_reservation_arithmetic_does_not_round_down_at_cap_boundary():
    frozen = policy(input_usd_per_million_tokens='0.000000000000000001',
                    output_usd_per_million_tokens='0.000000000000000002', hard_usd_cap='0.000000000000000001')
    assert Decimal(frozen.reservation_usd(3, 7)) == Decimal('0.000000000000000000000017')


def test_existing_selector_and_run_agent_use_cost_wrapper_without_policy_source_changes(tmp_path):
    frozen = policy(hard_usd_cap='0.004')
    ledger, client, wrapped = make(tmp_path, [response()], cost_policy=frozen)
    selector = ResponsesActionPolicy(wrapped, model=frozen.model, max_calls=2, max_output_tokens=100)
    session_policy = SessionPolicy(session_id='cost-loop', cohort_id='cpu',
        reference_config={'optimizations': {'compile_denoiser': False}},
        allowed_config_values={'optimizations.compile_denoiser': [False, True]},
        wall_seconds=100, max_tool_calls=3, max_evaluations=1, max_profiles=0)
    with AgentSession.create(tmp_path / 'session', policy=session_policy,
                             executor=lambda *args: None, clock=lambda: 1000.) as session:
        report = run_agent(session, selector, observation_builder=lambda *args: {'fixture': 'only'}, max_turns=2)
        assert len(client.requests) == 1
        assert report['turns'][0]['outcome']['ok'] is True
        assert report['turns'][1]['outcome']['ok'] is False
        assert 'cap would be exceeded' in report['turns'][1]['outcome']['error']['message']
    assert ledger.state['attempts'] == 1 and Decimal(ledger.state['charged_usd']) <= Decimal(frozen.hard_usd_cap)
    ledger.close()


@pytest.mark.parametrize('override', [{'input_usd_per_million_tokens': 2.0}, {'hard_usd_cap': 'NaN'},
    {'hard_usd_cap': '0'}, {'output_usd_per_million_tokens': '-1'}, {'input_bound_id': ''},
    {'input_bound_kind': 'guess'}, {'max_output_tokens': True}, {'max_request_utf8_bytes': 20000000}])
def test_policy_requires_explicit_finite_prices_and_bounds(override):
    with pytest.raises(ValueError):
        policy(**override)


@pytest.mark.parametrize('override', [{'input_bound_kind': 'verified_context_limit'},
    {'input_token_limit': 400000}, {'service_tier': 'auto'},
    {'input_bound_kind': 'verified_context_limit', 'input_token_limit': True}])
def test_model_context_bound_and_standard_tier_require_explicit_policy(override):
    with pytest.raises(ValueError):
        policy(**override)
