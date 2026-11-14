"""Trusted pre-call USD reservations for the tool-free Responses selector.

No model, price, API key, tokenizer or request is chosen here. The caller freezes
prices and supplies an input-token upper-bound function for the EXACT outgoing
request, including instructions and JSON schema. For explicit bounds the whole
canonical UTF-8 request length is a required conservative floor; the caller must
also cover provider framing/overhead. A verified tokenizer/counting contract may
instead supply a smaller bound. A caller-verified exact model context limit may
instead reserve that entire limit every call. These assertions are not verified
by this adapter.

Every dispatched request permanently charges its FULL reservation, even when it
succeeds. Missing usage, refusals and failures never refund budget. Usage is audit
evidence only. A reported bound violation halts future requests; it cannot undo a
provider charge. The cap therefore depends on correct caller-supplied pricing,
token bounds and provider compliance. Taxes and external account traffic are not
included. Private ledger storage must be outside agent write access; hashes and
flock are integrity/concurrency controls, not an OS sandbox.

Output reservations use max_output_tokens, including reasoning/non-visible tokens:
https://developers.openai.com/api/reference/resources/responses/methods/create
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal, localcontext
import fcntl
import math
import os
from pathlib import Path
import re
from types import SimpleNamespace
import uuid

from .agent_session import _bytes, _copy, _hash, _read, _sync_directory, _write_new


class CostBudgetError(RuntimeError):
    """No request admitted, or a charged request breached its frozen contract."""


def _integer(value, name, *, zero=False):
    if type(value) is not int or value < int(not zero) or value > 2**63 - 1:
        raise ValueError(f'{name} must be a bounded ' + ('nonnegative' if zero else 'positive') + ' integer')
    return value


def _decimal(value, name, *, zero=False):
    if type(value) is not str or not re.fullmatch(r'(?:0|[1-9][0-9]{0,12})(?:\.[0-9]{1,18})?', value):
        raise ValueError(f'{name} must be an explicit non-exponent decimal string')
    parsed = Decimal(value)
    if parsed < 0 or (not zero and parsed == 0):
        raise ValueError(f'{name} must be positive')
    return parsed


def _money(value):
    with localcontext() as ctx:
        ctx.prec = 80
        return format(value.normalize(), 'f')


@dataclass(frozen=True)
class APICostPolicy:
    budget_id: str
    model: str
    input_usd_per_million_tokens: str
    output_usd_per_million_tokens: str
    hard_usd_cap: str
    max_output_tokens: int
    input_bound_kind: str
    input_bound_id: str
    max_request_utf8_bytes: int = 2097152
    input_token_limit: int | None = None
    service_tier: str = 'default'

    def __post_init__(self):
        for name in ('budget_id', 'model', 'input_bound_id'):
            if type(getattr(self, name)) is not str or not getattr(self, name).strip():
                raise ValueError(f'explicit {name} required')
        if self.input_bound_kind not in ('explicit_upper_bound', 'verified_tokenizer', 'verified_context_limit'):
            raise ValueError('an explicit upper bound or caller-verified tokenizer contract is required')
        if self.input_bound_kind == 'verified_context_limit':
            _integer(self.input_token_limit, 'verified model context input token limit')
        elif self.input_token_limit is not None:
            raise ValueError('input_token_limit is reserved for a verified context-limit contract')
        if self.service_tier != 'default':
            raise ValueError('only an explicitly priced default service tier is supported')
        for name in ('input_usd_per_million_tokens', 'output_usd_per_million_tokens', 'hard_usd_cap'):
            value = _decimal(getattr(self, name), name, zero=name != 'hard_usd_cap')
            object.__setattr__(self, name, _money(value))
        _integer(self.max_output_tokens, 'max_output_tokens')
        _integer(self.max_request_utf8_bytes, 'max_request_utf8_bytes')
        if self.max_request_utf8_bytes > 16777216:
            raise ValueError('request bound exceeds this text-only adapter limit')

    def to_dict(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def reservation_usd(self, input_tokens, output_tokens):
        _integer(input_tokens, 'input_tokens')
        _integer(output_tokens, 'output_tokens')
        with localcontext() as ctx:
            ctx.prec = 80
            return _money((Decimal(self.input_usd_per_million_tokens) * input_tokens +
                           Decimal(self.output_usd_per_million_tokens) * output_tokens) / 1000000)


class APICostLedger:
    """Exclusive durable reservation owner; open requires the caller-pinned policy.

    The input_token_bound callable receives a defensive copy of the entire exact
    transport request and returns its trusted upper token bound. It must include
    provider overhead and must not itself make an unbudgeted provider call. Its
    verified identity/contract is pinned in policy.input_bound_id. An interrupted
    transport retains its full charge and permanently halts this ledger.
    """

    @classmethod
    def create(cls, directory, *, policy, input_token_bound=None):
        if (not isinstance(policy, APICostPolicy) or
                (policy.input_bound_kind != 'verified_context_limit' and not callable(input_token_bound))):
            raise ValueError('frozen APICostPolicy and trusted input-token bound required')
        directory = Path(directory).absolute()
        directory.mkdir(parents=True, exist_ok=False, mode=0o700)
        (directory / 'events').mkdir(mode=0o700)
        self = cls._lock(directory, input_token_bound)
        try:
            manifest = {'policy': policy.to_dict(), 'policy_sha256': _hash(policy.to_dict()),
                        'ledger_id': uuid.uuid4().hex}
            _write_new(directory / 'policy.json', {**manifest, 'sha256': _hash(manifest)})
            _sync_directory(directory.parent)
            self._load(policy)
            return self
        except BaseException:
            self.close()
            raise

    @classmethod
    def open(cls, directory, *, expected_policy, input_token_bound=None):
        if (not isinstance(expected_policy, APICostPolicy) or
                (expected_policy.input_bound_kind != 'verified_context_limit' and not callable(input_token_bound))):
            raise ValueError('caller-pinned APICostPolicy and trusted input-token bound required')
        self = cls._lock(Path(directory).absolute(), input_token_bound)
        try:
            self._load(expected_policy)
            if self._state['pending'] is not None:
                self._append('finished', {'request_id': self._state['pending']['request_id'],
                    'outcome': 'interrupted', 'usage': None, 'usage_status': 'unknown', 'bound_violation': False})
            return self
        except BaseException:
            self.close()
            raise

    @classmethod
    def _lock(cls, directory, input_token_bound):
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError('real existing ledger directory required')
        self = object.__new__(cls)
        self.directory, self.input_token_bound = directory, input_token_bound
        self._closed = False
        fd = os.open(directory / '.cost.lock', os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0), 0o600)
        self._lock_file = os.fdopen(fd, 'a+b')
        try:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock_file.close()
            raise CostBudgetError('another controller owns the API cost ledger') from exc
        return self

    def _transition(self, state, kind, payload):
        state = deepcopy(state)
        if kind == 'reserved':
            if state['halted'] or state['pending'] is not None:
                raise CostBudgetError('cost ledger halted or request unfinished')
            _integer(payload['input_tokens'], 'input token reservation')
            _integer(payload['output_tokens'], 'output token reservation')
            _integer(payload['request_utf8_bytes'], 'request bytes')
            if payload['request_utf8_bytes'] > self.policy.max_request_utf8_bytes:
                raise CostBudgetError('request exceeds frozen transport bound')
            if self.policy.input_bound_kind == 'explicit_upper_bound' and payload['input_tokens'] < payload['request_utf8_bytes']:
                raise CostBudgetError('explicit input reservation is below the full transport UTF-8 byte floor')
            if self.policy.input_bound_kind == 'verified_context_limit' and payload['input_tokens'] != self.policy.input_token_limit:
                raise CostBudgetError('reservation must cover the entire caller-verified model context limit')
            if payload['output_tokens'] > self.policy.max_output_tokens:
                raise CostBudgetError('output token request exceeds frozen maximum')
            if payload['reservation_usd'] != self.policy.reservation_usd(payload['input_tokens'], payload['output_tokens']):
                raise CostBudgetError('reservation does not match frozen prices')
            if not re.fullmatch('[0-9a-f]{64}', payload['request_sha256']):
                raise CostBudgetError('invalid request identity')
            if payload['request_id'] != state['attempts'] + 1:
                raise CostBudgetError('API request sequence mismatch')
            with localcontext() as ctx:
                ctx.prec = 80
                charged = Decimal(state['charged_usd']) + Decimal(payload['reservation_usd'])
            if charged > Decimal(self.policy.hard_usd_cap):
                raise CostBudgetError('hard API USD cap would be exceeded; request not sent')
            state.update(charged_usd=_money(charged), pending=deepcopy(payload), attempts=state['attempts'] + 1)
        elif kind == 'finished':
            if state['pending'] is None or payload['request_id'] != state['pending']['request_id']:
                raise CostBudgetError('completion does not match a reserved request')
            if payload['outcome'] not in ('returned', 'raised', 'interrupted'):
                raise CostBudgetError('unknown request outcome')
            usage_status, violation = self._usage(payload.get('usage'), state['pending'])
            violation = violation or payload.get('returned_service_tier') not in (None, self.policy.service_tier)
            violation = violation or payload.get('returned_model') not in (None, self.policy.model)
            if payload['usage_status'] != usage_status or payload['bound_violation'] is not violation:
                raise CostBudgetError('usage receipt does not match reservation audit')
            state['halted'] = violation or payload['outcome'] == 'interrupted'
            state['last_receipt'] = deepcopy(payload)
            state['pending'] = None
        else:
            raise CostBudgetError('unknown cost ledger event')
        return state

    @staticmethod
    def _usage(usage, reservation):
        if not isinstance(usage, dict):
            return 'unknown', False
        counts = [usage.get(name) for name in ('input_tokens', 'output_tokens', 'total_tokens')]
        violation = any(type(count) is int and count > maximum for count, maximum in
            zip(counts[:2], (reservation['input_tokens'], reservation['output_tokens'])))
        if any(type(value) is not int or value < 0 for value in counts) or counts[2] != counts[0] + counts[1]:
            return 'unknown', violation
        details = usage.get('output_tokens_details')
        if details is not None:
            if not isinstance(details, dict):
                return 'unknown', violation
            reasoning = details.get('reasoning_tokens')
            if reasoning is not None and (type(reasoning) is not int or not 0 <= reasoning <= counts[1]):
                return 'unknown', violation or (type(reasoning) is int and reasoning > reservation['output_tokens'])
        return 'reported', violation

    def _load(self, expected_policy):
        manifest = _read(self.directory / 'policy.json')
        if (not isinstance(manifest, dict) or set(manifest) != {'policy', 'policy_sha256', 'ledger_id', 'sha256'} or
                manifest['policy'] != expected_policy.to_dict() or manifest['policy_sha256'] != _hash(expected_policy.to_dict()) or
                type(manifest['ledger_id']) is not str or not re.fullmatch('[0-9a-f]{32}', manifest['ledger_id']) or
                manifest['sha256'] != _hash({k: v for k, v in manifest.items() if k != 'sha256'})):
            raise CostBudgetError('API policy/prices/cap differ from caller-pinned policy')
        self.policy = expected_policy
        self._ledger_id = manifest['ledger_id']
        self._head, self._sequence = manifest['sha256'], 0
        self._state = {'charged_usd': '0', 'attempts': 0, 'pending': None, 'halted': False, 'last_receipt': None}
        for index, path in enumerate(sorted((self.directory / 'events').iterdir()), 1):
            if path.name != f'{index:08d}.json':
                raise CostBudgetError('cost event missing or unexpected artifact')
            event = _read(path)
            if (set(event) != {'sequence', 'previous_hash', 'kind', 'payload', 'state', 'sha256'} or
                    event['sequence'] != index or event['previous_hash'] != self._head or
                    event['sha256'] != _hash({k: v for k, v in event.items() if k != 'sha256'})):
                raise CostBudgetError('cost event hash chain mismatch')
            state = self._transition(self._state, event['kind'], event['payload'])
            if state != event['state']:
                raise CostBudgetError('cost ledger state does not match replay')
            self._state, self._head, self._sequence = state, event['sha256'], index

    def _append(self, kind, payload):
        payload = _copy(payload)
        state = self._transition(self._state, kind, payload)
        event = {'sequence': self._sequence + 1, 'previous_hash': self._head,
                 'kind': kind, 'payload': payload, 'state': state}
        event['sha256'] = _hash(event)
        _write_new(self.directory / 'events' / f'{event["sequence"]:08d}.json', event)
        self._state, self._head, self._sequence = state, event['sha256'], event['sequence']

    def _refresh(self):
        if self._closed:
            raise CostBudgetError('cost ledger is closed')
        head = self._head
        self._load(self.policy)
        if self._head != head:
            raise CostBudgetError('cost ledger changed outside its owning controller')

    def reserve(self, request):
        self._refresh()
        request = _copy(request)
        self._validate_request(request)
        byte_count = len(_bytes(request))
        if byte_count > self.policy.max_request_utf8_bytes:
            raise CostBudgetError('request exceeds frozen transport bound')
        if self._state['halted'] or self._state['pending'] is not None:
            raise CostBudgetError('cost ledger halted or request unfinished')
        tokens = (self.policy.input_token_limit if self.policy.input_bound_kind == 'verified_context_limit'
                  else self.input_token_bound(deepcopy(request)))
        _integer(tokens, 'trusted input token bound')
        reservation = {'request_id': self._state['attempts'] + 1, 'request_sha256': _hash(request),
                       'request_utf8_bytes': byte_count, 'input_tokens': tokens,
                       'output_tokens': request['max_output_tokens'],
                       'reservation_usd': self.policy.reservation_usd(tokens, request['max_output_tokens'])}
        self._append('reserved', reservation)
        return deepcopy(reservation)

    def _validate_request(self, request):
        if not isinstance(request, dict) or set(request) != {
                'model', 'instructions', 'input', 'tools', 'store', 'max_output_tokens', 'text', 'service_tier'}:
            raise CostBudgetError('only the exact bounded text-only selector transport is supported')
        if request['model'] != self.policy.model or request['tools'] != [] or request['store'] is not False:
            raise CostBudgetError('model/tools/store differ from the admitted selector contract')
        if request['service_tier'] != self.policy.service_tier:
            raise CostBudgetError('request service tier differs from frozen pricing contract')
        if type(request['instructions']) is not str:
            raise CostBudgetError('instructions must be explicit text')
        inputs = request['input']
        if (not isinstance(inputs, list) or len(inputs) != 1 or not isinstance(inputs[0], dict) or
                set(inputs[0]) != {'role', 'content'} or inputs[0]['role'] != 'user' or type(inputs[0]['content']) is not str):
            raise CostBudgetError('one explicit user text input required; no hidden conversation, files or images')
        if len(inputs[0]['content'].encode()) > 1048576:
            raise CostBudgetError('input exceeds ResponsesActionPolicy observation bound')
        text = request['text']
        if (not isinstance(text, dict) or set(text) != {'format'} or not isinstance(text['format'], dict) or
                set(text['format']) != {'type', 'name', 'schema', 'strict'} or
                text['format']['type'] != 'json_schema' or text['format']['strict'] is not True or
                text['format']['name'] != 'drivebench_action' or not isinstance(text['format']['schema'], dict)):
            raise CostBudgetError('explicit strict action schema required')
        _integer(request['max_output_tokens'], 'max_output_tokens')

    def finish(self, reservation, *, outcome, usage=None, response_id=None, error_type=None,
               returned_service_tier=None, returned_model=None):
        self._refresh()
        if self._state['pending'] != reservation:
            raise CostBudgetError('reservation mismatch; transport completion rejected')
        try:
            usage = _copy(usage)
            if len(_bytes(usage)) > 65536:
                usage = None
        except (ValueError, TypeError):
            usage = None
        status, violation = self._usage(usage, reservation)
        violation = violation or returned_service_tier not in (None, self.policy.service_tier)
        violation = violation or returned_model not in (None, self.policy.model)
        self._append('finished', {'request_id': reservation['request_id'], 'outcome': outcome,
            'usage': usage, 'usage_status': status, 'bound_violation': violation,
            'response_id': response_id if type(response_id) is str else None,
            'returned_service_tier': returned_service_tier,
            'returned_model': returned_model,
            'error_type': error_type if type(error_type) is str else None})
        if violation:
            raise CostBudgetError('provider-reported model, tier or tokens exceeded the reserved bounds; further API calls halted')

    @property
    def state(self):
        if self._closed:
            raise CostBudgetError('cost ledger is closed')
        with localcontext() as ctx:
            ctx.prec = 80
            remaining = Decimal(self.policy.hard_usd_cap) - Decimal(self._state['charged_usd'])
        return {**deepcopy(self._state), 'remaining_usd': _money(remaining),
                'ledger_id': self._ledger_id,
                'policy_sha256': _hash(self.policy.to_dict()), 'ledger_sha256': self._head,
                'charges_are_full_reservations': True, 'provider_invoice_verified': False}

    def close(self):
        if not self._closed:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            self._lock_file.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class BudgetedResponsesClient:
    """Inject this instead of a bare official client into ResponsesActionPolicy.

    with_options preserves the same ledger, disables SDK retries and accepts only
    timeout/retry options. Raw client use outside this adapter is not controlled.
    No source change to the selector or run_agent is required.
    """

    def __init__(self, client, ledger):
        if not isinstance(ledger, APICostLedger) or not callable(getattr(client, 'with_options', None)):
            raise ValueError('official client and trusted APICostLedger required')
        self._client, self.ledger = client.with_options(max_retries=0), ledger
        self.responses = SimpleNamespace(create=self._create)

    def with_options(self, **options):
        if set(options) - {'max_retries', 'timeout'} or options.get('max_retries', 0) != 0:
            raise CostBudgetError('automatic retries and transport overrides are not admitted')
        if 'timeout' in options:
            timeout = options['timeout']
            if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 300:
                raise CostBudgetError('finite bounded transport timeout required')
        instance = object.__new__(type(self))
        instance._client = self._client.with_options(**{**options, 'max_retries': 0})
        instance.ledger = self.ledger
        instance.responses = SimpleNamespace(create=instance._create)
        return instance

    @property
    def cost_binding(self):
        state = self.ledger.state
        return {'ledger_id': state['ledger_id'], 'policy_sha256': state['policy_sha256'],
                'full_reservation_charged_per_attempt': True}

    def _create(self, **request):
        # Snapshot before the bound callback: it cannot mutate what gets sent.
        request = _copy(request)
        reservation = self.ledger.reserve(request)
        try:
            response = self._client.responses.create(**request)
        except Exception as exc:
            self.ledger.finish(reservation, outcome='raised', error_type=type(exc).__name__)
            raise
        # Invalid or missing usage receives the same full charge. Parsing errors
        # cannot silently discard an already attempted request.
        usage, response_id, service_tier, model = None, None, None, None
        try:
            doc = response.model_dump(mode='json') if callable(getattr(response, 'model_dump', None)) else response
            if isinstance(doc, dict):
                usage, response_id, service_tier, model = doc.get('usage'), doc.get('id'), doc.get('service_tier'), doc.get('model')
        except Exception:
            pass
        self.ledger.finish(reservation, outcome='returned', usage=usage, response_id=response_id,
                           returned_service_tier=service_tier, returned_model=model)
        return response
