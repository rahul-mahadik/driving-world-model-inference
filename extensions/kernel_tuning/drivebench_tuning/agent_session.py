"""Durable, config-only agent tools backed by a caller-owned trusted executor.

The agent supplies JSON arguments, never Python, shell commands, file paths or
measurements. The controller and executor must run outside the agent process
and their storage must not be writable by the agent. Hash chaining detects
corruption; it is not an OS sandbox or authentication against a writer who can
replace the entire ledger. Pin ``expected_policy`` on reopening a session.

This module enforces admission and records a wall deadline. A trusted external
supervisor must interrupt stuck compiler/CUDA calls and own resource cleanup.
There is no automatic provisioning, funding, deadline extension or reset.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time
from types import MappingProxyType


TOOLS = frozenset({'inspect', 'propose', 'profile', 'trial', 'submit'})
TERMINAL = frozenset({'completed', 'final_failed', 'aborted', 'expired'})


class SessionError(RuntimeError):
    """An admission, execution or integrity failure; no budget is refunded."""


def _freeze(value):
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, Mapping) and all(type(key) is str for key in value):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    raise ValueError('finite JSON values with string object keys required')


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _copy(value):
    try:
        return _plain(_freeze(value))
    except RecursionError as exc:
        raise ValueError('cyclic or excessively nested JSON') from exc


def _bytes(value):
    return json.dumps(_copy(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _hash(value):
    return hashlib.sha256(_bytes(value)).hexdigest()


def _positive(value, name, *, integer=False, zero=False):
    if (type(value) is not int if integer else type(value) not in (int, float)):
        raise ValueError(f'{name} has an invalid numeric type')
    if not math.isfinite(value) or value < 0 or (value == 0 and not zero):
        raise ValueError(f'{name} must be finite and ' + ('nonnegative' if zero else 'positive'))
    return value


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def _read(path):
    if path.is_symlink() or not path.is_file():
        raise SessionError(f'invalid ledger artifact: {path.name}')
    try:
        return _copy(json.loads(path.read_bytes(), object_pairs_hook=_unique_object,
                                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value))))
    except (ValueError, RecursionError) as exc:
        raise SessionError(f'invalid JSON ledger artifact: {path.name}') from exc


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new(path, value):
    data = _bytes(value) + b'\n'
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    with os.fdopen(descriptor, 'wb') as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())
        os.fchmod(output.fileno(), 0o400)
    _sync_directory(path.parent)


@dataclass(frozen=True)
class SessionPolicy:
    session_id: str
    cohort_id: str
    reference_config: Mapping
    allowed_config_values: Mapping
    wall_seconds: float
    max_tool_calls: int
    max_evaluations: int
    max_profiles: int
    final_evaluations: int = 1
    metadata: Mapping = field(default_factory=dict)
    final_reserve_seconds: float = 0.0

    def __post_init__(self):
        if any(type(value) is not str or not value.strip() for value in (self.session_id, self.cohort_id)):
            raise ValueError('session_id and cohort_id must be nonempty strings')
        _positive(self.wall_seconds, 'wall_seconds')
        _positive(self.final_reserve_seconds, 'final_reserve_seconds', zero=True)
        if self.final_reserve_seconds >= self.wall_seconds:
            raise ValueError('final wall reserve must be smaller than the session wall budget')
        for name in ('max_tool_calls', 'max_evaluations', 'final_evaluations'):
            _positive(getattr(self, name), name, integer=True)
        _positive(self.max_profiles, 'max_profiles', integer=True, zero=True)
        if self.final_evaluations > self.max_evaluations:
            raise ValueError('final reservation exceeds total evaluation budget')
        for name in ('reference_config', 'allowed_config_values', 'metadata'):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise ValueError(f'{name} must be a JSON object')
            object.__setattr__(self, name, _freeze(_copy(value)))
        if set(self.reference_config) - {'generation', 'optimizations'}:
            raise ValueError('reference config contains only generation and optimizations')
        if any(not isinstance(value, Mapping) for value in self.reference_config.values()):
            raise ValueError('config sections must be objects')
        for path, values in self.allowed_config_values.items():
            parts = path.split('.')
            if (len(parts) != 2 or parts[0] not in ('generation', 'optimizations') or
                not parts[1].isidentifier() or parts[1].startswith('_') or
                parts[1] not in self.reference_config.get(parts[0], {})):
                raise ValueError('allowlist paths must identify existing declarative config fields')
            if not isinstance(values, tuple) or not values:
                raise ValueError('each allowlist field needs a nonempty sequence of allowed JSON values')
            if len({_hash(value) for value in values}) != len(values):
                raise ValueError('duplicate allowed values')

    def to_dict(self):
        return {name: _plain(getattr(self, name)) for name in self.__dataclass_fields__}


class AgentSession:
    """Single-owner append-only session. ``executor`` is trusted controller code.

    ``executor(kind, proposal, context)`` handles profile/trial/final calls and
    returns finite JSON with the exact ``cohort_id``. Final receipts must include
    status ok/pass (accepted) or fail/inconclusive/error/incorrect/failed
    (rejected). Its output is evidence,
    not agent input; correctness, scoring and private data remain executor-owned.
    ``final_evaluations`` counts the final executor's reserved evaluation units.
    A supervisor must enforce that executor's internal call count and timeout.

    Tool forms: inspect({}); propose({overrides, parent_id?}); and
    profile/trial/submit({proposal_id}). Every admitted call consumes tool budget,
    including malformed arguments and failed evaluations. One tool slot and
    final evaluation units are reserved for submission. A valid submission is
    irrevocable before execution, even if it fails. Opening an interrupted
    session permanently aborts it; restarting never refunds counters or time.
    """

    @classmethod
    def create(cls, directory, *, policy, executor, clock=time.time):
        if not isinstance(policy, SessionPolicy) or not callable(executor) or not callable(clock):
            raise ValueError('SessionPolicy and trusted executor/clock callables required')
        directory = Path(directory).absolute()
        directory.mkdir(parents=True, exist_ok=False, mode=0o700)
        for name in ('events', 'proposals'):
            (directory / name).mkdir(mode=0o700)
        self = cls._locked(directory, executor, clock)
        try:
            now = self._now()
            policy_doc = policy.to_dict()
            manifest = {'schema_version': 1, 'policy': policy_doc, 'policy_sha256': _hash(policy_doc),
                        'created_at': now, 'deadline_unix': now + policy.wall_seconds}
            _write_new(directory / 'policy.json', manifest)
            self.policy, self._manifest = policy, manifest
            self._state = self._initial_state(manifest)
            self._last_hash = _hash(manifest)
            self._sequence = 0
            reference = self._snapshot(None, policy_doc['reference_config'], {}, 0, 'reference')
            _write_new(directory / 'proposals' / (reference['proposal_id'] + '.json'), reference)
            self._append('created', {'reference_proposal_id': reference['proposal_id']}, now=now)
            return self
        except BaseException:
            self.close()
            raise

    @classmethod
    def open(cls, directory, *, executor, expected_policy=None, clock=time.time):
        self = cls._locked(Path(directory).absolute(), executor, clock)
        try:
            self._load(expected_policy)
            now = self._now()
            if now < self._state['last_timestamp']:
                raise SessionError('wall clock moved backwards; no execution admitted')
            if self._state['pending'] is not None:
                self._append('recovery_aborted', {'reason': 'interrupted operation; execution outcome unknown'}, now=now)
            return self
        except BaseException:
            self.close()
            raise

    @classmethod
    def _locked(cls, directory, executor, clock):
        if not callable(executor) or not callable(clock) or directory.is_symlink() or not directory.is_dir():
            raise ValueError('existing real directory and trusted callables required')
        self = object.__new__(cls)
        self.directory, self.executor, self.clock = directory, executor, clock
        self._closed = False
        lock_path = directory / '.controller.lock'
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | getattr(os, 'O_NOFOLLOW', 0), 0o600)
        self._lock = os.fdopen(descriptor, 'a+b')
        try:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock.close()
            raise SessionError('another controller owns this session') from exc
        return self

    def _now(self):
        value = self.clock()
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise SessionError('invalid controller wall clock')
        return float(value)

    def controller_time(self):
        """Read the trusted finite clock, rejecting rollback behind session events."""
        if self._closed:
            raise SessionError('controller is closed')
        now = self._now()
        if now < self._state['last_timestamp']:
            raise SessionError('wall clock moved backwards; no execution admitted')
        return now

    @staticmethod
    def _initial_state(manifest):
        return {'status': 'active', 'tool_calls_used': 0, 'evaluations_used': 0, 'profiles_used': 0,
                'pending': None, 'latest_proposal_id': None, 'reference_proposal_id': None,
                'selected_proposal_id': None, 'last_result': None,
                'created_at': manifest['created_at'], 'deadline_unix': manifest['deadline_unix'],
                'last_timestamp': manifest['created_at'], 'last_sequence': 0}

    @staticmethod
    def _snapshot(parent_id, config, overrides, sequence, kind='proposal'):
        body = {'parent_id': parent_id, 'config': config, 'overrides': overrides,
                'sequence': sequence, 'kind': kind}
        return {**_copy(body), 'proposal_id': _hash(body)}

    def _proposal(self, proposal_id):
        if (type(proposal_id) is not str or len(proposal_id) != 64 or
                any(char not in '0123456789abcdef' for char in proposal_id)):
            raise ValueError('proposal_id must be a controller-issued SHA-256 identity')
        proposal = _read(self.directory / 'proposals' / (proposal_id + '.json'))
        if not isinstance(proposal, dict) or proposal.get('proposal_id') != proposal_id:
            raise SessionError('proposal identity mismatch')
        if _hash({k: v for k, v in proposal.items() if k != 'proposal_id'}) != proposal_id:
            raise SessionError('proposal snapshot hash mismatch')
        return proposal

    def _merged(self, parent_id, overrides):
        if not isinstance(overrides, dict) or set(overrides) - {'generation', 'optimizations'}:
            raise ValueError('only allowlisted generation/optimizations overrides accepted; no code or paths')
        config = self._proposal(parent_id)['config']
        for section, settings in overrides.items():
            if not isinstance(settings, dict):
                raise ValueError('override sections must be objects')
            for key, value in settings.items():
                permitted = self.policy.allowed_config_values.get(f'{section}.{key}')
                if permitted is None or not any(_bytes(value) == _bytes(item) for item in permitted):
                    raise ValueError(f'configuration change not admitted: {section}.{key}')
                config[section][key] = value
        return config

    def _transition(self, state, event):
        state = deepcopy(state)
        kind, payload, now = event['type'], event['payload'], event['at']
        if type(now) not in (int, float) or not math.isfinite(now) or now < state['last_timestamp']:
            raise SessionError('nonmonotonic event time')
        if kind == 'created':
            if state['last_sequence'] or state['reference_proposal_id'] is not None:
                raise SessionError('duplicate creation')
            expected = self._snapshot(None, self.policy.to_dict()['reference_config'], {}, 0, 'reference')
            if payload != {'reference_proposal_id': expected['proposal_id']} or self._proposal(expected['proposal_id']) != expected:
                raise SessionError('reference snapshot does not match immutable policy')
            state['reference_proposal_id'] = state['latest_proposal_id'] = expected['proposal_id']
        elif kind == 'tool_started':
            if state['status'] != 'active' or state['pending'] is not None or now >= state['deadline_unix']:
                raise SessionError('tool started outside an active admissible session')
            limit = self.policy.max_tool_calls if payload['tool'] == 'submit' else self.policy.max_tool_calls - 1
            if payload['tool'] != 'submit' and now >= state['deadline_unix'] - self.policy.final_reserve_seconds:
                raise SessionError('development wall budget exhausted; final window reserved')
            if state['tool_calls_used'] >= limit:
                raise SessionError('tool budget exhausted; final submission slot reserved')
            state['tool_calls_used'] += 1
            state['pending'] = {'tool': payload['tool'], 'arguments': payload['arguments']}
        elif kind == 'proposal_created':
            pending = state['pending']
            if not pending or pending['tool'] != 'propose':
                raise SessionError('proposal event outside proposal tool')
            args = pending['arguments']
            parent = args.get('parent_id', state['latest_proposal_id'])
            expected = self._snapshot(parent, self._merged(parent, args['overrides']), args['overrides'], event['sequence'])
            if payload != {'proposal_id': expected['proposal_id']} or self._proposal(expected['proposal_id']) != expected:
                raise SessionError('proposal snapshot does not match admitted overrides')
            state['latest_proposal_id'] = expected['proposal_id']
        elif kind == 'execution_started':
            pending = state['pending']
            if not pending or pending['tool'] not in ('profile', 'trial', 'submit') or 'kind' in pending:
                raise SessionError('execution outside an admitted tool')
            expected_kind = {'profile': 'profile', 'trial': 'trial', 'submit': 'final'}[pending['tool']]
            cutoff = state['deadline_unix'] - (0 if expected_kind == 'final' else self.policy.final_reserve_seconds)
            if now >= cutoff:
                raise SessionError('execution admission missed its immutable wall cutoff')
            if payload['kind'] != expected_kind or payload['proposal_id'] != pending['arguments']['proposal_id']:
                raise SessionError('execution/proposal binding mismatch')
            self._proposal(payload['proposal_id'])
            units = self.policy.final_evaluations if expected_kind == 'final' else int(expected_kind == 'trial')
            if payload['evaluation_units'] != units:
                raise SessionError('evaluation reservation mismatch')
            if expected_kind == 'profile':
                if state['profiles_used'] >= self.policy.max_profiles:
                    raise SessionError('profile budget exhausted')
                state['profiles_used'] += 1
            else:
                limit = self.policy.max_evaluations if expected_kind == 'final' else self.policy.max_evaluations - self.policy.final_evaluations
                if state['evaluations_used'] + units > limit:
                    raise SessionError('evaluation budget exhausted; final evaluation reserved')
                state['evaluations_used'] += units
            state['pending'].update(payload)
            if expected_kind == 'final':
                state['status'] = 'submitted'
                state['selected_proposal_id'] = payload['proposal_id']
        elif kind in ('tool_finished', 'tool_failed'):
            pending = state['pending']
            if pending is None:
                raise SessionError('completion without a pending tool')
            cutoff = state['deadline_unix'] - (self.policy.final_reserve_seconds
                if pending.get('kind') in ('trial', 'profile') else 0)
            if kind == 'tool_finished' and now >= cutoff:
                raise SessionError('completion missed the immutable wall deadline')
            if kind == 'tool_finished' and 'kind' in pending:
                result = payload['result']
                if not isinstance(result, dict) or result.get('cohort_id') != self.policy.cohort_id:
                    raise SessionError('executor cohort binding mismatch')
                if pending['kind'] == 'final' and result.get('status') not in (
                        'ok', 'pass', 'fail', 'inconclusive', 'error', 'incorrect', 'failed'):
                    raise SessionError('final executor receipt requires an explicit recognized status')
            state['last_result'] = {'event_sequence': event['sequence'], 'tool': pending['tool'],
                                    'successful_tool': kind == 'tool_finished'}
            if state['status'] == 'submitted':
                result_status = payload.get('result', {}).get('status')
                state['status'] = ('completed' if kind == 'tool_finished' and result_status in ('ok', 'pass') else 'final_failed')
            if now >= state['deadline_unix']:
                state['status'] = 'expired'
            state['pending'] = None
        elif kind == 'recovery_aborted':
            if state['pending'] is None:
                raise SessionError('recovery without an interrupted operation')
            state['status'] = 'aborted'
            state['pending'] = None
        elif kind == 'expired':
            if state['status'] != 'active' or state['pending'] is not None or now < state['deadline_unix']:
                raise SessionError('invalid deadline expiry')
            state['status'] = 'expired'
        else:
            raise SessionError('unknown ledger event')
        state['last_timestamp'], state['last_sequence'] = now, event['sequence']
        return state

    def _append(self, kind, payload, *, now=None):
        event = {'sequence': self._sequence + 1, 'previous_hash': self._last_hash,
                 'at': self._now() if now is None else now, 'type': kind, 'payload': _copy(payload)}
        state = self._transition(self._state, event)
        event['state'] = state
        event['sha256'] = _hash(event)
        _write_new(self.directory / 'events' / f'{event["sequence"]:08d}.json', event)
        self._state, self._sequence, self._last_hash = state, event['sequence'], event['sha256']

    def _load(self, expected_policy=None):
        manifest = _read(self.directory / 'policy.json')
        if set(manifest) != {'schema_version', 'policy', 'policy_sha256', 'created_at', 'deadline_unix'} or manifest['schema_version'] != 1:
            raise SessionError('invalid session manifest')
        policy = SessionPolicy(**manifest['policy'])
        if _hash(policy.to_dict()) != manifest['policy_sha256']:
            raise SessionError('policy hash mismatch')
        if expected_policy is not None and (not isinstance(expected_policy, SessionPolicy) or expected_policy.to_dict() != policy.to_dict()):
            raise SessionError('stored policy differs from caller-pinned policy')
        if (type(manifest['created_at']) not in (int, float) or not math.isfinite(manifest['created_at']) or
                manifest['deadline_unix'] != manifest['created_at'] + policy.wall_seconds):
            raise SessionError('immutable deadline mismatch')
        self.policy, self._manifest = policy, manifest
        self._state, self._last_hash, self._sequence = self._initial_state(manifest), _hash(manifest), 0
        paths = sorted((self.directory / 'events').iterdir())
        if not paths:
            raise SessionError('session creation incomplete')
        for sequence, path in enumerate(paths, 1):
            if path.name != f'{sequence:08d}.json':
                raise SessionError('event sequence missing or unexpected artifact')
            event = _read(path)
            if set(event) != {'sequence', 'previous_hash', 'at', 'type', 'payload', 'state', 'sha256'}:
                raise SessionError('invalid event schema')
            if (event['sequence'] != sequence or event['previous_hash'] != self._last_hash or
                    event['sha256'] != _hash({k: v for k, v in event.items() if k != 'sha256'})):
                raise SessionError('event hash chain mismatch')
            state = self._transition(self._state, event)
            if state != event['state']:
                raise SessionError('ledger state does not match replay')
            self._state, self._sequence, self._last_hash = state, sequence, event['sha256']
        expected_proposals = {self._state['reference_proposal_id']}
        for path in paths:
            event = _read(path)
            if event['type'] == 'proposal_created':
                expected_proposals.add(event['payload']['proposal_id'])
        actual_proposals = {path.name for path in (self.directory / 'proposals').iterdir()}
        if actual_proposals != {identity + '.json' for identity in expected_proposals}:
            raise SessionError('uncommitted or unexpected proposal snapshot; execution denied')

    @property
    def state(self):
        if self._closed:
            raise SessionError('controller is closed')
        return {**deepcopy(self._state), 'session_id': self.policy.session_id,
                'cohort_id': self.policy.cohort_id, 'policy_sha256': self._manifest['policy_sha256'],
                'ledger_sha256': self._last_hash,
                'tool_calls_remaining': self.policy.max_tool_calls - self._state['tool_calls_used'],
                'evaluations_remaining': self.policy.max_evaluations - self._state['evaluations_used'],
                'profiles_remaining': self.policy.max_profiles - self._state['profiles_used'],
                'final_evaluations_reserved': 0 if self._state['selected_proposal_id'] else self.policy.final_evaluations}

    def admit_selector(self):
        """Return trusted remaining time, or None for an expired/terminal session.

        This does not consume tool budget. Call immediately before a potentially
        billed selector request; a provider timeout must also be capped to the
        returned remaining_seconds. Model-visible timestamps are not authority.
        """
        if self._closed:
            raise SessionError('controller is closed')
        # Re-read the protected ledger before each request; no stale local counters.
        expected_head = self._last_hash
        self._load(self.policy)
        if self._last_hash != expected_head:
            raise SessionError('ledger changed outside the owning controller')
        now = self.controller_time()
        if self._state['status'] in TERMINAL:
            return None
        if self._state['status'] != 'active' or self._state['pending'] is not None:
            raise SessionError('session has an unfinished operation')
        if now >= self._state['deadline_unix']:
            self._append('expired', {}, now=now)
            return None
        return {'admitted_at': now, 'deadline_unix': self._state['deadline_unix'],
                'remaining_seconds': self._state['deadline_unix'] - now}

    def call(self, tool, arguments=None):
        admission = self.admit_selector()
        if admission is None:
            raise SessionError('session terminal or wall deadline exhausted')
        now = admission['admitted_at']
        args = _copy({} if arguments is None else arguments)
        if not isinstance(args, dict) or len(_bytes(args)) > 262144:
            raise ValueError('tool arguments must be a bounded JSON object')
        if type(tool) is not str:
            raise ValueError('tool name must be text')
        self._append('tool_started', {'tool': tool, 'arguments': args}, now=now)
        raw_result = None
        completion_deadline = self._state['deadline_unix']
        try:
            if tool not in TOOLS:
                raise ValueError('unknown tool; code, filesystem and shell execution are not supported')
            if tool == 'inspect':
                if args:
                    raise ValueError('inspect takes no arguments')
                result = {'policy': self.policy.to_dict(), 'latest_proposal': self._proposal(self._state['latest_proposal_id'])}
            elif tool == 'propose':
                if not {'overrides'} <= set(args) <= {'overrides', 'parent_id'}:
                    raise ValueError('propose accepts overrides and optional parent_id only')
                parent = args.get('parent_id', self._state['latest_proposal_id'])
                config = self._merged(parent, args['overrides'])
                proposal = self._snapshot(parent, config, args['overrides'], self._sequence + 1)
                _write_new(self.directory / 'proposals' / (proposal['proposal_id'] + '.json'), proposal)
                self._append('proposal_created', {'proposal_id': proposal['proposal_id']})
                result = proposal
            else:
                if set(args) != {'proposal_id'}:
                    raise ValueError('evaluation tools accept proposal_id only; measurements cannot be supplied')
                proposal = self._proposal(args['proposal_id'])
                kind = {'profile': 'profile', 'trial': 'trial', 'submit': 'final'}[tool]
                units = self.policy.final_evaluations if kind == 'final' else int(kind == 'trial')
                self._append('execution_started', {'kind': kind, 'proposal_id': proposal['proposal_id'], 'evaluation_units': units})
                completion_deadline -= 0 if kind == 'final' else self.policy.final_reserve_seconds
                context = {'session_id': self.policy.session_id, 'cohort_id': self.policy.cohort_id,
                           'policy_sha256': self._manifest['policy_sha256'],
                           'deadline_unix': self._state['deadline_unix'],
                           'execution_deadline_unix': completion_deadline,
                           'remaining_seconds': max(0., completion_deadline - self._now()),
                           'evaluation_units': units, 'metadata': self.policy.to_dict()['metadata']}
                if context['remaining_seconds'] <= 0:
                    raise SessionError('execution dispatch missed its immutable wall deadline')
                raw_result = self.executor(kind, deepcopy(proposal), context)
                result = _copy(raw_result)
                if not isinstance(result, dict) or result.get('cohort_id') != self.policy.cohort_id:
                    raise ValueError('trusted executor returned missing or mismatched cohort_id')
                if len(_bytes(result)) > 8388608:
                    raise ValueError('executor result exceeds bounded JSON receipt size')
            if self._now() >= completion_deadline:
                raise SessionError('executor/tool completed after its immutable wall deadline')
            self._append('tool_finished', {'result': result})
            return {'tool': tool, 'result': deepcopy(result), 'state': self.state}
        except Exception as exc:
            failure = {'error': {'type': type(exc).__name__, 'message': str(exc)}}
            # Preserve valid raw executor receipts even if a binding/deadline check failed.
            try:
                if raw_result is not None and len(_bytes(raw_result)) <= 8388608:
                    failure['executor_result'] = _copy(raw_result)
            except (ValueError, TypeError):
                failure['executor_result_unserializable'] = True
            self._append('tool_failed', failure)
            raise

    def close(self):
        if not self._closed:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
            self._lock.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
