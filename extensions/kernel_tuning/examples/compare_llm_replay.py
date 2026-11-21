"""Evaluate a named API policy on measured operator data with a dollar cap.

Prices are supplied explicitly because provider pricing can change. Credentials
and cost ledgers belong outside the repository; outputs contain no API key.
"""
import argparse
from copy import deepcopy
from decimal import Decimal
import json
import os
from pathlib import Path
import stat
import signal
import time
from types import SimpleNamespace

from drivebench_tuning.agent_budget import APICostLedger, APICostPolicy, BudgetedResponsesClient
from drivebench_tuning.replay import MeasuredTable
from drivebench_tuning.replay_cli import write_json, selection_manifest
from drivebench_tuning.replay_policy import ResponsesIndexPolicy, run_replay_policy, public_policy_observation
from drivebench_tuning.evaluation import PolicyActionRejected
from drivebench_tuning.suite import load_suite


class DeadlineTransport:
    """Bound the existing SDK call to one absolute stage deadline.

    SDK socket timeouts are also reduced to remaining time. A process timer
    supplies the absolute wall bound because a socket timeout alone can restart
    across request phases. This CLI runs on the Unix main thread.
    """
    def __init__(self, client, deadline, *, clock=time.monotonic, options=None):
        self.client, self.deadline, self.clock = client, deadline, clock
        self.options = dict(options or {})
        self.responses = SimpleNamespace(create=self.create)

    def with_options(self, **options):
        if set(options) - {'max_retries', 'timeout'} or options.get('max_retries', 0) != 0:
            raise ValueError('Only bounded timeout and disabled retries are permitted')
        return DeadlineTransport(self.client, self.deadline, clock=self.clock,
                                 options={**self.options, **options})

    def create(self, **request):
        remaining = self.deadline - self.clock()
        if remaining <= 0:
            raise TimeoutError('policy_stage_wall_deadline')
        if signal.getitimer(signal.ITIMER_REAL)[0] > 0:
            raise RuntimeError('Existing process alarm cannot be replaced')
        def expired(_signum, _frame):
            raise TimeoutError('policy_stage_wall_deadline')
        prior = signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, remaining)
        try:
            timeout = min(float(self.options.get('timeout', 60)), remaining, 60)
            response = self.client.with_options(max_retries=0, timeout=timeout).responses.create(**request)
            if self.clock() >= self.deadline:
                raise TimeoutError('policy_stage_wall_deadline')
            return response
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, prior)


class DurableTurns:
    """Append and fsync before and after each policy turn; no secret/error text."""
    def __init__(self, path):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND, 0o600)
        self.stream = os.fdopen(fd, 'w')
        directory_fd = os.open(Path(path).parent, os.O_RDONLY)
        try: os.fsync(directory_fd)
        finally: os.close(directory_fd)

    def append(self, event):
        raw = json.dumps(event, sort_keys=True, allow_nan=False) + '\n'
        if len(raw.encode()) > 131072:
            raise ValueError('Policy turn evidence exceeds fixed bound')
        self.stream.write(raw); self.stream.flush(); os.fsync(self.stream.fileno())

    def close(self):
        self.stream.close()


def sanitized_receipt(receipt):
    """Retain fixed scalar metadata and token counts, never raw provider errors."""
    if receipt is None:
        return None
    result = {}
    for field in ('call_index', 'evaluation_episode_index', 'trials_used', 'action'):
        value = receipt.get(field)
        if type(value) is int:
            result[field] = value
    for field in ('task_id', 'status', 'error_type', 'response_id', 'returned_model', 'provider_status'):
        value = receipt.get(field)
        if type(value) is str and len(value) <= 256:
            result[field] = value
    value = receipt.get('wall_seconds')
    if type(value) in (int, float) and 0 <= value < 86400:
        result['wall_seconds'] = value
    usage = receipt.get('usage')
    if isinstance(usage, dict):
        result['usage'] = {k: usage[k] for k in ('input_tokens', 'output_tokens', 'total_tokens')
                           if type(usage.get(k)) is int and usage[k] >= 0}
    return result


class DurableDeadlinePolicy:
    def __init__(self, selector, suite, journal, deadline, *, max_wall_seconds, clock=time.monotonic):
        self.selector, self.suite, self.journal = selector, suite, journal
        self.deadline, self.clock = deadline, clock
        self.turns = 0
        self.deadline_rejections = 0
        self.configuration = {**deepcopy(selector.configuration),
                              'max_wall_seconds': max_wall_seconds,
                              'absolute_stage_deadline': True,
                              'durable_turn_journal': 'policy-turns.jsonl'}

    @property
    def receipts(self):
        return self.selector.receipts

    def __call__(self, observation, info):
        self.turns += 1
        public = public_policy_observation(self.suite, observation, info)
        before = len(self.selector.receipts)
        base = {'turn': self.turns, 'task_id': info['task_id'],
                'evaluation_episode_index': info.get('evaluation_episode_index'),
                'evaluation_seed': info.get('evaluation_seed'), 'trials_used': info['trials_used']}
        self.journal.append({**base, 'event': 'started', 'public_observation': public})
        try:
            if self.clock() >= self.deadline:
                self.deadline_rejections += 1
                raise PolicyActionRejected('policy_stage_wall_deadline')
            action = self.selector(observation, info)
        except BaseException as exc:
            receipt = self.selector.receipts[-1] if len(self.selector.receipts) > before else None
            # Exception text is deliberately excluded, including SDK request data.
            self.journal.append({**base, 'event': 'finished', 'status': 'rejected',
                'error_type': type(exc).__name__, 'api_receipt': sanitized_receipt(receipt)})
            raise
        self.journal.append({**base, 'event': 'finished', 'status': 'completed',
                             'action': action, 'api_receipt': sanitized_receipt(self.selector.receipts[-1])})
        return action


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--table', required=True, type=Path)
    p.add_argument('--suite', help='Optional explicit packaged suite ID or JSON path; otherwise resolve exact table suite')
    p.add_argument('--study-protocol', type=Path, help='Optional frozen generalization-study protocol to enforce')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--key-file', required=True, type=Path)
    p.add_argument('--private-ledger', required=True, type=Path)
    p.add_argument('--model', required=True)
    p.add_argument('--input-usd-per-million', required=True)
    p.add_argument('--output-usd-per-million', required=True)
    p.add_argument('--cap-usd', required=True)
    p.add_argument('--repeats', default=3, type=int)
    p.add_argument('--seed', default=1901, type=int)
    p.add_argument('--split', choices=('train', 'validation', 'test'), default='test')
    p.add_argument('--trials', default=4, type=int)
    p.add_argument('--max-wall-seconds', default=600, type=float)
    args = p.parse_args()
    if not 0 < args.max_wall_seconds <= 3600:
        p.error('max-wall-seconds must be in (0,3600]')
    stage_started = time.monotonic()
    stage_deadline = stage_started + args.max_wall_seconds
    repo = Path(__file__).resolve().parents[3]
    key_path = args.key_file.expanduser()
    ledger_path = args.private_ledger.expanduser()
    if (key_path.is_symlink() or key_path.resolve().is_relative_to(repo)
            or ledger_path.resolve().is_relative_to(repo)
            or stat.S_IMODE(key_path.stat().st_mode) != 0o600
            or key_path.stat().st_uid != os.getuid()):
        raise ValueError('Use an owned mode-0600 key file and a new private ledger outside the repository')
    table = MeasuredTable.load(args.table, suite=load_suite(args.suite) if args.suite else None)
    if not 1 <= args.repeats <= 20 or not 1 <= args.trials < len(table.suite.candidates):
        raise ValueError('Use 1..20 repeats and a trial budget smaller than the candidate set')
    calls = len(table.suite.splits[args.split]) * args.repeats * args.trials
    study_protocol_sha256 = None
    if args.study_protocol is not None:
        from drivebench_tuning.generalization_cli import read_json, validate_protocol
        from drivebench_tuning.generalization import digest
        frozen = validate_protocol(read_json(args.study_protocol), table.suite)
        supplement = frozen.get('llm_supplement')
        specs = [m for m in (supplement or {}).get('models', []) if m['model'] == args.model]
        if (len(specs) != 1 or args.split != 'test' or args.seed != frozen['seed']
                or args.repeats != supplement['repeats'] or args.trials != supplement['trials']
                or args.max_wall_seconds != supplement['max_wall_seconds']
                or calls != supplement['max_calls_per_model']
                or table._document['protocol']['source_hashes'] != frozen['source_hashes']):
            raise ValueError('API run differs from the frozen study protocol')
        spec = specs[0]
        if any(Decimal(actual) != Decimal(spec[field]) for field, actual in (
            ('input_usd_per_million', args.input_usd_per_million),
            ('output_usd_per_million', args.output_usd_per_million), ('cap_usd', args.cap_usd))):
            raise ValueError('API prices or reservation cap differ from frozen study')
        study_protocol_sha256 = digest(frozen)
    policy = APICostPolicy(budget_id=args.output.name, model=args.model,
        input_usd_per_million_tokens=args.input_usd_per_million,
        output_usd_per_million_tokens=args.output_usd_per_million,
        hard_usd_cap=args.cap_usd, max_output_tokens=1024,
        input_bound_kind='explicit_upper_bound',
        input_bound_id='canonical-utf8-bytes-plus-2048-provider-framing-v1')
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/'plan.json', {'table_sha256': table.sha256, 'policy': policy.to_dict(),
        'frozen_study_protocol_sha256': study_protocol_sha256,
        'max_calls': calls, 'max_wall_seconds': args.max_wall_seconds, 'split': args.split, 'repeats': args.repeats, 'seed': args.seed,
        'trial_budget': args.trials, 'backend': 'measured_replay', 'training_performed': False,
        'budget_note': 'Full upper-bound reservations retained; reported usage costs are separate estimates.'})
    # Explicit conservative token bound covers the complete request, including
    # schema and instructions, plus 2048 tokens for provider framing.
    def input_bound(request):
        return len(json.dumps(request, sort_keys=True, separators=(',', ':'),
                              ensure_ascii=False, allow_nan=False).encode()) + 2048
    ledger = APICostLedger.create(ledger_path, policy=policy, input_token_bound=input_bound)
    client = None
    turn_journal = DurableTurns(args.output/'policy-turns.jsonl')
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key_path.read_text().strip(), max_retries=0, timeout=60)
        selector = ResponsesIndexPolicy(BudgetedResponsesClient(DeadlineTransport(client, stage_deadline), ledger), suite=table.suite,
            model=args.model, max_calls=calls, max_output_tokens=1024)
        durable_selector = DurableDeadlinePolicy(selector, table.suite, turn_journal, stage_deadline,
            max_wall_seconds=args.max_wall_seconds)
        result = run_replay_policy(table, durable_selector, name=args.model, split=args.split,
            repeats=args.repeats, seed=args.seed, trials=args.trials)
        usage_cost = Decimal('0')
        known_usage = 0
        for receipt in result['policy_api_receipts']:
            usage = receipt.get('usage')
            if isinstance(usage, dict) and all(type(usage.get(k)) is int for k in ('input_tokens', 'output_tokens')):
                usage_cost += (Decimal(usage['input_tokens']) * Decimal(args.input_usd_per_million)
                               + Decimal(usage['output_tokens']) * Decimal(args.output_usd_per_million)) / 1_000_000
                known_usage += 1
        result['stage_wall_seconds'] = time.monotonic() - stage_started
        if study_protocol_sha256 is not None:
            result['frozen_study_protocol_sha256'] = study_protocol_sha256
        result['deadline_rejections'] = durable_selector.deadline_rejections
        result['api_cost'] = {'reserved_usd': ledger.state['charged_usd'],
            'reported_usage_estimated_usd': str(usage_cost), 'usage_receipts': known_usage,
            'attempts': ledger.state['attempts'], 'cap_usd': args.cap_usd,
            'cached_input_charged_at_full_price_in_estimate': True}
        write_json(args.output/'result.json', result)
        write_json(args.output/'confirmation-request.json', selection_manifest(table, [result]))
        complete = result['status'] == 'completed'
        write_json(args.output/('completed.json' if complete else 'failed.json'),
            {'completed': complete, 'backend': 'measured_replay', 'fresh_gpu_measurements': 0,
             'api_calls_attempted': ledger.state['attempts'], 'reserved_usd': ledger.state['charged_usd']})
        print(json.dumps({'status': result['status'], 'api_calls_attempted': ledger.state['attempts'],
                          'reserved_usd': ledger.state['charged_usd']}))
        return 0 if complete else 1
    except BaseException as exc:
        write_json(args.output/'interrupted.json', {'completed': False, 'error_type': type(exc).__name__,
            'stage_wall_seconds': time.monotonic() - stage_started,
            'api_calls_attempted': ledger.state['attempts'], 'reserved_usd': ledger.state['charged_usd'],
            'durable_turn_journal': 'policy-turns.jsonl', 'fresh_gpu_measurements': 0})
        raise
    finally:
        turn_journal.close()
        ledger.close()
        if client is not None:
            client.close()


if __name__ == '__main__':
    raise SystemExit(main())
