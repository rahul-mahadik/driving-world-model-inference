"""Portable configuration-search benchmark; never provisions compute."""
import argparse
import hashlib
from importlib.resources import files
import json
from pathlib import Path
import time

from .adapters import OperatorEvaluator, SyntheticEvaluator
from .benchmark import CudaEvaluator, EvidenceJournal, source_hashes, utc_now
from .evaluation import evaluate_policy, grid_policy, random_policy
from .search import BudgetedTuningEnv
from .suite import load_suite


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('describe', help='Print the bundled versioned suite and identity')
    export = sub.add_parser('export-suite', help='Copy bundled suite to a new file for custom experiments')
    export.add_argument('--output', required=True, type=Path)
    run = sub.add_parser('run', help='Evaluate a baseline search policy using fresh trials')
    run.add_argument('--suite', type=Path, help='JSON suite file; defaults to packaged operator-search-v1')
    run.add_argument('--backend', choices=('synthetic', 'cuda'), required=True)
    run.add_argument('--split', choices=('train', 'validation', 'test'), default='validation')
    run.add_argument('--policy', choices=('random', 'grid'), default='random')
    run.add_argument('--output', required=True, type=Path)
    run.add_argument('--trials', type=int, help='Candidate calls per task; default is suite budget')
    run.add_argument('--repeats', type=int, default=3)
    run.add_argument('--seed', type=int, default=1901)
    run.add_argument('--memory-weight', type=float, help='Default is latency-only; weights change task identity')
    run.add_argument('--max-seconds', type=int, default=900, help='CUDA total inner wall-time budget; use an outer timeout')
    run.add_argument('--repetitions', type=int, default=30)
    run.add_argument('--warmups', type=int, default=5)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == 'describe':
        suite = load_suite()
        print(json.dumps({'suite_sha256': suite.sha256, 'suite': suite.document}, indent=2))
        return 0
    if args.command == 'export-suite':
        with args.output.open('xb') as f:
            f.write(files('drivebench_tuning').joinpath('suites/operator-search-v1.json').read_bytes())
        print(json.dumps({'suite_path': str(args.output.resolve())}))
        return 0
    suite = load_suite(args.suite)
    trials = args.trials if args.trials is not None else suite.document['default_trial_budget']
    weight = args.memory_weight if args.memory_weight is not None else suite.document['default_memory_weight']
    if not 0 < trials < len(suite.candidates):
        raise ValueError('Search trials must be positive and smaller than the candidate set; exhaustive timing uses drivebench-tune benchmark')
    if not 1 <= args.repeats <= 100 or not 1 <= args.max_seconds <= 7200 or not 0 <= args.seed < 2**32:
        raise ValueError('Use repeats 1..100, max-seconds 1..7200 and uint32 seed')
    # Validate objective without creating a run or touching CUDA.
    probe = BudgetedTuningEnv(suite.tasks, suite.candidates, SyntheticEvaluator(),
                             trial_budget=trials, memory_weight=weight, allow_synthetic=True)
    probe.close()
    journal = EvidenceJournal(args.output)
    start = time.monotonic()
    plan = {'schema_version': 1, 'created_at': utc_now(), 'suite_id': suite.document['suite_id'],
            'suite_sha256': suite.sha256, 'suite': suite.document,
            'command': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
            'effective_trial_budget': trials, 'memory_weight': weight,
            'objective': 'best_correct_cost_within_candidate_budget', 'reward': 'incumbent_cost_improvement',
            'baseline_calls_per_episode': 1, 'source_hashes': source_hashes(),
            'synthetic': args.backend == 'synthetic', 'training_performed': False,
            'training_candidate_calls': 0, 'training_baseline_calls': 0,
            'test_split_policy': 'Open split. Use validation for choices; preregister policy before test. Not secret or historically untouched.',
            'selected_config_confirmation_performed': False}
    plan['protocol_sha256'] = hashlib.sha256(json.dumps({
        'suite_sha256': suite.sha256, 'trials': trials, 'memory_weight': weight,
        'warmups': args.warmups, 'repetitions': args.repetitions,
        'objective': plan['objective'], 'source_hashes': plan['source_hashes']}, sort_keys=True).encode()).hexdigest()
    journal.write('plan.json', plan)
    evaluator = None
    environments = []
    try:
        evaluator = (SyntheticEvaluator(journal) if args.backend == 'synthetic' else
                     OperatorEvaluator(CudaEvaluator(journal, repetitions=args.repetitions,
                                                      warmups=args.warmups, max_seconds=args.max_seconds)))
        def factory():
            env = BudgetedTuningEnv(suite.tasks, suite.candidates, evaluator, trial_budget=trials,
                                   memory_weight=weight, allow_synthetic=args.backend == 'synthetic')
            environments.append(env)
            return env
        policy = random_policy(args.seed) if args.policy == 'random' else grid_policy
        result = evaluate_policy(factory, policy, suite.splits[args.split], repeats=args.repeats,
                                 seed=args.seed, split=args.split)
        result['policy_name'] = args.policy
        result['suite_sha256'] = suite.sha256
        result['protocol_sha256'] = plan['protocol_sha256']
        result['wall_seconds'] = time.monotonic() - start
        journal.write('result.json', result)
        journal.write('environment-journals.json', [env.journal for env in environments])
        # Success is written only after cleanup. A failed close cannot leave a success marker.
        evaluator.close()
        evaluator = None
        journal.write('completed.json', {'completed': True, 'at': utc_now(), 'backend': args.backend,
                      'synthetic': args.backend == 'synthetic', 'suite_sha256': suite.sha256,
                      'protocol_sha256': plan['protocol_sha256'], 'rl_improvement_validated': False,
                      'quality_preserving': False, 'selected_config_confirmation_performed': False})
        print(json.dumps({'completed': True, 'synthetic': args.backend == 'synthetic',
                          'directory': str(args.output.resolve())}))
        return 0
    except Exception as exc:
        if hasattr(exc, 'report'):
            journal.write('partial-evaluation.json', exc.report)
        journal.write('failed.json', {'at': utc_now(), 'error_type': type(exc).__name__, 'error': str(exc),
                                     'completed': False, 'synthetic': args.backend == 'synthetic'})
        print(json.dumps({'completed': False, 'error_type': type(exc).__name__, 'error': str(exc)}))
        return 1
    finally:
        if not (args.output / 'environment-journals.json').exists():
            journal.write('environment-journals.json', [env.journal for env in environments])
        if evaluator is not None:
            try:
                evaluator.close()
            except Exception as exc:
                journal.write('cleanup-failed.json', {'error_type': type(exc).__name__, 'error': str(exc)})


if __name__ == '__main__':
    raise SystemExit(main())
