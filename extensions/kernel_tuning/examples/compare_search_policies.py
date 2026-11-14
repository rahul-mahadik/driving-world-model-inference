"""Serial, equal-trial grid/random example. Search outcomes need fresh confirmation."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from drivebench_tuning.adapters import OperatorEvaluator, SyntheticEvaluator
from drivebench_tuning.benchmark import CudaEvaluator, EvidenceJournal, source_hashes, utc_now
from drivebench_tuning.evaluation import evaluate_policy, grid_policy, random_policy
from drivebench_tuning.search import BudgetedTuningEnv
from drivebench_tuning.suite import load_suite


def run(output, *, backend='synthetic', split='validation', repeats=2, trials=4,
        seed=1901, max_seconds=900):
    if backend not in ('synthetic', 'cuda') or split not in ('train', 'validation', 'test'):
        raise ValueError('explicit supported backend and split required')
    if type(repeats) is not int or not 2 <= repeats <= 100 or repeats % 2:
        raise ValueError('use an even repeat count in 2..100 for balanced policy order')
    suite = load_suite()
    if type(trials) is not int or not 0 < trials < len(suite.candidates):
        raise ValueError('trial budget must be smaller than the candidate set')
    if type(seed) is not int or not 0 <= seed < 2**32 - 10000:
        raise ValueError('bounded nonnegative integer seed required')
    if type(max_seconds) is not int or not 1 <= max_seconds <= 7200:
        raise ValueError('max_seconds must be in 1..7200; use an outer process timeout')
    schedule = [dict(task_index=task, repeat=rep, policy=name,
                     episode_seed=seed + task * repeats + rep)
                for task in suite.splits[split] for rep in range(repeats)
                for name in (('grid', 'random') if rep % 2 == 0 else ('random', 'grid'))]
    journal = EvidenceJournal(Path(output))
    journal.write('plan.json', {'created_at': utc_now(), 'backend': backend,
        'synthetic': backend == 'synthetic', 'split': split, 'suite': suite.document,
        'suite_sha256': suite.sha256, 'source_hashes': source_hashes(),
        'example_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'schedule': schedule, 'candidate_trials_per_episode': trials,
        'reference_trials_per_episode': 1, 'warmups': 5, 'timed_repetitions': 30,
        'max_inner_cuda_seconds': max_seconds, 'training_performed': False,
        'training_calls': 0, 'selected_config_confirmation_performed': False,
        'policy_superiority_established': False})
    evaluator = None
    results = []
    started = time.monotonic()
    try:
        evaluator = (SyntheticEvaluator(journal) if backend == 'synthetic' else
                     OperatorEvaluator(CudaEvaluator(journal, repetitions=30, warmups=5,
                                                      max_seconds=max_seconds)))
        cohort, seen_trials, task_inputs = None, set(), {}
        for index, item in enumerate(schedule):
            policy = grid_policy if item['policy'] == 'grid' else random_policy(seed)
            def factory():
                return BudgetedTuningEnv(suite.tasks, suite.candidates, evaluator,
                    trial_budget=trials, memory_weight=0, allow_synthetic=backend == 'synthetic')
            report = evaluate_policy(factory, policy, [item['task_index']], repeats=1,
                                     seed=item['episode_seed'], split=split,
                                     training_metadata={'performed': False, 'calls': 0})
            results.append({'schedule': item, 'report': report})
            # Save before checking cross-policy bindings, preserving rejected evidence.
            journal.write(f'episode-{index:04d}.json', results[-1])
            episode = report['episodes'][0]
            identity = (episode['initial_info']['backend'], episode['initial_info']['cohort_id'])
            if cohort is not None and identity != cohort:
                raise ValueError('policy comparison changed its runtime cohort')
            cohort = identity
            task_id = episode['task_id']
            input_digest = episode['initial_info']['input_digest']
            if task_id in task_inputs and task_inputs[task_id] != input_digest:
                raise ValueError('policy comparison changed a task input')
            task_inputs[task_id] = input_digest
            for entry in episode['journal']:
                measurement = entry.get('measurement')
                if measurement:
                    if measurement['trial_id'] in seen_trials:
                        raise ValueError('measurement reused across policy episodes')
                    seen_trials.add(measurement['trial_id'])
        evaluator.close()
        evaluator = None
        totals = {name: {key: sum(x['report']['counts'][key] for x in results
                                 if x['schedule']['policy'] == name)
                        for key in ('baseline_calls', 'candidate_calls', 'failed_or_incorrect_trials')}
                  for name in ('grid', 'random')}
        report = {'status': 'completed', 'synthetic': backend == 'synthetic',
            'cohort': cohort, 'wall_seconds': time.monotonic() - started,
            'counts_by_policy': totals, 'episodes': results,
            'selected_config_confirmation_performed': False,
            'policy_superiority_established': False,
            'scope': 'Observed finite-budget search outcomes; no pooled speedup or confirmed winner.'}
        journal.write('comparison.json', report)
        journal.write('completed.json', {'completed': True, 'at': utc_now(),
                                         'synthetic': backend == 'synthetic'})
        return report
    except Exception as exc:
        journal.write('failed.json', {'completed': False, 'at': utc_now(),
            'error_type': type(exc).__name__, 'error': str(exc),
            'finished_episodes': results, 'partial_episode': getattr(exc, 'report', None)})
        raise
    finally:
        if evaluator is not None:
            try:
                evaluator.close()
            except Exception as exc:
                journal.write('cleanup-failed.json', {'error_type': type(exc).__name__, 'error': str(exc)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--backend', choices=('synthetic', 'cuda'), default='synthetic')
    parser.add_argument('--split', choices=('train', 'validation', 'test'), default='validation')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--trials', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1901)
    parser.add_argument('--max-seconds', type=int, default=900)
    report = run(**vars(parser.parse_args()))
    print(json.dumps({k: v for k, v in report.items() if k != 'episodes'}, indent=2))


if __name__ == '__main__':
    main()
