"""Run locally on an already admitted GPU; never provisions compute or funding."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

from .benchmark import CudaEvaluator, EvidenceJournal, source_hashes, utc_now
from .contract import DEFAULT_CONFIGS
from .study import load_workloads, run_sweep


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['benchmark', 'tune'])
    p.add_argument('--workloads', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path, help='New directory; existing paths are refused')
    p.add_argument('--max-seconds', type=int, default=900, help='Inner budget; an outer process timeout is also required')
    p.add_argument('--repetitions', type=int, default=30)
    p.add_argument('--warmups', type=int, default=5)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--episodes', type=int, default=12)
    p.add_argument('--trials', type=int, default=8)
    p.add_argument('--seed', type=int, default=1901)
    p.add_argument('--memory-weight', type=float, default=.25)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    document, workloads = load_workloads(args.workloads)
    if not 1 <= args.max_seconds <= 7200 or not 1 <= args.episodes <= 100 or not 1 <= args.trials <= 64:
        raise ValueError('Use max-seconds 1..7200, episodes 1..100, trials 1..64')
    journal = EvidenceJournal(args.output)
    journal.write('plan.json', {'created_at': utc_now(), 'command': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        'workload_document': document, 'workload_file_sha256': hashlib.sha256(args.workloads.read_bytes()).hexdigest(),
        'configs': [asdict(c) for c in DEFAULT_CONFIGS], 'source_hashes': source_hashes(),
        'baseline': 'same-cohort eager operator', 'compiled_control': args.mode == 'benchmark',
        'historical_timing_reused_as_kernel_evidence': False,
        'training_and_evaluation_workloads': 'same workload keys; fresh measurements, no generalization claim'})
    evaluator = None
    try:
        evaluator = CudaEvaluator(journal, repetitions=args.repetitions, warmups=args.warmups, max_seconds=args.max_seconds)
        if args.mode == 'benchmark':
            result = run_sweep(evaluator, workloads, rounds=args.rounds, seed=args.seed)
            journal.write('result.json', result)
        else:
            from .env import KernelTuningEnv
            from .policy import train_q_learning, evaluate_policies
            # Individual environments own their search state. The shared evaluator
            # owns one serial GPU/runtime and an absolute wall-time budget; its close
            # releases tensors but does not reset that budget or its append-only log.
            def make_env():
                return KernelTuningEnv(workloads, DEFAULT_CONFIGS, evaluator, trial_budget=args.trials,
                                       memory_weight=args.memory_weight)
            env = make_env()
            try:
                trained = train_q_learning(env, episodes=args.episodes, seed=args.seed)
            finally:
                env.close()
            journal.write('training.json', trained.to_dict())
            evaluation = evaluate_policies(make_env, trained.policy, episodes_per_workload=2, seed=args.seed+1000)
            journal.write('evaluation.json', evaluation)
        cohort_id = evaluator.cohort_id
        evaluator.close()
        evaluator = None
        journal.write('completed.json', {'at': utc_now(), 'completed': True, 'backend': 'cuda',
                      'cohort_id': cohort_id, 'mode': args.mode,
                      'whole_model_kernel_benefit_validated': False, 'quality_preserving': False,
                      'source_hashes': source_hashes()})
        print(json.dumps({'completed': True, 'directory': str(args.output.resolve())}))
        return 0
    except Exception as exc:
        journal.write('failed.json', {'at': utc_now(), 'completed': False,
                                     'error_type': type(exc).__name__, 'error': str(exc)})
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    finally:
        if evaluator is not None:
            try:
                evaluator.close()
            except Exception as cleanup_exc:
                journal.write('cleanup-failed.json', {'at': utc_now(), 'error_type': type(cleanup_exc).__name__,
                                                     'error': str(cleanup_exc)})
                print('CUDA cleanup failed; preserve the failure log and end the outer job.', file=sys.stderr)


if __name__ == '__main__':
    raise SystemExit(main())
