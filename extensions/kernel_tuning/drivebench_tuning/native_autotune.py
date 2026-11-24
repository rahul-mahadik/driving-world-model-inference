"""Native Triton autotuning and live, pretrained PPO on isolated operators.

The native Triton 3.6 Autotuner selects among the existing 24 configurations.
A small ``run`` adapter treats GroupNorm's three kernels as one operation; it
does not change their math. The documented ``do_bench`` hook uses the same CUDA
event boundary as the existing evaluator. Selection samples are never reused
as confirmation samples. This is not an equal-search-budget comparison:
native tuning evaluates 24 configurations and PPO has eight query turns.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

from .adapters import OperatorEvaluator, _digest
from .benchmark import CudaEvaluator, EvidenceJournal, percentile, source_hashes, utc_now
from .contract import DTYPE_TOLERANCES, Workload
from .evaluation import _json
from .search import BudgetedTuningEnv
from .suite import load_suite


PROTOCOL = {
    'version': 'ppo-native-autotune-v1', 'suite_id': 'operator-generalization-v2',
    'split': 'test', 'native_configurations': 24, 'ppo_trial_budget': 8,
    'native_engine': 'triton.autotune / triton.runtime.autotuner.Autotuner',
    'native_adapter': 'whole-operator run adapter around unchanged fused operator wrappers',
    'native_do_bench': '5 warmups / 30 separate synchronized CUDA-event samples; median, p20, p80',
    'correctness': 'unchanged dtype-specific operator numerical gate before tuning each configuration',
    'autotune_disk_results': False, 'fresh_tuner_per_task_and_process': True,
    'selector_order': 'alternate native/PPO by repeat_index + shuffled task position',
    'compilation_cache': 'shared within process; actual selector order disclosed; host owns fresh-process cache directories',
    'native_selection_wall_includes': 'input preparation, first-call compilation, correctness, warmups, samples and final launch',
    'native_reference_fallback': 'three fresh selection probes of eager/compiled/native; choose fastest correct probe before any confirmation',
    'ppo_selection_wall_includes': 'fresh eager reset, eight real GPU queries and CPU policy inference',
    'confirmation': 'fresh eager, compiled, native-selected and PPO-selected measurements after both selectors finish',
    'confirmation_order': 'rotate four roles by repeat_index + shuffled task position',
    'equal_search_budget': False, 'fresh_oracle_measured': False,
    'regret_claim_from_live_confirmation': False, 'video_quality_preserving': False,
}


def native_candidates(suite):
    """Require the exact published 6-block by 4-warp grid, without pruning."""
    rows = tuple(c for c in suite.candidates if c.config.get('implementation') == 'triton')
    observed = {(c.config['block_size'], c.config['num_warps']) for c in rows}
    expected = {(b, w) for b in (64, 128, 256, 512, 1024, 2048) for w in (1, 2, 4, 8)}
    if len(rows) != 24 or observed != expected:
        raise ValueError('Native comparison requires the exact 24-configuration published grid')
    return rows


def validate_input_bindings(document, suite):
    if document.get('suite_sha256') != suite.sha256:
        raise ValueError('Input binding suite differs from the frozen suite')
    values = document.get('expected_input_digests')
    if not isinstance(values, dict):
        raise ValueError('Expected measured input digests required')
    result = {}
    for index in suite.splits['test']:
        key = suite.tasks[index].task_id
        value = values.get(key)
        if (not isinstance(value, str) or len(value) != 64
                or any(c not in '0123456789abcdef' for c in value)):
            raise ValueError('Every test task needs its original SHA256 input binding')
        result[key] = value
    return result


def _operator_marker():
    """Base-function identity for Triton's whole-operator run protocol."""


class WholeOperatorLaunch:
    """Triton launch protocol for the unchanged allocation + kernel pipeline.

    Triton's pinned Autotuner calls ``run`` with Config metadata. Its ``fn``
    attribute traversal is used only for identity; disk autotune caching is off.
    This explicit adapter is needed because GroupNorm consists of three JIT
    kernels whose block/warp settings must change together.
    """
    arg_names = ['X', 'Weight', 'Bias', 'TASK_ID']
    fn = staticmethod(_operator_marker)

    def __init__(self, workload, candidate_map, on_configuration_start=None):
        self.workload, self.candidate_map = workload, candidate_map
        self.current = None
        self.last_output = None
        self._started = set()
        self._on_configuration_start = on_configuration_start

    def run(self, X, Weight, Bias, TASK_ID, *, BLOCK_SIZE, num_warps,
            num_stages=3, num_ctas=1, maxnreg=None, ir_override=None):
        if (num_stages, num_ctas, maxnreg, ir_override) != (3, 1, None, None):
            raise ValueError('Unexpected native tuning compiler metadata')
        key = (BLOCK_SIZE, num_warps)
        if key not in self.candidate_map:
            raise ValueError('Native tuner supplied a config outside the declared grid')
        self.current = self.candidate_map[key]
        if key not in self._started:
            self._started.add(key)
            if self._on_configuration_start is not None:
                self._on_configuration_start(self.current)
        self.last_output = None
        from .kernels import fused_geglu, fused_groupnorm_silu
        if self.workload.operator == 'geglu':
            self.last_output = fused_geglu(X, block_size=BLOCK_SIZE, num_warps=num_warps)
        else:
            self.last_output = fused_groupnorm_silu(X, Weight, Bias,
                num_groups=self.workload.num_groups, eps=self.workload.eps,
                block_size=BLOCK_SIZE, num_warps=num_warps)
        return self.last_output


def _verify_candidate_output(runner, launcher, workload):
    torch = runner.torch
    actual = launcher.last_output.detach().cpu()
    reference = runner._expected
    tolerance = DTYPE_TOLERANCES[workload.dtype]
    shape = actual.shape == reference.shape and actual.dtype == reference.dtype
    finite = bool(torch.isfinite(actual).all().item())
    passed = shape and finite and bool(torch.allclose(actual, reference, **tolerance))
    error = (actual.float() - reference.float()).abs() if shape and finite else None
    return {'passed': passed, 'shape_dtype_match': shape, 'finite': finite, **tolerance,
            'max_abs_error': float(error.max()) if error is not None else None,
            'rmse': float(error.square().mean().sqrt()) if error is not None else None}


def native_select(runner, task, candidates, expected_digest, journal):
    """Run a new native Autotuner, retaining every tuning sample and failure."""
    import triton
    if triton.__version__ != '3.6.0':
        raise RuntimeError('Pinned Triton 3.6.0 required')
    workload = Workload(**task.to_dict()['metadata']['workload'])
    torch = runner.torch
    started = time.perf_counter()
    runner._prepare(workload)
    if _digest(runner._input_hashes) != expected_digest:
        raise ValueError('Native selection inputs differ from measured table')
    candidate_map = {(c.config['block_size'], c.config['num_warps']): c for c in candidates}
    launcher = WholeOperatorLaunch(workload, candidate_map, lambda c: journal.append({
        'event': 'native_tuning_candidate_started', 'task_id': task.task_id,
        'candidate_id': c.candidate_id, 'config': c.to_dict()['config'],
        'cohort_id': runner.cohort_id, 'input_digest': expected_digest}))
    tuning_rows = []

    def bench(fn, quantiles):
        runner._check_deadline()
        row = {'task_id': task.task_id, 'candidate_id': None, 'status': 'started',
               'cohort_id': runner.cohort_id, 'input_digest': expected_digest}
        prep = time.perf_counter()
        try:
            fn()  # Compilation and a correctness output, outside timed samples.
            row['candidate_id'] = launcher.current.candidate_id
            row['config'] = launcher.current.to_dict()['config']
            torch.cuda.synchronize(runner.device)
            row['first_invocation_seconds'] = time.perf_counter() - prep
            row['correctness'] = _verify_candidate_output(runner, launcher, workload)
            runner._verify_inputs()
            if not row['correctness']['passed']:
                row['status'] = 'incorrect'
                return [float('inf')] * len(quantiles)
            for _ in range(runner.warmups):
                runner._check_deadline()
                fn()
            torch.cuda.synchronize(runner.device)
            row['preparation_seconds'] = time.perf_counter() - prep
            samples = []
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record(); end.record(); end.synchronize()
            for _ in range(runner.repetitions):
                runner._check_deadline()
                torch.cuda.synchronize(runner.device)
                launcher.last_output = None  # Release prior output outside timing, like CudaEvaluator.
                begin.record()
                fn()
                end.record(); end.synchronize()
                samples.append(float(begin.elapsed_time(end)))
            runner._verify_inputs()
            if any(not math.isfinite(x) or x <= 0 for x in samples):
                raise RuntimeError('Invalid native autotune CUDA-event sample')
            row.update(status='ok', cuda_event_ms=samples, median_ms=statistics.median(samples),
                       input_integrity_verified=True)
            return [percentile(samples, q) for q in quantiles]
        except Exception as exc:
            if launcher.current is not None:
                row.update(candidate_id=launcher.current.candidate_id,
                           config=launcher.current.to_dict()['config'])
            row.update(status='error', error_type=type(exc).__name__, error=str(exc))
            if isinstance(exc, (torch.cuda.OutOfMemoryError, runner._out_of_resources)):
                torch.cuda.empty_cache()
                return [float('inf')] * len(quantiles)
            raise
        finally:
            row['selection_candidate_wall_seconds'] = time.perf_counter() - prep
            launcher.last_output = None
            tuning_rows.append(row)
            journal.append({'event': 'native_tuning_candidate', 'record': row})

    configs = [triton.Config({'BLOCK_SIZE': c.config['block_size']},
                            num_warps=c.config['num_warps']) for c in candidates]
    tuner = triton.autotune(configs=configs, key=['TASK_ID'], do_bench=bench,
                           cache_results=False)(launcher)
    if tuner.cache or tuner.cache_results:
        raise RuntimeError('Native tuner must begin with an empty decision cache')
    inputs = runner._inputs
    with torch.cuda.device(runner.device), torch.no_grad():
        tuner.run(inputs[0], inputs[1] if len(inputs) > 1 else None,
                  inputs[2] if len(inputs) > 2 else None, task.task_id)
        torch.cuda.synchronize(runner.device)
        runner._verify_inputs()
        final_check = _verify_candidate_output(runner, launcher, workload)
        launcher.last_output = None
    # An all-infinite grid is a failed selection, not a viable first config.
    best = tuner.best_config
    selected = candidate_map[(best.kwargs['BLOCK_SIZE'], best.num_warps)]
    selected_rows = [r for r in tuning_rows if r['candidate_id'] == selected.candidate_id]
    if len(tuning_rows) != len(candidates) or len({r['candidate_id'] for r in tuning_rows}) != len(candidates):
        raise RuntimeError('Native tuner did not attempt every declared configuration exactly once')
    if not selected_rows or selected_rows[0]['status'] != 'ok' or not final_check['passed']:
        raise RuntimeError('Native tuning found no correct admissible selection')
    native_timings = [{'candidate_id': candidate_map[(c.kwargs['BLOCK_SIZE'], c.num_warps)].candidate_id,
                      'quantiles_ms': [v if math.isfinite(v) else None for v in values]}
                     for c, values in tuner.configs_timings.items()]
    result = {'task_id': task.task_id, 'selected_candidate_id': selected.candidate_id,
              'cohort_id': runner.cohort_id, 'input_digest': expected_digest,
              'selection_wall_seconds': time.perf_counter() - started,
              'native_bench_time_seconds': tuner.bench_time,
              'configs_attempted': len(tuning_rows), 'records': tuning_rows,
              'native_configs_timings': native_timings, 'selected_correctness': final_check,
              'autotune_cache_results': bool(tuner.cache_results), 'initial_decision_cache_empty': True}
    journal.append({'event': 'native_selection_completed', 'result': result})
    return result


def ppo_select(env, policy, task_index, expected_digest, repeat_index, journal):
    started = time.perf_counter()
    observation, info = env.reset(seed=4101 + repeat_index, options={'task_index': task_index})
    if info['input_digest'] != expected_digest or info['backend'] != 'cuda':
        raise ValueError('PPO selection input/backend differs from frozen measured task')
    initial = _json(info)
    initial_observation = _json(observation)
    turns = []
    for turn in range(8):
        preaction_observation = _json(observation)
        inference_started = time.perf_counter()
        action = int(policy(observation, info))
        inference_seconds = time.perf_counter() - inference_started
        observation, reward, terminated, truncated, info = env.step(action)
        if truncated or terminated != (turn == 7):
            raise RuntimeError('PPO live environment changed its eight-turn boundary')
        row = {'turn': turn, 'action': action, 'reward': reward,
               'preaction_observation': preaction_observation,
               'cpu_policy_seconds': inference_seconds, 'observation': _json(observation), 'info': _json(info)}
        turns.append(row)
        journal.append({'event': 'ppo_live_turn', 'task_id': info['task_id'], 'record': row})
    result = {'task_id': info['task_id'], 'selected_candidate_id': info['incumbent_candidate_id'],
              'initial_info': initial, 'initial_observation': initial_observation,
              'turns': turns, 'final_info': _json(info),
              'candidate_turns': len(turns), 'selection_wall_seconds': time.perf_counter() - started,
              'policy_updates_permitted': False}
    journal.append({'event': 'ppo_live_selection_completed', 'result': result})
    return result


def confirm_selections(evaluator, task, selections, *, expected_digest, offset, journal):
    """Always make four new measurements, including duplicate selected roles."""
    roles = ['eager', 'compiled', 'native', 'ppo']
    roles = roles[offset % 4:] + roles[:offset % 4]
    rows = {}
    ids = set()
    for role in roles:
        candidate = selections[role]
        trial = evaluator.baseline(task) if candidate is None else evaluator.evaluate(task, candidate)
        if (trial.backend != 'cuda' or trial.input_digest != expected_digest or trial.task_id != task.task_id
                or trial.candidate_id != (candidate.candidate_id if candidate else None)
                or trial.cohort_id != evaluator.runner.cohort_id or trial.trial_id in ids):
            raise ValueError('Fresh confirmation identity/input/cohort binding failed')
        ids.add(trial.trial_id)
        rows[role] = trial.to_dict()
        journal.append({'event': 'autotune_ppo_confirmation', 'task_id': task.task_id,
                        'role': role, 'measurement': rows[role]})
    return {'order': roles, 'measurements': rows}


def reference_fallback(evaluator, task, native_candidate, compiled_candidate, *, expected_digest, offset, journal):
    """Select a native/reference fallback from fresh probes, before confirmation."""
    started = time.perf_counter()
    roles = ['eager', 'compiled', 'native']
    roles = roles[offset % 3:] + roles[:offset % 3]
    candidates = {'eager': None, 'compiled': compiled_candidate, 'native': native_candidate}
    probes = {}
    for role in roles:
        candidate = candidates[role]
        trial = evaluator.baseline(task) if candidate is None else evaluator.evaluate(task, candidate)
        if (trial.backend != 'cuda' or trial.task_id != task.task_id or trial.input_digest != expected_digest
                or trial.cohort_id != evaluator.runner.cohort_id
                or trial.candidate_id != (candidate.candidate_id if candidate else None)):
            raise ValueError('Native reference fallback probe binding failed')
        probes[role] = trial.to_dict()
        journal.append({'event': 'native_reference_fallback_probe', 'task_id': task.task_id,
                        'role': role, 'measurement': probes[role]})
    admissible = [r for r in ['eager', 'compiled', 'native'] if probes[r]['correct']]
    if not admissible:
        raise RuntimeError('No correct native/reference fallback probe')
    selected = min(admissible, key=lambda r: probes[r]['latency_ms'])
    result = {'order': roles, 'probes': probes, 'selected_role': selected,
              'selected_candidate_id': probes[selected]['candidate_id'],
              'selection_wall_seconds': time.perf_counter() - started,
              'selected_before_confirmation': True}
    journal.append({'event': 'native_reference_fallback_selected', 'task_id': task.task_id, 'result': result})
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', default='operator-generalization-v2')
    p.add_argument('--ppo-checkpoint', required=True, type=Path)
    p.add_argument('--ppo-manifest', required=True, type=Path)
    p.add_argument('--input-bindings', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--repeat-index', required=True, type=int)
    p.add_argument('--max-seconds', default=1800, type=int)
    p.add_argument('--repetitions', default=30, type=int)
    p.add_argument('--warmups', default=5, type=int)
    args = p.parse_args(argv)
    if (args.repeat_index not in (0, 1, 2) or not 1 <= args.max_seconds <= 7200
            or args.repetitions != 30 or args.warmups != 5):
        p.error('This supplement requires repeat0..2, thirty samples and five warmups')
    suite = load_suite(args.suite)
    if suite.document['suite_id'] != PROTOCOL['suite_id']:
        p.error('Only the frozen generalization suite is supported')
    candidates = native_candidates(suite)
    bindings_raw = args.input_bindings.read_bytes()
    bindings = validate_input_bindings(json.loads(bindings_raw), suite)
    from .ppo import PPOSelector
    policy = PPOSelector.load(args.ppo_checkpoint, args.ppo_manifest, suite)
    if (policy.manifest.get('seed') != 3101
            or policy.manifest.get('training', {}).get('performed') is not True):
        raise ValueError('Live supplement requires the prospectively selected trained PPO seed3101')
    journal = EvidenceJournal(args.output)
    journal.write('plan.json', {'protocol': PROTOCOL, 'created_at': utc_now(),
        'suite_sha256': suite.sha256, 'source_hashes': source_hashes(),
        'repeat_index': args.repeat_index, 'max_seconds': args.max_seconds,
        'checkpoint_sha256': hashlib.sha256(args.ppo_checkpoint.read_bytes()).hexdigest(),
        'ppo_manifest_sha256': hashlib.sha256(args.ppo_manifest.read_bytes()).hexdigest(),
        'input_bindings_sha256': hashlib.sha256(bindings_raw).hexdigest(),
        'expected_input_digests': bindings, 'native_candidate_ids': [c.candidate_id for c in candidates]})
    runner = None
    try:
        runner = CudaEvaluator(journal, repetitions=30, warmups=5, max_seconds=args.max_seconds)
        evaluator = OperatorEvaluator(runner)
        env = BudgetedTuningEnv(suite.tasks, suite.candidates, evaluator, trial_budget=8, memory_weight=0)
        by_id = {c.candidate_id: c for c in suite.candidates}
        compiled = next(c for c in suite.candidates if c.config['implementation'] == 'compiled')
        indices = list(suite.splits['test'])
        random.Random(5101 + args.repeat_index).shuffle(indices)
        results = []
        for position, index in enumerate(indices):
            runner._check_deadline()
            task = suite.tasks[index]
            order = ['native', 'ppo'] if (position + args.repeat_index) % 2 == 0 else ['ppo', 'native']
            chosen = {}
            for name in order:
                chosen[name] = (native_select(runner, task, candidates, bindings[task.task_id], journal)
                    if name == 'native' else ppo_select(env, policy, index, bindings[task.task_id], args.repeat_index, journal))
                if name == 'native':
                    chosen[name]['reference_fallback'] = reference_fallback(evaluator, task,
                        by_id[chosen[name]['selected_candidate_id']], compiled,
                        expected_digest=bindings[task.task_id], offset=position + args.repeat_index, journal=journal)
            selections = {'eager': None, 'compiled': compiled,
                          'native': by_id[chosen['native']['selected_candidate_id']],
                          'ppo': by_id.get(chosen['ppo']['selected_candidate_id'])}
            confirmation = confirm_selections(evaluator, task, selections,
                expected_digest=bindings[task.task_id], offset=position + args.repeat_index, journal=journal)
            row = {'task_id': task.task_id, 'selector_order': order, 'selectors': chosen,
                   'confirmation': confirmation}
            results.append(row)
            journal.append({'event': 'autotune_ppo_task_completed', 'result': row})
        runner.close(); runner = None
        journal.write('result.json', {'schema_version': 1, 'protocol': PROTOCOL,
            'suite_sha256': suite.sha256, 'cohort_id': evaluator.runner.cohort_id,
            'repeat_index': args.repeat_index, 'results': results,
            'scope': 'Isolated operator selection cost and fresh confirmation; no live oracle or video-quality claim'})
        journal.write('completed.json', {'completed': True, 'at': utc_now(), 'tasks': len(results),
            'native_config_trials': len(results) * 24, 'ppo_queries': len(results) * 8,
            'ppo_resets': len(results), 'confirmations': len(results) * 4,
            'reference_fallback_probes': len(results) * 3,
            'ppo_training_on_this_gpu': False})
        return 0
    except Exception as exc:
        journal.write('failed.json', {'completed': False, 'at': utc_now(),
            'error_type': type(exc).__name__, 'error': str(exc)})
        return 1
    finally:
        if runner is not None:
            try:
                runner.close()
            except Exception as exc:
                journal.write('cleanup-failed.json', {'error_type': type(exc).__name__, 'error': str(exc)})


if __name__ == '__main__':
    raise SystemExit(main())
