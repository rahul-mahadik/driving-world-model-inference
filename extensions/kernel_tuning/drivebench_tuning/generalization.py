"""Train-only supervised cost model and Optuna TPE configuration selectors.

The model predicts normalized operator latency from public task/configuration
metadata. This is supervised regression, not RL training. Neither selector owns
an evaluator or a measured table. Only training explicitly reads TRAIN cells.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .evaluation import PolicyActionRejected
from .suite import _strict_object, _reject_constant

FEATURE_VERSION = 'task-quadratic-launch-interactions-v1'
MODEL_KIND = 'train_only_ridge_log_cost'
OPTUNA_VERSION = '5.1.0'
RIDGE_ALPHA = 1.0
FAILURE_COST = 2.0


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def structured_tpe_space(suite):
    configs = [c.to_dict()['config'] for c in suite.candidates]
    triton = [c for c in configs if c['implementation'] == 'triton']
    if sum(c == {'implementation': 'compiled'} for c in configs) != 1 or not triton:
        raise ValueError('TPE requires one compiled candidate and a Triton configuration grid')
    blocks = sorted({int(math.log2(c['block_size'])) for c in triton})
    warps = sorted({int(math.log2(c['num_warps'])) for c in triton})
    if (blocks != list(range(blocks[0], blocks[-1] + 1))
            or warps != list(range(warps[0], warps[-1] + 1))
            or {(c['block_size'], c['num_warps']) for c in triton}
            != {(2**b, 2**w) for b in blocks for w in warps}):
        raise ValueError('TPE requires the full contiguous block/warp exponent grid')
    return {'implementation': ['compiled', 'triton'],
            'triton_block_exponent': [blocks[0], blocks[-1]],
            'triton_warp_exponent': [warps[0], warps[-1]]}


def _features(task, candidate):
    t = np.asarray(task.features, dtype=np.float64)
    config = candidate.to_dict()['config']
    implementation = config.get('implementation')
    if implementation == 'compiled':
        c = np.array([1., 0., 0.])
    elif implementation == 'triton':
        c = np.array([0., math.log2(config['block_size']) / 16., math.log2(config['num_warps']) / 5.])
    else:
        raise ValueError('cost model supports compiled and Triton launch configurations')
    values = np.concatenate([t, t * t, c, np.outer(t, c).ravel()])
    if not np.isfinite(values).all():
        raise ValueError('finite public task/configuration features required')
    return values


def _training_ids(suite):
    ids = [suite.tasks[i].task_id for i in suite.splits['train']]
    if any(suite.tasks[i].metadata.get('split') != 'train' for i in suite.splits['train']):
        raise ValueError('training split disagrees with task metadata')
    return ids


def train_cost_model(table, *, alpha=RIDGE_ALPHA):
    """Fit only TRAIN outcomes; fixed regularization, no validation/test tuning."""
    if type(alpha) not in (int, float) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('positive finite ridge regularization required')
    suite = table.suite
    training_ids = _training_ids(suite)
    if not set(training_ids).issubset({t.task_id for t in getattr(table, 'measured_tasks', suite.tasks)}):
        raise ValueError('table does not contain the complete training split')
    data, targets, records = [], [], []
    for task_index in suite.splits['train']:
        task = suite.tasks[task_index]
        references = [table.row(task.task_id, None, repeat) for repeat in table.repeat_indices]
        if not all(r['correct'] and r['status'] == 'ok' for r in references):
            raise ValueError('every training reference must pass')
        for candidate in suite.candidates:
            rows = [table.row(task.task_id, candidate.candidate_id, repeat) for repeat in table.repeat_indices]
            all_pass = all(r['correct'] and r['status'] == 'ok' for r in rows)
            ratios = [r['latency_ms'] / ref['latency_ms'] for r, ref in zip(rows, references)] if all_pass else []
            target = min(float(np.median(ratios)), FAILURE_COST) if all_pass else FAILURE_COST
            data.append(_features(task, candidate)); targets.append(math.log(target))
            records.append({'task_id': task.task_id, 'candidate_id': candidate.candidate_id,
                            'all_repeats_correct': all_pass, 'target_log_cost': math.log(target),
                            'source_trial_ids': [r['trial_id'] for r in rows],
                            'reference_trial_ids': [r['trial_id'] for r in references],
                            'input_digests': sorted({r['input_digest'] for r in rows + references})})
    x, y = np.asarray(data), np.asarray(targets)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-12] = 1.
    z = (x - mean) / scale
    intercept = float(y.mean())
    coefficients = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y - intercept))
    if not np.isfinite(coefficients).all():
        raise ValueError('nonfinite fitted model')
    return {'schema_version': 1, 'kind': MODEL_KIND, 'feature_version': FEATURE_VERSION,
            'suite_sha256': suite.sha256, 'candidates': [c.to_dict() for c in suite.candidates],
            'task_feature_count': len(suite.tasks[0].features), 'feature_count': x.shape[1],
            'parameters': {'alpha': float(alpha), 'failure_cost': FAILURE_COST,
                           'target': 'log(min(median paired latency ratio, 2)); failed cell -> log(2)',
                           'memory_weight': 0.0, 'hyperparameter_selection': 'fixed prospectively'},
            'mean': mean.tolist(), 'scale': scale.tolist(), 'coefficients': coefficients.tolist(),
            'intercept': intercept,
            'training': {'performed': True, 'method': 'supervised ridge regression', 'rl_training': False,
                         'training_split': 'train', 'task_ids': training_ids, 'cell_count': len(records),
                         'outcome_splits_read': ['train'], 'validation_or_test_outcomes_used': False,
                         'training_table_sha256': table.sha256, 'training_cohort_id': table.cohort_id,
                         'training_split_sha256': digest([suite.tasks[i].to_dict() for i in suite.splits['train']]),
                         'training_records_sha256': digest(records), 'records': records,
                         'numpy_version': np.__version__, 'gpu_measurements_performed_by_training': 0}}


def validate_cost_model(document, suite, *, expected_table_sha256=None, expected_cohort_id=None):
    """Reject incompatible/tampered structure; provenance is not a signature."""
    d = deepcopy(document)
    if (d.get('schema_version') != 1 or d.get('kind') != MODEL_KIND
            or d.get('feature_version') != FEATURE_VERSION or d.get('suite_sha256') != suite.sha256
            or d.get('candidates') != [c.to_dict() for c in suite.candidates]
            or d.get('task_feature_count') != len(suite.tasks[0].features)):
        raise ValueError('model and suite/feature/candidate identities differ')
    count = len(_features(suite.tasks[0], suite.candidates[0]))
    if d.get('feature_count') != count:
        raise ValueError('model feature dimension differs')
    for key in ('mean', 'scale', 'coefficients'):
        value = np.asarray(d.get(key), dtype=np.float64)
        if value.shape != (count,) or not np.isfinite(value).all():
            raise ValueError('finite model vectors with exact feature dimensions required')
        if key == 'scale' and np.any(value <= 0):
            raise ValueError('positive model normalization scales required')
    if type(d.get('intercept')) not in (int, float) or not math.isfinite(d['intercept']):
        raise ValueError('finite model intercept required')
    p, t = d.get('parameters', {}), d.get('training', {})
    if (type(p.get('alpha')) not in (int, float) or not math.isfinite(p['alpha']) or p['alpha'] <= 0
            or p.get('failure_cost') != FAILURE_COST or p.get('memory_weight') != 0.0):
        raise ValueError('unsupported model objective/hyperparameters')
    if (t.get('performed') is not True or t.get('rl_training') is not False
            or t.get('training_split') != 'train' or t.get('task_ids') != _training_ids(suite)
            or t.get('outcome_splits_read') != ['train'] or t.get('validation_or_test_outcomes_used') is not False
            or t.get('training_split_sha256') != digest([suite.tasks[i].to_dict() for i in suite.splits['train']])):
        raise ValueError('model is not bound exclusively to the declared training split')
    records = t.get('records', [])
    expected = {(task_id, c.candidate_id) for task_id in _training_ids(suite) for c in suite.candidates}
    actual = [(r.get('task_id'), r.get('candidate_id')) for r in records]
    if len(actual) != len(expected) or set(actual) != expected or t.get('cell_count') != len(records):
        raise ValueError('training records do not exactly cover TRAIN candidates')
    if t.get('training_records_sha256') != digest(records):
        raise ValueError('training record digest differs')
    for key in ('training_table_sha256', 'training_cohort_id'):
        value = t.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('valid model provenance digests required')
    if expected_table_sha256 is not None and t['training_table_sha256'] != expected_table_sha256:
        raise ValueError('model was trained from another table')
    if expected_cohort_id is not None and t['training_cohort_id'] != expected_cohort_id:
        raise ValueError('model was trained on another GPU/runtime cohort')
    return d


def load_cost_model(path, suite, **bindings):
    raw = Path(path).read_bytes()
    if len(raw) > 10_000_000:
        raise ValueError('model exceeds size bound')
    return validate_cost_model(json.loads(raw, object_pairs_hook=_strict_object, parse_constant=_reject_constant),
                               suite, **bindings)


class RidgeSelector:
    """Rank unqueried settings using fixed TRAIN-fitted weights and metadata."""
    def __init__(self, model, *, suite, expected_table_sha256=None, expected_cohort_id=None):
        self._model = validate_cost_model(model, suite, expected_table_sha256=expected_table_sha256,
                                         expected_cohort_id=expected_cohort_id)
        self._suite = suite
        self.training_metadata = {k: deepcopy(v) for k, v in self._model['training'].items() if k != 'records'}
        self.configuration = {'kind': MODEL_KIND, 'model_sha256': digest(self._model),
                              'feature_version': FEATURE_VERSION, 'online_updates': False,
                              'sampling': 'ascending predicted cost without replacement; ties suite order',
                              'training': self.training_metadata}

    def __call__(self, observation, info):
        task = self._suite.tasks[info['task_index']]
        if task.task_id != info['task_id'] or info['objective']['memory_weight'] != 0.0:
            raise ValueError('selector task or objective differs from trained contract')
        available = [i for i, count in enumerate(observation['action_counts']) if count == 0]
        if not available:
            raise PolicyActionRejected('learned_selector_exhausted')
        m = self._model
        scores = [float(((_features(task, self._suite.candidates[i]) - np.asarray(m['mean'])) /
                         np.asarray(m['scale'])) @ np.asarray(m['coefficients']) + m['intercept']) for i in available]
        return min(zip(scores, available))[1]


class TPESelector:
    """Actual Optuna ask/tell; each suggestion, including repeats, costs a turn."""
    def __init__(self, *, suite, seed=2101, startup_trials=3):
        import optuna
        if optuna.__version__ != OPTUNA_VERSION:
            raise ValueError(f'Optuna {OPTUNA_VERSION} required for the frozen policy protocol')
        if type(seed) is not int or not 0 <= seed < 2**32 or type(startup_trials) is not int or not 1 <= startup_trials < 8:
            raise ValueError('bounded seed and startup count below the eight-turn budget required')
        self._optuna, self._suite = optuna, suite
        self._space = structured_tpe_space(suite)
        self.seed, self.startup_trials = seed, startup_trials
        self._study, self._pending = None, None
        self.search_receipts = []
        self.configuration = {'kind': 'optuna_tpe', 'optuna_version': OPTUNA_VERSION,
                              'seed': seed, 'n_startup_trials': startup_trials, 'n_ei_candidates': 24,
                              'multivariate': False, 'constant_liar': False,
                              'search_space': self._space,
                              'successful_cost': 'min(observed paired latency ratio, 2)',
                              'failure_cost': FAILURE_COST, 'repeat_resampling': False,
                              'cross_episode_learning': False, 'training_performed': False}

    def __call__(self, observation, info):
        if info['objective']['memory_weight'] != 0.0:
            raise ValueError('TPE protocol uses latency-only objective')
        if info['trials_used'] == 0:
            if self._pending is not None:
                raise ValueError('previous TPE trial was never observed')
            episode_seed = int(info.get('evaluation_seed', 0))
            sampler_seed = int(hashlib.sha256(f'{self.seed}:{episode_seed}'.encode()).hexdigest()[:8], 16)
            sampler = self._optuna.samplers.TPESampler(seed=sampler_seed, n_startup_trials=self.startup_trials,
                n_ei_candidates=24, multivariate=False, constant_liar=False)
            self._study = self._optuna.create_study(direction='minimize', sampler=sampler)
        if self._study is None or self._pending is not None:
            raise ValueError('TPE requires one observation for each previous suggestion')
        trial = self._study.ask()
        implementation = trial.suggest_categorical('implementation', self._space['implementation'])
        config = {'implementation': implementation}
        if implementation == 'triton':
            block = trial.suggest_int('block_exponent', *self._space['triton_block_exponent'])
            warps = trial.suggest_int('warp_exponent', *self._space['triton_warp_exponent'])
            config.update(block_size=2**block, num_warps=2**warps)
        action = next(i for i, c in enumerate(self._suite.candidates) if c.to_dict()['config'] == config)
        candidate_id = self._suite.candidates[action].candidate_id
        receipt = {'task_id': info['task_id'], 'episode_index': info.get('evaluation_episode_index'),
                   'episode_seed': info.get('evaluation_seed'), 'trial_index': info['trials_used'],
                   'optuna_trial_number': trial.number, 'action': action, 'candidate_id': candidate_id,
                   'suggested_parameters': dict(trial.params),
                   'sampler_phase': 'startup' if trial.number < self.startup_trials else 'tpe',
                   'repeated_action': bool(observation['action_counts'][action]), 'status': 'suggested'}
        self.search_receipts.append(receipt)
        self._pending = (trial, receipt)
        return action

    def observe(self, observation, info):
        """Called once after the environment supplies this turn's own outcome."""
        if self._pending is None:
            raise ValueError('unexpected TPE observation')
        trial, receipt = self._pending
        measurement = info['measurement']
        if (measurement is None or measurement['candidate_id'] != receipt['candidate_id']
                or info['task_id'] != receipt['task_id'] or info['trials_used'] != receipt['trial_index'] + 1):
            raise ValueError('TPE feedback differs from its pending query')
        cost = (min(measurement['latency_ms'] / info['baseline']['latency_ms'], FAILURE_COST)
                if measurement['status'] == 'ok' and measurement['correct'] else FAILURE_COST)
        self._study.tell(trial, cost)
        receipt.update(status='observed', observed_status=measurement['status'],
                       objective_value=cost, source_trial_id=measurement['evidence'].get('source_trial_id'))
        self._pending = None
