"""Bridge an injected InferenceEngine to the shared configuration-search API.

No model or GPU is imported/provisioned here. The caller owns model loading,
runtime verification, execution timeouts and final resource cleanup. The
included example runs the core's explicit CPU test double. Real Vista execution
through this bridge has not been validated.
"""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import time
import uuid

import numpy as np

from .search import CandidateSpec, TaskSpec, TrialResult, _freeze_json, _thaw_json


def _array(value, shape=None):
    array = np.asarray(value, dtype=np.float32)
    if not np.isfinite(array).all() or (shape is not None and array.shape != shape):
        raise ValueError('invalid inference input shape or nonfinite values')
    # An immutable bytes owner prevents callers from re-enabling writeability.
    return np.frombuffer(np.ascontiguousarray(array).tobytes(), dtype=np.float32).reshape(array.shape)


@dataclass(frozen=True)
class InferenceRequest:
    observed_frames: np.ndarray
    trajectories: tuple
    seeds: tuple
    output_ids: tuple | None = None

    def __post_init__(self):
        frames = _array(self.observed_frames)
        if frames.ndim != 4 or frames.shape[1] != 3 or 0 in frames.shape or frames.min() < -1 or frames.max() > 1:
            raise ValueError('observed_frames must be nonempty normalized TCHW RGB')
        trajectories = tuple(_array(value).reshape(8) for value in self.trajectories)
        seeds = tuple(self.seeds)
        if not trajectories or len(trajectories) != len(seeds):
            raise ValueError('one seed per nonempty trajectory is required')
        if any(type(seed) is not int or not 0 <= seed < 2**63 for seed in seeds):
            raise ValueError('seeds must be uint63 integers')
        ids = tuple(self.output_ids) if self.output_ids is not None else tuple(f'future-{i}' for i in range(len(seeds)))
        if len(ids) != len(seeds) or len(set(ids)) != len(ids) or any(not isinstance(i, str) or not i for i in ids):
            raise ValueError('output IDs must be nonempty, unique and match the trajectories')
        object.__setattr__(self, 'observed_frames', frames)
        object.__setattr__(self, 'trajectories', trajectories)
        object.__setattr__(self, 'seeds', seeds)
        object.__setattr__(self, 'output_ids', ids)


@dataclass(frozen=True)
class QualityDecision:
    passed: bool
    evidence: Mapping

    def __post_init__(self):
        if type(self.passed) is not bool or not isinstance(self.evidence, Mapping) or not self.evidence:
            raise ValueError('quality decision needs a boolean and nonempty policy/measurement evidence')
        object.__setattr__(self, 'evidence', _freeze_json(self.evidence))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def _input_digest(task, frames, trajectories, seeds, ids):
    return _digest({'task': task.to_dict(), 'frames': _tensor_digest(frames),
                    'trajectories': [_tensor_digest(x) for x in trajectories],
                    'seeds': list(seeds), 'output_ids': list(ids)})


def _tensor_digest(value):
    array = np.asarray(value)
    return {'shape': list(array.shape), 'dtype': str(array.dtype),
            'sha256': hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()}


def _result_digest(result):
    return _digest({'frames': {key: _tensor_digest(value) for key, value in result.frames.items()},
                    'ids': result.candidate_ids, 'seeds': result.seeds,
                    'config': result.effective_config, 'status': result.status,
                    'failures': result.failures, 'fallbacks': result.fallbacks})


def _settings(value):
    value = CandidateSpec('settings', value).to_dict()['config']
    if set(value) - {'generation', 'optimizations'} or any(not isinstance(v, dict) for v in value.values()):
        raise ValueError('inference configs contain only generation and optimizations objects')
    return {key: value.get(key, {}) for key in ('generation', 'optimizations')}


def _model_content_identity(identity):
    # These core backend fields describe the chosen execution path rather than
    # model contents. Everything else must remain identical, including unknown
    # identity fields introduced by an injected backend.
    execution_fields = {'candidate_execution', 'candidate_batch_limit', 'candidate_batch_limit_basis',
                        'candidate_batch_agreement', 'decoder_chunk_quality', 'compiler_runtime',
                        'compiler_feature_boundary'}
    return {key: deepcopy(value) for key, value in identity.items() if key not in execution_fields}


class InferenceEvaluator:
    """Measure one first request for each injected engine configuration.

    ``engine_factory(task, config)`` returns a fresh core InferenceEngine.
    Configs are generation/optimizations dictionaries; candidate dictionaries
    override the reference dictionaries. An optional mapper(task,candidate)
    supplies those overrides for other candidate payload formats.

    ``quality_gate(task,candidate,result,reference)`` is mandatory, including
    baseline calls (candidate/reference are None there). Return QualityDecision;
    the callback must document a suitable quality policy. Output shape/range
    checks in this bridge are not a visual-quality assessment.

    Timing excludes engine factory initialization, includes prepare_context,
    generate and synchronization, and includes any first-use compilation. There
    are zero warmups and one request sample per trial. This is a first-request
    boundary, not the historical warm steady-state Vista benchmark. CUDA memory
    is the engine-reported full-model peak allocated bytes after resetting peak
    counters. Synthetic memory is an explicitly labeled output-nbytes proxy.

    Runtime/cohort identity is caller supplied; CUDA additionally requires a
    CUDA device on the injected backend. This check does not replace runtime
    admission or prove the hardware identity. TrialResult.input_digest binds
    the inputs and actual reference config/model identity/runtime evidence so
    a harness can reject reference-contract drift across fresh evaluators.
    The raw input digest is also retained in evidence. Optional engine_cleanup(engine)
    runs after every attempt, before completion; absent it Python references are
    released, which is not a cloud-resource cleanup procedure. ``event_sink``
    can persist started/finished/failed records (e.g. EvidenceJournal.append).
    """

    def __init__(self, request_loader, engine_factory, quality_gate, reference_config, *,
                 backend, cohort_id, runtime_evidence=None, candidate_config_mapper=None,
                 engine_cleanup=None, event_sink=None):
        if not all(callable(fn) for fn in (request_loader, engine_factory, quality_gate)):
            raise ValueError('request_loader, engine_factory and quality_gate are mandatory callables')
        for fn in (candidate_config_mapper, engine_cleanup, event_sink):
            if fn is not None and not callable(fn):
                raise ValueError('optional callbacks must be callable')
        if backend not in ('cuda', 'synthetic') or not isinstance(cohort_id, str) or not cohort_id:
            raise ValueError('explicit backend and nonempty cohort_id required')
        if backend == 'cuda' and not runtime_evidence:
            raise ValueError('CUDA needs caller-verified runtime evidence; no automatic fallback')
        self.request_loader, self.engine_factory, self.quality_gate = request_loader, engine_factory, quality_gate
        self.reference_config = _freeze_json(_settings(reference_config))
        self.backend, self.cohort_id = backend, cohort_id
        self.runtime_evidence = _freeze_json(runtime_evidence or {})
        self.mapper, self.cleanup, self.event_sink = candidate_config_mapper, engine_cleanup, event_sink
        self._references = {}
        self._journal = []

    @property
    def journal(self):
        return deepcopy(self._journal)

    def _event(self, event):
        copied = _thaw_json(_freeze_json(event))
        self._journal.append(copied)
        if self.event_sink is not None:
            self.event_sink(deepcopy(copied))

    def baseline(self, task):
        return self._run(task, None)

    def evaluate(self, task, candidate):
        if task.task_id not in self._references:
            raise RuntimeError('measure an accepted reference before evaluating candidates')
        return self._run(task, candidate)

    def _run(self, task, candidate):
        trial_id = str(uuid.uuid4())
        self._event({'event': 'started', 'trial_id': trial_id, 'task_id': task.task_id,
                     'candidate_id': None if candidate is None else candidate.candidate_id,
                     'backend': self.backend})
        engine = context = None
        evidence = {'runtime': _thaw_json(self.runtime_evidence)}
        input_digest = None
        try:
            reference = self._references.get(task.task_id)
            if candidate is None:
                self._references.pop(task.task_id, None)  # Failed new reset cannot reuse an old reference.
            request = self.request_loader(task)
            if not isinstance(request, InferenceRequest):
                raise ValueError('request_loader must return InferenceRequest')
            frames = request.observed_frames.copy()
            actions = [action.copy() for action in request.trajectories]
            seeds, ids = list(request.seeds), list(request.output_ids)
            input_digest = _input_digest(task, frames, actions, seeds, ids)
            if candidate is not None and input_digest != reference['input_digest']:
                raise ValueError('inference inputs or task identity changed since the reference')
            config = _thaw_json(self.reference_config)
            if candidate is not None:
                override = _settings(self.mapper(task, candidate) if self.mapper else candidate.to_dict()['config'])
                config = {key: {**config[key], **override[key]} for key in config}
            evidence = {'runtime': _thaw_json(self.runtime_evidence), 'mapped_config': config,
                        'latency_boundary': 'first request: prepare_context + generate + synchronization; initialization excluded; first-use compilation included',
                        'warmups': 0, 'request_samples': 1,
                        'memory_boundary': ('synthetic returned-output nbytes proxy; not GPU allocation'
                                            if self.backend == 'synthetic' else 'engine-reported full-model peak allocated bytes, including resident model'),
                        'runtime_identity_source': 'caller supplied; backend CUDA device checked but full runtime admission remains external',
                        'quality_preservation_claim': False, 'vista_gpu_bridge_validated': False}
            execution_error = None
            try:
                started = time.perf_counter()
                engine = self.engine_factory(task, deepcopy(config))
                evidence['engine_initialization_seconds'] = time.perf_counter() - started
                generation = engine.generation.to_dict()
                optimizations = engine.optimizations.to_dict()
                actual = {'generation': generation, 'optimizations': optimizations}
                for section, values in config.items():
                    if any(key not in actual[section] or actual[section][key] != value for key, value in values.items()):
                        raise ValueError('engine factory ignored or changed mapped configuration')
                if candidate is not None:
                    expected = {key: {**reference['engine_config'][key], **config[key]} for key in config}
                    if actual != expected:
                        raise ValueError('engine factory changed an unspecified configuration')
                # Fixed semantic task; decode chunk size and execution options may change.
                semantics = {'generation': {k: v for k, v in generation.items() if k != 'decode_chunk_size'},
                             'model': engine.model_config}
                if candidate is not None and semantics != reference['semantics']:
                    raise ValueError('candidate changed model or generation semantics')
                backend_identity = getattr(engine.backend, 'identity', {})
                content_identity = _model_content_identity(backend_identity)
                if candidate is not None and content_identity != reference['model_content_identity']:
                    raise ValueError('actual backend model-content identity changed since the reference')
                if self.backend == 'cuda':
                    if backend_identity.get('backend') == 'synthetic' or getattr(getattr(engine.backend, 'device', None), 'type', None) != 'cuda':
                        raise ValueError('CUDA inference requested but injected engine is not a CUDA backend')
                elif backend_identity.get('backend') != 'synthetic':
                    raise ValueError('synthetic mode requires an explicitly identified synthetic backend')
                evidence['engine_config'] = actual
                evidence['model_identity'] = deepcopy(backend_identity)
                reference_contract = {'raw_input_digest': input_digest,
                                      'engine_config': actual if candidate is None else reference['engine_config'],
                                      'semantics': semantics,
                                      'model_content_identity': content_identity,
                                      'runtime': _thaw_json(self.runtime_evidence)}
                measurement_digest = _digest(reference_contract)
                if candidate is not None and measurement_digest != reference['measurement_digest']:
                    raise ValueError('reference runtime or inference contract changed since the reference')
                evidence['raw_input_digest'] = input_digest
                evidence['reference_contract_digest'] = measurement_digest
                engine.synchronize()
                engine.reset_peak_memory()
                started = time.perf_counter()
                context = engine.prepare_context(frames)
                result = engine.generate(context, actions, seeds, candidate_ids=ids)
                engine.synchronize()
                elapsed_ms = (time.perf_counter() - started) * 1000
                evidence['request_latency_ms'] = elapsed_ms
                if input_digest != _input_digest(task, frames, actions, seeds, ids):
                    raise ValueError('inference engine mutated request inputs')
                if list(result.candidate_ids) != list(request.output_ids) or list(result.seeds) != list(request.seeds):
                    raise ValueError('inference output identity does not match the request')
                effective = result.effective_config
                if (any(effective.get(k) != v for k, v in generation.items()) or
                    effective.get('optimizations') != optimizations or
                    effective.get('candidate_count') != len(ids) or
                    effective.get('requested_batch_size') != optimizations['max_batch_size']):
                    raise ValueError('reported effective configuration does not match the injected engine/request')
                evidence['effective_config'] = deepcopy(effective)
                evidence['stage_timings_s'] = deepcopy(result.stage_timings_s)
                evidence['fallbacks'] = deepcopy(result.fallbacks)
                evidence['failures'] = deepcopy(result.failures)
                evidence['cache_stats'] = deepcopy(result.cache_stats)
                evidence['observed_engine_peak_allocated_bytes'] = result.peak_allocated_bytes
                evidence['observed_engine_peak_reserved_bytes'] = result.peak_reserved_bytes
                if self.backend == 'synthetic':
                    evidence['observed_synthetic_output_nbytes'] = sum(np.asarray(value).nbytes for value in result.frames.values())
                if result.status != 'ok' or result.failures or result.cache_stats.get('oom_attempts'):
                    outcome = TrialResult(trial_id, task.task_id, None if candidate is None else candidate.candidate_id,
                                          self.backend, self.cohort_id, measurement_digest, 'error', False, None, None,
                                          {**evidence, 'execution_status': result.status, 'quality_gate_not_run': True})
                else:
                    shape = (generation['total_frames'], generation['height'], generation['width'], 3)
                    if set(result.frames) != set(ids) or any(np.asarray(value).shape != shape or not np.isfinite(value).all()
                        or value.min() < 0 or value.max() > 1 for value in result.frames.values()):
                        raise ValueError('invalid inference output tensors')
                    checked = deepcopy(result)
                    checked_reference = deepcopy(reference['result']) if candidate is not None else None
                    before = _result_digest(checked)
                    ref_before = _result_digest(checked_reference) if checked_reference is not None else None
                    decision = self.quality_gate(task, candidate, checked, checked_reference)
                    if not isinstance(decision, QualityDecision):
                        raise ValueError('quality_gate must return QualityDecision, not a bare boolean')
                    if before != _result_digest(checked) or (checked_reference is not None and ref_before != _result_digest(checked_reference)):
                        raise ValueError('quality callback mutated evaluated or reference results')
                    evidence['quality'] = _thaw_json(decision.evidence)
                    evidence['output_digest'] = _result_digest(result)
                    memory = (sum(value.nbytes for value in result.frames.values()) if self.backend == 'synthetic'
                              else result.peak_allocated_bytes)
                    outcome = TrialResult(trial_id, task.task_id, None if candidate is None else candidate.candidate_id,
                                          self.backend, self.cohort_id, measurement_digest,
                                          'ok' if decision.passed else 'incorrect', decision.passed,
                                          elapsed_ms if decision.passed else None, memory if decision.passed else None, evidence)
                    if candidate is None and decision.passed:
                        pending_reference = {'result': deepcopy(result), 'input_digest': input_digest,
                                             'engine_config': deepcopy(actual), 'semantics': deepcopy(semantics),
                                             'model_content_identity': content_identity,
                                             'measurement_digest': measurement_digest}
            except Exception as exc:
                execution_error = exc
                raise
            finally:
                context = None
                try:
                    if engine is not None and self.cleanup is not None:
                        self.cleanup(engine)
                except Exception as exc:
                    evidence['cleanup_error'] = {'type': type(exc).__name__, 'message': str(exc)}
                    if execution_error is None:
                        raise
                finally:
                    engine = None
            if candidate is None and outcome.status == 'ok':
                self._references[task.task_id] = pending_reference
            self._event({'event': 'finished', 'trial_id': trial_id, 'measurement': outcome.to_dict()})
            return outcome
        except Exception as exc:
            self._event({'event': 'failed', 'trial_id': trial_id,
                         'task_id': task.task_id, 'candidate_id': None if candidate is None else candidate.candidate_id,
                         'input_digest': input_digest, 'evidence': evidence,
                         'error': {'type': type(exc).__name__, 'message': str(exc)}})
            raise

    def close(self):
        # Reusable adapter: releasing reference outputs does not reset any caller deadline.
        self._references.clear()
