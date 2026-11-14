"""Trusted, configuration-only batch execution for the Vista agent study.

No agent Python is imported. Call this in a fresh, externally time-limited worker
for each batch. The caller owns the GPU lease, source/runtime pinning, private
manifests, quality executor, and termination of a stuck CUDA/compiler call.
This adapter is CPU-contract tested; real Vista execution is not yet validated.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np


ALLOWED_VALUES = {'generation.dtype': ['fp16', 'bf16'],
                  'generation.decode_chunk_size': [1, 2, 7, 14],
                  'optimizations.compile_denoiser': [False, True]}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _write(path, value):
    with Path(path).open('x') as output:
        output.write(canonical(value) + '\n')


def _now():
    return datetime.now(timezone.utc).isoformat()


def reference_settings(*, synthetic=False):
    from drivebench.types import GenerationConfig, OptimizationConfig
    generation = GenerationConfig(**({'total_frames': 3, 'height': 8, 'width': 8, 'steps': 2}
                                      if synthetic else {}))
    return {'generation': generation.to_dict(), 'optimizations': OptimizationConfig().to_dict()}


def validate_settings(settings, *, synthetic=False):
    """No paths, model identity, semantic-work reductions, or executable hooks."""
    reference = reference_settings(synthetic=synthetic)
    if not isinstance(settings, dict) or set(settings) != set(reference):
        raise ValueError('exact generation and optimizations sections required')
    for section, expected in reference.items():
        actual = settings[section]
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError('full declared settings required; unknown or missing fields')
        for key, value in actual.items():
            path = f'{section}.{key}'
            allowed = ALLOWED_VALUES.get(path, [expected[key]])
            if not any(type(value) is type(item) and value == item for item in allowed):
                raise ValueError(f'unsupported or fixed setting: {path}')
    return json.loads(canonical(settings))


def model_for_settings(model, settings, *, synthetic=False):
    settings = validate_settings(settings, synthetic=synthetic)
    if synthetic:
        if model != {'backend': 'synthetic'}:
            raise ValueError('synthetic fixture requires its explicit backend')
        return dict(model)
    # The evaluator, not the agent, supplies paths and weight/source identities.
    required = {'checkpoint', 'checkpoint_sha256', 'upstream_dir', 'upstream_commit', 'inference_config'}
    if set(model) != required or any(not isinstance(v, str) or not v for v in model.values()):
        raise ValueError('trusted model identity and paths required')
    compiled = settings['optimizations']['compile_denoiser']
    reference = settings == reference_settings()
    return {**model, 'backend': 'upstream' if reference else 'vista',
            'compiler_runtime': 'registered_einops_attention_v1' if compiled else None,
            'compiler_feature_boundary': None}


def probe_runtime(*, synthetic=False):
    if synthetic:
        return {'kind': 'synthetic', 'numpy': np.__version__, 'gpu': None}
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('exactly one visible CUDA GPU required')
    result = subprocess.run(['nvidia-smi', '--query-gpu=uuid,driver_version', '--format=csv,noheader'],
                            capture_output=True, text=True, check=True, timeout=15)
    rows = [row.strip().split(', ') for row in result.stdout.splitlines() if row.strip()]
    if len(rows) != 1 or len(rows[0]) != 2:
        raise ValueError('one physical GPU identity required; pooled hosts unsupported')
    packages = {name: importlib.metadata.version(name) for name in
                ('torch', 'numpy', 'triton', 'xformers', 'einops', 'safetensors')}
    # Observe backend permissions without selecting or forcing an SDPA kernel.
    # These are runtime settings, not evidence of which kernel actually ran.
    sdp = {}
    for name in ('flash_sdp_enabled', 'mem_efficient_sdp_enabled', 'math_sdp_enabled'):
        query = getattr(torch.backends.cuda, name, None)
        if not callable(query):
            raise ValueError('required SDPA runtime flag unavailable: ' + name)
        value = query()
        if type(value) is not bool:
            raise ValueError('SDPA runtime flag must return a boolean: ' + name)
        sdp[name] = value
    cudnn_available = hasattr(torch.backends.cuda, 'cudnn_sdp_enabled')
    cudnn_query = getattr(torch.backends.cuda, 'cudnn_sdp_enabled', None)
    if cudnn_available and not callable(cudnn_query):
        raise ValueError('invalid optional cuDNN SDPA runtime query')
    cudnn_enabled = cudnn_query() if cudnn_available else None
    if cudnn_available and type(cudnn_enabled) is not bool:
        raise ValueError('cuDNN SDPA runtime flag must return a boolean')
    sdp.update(cudnn_sdp_enabled_available=cudnn_available, cudnn_sdp_enabled=cudnn_enabled)
    return {'kind': 'cuda', 'gpu_uuid': rows[0][0], 'driver': rows[0][1],
            'device': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda, 'packages': packages,
            'sdp': sdp,
            'cudnn_version': torch.backends.cudnn.version(),
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'cuda_launch_blocking': os.environ.get('CUDA_LAUNCH_BLOCKING')}


def _compiler_snapshot(synthetic):
    from drivebench.benchmark import _compiler_state
    return {} if synthetic else _compiler_state()


@dataclass(frozen=True)
class BatchRequest:
    record_id: str
    scene_id: str
    seed: int
    replicate: int
    observed_frames: object
    trajectory: object

    def __post_init__(self):
        if any(not isinstance(v, str) or not v for v in (self.record_id, self.scene_id)):
            raise ValueError('record and scene IDs required')
        if type(self.seed) is not int or not 0 <= self.seed < 2**63 or type(self.replicate) is not int or self.replicate < 0:
            raise ValueError('invalid seed or replicate')
        frames = np.asarray(self.observed_frames, dtype=np.float32)
        trajectory = np.asarray(self.trajectory, dtype=np.float32)
        if (frames.ndim != 4 or frames.shape[1] != 3 or not frames.size or not np.isfinite(frames).all()
                or frames.min() < -1 or frames.max() > 1 or trajectory.shape not in ((8,), (4, 2))
                or not np.isfinite(trajectory).all()):
            raise ValueError('invalid conditioning or trajectory')
        for name, array in [('observed_frames', frames), ('trajectory', trajectory.reshape(8))]:
            frozen = np.frombuffer(np.ascontiguousarray(array).tobytes(), dtype=np.float32).reshape(array.shape)
            object.__setattr__(self, name, frozen)

    @property
    def input_sha256(self):
        return digest({'frames': hashlib.sha256(self.observed_frames.tobytes()).hexdigest(),
                       'shape': list(self.observed_frames.shape),
                       'trajectory': self.trajectory.tolist(), 'seed': self.seed, 'scene_id': self.scene_id})


def run_batch(directory, *, settings, model, requests, cohort_id, expected_runtime,
              source_files, deadline_unix, warmup_requests=1, synthetic=False,
              minimum_free_bytes=2_000_000_000):
    """Run one persistent engine, preserve every array, exclude warmup/export.

    Preparation starts from predecoded normalized arrays. Thus the measured
    boundary includes prepare_context through host-ready RGB, but excludes disk
    image decoding, engine construction, compilation/warmups and output saving.
    It is a NEW timing boundary and cannot reuse the old H100 speedup claim.
    source_files maps trusted absolute source paths to their prespecified hashes.
    Records are written per request; failed/partial directories are never reused.
    """
    from drivebench.engine import InferenceEngine
    from drivebench.types import GenerationConfig, OptimizationConfig
    from drivebench.benchmark import _compilation_count
    settings = validate_settings(settings, synthetic=synthetic)
    actual_model = model_for_settings(model, settings, synthetic=synthetic)
    if not isinstance(cohort_id, str) or not cohort_id or type(synthetic) is not bool:
        raise ValueError('explicit cohort and execution kind required')
    if type(deadline_unix) not in (int, float) or not math.isfinite(deadline_unix) or deadline_unix <= time.time():
        raise ValueError('future finite execution deadline required')
    if type(warmup_requests) is not int or warmup_requests < 1:
        raise ValueError('at least one explicit warmup required')
    if type(minimum_free_bytes) is not int or minimum_free_bytes < 0:
        raise ValueError('nonnegative storage reserve required')
    requests = tuple(requests)
    if not requests or any(not isinstance(r, BatchRequest) for r in requests):
        raise ValueError('nonempty trusted BatchRequest list required')
    if len({r.record_id for r in requests}) != len(requests):
        raise ValueError('duplicate measurement IDs')
    if len({(r.scene_id, r.seed, r.replicate) for r in requests}) != len(requests):
        raise ValueError('duplicate scene/seed/replicate cells')
    if not synthetic and not source_files:
        raise ValueError('real execution requires frozen evaluator source hashes')
    def check_boundary():
        if time.time() >= deadline_unix:
            raise TimeoutError('immutable execution deadline reached')
        if probe_runtime(synthetic=synthetic) != expected_runtime:
            raise ValueError('physical GPU/software runtime drift')
        if any(file_hash(path) != expected for path, expected in source_files.items()):
            raise ValueError('trusted evaluator source drift')
    check_boundary()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    array_bytes = np.prod([settings['generation'][key] for key in ('total_frames', 'height', 'width')]) * 3 * 4
    required_bytes = int(array_bytes) * (len(requests) + warmup_requests) + minimum_free_bytes
    if shutil.disk_usage(directory).free < required_bytes:
        _write(directory / 'failure.json', {'status': 'not_started', 'reason': 'insufficient_output_storage'})
        raise ValueError('insufficient output storage and reserve')
    manifest = {'cohort_id': cohort_id, 'settings': settings, 'settings_sha256': digest(settings),
                'runtime': expected_runtime, 'runtime_sha256': digest(expected_runtime),
                'source_files': source_files, 'model': actual_model, 'created_at': _now(),
                'deadline_unix': deadline_unix, 'synthetic': synthetic,
                'warmup_requests': warmup_requests, 'measurement_count': len(requests),
                'inputs': [{'record_id': r.record_id, 'input_sha256': r.input_sha256} for r in requests]}
    _write(directory / 'manifest.json', manifest)
    rows, warmups, engine = [], [], None
    context = generated = frame = None
    started = time.perf_counter()
    try:
        engine = InferenceEngine(model=actual_model, generation=GenerationConfig(**settings['generation']),
                                 optimizations=OptimizationConfig(**settings['optimizations']))
        engine.synchronize()
        initialization_s = time.perf_counter() - started
        _write(directory / 'engine.json', {'initialization_s': initialization_s,
                                          'backend_identity': engine.backend.identity})
        sequence = [(requests[0], True) for _ in range(warmup_requests)] + [(r, False) for r in requests]
        for index, (request, is_warmup) in enumerate(sequence):
            check_boundary()
            if shutil.disk_usage(directory).free < int(array_bytes) + minimum_free_bytes:
                raise ValueError('output storage reserve exhausted')
            input_hash = request.input_sha256
            engine.synchronize()
            engine.reset_peak_memory()
            compiler_before = _compiler_snapshot(synthetic)
            start = time.perf_counter()
            context = engine.prepare_context(request.observed_frames)
            generated = engine.generate(context, [request.trajectory], [request.seed], candidate_ids=['future'])
            engine.synchronize()
            latency_s = time.perf_counter() - start
            compiler_after = _compiler_snapshot(synthetic)
            compiled_during_request = _compilation_count(compiler_after) - _compilation_count(compiler_before)
            compiler_observable = synthetic or not settings['optimizations']['compile_denoiser'] or _compilation_count(compiler_after) > 0
            # Raw evidence is saved even if subsequent correctness checks fail.
            array_path = directory / f'{index:05d}.npy'
            if 'future' in generated.frames:
                with array_path.open('xb') as output:
                    np.save(output, generated.frames['future'], allow_pickle=False)
            frame = generated.frames.get('future')
            expected_shape = tuple(settings['generation'][k] for k in ('total_frames', 'height', 'width')) + (3,)
            expected_effective = {**settings['generation'], 'candidate_count': 1, 'requested_batch_size': 1,
                                  'actual_batch_sizes': [1], 'optimizations': settings['optimizations']}
            valid = (generated.effective_config == expected_effective and compiler_observable
                     and compiled_during_request >= 0 and (is_warmup or compiled_during_request == 0) and generated.status == 'ok' and not generated.failures and
                     (synthetic or not generated.fallbacks) and generated.denoiser_evaluations == settings['generation']['steps']
                     and not generated.cache_stats.get('context_hit') and not generated.cache_stats.get('oom_attempts')
                     and generated.cache_stats.get('denoiser_evaluations_complete') is True
                     and generated.seeds == [request.seed] and generated.candidate_ids == ['future']
                     and request.input_sha256 == input_hash and frame is not None and frame.shape == expected_shape
                     and frame.dtype == np.float32 and np.isfinite(frame).all() and 0 <= frame.min() <= frame.max() <= 1)
            row = {'record_id': request.record_id, 'scene_id': request.scene_id, 'seed': request.seed,
                   'replicate': request.replicate, 'warmup': is_warmup, 'observed_at': _now(),
                   'latency_ms': latency_s * 1000, 'input_sha256': input_hash,
                   'output_file': array_path.name if array_path.exists() else None,
                   'output_sha256': file_hash(array_path) if array_path.exists() else None,
                   'runtime_sha256': digest(expected_runtime), 'settings_sha256': digest(settings),
                   'peak_allocated_bytes': generated.peak_allocated_bytes, 'peak_reserved_bytes': generated.peak_reserved_bytes,
                   'stage_timings_s': generated.stage_timings_s, 'denoiser_evaluations': generated.denoiser_evaluations,
                   'cache_stats': generated.cache_stats, 'fallbacks': generated.fallbacks,
                   'effective_config': generated.effective_config,
                   'compiler_before': compiler_before, 'compiler_after': compiler_after,
                   'compiled_graph_delta': compiled_during_request,
                   'failures': generated.failures, 'valid': bool(valid), 'synthetic': synthetic}
            _write(directory / f'{index:05d}.json', row)
            (warmups if is_warmup else rows).append(row)
            if not valid:
                raise ValueError('correctness/execution invariant failed; raw evidence retained')
            check_boundary()
        result = {'status': 'ok', 'cohort_id': cohort_id, 'synthetic': synthetic,
                  'rows': rows, 'warmups': warmups, 'initialization_s': initialization_s,
                  'elapsed_s': time.perf_counter() - started, 'quality_assessed': False,
                  'headline_eligible': False, 'timing_boundary': 'normalized conditioning through host-ready RGB; no image file decoding',
                  'completed_at': _now()}
    except BaseException as error:
        _write(directory / 'failure.json', {'status': 'failed', 'error_type': type(error).__name__,
                                          'finished_measurements': len(rows), 'finished_warmups': len(warmups),
                                          'elapsed_s': time.perf_counter() - started, 'at': _now()})
        raise
    finally:
        # Drop context/model references before allocator cleanup and before success.
        context = generated = frame = engine = None
        try:
            gc.collect()
            if not synthetic:
                import torch
                torch.cuda.empty_cache()
        except BaseException as cleanup_error:
            _write(directory / 'cleanup-failure.json', {'status': 'failed', 'error_type': type(cleanup_error).__name__, 'at': _now()})
            raise
    check_boundary()
    _write(directory / 'cleanup.json', {'status': 'completed', 'at': _now(), 'worker_exit_still_required': True})
    result.update(manifest_sha256=file_hash(directory / 'manifest.json'),
                  engine_sha256=file_hash(directory / 'engine.json'),
                  cleanup_sha256=file_hash(directory / 'cleanup.json'), completed_at=_now())
    _write(directory / 'completed.json', result)
    return result
