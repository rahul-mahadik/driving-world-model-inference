#!/usr/bin/env python3
"""One cold full-length candidate-batch capacity attempt, without OOM fallback.

This diagnostic deliberately calls the backend once rather than the engine's
retrying scheduler. A failed size is retained as failed; smaller sizes require a
new process and output directory. No numerical-equivalence or quality claim.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import yaml

from drivebench.config import digest, file_sha256, load_config, validate_run_config
from drivebench.types import GenerationConfig, OptimizationConfig


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def config_files(path):
    result = {}
    while path is not None:
        path = Path(path).resolve()
        if str(path) in result:
            raise ValueError('Config inheritance cycle')
        result[str(path)] = file_sha256(path)
        parent = yaml.safe_load(path.read_text()).get('extends')
        path = path.parent / parent if parent else None
    return result


def inspect_config(path):
    config = load_config(path)
    generation, options = validate_run_config(config)
    size = config.get('memory_probe', {}).get('batch_size')
    index = config.get('memory_probe', {}).get('request_index')
    if type(size) is not int or not 1 <= size <= 16:
        raise ValueError('Explicit integer capacity size must be in [1,16]')
    if type(index) is not int or not 0 <= index < 32:
        raise ValueError('Capacity screening is restricted to a development request')
    if generation != GenerationConfig() or options != replace(OptimizationConfig(), max_batch_size=size):
        raise ValueError('Change only candidate batch size at the unchanged primary generation settings')
    if (config['model'].get('backend') != 'vista' or config['model'].get('compiler_runtime')
            or config['model'].get('compiler_feature_boundary') or config.get('numerical_validation') is not None
            or config.get('benchmark', {}).get('headline') is not False
            or config.get('benchmark', {}).get('candidate_count') != size
            or config.get('data', {}).get('verify_checksums') is not True):
        raise ValueError('Capacity probes require eager Vista, checksums, no inherited proof and headline=false')
    return config, generation, options


def execute_attempt(engine, frames, request, generation, size, *, clock=time.perf_counter):
    """Execute exactly one backend call; return a serializable result and status."""
    backend = engine.backend
    if len(request['seeds']) != size or len(request['trajectories']) != size or size > backend.max_batch_size:
        raise ValueError('Requested capacity does not match the exact backend call')
    engine.synchronize()
    engine.reset_peak_memory()
    row = {'requested_batch_size': size, 'attempted_batch_sizes': [size], 'actual_batch_sizes': [],
           'oom_fallback_enabled': False, 'retry_count': 0, 'hardware_capacity_established': False,
           'phase': 'prepare_context', 'status': 'running'}
    begin = clock()
    recover = False
    try:
        context = engine.prepare_context(frames)
        row['prepare_context_s'] = context.preparation_s
        row['phase'] = 'generate_batch'
        batch = backend.generate_batch(context.payload, request['trajectories'], request['seeds'], generation)
        engine.synchronize()
        row['wall_s_excluding_file_decode_and_export'] = clock() - begin
        expected = (size, generation.total_frames, generation.height, generation.width, 3)
        if batch.frames.shape != expected or batch.frames.dtype != np.float32:
            raise ValueError('Capacity output shape or dtype differs')
        for candidate in batch.frames:
            if not np.isfinite(candidate).all() or candidate.min() < 0 or candidate.max() > 1:
                raise ValueError('Invalid capacity output values')
        if batch.fallbacks or batch.denoiser_evaluations != generation.steps:
            raise ValueError('Capacity call did not complete the exact unchanged sampler work')
        from drivebench.validate import _array_sha256
        row.update(status='requested_batch_fits', actual_batch_sizes=[size], phase='complete',
                   denoiser_evaluations=batch.denoiser_evaluations, stage_timings_s=batch.stage_timings_s,
                   gpu_inference_s=batch.gpu_inference_s, cache_stats=batch.cache_stats,
                   output_sha256={key: _array_sha256(value) for key, value in zip(request['candidate_ids'], batch.frames)})
        code = 0
    except Exception as error:
        recover = isinstance(error, MemoryError) or type(error).__name__ in {'OutOfMemoryError', 'CUDAOutOfMemoryError'}
        row.update(status='requested_batch_oom' if recover else 'failed',
                   wall_s_excluding_file_decode_and_export=clock() - begin,
                   error={'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()})
        code = 2 if recover else 1
    finally:
        allocated, reserved = backend.peak_memory()
        row.update(peak_allocated_bytes=allocated, peak_reserved_bytes=reserved)
    # Leave the exception block before allocator cleanup releases failed tensors.
    if recover:
        backend.release_after_oom()
        row['oom_cleanup_completed'] = True
    return row, code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    config, generation, options = inspect_config(args.config)
    output = args.output or Path(config['output'])
    if output.is_symlink():
        raise ValueError('Output directory must not be a symlink')
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    bound = config_files(args.config)
    for path in [Path(__file__), Path(config['data']['manifest']),
                 Path(config['data']['manifest'] + '.meta.json'),
                 Path(config['environment_verification']), Path(config['environment_lock'])]:
        bound[str(path.resolve())] = file_sha256(path)
    report = {'schema_version': 1, 'status': 'running', 'started_at': now(), 'config': config,
              'bound_file_sha256': bound, 'generation': generation.to_dict(), 'optimizations': options.to_dict(),
              'headline_eligible': False, 'quality_validated': False, 'cold_request': True,
              'scope': 'One development request capacity screen; no largest-hardware-batch claim',
              'timing_scope': 'Synchronized prepare_context + one backend generation; excludes file decode, model load, export',
              'no_retry_or_smaller_batch_fallback': True}
    report_path = output / 'memory-probe.json'
    save(report_path, report)
    handlers = {}
    def interrupted(signum, _frame):
        report['termination_signal'] = signum
        raise SystemExit(128 + signum)
    for sig in (signal.SIGINT, signal.SIGTERM):
        handlers[sig] = signal.signal(sig, interrupted)
    code = 1
    try:
        from drivebench.benchmark import _validate_reference_environment
        from drivebench.benchmarks.environment import collect_environment, source_identity
        from drivebench.data import inference_request, load_manifest, read_preprocess, verify_input_checksums
        from drivebench.engine import InferenceEngine
        from drivebench.validate import input_provenance
        report['implementation_sha256'] = source_identity()
        environment = collect_environment()
        save(output / 'environment.json', environment)
        report['environment_gate'] = _validate_reference_environment(config, environment)
        records = load_manifest(config['data']['manifest'])
        index, size = config['memory_probe']['request_index'], config['memory_probe']['batch_size']
        record = records[index]
        verify_input_checksums(record, config['data']['root'], include_future=False)
        request = inference_request(record, config['data']['root'], candidate_count=size)
        frames = read_preprocess(request['observed_frames'], generation.height, generation.width)
        provenance = input_provenance(config['data']['manifest'], records, record, request, frames, request_index=index)
        if provenance['partition_role'] != 'development' or provenance['synthetic'] or len(records) != 32:
            raise ValueError('Capacity probe requires the real 32-request development manifest')
        report['input_provenance'] = provenance
        save(report_path, report)
        start = time.perf_counter()
        engine = InferenceEngine(config['model'], generation, options)
        engine.synchronize()
        report.update(model_load_s=time.perf_counter() - start, model_identity=engine.backend.identity)
        save(report_path, report)
        report['execution'], code = execute_attempt(engine, frames, request, generation, size)
        report['status'] = report['execution']['status']
    except BaseException as error:
        report.update(status='failed', error={'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()})
        code = error.code if isinstance(error, SystemExit) and type(error.code) is int and error.code else 1
    finally:
        try:
            report['bound_files_unchanged'] = all(Path(path).is_file() and file_sha256(path) == sha for path, sha in bound.items())
            report['source_unchanged'] = report.get('implementation_sha256') == source_identity()
            if not report['bound_files_unchanged'] or not report['source_unchanged']:
                report['status'], code = 'source_or_inputs_changed', 1
        except Exception as error:
            report['finalization_error'] = str(error)
            code = 1
        report.update(exit_code=code, finished_at=now())
        save(report_path, report)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    print(json.dumps({'status': report['status'], 'exit_code': code, 'report': str(report_path)}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
