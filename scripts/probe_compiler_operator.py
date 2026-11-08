#!/usr/bin/env python3
"""Compare eager and integrated compiled denoising on one captured real input.

This is a standalone component diagnostic, not a numerical acceptance gate.
The sampler is deliberately stopped before its first denoiser evaluation. The
same backend, EMA weights and captured tensors are then used for one eager and
one compiled call. No complete sampler rollout or video decoding occurs.
Run only through the external serialized launcher with a 300-second timeout.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from drivebench.config import file_sha256, load_config, validate_run_config
from drivebench.types import GenerationConfig, OptimizationConfig
from scripts.probe_candidate_memory import config_files, save

SOURCE = 'd8a32813492c0939a0a91573a945c914938972419946065e6822f6209fa470cb'


def inspect_config(path):
    config = load_config(path)
    generation, options = validate_run_config(config)
    index = config.get('operator_probe', {}).get('request_index')
    if (generation != GenerationConfig()
            or options != replace(OptimizationConfig(), compile_denoiser=True)
            or config['model'].get('backend') != 'vista'
            or config['model'].get('compiler_runtime') != 'registered_einops_attention_v1'
            or config['model'].get('compiler_feature_boundary') is not None
            or config['data'].get('verify_checksums') is not True
            or config.get('numerical_validation') is not None
            or config.get('benchmark', {}).get('headline') is not False
            or type(index) is not int or not 0 <= index < 32):
        raise ValueError('Require unchanged primary FP16 generation, B1 registered default compiler, no feature/cache changes and a development request')
    return config, generation, options


class CapturedFirstCall(Exception):
    """Intentional early stop; never reported as a completed generation."""


def capture_first_input(backend, payload, request, generation):
    captured = {}
    original = backend._denoiser
    def capture(x, sigma, condition, mask):
        if captured:
            raise RuntimeError('More than one denoiser input was captured')
        captured.update(x=x, sigma=sigma, condition=condition, mask=mask)
        raise CapturedFirstCall()
    backend._denoiser = capture
    try:
        try:
            backend.generate_batch(payload, request['trajectories'], request['seeds'], generation)
        except CapturedFirstCall:
            pass
    finally:
        backend._denoiser = original
    if not captured:
        raise RuntimeError('Sampler did not reach its first denoiser call')
    return captured


def array_hash(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast('B')).hexdigest()


def tensor_fingerprint(tensor):
    array = tensor.detach().contiguous().cpu().numpy()
    if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
        raise ValueError('Captured tensor must contain finite floating-point values')
    return {'shape': list(tensor.shape), 'dtype': str(tensor.dtype), 'stride': list(tensor.stride()),
            'device_type': tensor.device.type, 'sha256': array_hash(array)}


def input_fingerprints(values):
    if set(values) != {'x', 'sigma', 'condition', 'mask'} or set(values['condition']) != {'vector', 'crossattn', 'concat'}:
        raise ValueError('Unexpected captured denoiser input structure')
    tensors = {key: values[key] for key in ('x', 'sigma', 'mask')}
    tensors.update({'condition.' + key: tensor for key, tensor in values['condition'].items()})
    if tuple(values['x'].shape) != (50, 4, 72, 128):
        raise ValueError('Require primary B1 CFG latent shape [50,4,72,128]')
    if any(tensor.shape[0] != 50 or tensor.device.type != 'cuda' for tensor in tensors.values()):
        raise ValueError('Captured tensors must preserve the full unconditional/conditional 25-frame CUDA layout')
    return {name: tensor_fingerprint(tensor) for name, tensor in sorted(tensors.items())}


def compare_arrays(reference, actual):
    if reference.shape != actual.shape or not reference.size:
        raise ValueError('Denoiser output shapes differ or are empty')
    if any(not np.issubdtype(x.dtype, np.floating) for x in (reference, actual)):
        raise ValueError('Denoiser outputs must be floating-point arrays')
    maximum = squared = 0.0
    different = total = 0
    for left, right in zip(reference, actual):
        left, right = left.astype(np.float64), right.astype(np.float64)
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError('Nonfinite denoiser output')
        difference = left - right
        maximum = max(maximum, float(np.abs(difference).max(initial=0)))
        squared += float(np.square(difference).sum())
        different += int(np.count_nonzero(difference))
        total += difference.size
    return {'atol': 0, 'rtol': 0, 'passed_exact': different == 0 and reference.dtype == actual.dtype,
            'max_absolute_error': maximum, 'rmse': float(np.sqrt(squared / total)),
            'different_values': different, 'values': total,
            'reference_dtype': str(reference.dtype), 'compiled_dtype': str(actual.dtype),
            'dtype_equal': reference.dtype == actual.dtype}


def rng_fingerprint(torch, device):
    return {'cpu': array_hash(torch.random.get_rng_state().cpu().numpy()),
            'cuda': array_hash(torch.cuda.get_rng_state(device).cpu().numpy())}


def run_diagnostic(backend, payload, request, generation, *, progress=None):
    import torch
    from drivebench.models.vista import _EXECUTION_LOCK
    compiled = backend._denoiser
    eager = getattr(compiled, '_torchdynamo_orig_callable', None)
    if not callable(eager) or eager is compiled or backend._compiler_runtime is None:
        raise ValueError('Require the integrated compiled callable and its preserved original')
    if len(request['seeds']) != 1 or len(request['trajectories']) != 1:
        raise ValueError('Component diagnostic requires one candidate')
    captured = capture_first_input(backend, payload, request, generation)
    capture_runtime = copy.deepcopy(backend._compiler_runtime.last_execution)
    if not capture_runtime or capture_runtime.get('restoration_complete') is not True:
        raise RuntimeError('Capture did not restore the compiler runtime')
    baseline_inputs = input_fingerprints(captured)
    outputs, rows = {}, []
    if progress is not None:
        progress({'stage': 'captured_before_component_calls', 'input_fingerprints': baseline_inputs,
                  'capture_runtime': capture_runtime, 'sampler_rollouts_completed': 0, 'video_decoder_calls': 0})
    backend.reset_peak_memory()
    # Both calls see the same EMA parameters and same autocast policy. The
    # compiler runtime is activated only for its own compiled call.
    with _EXECUTION_LOCK, torch.cuda.device(backend.device), torch.no_grad(), backend._autocast(), backend.model.ema_scope('Compiler component diagnostic'):
        for label, function in [('eager', eager), ('compiled', compiled)]:
            before = input_fingerprints(captured)
            if before != baseline_inputs:
                raise RuntimeError('Captured input changed before ' + label)
            rng_before = rng_fingerprint(torch, backend.device)
            backend.synchronize()
            started = time.perf_counter()
            if label == 'compiled':
                with backend._compiler_scope(require_graphs=True):
                    out = function(captured['x'], captured['sigma'], captured['condition'], captured['mask'])
            else:
                out = function(captured['x'], captured['sigma'], captured['condition'], captured['mask'])
            backend.synchronize()
            elapsed = time.perf_counter() - started
            array = out.detach().contiguous().cpu().numpy().copy()
            del out
            if tuple(array.shape) != (50, 4, 72, 128):
                raise ValueError('Unexpected denoiser component output shape')
            after = input_fingerprints(captured)
            if after != baseline_inputs:
                raise RuntimeError('Denoiser mutated a captured input during ' + label)
            rng_after = rng_fingerprint(torch, backend.device)
            outputs[label] = array
            rows.append({'label': label, 'diagnostic_wall_s': elapsed, 'shape': list(array.shape),
                'dtype': str(array.dtype), 'output_sha256': array_hash(array),
                'input_hashes_before': before, 'input_hashes_after': after,
                'rng_state_before': rng_before, 'rng_state_after': rng_after,
                'rng_state_unchanged': rng_before == rng_after})
            if progress is not None:
                progress({'stage': label + '_component_complete', 'executions': copy.deepcopy(rows)})
    execution = copy.deepcopy(backend._compiler_runtime.last_execution)
    if (execution.get('passed') is not True or execution.get('restoration_complete') is not True
            or execution.get('graphs_created_by_runtime', 0) < 1 or execution.get('kernels_created_by_runtime', 0) < 1):
        raise RuntimeError('Missing positive integrated compilation/restoration evidence')
    comparison = compare_arrays(outputs['eager'], outputs['compiled'])
    allocated, reserved = backend.peak_memory()
    return {'capture_count': 1, 'denoiser_evaluations': 2, 'sampler_rollouts_completed': 0,
            'video_decoder_calls': 0, 'candidate_count': 1,
            'cfg_order': '25 unconditional frames followed by 25 conditional frames',
            'autocast_policy': 'fp16; actual input/output dtypes recorded',
            'same_backend_and_ema_scope': True, 'input_tensors_reused_without_resampling': True,
            'callable_restored_after_capture': backend._denoiser is compiled,
            'capture_runtime': capture_runtime, 'compiled_runtime': execution,
            'executions': rows, 'comparison': comparison,
            'peak_allocated_bytes': allocated, 'peak_reserved_bytes': reserved,
            'timing_scope': 'Instrumented component observation only. Compiled call includes lazy compilation; not steady-state latency.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    config, generation, options = inspect_config(args.config)
    if args.output.is_symlink():
        raise ValueError('Output symlink is forbidden')
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    bindings = config_files(args.config)
    for path in [Path(__file__), Path('scripts/probe_candidate_memory.py'), Path(config['data']['manifest']),
                 Path(config['data']['manifest'] + '.meta.json'), Path(config['environment_verification']), Path(config['environment_lock'])]:
        bindings[str(path.resolve())] = file_sha256(path)
    report = {'schema_version': 1, 'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
              'config': config, 'bound_file_sha256': bindings, 'headline_eligible': False, 'quality_validated': False,
              'numerical_acceptance_gate': False, 'proposed_external_timeout_seconds': 300,
              'scope': 'One real captured first-step denoiser component input; no rollout, acceptance, timing or quality claim'}
    target = output / 'compiler-operator-probe.json'; save(target, report)
    handlers = {}
    def stop(signum, _frame):
        report['termination_signal'] = signum
        raise SystemExit(128 + signum)
    for sig in (signal.SIGINT, signal.SIGTERM):
        handlers[sig] = signal.signal(sig, stop)
    code = 1
    try:
        from drivebench.benchmark import _validate_reference_environment
        from drivebench.benchmarks.environment import collect_environment, source_identity
        from drivebench.data import inference_request, load_manifest, read_preprocess, verify_input_checksums
        from drivebench.engine import InferenceEngine
        from drivebench.validate import input_provenance
        report['implementation_sha256'] = source_identity()
        if report['implementation_sha256'] != SOURCE:
            raise ValueError('Require the frozen d8 package')
        environment = collect_environment(); save(output / 'environment.json', environment)
        report['environment_gate'] = _validate_reference_environment(config, environment)
        records = load_manifest(config['data']['manifest'])
        index = config['operator_probe']['request_index']; record = records[index]
        verify_input_checksums(record, config['data']['root'], include_future=False)
        request = inference_request(record, config['data']['root'], candidate_count=1)
        frames = read_preprocess(request['observed_frames'], generation.height, generation.width)
        provenance = input_provenance(config['data']['manifest'], records, record, request, frames, request_index=index)
        if provenance['partition_role'] != 'development' or provenance['synthetic'] or len(records) != 32:
            raise ValueError('Use the verified real32-request development selection only')
        report['input_provenance'] = provenance
        engine = InferenceEngine(config['model'], generation, options)
        report['model_identity'] = engine.backend.identity; save(target, report)
        context = engine.prepare_context(frames)
        report['diagnostic'] = {}
        def progress(values):
            report['diagnostic'].update(values); save(target, report)
        report['diagnostic'] = run_diagnostic(engine.backend, context.payload, request, generation, progress=progress)
        report['status'] = 'diagnostic_complete_exact' if report['diagnostic']['comparison']['passed_exact'] else 'diagnostic_complete_strict_failed'
        code = 0 if report['diagnostic']['comparison']['passed_exact'] else 2
    except BaseException as error:
        report.update(status='failed', error={'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()})
        if 'engine' in locals() and getattr(engine.backend, '_compiler_runtime', None) is not None:
            report['runtime_at_failure'] = copy.deepcopy(engine.backend._compiler_runtime.last_execution)
        code = error.code if isinstance(error, SystemExit) and type(error.code) is int and error.code else 1
    finally:
        try:
            report['bound_files_unchanged'] = all(Path(path).is_file() and file_sha256(path) == checksum for path, checksum in bindings.items())
            report['source_unchanged'] = report.get('implementation_sha256') == source_identity()
            if not report['bound_files_unchanged'] or not report['source_unchanged']:
                report['status'], code = 'source_or_inputs_changed', 1
        except Exception as error:
            report['finalization_error'] = str(error); code = 1
        report.update(exit_code=code, finished_at=datetime.now(timezone.utc).isoformat()); save(target, report)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    print(json.dumps({'status': report['status'], 'report': str(target), 'exit_code': code}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
