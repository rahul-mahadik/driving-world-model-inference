#!/usr/bin/env python3
"""Instrumented A100 first-step repeatability localization; never an acceptance gate."""
from __future__ import annotations
import argparse
import contextlib
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
SOURCE = 'd8a32813492c0939a0a91573a945c914938972419946065e6822f6209fa470cb'
DEADLINE = 1793168129.682277
INNER_SECONDS = 840
PINS = {
    'outputs/host/verify-a100-pilot-v1.py': '157f988562c27b713a33f5987b9c5a7c137ff2771453569744f286ee9bf6425f',
    'outputs/host/a100-pilot-01/configs/baseline.yaml': '1942e4782b6018eca25d93e928dcad0501e639dab64c64acaf91129a427981ad',
    'scripts/probe_compiler_operator.py': '69e6312f3dbb895e65d5842c28d4bf7e131fe32d934967587f995e79cd6bf392',
    'scripts/probe_candidate_memory.py': '16407cb62e23a46192cb1132aa973f8629530545766694c0d901f41737ec8d29',
}
DISPATCH_SHA = '2c94dd2fb18d38bd16f57258e6541f7bf0caec181db2f04e2d1675c2cfa86461'
SETTINGS = {'request_index': 0, 'candidate_count': 1, 'captures': 2, 'replays': 3,
            'inner_seconds': INNER_SECONDS, 'outer_seconds': 900,
            'change_backend_or_flags': False, 'full_rollouts': 0, 'acceptance_gate': False}


def stamp(): return datetime.now(timezone.utc).isoformat()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''): value.update(chunk)
    return value.hexdigest()


def save(path, value):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def frozen_inputs():
    actual = {name: sha(ROOT / name) for name in PINS}
    if actual != PINS: raise ValueError('Pinned standalone helper or baseline config changed')
    from drivebench.benchmarks.environment import source_identity
    if source_identity() != SOURCE: raise ValueError('Frozen engine source changed')
    return actual


def load_verifier():
    path = ROOT / 'outputs/host/verify-a100-pilot-v1.py'
    spec = importlib.util.spec_from_file_location('a100_repeatability_gate', path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def scoped_config(path, verifier):
    from drivebench.config import load_config, validate_run_config
    config = load_config(path)
    baseline = verifier.load_config(ROOT / 'outputs/host/a100-pilot-01/configs/baseline.yaml')
    expected = copy.deepcopy(baseline)
    expected.update(name='a100-first-step-repeatability-v1', output='/workspace/outputs/a100-repeatability-v1',
                    numerical_validation=None, repeatability_probe=SETTINGS)
    if config != expected: raise ValueError('Only the scoped first-step diagnostic config is allowed')
    verifier.check_config(config, 'agreement', 1)
    generation, options = validate_run_config(config)
    return config, generation, options


def flags(torch):
    return {'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_enabled': torch.backends.cudnn.enabled, 'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'allow_fp16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'cudnn_version': torch.backends.cudnn.version()}


class CapturedFirstCall(Exception):
    """Intentional sampler exit before the first denoiser evaluation."""


def capture(backend, payload, request, generation, torch):
    captured = {}; had_override = '_denoiser' in vars(backend)
    prior_override = vars(backend).get('_denoiser')
    def intercept(x, sigma, condition, mask):
        if captured: raise RuntimeError('More than one first input was captured')
        captured.update(values={'x': x, 'sigma': sigma, 'condition': condition, 'mask': mask},
                        rng_cpu=torch.random.get_rng_state().clone(),
                        rng_cuda=torch.cuda.get_rng_state(backend.device).clone(),
                        grad_enabled=torch.is_grad_enabled(), autocast_enabled=torch.is_autocast_enabled(),
                        autocast_dtype=str(torch.get_autocast_gpu_dtype()))
        raise CapturedFirstCall()
    backend._denoiser = intercept
    try:
        try: backend.generate_batch(payload, request['trajectories'], request['seeds'], generation)
        except CapturedFirstCall: pass
    finally:
        if had_override: backend._denoiser = prior_override
        else: del backend._denoiser
    if not captured: raise RuntimeError('First denoiser input was not captured')
    return captured


def flat(values):
    return {**{key: values[key] for key in ('x', 'sigma', 'mask')},
            **{'condition.' + key: value for key, value in values['condition'].items()}}


def compare(a, b, helper):
    result = helper.compare_arrays(a, b)
    result['actual_dtype'] = result.pop('compiled_dtype')
    return result


def tensor_array(tensor): return tensor.detach().contiguous().cpu().numpy()


@contextlib.contextmanager
def observations(backend, state, helper, fmha):
    """Observe actual dispatch results; never force an operator or change its input."""
    original = fmha._dispatch_fw
    had_override = '_denoiser' in vars(backend)
    prior_override = vars(backend).get('_denoiser')
    if sha(inspect.getsourcefile(original)) != DISPATCH_SHA:
        raise ValueError('Pinned native dispatch implementation differs')
    handles = []
    def dispatch(inp):
        op = original(inp)
        state['attention'].append({'phase': state['phase'], 'operator': op.NAME, 'operator_module': op.__module__,
            'q_shape': list(inp.query.shape), 'k_shape': list(inp.key.shape), 'v_shape': list(inp.value.shape),
            'q_stride': list(inp.query.stride()), 'k_stride': list(inp.key.stride()), 'v_stride': list(inp.value.stride()),
            'dtype': str(inp.query.dtype), 'dropout_p': inp.p, 'scale': inp.scale,
            'attn_bias_type': type(inp.attn_bias).__name__})
        return op
    def hook(name):
        def record(module, inputs, output):
            if state['phase'].startswith('replay-'):
                state['blocks'].setdefault(state['phase'], []).append(
                    {'module': name, **helper.tensor_fingerprint(output)})
        return record
    network = backend.model.model.diffusion_model
    blocks = [(f'input_blocks.{i}', block) for i, block in enumerate(network.input_blocks)]
    blocks += [('middle_block', network.middle_block)]
    blocks += [(f'output_blocks.{i}', block) for i, block in enumerate(network.output_blocks)]
    blocks += [('out', network.out)]
    try:
        fmha._dispatch_fw = dispatch
        for name, block in blocks: handles.append(block.register_forward_hook(hook(name)))
        yield
    finally:
        for handle in handles: handle.remove()
        fmha._dispatch_fw = original
        state['attention_dispatch_restored'] = fmha._dispatch_fw is original
        state['block_hooks_removed'] = True
        state['denoiser_attribute_restored'] = ('_denoiser' in vars(backend)) == had_override and (
            not had_override or vars(backend)['_denoiser'] is prior_override)


def first_different_block(reference, actual):
    if [x['module'] for x in reference] != [x['module'] for x in actual]:
        raise ValueError('Observed block execution order changed')
    return next((x['module'] for x, y in zip(reference, actual) if x != y), None)


def same_callable(left, right):
    return left is right or (getattr(left, '__self__', None) is not None
        and getattr(left, '__self__', None) is getattr(right, '__self__', None)
        and getattr(left, '__func__', None) is getattr(right, '__func__', None))


def diagnose(engine, request, generation, directory, helper, progress):
    import numpy as np
    import torch
    import xformers.ops.fmha as fmha
    from drivebench.models.vista import _EXECUTION_LOCK
    backend = engine.backend; original = backend._denoiser
    settings = flags(torch)
    training = {name: module.training for name, module in backend.model.named_modules()}
    state = {'phase': 'prepare_context', 'attention': [], 'blocks': {},
             'attention_dispatch_restored': False, 'block_hooks_removed': False}
    progress.update(runtime_flags=settings, module_training=training, observations=state)
    captures = []; output_rows = []; first_output = None
    with observations(backend, state, helper, fmha):
        context = engine.prepare_context(request['observed_frames'])
        progress['context'] = {'cache_hit': context.cache_hit, 'preparation_s': context.preparation_s,
                               'cache_key': context.cache_key, 'cache_stats': engine.cache.stats()}
        for index in range(2):
            state['phase'] = 'capture-' + str(index)
            value = capture(backend, context.payload, request, generation, torch)
            fingerprints = helper.input_fingerprints(value['values'])
            arrays = {name: tensor_array(tensor) for name, tensor in flat(value['values']).items()}
            arrays.update(rng_cpu=value['rng_cpu'].cpu().numpy(), rng_cuda=value['rng_cuda'].cpu().numpy())
            path = directory / f'capture-{index}.npz'
            with path.open('xb') as stream:
                np.savez(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
            row = {'capture_index': index, 'input_fingerprints': fingerprints,
                   'rng_cpu_sha256': helper.array_hash(arrays['rng_cpu']),
                   'rng_cuda_sha256': helper.array_hash(arrays['rng_cuda']),
                   'grad_enabled': value['grad_enabled'], 'autocast_enabled': value['autocast_enabled'],
                   'autocast_dtype': value['autocast_dtype'], 'artifact_sha256': sha(path),
                   'intentional_early_stop': True, 'sampler_rollouts_completed': 0, 'video_decoder_calls': 0}
            save(directory / f'capture-{index}.json', row)
            captures.append(value); progress.setdefault('captures', []).append(row)
        progress['same_seed_construction_comparisons'] = {
            name: compare(tensor_array(tensor), tensor_array(flat(captures[1]['values'])[name]), helper)
            for name, tensor in flat(captures[0]['values']).items()}
        selected = captures[0]; baseline = progress['captures'][0]['input_fingerprints']
        if helper.input_fingerprints(selected['values']) != baseline:
            raise ValueError('Second input construction mutated the original captured input')
        with _EXECUTION_LOCK, torch.cuda.device(backend.device), torch.random.fork_rng(devices=[backend.device.index]):
            for index in range(3):
                state['phase'] = f'replay-{index}'
                torch.random.set_rng_state(selected['rng_cpu'])
                torch.cuda.set_rng_state(selected['rng_cuda'], backend.device)
                rng_before = helper.rng_fingerprint(torch, backend.device)
                if helper.input_fingerprints(selected['values']) != baseline: raise ValueError('Captured input mutated')
                begin = time.monotonic(); x = selected['values']
                # Match the original per-candidate autocast/EMA lifetime, including cast-cache invalidation.
                with torch.no_grad(), backend._autocast(), backend.model.ema_scope('First-step repeatability diagnostic'):
                    output = original(x['x'], x['sigma'], x['condition'], x['mask']); backend.synchronize()
                array = tensor_array(output)
                if array.shape != (50, 4, 72, 128) or not np.isfinite(array).all(): raise ValueError('Invalid denoiser output')
                path = directory / f'replay-{index}.npy'
                with path.open('xb') as stream:
                    np.save(stream, array, allow_pickle=False); stream.flush(); os.fsync(stream.fileno())
                row = {'replay_index': index, 'output_sha256': helper.array_hash(array), 'artifact_sha256': sha(path),
                       'output_dtype': str(output.dtype), 'output_shape': list(array.shape),
                       'rng_before': rng_before, 'rng_after': helper.rng_fingerprint(torch, backend.device),
                       'inputs_unchanged': helper.input_fingerprints(x) == baseline,
                       'instrumented_wall_s': time.monotonic() - begin}
                row['rng_unchanged'] = row['rng_before'] == row['rng_after']
                if first_output is None: first_output = array.copy()
                row['comparison_to_first'] = compare(first_output, array, helper)
                row['first_different_block'] = first_different_block(state['blocks']['replay-0'], state['blocks'][state['phase']])
                save(directory / f'replay-{index}.json', row); output_rows.append(row)
                progress['replays'] = output_rows
                if not row['inputs_unchanged']: raise ValueError('Denoiser mutated captured input')
    progress.update(denoiser_callable_restored=same_callable(backend._denoiser, original), runtime_flags_unchanged=flags(torch) == settings,
                    module_training_unchanged={n: m.training for n, m in backend.model.named_modules()} == training,
                    exact_repeat_observed=all(r['comparison_to_first']['passed_exact'] for r in output_rows),
                    denoiser_evaluations=3, first_input_constructions=2, sampler_rollouts_completed=0, video_decoder_calls=0)
    if not all(progress[key] for key in ('denoiser_callable_restored','runtime_flags_unchanged','module_training_unchanged')):
        raise ValueError('Probe changed runtime/callable/module state')


def run(config_path, output):
    bindings = frozen_inputs(); verifier = load_verifier()
    config, generation, options = scoped_config(config_path, verifier)
    verifier.deadline_check(DEADLINE, 900)
    output = Path(output)
    if output.resolve() != (ROOT / 'outputs/a100-repeatability-v1').resolve():
        raise ValueError('Use the exact fresh diagnostic output directory')
    output.mkdir(parents=True, exist_ok=False)
    from drivebench.benchmarks.environment import collect_environment, reference_identity, source_identity
    from drivebench.data import load_manifest, inference_request, verify_input_checksums
    from drivebench.engine import InferenceEngine
    from scripts import probe_compiler_operator as helper
    bindings.update({str(Path(config_path).resolve()): sha(config_path), str(Path(__file__).resolve()): sha(__file__)})
    for name in (config['data']['manifest'], config['data']['manifest'] + '.meta.json',
                 config['environment_verification'], config['environment_lock']): bindings[name] = sha(name)
    proof = json.loads(Path(config['environment_verification']).read_text())
    bindings.update(proof['native_xformers_provenance']['artifact_sha256'])
    value = {'type': 'a100_first_step_repeatability_diagnostic_v1', 'started_at': stamp(), 'status': 'running',
             'source_identity': SOURCE, 'input_sha256': bindings, 'settings': SETTINGS,
             'numerical_acceptance': False, 'headline_eligible': False, 'full_generation_passed': False,
             'scope': 'Instrumented first-step localization on one A100 development scene; no backend/flag/tolerance changes.'}
    save(output / 'attempt.json', value)
    def timeout(signum, frame): raise TimeoutError('840-second diagnostic bound reached')
    old = signal.signal(signal.SIGALRM, timeout); signal.alarm(INNER_SECONDS)
    code = 1
    try:
        value['environment_gate'] = verifier.gate(config, DEADLINE)
        save(output / 'environment-gate.json', value['environment_gate'])
        records = load_manifest(config['data']['manifest'])
        if len(records) != 1 or records[0]['request_id'] != verifier.SCENE_REQUEST:
            raise ValueError('Unexpected diagnostic request')
        verify_input_checksums(records[0], config['data']['root'], include_future=False)
        request = inference_request(records[0], config['data']['root'], candidate_count=1)
        value['request'] = {'request_id': request['request_id'], 'candidate_ids': request['candidate_ids'],
                            'seeds': request['seeds'], 'trajectories': [x.tolist() for x in request['trajectories']]}
        engine = InferenceEngine(config['model'], generation, options)
        value['model_identity'] = engine.backend.identity
        diagnose(engine, request, generation, output, helper, value)
        verify_input_checksums(records[0], config['data']['root'], include_future=False)
        if reference_identity(collect_environment()) != value['environment_gate']['runtime_identity']:
            raise ValueError('Runtime identity changed')
        if source_identity() != SOURCE or any(sha(Path(name) if Path(name).is_absolute() else ROOT/name) != h
                                              for name, h in bindings.items()):
            raise ValueError('Source or diagnostic input changed')
        value['inputs_and_source_unchanged'] = True
        value['status'] = 'completed'; code = 0
    except BaseException as error:
        value['status'] = 'failed'; value['error'] = {'type': type(error).__name__, 'message': str(error)}
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old)
        value['finished_at'] = stamp()
        value['artifact_sha256'] = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
        save(output / 'probe.json', value)
    print(json.dumps({'status': value['status'], 'exact_repeat_observed': value.get('exact_repeat_observed'),
                      'output': str(output), 'numerical_acceptance': False}))
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); raise SystemExit(run(args.config, args.output))
