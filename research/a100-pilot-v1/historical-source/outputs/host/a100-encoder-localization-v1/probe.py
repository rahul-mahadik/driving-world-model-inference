#!/usr/bin/env python3
"""Two original early captures to localize first-stage encoder divergence; no rollout/gate."""
from __future__ import annotations
import argparse
import contextlib
import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
BASE = ROOT/'outputs/host/a100-repeatability-v1/probe.py'
BASE_SHA = '505b688898f5ad6ddffc3714a478c7a5d7507efb062107de4cd2b6aeebea51e3'
SETTINGS = {'captures': 2, 'denoiser_evaluations': 0, 'full_rollouts': 0,
            'inner_seconds': 840, 'outer_seconds': 900, 'change_backend_or_flags': False,
            'save_first_frame_only': True, 'acceptance_gate': False}


def load_base():
    import hashlib
    if hashlib.sha256(BASE.read_bytes()).hexdigest() != BASE_SHA:
        raise ValueError('Reviewed first-step helper changed')
    spec = importlib.util.spec_from_file_location('a100_localization_base', BASE)
    base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
    return base


def scoped_config(path, base, verifier):
    from drivebench.config import load_config, validate_run_config
    cfg = load_config(path)
    expected = copy.deepcopy(verifier.load_config(ROOT/'outputs/host/a100-pilot-01/configs/baseline.yaml'))
    expected.update(name='a100-encoder-localization-v1', output='/workspace/outputs/a100-encoder-localization-v1',
                    numerical_validation=None, encoder_localization=SETTINGS)
    if cfg != expected: raise ValueError('Only exact original-baseline encoder localization configuration is allowed')
    verifier.check_config(cfg, 'agreement', 1)
    generation, options = validate_run_config(cfg)
    return cfg, generation, options


class Recorder:
    def __init__(self, directory, base, helper, torch, device, progress):
        self.directory, self.base, self.helper = directory, base, helper
        self.torch, self.device, self.progress = torch, device, progress
        self.capture_index = 0; self.counts = {}; self.reference = {}; self.orders = [[], []]
        progress['tensor_observations'] = []; progress['rng_observations'] = []

    def rng(self, label):
        row = {'capture': self.capture_index, 'label': label, **self.helper.rng_fingerprint(self.torch, self.device)}
        self.progress['rng_observations'].append(row)

    def tensor(self, label, tensor):
        import numpy as np
        count = self.counts.get(label, 0); self.counts[label] = count + 1
        key = label + '-' + str(count)
        self.orders[self.capture_index].append(key)
        array = self.base.tensor_array(tensor).copy()
        if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
            raise ValueError('Nonfinite or nonfloating observed tensor')
        path = self.directory/f'capture-{self.capture_index}-{key}-first.npy'
        with path.open('xb') as stream:
            np.save(stream, array[:1], allow_pickle=False); stream.flush(); os.fsync(stream.fileno())
        row = {'capture': self.capture_index, 'label': key, 'shape': list(tensor.shape),
               'stride': list(tensor.stride()), 'dtype': str(tensor.dtype), 'device_type': tensor.device.type,
               'full_array_sha256': self.helper.array_hash(array),
               'first_slice_file': path.name, 'first_slice_file_sha256': self.base.sha(path),
               'autocast_enabled': self.torch.is_autocast_enabled(),
               'grad_enabled': self.torch.is_grad_enabled()}
        if self.capture_index == 0: self.reference[key] = array
        else:
            if key not in self.reference: raise ValueError('Tensor observation sequence changed')
            left = self.reference[key]
            row['comparison_to_first_capture'] = self.base.compare(left, array, self.helper)
            row['per_leading_index'] = [dict(index=i, **self.base.compare(left[i:i+1], array[i:i+1], self.helper))
                                        for i in range(len(array))]
        self.progress['tensor_observations'].append(row)


@contextlib.contextmanager
def replace_attribute(owner, name, value):
    present = name in vars(owner); prior = vars(owner).get(name)
    setattr(owner, name, value)
    try: yield
    finally:
        if present: setattr(owner, name, prior)
        else: delattr(owner, name)


@contextlib.contextmanager
def instrument(backend, generation, recorder, progress):
    """Observe existing outputs, preserving call arguments, result identity, and RNG draws."""
    handles = []; sampler = backend._sampler(generation)
    original_encode, original_prepare = backend._encode, sampler.prepare_sampling_loop
    def encoder_post(_module, _inputs, output): recorder.tensor('encoder-parameters', output)
    def regularizer_pre(_module, inputs):
        recorder.rng('before-regularizer'); recorder.tensor('regularizer-parameters', inputs[0])
    def regularizer_post(_module, _inputs, output):
        recorder.rng('after-regularizer'); recorder.tensor('regularizer-sampled', output[0])
    def encode(payload, stats):
        recorder.rng('before-encode'); recorder.tensor('encode-images', payload['images'])
        result = original_encode(payload, stats)
        recorder.rng('after-encode'); recorder.tensor('encode-scaled-z', result)
        return result
    def prepare(x, *args, **kwargs):
        recorder.rng('before-sampler-prepare'); recorder.tensor('sampler-incoming-noise', x)
        result = original_prepare(x, *args, **kwargs)
        recorder.tensor('sampler-scaled-noise', result[0]); recorder.rng('after-sampler-prepare')
        return result
    try:
        handles.append(backend.model.first_stage_model.encoder.register_forward_hook(encoder_post))
        handles.append(backend.model.first_stage_model.regularization.register_forward_pre_hook(regularizer_pre))
        handles.append(backend.model.first_stage_model.regularization.register_forward_hook(regularizer_post))
        with replace_attribute(backend, '_encode', encode), replace_attribute(sampler, 'prepare_sampling_loop', prepare):
            yield
    finally:
        for handle in handles: handle.remove()
        progress['instrumentation_restored'] = recorder.base.same_callable(backend._encode, original_encode) and recorder.base.same_callable(sampler.prepare_sampling_loop, original_prepare)
        progress['forward_hooks_removed'] = True


def diagnose(engine, request, generation, output, base, helper, progress):
    import torch
    backend = engine.backend; original = backend._denoiser
    original_flags = base.flags(torch)
    training = {name: module.training for name, module in backend.model.named_modules()}
    progress.update(runtime_flags=original_flags, module_training=training,
                    sdp_flags={name: getattr(torch.backends.cuda, name)() for name in
                               ('flash_sdp_enabled', 'mem_efficient_sdp_enabled', 'math_sdp_enabled')})
    recorder = Recorder(output, base, helper, torch, backend.device, progress)
    context = engine.prepare_context(request['observed_frames'])
    progress['context'] = {'cache_key': context.cache_key, 'cache_hit': context.cache_hit, 'cache_stats': engine.cache.stats()}
    try:
        with instrument(backend, generation, recorder, progress):
            for index in range(2):
                recorder.capture_index = index; recorder.counts = {}
                value = base.capture(backend, context.payload, request, generation, torch)
                helper.input_fingerprints(value['values'])  # real primary CFG/CUDA layout assertion
                for name, tensor in base.flat(value['values']).items(): recorder.tensor('captured-' + name, tensor)
                progress.setdefault('capture_rng', []).append({'capture': index,
                    'cpu': helper.array_hash(value['rng_cpu'].cpu().numpy()),
                    'cuda': helper.array_hash(value['rng_cuda'].cpu().numpy())})
                base.save(output/f'capture-{index}-observations.json', {
                    'capture': index, 'observations': [x for x in progress['tensor_observations'] if x['capture']==index],
                    'rng': [x for x in progress['rng_observations'] if x['capture']==index],
                    'intentional_early_stop': True, 'denoiser_evaluations': 0, 'full_rollouts': 0})
        if recorder.orders[0] != recorder.orders[1]: raise ValueError('Observed execution order changed')
        for label, expected in [('encoder-parameters',2), ('regularizer-parameters',2), ('regularizer-sampled',2),
                                ('encode-images',1), ('encode-scaled-z',1), ('sampler-incoming-noise',1), ('sampler-scaled-noise',1)]:
            if recorder.counts.get(label) != expected: raise ValueError('Original 14+11 encode/chunk execution changed: '+label)
        progress['first_different_observation'] = next((r['label'] for r in progress['tensor_observations']
            if r['capture']==1 and not r['comparison_to_first_capture']['passed_exact']), None)
        progress.update(denoiser_evaluations=0, first_input_constructions=2, sampler_rollouts_completed=0, video_decoder_calls=0)
    finally:
        progress['denoiser_callable_restored'] = base.same_callable(backend._denoiser, original)
        progress['runtime_flags_unchanged'] = base.flags(torch) == original_flags
        progress['module_training_unchanged'] = {n: m.training for n,m in backend.model.named_modules()} == training
    if not all(progress[k] for k in ('instrumentation_restored','forward_hooks_removed','denoiser_callable_restored',
                                     'runtime_flags_unchanged','module_training_unchanged')):
        raise ValueError('Instrumentation changed runtime/callable state')


def run(config_path, output):
    base = load_base(); bindings = base.frozen_inputs(); verifier = base.load_verifier()
    config, generation, options = scoped_config(config_path, base, verifier)
    verifier.deadline_check(base.DEADLINE, 900)
    output = Path(output)
    if output.resolve() != (ROOT/'outputs/a100-encoder-localization-v1').resolve(): raise ValueError('Unexpected output path')
    output.mkdir(parents=True, exist_ok=False)
    from drivebench.benchmarks.environment import collect_environment, reference_identity, source_identity
    from drivebench.data import load_manifest, inference_request, verify_input_checksums
    from drivebench.engine import InferenceEngine
    from scripts import probe_compiler_operator as helper
    for path in (BASE, Path(__file__), Path(config_path), Path(config['data']['manifest']),
                 Path(config['data']['manifest']+'.meta.json'), Path(config['environment_verification']), Path(config['environment_lock'])):
        bindings[str(path.resolve())] = base.sha(path)
    proof = json.loads(Path(config['environment_verification']).read_text())
    bindings.update(proof['native_xformers_provenance']['artifact_sha256'])
    value = {'type':'a100_encoder_localization_v1', 'started_at':base.stamp(), 'status':'running',
             'source_identity':base.SOURCE, 'input_sha256':bindings, 'settings':SETTINGS,
             'numerical_acceptance':False, 'headline_eligible':False, 'full_generation_passed':False,
             'scope':'Instrumented two early captures; first-stage parameters/posterior/noise only; no denoising or decoding.'}
    base.save(output/'attempt.json',value)
    def timeout(_signum,_frame): raise TimeoutError('840-second localization bound reached')
    previous = signal.signal(signal.SIGALRM, timeout); signal.alarm(840); code=1
    try:
        value['environment_gate'] = verifier.gate(config, base.DEADLINE)
        base.save(output/'environment-gate.json',value['environment_gate'])
        records=load_manifest(config['data']['manifest'])
        if len(records)!=1 or records[0]['request_id']!=verifier.SCENE_REQUEST: raise ValueError('Unexpected request')
        verify_input_checksums(records[0], config['data']['root'], include_future=False)
        request=inference_request(records[0],config['data']['root'],candidate_count=1)
        value['request']={'request_id':request['request_id'],'candidate_ids':request['candidate_ids'],'seeds':request['seeds']}
        engine=InferenceEngine(config['model'],generation,options); value['model_identity']=engine.backend.identity
        diagnose(engine,request,generation,output,base,helper,value)
        verify_input_checksums(records[0],config['data']['root'],include_future=False)
        if reference_identity(collect_environment())!=value['environment_gate']['runtime_identity']: raise ValueError('Runtime identity changed')
        if source_identity()!=base.SOURCE or any(base.sha(Path(n) if Path(n).is_absolute() else ROOT/n)!=h for n,h in bindings.items()):
            raise ValueError('Source or input changed')
        value['inputs_and_source_unchanged']=True; value['status']='completed'; code=0
    except BaseException as error: value.update(status='failed',error={'type':type(error).__name__,'message':str(error)})
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM,previous)
        value['finished_at']=base.stamp(); value['artifact_sha256']={p.name:base.sha(p) for p in output.iterdir() if p.is_file()}
        base.save(output/'probe.json',value)
    print(json.dumps({'status':value['status'],'first_different_observation':value.get('first_different_observation'),
                      'numerical_acceptance':False,'output':str(output)}))
    return code


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True); parser.add_argument('--output',required=True)
    args=parser.parse_args(); raise SystemExit(run(args.config,args.output))
