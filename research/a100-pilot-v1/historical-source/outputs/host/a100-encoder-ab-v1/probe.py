#!/usr/bin/env python3
"""Scoped first-stage diagnostic interventions; never a runtime/correctness acceptance proof."""
from __future__ import annotations
import argparse
import contextlib
import copy
import importlib.util
import json
from pathlib import Path
import signal
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
ENCODER=ROOT/'outputs/host/a100-encoder-localization-v1/probe.py'
ENCODER_SHA='f4a491af06a462e901b30e72c4d7ae3b30adfe6bb10fc07e6aec73fbef37e0d1'
PHASES=('original','cudnn_deterministic','sdpa_math','original_after')
SETTINGS={'phases':list(PHASES),'captures_per_phase':2,'scope':'backend._encode only',
          'denoiser_evaluations':0,'full_rollouts':0,'inner_seconds':840,'outer_seconds':900,
          'change_inference_configuration':False,'acceptance_gate':False,'oom_fallback':False}


def load_encoder():
    import hashlib
    if hashlib.sha256(ENCODER.read_bytes()).hexdigest()!=ENCODER_SHA:raise ValueError('Reviewed encoder helper changed')
    spec=importlib.util.spec_from_file_location('a100_ab_encoder',ENCODER)
    encoder=importlib.util.module_from_spec(spec);spec.loader.exec_module(encoder)
    return encoder


def scoped_config(path,base,verifier):
    from drivebench.config import load_config,validate_run_config
    cfg=load_config(path)
    expected=copy.deepcopy(verifier.load_config(ROOT/'outputs/host/a100-pilot-01/configs/baseline.yaml'))
    expected.update(name='a100-encoder-ab-v1',output='/workspace/outputs/a100-encoder-ab-v1',
                    numerical_validation=None,encoder_ab=SETTINGS)
    if cfg!=expected:raise ValueError('Only the exact original-baseline scoped A/B diagnostic is allowed')
    verifier.check_config(cfg,'agreement',1)
    generation,options=validate_run_config(cfg)
    return cfg,generation,options


def snapshot(torch,base):
    return {'runtime':base.flags(torch),'sdp':{name:getattr(torch.backends.cuda,name)() for name in
            ('flash_sdp_enabled','mem_efficient_sdp_enabled','math_sdp_enabled')},
            'cudnn_benchmark_limit':torch.backends.cudnn.benchmark_limit}


@contextlib.contextmanager
def policy(torch,base,phase,row):
    if phase not in PHASES:raise ValueError('Unknown diagnostic phase')
    before=snapshot(torch,base);row['flags_before']=before
    expected=copy.deepcopy(before)
    if phase=='cudnn_deterministic':expected['runtime']['cudnn_deterministic']=True
    elif phase=='sdpa_math':expected['sdp']={'flash_sdp_enabled':False,'mem_efficient_sdp_enabled':False,'math_sdp_enabled':True}
    try:
        with contextlib.ExitStack() as stack:
            if phase=='cudnn_deterministic':
                torch.backends.cudnn.deterministic=True
                stack.callback(setattr,torch.backends.cudnn,'deterministic',before['runtime']['cudnn_deterministic'])
            elif phase=='sdpa_math':
                stack.enter_context(torch.backends.cuda.sdp_kernel(enable_flash=False,enable_mem_efficient=False,enable_math=True))
            row['flags_active']=snapshot(torch,base)
            if row['flags_active']!=expected:raise ValueError('Diagnostic changed unexpected runtime flags')
            yield
    finally:
        row['flags_after']=snapshot(torch,base)
        row['flags_restored']=row['flags_after']==before
        if not row['flags_restored']:raise ValueError('Diagnostic runtime flags were not restored')


@contextlib.contextmanager
def encode_policy(backend,torch,base,encoder,phase,progress):
    original=backend._encode
    calls=progress.setdefault('encode_policy_calls',[])
    def wrapped(payload,stats):
        row={'call':len(calls),'phase':phase,'scope':'backend._encode only','status':'running',
             'sdpa_selection_evidence':'math backend only permitted' if phase=='sdpa_math' else 'automatic backend; actual kernel not profiled'};calls.append(row)
        try:
            with policy(torch,base,phase,row):
                result=original(payload,stats)
                backend.synchronize()
            row['status']='completed';return result
        except BaseException as error:
            row.update(status='failed',error={'type':type(error).__name__,'message':str(error)});raise
    try:
        with encoder.replace_attribute(backend,'_encode',wrapped):yield
    finally:progress['policy_wrapper_restored']=base.same_callable(backend._encode,original)


def phase_summary(progress):
    rows={r['label']:r['comparison_to_first_capture'] for r in progress['tensor_observations'] if r['capture']==1}
    required=('encoder-parameters-0','encoder-parameters-1','encode-scaled-z-0','sampler-incoming-noise-0',
              'sampler-scaled-noise-0','captured-x-0')
    if any(key not in rows for key in required):raise ValueError('Phase lacks required encoder/noise evidence')
    return {'comparisons':{key:rows[key] for key in required},
            'all_observed_tensors_repeat_exactly':all(row['passed_exact'] for row in rows.values()),
            'first_different_observation':progress['first_different_observation'],
            'capture_rng_equal':progress['capture_rng'][0]['cpu']==progress['capture_rng'][1]['cpu'] and
                                progress['capture_rng'][0]['cuda']==progress['capture_rng'][1]['cuda'],
            'numerical_acceptance':False,'full_generation_passed':False}


def run(config_path,output):
    encoder=load_encoder();base=encoder.load_base();bindings=base.frozen_inputs();verifier=base.load_verifier()
    config,generation,options=scoped_config(config_path,base,verifier)
    verifier.deadline_check(base.DEADLINE,900)
    output=Path(output)
    if output.resolve()!=(ROOT/'outputs/a100-encoder-ab-v1').resolve():raise ValueError('Unexpected output directory')
    output.mkdir(parents=True,exist_ok=False)
    from drivebench.benchmarks.environment import collect_environment,reference_identity,source_identity
    from drivebench.data import load_manifest,inference_request,verify_input_checksums
    from drivebench.engine import InferenceEngine
    from scripts import probe_compiler_operator as helper
    for path in (ENCODER,encoder.BASE,Path(__file__),Path(config_path),Path(config['data']['manifest']),
                 Path(config['data']['manifest']+'.meta.json'),Path(config['environment_verification']),Path(config['environment_lock'])):
        bindings[str(path.resolve())]=base.sha(path)
    proof=json.loads(Path(config['environment_verification']).read_text())
    bindings.update(proof['native_xformers_provenance']['artifact_sha256'])
    value={'type':'a100_encoder_scoped_ab_diagnostic_v1','started_at':base.stamp(),'status':'running',
           'source_identity':base.SOURCE,'input_sha256':bindings,'settings':SETTINGS,'phases':[],
           'numerical_acceptance':False,'headline_eligible':False,'full_generation_passed':False,
           'runtime_proof_scope':'Original runtime admitted before experiment and checked after restoration; temporary phases are NOT covered by that proof.',
           'interpretation':'Two repetitions per ordered phase are localization evidence, not a specific-kernel or full-generation determinism proof.',
           'memory_scope':{'one_fp32_score_matrix_bytes':14*9216*9216*4,
                           'total_peak_not_predicted':True,'no_batch_chunk_or_backend_fallback':True}}
    base.save(output/'attempt.json',value)
    def timeout(_signum,_frame):raise TimeoutError('840-second diagnostic bound reached')
    previous=signal.signal(signal.SIGALRM,timeout);signal.alarm(840);code=1;engine=None;original_flags=None
    try:
        value['environment_gate']=verifier.gate(config,base.DEADLINE)
        base.save(output/'environment-gate.json',value['environment_gate'])
        records=load_manifest(config['data']['manifest'])
        if len(records)!=1 or records[0]['request_id']!=verifier.SCENE_REQUEST:raise ValueError('Unexpected request')
        verify_input_checksums(records[0],config['data']['root'],include_future=False)
        request=inference_request(records[0],config['data']['root'],candidate_count=1)
        value['request']={'request_id':request['request_id'],'candidate_ids':request['candidate_ids'],'seeds':request['seeds']}
        import torch
        engine=InferenceEngine(config['model'],generation,options);value['model_identity']=engine.backend.identity
        if engine.backend.model.disable_first_stage_autocast is not True or engine.backend.model.en_and_decode_n_samples_a_time!=14:
            raise ValueError('Original first-stage FP32/chunk configuration differs')
        original_flags=snapshot(torch,base);value['original_flags']=original_flags
        if original_flags['runtime']['cudnn_deterministic'] is not False:
            raise ValueError('Original runtime no longer has the observed nondeterministic cuDNN setting')
        for phase in PHASES:
            if snapshot(torch,base)!=original_flags:raise ValueError('Flags differ before phase')
            directory=output/phase;directory.mkdir(exist_ok=False)
            progress={'phase':phase,'status':'running','started_at':base.stamp(),'flags_before':snapshot(torch,base)}
            value['phases'].append(progress);base.save(directory/'attempt.json',progress)
            try:
                free,total=torch.cuda.mem_get_info(engine.backend.device)
                progress['memory_before']={'free_bytes':free,'total_bytes':total}
                engine.backend.reset_peak_memory()
                with encode_policy(engine.backend,torch,base,encoder,phase,progress):
                    encoder.diagnose(engine,request,generation,directory,base,helper,progress)
                if len(progress['encode_policy_calls'])!=2 or any(r['status']!='completed' or not r['flags_restored'] for r in progress['encode_policy_calls']):
                    raise ValueError('Phase did not complete exactly two restored encoder policies')
                if not progress['policy_wrapper_restored']:raise ValueError('Policy wrapper was not restored')
                progress['summary']=phase_summary(progress);progress['status']='completed'
            except BaseException as error:
                progress.update(status='failed',error={'type':type(error).__name__,'message':str(error)})
                raise
            finally:
                progress['flags_after']=snapshot(torch,base);progress['phase_flags_restored']=progress['flags_after']==original_flags
                progress['peak_allocated_bytes'],progress['peak_reserved_bytes']=engine.backend.peak_memory()
                progress['finished_at']=base.stamp()
                progress['artifact_sha256']={p.name:base.sha(p) for p in directory.iterdir() if p.is_file()}
                base.save(directory/'phase.json',progress)
            if not progress['phase_flags_restored']:raise ValueError('Phase runtime flags were not restored')
        verify_input_checksums(records[0],config['data']['root'],include_future=False)
        if reference_identity(collect_environment())!=value['environment_gate']['runtime_identity']:raise ValueError('Original runtime identity changed after experiment')
        if source_identity()!=base.SOURCE or any(base.sha(Path(n) if Path(n).is_absolute() else ROOT/n)!=h for n,h in bindings.items()):
            raise ValueError('Source or input changed')
        value.update(inputs_and_source_unchanged=True,original_runtime_restored=True,status='completed');code=0
    except BaseException as error:value.update(status='failed',error={'type':type(error).__name__,'message':str(error)})
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous)
        if original_flags is not None:value['final_flags_restored']=snapshot(torch,base)==original_flags
        value['finished_at']=base.stamp()
        value['artifact_sha256']={str(p.relative_to(output)):base.sha(p) for p in output.rglob('*') if p.is_file()}
        base.save(output/'probe.json',value)
    print(json.dumps({'status':value['status'],'numerical_acceptance':False,'output':str(output)}));return code


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();raise SystemExit(run(args.config,args.output))
