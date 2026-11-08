#!/usr/bin/env python3
"""Explicit process-wide A100 math-SDPA runtime; frozen engine and old evidence unchanged."""
from __future__ import annotations
import argparse
import contextlib
import copy
import importlib.util
import json
import math
import re
from pathlib import Path
import signal
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
AB=ROOT/'outputs/host/a100-encoder-ab-v1/probe.py'
AB_SHA='799eb4b9d92c6445eccc82d412a7c38a82c22eafcaa61edfbc6dbb6364cb4ad4'
AB_RESULT=ROOT/'outputs/a100-encoder-ab-v1/probe.json'
NAME='a100_process_math_sdpa_v1'
PROOF_DIR=ROOT/'outputs/environment-a100-math-sdpa-v1'
VALIDATE_DIR=ROOT/'outputs/a100-math-sdpa-v1'
CONFIG_PATH=ROOT/'outputs/host/a100-math-sdpa-v1/baseline.json'
POLICY_PATHS={'name':NAME,'verification':'/workspace/outputs/environment-a100-math-sdpa-v1/policy-verification.json',
              'lock':'/workspace/outputs/environment-a100-math-sdpa-v1/policy-lock.json'}
SMOKE_SHAPE=(1,1,9216,512)


def dependencies():
    import hashlib
    if hashlib.sha256(AB.read_bytes()).hexdigest()!=AB_SHA:raise ValueError('Reviewed A/B helper changed')
    spec=importlib.util.spec_from_file_location('math_sdpa_ab',AB)
    ab=importlib.util.module_from_spec(spec);spec.loader.exec_module(ab)
    encoder=ab.load_encoder();base=encoder.load_base();base.frozen_inputs()
    return ab,encoder,base,base.load_verifier()


def declaration(base):
    return {'name':NAME,'scope':'process-wide torch SDPA from startup through model teardown',
            'flash_sdp_enabled':False,'mem_efficient_sdp_enabled':False,'math_sdp_enabled':True,
            'xformers_attention_policy':'unchanged upstream','other_runtime_flags':'unchanged',
            'wrapper_sha256':base.sha(__file__),'engine_source_identity':base.SOURCE,
            'execution_contract':'same-process; every torch SDPA call checks active flags; no fallback'}


def config_checked(path,base,verifier):
    from drivebench.config import load_config,validate_run_config
    config=load_config(path)
    expected=copy.deepcopy(verifier.load_config(ROOT/'outputs/host/a100-pilot-01/configs/baseline.yaml'))
    expected.update(name='a100-math-sdpa-v1-baseline',output='/workspace/outputs/a100-math-sdpa-v1',
                    numerical_validation=None,algorithm_policy=POLICY_PATHS)
    if config!=expected:raise ValueError('Require exact declared math-SDPA baseline configuration')
    verifier.check_config(config,'all',2)
    return config


def file_map(paths,base):return {str(Path(p).resolve()):base.sha(p) for p in paths}


def recheck(bindings,base):
    for name,digest in bindings.items():
        p=Path(name) if Path(name).is_absolute() else ROOT/name
        if p.is_symlink() or not p.is_file() or base.sha(p)!=digest:raise ValueError('Bound artifact changed: '+str(p))
    from drivebench.benchmarks.environment import source_identity
    if source_identity()!=base.SOURCE:raise ValueError('Frozen engine source changed')


def ab_evidence(ab,base):
    result=json.loads(AB_RESULT.read_text())
    phases=result.get('phases',[])
    if (result.get('type')!='a100_encoder_scoped_ab_diagnostic_v1' or result.get('status')!='completed'
            or result.get('source_identity')!=base.SOURCE or result.get('settings')!=ab.SETTINGS
            or result.get('inputs_and_source_unchanged') is not True or result.get('original_runtime_restored') is not True
            or result.get('final_flags_restored') is not True or result.get('numerical_acceptance') is not False
            or [x.get('phase') for x in phases]!=list(ab.PHASES)):
        raise ValueError('Completed original/scoped A/B evidence required')
    for phase in phases:
        if (phase.get('status')!='completed' or phase.get('phase_flags_restored') is not True
                or phase.get('policy_wrapper_restored') is not True or phase.get('denoiser_evaluations')!=0
                or phase.get('first_input_constructions')!=2 or phase.get('summary',{}).get('capture_rng_equal') is not True):
            raise ValueError('A/B phase lifecycle or unchanged RNG evidence missing')
        expected=phase['phase']=='sdpa_math'
        if phase['summary'].get('all_observed_tensors_repeat_exactly') is not expected:
            raise ValueError('A/B evidence differs from the reviewed outcome')
        calls=phase.get('encode_policy_calls',[])
        if len(calls)!=2 or any(x.get('status')!='completed' or x.get('flags_restored') is not True for x in calls):
            raise ValueError('A/B policy activation/restoration incomplete')
        if expected and any(x.get('flags_active',{}).get('sdp')!=
            {'flash_sdp_enabled':False,'mem_efficient_sdp_enabled':False,'math_sdp_enabled':True} for x in calls):
            raise ValueError('A/B math policy was not active')
    bindings=file_map([AB_RESULT],base)
    artifacts=result.get('artifact_sha256',{})
    if not artifacts:raise ValueError('A/B artifact inventory missing')
    for name,digest in artifacts.items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe A/B artifact path')
        bindings[str(AB_RESULT.parent/relative)]=digest
    bindings.update(result.get('input_sha256',{}));recheck(bindings,base)
    return {'path':str(AB_RESULT),'sha256':base.sha(AB_RESULT),'scope':'Development localization rationale only; no full-generation acceptance.',
            'artifact_sha256':bindings}


def input_bindings(config_path,config,ab,encoder,base):
    paths=[Path(__file__),Path(config_path),AB,ab.ENCODER,encoder.BASE]
    paths += [ROOT/p for p in base.PINS]
    paths += [Path(config['data']['manifest']),Path(config['data']['manifest']+'.meta.json'),
              Path(config['environment_verification']),Path(config['environment_lock'])]
    return file_map(paths,base)


def assert_math(torch):
    flags={name:getattr(torch.backends.cuda,name)() for name in ('flash_sdp_enabled','mem_efficient_sdp_enabled','math_sdp_enabled')}
    if flags!={'flash_sdp_enabled':False,'mem_efficient_sdp_enabled':False,'math_sdp_enabled':True}:
        raise ValueError('Declared process-wide math-SDPA policy is not active')
    return flags


@contextlib.contextmanager
def activate(torch,ab,base,receipt):
    import torch.nn.functional as functional
    before=ab.snapshot(torch,base);receipt['flags_before']=before
    expected=copy.deepcopy(before);expected['sdp']={'flash_sdp_enabled':False,'mem_efficient_sdp_enabled':False,'math_sdp_enabled':True}
    original=functional.scaled_dot_product_attention;calls={};receipt['sdpa_calls']=calls;receipt['sdpa_call_count']=0
    def checked(*args,**kwargs):
        assert_math(torch)
        tensors=[args[i] if len(args)>i else kwargs[k] for i,k in enumerate(('query','key','value'))]
        geometry=json.dumps([{'shape':list(t.shape),'dtype':str(t.dtype),'device':t.device.type} for t in tensors],sort_keys=True)
        calls[geometry]=calls.get(geometry,0)+1;receipt['sdpa_call_count']+=1
        result=original(*args,**kwargs);assert_math(torch);return result
    try:
        with torch.backends.cuda.sdp_kernel(enable_flash=False,enable_mem_efficient=False,enable_math=True):
            receipt['flags_active']=ab.snapshot(torch,base)
            if receipt['flags_active']!=expected:raise ValueError('Unexpected runtime flag change while selecting math SDPA')
            functional.scaled_dot_product_attention=checked
            try:yield
            finally:
                receipt['active_flags_unchanged']=ab.snapshot(torch,base)==expected
                functional.scaled_dot_product_attention=original
    finally:
        receipt['callable_restored']=functional.scaled_dot_product_attention is original
        receipt['flags_after']=ab.snapshot(torch,base);receipt['flags_restored']=receipt['flags_after']==before
        if not receipt['callable_restored'] or not receipt['flags_restored']:raise ValueError('Math runtime restoration failed')


def trace_evidence(path):
    events=json.loads(Path(path).read_text()).get('traceEvents',[])
    kernels=[e for e in events if e.get('cat')=='kernel' and type(e.get('dur')) in (int,float) and math.isfinite(e['dur']) and e['dur']>0]
    math_events=[e for e in events if e.get('name')=='aten::_scaled_dot_product_attention_math']
    if (not kernels or len(math_events)!=1 or any(e.get('name') in
            {'aten::_scaled_dot_product_efficient_attention','aten::_scaled_dot_product_flash_attention'} for e in events)):
        raise ValueError('Fresh profile lacks exclusive math SDPA operator and positive GPU kernel duration')
    return {'math_operator_events':len(math_events),'cuda_kernel_events':len(kernels),
            'summed_cuda_kernel_duration_us':sum(e['dur'] for e in kernels),
            'cuda_kernel_names':sorted(set(e['name'] for e in kernels)),
            'duration_scope':'sum of instrumented kernel durations, not wall latency; no fresh hardware counters'}


def smoke(torch,output,base):
    import torch.nn.functional as functional
    from scripts.probe_compiler_operator import array_hash,rng_fingerprint
    device=torch.device('cuda',0);rows=[]
    with torch.random.fork_rng(devices=[0]),torch.no_grad():
        torch.random.default_generator.manual_seed(20261027);torch.cuda.manual_seed(20261027)
        q,k,v=[torch.randn(SMOKE_SHAPE,device=device,dtype=torch.float32) for _ in range(3)]
        inputs=[array_hash(t.cpu().numpy()) for t in (q,k,v)];before=rng_fingerprint(torch,device)
        first=None
        for index in range(3):
            result=functional.scaled_dot_product_attention(q,k,v,dropout_p=0.0,is_causal=False)
            torch.cuda.synchronize()
            if tuple(result.shape)!=SMOKE_SHAPE or result.dtype!=torch.float32 or not torch.isfinite(result).all().item():
                raise ValueError('Math SDPA smoke produced invalid output')
            digest=array_hash(result.cpu().numpy());first=digest if first is None else first
            rows.append({'repeat':index,'output_sha256':digest,'exact_to_first':digest==first})
        if not all(row['exact_to_first'] for row in rows):raise ValueError('Repeated math SDPA smoke differs')
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as profile:
            result=functional.scaled_dot_product_attention(q,k,v,dropout_p=0.0,is_causal=False);torch.cuda.synchronize()
        trace=output/'math-sdpa-trace.json';profile.export_chrome_trace(str(trace))
        if array_hash(result.cpu().numpy())!=first:raise ValueError('Profiled math SDPA output differs')
        if [array_hash(t.cpu().numpy()) for t in (q,k,v)]!=inputs or rng_fingerprint(torch,device)!=before:
            raise ValueError('Smoke changed input tensors or RNG state')
    return {'passed':True,'shape':list(SMOKE_SHAPE),'dtype':'torch.float32','dropout_p':0.0,'is_causal':False,
            'geometry_scope':'One spatial-attention batch; not full14-frame batch or full-model memory proof.',
            'input_sha256':inputs,'repetitions':rows,'profiled_repeat_exact':True,'rng_unchanged':True,
            'trace_sha256':base.sha(trace),'profile':trace_evidence(trace)}


def compound_identity(platform,base):return {'platform_identity':platform,'algorithm_policy':declaration(base)}


def smoke_valid(smoke):
    rows=smoke.get('repetitions',[]);inputs=smoke.get('input_sha256',[])
    return (smoke.get('passed') is True and smoke.get('shape')==list(SMOKE_SHAPE)
        and smoke.get('dtype')=='torch.float32' and smoke.get('dropout_p')==0.0 and smoke.get('is_causal') is False
        and smoke.get('profiled_repeat_exact') is True and smoke.get('rng_unchanged') is True
        and len(inputs)==3 and all(isinstance(x,str) and re.fullmatch(r'[0-9a-f]{64}',x) for x in inputs)
        and len(rows)==3 and [r.get('repeat') for r in rows]==[0,1,2]
        and all(r.get('exact_to_first') is True and isinstance(r.get('output_sha256'),str)
                and re.fullmatch(r'[0-9a-f]{64}',r['output_sha256']) for r in rows)
        and len({r['output_sha256'] for r in rows})==1)


def policy_gate(config,admitted,ab,encoder,base):
    import torch
    proof_path=Path(config['algorithm_policy']['verification']);lock_path=Path(config['algorithm_policy']['lock'])
    proof=json.loads(proof_path.read_text());lock=json.loads(lock_path.read_text())
    identity=compound_identity(admitted['runtime_identity'],base)
    if (proof.get('verification_type')!='explicit_a100_math_sdpa_runtime_v1' or proof.get('passed') is not True
            or proof.get('status')!='completed' or proof.get('inputs_and_source_unchanged') is not True
            or proof.get('platform_unchanged_after_restoration') is not True
            or proof.get('platform_gate')!=admitted or proof.get('runtime_identity')!=identity
            or proof.get('producer_sha256')!=base.sha(__file__) or proof.get('source_identity')!=base.SOURCE
            or proof.get('allocation_deadline_epoch')!=base.DEADLINE or proof.get('headline_eligible') is not False
            or proof.get('policy_execution',{}).get('flags_restored') is not True
            or proof.get('policy_execution',{}).get('active_flags_unchanged') is not True
            or proof.get('policy_execution',{}).get('callable_restored') is not True
            or proof.get('policy_execution',{}).get('sdpa_call_count')!=4
            or proof.get('policy_execution',{}).get('flags_before')!=ab.snapshot(torch,base)
            or not smoke_valid(proof.get('smoke',{}))):
        raise ValueError('Fresh explicit math-SDPA proof/lifecycle required')
    policy=proof['policy_execution'];expected_active=copy.deepcopy(policy['flags_before'])
    expected_active['sdp']={'flash_sdp_enabled':False,'mem_efficient_sdp_enabled':False,'math_sdp_enabled':True}
    if policy.get('flags_active')!=expected_active or policy.get('flags_after')!=policy['flags_before']:
        raise ValueError('Recorded policy flags do not match the declared one-factor runtime')
    expected_geometry=json.dumps([{'shape':list(SMOKE_SHAPE),'dtype':'torch.float32','device':'cuda'}]*3,sort_keys=True)
    if policy.get('sdpa_calls')!={expected_geometry:4}:raise ValueError('Fresh smoke actual SDPA geometry differs')
    expected={'schema_version':1,'runtime_identity':identity,'policy_verification_sha256':base.sha(proof_path)}
    if lock!=expected:raise ValueError('Explicit compound policy lock differs')
    current_ab=ab_evidence(ab,base)
    required=input_bindings(CONFIG_PATH,config,ab,encoder,base)
    required.update(current_ab.get('artifact_sha256',{}))
    if proof.get('input_sha256')!=required:raise ValueError('Policy proof input inventory is incomplete or changed')
    recheck(proof['input_sha256'],base)
    if proof.get('development_rationale')!=current_ab:raise ValueError('Development A/B evidence changed')
    artifacts=proof.get('artifact_sha256',{})
    if 'math-sdpa-trace.json' not in artifacts:raise ValueError('Fresh policy trace missing')
    for name,digest in artifacts.items():
        if Path(name).name!=name:raise ValueError('Unsafe policy artifact path')
        recheck({str(proof_path.parent/name):digest},base)
    trace=proof_path.parent/'math-sdpa-trace.json'
    if base.sha(trace)!=proof['smoke'].get('trace_sha256') or trace_evidence(trace)!=proof['smoke']['profile']:
        raise ValueError('Policy profile evidence differs')
    return {'passed':True,'runtime_identity':identity,'policy_verification_sha256':base.sha(proof_path),'policy_lock_sha256':base.sha(lock_path)}


def validate_same_process(config_path,output,torch,base,progress):
    import drivebench.validate as validation
    original=validation.reference_identity
    def extended(environment):
        assert_math(torch)
        return {**original(environment),'algorithm_policy':declaration(base)}
    validation.reference_identity=extended
    try:
        code=validation.main(['--config',str(config_path),'--mode','all','--candidates','2','--request-index','0',
                              '--atol','0','--rtol','0','--cache-bytes','2147483648','--output',str(output)])
        return code
    finally:
        validation.reference_identity=original
        progress['validator_identity_function_restored']=validation.reference_identity is original


def strict_result(path,config,admitted,base):
    report=json.loads(path.read_text());expected_runtime={**admitted['runtime_identity'],'algorithm_policy':declaration(base)}
    expected_checks={'upstream_agreement','candidate_order','unrelated_candidate_action_change',
                     'conditioning_cache_agreement','conditioning_cache_keys','conditioning_cache_exercised'}
    if (report.get('schema_version')!=2 or report.get('mode')!='all' or report.get('atol')!=0 or report.get('rtol')!=0
            or report.get('config')!=config or report.get('implementation_sha256')!=base.SOURCE
            or report.get('runtime_identity')!=expected_runtime or set(report.get('checks',{}))!=expected_checks
            or len(report.get('seeds',[]))!=2 or report.get('quality_validated') is not False):
        raise ValueError('Raw strict report contract differs')
    bindings=file_map([path],base)
    for label in ('upstream','refactored','isolation','uncached','cached'):
        run=report.get('runs',{}).get(label,{})
        kinds={'normal','reverse','other_candidates_after_action_change'} if label=='isolation' else {'normal'}
        if set(run.get('outputs',{}))!=kinds:raise ValueError('Missing original strict output artifacts')
        for kind in kinds:bindings[run['outputs'][kind]]=run.get('outputs_sha256',{}).get(kind)
    recheck(bindings,base)
    return {'raw_report_passed':report.get('passed') is True,'all_six_checks_passed':all(c.get('passed') is True for c in report['checks'].values()),
            'artifact_sha256':bindings,'output_files':7,'scope':'Selected development request under explicitly declared process-wide math SDPA; not original default-runtime agreement.'}


def run(command,config_path,output):
    ab,encoder,base,verifier=dependencies();config=config_checked(config_path,base,verifier)
    seconds=840 if command=='verify' else 3540
    verifier.deadline_check(base.DEADLINE,seconds+60)
    output=Path(output);expected=PROOF_DIR if command=='verify' else VALIDATE_DIR
    if output.resolve()!=expected.resolve():raise ValueError('Use the exact fresh output directory')
    output.mkdir(parents=True,exist_ok=False)
    bindings=input_bindings(config_path,config,ab,encoder,base)
    rationale=ab_evidence(ab,base);bindings.update(rationale['artifact_sha256'])
    if command=='validate':bindings.update(file_map([config['algorithm_policy']['verification'],config['algorithm_policy']['lock']],base))
    value={'type':'a100_math_sdpa_'+command+'_attempt_v1','started_at':base.stamp(),'status':'running',
           'source_identity':base.SOURCE,'producer_sha256':base.sha(__file__),'input_sha256':bindings,
           'allocation_deadline_epoch':base.DEADLINE,'algorithm_policy':declaration(base),'headline_eligible':False,
           'quality_validated':False,'original_default_runtime_strict_pass_claimed':False,
           'scope':'New declared A100 runtime; prior platform proof supplies only separately revalidated hardware evidence.'}
    base.save(output/'attempt.json',value)
    def timeout(_signum,_frame):raise TimeoutError('Bounded math-SDPA command expired')
    old=signal.signal(signal.SIGALRM,timeout);signal.alarm(seconds);code=1;execution={};admitted=None;proof_ready=False
    try:
        admitted=verifier.gate(config,base.DEADLINE);value['platform_gate']=admitted
        value['development_rationale']=rationale
        if command=='validate':
            value['policy_gate']=policy_gate(config,admitted,ab,encoder,base)
        import torch
        from drivebench.benchmarks.environment import collect_environment,reference_identity
        with activate(torch,ab,base,execution):
            if command=='verify':value['smoke']=smoke(torch,output,base)
            else:
                child_code=validate_same_process(config_path,output/'strict-all.json',torch,base,value)
                value['raw_validator_exit_code']=child_code
                value['strict_result']=strict_result(output/'strict-all.json',config,admitted,base)
                if child_code!=0 or not value['strict_result']['raw_report_passed'] or not value['strict_result']['all_six_checks_passed']:
                    raise ValueError('New runtime strict numerical checks failed; original and new failures preserved')
            if not execution['sdpa_call_count']:raise ValueError('Policy never observed a real SDPA call')
        if not execution.get('active_flags_unchanged') or not execution.get('flags_restored') or not execution.get('callable_restored'):
            raise ValueError('Policy lifetime/restoration evidence incomplete')
        if reference_identity(collect_environment())!=admitted['runtime_identity']:raise ValueError('Original platform changed after policy restoration')
        recheck(bindings,base);value.update(inputs_and_source_unchanged=True,platform_unchanged_after_restoration=True,status='completed');code=0
        proof_ready=command=='verify'
    except BaseException as error:value.update(status='failed',error={'type':type(error).__name__,'message':str(error)})
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old)
        value['policy_execution']=execution;value['finished_at']=base.stamp()
        value['artifact_sha256']={str(p.relative_to(output)):base.sha(p) for p in output.rglob('*') if p.is_file()}
        if command=='verify':
            value.update(verification_type='explicit_a100_math_sdpa_runtime_v1',passed=proof_ready,
                         runtime_identity=compound_identity(admitted['runtime_identity'],base) if admitted else None)
            proof_path=output/'policy-verification.json';base.save(proof_path,value)
            if proof_ready:base.save(output/'policy-lock.json',{'schema_version':1,'runtime_identity':value['runtime_identity'],
                                                               'policy_verification_sha256':base.sha(proof_path)})
        else:base.save(output/'completed.json',value)
    print(json.dumps({'status':value['status'],'command':command,'output':str(output),'headline_eligible':False}));return code


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['verify','validate'])
    parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();raise SystemExit(run(args.command,args.config,args.output))
