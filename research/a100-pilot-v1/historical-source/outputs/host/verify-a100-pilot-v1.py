#!/usr/bin/env python3
"""Explicit A100-only admission and bounded diagnostic launch; frozen core unchanged."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from drivebench.benchmarks.environment import collect_environment, command, reference_identity, source_identity
from drivebench.config import file_sha256, load_config, validate_run_config
from drivebench.verify_environment import CUDA_SMOKE, has_cuda_kernel_trace, has_hardware_counters

SOURCE = "d8a32813492c0939a0a91573a945c914938972419946065e6822f6209fa470cb"
VISTA = "cc9821b4253ca7987c32757613d2fc2448fa9f5d"
CHECKPOINT = "1b267bb753aa5c5e4d363f6e4c864dd4e343996dedc18b5ab20af6d20be34b98"
XFORMERS_COMMIT = "1dc3d7a3077ff896cd42d573db9fafeefc97b815"
TARGET = "single_nvidia_a100_sxm4_80gb_pilot_v1"
SCENE_REQUEST = "nuscenes-fcbccedd61424f1b85dcbf8f897f9754-054c2431ba324912bb4799b9217f5b7b"
CUDA_TEST = r'''
import json, torch, xformers, xformers.ops
assert torch.cuda.device_count() == 1
assert torch.cuda.get_device_capability(0) == (8, 0)
x = torch.ones(32, device='cuda'); x.mul_(3); torch.cuda.synchronize()
assert x.sum().item() == 96.0
cases=[]
with torch.no_grad():
    for dimension in (40, 64):
        for length in (77, 256):
            for flat in (False, True):
                qshape = (8,256,dimension) if flat else (1,256,8,dimension)
                kshape = (8,length,dimension) if flat else (1,length,8,dimension)
                q=torch.randn(qshape,device='cuda',dtype=torch.float16)
                k=torch.randn(kshape,device='cuda',dtype=torch.float16)
                v=torch.randn(kshape,device='cuda',dtype=torch.float16)
                out=xformers.ops.memory_efficient_attention(q,k,v)
                torch.cuda.synchronize()
                assert out.shape == q.shape and torch.isfinite(out).all().item()
                cases.append({'dimension':dimension,'key_length':length,'flat':flat,'passed':True})
print(json.dumps({'passed':True,'capability':[8,0],'tensor_sum':96.0,
 'attention_cases':cases,'torch_arch_list':torch.cuda.get_arch_list()}))
'''


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    """Fresh durable artifact; never replace earlier evidence."""
    data=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'wb') as stream:
        stream.write(data);stream.flush();os.fsync(stream.fileno())
    fd=os.open(Path(path).parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def checked_source():
    value=source_identity()
    if value != SOURCE:raise ValueError('Frozen drivebench package changed')
    return value


def deadline_check(deadline, required_seconds):
    if isinstance(deadline,bool) or not isinstance(deadline,(int,float)) or not math.isfinite(deadline):
        raise ValueError('Finite immutable allocation deadline required')
    if deadline <= time.time()+required_seconds+120:
        raise ValueError('Insufficient time before allocation cleanup deadline')


def environment_failures(env, dedicated):
    failures=[]
    if env.get('system')!='Linux':failures.append('Linux required')
    if (env.get('host_ram_bytes') or 0)<64_000_000_000:failures.append('At least 64 GB host RAM required')
    if len(env.get('gpus',[]))!=1:failures.append('Exactly one full GPU required')
    else:
        gpu=env['gpus'][0];name=gpu.get('name','').upper().replace(' ','')
        try:memory=float(gpu['memory_total_mib'])
        except (KeyError,TypeError,ValueError):memory=0
        if 'A100-SXM4-80GB' not in name or not 79000<=memory<=83000:
            failures.append('This separate pilot requires A100 SXM4 80GB, not H100/PCIe/40GB')
        if gpu.get('mig_mode')!='Disabled':failures.append('MIG must be disabled')
    if dedicated is not True:failures.append('Dedicated allocation attestation required')
    processes=env.get('gpu_processes',{})
    if processes.get('returncode')!=0 or processes.get('stdout','').strip():
        failures.append('GPU must be idle; compute-process inspection must succeed')
    torch=env.get('torch') or {}
    if (torch.get('version'),torch.get('cuda'),torch.get('cuda_available'))!=('2.0.1+cu118','11.8',True):
        failures.append('Pinned Torch2.0.1+cu118 with available CUDA required')
    packages={key.lower():value for key,value in env.get('packages',{}).items()}
    if packages.get('triton')!='2.0.0' or packages.get('xformers')!='0.0.20':
        failures.append('Pinned Triton2.0.0/xFormers0.0.20 required')
    if not str(env.get('python','')).startswith('3.10.'):
        failures.append('Pinned Python3.10 required')
    build=env.get('xformers_build') or {}
    if not build.get('cpp_lib_sha256') or not build.get('installed_wheel_sha256'):
        failures.append('Installed xFormers build and wheel SHA256 required')
    tools=env.get('tools',{})
    for name in ('nvcc','nsys','ncu'):
        if tools.get(name,{}).get('returncode')!=0:failures.append(name+' unavailable')
    if 'release 11.8,' not in tools.get('nvcc',{}).get('stdout',''):
        failures.append('CUDA11.8 compiler required')
    return failures


def cuda_evidence(result):
    if result.get('returncode')!=0:return False
    try:value=json.loads(result['stdout'].splitlines()[-1])
    except (KeyError,ValueError,IndexError):return False
    cases=value.get('attention_cases',[])
    expected={(d,k,f) for d in (40,64) for k in (77,256) for f in (False,True)}
    actual={(x.get('dimension'),x.get('key_length'),x.get('flat')) for x in cases if x.get('passed') is True}
    return (value.get('passed') is True and value.get('capability')==[8,0]
            and value.get('tensor_sum')==96.0 and len(cases)==8 and actual==expected)


def bootstrap_identity(env, *, root=ROOT, metadata_path=None, prefix=None):
    directory=root/'outputs/environments/a100-cu118-sm80-pilot-01'
    venv=root/'outputs/venvs/vista-a100-cu118'
    if Path(sys.prefix if prefix is None else prefix).resolve()!=venv.resolve():
        raise ValueError('Use the separately built A100 virtual environment')
    receipt_path=directory/'build-receipt.json';input_path=directory/'input.json'
    receipt=json.loads(receipt_path.read_text());inputs=json.loads(input_path.read_text())
    helper=root/'outputs/host/bootstrap-a100-sm80-v1.py'
    if metadata_path is None:
        metadata_path=Path(importlib.metadata.distribution('xformers').locate_file('xformers/cpp_lib.json'))
    metadata_path=Path(metadata_path)
    if not metadata_path.resolve().is_relative_to(venv.resolve()):
        raise ValueError('xFormers metadata resolves outside new A100 environment')
    metadata=json.loads(metadata_path.read_text());wheels=list((directory/'wheels').glob('xformers-0.0.20-*.whl'))
    if len(wheels)!=1:raise ValueError('Exactly one new native-sm80 wheel required')
    wheel_sha=file_sha256(wheels[0]);build=env.get('xformers_build') or {}
    if (receipt.get('passed') is not True or receipt.get('source_environment_unchanged') is not True
            or receipt.get('reference_h100_verification_passed') is not False
            or receipt.get('input_sha256')!=file_sha256(input_path)
            or receipt.get('helper_sha256')!=file_sha256(helper) or inputs.get('helper_sha256')!=file_sha256(helper)
            or inputs.get('xformers_commit')!=XFORMERS_COMMIT or inputs.get('torch_cuda_arch_list')!='8.0'
            or receipt.get('wheel_sha256')!=wheel_sha or build.get('installed_wheel_sha256')!=wheel_sha
            or build.get('cpp_lib_sha256')!=file_sha256(metadata_path)
            or receipt.get('installed',{}).get('source_sha256')!=SOURCE
            or receipt.get('installed',{}).get('build')!=metadata
            or metadata.get('env',{}).get('TORCH_CUDA_ARCH_LIST')!='8.0'
            or metadata.get('version',{}).get('cuda')!=1108
            or metadata.get('version',{}).get('torch')!='2.0.1+cu118'):
        raise ValueError('Native-sm80 source/build/wheel provenance differs')
    paths=[receipt_path,input_path,helper,wheels[0],metadata_path]
    return {'xformers_commit':XFORMERS_COMMIT,'torch_cuda_arch_list':'8.0',
            'artifact_sha256':{str(p):file_sha256(p) for p in paths}}


def verify(output, *, dedicated, deadline):
    deadline_check(deadline,900);checked_source()
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    env=collect_environment();failures=environment_failures(env,dedicated)
    native=None
    try:native=bootstrap_identity(env)
    except (OSError,ValueError,KeyError,ImportError) as error:failures.append('Native bootstrap admission failed: '+str(error))
    proof={'torch_cuda_execution_passed':False,'attention_smoke_passed':False,
           'hardware_counters_collected':False,'nsys_trace_collected':False}
    if not failures:
        proof['torch_cuda']=command([sys.executable,'-c',CUDA_TEST],timeout=180)
        proof['torch_cuda_execution_passed']=cuda_evidence(proof['torch_cuda'])
        proof['attention_smoke_passed']=proof['torch_cuda_execution_passed']
        if proof['torch_cuda_execution_passed']:
            source=output/'cuda_smoke.cu';binary=output/'cuda_smoke';source.write_text(CUDA_SMOKE)
            proof['compile']=command(['nvcc','-O2','-arch=sm_80',str(source.resolve()),'-o',str(binary.resolve())],timeout=120)
            if proof['compile']['returncode']==0:
                proof['nsys']=command(['nsys','profile','--trace=cuda,nvtx,osrt','--sample=none','--force-overwrite=true',
                    '-o',str((output/'smoke-systems').resolve()),str(binary.resolve())],timeout=120)
                proof['ncu']=command(['ncu','--metrics','sm__cycles_elapsed.avg,gpu__time_duration.sum','--csv','--page','raw',
                    '--force-overwrite','-o',str((output/'smoke-compute').resolve()),str(binary.resolve())],timeout=120)
                proof['hardware_counters_collected']=(proof['ncu']['returncode']==0 and has_hardware_counters(proof['ncu']['stdout']))
                if proof['nsys']['returncode']==0 and (output/'smoke-systems.nsys-rep').is_file():
                    proof['nsys_stats']=command(['nsys','stats','--report','cuda_gpu_kern_sum','--format','csv',str((output/'smoke-systems.nsys-rep').resolve())],timeout=120)
                    proof['nsys_trace_collected']=(proof['nsys_stats']['returncode']==0 and has_cuda_kernel_trace(proof['nsys_stats']['stdout'],kernel='drivebench_smoke'))
    for key in ('torch_cuda_execution_passed','attention_smoke_passed','hardware_counters_collected','nsys_trace_collected'):
        if not proof[key]:failures.append(key+' has not passed')
    after=collect_environment()
    if reference_identity(after)!=reference_identity(env):failures.append('Runtime changed during verification')
    failures.extend(environment_failures(after,dedicated));checked_source()
    try:
        if bootstrap_identity(after)!=native:failures.append('Native bootstrap changed during verification')
    except (OSError,ValueError,KeyError,ImportError) as error:failures.append('Native bootstrap final check failed: '+str(error))
    artifacts={p.name:file_sha256(p) for p in output.iterdir() if p.is_file()}
    for name in ('smoke-systems.nsys-rep','smoke-compute.ncu-rep'):
        if name not in artifacts or (output/name).stat().st_size==0:failures.append('Missing nonempty profiler artifact: '+name)
    result={'schema_version':1,'verification_type':'explicit_a100_pilot_environment_v1','target_contract':TARGET,
        'created_at':stamp(),'allocation_deadline_epoch':deadline,'passed':not failures,'failures':sorted(set(failures)),
        'dedicated_attestation':dedicated,'environment':env,'profiling_proof':proof,'artifact_sha256':artifacts,
        'producer':{'helper_sha256':file_sha256(__file__),'source_identity':SOURCE},
        'native_xformers_provenance':native,
        'scope':'Separate A100 pilot. No H100 environment proof or study acceptance is inherited.',
        'headline_eligible':False,'h100_equivalence_claim':False}
    save(output/'verification.json',result)
    if result['passed']:save(output/'reference-lock.json',reference_identity(env))
    return result


def gate(config, deadline, *, current=None):
    checked_source();proof_path=Path(config['environment_verification']);lock_path=Path(config['environment_lock'])
    proof=json.loads(proof_path.read_text());lock=json.loads(lock_path.read_text())
    if (proof.get('verification_type')!='explicit_a100_pilot_environment_v1' or proof.get('target_contract')!=TARGET
            or proof.get('passed') is not True or proof.get('failures') or proof.get('dedicated_attestation') is not True
            or proof.get('allocation_deadline_epoch')!=deadline or proof.get('h100_equivalence_claim') is not False
            or proof.get('producer')!={'helper_sha256':file_sha256(__file__),'source_identity':SOURCE}):
        raise ValueError('Fresh explicit A100 proof/producer/allocation binding required')
    current=collect_environment() if current is None else current
    failures=environment_failures(current,True)
    if failures:raise ValueError('; '.join(failures))
    if reference_identity(current)!=reference_identity(proof['environment']) or lock!=reference_identity(current):
        raise ValueError('A100 runtime differs from fresh proof or lock')
    if proof.get('native_xformers_provenance')!=bootstrap_identity(current):
        raise ValueError('Native xFormers bootstrap evidence changed')
    p=proof.get('profiling_proof',{})
    if (not cuda_evidence(p.get('torch_cuda',{})) or p.get('torch_cuda_execution_passed') is not True
            or p.get('attention_smoke_passed') is not True or p.get('hardware_counters_collected') is not True
            or p.get('nsys_trace_collected') is not True or p.get('ncu',{}).get('returncode')!=0
            or not has_hardware_counters(p.get('ncu',{}).get('stdout',''))
            or p.get('nsys_stats',{}).get('returncode')!=0
            or not has_cuda_kernel_trace(p.get('nsys_stats',{}).get('stdout',''),kernel='drivebench_smoke')):
        raise ValueError('Actual CUDA/attention/NSYS/SM-counter evidence required')
    artifacts=proof.get('artifact_sha256',{})
    if not {'smoke-systems.nsys-rep','smoke-compute.ncu-rep'}.issubset(artifacts):
        raise ValueError('Profiler artifact bindings missing')
    for name,digest in artifacts.items():
        path=proof_path.parent/name
        if Path(name).name!=name or path.is_symlink() or not path.is_file() or file_sha256(path)!=digest:
            raise ValueError('Changed profiler artifact')
    # Exercise the unchanged core gate too; it is runtime-bound, despite its H100 error text.
    from drivebench.benchmark import _validate_reference_environment
    core_gate=_validate_reference_environment(config,current)
    return {'passed':True,'target_contract':TARGET,'verification_sha256':file_sha256(proof_path),
            'lock_sha256':file_sha256(lock_path),'runtime_identity':reference_identity(current),'core_gate':core_gate}


def check_config(config, mode, candidates):
    from drivebench.data.manifest import load_manifest
    from drivebench.types import GenerationConfig
    generation,options=validate_run_config(config)
    if generation!=GenerationConfig() or config['model'].get('backend')!='vista':
        raise ValueError('Pilot requires original generation and Vista adapter')
    if config['model'].get('upstream_commit')!=VISTA or config['model'].get('checkpoint_sha256')!=CHECKPOINT:
        raise ValueError('Pinned Vista source and checkpoint required')
    if (options.compile_denoiser or options.compile_decoder or options.feature_cache_interval!=1
            or options.feature_cache_module is not None or options.max_batch_size!=1
            or options.attention_backend!='upstream' or any(config['model'].get(k) for k in ('compiler_runtime','compiler_feature_boundary'))):
        raise ValueError('First A100 pilot is serial eager, without compiler/feature/attention changes')
    if options.conditioning_cache_bytes not in (0,2147483648):raise ValueError('Pilot cache budget must be zero or 2GiB')
    if config.get('benchmark',{}).get('headline') is not False or config['data'].get('verify_checksums') is not True:
        raise ValueError('Pilot must remain non-headline with verified input checksums')
    records=load_manifest(config['data']['manifest'])
    if len(records)!=1 or records[0].get('request_id')!=SCENE_REQUEST:
        raise ValueError('Pilot is limited to the preselected scene-0103 development request')
    bench=config['benchmark']
    if mode=='benchmark':
        if (bench.get('workload')!='single_future' or bench.get('candidate_count')!=1
                or not 1<=bench.get('requests',0)<=3 or not 1<=bench.get('warmup_requests',0)<=3
                or not bench['warmup_requests']<=bench.get('max_warmup_requests',0)<=3):
            raise ValueError('Measured pilot limited to 1..3 single-future requests and 1..3 warmups')
    elif candidates not in (1,2) or (mode in ('all','isolation') and candidates!=2):
        raise ValueError('Selected strict diagnostic requires one or two candidates')


def checked_result(output, mode, config, admitted):
    if mode=='benchmark':
        names=('config.json','environment.json','measurements.jsonl','summary.json')
        if any(not (output/name).is_file() for name in names):raise ValueError('Incomplete benchmark artifacts')
        summary=json.loads((output/'summary.json').read_text())
        rows=[json.loads(line) for line in (output/'measurements.jsonl').read_text().splitlines() if line]
        if (summary.get('failures')!=0 or summary.get('headline_eligible') is not False
                or len(rows)!=config['benchmark']['requests'] or any(r.get('status')!='ok' for r in rows)
                or reference_identity(json.loads((output/'environment.json').read_text()))!=admitted['runtime_identity']):
            raise ValueError('Measured pilot result incomplete, failed, or mismatched')
        return {name:file_sha256(output/name) for name in names}
    report=json.loads(output.read_text())
    if (report.get('passed') is not True or report.get('mode')!=mode or report.get('atol')!=0
            or report.get('rtol')!=0 or report.get('config')!=config
            or report.get('implementation_sha256')!=SOURCE or report.get('runtime_identity')!=admitted['runtime_identity']
            or report.get('request_id')!=SCENE_REQUEST):
        raise ValueError('Strict numerical diagnostic result is not the requested A100 run')
    return {'numerical_report_sha256':file_sha256(output)}


def numerical_prerequisite(config, admitted, deadline):
    """Diagnostic acceptance is distinct from the intentionally false headline gate."""
    path=Path(config.get('numerical_validation') or '')
    checked_result(path,'all',config,admitted)
    report=json.loads(path.read_text());completion_path=path.with_name(path.name+'.a100-completed.json')
    completion=json.loads(completion_path.read_text())
    if (completion.get('status')!='completed' or completion.get('type')!='bounded_a100_pilot_attempt_v1'
            or completion.get('target_contract')!=TARGET or completion.get('source_identity')!=SOURCE
            or completion.get('helper_sha256')!=file_sha256(__file__)
            or completion.get('exit_code')!=0 or completion.get('timed_out') is not False
            or completion.get('inputs_and_source_unchanged') is not True or completion.get('runtime_unchanged') is not True
            or completion.get('allocation_deadline_epoch')!=deadline
            or completion.get('environment_gate')!=admitted
            or completion.get('result_artifact_sha256',{}).get('numerical_report_sha256')!=file_sha256(path)):
        raise ValueError('Same-pilot completed strict-all execution receipt required')
    required={'upstream_agreement','candidate_order','unrelated_candidate_action_change',
              'conditioning_cache_agreement','conditioning_cache_keys','conditioning_cache_exercised'}
    if not required.issubset(report.get('checks',{})) or any(report['checks'][key].get('passed') is not True for key in required):
        raise ValueError('Full strict/cache/isolation checks required before measurement')
    if len(report.get('seeds',[]))!=2:raise ValueError('Two-candidate semantics required')
    bindings={str(path):file_sha256(path),str(completion_path):file_sha256(completion_path)}
    argv=completion.get('argv',[])
    try:tested_config_path=Path(argv[argv.index('--config')+1])
    except (ValueError,IndexError,TypeError):raise ValueError('Strict execution config provenance missing')
    manifest=Path(config['data']['manifest'])
    required_inputs={str(tested_config_path),str(manifest),str(manifest.with_suffix('.jsonl.meta.json')),
                     str(Path(config['environment_verification'])),str(Path(config['environment_lock']))}
    original_inputs=completion.get('input_sha256',{})
    if not required_inputs.issubset(original_inputs) or load_config(tested_config_path)!=config:
        raise ValueError('Strict execution config/input binding incomplete or changed')
    for name,digest in original_inputs.items():
        if file_sha256(name)!=digest:raise ValueError('Strict execution input changed: '+name)
    bindings.update(original_inputs)
    for label in ('upstream','refactored','isolation','uncached','cached'):
        details=report.get('runs',{}).get(label,{})
        needed={'normal','reverse','other_candidates_after_action_change'} if label=='isolation' else {'normal'}
        if not needed.issubset(details.get('outputs',{})):raise ValueError('Missing original numerical output evidence')
        for key in needed:
            output=Path(details['outputs'][key]);expected=details.get('outputs_sha256',{}).get(key)
            if not output.is_file() or output.is_symlink() or file_sha256(output)!=expected:
                raise ValueError('Numerical output hash differs')
            bindings[str(output)]=expected
    return bindings


def run(config_path, mode, output, *, candidates, timeout, deadline):
    if not 60<=timeout<=4500:raise ValueError('Pilot timeout must be 60..4500 seconds')
    deadline_check(deadline,timeout+60)
    config_path=Path(config_path);config=load_config(config_path);check_config(config,mode,candidates)
    admitted=gate(config,deadline);output=Path(output)
    attempt=output.with_name(output.name+'.a100-attempt.json');final=output.with_name(output.name+'.a100-completed.json')
    artifacts=output.parent/(output.stem+'-outputs');log=output.with_name(output.name+'.log')
    if any(p.exists() or p.is_symlink() for p in (output,attempt,final,log,artifacts)):
        raise FileExistsError('Fresh pilot output/attempt required; preserve prior failures')
    output.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(output.parent).free<16*1024**3:
        raise ValueError('At least 16 GiB free storage required before pilot')
    bindings={str(config_path):file_sha256(config_path),str(Path(config['data']['manifest'])):file_sha256(config['data']['manifest']),
              str(Path(config['environment_verification'])):file_sha256(config['environment_verification']),
              str(Path(config['environment_lock'])):file_sha256(config['environment_lock'])}
    manifest_meta=Path(config['data']['manifest']).with_suffix('.jsonl.meta.json')
    if not manifest_meta.is_file():raise ValueError('Verified manifest metadata required')
    bindings[str(manifest_meta)]=file_sha256(manifest_meta)
    if mode=='benchmark':bindings.update(numerical_prerequisite(config,admitted,deadline))
    argv=[sys.executable,'-m','drivebench.benchmark','--config',str(config_path),'--output',str(output)] if mode=='benchmark' else [
        sys.executable,'-m','drivebench.validate','--config',str(config_path),'--mode',mode,'--candidates',str(candidates),
        '--request-index','0','--atol','0','--rtol','0','--cache-bytes','2147483648','--output',str(output)]
    value={'schema_version':1,'type':'bounded_a100_pilot_attempt_v1','target_contract':TARGET,'started_at':stamp(),
           'allocation_deadline_epoch':deadline,'timeout_seconds':timeout,'argv':argv,'input_sha256':bindings,
           'helper_sha256':file_sha256(__file__),'source_identity':SOURCE,'environment_gate':admitted,
           'scope':'One preselected development clip; separate A100 diagnostics; no headline or H100 adoption.'}
    save(attempt,value);timed_out=False;started=time.monotonic()
    with log.open('xb') as stream:
        child=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(child.pid,signal.SIGTERM)
            try:code=child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=30)
        stream.flush();os.fsync(stream.fileno())
    post_error=None;unchanged=False;runtime_unchanged=False
    try:
        unchanged=all(file_sha256(path)==digest for path,digest in bindings.items()) and source_identity()==SOURCE
        runtime_unchanged=reference_identity(collect_environment())==admitted['runtime_identity']
    except (OSError,ValueError,KeyError,TypeError) as error:post_error=type(error).__name__+': '+str(error)
    result_artifacts={};result_error=None
    if code==0 and not timed_out:
        try:result_artifacts=checked_result(output,mode,config,admitted)
        except (OSError,ValueError,KeyError,TypeError) as error:result_error=str(error)
    value.update(finished_at=stamp(),exit_code=code,timed_out=timed_out,elapsed_seconds=time.monotonic()-started,
                 inputs_and_source_unchanged=unchanged,runtime_unchanged=runtime_unchanged,log_sha256=file_sha256(log),
                 result_artifact_sha256=result_artifacts,result_error=result_error,
                 post_execution_error=post_error,
                 status='completed' if code==0 and not timed_out and unchanged and runtime_unchanged and result_error is None else 'failed')
    save(final,value)
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    v=commands.add_parser('verify');v.add_argument('--output',required=True);v.add_argument('--dedicated',action='store_true')
    v.add_argument('--allocation-deadline-epoch',type=float,required=True)
    r=commands.add_parser('run');r.add_argument('--config',required=True);r.add_argument('--mode',choices=['agreement','all','cache','isolation','benchmark'],required=True)
    r.add_argument('--output',required=True);r.add_argument('--candidates',type=int,default=2);r.add_argument('--timeout-seconds',type=int,required=True)
    r.add_argument('--allocation-deadline-epoch',type=float,required=True)
    args=parser.parse_args()
    if args.command=='verify':
        result=verify(args.output,dedicated=args.dedicated,deadline=args.allocation_deadline_epoch)
        print(json.dumps({'passed':result['passed'],'failures':result['failures'],'output':args.output}));return 0 if result['passed'] else 1
    result=run(args.config,args.mode,args.output,candidates=args.candidates,timeout=args.timeout_seconds,deadline=args.allocation_deadline_epoch)
    print(json.dumps({'status':result['status'],'exit_code':result['exit_code'],'output':args.output}));return 0 if result['status']=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
