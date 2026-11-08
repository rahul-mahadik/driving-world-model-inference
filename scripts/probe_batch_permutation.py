#!/usr/bin/env python3
"""Capture one actual first-step B2 denoiser input and permute it exactly.

Three instrumented UNet calls (normal, reversed, normal repeat) distinguish an
observed position effect from repeat nondeterminism. No 50-step rollout or video
decode occurs. Block outputs are temporary CPU arrays; only comparisons/shapes
are saved. This is a cause-localization diagnostic, never a quality proof.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import signal
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from drivebench.config import file_sha256
from scripts.probe_candidate_memory import config_files, inspect_config, save


def cfg_permutation(frames=25):
    """Reverse B2 candidates independently inside unconditional/conditional halves."""
    return np.r_[np.arange(frames,2*frames),np.arange(frames),
                 np.arange(3*frames,4*frames),np.arange(2*frames,3*frames)].astype(np.int64)


def compare_permuted(reference, candidate, permutation):
    if reference.shape != candidate.shape or len(reference) != len(permutation):
        raise ValueError('Block output layout changed')
    total=outside=0; square=maximum=0.0
    for index, source in enumerate(permutation):
        left=np.asarray(reference[index],dtype=np.float64)
        right=np.asarray(candidate[source],dtype=np.float64)
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError('Nonfinite diagnostic block output')
        difference=np.abs(left-right)
        total+=difference.size
        outside+=int(np.count_nonzero(difference))
        maximum=max(maximum,float(difference.max(initial=0)))
        square+=float(np.square(difference).sum())
    return {'passed_exact':outside==0,'max_absolute_error':maximum,
            'rmse':float(np.sqrt(square/total)) if total else 0.0,'different_values':outside,'values':total}


@contextlib.contextmanager
def block_capture(backend, *, memory_limit=16*1024**3):
    """Keep normal outputs on CPU; restore all module/native wrappers on failure."""
    import xformers.ops as ops
    from drivebench.runtime.compiler import NATIVE_SOURCE_SHA256
    native=ops.memory_efficient_attention
    if (getattr(native,'_torchdynamo_orig_callable',None) is not None
            or file_sha256(inspect.getsourcefile(native)) != NATIVE_SOURCE_SHA256):
        raise ValueError('Require unchanged pinned native xFormers attention')
    network=backend.model.model.diffusion_model
    blocks=[(f'input_blocks.{i}',module) for i,module in enumerate(network.input_blocks)]
    blocks+=[('middle_block',network.middle_block)]
    blocks += [(f'output_blocks.{i}',module) for i,module in enumerate(network.output_blocks)]
    state={'pass':'normal','block':None,'outputs':{},'bytes':0,'rows':{},'attention':{}}
    originals=[]
    def attention(q,k,v,*args,**kwargs):
        state['attention'].setdefault(state['pass'],[]).append({
            'block':state['block'],'q_shape':list(q.shape),'k_shape':list(k.shape),'v_shape':list(v.shape),
            'q_stride':list(q.stride()),'dtype':str(q.dtype),
            'operator_argument':str(kwargs.get('op')),'attn_bias_type':type(kwargs.get('attn_bias')).__name__})
        return native(q,k,v,*args,**kwargs)
    try:
        ops.memory_efficient_attention=attention
        for name,module in blocks:
            original=module.forward;originals.append((module,original))
            def forward(*args,_name=name,_original=original,**kwargs):
                previous=state['block'];state['block']=_name
                try:
                    out=_original(*args,**kwargs)
                    if out.ndim != 4 or out.shape[0] != 100:
                        raise ValueError('Expected full-block ordered B2 CFG tensor [100,C,H,W]')
                    size=out.numel()*out.element_size()
                    if state['pass']=='normal' and state['bytes']+size>memory_limit:
                        raise MemoryError('Declared temporary CPU snapshot budget exceeded')
                    array=out.detach().contiguous().cpu().numpy()
                    row={'block':_name,'shape':list(array.shape),'dtype':str(array.dtype),
                         'sha256':hashlib.sha256(memoryview(array).cast('B')).hexdigest()}
                    if state['pass']=='normal':
                        if _name in state['outputs']:raise ValueError('Block executed more than once in a denoiser call')
                        state['outputs'][_name]=array;state['bytes']+=size
                    else:
                        permutation=cfg_permutation() if state['pass']=='reversed' else np.arange(100)
                        row['comparison']=compare_permuted(state['outputs'][_name],array,permutation)
                    state['rows'].setdefault(state['pass'],[]).append(row)
                    return out
                finally:
                    state['block']=previous
            module.forward=forward
        yield state
    finally:
        for module,original in originals:module.forward=original
        ops.memory_efficient_attention=native


def run_diagnostic(backend,payload,request,generation):
    import torch
    from drivebench.models.vista import _EXECUTION_LOCK
    captured={}
    original=backend._denoiser
    class CapturedFirstCall(Exception):pass
    def capture(x,sigma,condition,mask):
        if captured:raise RuntimeError('Captured more than one denoiser input')
        captured.update(x=x,sigma=sigma,condition=condition,mask=mask)
        raise CapturedFirstCall()
    backend._denoiser=capture
    try:
        try:
            backend.generate_batch(payload,request['trajectories'],request['seeds'],generation)
        except CapturedFirstCall:
            pass
    finally:
        backend._denoiser=original
    if not captured:raise RuntimeError('First denoiser call was not reached')
    if captured['x'].shape[0]!=100 or set(captured['condition'])!={'vector','crossattn','concat'}:
        raise ValueError('Captured inputs differ from the audited B2 CFG layout')
    index=torch.as_tensor(cfg_permutation(),device=backend.device)
    reverse=lambda value:value.index_select(0,index)
    reversed_inputs={'x':reverse(captured['x']),'sigma':reverse(captured['sigma']),
                     'mask':reverse(captured['mask']),
                     'condition':{key:reverse(value) for key,value in captured['condition'].items()}}
    pairs=[(captured[key],reversed_inputs[key]) for key in ('x','sigma','mask')]
    pairs += [(captured['condition'][key],reversed_inputs['condition'][key]) for key in captured['condition']]
    if not all(torch.equal(a,reverse(b)) for a,b in pairs):
        raise RuntimeError('Permutation changed a captured input value')
    output_rows=[]
    baseline=None
    backend.reset_peak_memory()
    with _EXECUTION_LOCK,torch.cuda.device(backend.device),torch.no_grad(),backend._autocast(),backend.model.ema_scope('Permutation diagnostic'):
        with block_capture(backend) as state:
            for name,values in [('normal',captured),('reversed',reversed_inputs),('normal_repeat',captured)]:
                state['pass']=name
                backend.synchronize();start=time.perf_counter()
                out=original(values['x'],values['sigma'],values['condition'],values['mask'])
                backend.synchronize()
                array=out.detach().float().cpu().numpy()
                row={'pass':name,'instrumented_wall_s':time.perf_counter()-start,'shape':list(array.shape)}
                if baseline is None:baseline=array
                else:row['comparison']=compare_permuted(baseline,array,cfg_permutation() if name=='reversed' else np.arange(100))
                output_rows.append(row)
            report={'input_permutation_exact':True,'denoiser_evaluations':3,'sampler_rollouts_completed':0,
                    'video_decoder_calls':0,'cfg_order':'UC candidate0, UC candidate1, C candidate0, C candidate1',
                    'blocks':state['rows'],'attention_calls':state['attention'],'denoiser_output':output_rows,
                    'temporary_cpu_snapshot_bytes':state['bytes'],'temporary_cpu_snapshot_limit_bytes':16*1024**3}
            report['first_observed_divergent_block']={name:next((row['block'] for row in state['rows'][name]
                if not row['comparison']['passed_exact']),None) for name in ('reversed','normal_repeat')}
        report['forward_and_attention_wrappers_restored']=True
    allocated,reserved=backend.peak_memory()
    report.update(peak_allocated_bytes=allocated,peak_reserved_bytes=reserved)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/study_batch_permutation.yaml'))
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    config,generation,options=inspect_config(args.config)
    if options.max_batch_size!=2:raise ValueError('Diagnostic is restricted to B2')
    output=args.output or Path(config['output'])
    if output.is_symlink():raise ValueError('Refusing output symlink')
    output=output.resolve();output.mkdir(parents=True,exist_ok=False)
    bindings=config_files(args.config)
    for path in [Path(__file__),Path('scripts/probe_candidate_memory.py'),Path(config['data']['manifest']),
                 Path(config['data']['manifest']+'.meta.json'),Path(config['environment_verification']),Path(config['environment_lock'])]:
        bindings[str(path.resolve())]=file_sha256(path)
    report={'schema_version':1,'status':'running','started_at':datetime.now(timezone.utc).isoformat(),
            'config':config,'bound_file_sha256':bindings,'headline_eligible':False,'quality_validated':False,
            'scope':'Three instrumented evaluations of one captured first-step input; no timing or full-rollout claim',
            'interpretation_limit':'First differing full-block output localizes observations; it does not prove a specific kernel cause'}
    report_path=output/'permutation-probe.json';save(report_path,report)
    handlers={}
    def interrupted(signum,_frame):
        report['termination_signal']=signum
        raise SystemExit(128+signum)
    for sig in (signal.SIGINT,signal.SIGTERM):handlers[sig]=signal.signal(sig,interrupted)
    code=1
    try:
        from drivebench.benchmark import _validate_reference_environment
        from drivebench.benchmarks.environment import collect_environment,source_identity
        from drivebench.data import inference_request,load_manifest,read_preprocess,verify_input_checksums
        from drivebench.engine import InferenceEngine
        from drivebench.validate import input_provenance
        report['implementation_sha256']=source_identity()
        environment=collect_environment();save(output/'environment.json',environment)
        report['environment_gate']=_validate_reference_environment(config,environment)
        records=load_manifest(config['data']['manifest']);index=config['memory_probe']['request_index'];record=records[index]
        verify_input_checksums(record,config['data']['root'],include_future=False)
        request=inference_request(record,config['data']['root'],candidate_count=2)
        frames=read_preprocess(request['observed_frames'],generation.height,generation.width)
        provenance=input_provenance(config['data']['manifest'],records,record,request,frames,request_index=index)
        if provenance['partition_role']!='development' or provenance['synthetic'] or len(records)!=32:
            raise ValueError('Use only the verified real 32-request development selection')
        report['input_provenance']=provenance
        engine=InferenceEngine(config['model'],generation,options)
        report['model_identity']=engine.backend.identity;save(report_path,report)
        context=engine.prepare_context(frames)
        report['diagnostic']=run_diagnostic(engine.backend,context.payload,request,generation)
        report['status']='diagnostic_complete';code=0
    except BaseException as error:
        report.update(status='failed',error={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
        code=error.code if isinstance(error,SystemExit) and type(error.code) is int and error.code else 1
    finally:
        try:
            report['bound_files_unchanged']=all(Path(path).is_file() and file_sha256(path)==sha for path,sha in bindings.items())
            report['source_unchanged']=report.get('implementation_sha256')==source_identity()
            if not report['source_unchanged'] or not report['bound_files_unchanged']:report['status'],code='source_or_inputs_changed',1
        except Exception as error:report['finalization_error']=str(error);code=1
        report.update(exit_code=code,finished_at=datetime.now(timezone.utc).isoformat());save(report_path,report)
        for sig,handler in handlers.items():signal.signal(sig,handler)
    print(json.dumps({'status':report['status'],'report':str(report_path),'exit_code':code}))
    return code


if __name__=='__main__':raise SystemExit(main())
