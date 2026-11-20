"""One separately profiled modern request through the unchanged profiler.

Only the in-process Torch profiler is supported. The legacy Nsight subprocess
entrypoint would lose the explicit modern backend and is deliberately rejected.
An existing external bounded runner still owns termination and cleanup.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import math
import sys
from pathlib import Path
import time

from drivebench.config import digest, file_sha256
from .restart_measurement import (_archive_modern_environment, _modern_sources, _measurement_sources,
    _number, _pin, _read, _validate_plan, _verify_archived_environment, _write)


def request_window(trace):
    import ijson
    windows = []
    with Path(trace).open('rb') as stream:
        for event in ijson.items(stream,'traceEvents.item',use_float=True):
            if event.get('ph') != 'X': continue
            if event.get('cat') == 'user_annotation' and event.get('name') == 'drivebench::profiled_request':
                start,duration = event.get('ts'),event.get('dur')
                if not all(type(x) in (int,float) and math.isfinite(x) for x in (start,duration)) or duration <= 0:
                    raise ValueError('finite positive recorded request window required')
                windows.append({'start_us':start,'end_us':start+duration})
    if len(windows) != 1:
        raise ValueError('exactly one recorded request annotation required')
    window = windows[0]
    # A second streaming pass avoids retaining millions of kernel intervals.
    with Path(trace).open('rb') as stream:
        for event in ijson.items(stream,'traceEvents.item',use_float=True):
            if event.get('ph') != 'X' or event.get('cat') != 'kernel': continue
            start,duration = event.get('ts'),event.get('dur')
            if not all(type(x) in (int,float) and math.isfinite(x) for x in (start,duration)) or duration <= 0:
                raise ValueError('finite positive CUDA kernel interval required')
            if start < window['end_us'] and start+duration > window['start_us']: return window
    raise ValueError('actual CUDA kernel activity inside the request annotation required')


def profile_modern(plan_path, plan_sha256, role, output, *, deadline_unix, tool='torch'):
    if tool != 'torch': raise ValueError('modern wrapper supports only the in-process torch profiler')
    if file_sha256(plan_path) != plan_sha256: raise ValueError('plan hash mismatch')
    plan = _read(plan_path); _validate_plan(plan)
    if role not in plan['roles']: raise ValueError('unknown modern role')
    _number(deadline_unix,'deadline',positive=True)
    if time.time() >= deadline_unix: raise TimeoutError('immutable profile deadline expired')
    output = Path(output); output.mkdir(parents=True,exist_ok=False)
    _write(output/'started.json',{'plan_sha256':plan_sha256,'role':role,'profiled':True,
           'headline_eligible':False,'quality_preserving':False,'deadline_unix':deadline_unix})
    try:
        from drivebench import profile as legacy
        from drivebench.benchmarks.environment import source_identity
        from drivebench.engine import InferenceEngine
        from drivebench_tuning.modern_vista import ModernVistaBackend
        import torch
        legacy_before,measurement_before,port_before = source_identity(),_measurement_sources(),_modern_sources()
        source_before = digest({'legacy':legacy_before,'measurement':measurement_before})
        original = deepcopy(plan['configs'][role]); translated = deepcopy(original)
        translated['model']['backend'] = 'vista'
        translated = _archive_modern_environment(translated,plan['hardware'],output)
        _write(output/'original_modern_config.json',original)

        class ProfileEngine(InferenceEngine):
            def __init__(self,model,generation,optimizations):
                backend = ModernVistaBackend(original['model'],generation,optimizations)
                super().__init__(original['model'],generation,optimizations,backend=backend)
                if backend.identity.get('implementation_sha256') != port_before:
                    raise ValueError('constructed modern implementation differs from source pins')
                _write(output/'model_identity.json',backend.identity)
                self.request_range = None

            def prepare_context(self,*args,**kwargs):
                self.request_range = torch.profiler.record_function('drivebench::profiled_request')
                self.request_range.__enter__()
                try: return super().prepare_context(*args,**kwargs)
                except BaseException:
                    self.request_range.__exit__(*sys.exc_info());self.request_range=None
                    raise

            def generate(self,*args,**kwargs):
                try: return super().generate(*args,**kwargs)
                finally:
                    if self.request_range is not None:
                        self.request_range.__exit__(*sys.exc_info());self.request_range=None

        old_loader,old_engine = legacy.load_config,legacy.InferenceEngine
        old_gate = legacy._validate_reference_environment
        from .modern_verify_environment import validate_reference_environment
        try:
            legacy.load_config = lambda unused: deepcopy(translated)
            legacy.InferenceEngine = ProfileEngine
            legacy._validate_reference_environment = validate_reference_environment
            legacy.profile(str(plan_path),tool='torch',output=str(output))
        finally:
            legacy.load_config,legacy.InferenceEngine = old_loader,old_engine
            legacy._validate_reference_environment = old_gate
        if time.time() >= deadline_unix: raise TimeoutError('profile exceeded immutable deadline')
        if (source_identity() != legacy_before or _measurement_sources() != measurement_before
                or _modern_sources() != port_before):
            raise ValueError('source changed during modern profile')
        if file_sha256(original['data']['manifest']) != plan['input_manifest_sha256']:
            raise ValueError('profile input manifest changed')
        gate = _verify_archived_environment(output,plan['hardware'],_read(output/'environment.json'))
        warmup = _read(output/'warmup.json')
        before = warmup['compiler_states'][-1]
        after = legacy._compiler_state()
        if legacy._compilation_count(after) != legacy._compilation_count(before):
            raise ValueError('compiler graph count changed inside diagnostic request')
        if role == 'modern_compiled' and legacy._compilation_count(before) < 1:
            raise ValueError('compiled profile requires positive captured graph evidence')
        window = request_window(output/'torch-trace.json')
        files = ('original_modern_config.json','config.json','environment.json','model_identity.json',
                 'modern_environment_verification.json','modern_runtime_lock.json','warmup.json',
                 'torch-trace.json','operators.txt','profile_metadata.json')
        pins = {name:_pin(output/name) for name in files}
        if time.time() >= deadline_unix: raise TimeoutError('profile evidence hashing exceeded deadline')
        record = {'status':'completed','profiled':True,'headline_eligible':False,'quality_preserving':False,
                  'plan_sha256':plan_sha256,'role':role,'original_modern_config_sha256':digest(original),
                  'source_identity':source_before,'legacy_source_identity':legacy_before,
                  'measurement_implementation_sha256':measurement_before,'modern_implementation_sha256':port_before,
                  'source_unchanged':True,'environment_gate':gate,'request_window':window,
                  'window_boundary':'prepare_context through synchronized generate; excludes profiler setup and export',
                  'compiler_before':before,'compiler_after':after,'artifact_pins':pins,
                  'external_process_exit_and_cleanup_still_required':True}
        _write(output/'modern_profile.json',record)
        return record
    except BaseException as error:
        _write(output/'modern_profile_failure.json',{'status':'failed','profiled':True,
                'headline_eligible':False,'error_type':type(error).__name__})
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True);p.add_argument('--plan-sha256',required=True)
    p.add_argument('--role',choices=['modern_eager','modern_compiled'],required=True)
    p.add_argument('--output',required=True);p.add_argument('--deadline-unix',required=True,type=float)
    p.add_argument('--tool',choices=['torch'],default='torch')
    args=p.parse_args()
    profile_modern(args.plan,args.plan_sha256,args.role,args.output,
                   deadline_unix=args.deadline_unix,tool=args.tool)


if __name__=='__main__':main()
