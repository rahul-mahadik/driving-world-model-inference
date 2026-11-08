#!/usr/bin/env python3
"""Bounded operator-only native xFormers/forced SDPA smoke; no model or rollout.

Use the existing serialized host queue with a 180-second outer timeout. The
10–60 second internal budget bounds operator work between synchronized calls;
a blocked CUDA call still needs the external container timeout. Input shapes
come from the pinned first-step permutation observation, values are synthetic.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
from pathlib import Path
import signal
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = 'd8a32813492c0939a0a91573a945c914938972419946065e6822f6209fa470cb'
OBSERVATION_SHA = '7ba451437ef2d3734e1fb98fa320eb27385d1d9adea58a6cce63ab9657796d7f'
NATIVE_SHA = '6ee2ed49dfd70c3ccce4e046c3d15c2da35e5963f77eaa591627883a11f08fb9'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, document):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def contiguous_stride(shape):
    return [shape[1] * shape[2], shape[2], 1]


def cases_from_observation(path):
    if sha(path) != OBSERVATION_SHA:
        raise ValueError('Observed-shape evidence changed')
    document = json.loads(Path(path).read_text())
    if (document.get('implementation_sha256') != SOURCE or document.get('status') != 'diagnostic_complete'
            or document.get('source_unchanged') is not True or document.get('bound_files_unchanged') is not True):
        raise ValueError('Require the completed frozen-source permutation observation')
    calls = document['diagnostic']['attention_calls']['normal']
    cases = {}
    for call in calls:
        shapes = [call[name] for name in ('q_shape', 'k_shape', 'v_shape')]
        if (any(len(shape) != 3 or any(type(x) is not int or x < 1 for x in shape) or shape[-1] != 64 for shape in shapes)
                or shapes[1] != shapes[2] or len({shape[0] for shape in shapes}) != 1
                or call.get('dtype') != 'torch.float16' or call.get('attn_bias_type') != 'NoneType'
                or call.get('operator_argument') != 'None' or call.get('q_stride') != contiguous_stride(shapes[0])):
            raise ValueError('Observed attention differs from the audited unmasked FP16/head64 contract')
        key = tuple(tuple(shape) for shape in shapes)
        if key not in cases:
            identity = json.dumps(key, separators=(',', ':'))
            cases[key] = dict(case_id='shape-' + hashlib.sha256(identity.encode()).hexdigest()[:12],
                              q_shape=shapes[0], k_shape=shapes[1], v_shape=shapes[2], q_stride=call['q_stride'],
                              k_stride=contiguous_stride(shapes[1]), v_stride=contiguous_stride(shapes[2]),
                              seed=2026 + int(hashlib.sha256(identity.encode()).hexdigest()[:7], 16), observations=0)
        cases[key]['observations'] += 1
    if len(calls) != 144 or len(cases) != 18:
        raise ValueError('Expected all144 observed calls and18 unique shape triples')
    return list(cases.values())


def validate_semantics(q, k, v, *, attn_bias=None, p=0.0, scale=None, op=None):
    if attn_bias is not None or p != 0.0 or scale is not None or op is not None:
        raise ValueError('Only no mask, zero dropout, default 1/sqrt(64) scale and automatic native op are audited')
    tensors = (q, k, v)
    if (any(t.ndim != 3 or t.shape[-1] != 64 or str(t.dtype) != 'torch.float16' or t.device.type != 'cuda' for t in tensors)
            or k.shape != v.shape or len({t.shape[0] for t in tensors}) != 1
            or len({str(t.device) for t in tensors}) != 1
            or any(tuple(t.stride()) != tuple(contiguous_stride(t.shape)) for t in tensors)):
        raise ValueError('Expected same-device contiguous FP16 [B*heads,L,64] inputs')


def sdpa(torch, q, k, v, **kwargs):
    validate_semantics(q, k, v, **kwargs)
    # One synthetic head preserves each already-flattened upstream head/sequence.
    # Math and flash are explicitly disabled, so unavailable efficient kernels fail.
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True):
        return torch.nn.functional.scaled_dot_product_attention(
            q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), attn_mask=None,
            dropout_p=0.0, is_causal=False).squeeze(1)


class BudgetExpired(RuntimeError):
    pass


class Budget:
    def __init__(self, seconds, clock=time.monotonic):
        if type(seconds) is not int or not 10 <= seconds <= 60:
            raise ValueError('Internal operator budget must be an integer10–60seconds')
        self.clock, self.started, self.seconds = clock, clock(), seconds

    def check(self):
        if self.clock() - self.started >= self.seconds:
            raise BudgetExpired('Internal operator-work time budget expired; coverage remains incomplete')


def make_inputs(torch, case):
    generator = torch.Generator(device='cuda').manual_seed(case['seed'])
    tensors = tuple(torch.randn(case[key], generator=generator, device='cuda', dtype=torch.float16)
                    for key in ('q_shape', 'k_shape', 'v_shape'))
    validate_semantics(*tensors)
    if [list(t.stride()) for t in tensors] != [case[key] for key in ('q_stride', 'k_stride', 'v_stride')]:
        raise ValueError('Generated tensor strides differ from declared case')
    return tensors


def comparison(torch, left, right, budget):
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError('Operator output shape/dtype differs')
    total = different = 0
    maximum = square = 0.0
    # Bounded float32 comparison scratch instead of another full-size float array.
    rows = max(1, (4 * 1024**2) // math.prod(left.shape[1:]))
    for index in range(0, len(left), rows):
        budget.check()
        a, b = left[index:index+rows].float(), right[index:index+rows].float()
        if not torch.isfinite(a).all().item() or not torch.isfinite(b).all().item():
            raise ValueError('Nonfinite attention output')
        delta = a - b
        maximum = max(maximum, delta.abs().max().item())
        square += delta.square().sum(dtype=torch.float64).item()
        different += int(torch.count_nonzero(delta).item())
        total += delta.numel()
    return dict(atol=0, rtol=0, passed_exact=different == 0, max_absolute_error=maximum,
                rmse=math.sqrt(square / total), different_values=different, values=total)


def timed(torch, fn, tensors, budget):
    budget.check()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    output = fn(*tensors)
    stop.record()
    stop.synchronize()
    elapsed = start.elapsed_time(stop)
    del output
    budget.check()
    return elapsed


def trace_evidence(path, backend, expected_calls):
    trace = json.loads(Path(path).read_text())
    events = trace.get('traceEvents', [])
    efficient = [e for e in events if e.get('name') == 'aten::_scaled_dot_product_efficient_attention']
    forbidden = [e['name'] for e in events if e.get('name') in
                 {'aten::_scaled_dot_product_attention_math', 'aten::_scaled_dot_product_flash_attention'}]
    kernels = [e for e in events if e.get('cat') == 'kernel' and e.get('dur', 0) > 0
               and any(word in e.get('name', '').lower() for word in ('attention', 'fmha'))]
    if not kernels or (backend == 'sdpa' and (len(efficient) != expected_calls or forbidden)):
        raise ValueError('Separate profiler pass lacks required efficient-attention dispatch/kernel evidence')
    return dict(trace_sha256=sha(path), efficient_operator_calls=len(efficient), forbidden_operator_calls=forbidden,
                attention_kernel_events=len(kernels), attention_kernel_names=sorted({e['name'] for e in kernels}),
                scope='Instrumented dispatch proof only; excluded from CUDA-event timing samples')


def run(torch, native, cases, output, report, budget):
    functions = {'native_xformers': lambda *xs: native(*xs, attn_bias=None, p=0.0, scale=None, op=None),
                 'sdpa': lambda *xs: sdpa(torch, *xs)}
    # Tiny self/cross calls must execute before any full observed-shape allocation.
    report['stage'] = 'tiny_eligibility'
    for length in (25, 1):
        budget.check()
        case = dict(q_shape=[2,25,64], k_shape=[2,length,64], v_shape=[2,length,64],
                    q_stride=[1600,64,1], k_stride=[length*64,64,1], v_stride=[length*64,64,1], seed=2026+length)
        tensors = make_inputs(torch, case)
        left, right = functions['native_xformers'](*tensors), functions['sdpa'](*tensors)
        torch.cuda.synchronize()
        report['tiny_eligibility'].append({'key_length': length, 'comparison': comparison(torch,left,right,budget)})
        del left, right, tensors
    report['stage'] = 'unprofiled_operator_timings'
    for case in cases:
        budget.check()
        tensors = make_inputs(torch, case)
        left, right = functions['native_xformers'](*tensors), functions['sdpa'](*tensors)
        torch.cuda.synchronize()
        row = {**case, 'comparison': comparison(torch,left,right,budget), 'cuda_event_ms': {name: [] for name in functions}}
        del left, right
        # The numerical calls are the one warmup/operator; timed order alternates.
        for repetition in range(3):
            order = list(functions) if repetition % 2 == 0 else list(reversed(functions))
            for name in order:
                row['cuda_event_ms'][name].append(timed(torch, functions[name], tensors, budget))
        row['median_cuda_event_ms'] = {name: statistics.median(values) for name, values in row['cuda_event_ms'].items()}
        report['cases'].append(row)
        save(output/'operator-probe.json',report)
        del tensors
    report['stage'] = 'separate_instrumented_dispatch_proof'
    for name, fn in functions.items():
        budget.check()
        trace = output / (name + '-trace.json')
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profiler:
            for case in cases:
                budget.check()
                tensors = make_inputs(torch,case)
                result = fn(*tensors)
                torch.cuda.synchronize()
                del result, tensors
        profiler.export_chrome_trace(str(trace))
        report['profiler'][name] = trace_evidence(trace,name,len(cases))
        save(output/'operator-probe.json',report)
    budget.check()
    return all(row['comparison']['passed_exact'] for row in report['cases'])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observations',type=Path,default=Path('outputs/study-probes/batch-permutation-attempt-2/permutation-probe.json'))
    parser.add_argument('--environment-verification',type=Path,default=Path('outputs/environment-study-01/verification.json'))
    parser.add_argument('--environment-lock',type=Path,default=Path('outputs/environment-study-01/reference-lock.json'))
    parser.add_argument('--seconds',type=int,default=60)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if not 10 <= args.seconds <= 60:parser.error('--seconds must be10–60')
    cases=cases_from_observation(args.observations)
    if args.output.is_symlink():raise ValueError('Output symlink is forbidden')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    bindings={str(p.resolve()):sha(p) for p in [Path(__file__),args.observations,args.environment_verification,args.environment_lock]}
    report=dict(schema_version=1,status='running',started_at=datetime.now(timezone.utc).isoformat(),
                bound_file_sha256=bindings,expected_source_identity=SOURCE,operator_budget_seconds=args.seconds,
                proposed_external_timeout_seconds=180,headline_eligible=False,quality_validated=False,
                full_rollouts=0,model_loaded=False,stage='runtime_verification',tiny_eligibility=[],cases=[],profiler={},
                scope='Synthetic fixed-seed q/k/v values at18 observed B2 first-step denoiser shapes; operator-only discovery',
                semantics={'dtype':'fp16','head_dimension':64,'attn_mask':None,'dropout_p':0.0,'scale':'1/sqrt(64)',
                           'is_causal':False,'native_op':'automatic upstream','sdpa_backends':{'flash':False,'math':False,'memory_efficient':True}},
                stride_provenance='Q strides observed directly. K/V strides were not logged; declared contiguous from pinned Vista construction, not claimed observed.',
                timing_policy='One numerical warmup/operator,3alternating CUDA-event samples/operator/shape; input allocation and comparisons excluded. Separate profiler passes provide no timing samples.')
    save(output/'operator-probe.json',report)
    old_handlers={}
    def stop(signum,_frame):raise SystemExit(128+signum)
    for sig in (signal.SIGINT,signal.SIGTERM):old_handlers[sig]=signal.signal(sig,stop)
    code=1
    try:
        import torch
        import xformers
        import xformers.ops as ops
        from drivebench.benchmark import _validate_reference_environment
        from drivebench.benchmarks.environment import collect_environment,source_identity
        if source_identity()!=SOURCE or torch.__version__!='2.0.1+cu118' or xformers.__version__!='0.0.20':
            raise ValueError('Require frozen d8 source/Torch2.0.1+cu118/native xFormers0.0.20')
        native=ops.memory_efficient_attention
        if getattr(native,'_torchdynamo_orig_callable',None) is not None or sha(inspect.getsourcefile(native))!=NATIVE_SHA:
            raise ValueError('Native xFormers callable changed')
        report['native_callable_sha256']=NATIVE_SHA
        environment=collect_environment();save(output/'environment.json',environment)
        report['environment_gate']=_validate_reference_environment({'environment_verification':str(args.environment_verification),
            'environment_lock':str(args.environment_lock)},environment)
        settings=(torch.backends.cuda.flash_sdp_enabled(),torch.backends.cuda.math_sdp_enabled(),torch.backends.cuda.mem_efficient_sdp_enabled())
        budget=Budget(args.seconds)
        try:
            with torch.cuda.device(0),torch.no_grad():
                torch.cuda.reset_peak_memory_stats()
                exact=run(torch,native,cases,output,report,budget)
                report['status']='execution_complete_exact' if exact else 'execution_complete_strict_failed'
                report['strict_passed']=exact;code=0 if exact else 2
        finally:
            after=(torch.backends.cuda.flash_sdp_enabled(),torch.backends.cuda.math_sdp_enabled(),torch.backends.cuda.mem_efficient_sdp_enabled())
            report['sdpa_backend_settings_restored']=settings==after
            report['native_callable_unchanged']=ops.memory_efficient_attention is native
            report['operator_work_wall_seconds']=time.monotonic()-budget.started
            report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            report['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
            if settings!=after or ops.memory_efficient_attention is not native:
                raise RuntimeError('Process attention settings changed')
    except BaseException as error:
        report.update(status='operator_budget_exhausted' if isinstance(error,BudgetExpired) else 'failed',
                      error={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
        code=3 if isinstance(error,BudgetExpired) else 1
    finally:
        try:
            from drivebench.benchmarks.environment import source_identity
            report['source_unchanged']=source_identity()==SOURCE
            report['bound_files_unchanged']=all(sha(path)==value for path,value in bindings.items())
            if not report['source_unchanged'] or not report['bound_files_unchanged']:
                report['status']='source_or_inputs_changed';code=1
        except Exception as error:report['finalization_error']=str(error);code=1
        report.update(exit_code=code,completed_shape_cases=len(report['cases']),expected_shape_cases=len(cases),
                      finished_at=datetime.now(timezone.utc).isoformat())
        save(output/'operator-probe.json',report)
        for sig,handler in old_handlers.items():signal.signal(sig,handler)
    print(json.dumps({'status':report['status'],'exit_code':code,'report':str(output/'operator-probe.json')}))
    return code


if __name__=='__main__':raise SystemExit(main())
