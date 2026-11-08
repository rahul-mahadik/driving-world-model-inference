import copy
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import probe_attention_operators as probe


@pytest.fixture
def observation(tmp_path,monkeypatch):
    # Synthetic test catalog reproduces the18 observed shape classes, without
    # making the CPU suite depend on ignored/private GPU report files.
    calls=[]
    for batch,length in [(500,9216),(1000,2304),(2000,576),(2000,144),
                         (32768,25),(20480,25),(26624,25),(13312,25),(11520,25)]:
        for keys in (length,1):
            calls.append(dict(q_shape=[batch,length,64],k_shape=[batch,keys,64],v_shape=[batch,keys,64],
                              q_stride=[length*64,64,1],dtype='torch.float16',attn_bias_type='NoneType',operator_argument='None'))
    document=dict(implementation_sha256=probe.SOURCE,status='diagnostic_complete',source_unchanged=True,
                  bound_files_unchanged=True,diagnostic={'attention_calls':{'normal':calls*8}})
    path=tmp_path/'source.json';path.write_text(json.dumps(document))
    monkeypatch.setattr(probe,'OBSERVATION_SHA',probe.sha(path))
    return path


def test_recorded_cases_cover_all_calls_and_distinguish_stride_provenance(observation):
    cases=probe.cases_from_observation(observation)
    assert len(cases)==18 and sum(case['observations'] for case in cases)==144
    assert len({case['seed'] for case in cases})==18
    assert all(case['q_stride']==probe.contiguous_stride(case['q_shape']) for case in cases)
    assert max(case['q_shape'][0] for case in cases)==32768
    assert any(case['k_shape'][1]==1 for case in cases)


@pytest.mark.parametrize('mutation',['hash','dtype','mask','stride','head','missing'])
def test_bad_observation_fails_before_torch_import(tmp_path,monkeypatch,mutation,observation):
    document=json.loads(observation.read_text())
    call=document['diagnostic']['attention_calls']['normal'][0]
    if mutation=='dtype':call['dtype']='torch.float32'
    if mutation=='mask':call['attn_bias_type']='LowerTriangularMask'
    if mutation=='stride':call['q_stride']=[1,2,3]
    if mutation=='head':call['q_shape'][-1]=128
    if mutation=='missing':document['diagnostic']['attention_calls']['normal'].pop()
    path=tmp_path/'observation.json';path.write_text(json.dumps(document)+'\n')
    if mutation!='hash':monkeypatch.setattr(probe,'OBSERVATION_SHA',probe.sha(path))
    with pytest.raises(ValueError):probe.cases_from_observation(path)


class Tensor:
    def __init__(self,shape=(2,25,64),stride=None):
        self.shape=shape;self.ndim=len(shape);self.dtype='torch.float16';self.device=SimpleNamespace(type='cuda')
        self._stride=stride or probe.contiguous_stride(shape)
    def stride(self):return self._stride
    def unsqueeze(self,dim):
        assert dim==1
        return SimpleNamespace(shape=(self.shape[0],1,*self.shape[1:]),original=self)


@pytest.mark.parametrize('fail',[False,True])
def test_forced_efficient_backend_no_fallback_and_restoration(fail):
    flags={'flash':True,'math':True,'efficient':True};events=[]
    @contextmanager
    def context(**kwargs):
        old=flags.copy();events.append(kwargs)
        flags.update(flash=kwargs['enable_flash'],math=kwargs['enable_math'],efficient=kwargs['enable_mem_efficient'])
        try:yield
        finally:flags.update(old)
    q,k,v=Tensor(),Tensor((2,1,64)),Tensor((2,1,64))
    # Same-device equality uses str(device); SimpleNamespace has stable values.
    def operator(a,b,c,**kwargs):
        assert a.shape==(2,1,25,64) and b.shape==(2,1,1,64)
        assert kwargs=={'attn_mask':None,'dropout_p':0.0,'is_causal':False}
        assert flags=={'flash':False,'math':False,'efficient':True}
        if fail:raise RuntimeError('no available kernel')
        return SimpleNamespace(squeeze=lambda dim:'result' if dim==1 else None)
    torch=SimpleNamespace(backends=SimpleNamespace(cuda=SimpleNamespace(sdp_kernel=context)),
                          nn=SimpleNamespace(functional=SimpleNamespace(scaled_dot_product_attention=operator)))
    if fail:
        with pytest.raises(RuntimeError,match='no available kernel'):probe.sdpa(torch,q,k,v)
    else:assert probe.sdpa(torch,q,k,v)=='result'
    assert flags=={'flash':True,'math':True,'efficient':True} and len(events)==1


@pytest.mark.parametrize('kwargs',[{'attn_bias':'mask'},{'p':.1},{'scale':.125},{'op':'override'}])
def test_undeclared_semantics_rejected(kwargs):
    with pytest.raises(ValueError):probe.validate_semantics(Tensor(),Tensor(),Tensor(),**kwargs)


@pytest.mark.parametrize('mutation',['none','no_kernel','no_dispatch','math','flash','count'])
def test_profiler_evidence_separate_dispatch_and_gpu_kernel_required(tmp_path,mutation):
    events=[{'name':'aten::_scaled_dot_product_efficient_attention','cat':'cpu_op'},
            {'name':'cutlass::AttentionKernel<float>','cat':'kernel','dur':10}]
    if mutation=='no_kernel':events.pop()
    if mutation=='no_dispatch':events.pop(0)
    if mutation in ['math','flash']:events.append({'name':'aten::_scaled_dot_product_attention_math' if mutation=='math' else 'aten::_scaled_dot_product_flash_attention','cat':'cpu_op'})
    path=tmp_path/'trace.json';path.write_text(json.dumps({'traceEvents':events}))
    if mutation=='none':assert probe.trace_evidence(path,'sdpa',1)['efficient_operator_calls']==1
    else:
        with pytest.raises(ValueError):probe.trace_evidence(path,'sdpa',2 if mutation=='count' else 1)


def test_budget_is_bounded_and_expiration_is_not_unsupported():
    for value in [True,0,9,61,10.5]:
        with pytest.raises(ValueError):probe.Budget(value)
    now=[100.0];budget=probe.Budget(10,clock=lambda:now[0]);budget.check()
    now[0]=110
    with pytest.raises(probe.BudgetExpired):budget.check()


def test_tiny_device_failure_prevents_observed_shape_allocations(tmp_path,monkeypatch):
    allocations=[]
    def inputs(torch,case):allocations.append(case['q_shape']);return (Tensor(),Tensor(),Tensor())
    monkeypatch.setattr(probe,'make_inputs',inputs)
    monkeypatch.setattr(probe,'sdpa',lambda *args:(_ for _ in ()).throw(RuntimeError('no available kernel')))
    report={'tiny_eligibility':[],'cases':[],'profiler':{}}
    with pytest.raises(RuntimeError):probe.run(SimpleNamespace(),lambda *args,**kwargs:Tensor(),
        [{'q_shape':[500,9216,64]}],tmp_path,report,probe.Budget(10))
    assert allocations==[[2,25,64]] and report['stage']=='tiny_eligibility' and not report['cases']
