import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import time
import pytest
spec=importlib.util.spec_from_file_location('fixed_assets',Path(__file__).resolve().parents[3]/'scripts/fetch_modern_assets.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def proof(tmp_path):
    gpu={'uuid':'TEST-ONLY','driver_version':'580.99'}
    env={'gpus':[gpu],'gpu':gpu}
    r={'proof_kind':m.PROOF_KIND,'passed':True,'dedicated_attestation':True,'failures':[],
       'legacy_h100_proof':False,'hardware_policy':{'schema_version':1,'gpu_name':'NVIDIA A40','minimum_memory_mib':45000,'minimum_host_ram_bytes':64*2**30,'mig_supported':False},
       'profiling_proof':{'modern_runtime_passed':True,'torch_cuda_execution_passed':True,'nsys_trace_collected':True,'hardware_counters_collected':False,'hardware_counters_required':False,'hardware_counter_status':'not_requested_counter_free_protocol','nsys':{'returncode':0},'nsys_stats':{'returncode':0,'stdout':'Total Time (ns),Instances,Name\n100,1,drivebench_smoke\n'}},
       'environment':env,'environment_after':copy.deepcopy(env)}
    p=tmp_path/'verification.json';p.write_text(json.dumps(r));(tmp_path/'smoke-systems.nsys-rep').write_bytes(b'TEST NOT REAL')
    return p,m.sha(p),r


def test_rejects_wrong_or_old_proof_before_fetch(tmp_path,monkeypatch):
    p,h,r=proof(tmp_path);monkeypatch.setattr(m,'current_gpu_identity',lambda:('TEST-ONLY','580.99'))
    assert m.require_proof(p,h)['passed']
    for mutate in (lambda x:x.update(proof_kind='modern_declared_gpu_v1'),lambda x:x.update(passed=False),lambda x:x['profiling_proof'].update(hardware_counters_required=True),lambda x:x['profiling_proof'].update(nsys_trace_collected=False),lambda x:x['profiling_proof']['nsys_stats'].update(stdout='')):
        bad=copy.deepcopy(r);mutate(bad);p.write_text(json.dumps(bad))
        with pytest.raises(RuntimeError):m.require_proof(p,m.sha(p))


def test_wrong_hash_device_or_missing_trace_rejected(tmp_path,monkeypatch):
    p,h,r=proof(tmp_path);monkeypatch.setattr(m,'current_gpu_identity',lambda:('TEST-ONLY','580.99'))
    with pytest.raises(RuntimeError):m.require_proof(p,'0'*64)
    monkeypatch.setattr(m,'current_gpu_identity',lambda:('DIFFERENT','580.99'))
    with pytest.raises(RuntimeError):m.require_proof(p,h)
    monkeypatch.setattr(m,'current_gpu_identity',lambda:('TEST-ONLY','580.99'))
    (tmp_path/'smoke-systems.nsys-rep').write_bytes(b'')
    with pytest.raises(RuntimeError):m.require_proof(p,h)


class Response(io.BytesIO):
    def __init__(self,data,header=None):super().__init__(data);self.headers={} if header is None else {'Content-Length':str(header)}


def fixture_fetch(tmp_path,monkeypatch,data=b'abc',expected=b'abc',header=3):
    calls=[]
    monkeypatch.setattr(m,'ASSETS',{'vista':{'url':'https://example.invalid/test','sha256':hashlib.sha256(expected).hexdigest(),'bytes':len(expected),'relative_path':'checkpoints/vista.safetensors'}})
    monkeypatch.setattr(m,'require_proof',lambda *x:calls.append('proof'))
    monkeypatch.setattr(m.urllib.request,'urlopen',lambda *a,**kw:Response(data,header))
    return calls


def test_fixed_bytes_and_partial_hardlink_preserved(tmp_path,monkeypatch):
    calls=fixture_fetch(tmp_path,monkeypatch)
    r=m.fetch('vista',tmp_path,'proof','hash',time.time()+10)
    dest=tmp_path/'checkpoints/vista.safetensors'
    assert dest.read_bytes()==b'abc' and dest.with_suffix('.safetensors.part').read_bytes()==b'abc'
    assert len(calls)==2 and r['model_loaded'] is False
    with pytest.raises(FileExistsError):m.fetch('vista',tmp_path,'proof','hash',time.time()+10)


@pytest.mark.parametrize('data,header',[(b'ab',3),(b'abcd',3),(b'xyz',3),(b'abc',4)])
def test_incomplete_oversize_wrong_hash_header_preserved(tmp_path,monkeypatch,data,header):
    fixture_fetch(tmp_path,monkeypatch,data=data,header=header)
    with pytest.raises(RuntimeError):m.fetch('vista',tmp_path,'proof','hash',time.time()+10)
    assert not (tmp_path/'checkpoints/vista.safetensors').exists()
    assert (tmp_path/'checkpoints/vista.safetensors.part').exists()


@pytest.mark.parametrize('deadline',[0,float('nan'),float('inf')])
def test_bad_deadline_prevents_work(tmp_path,monkeypatch,deadline):
    calls=fixture_fetch(tmp_path,monkeypatch)
    with pytest.raises(TimeoutError):m.fetch('vista',tmp_path,'proof','hash',deadline)
    assert not calls
