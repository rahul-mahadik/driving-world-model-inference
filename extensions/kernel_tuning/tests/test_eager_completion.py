"""Single-eager terminal routing; strict artifact checks remain in read_restart."""
import importlib.util
import json
from pathlib import Path
import pytest
from drivebench_tuning import eager_completion as m

def fixture(tmp_path):
    plan={'mode':'feasibility','warm_requests_per_process':1,'scene_ids':['scene'],
          'configs':{'modern_eager':{'model':{'backend':'modern_vista'},
                     'generation':{'steps':50,'total_frames':25,'conditioning_frames':1,
                                   'height':576,'width':1024,'decode_chunk_size':14,'dtype':'fp16'},
                     'optimizations':{'compile_denoiser':False}}},
          'schedule':[{'role':'modern_eager','run_id':'000-modern_eager'},
                      {'role':'modern_compiled','run_id':'001-modern_compiled'}],
          'hardware':{'gpu_name':'NVIDIA A40'}}
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan))
    return plan,path,tmp_path/'run',tmp_path/'terminal.json'

def test_only_validated_eager_and_no_comparison_claim(tmp_path,monkeypatch):
    plan,path,run,out=fixture(tmp_path);calls=[]
    def strict(directory,p,sha):
        calls.append((directory,p,sha));return {'role':'modern_eager','run_id':'000-modern_eager'}
    monkeypatch.setattr(m,'read_restart',strict)
    result=m.verify(path,m.file_sha256(path),run,out)
    assert calls==[(run,plan,m.file_sha256(path))]
    assert result['executed_roles']==['modern_eager'] and result['speedup'] is None
    assert not any(result[k] for k in ['comparison_complete','compiled_execution_requested','profile_execution_requested','quality_preserving','headline_eligible'])
    assert json.loads(out.read_text())==result

@pytest.mark.parametrize('mutate',[lambda p:p.update(mode='paired'),lambda p:p.update(warm_requests_per_process=3),lambda p:p.update(scene_ids=['a','b']),
    lambda p:p['configs']['modern_eager']['generation'].update(steps=1),
    lambda p:p['configs']['modern_eager']['generation'].update(height=64),
    lambda p:p['configs']['modern_eager']['optimizations'].update(compile_denoiser=True)])
def test_changed_task_rejected_before_verifier(tmp_path,monkeypatch,mutate):
    plan,path,run,out=fixture(tmp_path);mutate(plan);path.write_text(json.dumps(plan))
    monkeypatch.setattr(m,'read_restart',lambda *a:pytest.fail('strict verifier should not be called'))
    with pytest.raises(ValueError):m.verify(path,m.file_sha256(path),run,out)
    assert not out.exists()

def test_hash_and_compiled_row_rejected(tmp_path,monkeypatch):
    plan,path,run,out=fixture(tmp_path)
    monkeypatch.setattr(m,'read_restart',lambda *a:{'role':'modern_compiled','run_id':'001-modern_compiled'})
    with pytest.raises(ValueError,match='hash'):m.verify(path,'0'*64,run,out)
    with pytest.raises(ValueError,match='eager'):m.verify(path,m.file_sha256(path),run,out)
    assert not out.exists()

def test_strict_validation_failure_propagates_and_no_overwrite(tmp_path,monkeypatch):
    plan,path,run,out=fixture(tmp_path)
    def fail(*a):raise ValueError('raw artifact differs')
    monkeypatch.setattr(m,'read_restart',fail)
    with pytest.raises(ValueError,match='raw artifact'):m.verify(path,m.file_sha256(path),run,out)
    assert not out.exists();out.write_text('preserved')
    with pytest.raises(FileExistsError):m.verify(path,m.file_sha256(path),run,out)
    assert out.read_text()=='preserved'


@pytest.mark.parametrize('tamper', [False, True])
def test_serialized_eager_restart_uses_real_reconstruction(tmp_path, tamper):
    from test_restart_measurement import restart_fixture
    folder, plan, _, record = restart_fixture(tmp_path, eager_only=True)
    path = tmp_path/'plan.json'
    path.write_text(json.dumps(plan))
    record['plan_sha256'] = m.file_sha256(path)
    (folder/'restart.json').write_text(json.dumps(record))
    output = tmp_path/'verified-eager.json'
    if tamper:
        next((folder/'benchmark/outputs').iterdir()).write_bytes(b'changed output')
        with pytest.raises(ValueError, match='raw output'):
            m.verify(path, m.file_sha256(path), folder, output)
        assert not output.exists()
    else:
        result = m.verify(path, m.file_sha256(path), folder, output)
        assert result['row']['steady_mean_s'] == 50.0
        assert result['row']['cold_prefix_requests'] == 3
        assert result['executed_roles'] == ['modern_eager']
        assert result['speedup'] is None
        assert not result['quality_validated']
