from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from drivebench.types import GenerationConfig
from scripts import probe_candidate_memory as probe


@pytest.mark.parametrize('size', range(1,17))
def test_capacity_configs_keep_exact_primary_settings_and_no_fallback(size):
    config,generation,options=probe.inspect_config(Path(f'configs/study_memory_batch{size}.yaml'))
    assert generation==GenerationConfig()
    assert options.max_batch_size==size
    assert config['benchmark']['headline'] is False
    assert config['memory_probe']['request_index']==28


@pytest.mark.parametrize('mutation', ['steps','cache','compiler','count','boolean','headline'])
def test_capacity_config_rejects_other_runtime_changes(tmp_path,mutation):
    config,_,_=probe.inspect_config(Path('configs/study_memory_batch4.yaml'))
    if mutation=='steps':config['generation']['steps']=30
    if mutation=='cache':config['optimizations']['conditioning_cache_bytes']=100
    if mutation=='compiler':config['optimizations']['compile_denoiser']=True
    if mutation=='count':config['benchmark']['candidate_count']=8
    if mutation=='boolean':config['memory_probe']['batch_size']=True
    if mutation=='headline':config['benchmark']['headline']=True
    path=tmp_path/'bad.yaml';path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):probe.inspect_config(path)


@pytest.mark.parametrize('failure', [None,MemoryError('simulated OOM'),ValueError('bad output')])
def test_capacity_attempt_calls_once_never_splits_and_preserves_peak(failure):
    generation=GenerationConfig(height=64,width=64)
    calls=[];cleanup=[]
    class Backend:
        max_batch_size=16
        def generate_batch(self,payload,actions,seeds,config):
            calls.append((len(seeds),config.steps))
            if failure:raise failure
            return SimpleNamespace(frames=np.zeros((4,25,64,64,3),np.float32),fallbacks=[],
                denoiser_evaluations=50,stage_timings_s={'denoising_gpu':3.0},gpu_inference_s=4.0,cache_stats={})
        def peak_memory(self):return (12345,23456)
        def release_after_oom(self):cleanup.append(True)
    backend=Backend()
    engine=SimpleNamespace(backend=backend,synchronize=lambda:None,reset_peak_memory=lambda:None,
                           prepare_context=lambda _:SimpleNamespace(payload={},preparation_s=0.5))
    request={'candidate_ids':[str(i) for i in range(4)],'seeds':[1,2,3,4],'trajectories':[np.zeros((4,2))]*4}
    times=iter([10.0,15.0])
    row,code=probe.execute_attempt(engine,None,request,generation,4,clock=lambda:next(times))
    assert calls==[(4,50)] and row['retry_count']==0 and not row['oom_fallback_enabled']
    assert row['peak_allocated_bytes']==12345 and row['peak_reserved_bytes']==23456
    assert row['wall_s_excluding_file_decode_and_export']==5.0
    if failure is None:
        assert code==0 and row['actual_batch_sizes']==[4] and len(row['output_sha256'])==4
    else:
        assert code==(2 if isinstance(failure,MemoryError) else 1) and row['actual_batch_sizes']==[]
    assert len(cleanup)==int(isinstance(failure,MemoryError))
