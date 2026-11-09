from copy import deepcopy
import json
import time
import numpy as np
import pytest
from drivebench_tuning.vista_agent import (
    ALLOWED_VALUES, BatchRequest, file_hash, model_for_settings, probe_runtime,
    reference_settings, run_batch, validate_settings,
)
from drivebench_tuning.agent_metrics import measure_pair, negative_control


def req(scene='scene-a', seed=11, replicate=0, rid='record-a'):
    return BatchRequest(rid, scene, seed, replicate, np.zeros((1,3,8,8),np.float32), np.zeros(8,np.float32))


def args():
    return dict(settings=reference_settings(synthetic=True), model={'backend':'synthetic'}, requests=[req()],
                cohort_id='cpu-fixture', expected_runtime=probe_runtime(synthetic=True), source_files={},
                deadline_unix=time.time()+60, synthetic=True, minimum_free_bytes=0)


def test_actual_core_engine_persistent_batch_and_raw_evidence(tmp_path):
    kw=args()
    kw['requests']=[req(),req('scene-b',12,0,'record-b')]
    out=run_batch(tmp_path/'batch',**kw)
    assert out['status']=='ok' and out['synthetic'] and not out['quality_assessed']
    assert len(out['rows'])==2 and len(out['warmups'])==1
    for row in out['rows']+out['warmups']:
        assert row['valid'] and row['latency_ms']>0 and row['denoiser_evaluations']==2
        p=tmp_path/'batch'/row['output_file']
        assert file_hash(p)==row['output_sha256']
        assert np.load(p,allow_pickle=False).shape==(3,8,8,3)
    assert json.loads((tmp_path/'batch'/'completed.json').read_text())==out
    with pytest.raises(FileExistsError):
        run_batch(tmp_path/'batch',**kw)


@pytest.mark.parametrize('path,value', [('generation.steps',1),('generation.height',64),('generation.dtype','fp32'),
    ('generation.decode_chunk_size',4),('optimizations.compile_denoiser',1),('optimizations.max_batch_size',2),
    ('optimizations.feature_cache_interval',2),('optimizations.feature_cache_module','os.system'),
    ('optimizations.conditioning_cache_bytes',100)])
def test_fixed_and_invalid_candidate_options(path,value):
    config=reference_settings()
    section,key=path.split('.')
    config[section][key]=value
    with pytest.raises(ValueError):validate_settings(config)


@pytest.mark.parametrize('path',list(ALLOWED_VALUES))
def test_allowed_configs_are_declared(path):
    for value in ALLOWED_VALUES[path]:
        config=reference_settings()
        section,key=path.split('.')
        config[section][key]=value
        assert validate_settings(config)==config


def test_model_paths_cannot_be_agent_settings():
    config=reference_settings()
    config['model']={'checkpoint':'/tmp/changed'}
    with pytest.raises(ValueError):validate_settings(config)
    with pytest.raises(ValueError):model_for_settings({},reference_settings())


def test_trusted_mapper_only_enables_pinned_compiler_runtime():
    model={key:'trusted' for key in ('checkpoint','checkpoint_sha256','upstream_dir','upstream_commit','inference_config')}
    config=reference_settings()
    assert model_for_settings(model,config)['backend']=='upstream'
    config['optimizations']['compile_denoiser']=True
    assert model_for_settings(model,config)['compiler_runtime']=='registered_einops_attention_v1'
    config['generation']['dtype']='bf16'
    config['optimizations']['compile_denoiser']=False
    assert model_for_settings(model,config)['backend']=='vista'


def test_inputs_are_immutable_and_content_bound():
    r=req()
    with pytest.raises(ValueError):r.observed_frames.setflags(write=True)
    assert r.input_sha256!=req(seed=12).input_sha256
    with pytest.raises(ValueError):BatchRequest('r','s',True,0,np.zeros((1,3,8,8)),np.zeros(8))
    with pytest.raises(ValueError):BatchRequest('r','s',1,0,np.full((1,3,8,8),np.nan),np.zeros(8))


def test_runtime_and_source_drift_before_engine(tmp_path):
    kw=args();kw['expected_runtime']={'kind':'fake'}
    with pytest.raises(ValueError,match='runtime drift'):run_batch(tmp_path/'batch',**kw)
    assert not (tmp_path/'batch').exists()
    p=tmp_path/'source';p.write_text('x')
    kw=args();kw['source_files']={str(p):'0'*64}
    with pytest.raises(ValueError,match='source drift'):run_batch(tmp_path/'batch',**kw)


def test_storage_and_deadline_fail_closed(tmp_path):
    kw=args();kw['minimum_free_bytes']=10**20
    with pytest.raises(ValueError,match='storage'):run_batch(tmp_path/'batch',**kw)
    assert json.loads((tmp_path/'batch'/'failure.json').read_text())['status']=='not_started'
    kw=args();kw['deadline_unix']=time.time()-1
    with pytest.raises(ValueError,match='deadline'):run_batch(tmp_path/'late',**kw)


def test_failed_invariant_preserves_actual_raw_array(tmp_path,monkeypatch):
    from drivebench.engine import InferenceEngine
    real=InferenceEngine.generate
    def corrupt(self,*a,**k):
        result=real(self,*a,**k);result.denoiser_evaluations=0;return result
    monkeypatch.setattr(InferenceEngine,'generate',corrupt)
    with pytest.raises(ValueError,match='invariant'):run_batch(tmp_path/'batch',**args())
    assert (tmp_path/'batch'/'00000.npy').is_file()
    assert not (tmp_path/'batch'/'completed.json').exists()
    assert json.loads((tmp_path/'batch'/'00000.json').read_text())['valid'] is False


def test_duplicate_requests_rejected(tmp_path):
    kw=args();kw['requests']=[req(),req(rid='different')]
    with pytest.raises(ValueError,match='duplicate scene'):run_batch(tmp_path/'batch',**kw)


class FakePerceptual:
    # Exact numeric test double, explicitly not LPIPS.
    def compare(self,a,b):return float(np.abs(a-b).mean())


def save_pair(tmp_path,a,b):
    paths=[tmp_path/'ref.npy',tmp_path/'cand.npy']
    for p,v in zip(paths,[a,b]):np.save(p,v,allow_pickle=False)
    return paths,dict(reference_sha256=file_hash(paths[0]),candidate_sha256=file_hash(paths[1]),perceptual_metric=FakePerceptual())


def test_array_metrics_exclude_conditioning_and_require_actual_hashes(tmp_path):
    a=np.zeros((3,8,8,3),np.float32);b=a.copy();b[0]=1
    paths,kw=save_pair(tmp_path,a,b)
    assert measure_pair(*paths,**kw)=={'mae':0.,'temporal_derivative_mae':0.,'lpips':0.}
    paths[1].write_bytes(b'bad')
    with pytest.raises(ValueError,match='hash'):measure_pair(*paths,**kw)


@pytest.mark.parametrize('value',[None,float('nan'),-1,True])
def test_missing_perceptual_never_falls_back(tmp_path,value):
    a=np.zeros((3,8,8,3),np.float32)
    paths,kw=save_pair(tmp_path,a,a)
    class Missing:
        def compare(self,a,b):return value
    kw['perceptual_metric']=Missing()
    with pytest.raises(ValueError,match='perceptual'):measure_pair(*paths,**kw)


def test_negative_controls_preserve_conditioning_without_mutation():
    a=np.random.default_rng(0).random((4,8,8,3),dtype=np.float32)
    original=a.copy()
    for control in ('freeze_future','black_future','reverse_future'):
        b=negative_control(a,control)
        assert np.array_equal(a,original) and np.array_equal(b[0],a[0]) and not np.array_equal(b[1:],a[1:])
    with pytest.raises(ValueError):negative_control(a,'code')


def test_cleanup_failure_never_writes_completion(tmp_path,monkeypatch):
    import drivebench_tuning.vista_agent as module
    def fail():raise RuntimeError('cleanup failed')
    monkeypatch.setattr(module.gc,'collect',fail)
    with pytest.raises(RuntimeError):run_batch(tmp_path/'batch',**args())
    assert (tmp_path/'batch'/'cleanup-failure.json').is_file()
    assert not (tmp_path/'batch'/'completed.json').exists()


def test_effective_settings_mismatch_invalidates_saved_result(tmp_path,monkeypatch):
    from drivebench.engine import InferenceEngine
    real=InferenceEngine.generate
    def wrong(self,*a,**k):
        result=real(self,*a,**k);result.effective_config['dtype']='bf16';return result
    monkeypatch.setattr(InferenceEngine,'generate',wrong)
    with pytest.raises(ValueError,match='invariant'):run_batch(tmp_path/'batch',**args())
    assert (tmp_path/'batch'/'00000.npy').is_file()
    assert not (tmp_path/'batch'/'completed.json').exists()


def test_warmup_compilation_allowed_but_measured_recompile_retained_and_rejected(tmp_path,monkeypatch):
    import drivebench_tuning.vista_agent as module
    counts=iter([0,1,1,2])
    monkeypatch.setattr(module,'_compiler_snapshot',lambda synthetic:{'stats':{'unique_graphs':next(counts)}})
    with pytest.raises(ValueError,match='invariant'):run_batch(tmp_path/'batch',**args())
    warmup=json.loads((tmp_path/'batch'/'00000.json').read_text())
    measured=json.loads((tmp_path/'batch'/'00001.json').read_text())
    assert warmup['valid'] and warmup['compiled_graph_delta']==1
    assert not measured['valid'] and measured['compiled_graph_delta']==1
    assert (tmp_path/'batch'/'00001.npy').is_file()
    assert not (tmp_path/'batch'/'completed.json').exists()
