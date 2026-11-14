"""CPU contract tests; no timing or quality result is produced by these fixtures."""
from copy import deepcopy
import json
import math
from pathlib import Path
import numpy as np
import pytest
from drivebench_tuning.restart_measurement import make_plan, paired_summary, read_restart
from drivebench.config import file_sha256,digest
from drivebench_tuning.attribution import partition,analyze_trace
from drivebench_tuning.study_cost import estimate


def configs(tmp_path):
    manifest=tmp_path/'scenes.jsonl'
    metadata={'units':'meters','coordinate_frame':'fixture_ego','axis_order':['lateral','longitudinal'],'timestamps_seconds':[.5,1,1.5,2],'provenance':'unit test fixture only','verified_against_annotation_source':True}
    manifest.write_text(''.join(json.dumps({'scene_id':f'scene-{i}','request_id':f'request-{i}','dataset':'nuscenes','observed_frames':[f'fixture-{i}.jpg'],'trajectory':[0]*8,'trajectory_metadata':metadata,'seeds':[11]})+'\n' for i in range(3)))
    environment={'gpu':{'uuid':'fixture-only','name':'NVIDIA RTX A6000','memory_total_mib':'49140'},'packages':{}}
    proof={'proof_kind':'modern_declared_gpu_v1','passed':True,'legacy_h100_proof':False,
           'hardware_policy':{'schema_version':1,'gpu_name':'NVIDIA RTX A6000','minimum_memory_mib':45000,'minimum_host_ram_bytes':32*2**30,'mig_supported':False},
           'environment':environment,'profiling_proof':{'modern_runtime_passed':True,'torch_cuda_execution_passed':True,
               'hardware_counters_collected':True,'nsys_trace_collected':True,
               'ncu':{'returncode':0,'stdout':'"Metric Name","Metric Value"\n"sm__cycles_elapsed.avg","1"'},
               'nsys_stats':{'returncode':0,'stdout':'"Total Time (ns)","Instances","Name"\n1,1,drivebench_smoke'}}}
    from drivebench.benchmarks.environment import reference_identity
    (tmp_path/'verification.json').write_text(json.dumps(proof));(tmp_path/'runtime-lock.json').write_text(json.dumps(reference_identity(environment)))
    eager={'environment_verification':str(tmp_path/'verification.json'),'environment_lock':str(tmp_path/'runtime-lock.json'),'model':{'backend':'modern_vista','modern_runtime':{'fixture_only':True}},
           'data':{'manifest':str(manifest),'root':str(tmp_path)},'generation':{},'optimizations':{'compile_denoiser':False},
           'benchmark':{'requests':3,'save_outputs':3,'warmup_requests':3,'max_warmup_requests':12,'headline':False}}
    compiled=deepcopy(eager);compiled['optimizations']['compile_denoiser']=True
    return eager,compiled


def test_plan_is_balanced_fresh_modern_only_with_fixed_three_scenes(tmp_path):
    plan=make_plan(*configs(tmp_path))
    assert len(plan['schedule'])==24 and plan['legacy_eager']['new_processes']==0
    for block in range(0,12,2):
        a=[r['role'] for r in plan['schedule'] if r['block']==block]
        b=[r['role'] for r in plan['schedule'] if r['block']==block+1]
        assert a==list(reversed(b))
    assert plan['resource_admitted'] is False


@pytest.mark.parametrize('key,value',[('steps',20),('dtype','bf16'),('guidance_scale',2.)])
def test_plan_rejects_semantic_difference_between_roles(tmp_path,key,value):
    a,b=configs(tmp_path);b['generation'][key]=value
    with pytest.raises(ValueError,match='differ only|Phase 1 fixes'):make_plan(a,b)


def test_plan_rejects_implicit_baseline_and_changed_scene_membership(tmp_path):
    a,b=configs(tmp_path);b['model']['backend']='vista'
    with pytest.raises(ValueError,match='explicit'):make_plan(a,b)
    a,b=configs(tmp_path);mp=Path(a['data']['manifest']);records=[json.loads(line) for line in mp.read_text().splitlines()]
    for record in records:record['scene_id']='same'
    mp.write_text(''.join(json.dumps(row)+'\n' for row in records))
    with pytest.raises(ValueError,match='distinct'):make_plan(a,b)


def rows(savings=(10.,10.,10.,10.)):
    result=[]
    for block,saving in enumerate(savings*2):
        for role in ('modern_eager','modern_compiled'):
            compiled=role=='modern_compiled'
            warm=50-saving if compiled else 50.
            result.append({'role':role,'block':block,'worker_id':f'{block}-{role}',
                           'runtime':{'uuid':'fixture'},'source_identity':'fixture','model_identity':{'fixture':'same'},
                           'initialization_s':10.,'first_request_s':140. if compiled else 50.,
                           'steady_mean_s':warm,'cold_prefix_requests':3,
                           'cold_prefix_s':10+3*warm+(90 if compiled else 0),
                           'peak_allocated_bytes':100,'peak_reserved_bytes':120})
    return result


def test_break_even_charges_complete_prefix_and_counts_served_requests():
    result=paired_summary(rows(),bootstrap_samples=100)
    assert result['break_even']['requests']==10 # strict gain after 90/10
    assert result['break_even']['no_finite_crossing_fraction']==0
    assert result['warm_throughput_gain']==pytest.approx(.25)
    assert result['baseline_table'][1]['compile_only_s'] is None
    assert result['quality_preserving'] is False


def test_uncertainty_retains_no_crossing_draws_not_conditioned_on_winners():
    result=paired_summary(rows((-2.,-1.,1.,2.)),bootstrap_samples=1000)
    assert result['break_even']['finite'] is False
    assert result['break_even']['upper_unbounded'] is True
    assert result['break_even']['ci95_requests'][1] is None
    assert result['break_even']['no_finite_crossing_fraction']>.4


@pytest.mark.parametrize('change',['runtime','source','worker','missing'])
def test_summary_rejects_pooled_or_incomplete_restarts(change):
    data=rows()
    if change=='runtime':data[0]['runtime']={'uuid':'another'}
    elif change=='source':data[0]['source_identity']='changed'
    elif change=='worker':data[0]['worker_id']=data[1]['worker_id']
    else:data.pop()
    with pytest.raises(ValueError):paired_summary(data,bootstrap_samples=100)


def event(name,start,end,resource='0'):
    return {'name':name,'start_us':start,'end_us':end,'resource':resource}


def test_nested_operator_partition_does_not_double_count_inclusive_parents():
    report=partition([event('parent',0,100),event('child',20,80)],0,120,nested=True)
    assert report['partition_us']=={'parent':40.,'child':60.,'unassigned_or_idle':20.}
    assert report['partition_sum_us']==120.


def test_overlapping_gpu_streams_get_one_overlap_bin():
    report=partition([event('a',0,80,'stream1'),event('b',40,100,'stream2')],0,100)
    assert report['partition_us']=={'a':40.,'overlapping_or_ambiguous':40.,'b':20.}
    assert report['partition_sum_us']==100.


def test_partial_non_nested_operator_overlap_remains_ambiguous():
    report=partition([event('a',0,80),event('b',40,100)],0,100,nested=True)
    assert report['partition_us']['overlapping_or_ambiguous']==40


def test_trace_lanes_are_separate_and_container_stage_is_excluded(tmp_path):
    events=[{'ph':'X','cat':cat,'name':name,'ts':start,'dur':duration,'pid':pid,'tid':0}
            for cat,name,start,duration,pid in [
                ('user_annotation','drivebench::gpu_inference',0,100,0),
                ('user_annotation','drivebench::denoising_gpu',10,70,0),
                ('cpu_op','aten::add',20,20,0),('kernel','native kernel',30,50,1)]]
    trace=tmp_path/'trace.json';trace.write_text(json.dumps({'traceEvents':events}))
    result=analyze_trace(trace,start_us=0,end_us=100)
    assert len(result['lanes'])==3
    assert all(lane['partition_sum_us']==100 for lane in result['lanes'].values())
    assert 'drivebench::gpu_inference' not in result['lanes']['host_stage_ranges']['partition_us']
    assert result['headline_eligible'] is False


def test_sweep_projection_keeps_time_assumption_distinct_from_request_lower_bound():
    result=estimate(1000,3,2)
    assert result['minimum_candidate_measurement_requests']==6000
    assert result['illustrative_gpu_hours']==[75.,100.]
    assert result['raw_bytes_one_copy']==1061683968000
    assert result['raw_bytes_all_copies']==2123367936000
    assert result['resource_admitted'] is False
    assert 'NOT a lower bound' in result['interpretation']


@pytest.mark.parametrize('arguments',[(0,3,2),(1000,0,2),(1000,3,False)])
def test_cost_rejects_invalid_grid(arguments):
    with pytest.raises(ValueError):estimate(*arguments)


def restart_fixture(tmp_path):
    from drivebench_tuning import restart_measurement as module
    plan=make_plan(*configs(tmp_path),blocks=8);plan_sha=digest(plan)
    folder=tmp_path/'restart';root=folder/'benchmark';root.mkdir(parents=True)
    job=next(row for row in plan['schedule'] if row['role']=='modern_compiled')
    expected=deepcopy(plan['configs'][job['role']]);expected['model']['backend']='vista';expected['profiled']=False;expected['data']['manifest_sha256']=plan['input_manifest_sha256']
    identity={'backend':'modern_vista','modern_runtime':plan['configs'][job['role']]['model']['modern_runtime'],
              'compilation':{'enabled':True,'target':'denoiser','backend':'inductor','mode':'default','fullgraph':False,'dynamic':False,'compile_cost':'lazy_first_use_must_be_measured','fallback_free_proven':False},'implementation_sha256':{'modern_vista.py':'a'*64,'modern_attention.py':'b'*64}}
    def row(index,*,measured=False):
        binding=plan['request_bindings'][index%3]
        value={'request_id':binding['request_id'],'scene_id':binding['scene_id'],'request_index':index,'repeat':index//3,
               'latency_s':50.,'status':'ok','failures':[],'fallbacks':[],
               'cache_stats':{'context_hit':False,'denoiser_evaluations_complete':True},
               'peak_allocated_bytes':10,'peak_reserved_bytes':20,'candidate_count':1,'batch_size':1,
               'input_condition':'uncached','denoiser_evaluations':50,'candidates':[],
               'effective_config':{**expected['generation'],'candidate_count':1,'requested_batch_size':1,
                                   'actual_batch_sizes':[1],'optimizations':expected['optimizations']}}
        if measured:
            relative=f'outputs/request-{index}.npy';target=root/relative;target.parent.mkdir(exist_ok=True)
            target.write_bytes(b'unit-test-only raw-output hash fixture')
            value['candidates']=[{key:binding[key] for key in ('candidate_id','seed','action_kind','trajectory_sha256')}]
            value['candidates'][0]['frames_path']=relative
        return value
    payloads={'config.json':expected,'model_identity.json':identity,
              'environment.json':{'gpu':{'uuid':'fixture-only','name':plan['hardware']['gpu_name'],'memory_total_mib':'49140'},'packages':{}},
              'environment_gate.json':{'passed':True,'verification_sha256':file_sha256(tmp_path/'verification.json')},
              'modern_environment_verification.json':json.loads((tmp_path/'verification.json').read_text()),
              'modern_runtime_lock.json':json.loads((tmp_path/'runtime-lock.json').read_text()),
              'cold_start.json':{'status':'ready','model_load_s':5.,'warmups':[row(i) for i in range(3)]},
              'summary.json':{'recompiled_during_measurement':False,'failures':0,'compiler_before':{'stats':{'unique_graphs':1}},'compiler_after':{'stats':{'unique_graphs':1}}}}
    for name,value in payloads.items():(root/name).write_text(json.dumps(value))
    (root/'measurements.jsonl').write_text(''.join(json.dumps(row(i,measured=True))+'\n' for i in range(3)))
    record={'status':'completed','synthetic':False,'plan_sha256':plan_sha,**job,
            'compiler_cache_initially_empty':True,'source_unchanged':True,'worker_id':'fixture-only','legacy_source_identity':'fixture-only',
            'measurement_implementation_sha256':{name:'c'*64 for name in ('restart_measurement.py','modern_profile.py','attribution.py','study_cost.py')},
            'original_modern_config_sha256':digest(plan['configs'][job['role']]),'modern_backend_source_sha256':'a'*64,'modern_implementation_sha256':{'modern_vista.py':'a'*64,'modern_attention.py':'b'*64},'worker_wall_s':300.,
            'raw_output_pins':{str(path.relative_to(root)):module._pin(path) for path in (root/'outputs').iterdir()},
            'artifact_pins':{path.name:module._pin(path) for path in root.iterdir() if path.is_file()}}
    record['source_identity']=digest({'legacy':record['legacy_source_identity'],'measurement':record['measurement_implementation_sha256']})
    (folder/'restart.json').write_text(json.dumps(record))
    return folder,plan,plan_sha,record


def test_read_restart_reconstructs_explicit_baseline_and_cold_prefix(tmp_path):
    folder,plan,sha,record=restart_fixture(tmp_path)
    result=read_restart(folder,plan,sha)
    assert result['cold_prefix_s']==155. and result['first_request_s']==50.
    assert result['model_identity']['backend']=='modern_vista'


@pytest.mark.parametrize('mutation',['missing_pin','hash_drift','wrong_backend','wrong_compile','wrong_compile_backend','manifest_hash','no_graph','unverified_environment','profiled','scene_order','fallback','no_work','wrong_seed','too_many_warmups','raw_drift','wrong_gpu'])
def test_read_restart_rejects_mislabeled_or_invalid_worker_evidence(tmp_path,mutation):
    from drivebench_tuning import restart_measurement as module
    folder,plan,sha,record=restart_fixture(tmp_path);root=folder/'benchmark'
    if mutation=='missing_pin':record['artifact_pins'].pop('environment.json')
    elif mutation=='hash_drift':(root/'cold_start.json').write_text('{}')
    elif mutation=='raw_drift':next((root/'outputs').iterdir()).write_bytes(b'changed')
    else:
        name={'wrong_backend':'model_identity.json','wrong_compile':'model_identity.json','wrong_compile_backend':'model_identity.json','no_graph':'summary.json','manifest_hash':'config.json','too_many_warmups':'cold_start.json','wrong_gpu':'environment.json',
              'unverified_environment':'environment_gate.json','profiled':'config.json',
              'scene_order':'measurements.jsonl','fallback':'measurements.jsonl','no_work':'measurements.jsonl','wrong_seed':'measurements.jsonl'}[mutation]
        if name.endswith('jsonl'):
            value=[json.loads(line) for line in (root/name).read_text().splitlines()]
            if mutation=='scene_order':value.reverse()
            elif mutation=='no_work':value[0].update(candidate_count=0,denoiser_evaluations=0,effective_config={'steps':1})
            elif mutation=='wrong_seed':value[0]['candidates'][0]['seed']=999
            else:value[0]['fallbacks']=['silently eager']
            (root/name).write_text(''.join(json.dumps(row)+'\n' for row in value))
        else:
            value=json.loads((root/name).read_text())
            if mutation=='wrong_backend':value['backend']='upstream'
            elif mutation=='wrong_compile':value['compilation']['enabled']=False
            elif mutation=='wrong_compile_backend':value['compilation'].update(backend='eager',target='decoder',fullgraph=True,dynamic=True)
            elif mutation=='manifest_hash':value['data']['manifest_sha256']='f'*64
            elif mutation=='too_many_warmups':value['warmups']=value['warmups']*5
            elif mutation=='wrong_gpu':value['gpu']['name']='Different GPU'
            elif mutation=='no_graph':value['compiler_before']['stats']['unique_graphs']=0
            elif mutation=='unverified_environment':value['passed']=False
            elif mutation=='profiled':value['profiled']=True
            (root/name).write_text(json.dumps(value))
        record['artifact_pins'][name]=module._pin(root/name)
    (folder/'restart.json').write_text(json.dumps(record))
    with pytest.raises(ValueError):read_restart(folder,plan,sha)


def test_collect_uses_explicit_factory_fresh_caches_and_preserves_failure(tmp_path,monkeypatch):
    import sys,types,os,time
    from drivebench_tuning import restart_measurement as module
    from drivebench import benchmark
    plan=make_plan(*configs(tmp_path),blocks=8);path=tmp_path/'plan.json';path.write_text(json.dumps(plan))
    monkeypatch.setattr(module,'_USED_PROCESS',False);monkeypatch.delitem(sys.modules,'torch',raising=False)
    fake=types.ModuleType('drivebench_tuning.modern_vista');calls=[]
    class Modern:
        def __init__(self,model,generation,optimizations):calls.append((model,generation,optimizations));self.identity={}
    fake.ModernVistaBackend=Modern
    monkeypatch.setitem(sys.modules,'drivebench_tuning.modern_vista',fake)
    monkeypatch.setattr(module,'_modern_sources',lambda:{'modern_vista.py':'a'*64,'modern_attention.py':'b'*64})
    original=benchmark.InferenceEngine
    observed=[]
    def rejected(config,output):
        from drivebench.types import GenerationConfig,OptimizationConfig
        assert benchmark.InferenceEngine is not original
        assert config['model']['backend']=='vista'
        benchmark.InferenceEngine(config['model'],GenerationConfig(),OptimizationConfig())
        for name in module._CACHE_VARS:
            directory=Path(os.environ[name]);assert directory.is_dir() and not list(directory.iterdir());observed.append(directory)
        raise RuntimeError('fixture intentional worker failure')
    monkeypatch.setattr(benchmark,'run_benchmark',rejected)
    with pytest.raises(RuntimeError,match='intentional'):
        module.collect_restart(path,file_sha256(path),plan['schedule'][0]['run_id'],tmp_path/'attempt',deadline_unix=time.time()+60)
    assert benchmark.InferenceEngine is original
    assert len(calls)==1 and calls[0][0]['backend']=='modern_vista'
    assert len(observed)==2 and all(p.exists() for p in observed)
    assert json.loads((tmp_path/'attempt/restart-failure.json').read_text())['status']=='failed'
    with pytest.raises(RuntimeError,match='fresh interpreter'):
        module.collect_restart(path,file_sha256(path),plan['schedule'][0]['run_id'],tmp_path/'another',deadline_unix=time.time()+60)


def test_summary_rejects_changed_weights_even_with_same_software_identity():
    data=rows();data[0]['model_identity']={'fixture':'different weights'}
    with pytest.raises(ValueError,match='weights'):paired_summary(data,bootstrap_samples=100)


def test_historical_row_remains_separate_and_unknown_cold_times_stay_null():
    from drivebench_tuning.restart_measurement import historical_legacy_row
    path=Path('results/2026-10-30-h100-tradeoff/summary.json')
    result=historical_legacy_row(path,file_sha256(path))
    assert result['baseline']=='legacy_eager' and result['warm_median_s']==pytest.approx(57.338388142)
    assert result['first_request_mean_s'] is None and result['pooled'] is False
    assert result['cross_runtime_speedup_claim'] is False


def test_public_modern_configs_differ_only_compile_flag():
    from drivebench.config import load_config
    from drivebench_tuning.restart_measurement import _effective
    eager=_effective(load_config('configs/research/modern-eager.yaml'))
    compiled=_effective(load_config('configs/research/modern-compiled.yaml'))
    assert compiled['optimizations']['compile_denoiser'] is True
    compiled['optimizations']['compile_denoiser']=False
    assert eager==compiled
    assert eager['model']['backend']=='modern_vista'
    assert eager['benchmark']['requests']==3 and eager['benchmark']['headline'] is False


def test_equal_warm_cost_with_lower_startup_is_already_faster():
    from drivebench_tuning.restart_measurement import _break_even
    assert _break_even(-10,0,3)==3
    assert math.isinf(_break_even(0,0,3))
    assert math.isinf(_break_even(10,0,3))
    assert math.isinf(_break_even(-10,-1,3)) # no lasting amortization


def test_feasibility_is_two_full_workload_workers_without_quality_claim(tmp_path):
    from drivebench_tuning.restart_measurement import feasibility_summary
    a,b=configs(tmp_path);manifest=Path(a['data']['manifest']);manifest.write_text(manifest.read_text().splitlines()[0]+'\n')
    for config in (a,b):config['benchmark'].update(requests=1,save_outputs=1)
    plan=make_plan(a,b,blocks=1,mode='feasibility',gpu_name='NVIDIA RTX A6000',minimum_memory_mib=45000,compute_usd_per_hour=.888)
    assert len(plan['schedule'])==2 and plan['warm_requests_per_process']==1
    data=rows()[:2]
    for row in data:row['worker_wall_s']=200
    result=feasibility_summary(data,plan)
    assert result['cost_per_quality_valid_request'] is None
    assert result['noise_or_speedup_claim'] is False
    assert result['subsequent_full_restart_study_admitted'] is False
    assert result['rows'][0]['illustrative_worker_compute_usd_per_completed_warm_request']==pytest.approx(.888*200/3600)


def test_bootstrap_keeps_four_process_reversal_groups():
    result=paired_summary(rows(),bootstrap_samples=100)
    assert result['randomized_reversal_groups']==4 and result['paired_blocks']==8
    assert result['statistical_unit']=='four-process AB/BA reversal group'


def test_cost_rate_override_is_an_explicit_assumption():
    result=estimate(1000,1,1,seconds_per_request=(10,20),compute_usd_per_hour=.888)
    assert result['compute_usd_per_hour_assumption']==.888
    assert result['illustrative_compute_usd']==pytest.approx([1000*10/3600*.888,1000*20/3600*.888])
    assert 'caller assumptions' in result['assumption_provenance']


@pytest.mark.parametrize('change',['legacy_kind','gpu','host_ram','mig','runtime','lock','missing_attention_pin','attention_hash'])
def test_restart_binds_full_modern_policy_runtime_and_port(tmp_path,change):
    from drivebench_tuning import restart_measurement as module
    folder,plan,sha,record=restart_fixture(tmp_path);root=folder/'benchmark'
    name='modern_environment_verification.json'
    if change=='lock':name='modern_runtime_lock.json'
    if change in ('missing_attention_pin','attention_hash'):name='model_identity.json'
    value=json.loads((root/name).read_text())
    if change=='legacy_kind':value['proof_kind']='historical_h100'
    elif change=='gpu':value['hardware_policy']['gpu_name']='another'
    elif change=='host_ram':value['hardware_policy']['minimum_host_ram_bytes']=1
    elif change=='mig':value['hardware_policy']['mig_supported']=True
    elif change=='runtime':value['environment']['gpu']['uuid']='another'
    elif change=='lock':value['gpu']['uuid']='another'
    elif change=='missing_attention_pin':value['implementation_sha256'].pop('modern_attention.py')
    elif change=='attention_hash':value['implementation_sha256']['modern_attention.py']='f'*64
    (root/name).write_text(json.dumps(value));record['artifact_pins'][name]=module._pin(root/name)
    if name=='modern_environment_verification.json':
        gate={'passed':True,'verification_sha256':file_sha256(root/name)}
        (root/'environment_gate.json').write_text(json.dumps(gate));record['artifact_pins']['environment_gate.json']=module._pin(root/'environment_gate.json')
    (folder/'restart.json').write_text(json.dumps(record))
    with pytest.raises(ValueError):read_restart(folder,plan,sha)


def test_modern_profile_rejects_legacy_subprocess_tools_before_any_work(tmp_path):
    from drivebench_tuning.modern_profile import profile_modern
    with pytest.raises(ValueError,match='in-process'):
        profile_modern('unused','unused','modern_eager',tmp_path/'out',deadline_unix=1,tool='nsys')
    assert not (tmp_path/'out').exists()


def test_modern_profile_scopes_actual_backend_and_records_bound_trace(tmp_path,monkeypatch):
    import sys,types,time
    from drivebench_tuning import modern_profile as module
    from drivebench import profile as legacy
    from drivebench.types import GenerationConfig,OptimizationConfig
    plan=make_plan(*configs(tmp_path),blocks=8);path=tmp_path/'plan.json';path.write_text(json.dumps(plan))
    pins={'modern_vista.py':'a'*64,'modern_attention.py':'b'*64}
    fake=types.ModuleType('drivebench_tuning.modern_vista');calls=[]
    class Modern:
        def __init__(self,model,generation,optimizations):
            calls.append(model);self.identity={'backend':'modern_vista','implementation_sha256':pins}
    fake.ModernVistaBackend=Modern
    monkeypatch.setitem(sys.modules,'drivebench_tuning.modern_vista',fake)
    monkeypatch.setitem(sys.modules,'torch',types.ModuleType('torch'))
    monkeypatch.setattr(module,'_modern_sources',lambda:dict(pins))
    before_loader,before_engine=legacy.load_config,legacy.InferenceEngine
    def fake_profile(unused,tool,output):
        assert tool=='torch';config=legacy.load_config(unused)
        assert config['model']['backend']=='vista'
        engine=legacy.InferenceEngine(config['model'],GenerationConfig(),OptimizationConfig())
        assert engine.backend.identity['backend']=='modern_vista'
        out=Path(output)
        payloads={'config.json':config,'environment.json':json.loads((tmp_path/'verification.json').read_text())['environment'],
                  'warmup.json':{'compiler_states':[{'stats':{'unique_graphs':0}}]},'profile_metadata.json':{'profiled':True},
                  'torch-trace.json':{'traceEvents':[{'ph':'X','cat':'user_annotation','name':'drivebench::profiled_request','ts':10,'dur':20},
                                                   {'ph':'X','cat':'kernel','name':'fixture CUDA event','ts':15,'dur':5}]}}
        for name,value in payloads.items():(out/name).write_text(json.dumps(value))
        (out/'operators.txt').write_text('Synthetic producer fixture; not GPU evidence')
    monkeypatch.setattr(legacy,'profile',fake_profile);monkeypatch.setattr(legacy,'_compiler_state',lambda:{'stats':{'unique_graphs':0}})
    result=module.profile_modern(path,file_sha256(path),'modern_eager',tmp_path/'profile',deadline_unix=time.time()+60)
    assert legacy.load_config is before_loader and legacy.InferenceEngine is before_engine
    assert calls[0]['backend']=='modern_vista'
    assert result['profiled'] is True and result['headline_eligible'] is False
    assert result['request_window']=={'start_us':10,'end_us':30}
    assert set(result['artifact_pins'])=={'original_modern_config.json','config.json','environment.json','model_identity.json',
        'modern_environment_verification.json','modern_runtime_lock.json','warmup.json','torch-trace.json','operators.txt','profile_metadata.json'}


@pytest.mark.parametrize('mode',['no_window','two_windows','no_kernels','bad_time','outside_only'])
def test_profile_window_requires_one_real_recorded_range(tmp_path,mode):
    from drivebench_tuning.modern_profile import request_window
    marker={'ph':'X','cat':'user_annotation','name':'drivebench::profiled_request','ts':10,'dur':20}
    kernel={'ph':'X','cat':'kernel','name':'fixture','ts':15,'dur':5}
    events=[marker,kernel]
    if mode=='no_window':events=[kernel]
    elif mode=='two_windows':events.append(marker)
    elif mode=='no_kernels':events=[marker]
    elif mode=='outside_only':kernel['ts']=40
    else:marker['dur']=-1
    path=tmp_path/'trace.json';path.write_text(json.dumps({'traceEvents':events}))
    with pytest.raises(ValueError):request_window(path)


def test_relocated_measurement_modules_have_independent_complete_source_pins():
    from drivebench_tuning.restart_measurement import _measurement_sources
    pins=_measurement_sources()
    assert set(pins)=={'restart_measurement.py','modern_profile.py','attribution.py','study_cost.py'}
    assert all(len(value)==64 for value in pins.values())
    assert not Path('drivebench/restart_measurement.py').exists()
    assert not Path('drivebench/modern_profile.py').exists()


def test_restart_rejects_rebound_measurement_implementation(tmp_path):
    folder,plan,sha,record=restart_fixture(tmp_path)
    record['measurement_implementation_sha256']['modern_profile.py']='f'*64
    (folder/'restart.json').write_text(json.dumps(record))
    with pytest.raises(ValueError,match='measurement implementation'):
        read_restart(folder,plan,sha)
