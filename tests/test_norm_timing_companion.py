"""Fabricated CPU fixtures only; no inference, qualification or GPU claims."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from drivebench.config import digest, file_sha256, validate_run_config
from drivebench.data.manifests import write_manifest
from drivebench.evaluation.io import load_run

SPEC=importlib.util.spec_from_file_location('norm_companion',Path(__file__).resolve().parents[1]/'scripts/run_norm_timing_companion.py')
norm=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(norm)


def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path


def fixture(tmp_path,monkeypatch):
    parent=tmp_path/'study';parent.mkdir()
    manifest=parent/'evaluation.jsonl'
    write_manifest(manifest,[{'request_id':f'e{i}','scene_id':f's{i}'} for i in range(128)],{'partition_role':'held_out_evaluation'})
    env={'gpu':{'uuid':'GPU-CPU-FIXTURE'},'torch':{'version':'fixture'},'python':'fixture',
         'packages':{'torch':'fixture'},'tools':{'ncu':{'stdout':'fixture'}}}
    verification=put(tmp_path/'environment.json',{'passed':True,'environment':env})
    lock=put(tmp_path/'lock.json',norm.pilot._runtime(env));inference=put(tmp_path/'inference.json',{'fixture':True})
    cfg={'model':{'backend':'upstream','upstream_commit':'a'*40,'checkpoint_sha256':'b'*64,'inference_config':str(inference)},
         'generation':{},'optimizations':{},'data':{'manifest':str(manifest),'verify_checksums':True},
         'benchmark':{'workload':'single_future','candidate_count':1,'warmup_requests':3,'max_warmup_requests':5,
                      'save_outputs':0,'input_condition':'uncached','requests':50,'headline':False},
         'environment_verification':str(verification),'environment_lock':str(lock)}
    gen,opt=validate_run_config(cfg);cfg['generation'],cfg['optimizations']=gen.to_dict(),opt.to_dict()
    pp={'reference':'upstream','base_config':cfg,'quality':{},'variants':{'upstream':{},'compiled':{'model':{
        'backend':'vista','compiler_runtime':'registered_einops_attention_v1'},'optimizations':{'compile_denoiser':True}}},
        'timing':{'warmups':3},'manifests':{'evaluation':{'path':str(manifest)},'development':{'path':str(manifest),'sha256':file_sha256(manifest),'metadata':{'scene_ids':[f'ds{i}' for i in range(32)]}}}}
    put(parent/'plan.json',pp);selected={'variants':['upstream','compiled']}
    for key in ('development_evaluation','quality_policy'):
        p=put(parent/(key+'.json'),{'scope':'fabricated CPU fixture'});selected[key]=str(p);selected[key+'_sha256']=file_sha256(p)
    put(parent/'selection.json',selected)
    monkeypatch.setattr(norm.study,'_load_plan',lambda p:(parent,copy.deepcopy(pp)))
    monkeypatch.setattr(norm.study,'_selection',lambda *args:json.loads((parent/'selection.json').read_text()))
    calibration={'configuration':cfg,'identity':norm.pilot._identity(cfg,env),'source_identity':norm.SOURCE,
        'reference_backend':'upstream','reference_optimizations':cfg['optimizations'],'reference_input_condition':'uncached',
        'reference_compiler_selectors':norm.pilot._compiler_selectors(cfg)}
    policy={'frozen_at':'2026-01-03T00:00:00+00:00','relative_latency_tolerance':.02,
            'evidence':{'development_scenes':[f'ds{i}' for i in range(32)]}}
    policy_path=put(tmp_path/'pilot.json',policy)
    monkeypatch.setattr(norm.pilot,'validate_pilot_policy',lambda p:(policy,calibration))
    proof=put(tmp_path/'strict.json',{'schema_version':2,'passed':True,'atol':0,'rtol':0,'implementation_sha256':norm.SOURCE,
        'runtime_identity':norm.pilot._runtime(env),'checks':{'upstream_agreement':{'passed':True}},
        'runs':{'upstream':{'model_identity':{},'outputs':{'normal':str(inference)},'outputs_sha256':{'normal':file_sha256(inference)}}}})
    cfg['numerical_validation']=str(proof)
    legacy=put(tmp_path/'legacy.json',{'schema_version':2,'passed':False,'atol':0,'rtol':0,'input_provenance':{'candidate_ids':['a','b']}})
    compiled_cfg=norm.study._variant_config(pp,'compiled',manifest,50,1,quality=False)
    identity={**cfg['model'],'backend':'vista','inference_config_sha256':file_sha256(inference),
              'compiler_runtime':{'name':'registered_einops_attention_v1','implementation_sha256':'f'*64}}
    qualified={'type':'compiler_norm_qualification_v1','qualified':True,'legacy_strict_passed':False,
        'implementation_sha256':norm.SOURCE,'runtime_identity':norm.pilot._runtime(env),
        'generation':compiled_cfg['generation'],'optimizations':compiled_cfg['optimizations'],'model_identity':identity,
        'development_manifest':{'sha256':file_sha256(manifest),'scene_ids':[f'ds{i}' for i in range(32)]},
        'exact_semantics':{'passed':True},'numerical':{'passed':True},'quality_development':{'passed':True},'localized_review':{'passed':True},
        'legacy_proof':{'path':str(legacy),'sha256':file_sha256(legacy)},'created_at':'2026-01-03T00:00:01+00:00',
        'policy':{'frozen_at':'2026-01-02T00:00:00+00:00'}}
    qp=put(tmp_path/'qualification.json',qualified)
    original_module=norm.module
    def module(name):
        if name=='qualify_compiler_numerics.py':
            return SimpleNamespace(verify=lambda p:{**norm.read(p),'report_path':str(p),'report_sha256':file_sha256(p)})
        return original_module(name)
    monkeypatch.setattr(norm,'module',module)
    monkeypatch.setattr(norm,'source_bindings',lambda:{'fixture-helper':'x'})
    monkeypatch.setattr(norm,'numerical_evidence',lambda cfg,*args:{'passed':cfg['model']['backend']=='upstream','report_path':cfg['numerical_validation'],'report_sha256':file_sha256(cfg['numerical_validation'])})
    kwargs={'baseline':'upstream','candidate':'compiled','candidate_count':1,'pilot_policy':str(policy_path),
            'baseline_proof':str(proof),'qualification':str(qp)}
    return parent,pp,tmp_path/'companion',kwargs,calibration,policy,env


def test_plan_keeps_failed_strict_and_exact_pilot_unmodified(tmp_path,monkeypatch):
    parent,pp,out,kw,*_=fixture(tmp_path,monkeypatch)
    before={str(p):file_sha256(p) for p in tmp_path.rglob('*') if p.is_file()}
    value=norm.plan_companion(parent,out,**kw)
    assert value['legacy_strict_passed']=={'upstream':True,'compiled':False}
    assert value['combined_headline_claim'] is False
    assert len(value['schedule'])==8
    assert [x['variant'] for x in value['schedule']]==['upstream','compiled','compiled','upstream','compiled','upstream','upstream','compiled']
    assert {n:sum(x['requests'] for x in value['schedule'] if x['variant']==n) for n in ('upstream','compiled')}=={'upstream':200,'compiled':200}
    assert norm._load(out)[1]==value
    assert all(file_sha256(p)==h for p,h in before.items())
    assert norm.read(value['configs']['compiled']['numerical_validation'])['passed'] is False


@pytest.mark.parametrize('fault',['qualified','semantics','numerical','quality','review','runtime','generation','options','model','legacy','created'])
def test_forged_or_wrong_qualification_cannot_plan(tmp_path,monkeypatch,fault):
    parent,pp,out,kw,*_=fixture(tmp_path,monkeypatch);q=norm.read(kw['qualification'])
    if fault=='qualified':q['qualified']=False
    elif fault in ('semantics','numerical','quality','review'):
        q[{'semantics':'exact_semantics','numerical':'numerical','quality':'quality_development','review':'localized_review'}[fault]]['passed']=False
    elif fault=='runtime':q['runtime_identity']['gpu']['uuid']='OTHER-GPU'
    elif fault=='generation':q['generation']['steps']=40
    elif fault=='options':q['optimizations']['max_batch_size']=2
    elif fault=='model':q['model_identity']['checkpoint_sha256']='z'*64
    elif fault=='legacy':q['legacy_strict_passed']=True
    elif fault=='created':q['created_at']='missing timezone'
    put(kw['qualification'],q)
    with pytest.raises(ValueError):norm.plan_companion(parent,out,**kw)
    assert not out.exists()


@pytest.mark.parametrize('fault',['selection','quality','strict','qualification','policy','config','schedule','count'])
def test_changed_evidence_rejects_resume_even_with_rehashed_plan(tmp_path,monkeypatch,fault):
    parent,pp,out,kw,*_=fixture(tmp_path,monkeypatch);value=norm.plan_companion(parent,out,**kw)
    if fault=='selection':put(parent/'selection.json',{'variants':['upstream']})
    elif fault=='quality':put(parent/'quality_policy.json',{'changed':True})
    elif fault=='strict':put(kw['baseline_proof'],{'changed':True})
    elif fault=='qualification':q=norm.read(kw['qualification']);q['new']='changed';put(kw['qualification'],q)
    elif fault=='policy':put(kw['pilot_policy'],{'changed':True})
    else:
        if fault=='config':value['configs']['compiled']['optimizations']['conditioning_cache_bytes']=4096
        if fault=='schedule':value['schedule'].reverse()
        if fault=='count':value['candidate_count']=True
        put(out/'plan.json',value);put(out/'plan-receipt.json',{'plan_sha256':file_sha256(out/'plan.json')})
    with pytest.raises((ValueError,KeyError)):norm._load(out)


def test_own_candidate4_pilot_required(tmp_path,monkeypatch):
    parent,pp,out,kw,cal,*_=fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='workload'):norm.plan_companion(parent,out,**{**kw,'candidate_count':4})
    assert not out.exists()
    cal['configuration']['benchmark'].update(candidate_count=4,workload='candidate_set')
    env=norm.read(cal['configuration']['environment_verification'])['environment']
    cal['identity']=norm.pilot._identity(cal['configuration'],env)
    value=norm.plan_companion(parent,out,**{**kw,'candidate_count':4})
    assert value['candidate_count']==4 and value['workload']=='candidate_set'


def fake_rows(scale=1.,block=0,count=1):
    return [{'request_id':f'e{i}','scene_id':f's{i}','repeat':0,'request_index':i,'status':'ok',
        'latency_s':float(scale*(1+i/1000)+block*.0001),'candidate_count':count,'fallbacks':[],
        'cache_stats':{'denoiser_evaluations_complete':True},
        'candidates':[{'candidate_id':f'e{i}:c{k}','seed':i*10+k,'action_kind':'recorded' if k==0 else 'alternative',
                       'trajectory_sha256':digest([i,k]),'frames_path':None} for k in range(count)]} for i in range(50)]


def test_statistics_preserves_raw_rows_and_legacy_flags():
    rows={'a':[fake_rows(1,i) for i in range(4)],'b':[fake_rows(.8,i) for i in range(4)]};before=copy.deepcopy(rows)
    result=norm.statistics(rows,['a','b'],{'relative_latency_tolerance':.02})
    assert rows==before and result['paired_requests']==200 and result['independent_scenes']==50
    assert result['status']=='pass' and result['metrics']['p50']['improvement_exceeds_tolerance_and_uncertainty']
    assert result['combined_headline_claim'] is False and result['quality_preserving'] is False
    assert result['metrics']['p95']['relative_latency_change']==pytest.approx(-.2,abs=.001)


@pytest.mark.parametrize('fault',['partial','action','seed','order','nan','zero','duplicate'])
def test_statistics_rejects_invalid_or_mismatched_observations(fault):
    rows={'a':[fake_rows(1,i) for i in range(4)],'b':[fake_rows(.8,i) for i in range(4)]}
    if fault=='partial':rows['b'][1].pop()
    if fault=='action':rows['b'][0][0]['candidates'][0]['trajectory_sha256']='e'*64
    if fault=='seed':rows['b'][0][0]['candidates'][0]['seed']+=1
    if fault=='order':rows['b'][0].reverse()
    if fault=='nan':rows['b'][0][0]['latency_s']=float('nan')
    if fault=='zero':rows['b'][0][0]['latency_s']=0
    if fault=='duplicate':rows['b'][0][1]=copy.deepcopy(rows['b'][0][0])
    with pytest.raises(ValueError):norm.statistics(rows,['a','b'],{'relative_latency_tolerance':.02})


def test_apparent_regression_requires_confirmation_not_rejection():
    result=norm.statistics({'a':[fake_rows(1,i) for i in range(4)],'b':[fake_rows(1.2,i) for i in range(4)]},['a','b'],{'relative_latency_tolerance':.02})
    assert result['status']=='rerun_required' and result['combined_headline_claim'] is False


def block_fixture(tmp_path,monkeypatch):
    parent,pp,out,kw,cal,policy,env=fixture(tmp_path,monkeypatch)
    value=norm.plan_companion(parent,out,**kw);value['created_at']='2026-01-03T01:00:00+00:00'
    cfg=copy.deepcopy(value['configs']['compiled']);path=tmp_path/'block'
    cfg['started_at']='2026-01-04T00:00:00+00:00'
    meta=norm.read(cfg['data']['manifest']+'.meta.json');cfg['data'].update(manifest_sha256=file_sha256(cfg['data']['manifest']),manifest_metadata=meta)
    identity=value['qualification']['model_identity'];runtime={'passed':True,'require_graphs':True,'restoration_complete':True,
        'runtime_implementation_sha256':'f'*64,'graphs_created_by_runtime':10,'kernels_created_by_runtime':20,
        'before':{'unique_graphs':10,'generated_kernels':20,'unimplemented':{}},
        'after':{'unique_graphs':10,'generated_kernels':20,'unimplemented':{}}}
    rows=fake_rows()
    for r in rows:
        r['cache_stats']['compiler_runtime']=copy.deepcopy(runtime)
        r['effective_config']={**cfg['generation'],'optimizations':cfg['optimizations'],'requested_batch_size':1,'actual_batch_sizes':[1]}
        r['denoiser_evaluations']=50
    cold={'status':'ready','compilation_and_initialization_excluded':True,'warmups':[dict(copy.deepcopy(rows[i]),compiler={'stats':{'unique_graphs':10}}) for i in range(3)]}
    gate=norm.numerical_evidence(cfg)
    summary={'backend':'vista','synthetic':False,'profiled':False,'headline_eligible':False,'numerical_validation':gate,
             'failures':0,'measured_requests':50,'successful_requests':50,'completed_futures':50,'oom_attempts':0,
             'recompiled_during_measurement':False,'compiler_before':{'stats':{'unique_graphs':10}},'compiler_after':{'stats':{'unique_graphs':10}}}
    for name,obj in [('config.json',cfg),('environment.json',env),('model_identity.json',identity),('cold_start.json',cold),
                     ('summary.json',summary),('numerical_gate.json',gate),('environment_gate.json',{'passed':True})]:put(path/name,obj)
    (path/'measurements.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return path,cfg,value,policy,cal


def test_block_validates_legacy_false_without_promoting_it(tmp_path,monkeypatch):
    path,cfg,value,policy,cal=block_fixture(tmp_path,monkeypatch)
    before={str(p):file_sha256(p) for p in path.iterdir()}
    result=norm.validate_block(load_run(path),cfg,value,policy,cal)
    assert result['norm_execution_eligible'] and result['legacy_numerical_passed'] is False and result['legacy_headline_eligible'] is False
    assert before=={str(p):file_sha256(p) for p in path.iterdir()}


@pytest.mark.parametrize('fault',['uuid','early','partial','failure','fallback','oom','export','missing_codegen','runtime_identity',
                                  'measured_codegen','warmup','warmup_drift','headline','legacy_pass','gate_tamper','batch','work','summary'])
def test_execution_failures_reject_norm_qualification(tmp_path,monkeypatch,fault):
    path,cfg,v,p,c=block_fixture(tmp_path,monkeypatch)
    rows=[json.loads(x) for x in (path/'measurements.jsonl').read_text().splitlines()]
    if fault=='uuid':env=norm.read(path/'environment.json');env['gpu']['uuid']='other';put(path/'environment.json',env)
    elif fault=='early':config=norm.read(path/'config.json');config['started_at']='2026-01-01T00:00:00Z';put(path/'config.json',config)
    elif fault in ('warmup','warmup_drift'):
        cold=norm.read(path/'cold_start.json')
        if fault=='warmup':cold['warmups'].pop()
        else:cold['warmups'][-1]['compiler']['stats']['unique_graphs']=11
        put(path/'cold_start.json',cold)
    elif fault in ('headline','summary'):
        s=norm.read(path/'summary.json');s['headline_eligible' if fault=='headline' else 'successful_requests']=True if fault=='headline' else 49;put(path/'summary.json',s)
    elif fault in ('legacy_pass','gate_tamper'):
        gate=norm.read(path/'numerical_gate.json');gate['passed']=True if fault=='legacy_pass' else False;gate['forged']=True;put(path/'numerical_gate.json',gate)
    else:
        if fault=='partial':rows.pop()
        if fault=='failure':rows[0]['status']='failed'
        if fault=='fallback':rows[0]['fallbacks']=['eager']
        if fault=='oom':rows[0]['cache_stats']['oom_attempts']=['retried']
        if fault=='export':rows[0]['candidates'][0]['frames_path']='out.npy'
        if fault=='missing_codegen':rows[0]['cache_stats']['compiler_runtime']['kernels_created_by_runtime']=0
        if fault=='runtime_identity':rows[0]['cache_stats']['compiler_runtime']['runtime_implementation_sha256']='x'*64
        if fault=='measured_codegen':rows[0]['cache_stats']['compiler_runtime']['after']['generated_kernels']=21
        if fault=='batch':rows[0]['effective_config']['actual_batch_sizes']=[2]
        if fault=='work':rows[0]['denoiser_evaluations']=49
        (path/'measurements.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError):norm.validate_block(load_run(path),cfg,v,p,c)


def test_partial_companion_cannot_report_complete(tmp_path,monkeypatch):
    parent,pp,out,kw,*_=fixture(tmp_path,monkeypatch);norm.plan_companion(parent,out,**kw)
    with pytest.raises(FileNotFoundError):norm.verify_completed(out)
    assert not (out/'completed.json').exists()


def executed_fixture(tmp_path,monkeypatch,*,fail_slot=None,mutate_after=None):
    from datetime import datetime,timedelta,timezone
    ticks=[datetime(2026,1,4,tzinfo=timezone.utc)]
    def clock():
        ticks[0]+=timedelta(seconds=1);return ticks[0].isoformat()
    monkeypatch.setattr(norm.study,'now',clock)
    parent,pp,out,kw,cal,policy,env=fixture(tmp_path,monkeypatch)
    value=norm.plan_companion(parent,out,**kw);calls=[]
    def execute(directory,name,cfg,records):
        index=int(directory.name);calls.append(index)
        if index==fail_slot:raise RuntimeError('fabricated subprocess failure')
        path=directory/'attempts'/name/'attempt-000'/name
        source_cfg=copy.deepcopy(cfg);put(path.parent/'config.yaml',source_cfg)
        stored=copy.deepcopy(cfg);stored['started_at']=clock()
        meta=norm.read(cfg['data']['manifest']+'.meta.json')
        stored['data'].update(manifest_metadata=meta,manifest_sha256=file_sha256(cfg['data']['manifest']))
        candidate=name=='compiled';graphs=10 if candidate else 0
        identity=value['qualification']['model_identity'] if candidate else {**cfg['model']}
        runtime={'passed':True,'require_graphs':True,'restoration_complete':True,'runtime_implementation_sha256':'f'*64,
                 'graphs_created_by_runtime':10,'kernels_created_by_runtime':20,
                 'before':{'unique_graphs':10,'generated_kernels':20,'unimplemented':{}},
                 'after':{'unique_graphs':10,'generated_kernels':20,'unimplemented':{}}}
        rows=fake_rows(.8 if candidate else 1,index)
        for r in rows:
            if candidate:r['cache_stats']['compiler_runtime']=copy.deepcopy(runtime)
            r['effective_config']={**cfg['generation'],'optimizations':cfg['optimizations'],'requested_batch_size':1,'actual_batch_sizes':[1]}
            r['denoiser_evaluations']=50
        cold={'status':'ready','compilation_and_initialization_excluded':True,
              'warmups':[dict(copy.deepcopy(rows[i]),compiler={'stats':{'unique_graphs':graphs}}) for i in range(3)]}
        gate=norm.numerical_evidence(cfg)
        summary={'backend':cfg['model']['backend'],'synthetic':False,'profiled':False,'headline_eligible':False,'numerical_validation':gate,
                 'failures':0,'measured_requests':50,'successful_requests':50,'completed_futures':50,'oom_attempts':0,
                 'recompiled_during_measurement':False,'compiler_before':{'stats':{'unique_graphs':graphs}},'compiler_after':{'stats':{'unique_graphs':graphs}}}
        for key,obj in [('config.json',stored),('environment.json',env),('model_identity.json',identity),('cold_start.json',cold),
                        ('summary.json',summary),('numerical_gate.json',gate),('environment_gate.json',{'passed':True})]:put(path/key,obj)
        (path/'measurements.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        put(path.parent/'process.json',{'exit_code':0,'config_sha256':file_sha256(path.parent/'config.yaml'),'finished_at':clock()})
        receipt={'path':str(path),'adopted':False,'outputs':[],'artifact_sha256':norm.artifact_sha256(load_run(path)),
                 'generation_source_evidence':cfg['study_provenance']}
        put(directory/'completed'/(name+'.json'),receipt)
        if index==mutate_after:
            q=norm.read(kw['qualification']);q['altered_during_generation']=True;put(kw['qualification'],q)
        return receipt
    monkeypatch.setattr(norm.study,'_execute_run',execute)
    return out,calls,value


def test_eight_actual_artifact_blocks_reconstruct_without_legacy_relabel(tmp_path,monkeypatch):
    out,calls,value=executed_fixture(tmp_path,monkeypatch)
    result=norm.run_companion(out)
    assert calls==list(range(8))
    assert result['timing_qualified'] is True and result['combined_headline_claim'] is False
    assert result['legacy_strict_passed']=={'upstream':True,'compiled':False}
    assert result['legacy_headline_eligible']=={'upstream':False,'compiled':False}
    assert result['comparison']['paired_requests']==200
    assert all(norm.read(Path(b['path'])/'summary.json')['headline_eligible'] is False for b in result['blocks'])
    before={str(p):file_sha256(p) for p in out.rglob('*') if p.is_file()}
    assert norm.verify_completed(out)==result
    assert norm.run_companion(out)==result and calls==list(range(8))
    assert before=={str(p):file_sha256(p) for p in out.rglob('*') if p.is_file()}


def test_subprocess_failure_never_commits_complete_result(tmp_path,monkeypatch):
    out,calls,value=executed_fixture(tmp_path,monkeypatch,fail_slot=2)
    with pytest.raises(RuntimeError,match='subprocess failure'):norm.run_companion(out)
    assert calls==[0,1,2] and not (out/'completed.json').exists()
    assert len(list(out.glob('blocks/*/norm-completed.json')))==2
    with pytest.raises(FileNotFoundError):norm.verify_completed(out)


def test_post_block_qualification_mutation_stops_before_next_slot(tmp_path,monkeypatch):
    out,calls,value=executed_fixture(tmp_path,monkeypatch,mutate_after=0)
    with pytest.raises(ValueError,match='Qualification'):norm.run_companion(out)
    assert calls==[0] and not (out/'completed.json').exists()


@pytest.mark.parametrize('fault',['copied','overlap','process_failed','source_changed','gate_changed','completed_changed'])
def test_completed_timing_is_revalidated_not_trusted_flag(tmp_path,monkeypatch,fault):
    out,calls,value=executed_fixture(tmp_path,monkeypatch);result=norm.run_companion(out)
    b=result['blocks'][1];path=Path(b['path']);receipt=out/'blocks/001/norm-completed.json'
    if fault=='copied':(path/'measurements.jsonl').write_bytes((Path(result['blocks'][0]['path'])/'measurements.jsonl').read_bytes())
    elif fault=='overlap':
        p=out/'blocks/001/norm-started.json';s=norm.read(p);s['started_at']='2026-01-01T00:00:00Z';put(p,s)
        r=norm.read(receipt);r['start_receipt_sha256']=file_sha256(p);put(receipt,r)
    elif fault=='process_failed':p=path.parent/'process.json';s=norm.read(p);s['exit_code']=1;put(p,s)
    elif fault=='source_changed':
        p=out/'blocks/001/completed/compiled.json';s=norm.read(p);s['generation_source_evidence']['generation_sources']={};put(p,s)
    elif fault=='gate_changed':p=path/'numerical_gate.json';s=norm.read(p);s['passed']=True;put(p,s)
    else:p=out/'completed.json';s=norm.read(p);s['combined_headline_claim']=True;put(p,s)
    with pytest.raises(ValueError):norm.verify_completed(out)


def test_fresh_baseline_proof_reaches_canonical_gate(tmp_path,monkeypatch):
    parent,pp,out,kw,cal,*_=fixture(tmp_path,monkeypatch)
    # A replacement allocation proof differs from the original pilot proof.
    fresh=put(tmp_path/'fresh-strict.json',norm.read(kw['baseline_proof']))
    reached=[]
    def gate(cfg,*args):
        reached.append(cfg['numerical_validation'])
        assert cfg['numerical_validation']==str(fresh)
        return {'passed':True}
    monkeypatch.setattr(norm,'numerical_evidence',gate)
    value=norm.plan_companion(parent,out,**{**kw,'baseline_proof':str(fresh)})
    assert reached==[str(fresh)]
    assert value['configs']['upstream']['numerical_validation']==str(fresh)
    assert cal['configuration']['numerical_validation']!=str(fresh)


@pytest.mark.parametrize('key',['sha256','scene_ids'])
def test_qualification_must_bind_parent_development_partition(tmp_path,monkeypatch,key):
    parent,pp,out,kw,*_=fixture(tmp_path,monkeypatch)
    q=norm.read(kw['qualification']);q['development_manifest'][key]='different'
    put(kw['qualification'],q)
    with pytest.raises(ValueError,match='development partition'):norm.plan_companion(parent,out,**kw)


def test_bounded_invocations_complete_one_new_block_and_resume(tmp_path,monkeypatch):
    out,calls,value=executed_fixture(tmp_path,monkeypatch)
    for index in range(7):
        result=norm.run_companion(out,max_new_blocks=1)
        assert result['status']=='paused' and result['incomplete'] is True
        assert result['completed_blocks']==index+1 and result['new_blocks_this_invocation']==1
        assert result['timing_qualified'] is False and result['reason']=='max_new_blocks'
        assert calls==list(range(index+1)) and not (out/'completed.json').exists()
    result=norm.run_companion(out,max_new_blocks=1)
    assert result['timing_qualified'] is True and calls==list(range(8))
    assert norm.verify_completed(out)==result
    assert len(list((out/'pauses').glob('*.json')))==7


def test_deadline_admission_pauses_before_any_new_work(tmp_path,monkeypatch):
    out,calls,value=executed_fixture(tmp_path,monkeypatch)
    result=norm.run_companion(out,max_new_blocks=1,deadline='2026-01-04T01:00:00Z',block_budget_seconds=4000)
    assert result['status']=='paused' and result['reason']=='deadline_admission'
    assert result['completed_blocks']==0 and calls==[] and not (out/'blocks').exists()
    assert not (out/'completed.json').exists()


@pytest.mark.parametrize('kw',[{'max_new_blocks':0},{'max_new_blocks':True},{'max_new_blocks':1.5},
    {'deadline':'2026-01-04T01:00:00Z'},{'block_budget_seconds':1},
    {'deadline':'no timezone','block_budget_seconds':1},
    {'deadline':'2026-01-04T01:00:00Z','block_budget_seconds':float('inf')},
    {'deadline':'2026-01-04T01:00:00Z','block_budget_seconds':-1},{'cleanup_seconds':239}])
def test_invalid_admission_limits_reject_before_execution(tmp_path,monkeypatch,kw):
    out,calls,value=executed_fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError):norm.run_companion(out,**kw)
    assert calls==[]


def test_partial_failed_block_is_never_automatically_restarted(tmp_path,monkeypatch):
    out,calls,value=executed_fixture(tmp_path,monkeypatch,fail_slot=1)
    with pytest.raises(RuntimeError):norm.run_companion(out)
    with pytest.raises(ValueError,match='automatic retry'):norm.run_companion(out,max_new_blocks=1)
    assert calls==[0,1] and not (out/'completed.json').exists()


def test_resume_revalidates_completed_prefix_before_new_block(tmp_path,monkeypatch):
    out,calls,value=executed_fixture(tmp_path,monkeypatch)
    norm.run_companion(out,max_new_blocks=1)
    receipt=norm.read(out/'blocks/000/norm-completed.json')
    path=Path(receipt['path'])/'summary.json';s=norm.read(path);s['successful_requests']=49;put(path,s)
    with pytest.raises(ValueError):norm.run_companion(out,max_new_blocks=1)
    assert calls==[0]


def confirmation_fixture(monkeypatch):
    common={'parent_plan_sha256':'a','selection_sha256':'b','baseline':'a','candidate':'b','candidate_count':1,
            'workload':'single_future','runtime_identity':{'uuid':'same'},'source_identity':'source',
            'qualification':{'report_sha256':'qual'},'pilot_policy':{'sha256':'pilot'},'configs':{'a':{},'b':{}}}
    def result(prefix,start,regress):
        return {**copy.deepcopy(common),'comparison':{'status':'rerun_required' if regress else 'pass',
            'metrics':{k:{'exceeds_tolerance_and_uncertainty':k in regress} for k in ('p50','p95')}},
            'blocks':[{'path':prefix+str(i),'start_receipt':{'started_at':f'2026-01-{start:02d}T00:{i:02d}:00Z'},
                       'finished_at':f'2026-01-{start:02d}T00:{i:02d}:30Z'} for i in range(8)]}
    a,b=result('primary',4,['p50']),result('confirmation',5,['p50'])
    monkeypatch.setattr(norm,'verify_completed',lambda p: a if p=='primary' else b)
    monkeypatch.setattr(norm,'load_run',lambda p:p)
    monkeypatch.setattr(norm,'timing_signature',lambda p:p)
    return a,b


def test_confirmation_requires_repeated_same_metric(monkeypatch):
    a,b=confirmation_fixture(monkeypatch)
    result=norm.confirm_completed('primary','confirmation')
    assert result['status']=='regression_confirmed' and result['repeated_metrics']==['p50']
    assert result['automatic_measurement_performed'] is False and result['combined_headline_claim'] is False
    b['comparison']['metrics']['p50']['exceeds_tolerance_and_uncertainty']=False
    b['comparison']['metrics']['p95']['exceeds_tolerance_and_uncertainty']=True
    assert norm.confirm_completed('primary','confirmation')['status']=='not_confirmed'


@pytest.mark.parametrize('fault',['no_regression','runtime','policy','qualification','implementation','copied','chronology'])
def test_confirmation_rejects_changed_or_nonindependent_evidence(monkeypatch,fault):
    a,b=confirmation_fixture(monkeypatch)
    if fault=='no_regression':a['comparison']['status']='pass'
    elif fault=='runtime':b['runtime_identity']['uuid']='other'
    elif fault=='policy':b['pilot_policy']['sha256']='other'
    elif fault=='qualification':b['qualification']['report_sha256']='other'
    elif fault=='implementation':b['configs']['b']['model']={'different':True}
    elif fault=='copied':b['blocks'][0]['path']=a['blocks'][0]['path']
    else:b['blocks'][0]['start_receipt']['started_at']='2026-01-03T00:00:00Z'
    with pytest.raises(ValueError):norm.confirm_completed('primary','confirmation')


def test_strict_baseline_uses_real_numerical_evidence_and_retained_hashes(tmp_path):
    # Reuse the canonical gate's real filesystem fixture, without replacing the gate.
    spec=importlib.util.spec_from_file_location('canonical_benchmark_fixture',Path(__file__).with_name('test_benchmark.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cfg,gen,opts,env,identity,records,proof,path=module._numerical_fixture(tmp_path)
    cfg['model']['backend']='upstream';identity['backend']='upstream'
    env.update(python='fixture',packages={'torch':'fixture'},tools={'ncu':{'stdout':'fixture'}})
    cfg.update(generation=gen.to_dict(),optimizations=opts.to_dict())
    proof.update(atol=0,rtol=0,runtime_identity=norm.pilot._runtime(env))
    put(path,proof)
    cfg['environment_verification']=str(put(tmp_path/'verified-env.json',{'environment':env}))
    calibration={'identity':{'runtime':norm.pilot._runtime(env)}}
    assert norm.strict_baseline(path,cfg,calibration)['passed'] is True
    Path(proof['runs']['upstream']['outputs']['normal']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='Bound artifact changed'):norm.strict_baseline(path,cfg,calibration)


def test_launch_configuration_cannot_differ_from_selected_block(tmp_path,monkeypatch):
    out,calls,value=executed_fixture(tmp_path,monkeypatch)
    result=norm.run_companion(out)
    run=Path(result['blocks'][0]['path']);launch=run.parent/'config.yaml';cfg=norm.read(launch)
    cfg['model']['upstream_commit']='c'*40;put(launch,cfg)
    process_path=run.parent/'process.json';p=norm.read(process_path);p['config_sha256']=file_sha256(launch);put(process_path,p)
    with pytest.raises(ValueError,match='launch config changed'):norm.verify_completed(out)


def test_canonical_artifact_map_contains_relative_filenames(tmp_path,monkeypatch):
    out,calls,value=executed_fixture(tmp_path,monkeypatch)
    norm.run_companion(out,max_new_blocks=1)
    receipt=norm.read(out/'blocks/000/norm-completed.json');run=load_run(receipt['path'])
    assert set(norm.artifact_sha256(run))=={'config.json','environment.json','measurements.jsonl','summary.json'}
    assert set(receipt['artifact_sha256'])==set(norm.artifact_sha256(run))|{'cold_start.json','model_identity.json','environment_gate.json','numerical_gate.json'}
    assert all(file_sha256(run.path/key)==sha for key,sha in receipt['artifact_sha256'].items())


def test_documented_full_qualification_counts_for_bounded_resume(tmp_path,monkeypatch):
    out,calls,value=executed_fixture(tmp_path,monkeypatch)
    original=norm.module;counts=[0]
    def module(name):
        result=original(name)
        if name!='qualify_compiler_numerics.py':return result
        def verify(path):
            counts[0]+=1
            return result.verify(path)
        return SimpleNamespace(verify=verify)
    monkeypatch.setattr(norm,'module',module)
    observed=[]
    for _ in range(8):
        before=counts[0];norm.run_companion(out,max_new_blocks=1);observed.append(counts[0]-before)
    assert observed==[4,5,6,7,8,9,10,14]
