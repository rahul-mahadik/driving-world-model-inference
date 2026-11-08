"""Tiny fabricated CPU evidence exercises contracts, never GPU/model claims."""
import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import yaml

from scripts import qualify_compiler_numerics as q
from drivebench.config import digest, file_sha256
from drivebench.data.manifests import write_manifest, inference_request
from drivebench.data.preprocess import read_preprocess
from drivebench.evaluation.io import artifact_sha256, load_run
from drivebench.evaluation.metrics import compare_videos
from drivebench.freeze import freeze_tolerances
from drivebench.types import GenerationConfig, OptimizationConfig
from drivebench.calibrate_pilot import _sources


def put(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)); return path


def bound(path):
    return {'path':str(path),'sha256':file_sha256(path)}


def fixture(tmp, monkeypatch, cache=False, badclip=False):
    monkeypatch.setattr(q,'PRIMARY',{**q.PRIMARY,'height':8,'width':8})
    generation=GenerationConfig(height=8,width=8).to_dict()
    options=OptimizationConfig(compile_denoiser=True,conditioning_cache_bytes=2048 if cache else 0).to_dict()
    image=tmp/'frame.png'; Image.new('RGB',(8,8),(80,100,120)).save(image)
    records=[{'request_id':f'request-{i:02}', 'scene_id':f'scene-{i:02}','dataset':'nuscenes',
              'observed_frames':['frame.png'],'frame_checksums':{'frame.png':file_sha256(image)},
              'trajectory':[0,1,0,2,0,3,0,4], 'trajectory_metadata':{'units':'meters','coordinate_frame':'fixture',
              'axis_order':['lateral','longitudinal'],'timestamps_seconds':[.5,1,1.5,2],
              'provenance':'FABRICATED CPU FIXTURE','verified_against_annotation_source':True}}
             for i in range(32)]
    manifest=tmp/'development.jsonl';write_manifest(manifest,records,{'partition_role':'development'})
    metadata=json.loads(manifest.with_suffix('.jsonl.meta.json').read_text())
    environment={'gpu':{'uuid':'CPU-FIXTURE-NOT-HARDWARE','name':'fixture'},'python':'fixture','packages':{}}
    runtime=q.reference_identity(environment)
    env=put(tmp/'environment-proof.json',{'passed':True,'environment':environment})
    lock=put(tmp/'lock.json',runtime)
    checkpoint=tmp/'weights';checkpoint.write_bytes(b'CPU fixture weights')
    inference=tmp/'inference.yaml';inference.write_text('fixture')
    compiler={'name':'registered_einops_attention_v1','suppress_errors':False,'cache_size_limit':512,
              'native_operator_unchanged':True,'source_sha256':{'drivebench/runtime/compiler.py':file_sha256(q.ROOT/'drivebench/runtime/compiler.py')}}
    compiler['implementation_sha256']=digest(compiler)
    identity={'backend':'vista','upstream_commit':'commit-fixture','checkpoint_sha256':file_sha256(checkpoint),
              'inference_config_sha256':file_sha256(inference),'compiler_runtime':compiler}
    model={'backend':'vista','upstream_dir':str(tmp),'upstream_commit':'commit-fixture','checkpoint':str(checkpoint),
           'checkpoint_sha256':file_sha256(checkpoint),'inference_config':str(inference),
           'compiler_runtime':'registered_einops_attention_v1','compiler_feature_boundary':None}
    config={'model':model,'generation':generation,'optimizations':options,'data':{'root':str(tmp),'manifest':str(manifest),'verify_checksums':True},
            'environment_verification':str(env),'environment_lock':str(lock),'benchmark':{'candidate_count':1,'save_outputs':32}}
    now=datetime.now(timezone.utc); early=(now-timedelta(minutes=5)).isoformat()
    config['started_at']=early
    req=inference_request(records[0],tmp,2);ids=req['candidate_ids'];frames=read_preprocess(req['observed_frames'],8,8)
    provenance=q.input_provenance(manifest,records,records[0],req,frames,request_index=0)
    proof={'schema_version':2,'mode':'all','created_at':early,'implementation_sha256':q.SOURCE,'config':config,
           'generation':generation,'optimizations':options,'environment':environment,'runtime_identity':runtime,
           'input_provenance':provenance,'input_provenance_sha256':digest(provenance),'validated_request_count':1,
           'atol':0,'rtol':0,'runs':{},'checks':{}}
    base=np.broadcast_to(np.linspace(.1,.8,25,dtype=np.float32)[:,None,None,None],(25,8,8,3)).copy()
    compiled=base+.002
    originals={key:base.copy() for key in ids};normals={key:compiled.copy() for key in ids}
    values={'upstream':{'normal':originals},'refactored':{'normal':normals},
            'isolation':{'normal':normals,'reverse':dict(reversed(list(normals.items()))),'other_candidates_after_action_change':{ids[1]:normals[ids[1]]}},
            'uncached':{'normal':normals},'cached':{'normal':normals}}
    stats={'oom_attempts':[],'denoiser_evaluations_complete':True,'compiler_runtime':{'passed':True,'restoration_complete':True,
            'require_graphs':True,'runtime_implementation_sha256':compiler['implementation_sha256'],
            'graphs_created_by_runtime':1,'kernels_created_by_runtime':1,
            'before':{'unimplemented':{},'unique_graphs':0,'generated_kernels':0},
            'after':{'unimplemented':{},'unique_graphs':1,'generated_kernels':1}},
            'conditioning_embedding_hits':2,'posterior_parameter_hits':2,'conditioning_cached_embedders':['cond_frames','cond_frames_without_noise']}
    for name,kinds in values.items():
        opts=copy.deepcopy(options)
        mid=copy.deepcopy(identity)
        if name=='upstream':opts=OptimizationConfig().to_dict();mid['backend']='upstream';mid.pop('compiler_runtime')
        if name=='uncached':opts['conditioning_cache_bytes']=0
        if name=='cached':opts['conditioning_cache_bytes']=2048
        effective={**generation,'optimizations':opts,'candidate_count':2,'actual_batch_sizes':[1,1],'requested_batch_size':1}
        details={'model_identity':mid,'effective_config':effective,'candidate_count':2,'fallbacks':[],
                 'denoiser_evaluations':100,'cache_stats':copy.deepcopy(stats),'outputs':{},'outputs_sha256':{},'executions':{}}
        for kind,arrays in kinds.items():
            path=tmp/'proof-arrays'/f'{name}-{kind}.npz';path.parent.mkdir(exist_ok=True)
            np.savez(path,**arrays);details['outputs'][kind]=str(path);details['outputs_sha256'][kind]=file_sha256(path)
        for label in ('normal','reverse','action_change') if name=='isolation' else ('normal',):
            details['executions'][label]={'status':'ok',**{k:copy.deepcopy(details[k]) for k in ('effective_config','fallbacks','denoiser_evaluations','cache_stats')}}
        if name=='cached':details['cache_key_checks']={k:True for k in ('unchanged_input_hit','unchanged_key_equal','changed_input_missed','changed_key_differs')}
        proof['runs'][name]=details
    refresh_checks(proof)
    proof_path=put(tmp/'strict.json',proof)
    study=tmp/'study'; gen_sources=q.generation_sources()
    plan={'reference':'upstream','optimized_reference':'compiled-conditioning','base_config':config,'raw_output_policy':'fixture-retain-all',
          'generation_sources_at_plan':gen_sources,'manifests':{'development':{'path':str(manifest),'sha256':file_sha256(manifest),'metadata':metadata}}}
    plan_path=put(study/'plan.json',plan)
    source_runs={}; aggregate_runs={};outputs={};pairrows=[]
    for name in ('upstream','compiled-default'):
        src=tmp/'source-runs'/name;rows=[];receipt_outputs=[]
        cfg=copy.deepcopy(config);cfg['data'].update(manifest_sha256=file_sha256(manifest),manifest_metadata=metadata)
        cfg['study_provenance']={'generation_sources':gen_sources}
        mid=copy.deepcopy(identity)
        if name=='upstream':
            cfg['model']={**model,'backend':'upstream'};cfg['model'].pop('compiler_runtime');cfg['model'].pop('compiler_feature_boundary')
            cfg['optimizations']=OptimizationConfig().to_dict();mid['backend']='upstream';mid.pop('compiler_runtime')
        for i,record in enumerate(records):
            request=inference_request(record,tmp,1);candidate=request['candidate_ids'][0]
            array=base.copy() if name=='upstream' else base+np.float32(.03 if badclip and i==31 else .001+i*.00002)
            path=src/f'outputs/{i}.npy';path.parent.mkdir(parents=True,exist_ok=True);np.save(path,array)
            outputs[(name,i)]=(path,file_sha256(path),array)
            action_hash=digest(np.asarray(request['trajectories'][0]).tolist())
            rows.append({'request_id':record['request_id'],'scene_id':record['scene_id'],'status':'ok','repeat':0,'request_index':i,
                         'candidates':[{'candidate_id':candidate,'seed':request['seeds'][0],'trajectory_sha256':action_hash,'frames_path':f'outputs/{i}.npy'}]})
            receipt_outputs.append({'path':str(path),'sha256':file_sha256(path),'bytes':path.stat().st_size,'request_id':record['request_id'],'candidate_id':candidate})
        for filename,value in [('config.json',cfg),('environment.json',environment),('summary.json',{'backend':mid['backend'],'failures':0,'status':'ok'}),('model_identity.json',mid)]:put(src/filename,value)
        (src/'measurements.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        receipt={'path':str(src),'artifact_sha256':artifact_sha256(load_run(src)),'outputs':receipt_outputs,'adopted':False}
        source_runs[name]=receipt
        agg=study/'quality/development/aggregate/runs'/name
        acfg=copy.deepcopy(cfg);acfg['study_aggregation']={'source_runs':[receipt]}
        for filename,value in [('config.json',acfg),('environment.json',environment),('summary.json',{'backend':mid['backend'],'failures':0,'status':'ok'})]:put(agg/filename,value)
        (agg/'measurements.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));aggregate_runs[name]=load_run(agg)
    for i,record in enumerate(records):
        rr=aggregate_runs['compiled-default'].measurements[i];cc=rr['candidates'][0]
        metrics=compare_videos(outputs['upstream',i][2][1:],outputs['compiled-default',i][2][1:]);metrics['lpips']=.001
        pairrows.append({'request_id':record['request_id'],'scene_id':record['scene_id'],'repeat':0,'candidate_id':cc['candidate_id'],'seed':cc['seed'],
                         'trajectory_sha256':cc['trajectory_sha256'],'reference_output_sha256':outputs['upstream',i][1],
                         'candidate_output_sha256':outputs['compiled-default',i][1],'metrics':metrics})
    entry={'run':'compiled-default','run_path':str(aggregate_runs['compiled-default'].path),'artifact_sha256':artifact_sha256(aggregate_runs['compiled-default']),
           'synthetic':False,'complete_pairs':True,'errors':[],'matched_pairs':32,'pairs':pairrows,
           'aggregates':{m:{'max':max(p['metrics'][m] for p in pairrows),'pairs':32} for m in q.QUALITY}}
    evaluator_sources={'evaluate.py':file_sha256(q.ROOT/'drivebench/evaluate.py')}
    perceptual={'name':'LPIPS/alex','status':'available','provenance':{'fixture':True}}
    shard=study/'quality/development/shards/000';metric=shard/'metrics/upstream--compiled-default.json'
    metric_policy=tmp/'metric-policy.yaml';metric_policy.write_text('fixture')
    metric_document={'reference':source_runs['upstream']['path'],'reference_artifact_sha256':source_runs['upstream']['artifact_sha256'],
                     'runs':[{**entry,'artifact_sha256':source_runs['compiled-default']['artifact_sha256']}],
                     'evaluator_source_sha256':evaluator_sources,'perceptual':perceptual,'tolerances_path':str(metric_policy)}
    put(metric,metric_document)
    metric_receipt=metric.with_suffix('.receipt.json')
    put(metric_receipt,{'evaluation_sha256':file_sha256(metric),'reference_artifact_sha256':source_runs['upstream']['artifact_sha256'],
                       'candidate_artifact_sha256':source_runs['compiled-default']['artifact_sha256'],'raw_output_policy':plan['raw_output_policy'],
                       'evaluator_source_sha256':evaluator_sources,'tolerances_sha256':file_sha256(metric_policy),'review_artifact_sha256':{}})
    put(shard/'complete.json',{'runs':source_runs,'metric_artifacts':{str(metric):file_sha256(metric),str(metric_receipt):file_sha256(metric_receipt)}})
    entry['metric_artifacts']=[bound(metric)]
    document={'reference':str(aggregate_runs['upstream'].path),'reference_artifact_sha256':artifact_sha256(aggregate_runs['upstream']),
              'perceptual':perceptual,'evaluator_source_sha256':evaluator_sources,'runs':[entry]}
    evaluation=put(study/'quality/development/aggregate/evaluation-vs-upstream.json',document)
    put(study/'quality/development/aggregation-receipt.json',{'results':{'upstream':str(evaluation)},
            'artifact_sha256':{str(p):file_sha256(p) for p in evaluation.parent.rglob('*') if p.is_file()}})
    draft=tmp/'quality-draft.yaml';draft.write_text(yaml.safe_dump({'metrics':{m:{'maximum':v,'aggregation':'max'} for m,v in q.QUALITY.items()}}))
    quality=tmp/'quality-frozen.yaml';freeze_tolerances(evaluation,draft,quality)
    component_dir=tmp/'component';put(component_dir/'environment.json',environment)
    latent=[50,4,1,1]
    shapes={'x':latent,'condition.concat':latent,'sigma':[50],'mask':[50],'condition.crossattn':[50,1,3456],'condition.vector':[50,768]}
    inputs={k:{'sha256':'a'*64,'shape':shape,'dtype':'torch.float32','stride':[1]*len(shape),'device_type':'cuda'} for k,shape in shapes.items()}
    component_config=copy.deepcopy(config);component_config['optimizations']['conditioning_cache_bytes']=0
    component={'status':'diagnostic_complete_strict_failed','exit_code':2,'implementation_sha256':q.SOURCE,'source_unchanged':True,'bound_files_unchanged':True,'model_identity':identity,
               'config':component_config,'input_provenance':provenance,'environment_gate':{'passed':True,'verification_sha256':file_sha256(env)},
               'bound_file_sha256':{str(env):file_sha256(env)},'diagnostic':{'capture_count':1,'input_tensors_reused_without_resampling':True,
               'same_backend_and_ema_scope':True,'callable_restored_after_capture':True,
               'candidate_count':1,'denoiser_evaluations':2,'sampler_rollouts_completed':0,'video_decoder_calls':0,
               'autocast_policy':'fp16; actual input/output dtypes recorded','cfg_order':'25 unconditional frames followed by 25 conditional frames',
               'executions':[{'label':label,'input_hashes_before':inputs,'input_hashes_after':inputs,'rng_state_unchanged':True,
                              'shape':latent,'dtype':'float32','output_sha256':'b'*64,
                              'rng_state_before':{'cpu':'a'*64,'cuda':'b'*64},'rng_state_after':{'cpu':'a'*64,'cuda':'b'*64}} for label in ('eager','compiled')],
               'comparison':{'rmse':.001,'max_absolute_error':.05},'compiled_runtime':copy.deepcopy(stats['compiler_runtime'])}}
    component_path=put(component_dir/'compiler.json',component)
    frozen=datetime.now(timezone.utc).isoformat()
    policy={'type':q.POLICY_TYPE,'status':'frozen','created_at':frozen,'frozen_at':frozen,'source_identity':q.SOURCE,'held_out_not_started_at_freeze':True,
            'numerical':q.NUMERICAL,'review_rule':q.REVIEW_RULE,'study':bound(plan_path),'variant':'compiled-default',
            'development_evaluation':bound(evaluation),'quality_policy':bound(quality),'component_evidence':bound(component_path)}
    policy_path=put(tmp/'norm-policy.json',policy)
    numeric=q._raw_checks(proof,generation,options,ids)[0]
    dev_errors={r['request_id']:q.rgb_errors(outputs['upstream',i][2],outputs['compiled-default',i][2],1) for i,r in enumerate(records)}
    cases=q.review_cases(pairrows,proof,numeric,dev_errors)
    review={'type':'compiler_norm_localized_review_v1','created_at':datetime.now(timezone.utc).isoformat(),'policy_sha256':file_sha256(policy_path),
            'proof_sha256':file_sha256(proof_path),'rule':q.REVIEW_RULE,'held_out_consulted':False,'reviewer_kind':'agent_assisted','cases':[]}
    for case,binding in cases.items():
        review['cases'].append({'case_id':case,**binding,'frames_reviewed':list(range(1,25)), 'ordered_atlas_frames_reviewed':True,
             'native_error_frames_reviewed':q.REVIEW_RULE['native_error_frames'],'observations':'Fabricated fixture only', 'new_localized_failure':False,
             'failure_categories':{k:False for k in q.REVIEW_RULE['failure_categories']},'decision':'accept','preference':'tie','assets':[bound(image)]})
    review_path=put(tmp/'review.json',review)
    return {'proof':proof,'proof_path':proof_path,'policy':policy,'policy_path':policy_path,'review':review,'review_path':review_path,
            'output':tmp/'qualification.json','study':study,'generation':generation,'options':options,'ids':ids,'entry':entry,
            'evaluation':evaluation,'quality':quality,'component':component_path,'records':records,'outputs':outputs}


def refresh_checks(proof):
    g=proof['generation'];ids=proof['input_provenance']['candidate_ids'];r=proof['runs']
    a=lambda name,kind='normal',keys=ids:q.arrays(r[name]['outputs'][kind],keys,g)
    proof['checks']={'upstream_agreement':q.compare_outputs(a('upstream'),a('refactored'),atol=0,rtol=0),
                     'candidate_order':q.compare_outputs(a('isolation'),a('isolation','reverse'),atol=0,rtol=0),
                     'unrelated_candidate_action_change':q.compare_outputs({ids[1]:a('isolation')[ids[1]]},a('isolation','other_candidates_after_action_change',ids[1:]),atol=0,rtol=0),
                     'conditioning_cache_agreement':q.compare_outputs(a('uncached'),a('cached'),atol=0,rtol=0),
                     'conditioning_cache_keys':{**r['cached']['cache_key_checks'],'passed':all(r['cached']['cache_key_checks'].values())},
                     'conditioning_cache_exercised':q.cache_coverage(r['cached'])}
    proof['passed']=all(x['passed'] for x in proof['checks'].values())


def change(f,run,kind='normal',amount=.001,index=(1,0,0,0)):
    details=f['proof']['runs'][run];path=Path(details['outputs'][kind]);ids=f['ids']
    with np.load(path) as source:arr={k:source[k].copy() for k in source.files}
    arr[ids[-1]][index]+=amount;np.savez(path,**arr);details['outputs_sha256'][kind]=file_sha256(path)
    refresh_checks(f['proof']);put(f['proof_path'],f['proof'])
    f['review']['proof_sha256']=file_sha256(f['proof_path'])
    for row in f['review']['cases']:
        if row['case_id'].startswith('selected-proof:'):
            row['reference_output_sha256']=f['proof']['runs']['upstream']['outputs_sha256']['normal']
            row['candidate_output_sha256']=f['proof']['runs']['refactored']['outputs_sha256']['normal']
    put(f['review_path'],f['review'])


def create(f):return q.create(f['proof_path'],f['policy_path'],f['review_path'],f['output'])


def test_complete_cpu_contract_preserves_failed_legacy_proof(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);before=file_sha256(f['proof_path']);result=create(f)
    assert result['qualified'] and result['legacy_strict_passed'] is False
    assert result['quality_development']['raw_arrays_recomputed'] and len(result['quality_development']['numerical_by_request'])==32
    assert q.verify(f['output'])['report_sha256']==file_sha256(f['output'])
    assert file_sha256(f['proof_path'])==before
    with pytest.raises(ValueError,match='overwritten'):create(f)


def test_unused_cache_failure_does_not_block_default(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);change(f,'cached')
    result=create(f)
    assert result['qualified'] and not result['optional_cache_diagnostic']['agreement']['passed']
    assert not result['optional_cache_diagnostic']['required']


@pytest.mark.parametrize('run,kind',[('isolation','reverse'),('isolation','other_candidates_after_action_change'),('cached','normal')])
def test_exact_semantics_cannot_borrow_rgb_budget(tmp_path,monkeypatch,run,kind):
    f=fixture(tmp_path,monkeypatch,cache=True);change(f,run,kind,amount=.00001)
    result=create(f);assert not result['qualified'] and not result['exact_semantics']['passed']
    with pytest.raises(ValueError,match='failed'):q.verify(f['output'])


@pytest.mark.parametrize('fault',['fallback','batch','oom','calls','compiler','cache_budget','mode','runtime','source','nonzero',
                                'backend','manifest','synthetic','count','upstream_status','check_coverage','cache_claim'])
def test_execution_and_identity_fail_closed(tmp_path,monkeypatch,fault):
    f=fixture(tmp_path,monkeypatch);p=f['proof'];d=p['runs']['isolation']
    if fault=='fallback':d['fallbacks']=['serial fallback']
    elif fault=='batch':d['effective_config']['actual_batch_sizes']=[2]
    elif fault=='oom':d['cache_stats']['oom_attempts']=[{'retried':True}]
    elif fault=='calls':d['denoiser_evaluations']=99
    elif fault=='compiler':d['model_identity']['compiler_runtime']['name']='different'
    elif fault=='cache_budget':p['runs']['refactored']['effective_config']['optimizations']['conditioning_cache_bytes']=1
    elif fault=='mode':p['mode']='agreement'
    elif fault=='runtime':p['runtime_identity']['gpu']['uuid']='different'
    elif fault=='source':p['implementation_sha256']='0'*64
    elif fault=='nonzero':p['atol']=.6
    elif fault=='backend':p['runs']['refactored']['model_identity']['backend']='upstream'
    elif fault=='manifest':p['config']['data']['manifest']=str(tmp_path/'different.jsonl')
    elif fault=='synthetic':p['input_provenance']['synthetic']=True
    elif fault=='count':d['candidate_count']=1
    elif fault=='upstream_status':p['runs']['upstream']['executions']['normal']['status']='failed'
    elif fault=='check_coverage':p['checks'].pop('conditioning_cache_keys')
    else:p['checks']['conditioning_cache_keys']['changed_input_missed']=False
    put(f['proof_path'],p)
    with pytest.raises(ValueError):create(f)


@pytest.mark.parametrize('fault',['nan','inf','range','dtype','shape','ids'])
def test_raw_output_validation(tmp_path,fault):
    g={'total_frames':25,'height':2,'width':2};a=np.zeros((25,2,2,3),np.float32)
    if fault=='nan':a.flat[0]=np.nan
    elif fault=='inf':a.flat[0]=np.inf
    elif fault=='range':a.flat[0]=1.1
    elif fault=='dtype':a=a.astype(np.float64)
    elif fault=='shape':a=a[:-1]
    p=tmp_path/'arrays.npz';np.savez(p,**{'wrong' if fault=='ids' else 'a':a})
    with pytest.raises(ValueError):q.arrays(p,['a'],g)


def test_generated_frame_budget_not_pooled_or_diluted():
    a=np.zeros((25,2,2,3),np.float32);b=a.copy();b[0]=1
    assert q.rgb_errors(a,b,1)['rmse']==0
    b[1:]=.011
    assert not q.rgb_errors(a,b,1)['passed']
    b[1:]=0;b[7,1,0,2]=.5
    out=q.rgb_errors(a,b,1)
    assert out['max_abs_location_THWC']==[7,1,0,2] and out['per_frame'][6]['max_abs_error']==.5


@pytest.mark.parametrize('fault',['localized','category','uncertain','missing','native','frame'])
def test_localized_review_veto_and_coverage(tmp_path,monkeypatch,fault):
    f=fixture(tmp_path,monkeypatch);r=f['review'];case=r['cases'][0]
    if fault=='localized':case['new_localized_failure']=True
    elif fault=='category':case['failure_categories']['disappearance']=True
    elif fault=='uncertain':case['decision']='uncertain'
    elif fault=='missing':r['cases'].pop()
    elif fault=='native':case['native_error_frames_reviewed']=[]
    else:case['frames_reviewed']=list(range(1,24))
    put(f['review_path'],r)
    if fault in ('missing','native','frame'):
        with pytest.raises(ValueError):create(f)
    else:assert not create(f)['qualified']


def test_no_retrospective_qualification_and_later_heldout_allowed(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);create(f)
    path=f['study']/'quality/evaluation/attempt/config.json'
    put(path,{'started_at':datetime.now(timezone.utc).isoformat()})
    assert q.verify(f['output'])['qualified']
    f['output'].unlink()
    with pytest.raises(ValueError,match='Held-out'):create(f)


def test_changed_bound_artifact_never_reuses_declared_qualification(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);create(f)
    path=f['outputs']['compiled-default',7][0];a=np.load(path);a.flat[0]+=.001;np.save(path,a)
    with pytest.raises(ValueError,match='checksum'):q.verify(f['output'])


def test_unfrozen_and_fitted_policy_rejected(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);p=f['policy'];p['numerical']={**p['numerical'],'maximum':.6};put(f['policy_path'],p)
    with pytest.raises(ValueError,match='frozen norm policy'):create(f)


def test_review_selection_fixed_and_worst_even_below_threshold(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);numeric=q._raw_checks(f['proof'],f['generation'],f['options'],f['ids'])[0]
    errors={r['request_id']:q.rgb_errors(f['outputs']['upstream',i][2],f['outputs']['compiled-default',i][2],1) for i,r in enumerate(f['records'])}
    selected=q.review_cases(f['entry']['pairs'],f['proof'],numeric,errors)
    assert len(selected)>=6
    p=f['entry']['pairs'][-1];key='development:'+digest(p['request_id']+p['candidate_id'])
    assert key in selected


def test_one_bad_development_clip_cannot_hide_in_average(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch,badclip=True);result=create(f)
    assert not result['qualified'] and not result['quality_development']['passed']
    assert result['quality_development']['metrics_maximum']['rmse']>.01


@pytest.mark.parametrize('fault',['input_mutation','rng','status','dependency','runtime','source','output_shape','output_dtype','output_hash',
                                'input_shape','autocast','duplicate_execution','rng_hash'])
def test_component_evidence_is_diagnostic_but_mandatory(tmp_path,monkeypatch,fault):
    f=fixture(tmp_path,monkeypatch);value=q.read(f['component'])
    if fault=='input_mutation':value['diagnostic']['executions'][1]['input_hashes_after']['x']['sha256']='b'*64
    elif fault=='rng':value['diagnostic']['executions'][1]['rng_state_unchanged']=False
    elif fault=='status':value['status']='failed'
    elif fault=='dependency':value['bound_file_sha256']={}
    elif fault=='runtime':value['model_identity']['compiler_runtime']['name']='different'
    elif fault=='source':value['implementation_sha256']='b'*64
    elif fault=='output_shape':value['diagnostic']['executions'][0]['shape']=[1,4,1,1]
    elif fault=='output_dtype':value['diagnostic']['executions'][0]['dtype']='float16'
    elif fault=='output_hash':value['diagnostic']['executions'][0].pop('output_sha256')
    elif fault=='input_shape':value['diagnostic']['executions'][0]['input_hashes_before']['x']['shape']=[2]
    elif fault=='autocast':value['diagnostic']['autocast_policy']='bf16'
    elif fault=='duplicate_execution':value['diagnostic']['executions'].append(value['diagnostic']['executions'][0])
    else:value['diagnostic']['executions'][0]['rng_state_before']['cpu']='f'*64
    put(f['component'],value);f['policy']['component_evidence']=bound(f['component']);put(f['policy_path'],f['policy'])
    with pytest.raises(ValueError):create(f)


def test_policy_cannot_be_created_after_heldout_attempt(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch)
    put(f['study']/'quality/evaluation_second_seed/failed/config.json',{'started_at':f['proof']['created_at']})
    with pytest.raises(ValueError):create(f)


def test_report_timestamp_cannot_be_retroactively_backdated(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);result=create(f)
    result['created_at']=f['proof']['created_at'];put(f['output'],result)
    with pytest.raises(ValueError,match='predates'):q.verify(f['output'])


def test_declared_metric_policy_cannot_exceed_lpips_budget(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);value=yaml.safe_load(f['quality'].read_text());value['metrics']['lpips']['maximum']=.03
    f['quality'].write_text(yaml.safe_dump(value));f['policy']['quality_policy']=bound(f['quality']);put(f['policy_path'],f['policy'])
    with pytest.raises(ValueError,match='exceeds'):create(f)


def test_changed_main_pair_identity_rejects_even_with_new_aggregate_hash(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);value=q.read(f['evaluation']);value['runs'][0]['pairs'][0]['seed']+=1;put(f['evaluation'],value)
    f['policy']['development_evaluation']=bound(f['evaluation']);put(f['policy_path'],f['policy'])
    with pytest.raises(ValueError):create(f)


@pytest.mark.parametrize('fault',['missing_arrays','lpips_metric','shard_receipt','manifest_action','component_shape'])
def test_durable_development_and_component_dependencies(tmp_path,monkeypatch,fault):
    f=fixture(tmp_path,monkeypatch)
    if fault=='missing_arrays':f['outputs']['compiled-default',12][0].unlink()
    elif fault=='lpips_metric':
        path=f['study']/'quality/development/shards/000/metrics/upstream--compiled-default.json'
        value=q.read(path);value['runs'][0]['pairs'][0]['metrics']['lpips']=0;put(path,value)
    elif fault=='shard_receipt':
        path=f['study']/'quality/development/shards/000/complete.json'
        value=q.read(path);value['runs']['compiled-default']['outputs'].pop();put(path,value)
    elif fault=='manifest_action':
        f['records'][7]['trajectory'][0]=.2
        manifest=Path(f['proof']['input_provenance']['manifest_path'])
        # Even a complete new manifest metadata/hash cannot replace frozen source evidence.
        write_manifest(manifest,f['records'],{'partition_role':'development'})
    else:
        value=q.read(f['component']);value['config']['generation']['steps']=40;put(f['component'],value)
        f['policy']['component_evidence']=bound(f['component']);put(f['policy_path'],f['policy'])
    with pytest.raises((ValueError,FileNotFoundError)):create(f)


def test_unchanged_manifest_but_forged_development_action_cannot_match(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch)
    # Exercise the manifest join directly after earlier artifact validators, so
    # no amount of coherently rewritten run/aggregate hashes could waive it.
    original=q.candidates
    def forged(run):
        result=copy.deepcopy(original(run));key=next(iter(result))
        result[key][1]['trajectory_sha256']='0'*64
        return result
    monkeypatch.setattr(q,'candidates',forged)
    with pytest.raises(ValueError,match='input/output'):create(f)


def test_private_review_plan_is_unrated_and_binds_exact_native_indices(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);path=tmp_path/'review-plan.json'
    value=q.plan_review(f['proof_path'],f['policy_path'],path)
    assert value['qualified'] is False and value['status']=='unrated_private_export_plan'
    assert path.stat().st_mode & 0o777 == 0o600
    assert set(value['cases'])=={row['case_id'] for row in f['review']['cases']}
    for row in f['review']['cases']:
        assert value['cases'][row['case_id']]['native_frame_indices']==row['native_frame_indices']
    with pytest.raises(ValueError,match='overwritten'):q.plan_review(f['proof_path'],f['policy_path'],path)


def test_review_cannot_substitute_unreviewed_native_frame(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch)
    f['review']['cases'][0]['native_frame_indices']['maximum_frame_rmse']=0
    put(f['review_path'],f['review'])
    with pytest.raises(ValueError,match='Review outputs differ'):create(f)


@pytest.mark.parametrize('different_variant',[False,True])
def test_fresh_proof_requires_audited_runtime_overlay_for_replacement(tmp_path,monkeypatch,different_variant):
    f=fixture(tmp_path,monkeypatch)
    environment=copy.deepcopy(f['proof']['environment'])
    environment['gpu']['uuid']='NEW-CPU-FIXTURE-NOT-HARDWARE'
    if different_variant:environment['gpu']['name']='different-variant'
    env=put(tmp_path/'replacement/environment.json',{'passed':True,'environment':environment})
    lock=put(tmp_path/'replacement/lock.json',q.reference_identity(environment))
    overlay={'environment_verification':str(env),'environment_lock':str(lock)}
    overlay_path=tmp_path/'replacement/overlay.yaml';overlay_path.write_text(yaml.safe_dump(overlay))
    f['proof']['environment']=environment;f['proof']['runtime_identity']=q.reference_identity(environment)
    f['proof']['config'].update(overlay);put(f['proof_path'],f['proof'])
    f['policy']['runtime_config']=bound(overlay_path);put(f['policy_path'],f['policy'])
    f['review']['proof_sha256']=file_sha256(f['proof_path']);f['review']['policy_sha256']=file_sha256(f['policy_path'])
    put(f['review_path'],f['review'])
    if different_variant:
        with pytest.raises(ValueError,match='Replacement quality runtime'):create(f)
    else:
        value=create(f)
        assert value['qualified'] and value['runtime_identity']['gpu']['uuid']=='NEW-CPU-FIXTURE-NOT-HARDWARE'
        assert value['component_diagnostic']['runtime_identity']['gpu']['uuid']=='CPU-FIXTURE-NOT-HARDWARE'


@pytest.mark.parametrize('location',['top','nested','component'])
@pytest.mark.parametrize('fault',['identity','constructor_scope','boolean_counter','fallback_before','missing_after','reset'])
def test_compiler_execution_identity_and_positive_owned_counters(tmp_path,monkeypatch,location,fault):
    f=fixture(tmp_path,monkeypatch)
    if location=='component':
        owner=q.read(f['component']);runtime=owner['diagnostic']['compiled_runtime']
    else:
        owner=f['proof'];details=owner['runs']['isolation']
        if location=='nested':details=details['executions']['reverse']
        runtime=details['cache_stats']['compiler_runtime']
    if fault=='identity':runtime['runtime_implementation_sha256']='0'*64
    elif fault=='constructor_scope':runtime['require_graphs']=False
    elif fault=='boolean_counter':runtime['graphs_created_by_runtime']=True
    elif fault=='fallback_before':runtime['before']['unimplemented']={'unsupported':1}
    elif fault=='missing_after':runtime.pop('after')
    else:runtime['before']['unique_graphs']=2
    if location=='component':
        put(f['component'],owner);f['policy']['component_evidence']=bound(f['component']);put(f['policy_path'],f['policy'])
    else:put(f['proof_path'],owner)
    with pytest.raises(ValueError):create(f)
