"""Actual synthetic worker → streamed calibration → fresh trial/final scoring."""
from copy import deepcopy
from pathlib import Path
import itertools
import json
import time

import numpy as np
import pytest

from drivebench_tuning import agent_metrics_worker as worker, vista_agent, agent_scoring
from test_agent_evidence import setup, run, load, now


def pin(path):return {'path':str(path),'sha256':vista_agent.file_hash(path)}
def write(path,value):path.write_text(json.dumps(value));return pin(path)
def batch_pin(triple):
    directory,pins,role=triple
    return {'directory':str(directory),'role':role,**pins}


def fixture(tmp_path):
    f=setup(tmp_path,scene_count=4);f['dev']=f['records'][:2];f['final']=f['records'][2:]
    planned=now();reference=run({**f,'records':f['dev']},'calibration')
    gen=json.loads((reference[0]/'spec.json').read_text())
    bindings={'source_manifest':write(tmp_path/'sources.json',{'source_files':gen['source_files']}),
              'input_manifest':write(tmp_path/'inputs.json',{'manifests':[gen['manifest']]})}
    plan={'metric_margins':{k:1e-6 for k in worker.METRICS},'development_scenes':[r['scene_id'] for r in f['dev']],
          'final_scenes':[r['scene_id'] for r in f['final']],'seeds':[11,23],'replicates':[0,1],
          'plan_created_at':planned,'minimum_development_scenes':2,'minimum_final_scenes':2,
          'timing_min_blocks':2,'noise_floor_method':'maximum_paired_repeat_relative_difference'}
    common={'schema_version':1,'synthetic':True,'created_at':now(),'deadline_unix':time.time()+120,
            'minimum_free_bytes':0,'metric':{'kind':'synthetic_fixture'},
            'sources':{str(p):vista_agent.file_hash(p) for p in worker.required_source_paths()},'study_bindings':bindings}
    spec={**common,'kind':'calibrate','references':[batch_pin(reference)],'quality_plan':plan,'output':str(tmp_path/'calibrated')}
    f.update(common=common,calibration_spec=spec,reference=reference)
    return f


def execute(f,spec,name):
    path=f['root']/f'{name}.json';write(path,spec)
    return worker.run_spec(path,expected_sha256=vista_agent.file_hash(path))


def calibrate(f):
    result=execute(f,f['calibration_spec'],'cal-spec')
    assert result['result']['calibration_status']=='ready'
    path=Path(f['calibration_spec']['output'])/'quality-policy.json'
    return pin(path),agent_scoring.QualityPolicy.from_dict(json.loads(path.read_text()))


def trial(f,policy):
    started=now();ref=run({**f,'records':f['dev']},'trial-ref',reps=(0,))
    settings=vista_agent.reference_settings(synthetic=True);settings['generation']['decode_chunk_size']=2
    candidate=run({**f,'records':f['dev']},'trial-candidate','candidate',reps=(0,),settings=settings)
    spec={**f['common'],'kind':'trial','output':str(f['root']/'trial-metrics'),
          'references':[batch_pin(ref)],'candidates':[batch_pin(candidate)],'quality_policy':policy,
          'selection_started_at':started,'candidate_sha256':vista_agent.digest(settings),
          'trial_cells':[{'scene_id':r['scene_id'],'seed':seed,'replicate':0} for r in f['dev'] for seed in (11,23)]}
    return spec


def test_calibration_streams_controls_and_pins_fresh_trial(tmp_path):
    f=fixture(tmp_path);policy,parsed=calibrate(f);out=Path(f['calibration_spec']['output'])
    assert not (out/'derived-control.npy').exists()
    controls=sorted((out/'controls').glob('*.json'));assert len(controls)==24
    for path in controls:
        data=json.loads(path.read_text());recipe=data['recipe'];raw=np.load(recipe['reference_path'])
        from drivebench_tuning.agent_metrics import negative_control
        recreated=tmp_path/'recreated.npy';np.save(recreated,negative_control(raw,recipe['control']))
        assert vista_agent.file_hash(recreated)==recipe['derived_sha256']
    assert parsed.to_dict()['plan']['identity']['evidence_kind']=='synthetic'
    result=execute(f,trial(f,policy),'trial-spec')
    assert result['result']['screening_status']=='no_observed_breach'
    assert not result['result']['quality_assessed'] and not result['real_lpips']
    assert result['result']['quality_policy_sha256']==parsed.sha256
    assert result['result']['reference_median_latency_ms']>0
    assert result['result']['paired_cells']==4
    with pytest.raises(FileExistsError):execute(f,f['calibration_spec'],'again')


@pytest.mark.parametrize('change',['expired','bad_source','unknown_field','mixed_kind','source_bundle','input_bundle','fixture_real'])
def test_invalid_spec_and_binding_fail_closed(tmp_path,change):
    f=fixture(tmp_path);spec=f['calibration_spec']
    if change=='expired':spec['deadline_unix']=time.time()-1
    elif change=='bad_source':spec['sources'][str(worker.required_source_paths()[0])]='0'*64
    elif change=='unknown_field':spec['trust_me']=True
    elif change=='mixed_kind':spec['synthetic']=False
    elif change=='fixture_real':
        with pytest.raises(ValueError,match='forbidden'):
            with worker._metric({'kind':'synthetic_fixture'},False):pass
        return
    elif change=='source_bundle':spec['study_bindings']['source_manifest']=write(tmp_path/'wrong-source.json',{'source_files':{}})
    elif change=='input_bundle':spec['study_bindings']['input_manifest']=write(tmp_path/'wrong-input.json',{'manifests':[]})
    with pytest.raises(ValueError):execute(f,spec,'bad-spec')
    assert not (Path(spec['output'])/'completed.json').exists()


def test_failure_preserves_only_current_derived_scratch(tmp_path,monkeypatch):
    f=fixture(tmp_path);original=worker.agent_metrics.measure_pair
    def fail(a,b,**kw):
        if str(b).endswith('derived-control.npy'):raise RuntimeError('metric failed')
        return original(a,b,**kw)
    monkeypatch.setattr(worker.agent_metrics,'measure_pair',fail)
    with pytest.raises(RuntimeError,match='metric failed'):execute(f,f['calibration_spec'],'failed-spec')
    out=Path(f['calibration_spec']['output']);assert (out/'derived-control.npy').is_file()
    assert json.loads((out/'failure.json').read_text())['scratch_retained']
    assert len(list((f['reference'][0]/'batch').glob('*.npy')))==9


@pytest.mark.parametrize('change',['missing_cell','duplicate_cell','foreign_scene','before_selection','changed_policy','old_reference'])
def test_trial_cannot_score_wrong_or_stale_evidence(tmp_path,change):
    f=fixture(tmp_path);policy,_=calibrate(f);spec=trial(f,policy)
    if change=='missing_cell':spec['trial_cells'].pop()
    elif change=='duplicate_cell':spec['trial_cells'].append(spec['trial_cells'][0])
    elif change=='foreign_scene':spec['trial_cells'][0]['scene_id']='foreign'
    elif change=='before_selection':spec['selection_started_at']=now()
    elif change=='changed_policy':spec['quality_policy']['sha256']='0'*64
    elif change=='old_reference':spec['references']=[batch_pin(f['reference'])]
    with pytest.raises(ValueError):execute(f,spec,'invalid-trial')
    assert not (Path(spec['output'])/'completed.json').exists()


def test_final_actual_barrier_and_all_roles(tmp_path):
    from drivebench_tuning.agent_cohort import freeze_final_plan,selection_only_executor,POLICIES,ROLES
    from drivebench_tuning.agent_session import SessionPolicy,AgentSession
    f=fixture(tmp_path);policy_pin,policy=calibrate(f);started=now();sessions={}
    metadata={'quality_policy_sha256':policy.sha256,'identity_sha256':policy.identity_sha256,
              'source_manifest_sha256':f['common']['study_bindings']['source_manifest']['sha256'],
              'inputs_manifest_sha256':f['common']['study_bindings']['input_manifest']['sha256']}
    settings=vista_agent.reference_settings(synthetic=True)
    for name in POLICIES:
        config=SessionPolicy(session_id=name,cohort_id='evidence-fixture',reference_config=settings,
             allowed_config_values=vista_agent.ALLOWED_VALUES,wall_seconds=100,max_tool_calls=10,max_evaluations=5,max_profiles=0,metadata=metadata)
        def measured(proposal,context):return {'cohort_id':'evidence-fixture','status':'ok','screening_status':'no_observed_breach',
            'candidate_sha256':vista_agent.digest(proposal['config']),**metadata}
        directory=tmp_path/f'session-{name}'
        with AgentSession.create(directory,policy=config,executor=selection_only_executor(measured)) as session:
            pid=session.call('propose',{'overrides':{'generation':{'decode_chunk_size':2}}})['result']['proposal_id']
            session.call('trial',{'proposal_id':pid});session.call('submit',{'proposal_id':pid})
        sessions[name]=(directory,config)
    plan_path=tmp_path/'final-plan.json';blocks=policy.to_dict()['plan']['timing']['block_scenes']
    plan=freeze_final_plan(plan_path,sessions=sessions,quality_policy=policy,block_orders={b:list(ROLES) for b in blocks},
         source_manifest_sha256=metadata['source_manifest_sha256'],inputs_manifest_sha256=metadata['inputs_manifest_sha256'],
         deadline_unix=time.time()+100,synthetic=True)
    batches={role:[] for role in ROLES}
    for entry in plan['batches']:
        subset={**f,'records':[r for r in f['final'] if r['scene_id'] in entry['scenes']]}
        batches[entry['role']].append(batch_pin(run(subset,f'final-{entry["batch_index"]}',entry['role'],
            reps=(entry['replicate'],),settings=entry['config'])))
    for name in POLICIES:
        spec={**f['common'],'kind':'final','output':str(tmp_path/f'final-metrics-{name}'),
              'references':batches['reference'],'candidates':batches[name+'_selected'],
              'quality_policy':policy_pin,'selection_started_at':started,
              'candidate_sha256':plan['selections'][name]['candidate_sha256'],'final_plan':pin(plan_path)}
        result=execute(f,spec,f'final-spec-{name}')
        assert result['result']['quality_status']=='pass'
        assert result['result']['score']['status']=='pass'
        assert not result['quality_preserving_claim']
    spec['output']=str(tmp_path/'bad-binding');plan['source_manifest_sha256']='0'*64
    plan['sha256']=vista_agent.digest({k:v for k,v in plan.items() if k!='sha256'});spec['final_plan']=write(tmp_path/'altered-plan.json',plan)
    with pytest.raises(ValueError,match='aggregate manifest'):execute(f,spec,'bad-final')


def mocked_lpips(tmp_path, monkeypatch):
    """Exercise the real-mode import contract; this does not execute LPIPS."""
    import builtins
    import socket
    import sys
    from types import SimpleNamespace
    from drivebench.evaluation import metrics
    target=tmp_path/'staged';(target/'lpips').mkdir(parents=True)
    module=target/'lpips/__init__.py';module.write_text('# mocked LPIPS import')
    alexnet=tmp_path/'alexnet.py';alexnet.write_text('# mocked torchvision AlexNet source')
    file_record=lambda path:{'path':str(path),'bytes':path.stat().st_size,'sha256':vista_agent.file_hash(path)}
    setup={'package':{'target':str(target),'files':{'lpips/__init__.py':{}}},
        'wheel':{'sha256':'fd537af5828b69d2e6ffc0a397bd506dbc28ca183543617690844c08e102ec5e'},
        'alexnet_cached_weight':{'sha256':'7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02',
                                'path':str(tmp_path/'torch/hub/checkpoints/alexnet.pth')},
        'lpips_learned_weight':{'sha256':'df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0'},
        'cpu_verification':{'model_state_dict_sha256':'b'*64,'source_files':{
            'torchvision_alexnet':file_record(alexnet),**{k:file_record(module) for k in ('lpips','lpips_model','lpips_trunk')}}}}
    config={'kind':'lpips','setup':write(tmp_path/'setup.json',setup),'device':'cpu',
        'versions':{'torch':'test','torchvision':'test','numpy':'test','scipy':'test','lpips':'0.1.4'}}
    def downloader(*a,**kw):raise AssertionError('unprotected download')
    torch=SimpleNamespace(hub=SimpleNamespace(download_url_to_file=downloader))
    imported=builtins.__import__
    def fake_import(name,*a,**kw):return torch if name=='torch' else imported(name,*a,**kw)
    monkeypatch.setattr(builtins,'__import__',fake_import)
    monkeypatch.delitem(sys.modules,'torch',raising=False);monkeypatch.delitem(sys.modules,'lpips',raising=False)
    monkeypatch.setattr(worker.importlib.metadata,'version',lambda name:config['versions'][name])
    original_import_module=worker.importlib.import_module
    monkeypatch.setattr(worker.importlib,'import_module',lambda name,*a,**kw:
        SimpleNamespace(__file__=str(alexnet)) if name=='torchvision.models.alexnet' else original_import_module(name,*a,**kw))
    checks=[]
    monkeypatch.setattr(metrics,'verify_perceptual_setup',lambda path:checks.append(path) or setup)
    fake=SimpleNamespace(status='available',reason=None,provenance={'module_path':str(module),'state_dict_sha256':'b'*64})
    def construct(**kwargs):
        for operation in (socket.create_connection,torch.hub.download_url_to_file):
            with pytest.raises(RuntimeError,match='network forbidden'):operation(('example.invalid',443))
        return fake
    monkeypatch.setattr(metrics,'PerceptualMetric',construct)
    return config,setup,fake,checks,torch


def test_mocked_real_lpips_is_offline_state_bound_and_restores_process(tmp_path,monkeypatch):
    import os
    import socket
    import sys
    config,setup,fake,checks,torch=mocked_lpips(tmp_path,monkeypatch)
    originals=(sys.path[:],sys.dont_write_bytecode,os.environ.get('TORCH_HOME'),socket.create_connection,torch.hub.download_url_to_file)
    with worker._metric(config,False) as (metric,provenance):
        assert metric is fake and provenance['is_lpips']
        assert provenance['state_dict_sha256']==setup['cpu_verification']['model_state_dict_sha256']
        assert sys.dont_write_bytecode
        with pytest.raises(RuntimeError,match='network forbidden'):socket.create_connection(('example.invalid',443))
    assert len(checks)==2
    assert originals==(sys.path[:],sys.dont_write_bytecode,os.environ.get('TORCH_HOME'),socket.create_connection,torch.hub.download_url_to_file)


@pytest.mark.parametrize('failure',['asset','version','versions_type','import_root','state','unavailable','extra_file','preimport','post_extra','source','source_import','post_source'])
def test_mocked_real_lpips_refuses_substitution(tmp_path,monkeypatch,failure):
    import sys
    config,setup,fake,checks,torch=mocked_lpips(tmp_path,monkeypatch)
    if failure=='asset':setup['wheel']['sha256']='0'*64
    elif failure=='version':monkeypatch.setattr(worker.importlib.metadata,'version',lambda name:'wrong')
    elif failure=='versions_type':config['versions']=None
    elif failure=='import_root':fake.provenance['module_path']=str(tmp_path/'foreign.py')
    elif failure=='state':fake.provenance['state_dict_sha256']='c'*64
    elif failure=='unavailable':fake.status='unavailable';fake.reason='missing assets'
    elif failure=='extra_file':(Path(setup['package']['target'])/'lpips.py').write_text('unlisted')
    elif failure=='preimport':monkeypatch.setitem(sys.modules,'torch',torch)
    elif failure=='source':Path(setup['cpu_verification']['source_files']['torchvision_alexnet']['path']).write_text('drift')
    elif failure=='source_import':
        from types import SimpleNamespace
        monkeypatch.setattr(worker.importlib,'import_module',lambda name:SimpleNamespace(__file__=str(tmp_path/'foreign.py')))
    with pytest.raises(ValueError):
        with worker._metric(config,False):
            if failure=='post_extra':(Path(setup['package']['target'])/'new.py').write_text('drift')
            if failure=='post_source':Path(setup['cpu_verification']['source_files']['torchvision_alexnet']['path']).write_text('drift')


def test_calibration_freezes_observed_repeat_noise_without_widening_quality(tmp_path):
    f=fixture(tmp_path);_,policy=calibrate(f)
    floor=json.loads((Path(f['calibration_spec']['output'])/'timing-noise.json').read_text())
    data=load(f['reference']).to_dict();cells={}
    for row in data['rows']:cells.setdefault((row['scene_id'],row['seed']),[]).append(row['latency_ms'])
    expected=max(max(values)/min(values)-1 for values in cells.values())
    assert floor['frozen_floor']==floor['observed_floor']==expected
    assert floor['confidence_bound'] is False
    assert policy.to_dict()['plan']['metric_margins']==f['calibration_spec']['quality_plan']['metric_margins']


@pytest.mark.parametrize('corruption',['receipt','input','derived'])
def test_control_retirement_requires_verified_permanent_receipt_and_arrays(tmp_path,monkeypatch,corruption):
    f=fixture(tmp_path);original=worker._write
    def corrupt(path,value):
        original(path,value)
        if Path(path).parent.name=='controls':
            if corruption=='receipt':
                path.chmod(0o600);path.write_text('{}\n')
            elif corruption=='input':
                source=Path(value['recipe']['reference_path']);source.chmod(0o600);source.write_bytes(source.read_bytes()+b'changed')
            else:
                scratch=Path(f['calibration_spec']['output'])/'derived-control.npy';scratch.write_bytes(scratch.read_bytes()+b'changed')
    monkeypatch.setattr(worker,'_write',corrupt)
    with pytest.raises(ValueError):execute(f,f['calibration_spec'],'corrupt-control-spec')
    out=Path(f['calibration_spec']['output'])
    assert (out/'derived-control.npy').is_file() and (out/'controls/0000.json').is_file()
    assert json.loads((out/'failure.json').read_text())['scratch_retained']
    assert not (out/'completed.json').exists()
