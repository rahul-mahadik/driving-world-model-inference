"""Synthetic final evidence → blind native export → exhaustive new-tree inventory."""
from copy import deepcopy
import json
from pathlib import Path
import time

import pytest
from PIL import Image

from drivebench_tuning import agent_artifacts as artifacts, agent_metrics_worker, vista_agent
from drivebench_tuning.agent_cohort import freeze_final_plan, selection_only_executor, POLICIES, ROLES, COMPARISON_MODE, SINGLE_LLM_MODE
from drivebench_tuning.agent_session import SessionPolicy, AgentSession
from test_agent_metrics_worker import fixture, calibrate, write, pin, batch_pin, execute
from test_agent_evidence import run, now


def final_fixture(root, *, study_mode=COMPARISON_MODE, policy_names=POLICIES):
    f=fixture(root);review_at=now();quality_pin,quality=calibrate(f);started=now();sessions={}
    metadata={'quality_policy_sha256':quality.sha256,'identity_sha256':quality.identity_sha256,
        'source_manifest_sha256':f['common']['study_bindings']['source_manifest']['sha256'],
        'inputs_manifest_sha256':f['common']['study_bindings']['input_manifest']['sha256']}
    if study_mode == SINGLE_LLM_MODE:
        metadata.update(study_mode=study_mode, policy_names=list(policy_names))
    roles = ('reference', *(name+'_selected' for name in policy_names))
    settings=vista_agent.reference_settings(synthetic=True)
    for name in policy_names:
        policy=SessionPolicy(session_id=name,cohort_id='evidence-fixture',reference_config=settings,
            allowed_config_values=vista_agent.ALLOWED_VALUES,wall_seconds=120,max_tool_calls=10,
            max_evaluations=2,max_profiles=0,metadata=metadata)
        def measured(proposal,context):return {'cohort_id':'evidence-fixture','status':'ok','screening_status':'no_observed_breach',
            'candidate_sha256':vista_agent.digest(proposal['config']),**metadata}
        directory=root/f'session-{name}'
        with AgentSession.create(directory,policy=policy,executor=selection_only_executor(measured)) as session:
            pid=session.call('propose',{'overrides':{'generation':{'decode_chunk_size':2}}})['result']['proposal_id']
            session.call('trial',{'proposal_id':pid});session.call('submit',{'proposal_id':pid})
        sessions[name]=(directory,policy)
    plan_path=root/'final-plan.json';blocks=quality.to_dict()['plan']['timing']['block_scenes']
    plan=freeze_final_plan(plan_path,sessions=sessions,quality_policy=quality,block_orders={b:list(roles) for b in blocks},
        source_manifest_sha256=metadata['source_manifest_sha256'],inputs_manifest_sha256=metadata['inputs_manifest_sha256'],
        deadline_unix=time.time()+120,synthetic=True,study_mode=study_mode,policy_names=policy_names)
    batches={role:[] for role in roles}
    for entry in plan['batches']:
        subset={**f,'records':[r for r in f['final'] if r['scene_id'] in entry['scenes']]}
        batches[entry['role']].append(batch_pin(run(subset,f'final-{entry["batch_index"]}',entry['role'],
            reps=(entry['replicate'],),settings=entry['config'])))
    comparisons=[]
    for name in policy_names:
        metric={**f['common'],'kind':'final','output':str(root/f'final-metrics-{name}'),
            'references':batches['reference'],'candidates':batches[name+'_selected'],'quality_policy':quality_pin,
            'selection_started_at':started,'candidate_sha256':plan['selections'][name]['candidate_sha256'],'final_plan':pin(plan_path)}
        execute(f,metric,'metric-'+name)
        comparisons.append({'policy':name,'metric_completion':pin(Path(metric['output'])/'completed.json')})
    review={'schema_version':1,'created_at':review_at,'cells':[{'scene_id':r['scene_id'],'seed':11,'replicate':0} for r in f['final']],
            'review_seed':20261114,'maximum_cases':6}
    common={'schema_version':1,'synthetic':True,'created_at':now(),'deadline_unix':time.time()+120,
        'minimum_free_bytes':0,'maximum_output_bytes':50_000_000,
        'sources':{str(p):vista_agent.file_hash(p) for p in artifacts.required_source_paths()}}
    spec={**common,'kind':'export','output':str(root/'review'),'final_plan':pin(plan_path),
          'quality_policy':quality_pin,'review_plan':write(root/'review-plan.json',review),'comparisons':comparisons}
    return f,spec,common


def perform(root,spec,name):
    path=root/name;write(path,spec)
    return artifacts.run_spec(path,expected_sha256=vista_agent.file_hash(path))


def test_blind_native_export_and_inventory_preserve_complete_current_evidence(tmp_path):
    study=tmp_path/'study';study.mkdir();f,spec,common=final_fixture(study)
    completed=perform(study,spec,'export-spec.json');out=Path(spec['output'])
    assert completed['result']['cases']==6 and not completed['result']['visual_review_performed']
    assert not (out/'public/private_key.json').exists()
    assert (out/'private').stat().st_mode & 0o077 == 0
    key=json.loads((out/'private/rendering/private_key.json').read_text())
    for case in key['cases']:
        native=list((out/'public/native'/case['case_id']).glob('*.png'))
        assert len(native)==2
        with Image.open(native[0]) as image:assert image.size==(16,32)
        with Image.open(out/'public'/f'{case["case_id"]}.gif') as image:assert image.n_frames==2
    public=''.join(p.read_text() for p in (out/'public').glob('*') if p.suffix in ('.html','.csv'))
    assert all(r['scene_id'] not in public for r in f['final'])
    failed=study/'failed-attempt';failed.mkdir();(failed/'partial.npy').write_bytes(b'partial evidence');write(failed/'failure.json',{'status':'failed'})
    extra=tmp_path/'current-host-terminal.json';write(extra,{'status':'completed','scope':'fixture only'})
    inventory={**common,'kind':'inventory','output':str(tmp_path/'inventory'),'roots':{'study':str(study)},
        'extra_files':[str(extra)],'required_pins':[pin(out/'completed.json'),pin(extra)],'maximum_input_bytes':50_000_000}
    result=perform(tmp_path,inventory,'inventory-spec.json')
    records=json.loads((tmp_path/'inventory/inventory.json').read_text())['files']
    assert 'study/failed-attempt/partial.npy' in records and 'study/failed-attempt/failure.json' in records
    assert result['result']['two_identical_hash_passes'] and not result['result']['provider_or_process_cleanup_verified']
    assert (failed/'partial.npy').read_bytes()==b'partial evidence'
    with pytest.raises(FileExistsError):perform(study,spec,'another-export-spec.json')


@pytest.mark.parametrize('failure',['late_review','duplicate_cell','repeated_scene_seed','foreign_cell','missing_comparison','wrong_role','metric_changed','raw_changed'])
def test_export_rejects_stale_or_changed_provenance(tmp_path,failure):
    _,spec,_=final_fixture(tmp_path)
    review=json.loads(Path(spec['review_plan']['path']).read_text())
    if failure=='late_review':review['created_at']=now()
    elif failure=='duplicate_cell':review['cells'].append(review['cells'][0])
    elif failure=='repeated_scene_seed':review['cells'].append({**review['cells'][0],'replicate':1})
    elif failure=='foreign_cell':review['cells'][0]['scene_id']='foreign'
    elif failure=='missing_comparison':spec['comparisons'].pop()
    elif failure=='wrong_role':spec['comparisons'][0]['policy']='grid'
    elif failure=='metric_changed':
        path=Path(spec['comparisons'][0]['metric_completion']['path']).parent/'timing-score.json'
        path.chmod(0o600);path.write_text('{}\n')
    else:
        path=next((tmp_path/'final-0/batch').glob('*.npy'));path.chmod(0o600);path.write_bytes(path.read_bytes()+b'drift')
    if failure in ('late_review','duplicate_cell','repeated_scene_seed','foreign_cell'):
        spec['review_plan']=write(tmp_path/'altered-review-plan.json',review)
    with pytest.raises(ValueError):perform(tmp_path,spec,'invalid-export-spec.json')
    assert not (Path(spec['output'])/'completed.json').exists()
    assert (Path(spec['output'])/'failure.json').is_file()


def inventory_fixture(tmp_path):
    root=tmp_path/'evidence';root.mkdir();data=root/'data.bin';data.write_bytes(b'original')
    return {'schema_version':1,'synthetic':True,'kind':'inventory','created_at':now(),'deadline_unix':time.time()+30,
        'minimum_free_bytes':0,'maximum_output_bytes':1_000_000,'output':str(tmp_path/'inventory'),
        'sources':{str(p):vista_agent.file_hash(p) for p in artifacts.required_source_paths()},
        'roots':{'current':str(root)},'extra_files':[],'required_pins':[pin(data)],'maximum_input_bytes':1000}


@pytest.mark.parametrize('failure',['output_inside','overlap','symlink','missing_pin','byte_cap','deadline','changed_between_passes'])
def test_inventory_does_not_hide_partial_or_changed_evidence(tmp_path,monkeypatch,failure):
    spec=inventory_fixture(tmp_path);root=Path(spec['roots']['current'])
    if failure=='output_inside':spec['output']=str(root/'inventory')
    elif failure=='overlap':spec['roots']['parent']=str(tmp_path)
    elif failure=='symlink':(root/'link').symlink_to(root/'data.bin')
    elif failure=='missing_pin':spec['required_pins']=[{'path':str(tmp_path/'outside'),'sha256':'a'*64}]
    elif failure=='byte_cap':spec['maximum_input_bytes']=1
    elif failure=='deadline':spec['deadline_unix']=time.time()-1
    else:
        write_original=artifacts._write
        def altered(path,value):
            write_original(path,value)
            if Path(path).name=='inventory.json':(root/'new-partial.bin').write_bytes(b'appeared')
        monkeypatch.setattr(artifacts,'_write',altered)
    with pytest.raises(ValueError):perform(tmp_path,spec,'inventory-spec.json')
    assert not (Path(spec['output'])/'completed.json').exists()
    assert (root/'data.bin').read_bytes()==b'original'


@pytest.mark.parametrize('headroom',[0,-1])
def test_completion_receipt_is_admitted_at_exact_total_byte_boundary(tmp_path,headroom):
    # Include nested evidence and non-ASCII receipt text to exercise actual UTF-8 bytes.
    nested=tmp_path/'partial';nested.mkdir();(nested/'data.bin').write_bytes(b'retained evidence')
    receipt={'status':'ok','label':'native \u03bb frames','artifacts':{'partial/data.bin':{'bytes':17}}}
    encoded=artifacts._bytes(receipt)+b'\n'
    exact=sum(p.stat().st_size for p in tmp_path.rglob('*') if p.is_file())+len(encoded)
    if headroom==0:
        artifacts._write_completion(tmp_path,receipt,exact)
        assert (tmp_path/'completed.json').read_bytes()==encoded
        assert sum(p.stat().st_size for p in tmp_path.rglob('*') if p.is_file())==exact
    else:
        with pytest.raises(ValueError,match='completion receipt would exceed'):
            artifacts._write_completion(tmp_path,receipt,exact+headroom)
        assert not (tmp_path/'completed.json').exists()
        assert (nested/'data.bin').read_bytes()==b'retained evidence'
