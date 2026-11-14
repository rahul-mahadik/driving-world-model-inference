"""Concrete controller tests; actual CPU fixture workers, never Docker/GPU/API."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import pytest

from drivebench_tuning import agent_trials as trials, vista_agent, vista_batch, agent_metrics_worker
from drivebench_tuning.agent_session import AgentSession, SessionPolicy, SessionError
from drivebench_tuning.agent_cohort import selection_only_executor
from test_agent_metrics_worker import fixture, calibrate, write, pin, now


class FixtureTransport:
    """Tests only: executes real synthetic workers in process; no host-cleanup claim."""
    def __init__(self, **kwargs):
        self.max_jobs=kwargs.get('max_jobs',6);self.used=0;self.blocked=False;self.order=[]
    def container_path(self,path):return str(Path(path).absolute())
    def host_path(self,path):return Path(path)
    def __call__(self,kind,spec,*,spec_path,timeout_seconds,execution_deadline):
        assert spec['synthetic'] is True
        if self.blocked or self.used>=self.max_jobs:raise RuntimeError('fixture job budget')
        self.used+=1;self.blocked=True;self.order.append((kind,spec['output']))
        write(spec_path,spec);sha=vista_agent.file_hash(spec_path)
        host=spec_path.with_name(spec_path.stem+'-host.json');write(host,{'fixture_only':True})
        module= vista_batch if kind=='batch' else agent_metrics_worker
        completion=module.run_spec(spec_path,expected_sha256=sha)
        terminal=spec_path.with_name(spec_path.stem+'-terminal.json');write(terminal,{'synthetic_fixture_only':True})
        ending=Path(spec['output'])/('adapter-completed.json' if kind=='batch' else 'completed.json')
        self.blocked=False
        return {'spec_sha256':sha,'completion_sha256':vista_agent.file_hash(ending),'completion':completion,
                'terminal_receipt':str(terminal),'terminal_receipt_sha256':vista_agent.file_hash(terminal),
                'worker_exit_verified':True,'directory':spec['output']}


def episode(tmp_path, *, selector=None, count=2):
    f=fixture(tmp_path);quality_pin,quality=calibrate(f)
    batch=json.loads((f['reference'][0]/'spec.json').read_text())
    batch['requests']=[r for r in batch['requests'] if r['replicate']==0]
    batch['deadline_unix']=time.time()+7200
    metric={**f['common'],'kind':'trial','output':str(tmp_path/'unused'),'references':[],
            'candidates':[],'quality_policy':quality_pin,'selection_started_at':now(),
            'candidate_sha256':vista_agent.digest(batch['settings']),
            'trial_cells':[{k:r[k] for k in ('scene_id','seed','replicate')} for r in batch['requests']],
            'deadline_unix':time.time()+7200}
    metadata={'quality_policy_sha256':quality.sha256,'identity_sha256':quality.identity_sha256,
              'source_manifest_sha256':metric['study_bindings']['source_manifest']['sha256'],
              'inputs_manifest_sha256':metric['study_bindings']['input_manifest']['sha256']}
    policy=SessionPolicy(session_id='grid-fixture',cohort_id=batch['cohort_id'],reference_config=batch['settings'],
                         allowed_config_values=vista_agent.ALLOWED_VALUES,wall_seconds=3600,
                         max_tool_calls=20,max_evaluations=count+1,max_profiles=0,metadata=metadata)
    spec={'schema_version':1,'output':str(tmp_path/'episode'),'created_at':now(),'deadline_unix':time.time()+7200,
          'session_policy':policy.to_dict(),'batch_template':batch,'metric_template':metric,
          'transport':{'root':str(tmp_path),'launcher':str(tmp_path/'unused-launcher'),
                       'launcher_sha256':'a'*64,'python':sys.executable,'host_template':{},'max_jobs':3*count},
          'trial_timeouts':{'reference':1,'candidate':1,'metrics':1},
          'selector':selector or {'kind':'grid','seed':17},'max_turns':20}
    path=tmp_path/'episode-spec.json';write(path,spec)
    return f,quality,policy,spec,path


def test_concrete_grid_episode_uses_fresh_paired_workers_and_only_selects(tmp_path,monkeypatch):
    _,_,policy,spec,path=episode(tmp_path)
    monkeypatch.setattr(trials,'ExistingBoundedLauncher',FixtureTransport)
    result=trials.run_spec(path,expected_sha256=vista_agent.file_hash(path))
    assert result['status']=='ok' and result['jobs_used']==6
    assert result['selection_only'] and not result['final_evaluation_performed']
    assert not result['quality_preserving_claim']
    root=Path(spec['output'])
    assert json.loads((root/'trials/0000/intent.json').read_text())['order']==['reference','candidate']
    assert json.loads((root/'trials/0001/intent.json').read_text())['order']==['candidate','reference']
    assert len(list(root.glob('trials/*/reference/batch/completed.json')))==2
    assert len(list(root.glob('trials/*/candidate/batch/completed.json')))==2
    for p in root.glob('trials/*/completed.json'):
        receipt=json.loads(p.read_text());assert receipt['screening_status']=='no_observed_breach'
        assert not receipt['quality_assessed'] and receipt['paired_cells']==4
    with AgentSession.open(root/'session',expected_policy=policy,executor=lambda *_:None) as session:
        public=trials.public_observation(session)(session.state,{'ok':False,'error':{'secret':'private-path'}})
        assert public['trials_used']==2 and public['status']=='completed'
        assert 'private-path' not in json.dumps(public)
        assert all('source_manifest' not in k and 'quality_policy' not in k for k in public)
        assert len({vista_agent.digest(r['config']) for r in public['trial_history']})==2
    with pytest.raises(FileExistsError):trials.run_spec(path,expected_sha256=vista_agent.file_hash(path))


def test_predetermined_single_seed_trial_schedule_preserves_equal_work(tmp_path,monkeypatch):
    _,_,_,spec,path=episode(tmp_path,count=4)
    spec['trial_schedule']=[{'seed':seed,'replicate':0} for seed in [11,23,23,11]]
    write(path,spec);monkeypatch.setattr(trials,'ExistingBoundedLauncher',FixtureTransport)
    result=trials.run_spec(path,expected_sha256=vista_agent.file_hash(path))
    assert result['jobs_used']==12
    for i,seed in enumerate([11,23,23,11]):
        attempt=Path(spec['output'])/'trials'/f'{i:04d}'
        for role in ['reference','candidate']:
            batch=json.loads((attempt/f'{role}-spec.json').read_text())
            assert len(batch['requests'])==2 and {r['seed'] for r in batch['requests']}=={seed}
        scored=json.loads((attempt/'completed.json').read_text())
        assert scored['paired_cells']==2 and not scored['quality_assessed']
        assert {r['seed'] for r in json.loads((attempt/'intent.json').read_text())['trial_cells']}=={seed}


def test_individual_operations_resume_original_deadline_and_do_not_score_final(tmp_path,monkeypatch):
    _,_,_,spec,path=episode(tmp_path,selector={'kind':'external-json'},count=1)
    monkeypatch.setattr(trials,'ExistingBoundedLauncher',FixtureTransport)
    sha=vista_agent.file_hash(path)
    first=trials.step_spec(path,expected_sha256=sha,operation='initialize')
    state=json.loads((Path(spec['output'])/'session/policy.json').read_text())
    def act(action,name):
        p=tmp_path/name;write(p,action)
        return trials.step_spec(path,expected_sha256=sha,operation='act',action_path=p,action_sha256=vista_agent.file_hash(p))
    pid=first['latest_proposal']['proposal_id']
    out=act({'tool':'trial','arguments':{'proposal_id':pid}},'trial-action.json')
    assert out['trials_used']==1 and out['trial_history'][0]['eligible']
    assert out['execution_available']
    observed=trials.step_spec(path,expected_sha256=sha,operation='observe')
    assert observed['remaining_wall_seconds']<=first['remaining_wall_seconds']
    assert json.loads((Path(spec['output'])/'session/policy.json').read_text())==state
    out=act({'tool':'submit','arguments':{'proposal_id':pid}},'submit-action.json')
    assert out['status']=='completed' and not out['execution_available']
    receipt=json.loads((Path(spec['output'])/'completed.json').read_text())
    assert receipt['jobs_used']==3 and receipt['selection_only'] and not receipt['final_evaluation_performed']


def test_failed_worker_is_charged_preserved_and_blocks_every_further_call(tmp_path,monkeypatch):
    _,_,_,spec,path=episode(tmp_path,selector={'kind':'external-json'},count=2)
    class Failure(FixtureTransport):
        calls=0
        def __call__(self,*args,**kwargs):
            type(self).calls+=1;self.used+=1;self.blocked=True
            raise RuntimeError('/private/model/heldout-secret must never reach agent')
    monkeypatch.setattr(trials,'ExistingBoundedLauncher',Failure)
    sha=vista_agent.file_hash(path);first=trials.step_spec(path,expected_sha256=sha,operation='initialize')
    action=tmp_path/'action.json';write(action,{'tool':'trial','arguments':{'proposal_id':first['latest_proposal']['proposal_id']}})
    out=trials.step_spec(path,expected_sha256=sha,operation='act',action_path=action,action_sha256=vista_agent.file_hash(action))
    assert out['trials_used']==1 and not out['execution_available'] and not out['last_action_succeeded']
    assert 'heldout-secret' not in json.dumps(out)
    assert (Path(spec['output'])/'trials/0000/failed.json').is_file()
    with pytest.raises(SessionError,match='failed worker'):
        trials.step_spec(path,expected_sha256=sha,operation='act',action_path=action,action_sha256=vista_agent.file_hash(action))
    assert Failure.calls==1


def test_resumed_trial_receipt_must_match_durable_session(tmp_path,monkeypatch):
    _,_,_,spec,path=episode(tmp_path,selector={'kind':'external-json'},count=2)
    monkeypatch.setattr(trials,'ExistingBoundedLauncher',FixtureTransport)
    sha=vista_agent.file_hash(path);first=trials.step_spec(path,expected_sha256=sha,operation='initialize')
    action=tmp_path/'action.json';write(action,{'tool':'trial','arguments':{'proposal_id':first['latest_proposal']['proposal_id']}})
    trials.step_spec(path,expected_sha256=sha,operation='act',action_path=action,action_sha256=vista_agent.file_hash(action))
    receipt=Path(spec['output'])/'trials/0000/completed.json'
    content=json.loads(receipt.read_text());content['candidate_reference_latency_ratio']=1e-9
    receipt.chmod(0o600);write(receipt,content)
    with pytest.raises(SessionError,match='receipt differs from durable session'):
        trials.step_spec(path,expected_sha256=sha,operation='observe')


def test_interrupted_operator_action_does_not_replay(tmp_path,monkeypatch):
    _,_,_,spec,path=episode(tmp_path,selector={'kind':'external-json'},count=1)
    monkeypatch.setattr(trials,'ExistingBoundedLauncher',FixtureTransport)
    sha=vista_agent.file_hash(path);first=trials.step_spec(path,expected_sha256=sha,operation='initialize')
    write(Path(spec['output'])/'operator-actions/00000001-started.json',{'action':'unknown-outcome'})
    action=tmp_path/'action.json';write(action,{'tool':'trial','arguments':{'proposal_id':first['latest_proposal']['proposal_id']}})
    with pytest.raises(SessionError,match='operator action interrupted'):
        trials.step_spec(path,expected_sha256=sha,operation='act',action_path=action,action_sha256=vista_agent.file_hash(action))
    assert not list((Path(spec['output'])/'trials').iterdir())


def test_whole_trial_admission_happens_before_any_worker(tmp_path):
    _,quality,policy,spec,_=episode(tmp_path)
    transport=FixtureTransport()
    executor=trials.PairedTrialExecutor(directory=tmp_path/'trial-admission',transport=transport,
        batch_template=spec['batch_template'],metric_template=spec['metric_template'],quality_policy=quality,
        timeouts={'reference':1,'candidate':1,'metrics':1},max_trials=2,selection_started_at=now())
    context={'cohort_id':policy.cohort_id,'metadata':policy.to_dict()['metadata'],'execution_deadline_unix':time.time()+722}
    with pytest.raises(TimeoutError,match='whole trial'):
        executor({'config':spec['batch_template']['settings'],'proposal_id':'reference'},context)
    assert transport.used==0 and executor.used==0 and not list(executor.directory.iterdir())


def test_metric_result_cannot_substitute_another_cohort(tmp_path,monkeypatch):
    _,_,_,spec,path=episode(tmp_path,count=1)
    class WrongCohort(FixtureTransport):
        def __call__(self,kind,*args,**kwargs):
            result=super().__call__(kind,*args,**kwargs)
            if kind=='metrics':result['completion']['cohort_id']='other-cohort'
            return result
    monkeypatch.setattr(trials,'ExistingBoundedLauncher',WrongCohort)
    with pytest.raises(SessionError,match='worker failed'):
        trials.run_spec(path,expected_sha256=vista_agent.file_hash(path))
    assert (Path(spec['output'])/'trials/0000/failed.json').is_file()
    assert not (Path(spec['output'])/'completed.json').exists()


def test_random_baseline_is_reproducible_and_submits_only_eligible(tmp_path):
    a=trials.SearchBaseline(mode='random',seed=7,reference=vista_agent.reference_settings(),allowed_values=vista_agent.ALLOWED_VALUES)
    b=trials.SearchBaseline(mode='random',seed=7,reference=vista_agent.reference_settings(),allowed_values=vista_agent.ALLOWED_VALUES)
    assert a.configs==b.configs and len({vista_agent.digest(c) for c in a.configs})==16
    obs={'trial_history':[{'proposal_id':'bad','eligible':False,'candidate_reference_latency_ratio':None},
                          {'proposal_id':'good','eligible':True,'candidate_reference_latency_ratio':.9}],
         'trial_budget':2}
    assert a(obs)=={'tool':'submit','arguments':{'proposal_id':'good'}}
    obs['trial_history'][1]['eligible']=False
    with pytest.raises(SessionError,match='no measured candidate'):a(obs)


def launcher_fixture(tmp_path,monkeypatch,corrupt=None,portable=False,worker_kind='batch',isolated=False):
    root=tmp_path/'host';root.mkdir();(root/'outputs/host').mkdir(parents=True)
    launcher=root/(trials.PORTABLE_LAUNCHER if portable else 'outputs/host/fixture.py')
    launcher.parent.mkdir(parents=True,exist_ok=True);launcher.write_text('# fixture only\n')
    sha=vista_agent.file_hash(launcher)
    monkeypatch.setitem(trials.LAUNCHERS,'outputs/host/fixture.py',('fixture-v1',sha))
    source=root/'module.py';source.write_text('value = 1\n')
    host={'root':str(root),'metadata':'unused','container_name':'unused','fresh_outputs':[],
          'timeout_seconds':1,'allocation_deadline_epoch':time.time()+4000,'expected_source_identity':'fixture',
          'expected_file_hashes':{},'python_arguments':[],'scope':'fixture','environment_verification':{},'environment_lock':{}}
    if portable:host['runtime']={'image_id':'sha256:'+'a'*64,'python_executable':'/opt/python','environment_proof_kind':'fresh-proof'}
    spec={'output':'/workspace/result','deadline_unix':time.time()+2000,'source_files':{'/workspace/module.py':vista_agent.file_hash(source)},
          'synthetic':True,'cohort_id':'fixture','batch_id':'fixture-batch'}
    if worker_kind=='metrics':spec['sources']=spec.pop('source_files');spec['kind']='trial'
    def run(argv,**kwargs):
        path=Path(argv[-1]);submitted=json.loads(path.read_text());output=root/'result';output.mkdir()
        worker_path=path.with_name(path.name.replace('-host.json','.json'));worker_sha=vista_agent.file_hash(worker_path)
        completed={'status':'ok','spec_sha256':worker_sha,'schema_version':1,
                   'synthetic':True,'quality_preserving_claim':False,'cohort_id':'fixture','batch_id':'fixture-batch'}
        if worker_kind=='metrics':
            artifact=output/'metric-provenance.json';write(artifact,{'synthetic_fixture_only':True})
            completed.update(kind='trial',real_lpips=False,artifacts={'metric-provenance.json':vista_agent.file_hash(artifact)})
            if corrupt=='metric_kind':completed['kind']='final'
            if corrupt=='real_lpips':completed.pop('real_lpips')
            if corrupt=='artifact':artifact.write_text('changed\n')
            write(output/'completed.json',completed)
        else:write(output/'adapter-completed.json',completed)
        terminal={'status':'completed','exit_code':0,'specification_sha256':vista_agent.file_hash(path),
            'launcher_sha256':sha,'launcher_version':'portable-v1' if portable else 'fixture-v1','source_unchanged':True,'inputs_unchanged':True,
            'input_file_hashes':submitted['expected_file_hashes'],'source_identity':'fixture',
            'cleanup':[{'status':'absent'}],'allocation_deadline_epoch':submitted['allocation_deadline_epoch'],
            'timeout_seconds':submitted['timeout_seconds'],'cleanup_margin_seconds':240,
            'environment_verification':submitted['environment_verification'],'environment_lock':submitted['environment_lock']}
        if portable:terminal.update(launch_runtime=deepcopy(host['runtime']),network_disabled=True)
        if corrupt=='cleanup':terminal['cleanup']=[{'status':'unknown'}]
        if corrupt=='spec':terminal['specification_sha256']='f'*64
        if corrupt=='input':source.write_text('changed\n')
        if corrupt=='timeout':raise TimeoutError('unknown termination')
        if corrupt=='worker':write(output/'adapter-completed.json',{'status':'ok','spec_sha256':worker_sha})
        if corrupt=='runtime':terminal['launch_runtime']['image_id']='sha256:'+'b'*64
        if corrupt=='network':terminal['network_disabled']=False
        write(root/submitted['metadata'],terminal)
        return SimpleNamespace(returncode=0)
    extra={}
    if isolated:
        package=root/'outputs/reviewed-revision'
        for name in ['drivebench/__init__.py','extensions/kernel_tuning/drivebench_tuning/__init__.py']:
            file=package/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_text('')
        extra={'package_root':'/workspace/outputs/reviewed-revision'}
    transport=trials.ExistingBoundedLauncher(root=root,launcher=launcher,launcher_sha256=sha,python=sys.executable,
        host_template=host,max_jobs=1,run=run,**extra)
    return root,transport,spec


def test_launcher_checks_terminal_exit_cleanup_provenance_and_job_cap(tmp_path,monkeypatch):
    root,transport,spec=launcher_fixture(tmp_path,monkeypatch)
    result=transport('batch',spec,spec_path=root/'spec.json',timeout_seconds=1,execution_deadline=time.time()+2000)
    assert result['worker_exit_verified'] and transport.used==1 and not transport.blocked
    assert json.loads((root/'spec-host.json').read_text())['allocation_deadline_epoch']==transport.template['allocation_deadline_epoch']
    with pytest.raises(RuntimeError,match='job cap'):transport('batch',spec,spec_path=root/'spec2.json',timeout_seconds=1,execution_deadline=time.time()+2000)
    assert result['completion']['status']=='ok'


@pytest.mark.parametrize('failure',['cleanup','spec','input','timeout','worker'])
def test_launcher_failed_terminal_receipt_halts_without_retry(tmp_path,monkeypatch,failure):
    root,transport,spec=launcher_fixture(tmp_path,monkeypatch,failure)
    with pytest.raises((RuntimeError,ValueError,TimeoutError)):
        transport('batch',spec,spec_path=root/'spec.json',timeout_seconds=1,execution_deadline=time.time()+2000)
    assert transport.blocked and transport.used==1 and (root/'spec.controller.log').exists()
    with pytest.raises(RuntimeError,match='halted'):
        transport('batch',spec,spec_path=root/'retry.json',timeout_seconds=1,execution_deadline=time.time()+2000)


def test_launcher_paths_and_deadline_fail_before_dispatch(tmp_path,monkeypatch):
    root,transport,spec=launcher_fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='fixed /workspace'):transport.host_path('/etc/secrets')
    with pytest.raises(ValueError,match='fixed /workspace'):transport.host_path('/workspace/../secrets')
    with pytest.raises(TimeoutError,match='cleanup do not fit'):
        transport('batch',spec,spec_path=root/'spec.json',timeout_seconds=1,execution_deadline=time.time()+240)
    assert transport.used==0 and not (root/'spec.json').exists()


def test_portable_runner_requires_operator_runtime_and_exact_hash(tmp_path):
    root=tmp_path/'host';launcher=root/trials.PORTABLE_LAUNCHER;launcher.parent.mkdir(parents=True)
    launcher.write_text('# pinned portable contract fixture\n');sha=vista_agent.file_hash(launcher)
    runtime={'image_id':'sha256:'+'a'*64,'python_executable':'/opt/venv/bin/python','environment_proof_kind':'fresh-gpu-v1'}
    host={key:None for key in trials.LAUNCHER_KEYS};host.update(root=str(root),runtime=runtime)
    transport=trials.ExistingBoundedLauncher(root=root,launcher=launcher,launcher_sha256=sha,
                 python=sys.executable,host_template=host,max_jobs=3)
    assert transport.launcher_version=='portable-v1'
    host.pop('runtime')
    with pytest.raises(ValueError,match='host launcher template'):
        trials.ExistingBoundedLauncher(root=root,launcher=launcher,launcher_sha256=sha,
                 python=sys.executable,host_template=host,max_jobs=3)
    with pytest.raises(ValueError,match='launcher hash changed'):
        trials.ExistingBoundedLauncher(root=root,launcher=launcher,launcher_sha256='b'*64,
                 python=sys.executable,host_template=host,max_jobs=3)


@pytest.mark.parametrize('failure',[None,'runtime','network'])
def test_portable_terminal_runtime_image_and_network_binding(tmp_path,monkeypatch,failure):
    root,transport,spec=launcher_fixture(tmp_path,monkeypatch,failure,portable=True)
    kwargs={'spec_path':root/'spec.json','timeout_seconds':1,'execution_deadline':time.time()+2000}
    if failure:
        with pytest.raises(RuntimeError,match='termination/provenance'):transport('batch',spec,**kwargs)
        assert transport.blocked
    else:
        receipt=transport('batch',spec,**kwargs)
        assert receipt['worker_exit_verified'] and not transport.blocked
        submitted=json.loads((root/'spec-host.json').read_text())
        assert submitted['runtime']==transport.template['runtime']
        assert submitted['allocation_deadline_epoch']==transport.template['allocation_deadline_epoch']


@pytest.mark.parametrize('failure',[None,'metric_kind','real_lpips','artifact'])
def test_metrics_completion_kind_boolean_and_artifacts_are_required(tmp_path,monkeypatch,failure):
    root,transport,spec=launcher_fixture(tmp_path,monkeypatch,failure,worker_kind='metrics')
    kwargs={'spec_path':root/'spec.json','timeout_seconds':1,'execution_deadline':time.time()+2000}
    if failure:
        with pytest.raises(ValueError,match='metric completion'):transport('metrics',spec,**kwargs)
        assert transport.blocked
    else:assert transport('metrics',spec,**kwargs)['worker_exit_verified']


def test_isolated_package_root_is_explicit_in_bounded_worker_bootstrap(tmp_path,monkeypatch):
    root,transport,spec=launcher_fixture(tmp_path,monkeypatch,isolated=True)
    transport('batch',spec,spec_path=root/'spec.json',timeout_seconds=1,execution_deadline=time.time()+2000)
    arguments=json.loads((root/'spec-host.json').read_text())['python_arguments']
    assert arguments[0]=='-c'
    assert "['/workspace/outputs/reviewed-revision/extensions/kernel_tuning', '/workspace/outputs/reviewed-revision']" in arguments[1]
    assert "['/workspace/extensions/kernel_tuning'" not in arguments[1]
