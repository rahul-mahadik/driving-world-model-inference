import json
import time
from pathlib import Path

import pytest

from drivebench_tuning.agent_cohort import POLICIES, ROLES, freeze_final_plan, selection_only_executor
from drivebench_tuning.agent_demo import run_demo
from drivebench_tuning.agent_scoring import QualityPolicy
from drivebench_tuning.agent_session import AgentSession, SessionPolicy
from drivebench_tuning.vista_agent import ALLOWED_VALUES, digest, reference_settings


@pytest.fixture(scope='module')
def quality(tmp_path_factory):
    directory = tmp_path_factory.mktemp('cohort-fixture')/'demo'
    run_demo(directory)
    return QualityPolicy.from_dict(json.loads((directory/'quality-policy.json').read_text()))


def prepare(tmp_path, quality, *, mode='normal'):
    sessions = {}
    for name in POLICIES:
        metadata = {'quality_policy_sha256':quality.sha256,'identity_sha256':quality.identity_sha256,
                    'source_manifest_sha256':digest('source'),'inputs_manifest_sha256':digest('inputs')}
        policy = SessionPolicy(session_id=name,cohort_id='test',reference_config=reference_settings(synthetic=True),
            allowed_config_values=ALLOWED_VALUES,wall_seconds=60,max_tool_calls=20,max_evaluations=5,max_profiles=0,
            metadata=metadata)
        def trial(proposal, context):
            return {'cohort_id':'test','status':'ok','screening_status':'no_observed_breach',
                    'candidate_sha256':digest(proposal['config']), **metadata}
        executor = selection_only_executor(trial)
        if name == 'grid' and mode == 'actual_final':
            original = executor
            def executor(kind, proposal, context):
                result = original(kind, proposal, context)
                if kind == 'final': result['evaluation_performed'] = True
                return result
        directory = tmp_path/name
        with AgentSession.create(directory,policy=policy,executor=executor) as session:
            proposal_id = session.call('propose',{'overrides':{'generation':{'decode_chunk_size':1}}})['result']['proposal_id']
            if not (name == 'grid' and mode == 'unmeasured'):
                session.call('trial',{'proposal_id':proposal_id})
            if not (name == 'grid' and mode == 'unsubmitted'):
                session.call('submit',{'proposal_id':proposal_id})
        sessions[name] = (directory,policy)
    return sessions


def freeze(path, sessions, quality, **overrides):
    options = dict(sessions=sessions,quality_policy=quality,
        block_orders={b:list(ROLES) for b in quality.to_dict()['plan']['timing']['block_scenes']},
        source_manifest_sha256=digest('source'),inputs_manifest_sha256=digest('inputs'),
        deadline_unix=time.time()+60,synthetic=True)
    options.update(overrides)
    return freeze_final_plan(path,**options)


def test_all_selections_lock_before_balanced_final_plan(tmp_path, quality):
    sessions=prepare(tmp_path, quality)
    result=freeze(tmp_path/'final.json',sessions,quality)
    assert len(result['batches']) == 16
    assert set(result['selections']) == set(POLICIES)
    assert not result['gpu_execution_performed'] and not result['quality_preserving_claim']
    assert not result['shared_reference_comparisons_independent']
    for index in (0,8):
        assert [b['role'] for b in result['batches'][index:index+4]] == list(ROLES)
        assert [b['role'] for b in result['batches'][index+4:index+8]] == list(reversed(ROLES))
    with pytest.raises(FileExistsError): freeze(tmp_path/'final.json',sessions,quality)


@pytest.mark.parametrize('mode,match', [('unsubmitted','all policy'),('unmeasured','no eligible'),('actual_final','only lock')])
def test_refuses_early_unmeasured_or_already_scored_selection(tmp_path, quality, mode, match):
    sessions=prepare(tmp_path,quality,mode=mode)
    with pytest.raises(ValueError,match=match):freeze(tmp_path/'final.json',sessions,quality)
    assert not (tmp_path/'final.json').exists()


def test_missing_policy_and_deadline_do_not_admit(tmp_path,quality):
    sessions=prepare(tmp_path,quality)
    with pytest.raises(ValueError,match='three'):freeze(tmp_path/'final.json',{'llm':sessions['llm']},quality)
    with pytest.raises(ValueError,match='time remaining'):freeze(tmp_path/'final.json',sessions,quality,deadline_unix=0)


def test_manifest_binding_and_tampered_ledger_refuse(tmp_path,quality):
    sessions=prepare(tmp_path,quality)
    with pytest.raises(ValueError,match='identity drift'):
        freeze(tmp_path/'final.json',sessions,quality,inputs_manifest_sha256=digest('changed inputs'))
    event=next((tmp_path/'grid'/'events').glob('*.json'))
    event.chmod(0o600)
    body=json.loads(event.read_text());body['at'] += 1;event.write_text(json.dumps(body))
    with pytest.raises(RuntimeError,match='hash chain'):
        freeze(tmp_path/'final.json',sessions,quality)


def test_profile_cannot_execute_and_submit_is_selection_only():
    calls=[]
    executor=selection_only_executor(lambda *args:calls.append(args))
    proposal={'proposal_id':'p','config':{}}
    with pytest.raises(ValueError,match='no live profile'):executor('profile',proposal,{'cohort_id':'c'})
    result=executor('final',proposal,{'cohort_id':'c'})
    assert calls == [] and result['selection_only'] and not result['quality_assessed']


def test_clock_rollback_between_closed_sessions_is_rejected(tmp_path,quality):
    sessions=prepare(tmp_path,quality)
    t=time.time()
    values=iter([t,t+10,t+10,t+5,t+5])
    with pytest.raises(ValueError,match='monotone'):
        freeze(tmp_path/'final.json',sessions,quality,clock=lambda:next(values))
    assert not (tmp_path/'final.json').exists()


def test_quality_policy_must_reconstruct_before_admission(tmp_path,quality):
    sessions=prepare(tmp_path,quality)
    changed=quality.to_dict();changed['calibration']['status']='inconclusive'
    bad=QualityPolicy(json.dumps(changed,sort_keys=True,separators=(',',':')))
    with pytest.raises(ValueError,match='digest'):
        freeze(tmp_path/'final.json',sessions,bad)
