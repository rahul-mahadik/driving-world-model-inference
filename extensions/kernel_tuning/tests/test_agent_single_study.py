"""Explicit single-agent study admission and actual synthetic final/export flow."""
from copy import deepcopy
import json
from pathlib import Path
import time

import pytest

from drivebench_tuning import agent_evidence as evidence, vista_agent
from drivebench_tuning.agent_cohort import (
    COMPARISON_MODE, SINGLE_LLM_MODE, POLICIES, freeze_final_plan,
    final_plan_policies, selection_only_executor, study_policies,
)
from drivebench_tuning.agent_session import AgentSession, SessionPolicy
from test_agent_cohort import quality, prepare as three_sessions, freeze as freeze_three
from test_agent_artifacts import final_fixture, perform
from test_agent_evidence import final_fixture as pair_fixture


@pytest.mark.parametrize('mode,names', [
    (SINGLE_LLM_MODE, POLICIES), (COMPARISON_MODE, ['llm']), ('unknown', ['llm']),
    (SINGLE_LLM_MODE, []), (SINGLE_LLM_MODE, ['grid']), (SINGLE_LLM_MODE, ['llm','llm']),
    (SINGLE_LLM_MODE, 'llm'), (SINGLE_LLM_MODE, {'llm'}), (True, ['llm']),
    (COMPARISON_MODE, ['grid','random','llm']), (SINGLE_LLM_MODE, [True]),
])
def test_mode_and_ordered_policy_names_must_match_exactly(mode,names):
    with pytest.raises(ValueError): study_policies(mode,names)


def test_default_three_policy_plan_retains_legacy_contract(tmp_path,quality):
    plan=freeze_three(tmp_path/'default.json',three_sessions(tmp_path,quality),quality)
    assert 'study_mode' not in plan and 'policy_names' not in plan
    assert final_plan_policies(plan)==POLICIES
    assert len(plan['batches'])==16 and set(plan['selections'])==set(POLICIES)


def single_session(root, quality, *, failure=None):
    metadata={'quality_policy_sha256':quality.sha256,'identity_sha256':quality.identity_sha256,
        'source_manifest_sha256':vista_agent.digest('source'),'inputs_manifest_sha256':vista_agent.digest('inputs'),
        'study_mode':SINGLE_LLM_MODE,'policy_names':['llm']}
    if failure=='unregistered':metadata.pop('study_mode')
    policy=SessionPolicy(session_id='llm',cohort_id='single-fixture',
        reference_config=vista_agent.reference_settings(synthetic=True),allowed_config_values=vista_agent.ALLOWED_VALUES,
        wall_seconds=120,max_tool_calls=10,max_evaluations=5,max_profiles=0,metadata=metadata)
    def trial(proposal, context):
        return {'cohort_id':'single-fixture','status':'ok','screening_status':'no_observed_breach',
            'candidate_sha256':vista_agent.digest(proposal['config']),**metadata}
    executor=selection_only_executor(trial)
    if failure=='already_evaluated':
        wrapped=executor
        def executor(kind,proposal,context):
            result=wrapped(kind,proposal,context)
            if kind=='final':result['evaluation_performed']=True
            return result
    directory=root/'llm'
    with AgentSession.create(directory,policy=policy,executor=executor) as session:
        pid=session.call('propose',{'overrides':{'generation':{'decode_chunk_size':2}}})['result']['proposal_id']
        if failure!='unmeasured':session.call('trial',{'proposal_id':pid})
        if failure!='unsubmitted':session.call('submit',{'proposal_id':pid})
    options={'sessions':{'llm':(directory,policy)},'quality_policy':quality,
        'block_orders':{b:['reference','llm_selected'] for b in quality.to_dict()['plan']['timing']['block_scenes']},
        'source_manifest_sha256':metadata['source_manifest_sha256'],'inputs_manifest_sha256':metadata['inputs_manifest_sha256'],
        'deadline_unix':time.time()+120,'synthetic':True,'study_mode':SINGLE_LLM_MODE,'policy_names':['llm']}
    return options


def test_single_admission_is_explicit_and_retains_reverse_pair_order(tmp_path,quality):
    options=single_session(tmp_path,quality)
    plan=freeze_final_plan(tmp_path/'plan.json',**options)
    assert plan['study_mode']==SINGLE_LLM_MODE and plan['policy_names']==['llm']
    assert final_plan_policies(plan)==('llm',) and set(plan['selections'])=={'llm'}
    assert len(plan['batches'])==8
    assert [r['role'] for r in plan['batches']]==['reference','llm_selected','llm_selected','reference']*2
    assert not plan['quality_preserving_claim'] and not plan['gpu_execution_performed']
    assert plan['sha256']==vista_agent.digest({k:v for k,v in plan.items() if k!='sha256'})
    with pytest.raises(FileExistsError):freeze_final_plan(tmp_path/'plan.json',**options)


@pytest.mark.parametrize('failure',['unregistered','unmeasured','unsubmitted','already_evaluated','alias','legacy_default','wrong_identity','tampered_ledger'])
def test_single_mode_does_not_relax_selection_and_identity_barrier(tmp_path,quality,failure):
    options=single_session(tmp_path,quality,failure=failure)
    if failure=='alias':options['sessions']['random']=options['sessions']['llm']
    elif failure=='legacy_default':options.pop('study_mode');options.pop('policy_names')
    elif failure=='wrong_identity':options['inputs_manifest_sha256']='0'*64
    elif failure=='tampered_ledger':
        event=next((tmp_path/'llm/events').glob('*.json'));data=json.loads(event.read_text());data['at']+=1
        event.chmod(0o600);event.write_text(json.dumps(data))
    with pytest.raises((ValueError,RuntimeError)):freeze_final_plan(tmp_path/'plan.json',**options)
    assert not (tmp_path/'plan.json').exists()


def single_pairs(tmp_path):
    refs,candidates,args,plan,seal=pair_fixture(tmp_path)
    plan.update(study_mode=SINGLE_LLM_MODE,policy_names=['llm'])
    plan['selections']={'llm':plan['selections']['llm']}
    plan['batches']=[r for r in plan['batches'] if r['role'] in ('reference','llm_selected')]
    for index,row in enumerate(plan['batches']):row['batch_index']=index
    args['expected_final_plan_sha256']=seal()
    return refs,candidates,args,plan,seal


@pytest.mark.parametrize('failure',[None,'partial_mode','missing_mode','wrong_mode','alias','foreign_role','late_barrier','changed_candidate','wrong_identity','wrong_order','wrong_cells','expired'])
def test_single_final_pairs_keep_identity_freshness_cells_and_order(tmp_path,failure):
    refs,candidates,args,plan,seal=single_pairs(tmp_path)
    if failure=='partial_mode':plan.pop('policy_names')
    elif failure=='missing_mode':plan.pop('study_mode');plan.pop('policy_names')
    elif failure=='wrong_mode':plan['study_mode']=COMPARISON_MODE
    elif failure=='alias':plan['selections']['random']=deepcopy(plan['selections']['llm'])
    elif failure=='foreign_role':plan['batches'][0]['role']='random_selected'
    elif failure=='late_barrier':plan['all_selections_locked_at']=time.time()
    elif failure=='changed_candidate':plan['selections']['llm']['candidate_sha256']='0'*64
    elif failure=='wrong_identity':plan['identity_sha256']='0'*64
    elif failure=='wrong_order':plan['batches'][0]['batch_index']=20
    elif failure=='wrong_cells':plan['batches'][0]['scenes']=['foreign']
    elif failure=='expired':plan['deadline_unix']=plan['all_selections_locked_at']
    args['expected_final_plan_sha256']=seal()
    if failure:
        with pytest.raises(ValueError):evidence.pair_rows(refs,candidates,**args)
    else:
        result=evidence.pair_rows(refs,candidates,**args)
        assert len(result['quality_rows'])==len(result['timing_rows'])==8
        assert {r['order'] for r in result['timing_rows']}=={'reference-first','candidate-first'}


@pytest.mark.parametrize('failure',[None,'missing_comparison','alias','duplicate','case_cap'])
def test_single_study_calibration_final_metrics_and_blind_export(tmp_path,failure):
    _,spec,_=final_fixture(tmp_path,study_mode=SINGLE_LLM_MODE,policy_names=('llm',))
    completion=Path(spec['comparisons'][0]['metric_completion']['path'])
    score=json.loads((completion.parent/'timing-score.json').read_text())
    assert score['status']=='pass' and score['paired_samples']==8
    assert not score['quality_preserving_claim'] and not score['confidence_claim']
    assert len(spec['comparisons'])==1 and spec['comparisons'][0]['policy']=='llm'
    if failure=='missing_comparison':spec['comparisons']=[]
    elif failure=='alias':spec['comparisons'][0]['policy']='grid'
    elif failure=='duplicate':spec['comparisons'].append(deepcopy(spec['comparisons'][0]))
    elif failure=='case_cap':
        path=Path(spec['review_plan']['path']);review=json.loads(path.read_text());review['maximum_cases']=1
        path.write_text(json.dumps(review));spec['review_plan']['sha256']=vista_agent.file_hash(path)
    if failure:
        with pytest.raises(ValueError):perform(tmp_path,spec,'export-spec.json')
        assert not (Path(spec['output'])/'completed.json').exists()
    else:
        result=perform(tmp_path,spec,'export-spec.json')
        assert result['result']['cases']==2 and not result['result']['visual_review_performed']
        key=json.loads((Path(spec['output'])/'private/rendering/private_key.json').read_text())
        assert len(key['cases'])==2
        assert (Path(spec['output'])/'public/index.html').is_file()
