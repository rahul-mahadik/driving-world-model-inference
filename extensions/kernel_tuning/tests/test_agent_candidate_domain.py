"""Prospective task domains: real CPU fixture workers, never a GPU or API call."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from drivebench_tuning import agent_trials as trials, vista_agent
from drivebench_tuning.agent_session import SessionPolicy
from test_agent_trials import episode, FixtureTransport, write


def fp16_domain():
    values=deepcopy(vista_agent.ALLOWED_VALUES)
    values['generation.dtype']=['fp16']
    return values


def candidate_policy(values):
    return SessionPolicy(session_id='domain',cohort_id='cpu-fixture',
        reference_config=vista_agent.reference_settings(synthetic=True),
        allowed_config_values=values,wall_seconds=100,max_tool_calls=10,
        max_evaluations=2,max_profiles=0)


@pytest.mark.parametrize('values', [vista_agent.ALLOWED_VALUES, fp16_domain(),
    {'generation.dtype':['fp16'],'generation.decode_chunk_size':[14],
     'optimizations.compile_denoiser':[False]}])
def test_full_and_reference_containing_typed_subsets_are_admitted(values):
    policy=candidate_policy(values)
    before=policy.to_dict()
    trials._validate_candidate_domain(policy)
    assert policy.to_dict()==before  # Do not reorder a prospectively pinned grid.


BAD_DOMAINS=[
    {},
    {'generation.dtype':['fp16']},
    {**fp16_domain(),'generation.steps':[2]},
    {**fp16_domain(),'generation.dtype':[]},
    {**fp16_domain(),'generation.dtype':['fp32']},
    {**fp16_domain(),'generation.dtype':['bf16']},  # Reference cannot escape the domain.
    {**fp16_domain(),'generation.decode_chunk_size':[1,7]},
    {**fp16_domain(),'generation.decode_chunk_size':[True,14]},
    {**fp16_domain(),'generation.decode_chunk_size':[1.0,14]},
    {**fp16_domain(),'optimizations.compile_denoiser':[0,True]},
    {**fp16_domain(),'optimizations.compile_denoiser':[False,1]},
    {**fp16_domain(),'generation.dtype':['fp16','fp16']},
    {**fp16_domain(),'generation.dtype':'fp16'},
    {**fp16_domain(),'generation.dtype':[{'dtype':'fp16'}]},
]


@pytest.mark.parametrize('entrypoint',['run','step'])
@pytest.mark.parametrize('values',BAD_DOMAINS)
def test_invalid_domain_fails_both_entrypoints_before_transport(tmp_path,monkeypatch,entrypoint,values):
    _,_,_,spec,path=episode(tmp_path,selector={'kind':'external-json'} if entrypoint=='step' else None)
    spec['session_policy']['allowed_config_values']=values
    write(path,spec)
    def forbidden(**kwargs):
        pytest.fail('invalid domain reached transport construction')
    monkeypatch.setattr(trials,'ExistingBoundedLauncher',forbidden)
    with pytest.raises(ValueError):
        if entrypoint=='run':trials.run_spec(path,expected_sha256=vista_agent.file_hash(path))
        else:trials.step_spec(path,expected_sha256=vista_agent.file_hash(path),operation='initialize')
    assert not Path(spec['output']).exists()


@pytest.mark.parametrize('mode',['grid','random','external-json'])
def test_all_policies_share_fp16_domain_and_preserve_fresh_trials(tmp_path,monkeypatch,mode):
    selector={'kind':mode} if mode=='external-json' else {'kind':mode,'seed':17}
    _,_,_,spec,path=episode(tmp_path,selector=selector,count=2)
    spec['session_policy']['allowed_config_values']=fp16_domain();write(path,spec)
    monkeypatch.setattr(trials,'ExistingBoundedLauncher',FixtureTransport)
    sha=vista_agent.file_hash(path)
    if mode=='external-json':
        # The LLM-facing route uses the same public allowlist and immutable policy.
        first=trials.step_spec(path,expected_sha256=sha,operation='initialize')
        assert first['allowed_config_values']==fp16_domain()
        def act(action,label):
            target=tmp_path/(label+'.json');write(target,action)
            return trials.step_spec(path,expected_sha256=sha,operation='act',action_path=target,
                                    action_sha256=vista_agent.file_hash(target))
        denied=act({'tool':'propose','arguments':{'overrides':{'generation':{'dtype':'bf16'}}}},'unsupported')
        assert denied['last_action_succeeded'] is False and denied['trials_used']==0
        assert denied['latest_proposal']==first['latest_proposal']
        for i,chunk in enumerate([14,7]):
            proposed=act({'tool':'propose','arguments':{'overrides':{'generation':{'decode_chunk_size':chunk}}}},f'proposal-{i}')
            pid=proposed['latest_proposal']['proposal_id']
            observed=act({'tool':'trial','arguments':{'proposal_id':pid}},f'trial-{i}')
            assert observed['last_action_succeeded'] and observed['trials_used']==i+1
        selected=act({'tool':'submit','arguments':{'proposal_id':pid}},'submit')
        assert selected['status']=='completed'
        result=json.loads((Path(spec['output'])/'completed.json').read_text())
    else:
        result=trials.run_spec(path,expected_sha256=sha)
    assert result['session_policy']['allowed_config_values']==fp16_domain()
    assert result['jobs_used']==6 and result['selection_only'] and not result['final_evaluation_performed']
    assert not result['quality_preserving_claim']
    for specfile in Path(spec['output']).glob('trials/*/candidate-spec.json'):
        candidate=json.loads(specfile.read_text())
        assert candidate['settings']['generation']['dtype']=='fp16'
        assert len(candidate['requests'])==4
    for mode in ['grid','random']:
        baseline=trials.SearchBaseline(mode=mode,seed=17,
            reference=spec['session_policy']['reference_config'],allowed_values=fp16_domain())
        assert len(baseline.configs)==8
        assert {c['generation']['dtype'] for c in baseline.configs}=={'fp16'}
