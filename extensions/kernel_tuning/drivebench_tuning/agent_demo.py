"""Small end-to-end CPU contract demo. No real model, GPU, LPIPS, or LLM result."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import time

import numpy as np

from .agent_metrics import measure_pair, negative_control
from .agent_scoring import freeze_quality_policy, assess_proxy_quality, score_heldout_timing
from .agent_session import AgentSession, SessionPolicy
from .agent_policy import run_agent
from .vista_agent import BatchRequest, ALLOWED_VALUES, digest, file_hash, probe_runtime, reference_settings, run_batch


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    with path.open('x') as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write('\n')


class _PerceptualFixture:
    """Numeric contract test double; explicitly NOT LPIPS."""
    def compare(self, reference, candidate):
        return float(np.abs(candidate - reference).mean())


def run_demo(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    planned = now()
    settings = reference_settings(synthetic=True)
    runtime = probe_runtime(synthetic=True)
    identity = {'runtime_id': digest(runtime), 'reference_id': digest(settings), 'evidence_kind': 'synthetic',
                'metric_definition_sha256': digest({'fixture': 'mae substitutes LPIPS only in CPU contract test'})}
    identity_hash = digest(identity)
    dev, final, seeds, reps = ['dev-a','dev-b'], ['final-a','final-b'], [11,23], [0,1]
    deadline = time.time()+120
    def requests(scenes, repetitions):
        return [BatchRequest(f'{scene}-{seed}-{rep}',scene,seed,rep,np.zeros((1,3,8,8),np.float32),np.zeros(8,np.float32))
                for scene,seed,rep in itertools.product(scenes,seeds,repetitions)]
    def execute(name, config, reqs, cutoff=deadline):
        directory = output/name
        result = run_batch(directory, settings=config, model={'backend':'synthetic'}, requests=reqs,
                           cohort_id='cpu-agent-fixture', expected_runtime=runtime, source_files={},
                           deadline_unix=cutoff, synthetic=True, minimum_free_bytes=0)
        return [{**row, 'path': directory/row['output_file']} for row in result['rows']]
    def distances(a,b):
        return measure_pair(a['path'],b['path'],reference_sha256=a['output_sha256'],
                            candidate_sha256=b['output_sha256'],perceptual_metric=_PerceptualFixture())
    def common(row, label):
        return {key: row[key] for key in ('scene_id','seed','replicate','input_sha256','output_sha256')} | {
            'record_id': label, 'conditioning_sha256': digest({'scene':row['scene_id'],'conditioning':'zero','action':'zero'}),
            'identity_sha256':identity_hash,'observed_at':row['observed_at']}
    calibration = execute('calibration',settings,requests(dev,reps))
    cells = {(r['scene_id'],r['seed'],r['replicate']):r for r in calibration}
    rows,controls = [],[]
    for i,row in enumerate(calibration):
        repeat = cells[(row['scene_id'],row['seed'],1-row['replicate'])]
        other_seed = cells[(row['scene_id'],seeds[1-seeds.index(row['seed'])],row['replicate'])]
        rows.append(common(row,f'cal-{i}') | {'repeat_reference_sha256':repeat['output_sha256'],
            'seed_reference_sha256':other_seed['output_sha256'], 'repeat_metrics':distances(row,repeat),
            'seed_metrics':distances(row,other_seed)})
        for control in ('freeze_future','black_future','reverse_future'):
            path=output/f'control-{control}-{i}.npy'
            np.save(path,negative_control(np.load(row['path'],allow_pickle=False),control),allow_pickle=False)
            altered={**row,'path':path,'output_sha256':file_hash(path)}
            controls.append(common(altered,f'control-{control}-{i}') | {'control_id':control,
                'reference_output_sha256':row['output_sha256'],'metrics':distances(row,altered)})
    policy=freeze_quality_policy(identity=identity, metric_margins={k:1e-5 for k in ('mae','temporal_derivative_mae','lpips')},
        development_scenes=dev,final_scenes=final,seeds=seeds,replicates=reps,
        calibration_records=rows,negative_controls=controls,plan_created_at=planned,frozen_at=now(),
        minimum_development_scenes=2,minimum_final_scenes=2,timing_min_blocks=2,timing_noise_floor=0.99)
    write(output/'quality-policy.json',policy.to_dict())
    selection_started=now()
    call_count=0
    def executor(kind,proposal,context):
        nonlocal call_count
        call_count+=1
        cutoff=context['execution_deadline_unix']
        if kind in ('trial','profile'):
            observed=execute(f'{kind}-{call_count}',proposal['config'],requests(dev,[0]),cutoff)
            return {'cohort_id':'cpu-agent-fixture','status':'ok','synthetic':True,
                    'latency_ms':[r['latency_ms'] for r in observed],
                    'stage_timings_s':[r['stage_timings_s'] for r in observed],
                    'quality_status':'inconclusive: development subset only','reward':0.0}
        pairs, quality_rows = [],[]
        for rep in reps:
            configs = [('reference',settings),('candidate',proposal['config'])]
            if rep:configs.reverse()
            collected={name:execute(f'final-{rep}-{name}',config,requests(final,[rep]),cutoff) for name,config in configs}
            for a,b in zip(collected['reference'],collected['candidate']):
                q=common(b,f'quality-{b["record_id"]}') | {'candidate_sha256':digest(proposal['config']),
                    'reference_output_sha256':a['output_sha256'],'metrics':distances(a,b)}
                quality_rows.append(q)
                pairs.append({'record_id':f'pair-{b["record_id"]}','scene_id':b['scene_id'],'seed':b['seed'],'repeat':rep,
                    'block_id':f'block-{final.index(b["scene_id"])}','identity_sha256':identity_hash,
                    'candidate_sha256':digest(proposal['config']),'input_sha256':b['input_sha256'],
                    'observed_at':max(a['observed_at'],b['observed_at']),
                    'reference_measurement_id':f'reference-{a["record_id"]}',
                    'candidate_measurement_id':f'candidate-{b["record_id"]}',
                    'reference_latency_ms':a['latency_ms'],'candidate_latency_ms':b['latency_ms'],
                    'order':'candidate-first' if rep else 'reference-first'})
        assessment=assess_proxy_quality(policy,quality_rows,split='final',selection_started_at=selection_started)
        score=score_heldout_timing(policy,pairs,quality_assessment=assessment,selection_started_at=selection_started)
        write(output/'final-quality.json',assessment)
        write(output/'final-score.json',score)
        return {'cohort_id':'cpu-agent-fixture','status':'ok','synthetic':True,'quality':assessment['status'],'score':score}
    session_policy=SessionPolicy(session_id='cpu-agent-demo',cohort_id='cpu-agent-fixture',reference_config=settings,
        allowed_config_values=ALLOWED_VALUES,wall_seconds=90,max_tool_calls=8,max_evaluations=3,max_profiles=1,
        final_evaluations=1,final_reserve_seconds=10,metadata={'evidence_kind':'synthetic','quality_policy_sha256':policy.sha256})
    with AgentSession.create(output/'session',policy=session_policy,executor=executor) as session:
        class ScriptedSelector:
            index = 0
            configuration = {'kind': 'scripted CPU contract fixture; no LLM'}
            def __call__(self, observation):
                actions = [
                    {'tool': 'propose', 'arguments': {'overrides': {'generation': {'decode_chunk_size': 1}}}},
                    {'tool': 'profile', 'arguments': {'proposal_id': observation['latest_proposal_id']}},
                    {'tool': 'trial', 'arguments': {'proposal_id': observation['latest_proposal_id']}},
                    {'tool': 'submit', 'arguments': {'proposal_id': observation['latest_proposal_id']}},
                ]
                action = actions[self.index]
                self.index += 1
                return action
        def public_observation(state, outcome):
            # Deliberate disclosure: no evaluator paths or final data sent to a policy.
            return {'task': 'synthetic CPU contract fixture', 'allowed_values': ALLOWED_VALUES,
                    'latest_proposal_id': state['latest_proposal_id'],
                    'remaining_evaluations': state['evaluations_remaining'],
                    'last_call_ok': outcome['ok'] if outcome else None}
        loop = run_agent(session, ScriptedSelector(), observation_builder=public_observation, max_turns=4)
        final_outcome = loop['turns'][-1]['outcome']
        report={'scope':'CPU synthetic engine + numeric metric fixture + scripted policy; no GPU, LPIPS, Vista or LLM validation',
                'quality_calibration_status':policy.to_dict()['calibration']['status'],
                'real_quality_calibrated':False,'real_gpu_validation':False,'llm_demonstration':False,
                'final_state':session.state,'final_outcome':final_outcome,'agent_loop':loop,'completed_at':now()}
    write(output/'demo-report.json',report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    report=run_demo(args.output)
    print(json.dumps({'scope':report['scope'],'status':report['final_state']['status'],
                      'output':str(Path(args.output).resolve())},indent=2))


if __name__=='__main__':main()
