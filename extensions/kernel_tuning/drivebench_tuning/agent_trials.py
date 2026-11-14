"""Concrete config-only trial executor using an existing bounded host launcher.

The controller runs on the trusted GPU host, outside the launcher container. It
never provisions resources or opens SSH connections. The operator pins the
existing launcher, proof/lock, source files and worker templates. Every GPU job
uses that launcher; no Docker commands, retries or new supervisor live here.
The outer allocation guard and private controller filesystem remain required.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time

from .agent_budget import APICostLedger, APICostPolicy, BudgetedResponsesClient
from .agent_cohort import selection_only_executor
from .agent_policy import ResponsesActionPolicy, run_agent, validate_action
from .agent_scoring import QualityPolicy
from .agent_session import AgentSession, SessionPolicy, SessionError, _read, _write_new
from .vista_agent import ALLOWED_VALUES, digest, file_hash, reference_settings, validate_settings
from .vista_batch import _hash, _json, _keys


MOUNT = Path('/workspace')
LAUNCHERS = {'outputs/host/run-bounded-job-v2.py': (2, '2ea0ba70344a7b089ea28518459bcf8659f0eccdf352a069b6dced32fa029497'),
             'outputs/host/run-bounded-a100-job-v1.py': ('a100-diagnostic-v1', '8acf1240512bc75ef69ab41888794cb3fadd8ce83835b68c67121e98618d99a3')}
PORTABLE_LAUNCHER='tools/host/run-bounded-portable-v1.py'
WORKERS = {'batch': ('drivebench_tuning.vista_batch', 'adapter-completed.json'),
           'metrics': ('drivebench_tuning.agent_metrics_worker', 'completed.json')}
LAUNCHER_KEYS = {'root', 'metadata', 'container_name', 'fresh_outputs', 'timeout_seconds',
                'allocation_deadline_epoch', 'expected_source_identity', 'expected_file_hashes',
                'python_arguments', 'scope', 'environment_verification', 'environment_lock'}


def _now(): return datetime.now(timezone.utc).isoformat()

def _finite(value, name, positive=True):
    if type(value) not in (int, float) or not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f'invalid {name}')
    return value


def _pinned(path, sha):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or file_hash(path) != _hash(sha, 'artifact'):
        raise ValueError('pinned file missing or changed')
    return _read(path)


class ExistingBoundedLauncher:
    """Run the reviewed v2 host launcher; verify its terminal metadata.

    host_template has the existing launcher's exact schema. Dynamic argv,
    metadata, names, outputs, timeout and scope are filled by this adapter.
    It is never legitimate to run this controller *inside* the launcher: the
    controller must be outside the single-GPU lock held by each worker.
    """
    def __init__(self, *, root, launcher, launcher_sha256, python, host_template,
                 max_jobs, package_root='/workspace', clock=time.time, run=subprocess.run):
        self.root = Path(root).resolve()
        self.launcher = Path(launcher).resolve()
        self.python = str(Path(python).resolve())
        if not self.root.is_dir() or not self.launcher.is_relative_to(self.root):
            raise ValueError('launcher must be under the existing absolute host project root')
        if not Path(self.python).is_file(): raise ValueError('existing host Python required')
        if file_hash(self.launcher) != _hash(launcher_sha256, 'launcher'):
            raise ValueError('existing launcher hash changed')
        relative_launcher=self.launcher.relative_to(self.root).as_posix()
        portable=relative_launcher==PORTABLE_LAUNCHER
        selected=('portable-v1',launcher_sha256) if portable else LAUNCHERS.get(relative_launcher)
        if selected is None or selected[1]!=launcher_sha256:
            raise ValueError('unsupported or changed bounded host launcher')
        self.launcher_version=selected[0]
        _keys(host_template, LAUNCHER_KEYS | ({'runtime'} if portable else set()), 'host launcher template')
        if portable:
            _keys(host_template['runtime'],{'image_id','python_executable','environment_proof_kind'},'pinned launch runtime')
        if Path(host_template['root']).resolve() != self.root: raise ValueError('host root mismatch')
        if type(max_jobs) is not int or max_jobs <= 0: raise ValueError('positive job cap required')
        self.template = deepcopy(host_template)
        self.package_root=Path(package_root)
        package_host=self.host_path(package_root)
        if package_root!='/workspace' and (not package_host.is_dir() or
            not (package_host/'drivebench/__init__.py').is_file() or
            not (package_host/'extensions/kernel_tuning/drivebench_tuning/__init__.py').is_file()):
            raise ValueError('explicit isolated package root must contain both frozen core and tuning package')
        self.launcher_sha256, self.max_jobs = launcher_sha256, max_jobs
        self.clock, self.run, self.used, self.blocked = clock, run, 0, False

    def host_path(self, value):
        path = Path(value)
        if not path.is_absolute() or not path.is_relative_to(MOUNT) or '..' in path.parts:
            raise ValueError('worker paths must be under the fixed /workspace mount')
        out = self.root / path.relative_to(MOUNT)
        if out.is_symlink() or not out.resolve().is_relative_to(self.root):
            raise ValueError('worker path escapes host project')
        return out

    def container_path(self, path):
        path = Path(path).resolve()
        if not path.is_relative_to(self.root): raise ValueError('controller output must be inside host project')
        return str(MOUNT / path.relative_to(self.root))

    def __call__(self, kind, spec, *, spec_path, timeout_seconds, execution_deadline):
        if self.blocked or self.used >= self.max_jobs: raise RuntimeError('host transport halted or job cap exhausted')
        if kind not in WORKERS: raise ValueError('unknown trusted worker')
        if type(timeout_seconds) is not int or timeout_seconds < 1: raise ValueError('positive integer worker timeout required')
        now = self.clock(); _finite(now, 'clock')
        deadline = min(_finite(execution_deadline, 'execution deadline'), self.template['allocation_deadline_epoch'])
        # The existing launcher reserves240s after each worker. Its whole process
        # is bounded by that allowance; a timeout here is unknown cleanup and
        # permanently halts this transport rather than authorizing another job.
        if now + timeout_seconds + 240 >= deadline: raise TimeoutError('worker and cleanup do not fit')
        spec_path = Path(spec_path).resolve()
        if not spec_path.is_relative_to(self.root) or spec_path.exists():
            raise ValueError('fresh worker spec under host project required')
        if self.host_path(spec['output']).exists(): raise FileExistsError('worker output already exists')
        if spec['deadline_unix'] > deadline: raise ValueError('worker deadline exceeds controller admission')
        for source, expected in spec['source_files' if kind == 'batch' else 'sources'].items():
            if file_hash(self.host_path(source)) != expected: raise ValueError('worker source drift')
        _write_new(spec_path, spec); spec_sha = file_hash(spec_path)
        self.used += 1
        host = deepcopy(self.template)
        relative = spec_path.relative_to(self.root).as_posix()
        token = digest({'spec':spec_sha,'job':self.used})[:24]
        metadata = f'outputs/host/agent-{token}.json'
        module, completion = WORKERS[kind]
        search_paths=[str(self.package_root/'extensions/kernel_tuning'),str(self.package_root)]
        bootstrap = 'import sys;sys.path[:0]='+repr(search_paths)+';from ' + module + ' import main;raise SystemExit(main())'
        host.update(metadata=metadata, container_name='drivebench-agent-'+token,
                    fresh_outputs=[self.host_path(spec['output']).relative_to(self.root).as_posix()],
                    timeout_seconds=timeout_seconds,
                    python_arguments=['-c',bootstrap,'--spec',str(MOUNT/relative),'--spec-sha256',spec_sha],
                    scope='config-only agent '+kind+'; fixed inputs; no quality-preservation claim')
        hashes = host['expected_file_hashes']
        hashes[relative] = spec_sha
        for source, expected in spec['source_files' if kind == 'batch' else 'sources'].items():
            hashes[self.host_path(source).relative_to(self.root).as_posix()] = expected
        hashes[self.launcher.relative_to(self.root).as_posix()] = self.launcher_sha256
        if any(file_hash(self.root/name) != sha for name,sha in hashes.items()): raise ValueError('host job input drift')
        host_path = spec_path.with_name(spec_path.stem+'-host.json'); _write_new(host_path,host)
        self.blocked = True  # Clear only after exact successful termination evidence.
        with spec_path.with_suffix('.controller.log').open('x') as out:
            process = self.run([self.python,str(self.launcher),str(host_path)],cwd=self.root,
                               stdout=out,stderr=subprocess.STDOUT,
                               timeout=timeout_seconds+240,check=False)
        terminal_path = self.root/metadata; terminal = _read(terminal_path)
        if (process.returncode != 0 or terminal.get('status') != 'completed' or terminal.get('exit_code') != 0 or
            terminal.get('specification_sha256') != file_hash(host_path) or
            terminal.get('launcher_sha256') != self.launcher_sha256 or terminal.get('launcher_version') != self.launcher_version or
            terminal.get('source_unchanged') is not True or terminal.get('inputs_unchanged') is not True or
            terminal.get('input_file_hashes') != hashes or terminal.get('source_identity') != host['expected_source_identity'] or
            not terminal.get('cleanup') or terminal['cleanup'][-1].get('status') not in ('absent','removed') or
            terminal.get('allocation_deadline_epoch') != host['allocation_deadline_epoch'] or
            terminal.get('timeout_seconds') != timeout_seconds or terminal.get('cleanup_margin_seconds') != 240 or
            terminal.get('environment_verification') != host['environment_verification'] or
            terminal.get('environment_lock') != host['environment_lock'] or
            (self.launcher_version=='portable-v1' and (terminal.get('launch_runtime')!=host['runtime'] or
                                                       terminal.get('network_disabled') is not True)) or
            self.clock() >= deadline):
            raise RuntimeError('bounded launcher termination/provenance did not verify')
        for source, sha in hashes.items():
            if file_hash(self.root/source) != sha: raise ValueError('job inputs drifted after termination')
        end = self.host_path(spec['output'])/completion
        result = _read(end)
        if (result.get('spec_sha256') != spec_sha or result.get('status') != 'ok' or
            result.get('schema_version')!=1 or result.get('synthetic') is not spec['synthetic'] or
            result.get('quality_preserving_claim') is not False):
            raise ValueError('worker completion binding failed')
        if kind=='batch':
            if result.get('cohort_id')!=spec['cohort_id'] or result.get('batch_id')!=spec['batch_id']:
                raise ValueError('batch completion identity failed')
        else:
            if result.get('kind')!=spec['kind'] or result.get('real_lpips') is not (not spec['synthetic']):
                raise ValueError('metric completion kind failed')
            directory=end.parent;artifacts=result.get('artifacts')
            paths=list(directory.rglob('*'))
            if any(p.is_symlink() for p in paths):raise ValueError('metric artifact symlink')
            actual={p.relative_to(directory).as_posix():file_hash(p) for p in paths if p.is_file() and p!=end}
            if artifacts!=actual:raise ValueError('metric completion artifacts changed')
        self.blocked = False
        return {'spec_sha256':spec_sha,'completion_sha256':file_hash(end),'completion':result,
                'terminal_receipt':str(terminal_path),'terminal_receipt_sha256':file_hash(terminal_path),
                'worker_exit_verified':True,'directory':spec['output']}


def public_observation(session):
    """Project only allowed configs, charged budgets and scalar development feedback.

    Raw outcomes, private paths, errors, source/runtime identities, input/scene
    identifiers, metric assets and final observations never enter this view.
    """
    def build(state, last_outcome):
        latest = session._proposal(state['latest_proposal_id'])
        history = []; pending = None
        for p in sorted((session.directory/'events').glob('*.json')):
            event = _read(p)
            if event['type'] == 'execution_started': pending = event['payload']
            elif event['type'] in ('tool_finished','tool_failed') and pending:
                if pending['kind'] == 'trial':
                    result = event['payload'].get('result',{})
                    value = result.get('candidate_reference_latency_ratio')
                    eligible = (event['type']=='tool_finished' and result.get('status')=='ok' and
                                result.get('screening_status')=='no_observed_breach' and
                                type(value) in (int,float) and math.isfinite(value) and value>0)
                    history.append({'proposal_id':pending['proposal_id'],
                                    'config':session._proposal(pending['proposal_id'])['config'],
                                    'eligible':eligible,'candidate_reference_latency_ratio':value if eligible else None})
                pending = None
        return {'task':'Tune the declared Vista settings using charged development trials, then submit one tried proposal.',
                'allowed_config_values':session.policy.to_dict()['allowed_config_values'],
                'trial_budget':session.policy.max_evaluations-session.policy.final_evaluations,
                'trials_used':len(history),'tool_calls_used':state['tool_calls_used'],
                'max_tool_calls':session.policy.max_tool_calls,
                'latest_proposal':{'proposal_id':latest['proposal_id'],'config':latest['config']},
                'trial_history':history,'status':state['status'],
                'last_action_succeeded':None if last_outcome is None else bool(last_outcome.get('ok')),
                'final_feedback_available':False}
    return build


class SearchBaseline:
    """Equal-trial-budget random-without-replacement or fixed-grid selector.

    Actions derive entirely from the public observation, so a new selector
    object cannot reset consumed attempts. A failure remains a charged trial.
    Selection uses only eligible development measurements, never final data.
    """
    def __init__(self, *, mode, seed, reference, allowed_values):
        if mode not in ('random','grid') or type(seed) is not int: raise ValueError('explicit random/grid policy and integer seed required')
        self.configuration={'policy':mode,'seed':seed,'reference':deepcopy(reference),'allowed_config_values':deepcopy(allowed_values)}
        fields=sorted(allowed_values); self.configs=[]
        for values in itertools.product(*(allowed_values[f] for f in fields)):
            config=deepcopy(reference)
            for field,value in zip(fields,values):
                section,key=field.split('.');config[section][key]=value
            self.configs.append(config)
        if mode=='random': random.Random(seed).shuffle(self.configs)

    def __call__(self, observation):
        history=observation['trial_history'];budget=observation['trial_budget']
        if not 0<budget<=len(self.configs): raise ValueError('trial budget exceeds declared candidate space')
        if len(history)>=budget:
            eligible=[r for r in history if r['eligible']]
            if not eligible: raise SessionError('no measured candidate met provisional development screening')
            best=min(eligible,key=lambda r:r['candidate_reference_latency_ratio'])
            return {'tool':'submit','arguments':{'proposal_id':best['proposal_id']}}
        target=self.configs[len(history)];latest=observation['latest_proposal']
        if latest['config']==target:
            return {'tool':'trial','arguments':{'proposal_id':latest['proposal_id']}}
        overrides={}
        for field in self.configuration['allowed_config_values']:
            section,key=field.split('.')
            overrides.setdefault(section,{})[key]=target[section][key]
        return {'tool':'propose','arguments':{'overrides':overrides}}


class PairedTrialExecutor:
    """One charged trial = fresh reference + candidate + bounded metrics worker."""
    def __init__(self, *, directory, transport, batch_template, metric_template,
                 quality_policy, timeouts, max_trials, selection_started_at,
                 clock=time.time, resume=False, trial_schedule=None):
        self.directory=Path(directory).resolve()
        if not resume:self.directory.mkdir(exist_ok=False)
        elif self.directory.is_symlink() or not self.directory.is_dir():raise ValueError('real existing trial directory required')
        self.transport,self.batch,self.metric=transport,deepcopy(batch_template),deepcopy(metric_template)
        self.quality=QualityPolicy.from_dict(quality_policy.to_dict())
        self.timeouts=deepcopy(timeouts);self.max_trials=max_trials;self.started_at=selection_started_at
        self.clock,self.used,self.blocked=clock,0,False
        _keys(timeouts,{'reference','candidate','metrics'},'trial timeouts')
        if any(type(v) is not int or v<1 for v in timeouts.values()):raise ValueError('positive integer stage timeouts required')
        if type(max_trials) is not int or max_trials<1:raise ValueError('positive trial cap required')
        if self.quality.to_dict()['calibration']['status']!='ready':raise ValueError('ready development calibration required')
        if self.metric['kind']!='trial':raise ValueError('trial metrics template required')
        if self.metric['quality_policy']['sha256']!=file_hash(transport.host_path(self.metric['quality_policy']['path'])):
            raise ValueError('quality policy file changed')
        supplied=QualityPolicy.from_dict(_read(transport.host_path(self.metric['quality_policy']['path'])))
        if supplied.sha256!=self.quality.sha256:raise ValueError('metrics quality policy differs')
        cells=lambda rows:{(r['scene_id'],r['seed'],r['replicate']) for r in rows}
        if cells(self.batch['requests'])!=cells(self.metric['trial_cells']):raise ValueError('trial template cells differ')
        if len(cells(self.batch['requests']))!=len(self.batch['requests']):raise ValueError('trial cells duplicate')
        self.schedule=deepcopy(trial_schedule)
        if self.schedule is not None:
            if not isinstance(self.schedule,list) or len(self.schedule)!=max_trials:
                raise ValueError('trial schedule must contain exactly one cell selection per admitted trial')
            for choice in self.schedule:
                _keys(choice,{'seed','replicate'},'trial schedule cell')
                if any(type(v) is not int or v<0 for v in choice.values()):raise ValueError('nonnegative scheduled seed and replicate required')
                selected=[r for r in self.batch['requests'] if all(r[k]==v for k,v in choice.items())]
                if not selected or {r['scene_id'] for r in selected}!={r['scene_id'] for r in self.batch['requests']}:
                    raise ValueError('scheduled cell must cover every fixed trial scene')
        if self.batch['settings']!=reference_settings(synthetic=self.batch['synthetic']):raise ValueError('canonical paired reference required')
        if resume:
            attempts=sorted(self.directory.iterdir())
            if any(not p.is_dir() or p.is_symlink() or p.name!=f'{i:04d}' for i,p in enumerate(attempts)):
                raise ValueError('trial attempt sequence changed')
            self.used=len(attempts)
            if self.used>max_trials:raise ValueError('persisted trials exceed cap')
            for attempt in attempts:
                if not (attempt/'completed.json').is_file() or (attempt/'failed.json').exists():
                    self.blocked=True
                    continue
                result=_read(attempt/'completed.json')
                for worker in result['worker_receipts'].values():
                    _pinned(worker['terminal_receipt'],worker['terminal_receipt_sha256'])
                    completion=transport.host_path(worker['directory'])/('completed.json' if worker['directory'].endswith('/metrics') else 'adapter-completed.json')
                    _pinned(completion,worker['completion_sha256'])
            transport.used=len(list(self.directory.glob('*/*-host.json')))
            if transport.used>transport.max_jobs:raise ValueError('persisted jobs exceed cap')
            transport.blocked=self.blocked

    def __call__(self,proposal,context):
        if self.blocked or self.used>=self.max_trials:raise RuntimeError('paired trial executor halted or exhausted')
        settings=validate_settings(proposal['config'],synthetic=self.batch['synthetic'])
        if context['cohort_id']!=self.batch['cohort_id']:raise ValueError('trial/session cohort mismatch')
        metadata=context['metadata']
        expected={'quality_policy_sha256':self.quality.sha256,'identity_sha256':self.quality.identity_sha256,
                  'source_manifest_sha256':self.metric['study_bindings']['source_manifest']['sha256'],
                  'inputs_manifest_sha256':self.metric['study_bindings']['input_manifest']['sha256']}
        if any(metadata.get(k)!=v for k,v in expected.items()):raise ValueError('trial/session frozen identity mismatch')
        deadline=min(context['execution_deadline_unix'],self.batch['deadline_unix'],self.metric['deadline_unix'])
        if self.clock()+sum(self.timeouts.values())+3*240>=deadline:raise TimeoutError('whole trial and three cleanup margins do not fit')
        index=self.used;self.used+=1;directory=self.directory/f'{index:04d}';directory.mkdir()
        requests=deepcopy(self.batch['requests'])
        if self.schedule is not None:
            requests=[r for r in requests if all(r[k]==v for k,v in self.schedule[index].items())]
        trial_cells=[{k:r[k] for k in ('scene_id','seed','replicate')} for r in requests]
        _write_new(directory/'intent.json',{'proposal_id':proposal['proposal_id'],'candidate_sha256':digest(settings),
            'cohort_id':context['cohort_id'],'created_at':_now(),'deadline_unix':deadline,
            'trial_cells':trial_cells,
            'order':['reference','candidate'] if index%2==0 else ['candidate','reference']})
        self.blocked=True
        try:
            executed={}
            for role in (['reference','candidate'] if index%2==0 else ['candidate','reference']):
                spec=deepcopy(self.batch);spec.update(batch_id=f'{directory.parent.parent.name}-trial-{index}-{role}',
                    settings=self.batch['settings'] if role=='reference' else settings,
                    requests=requests,
                    output=self.transport.container_path(directory/role),created_at=_now(),deadline_unix=deadline)
                executed[role]=self.transport('batch',spec,spec_path=directory/f'{role}-spec.json',
                    timeout_seconds=self.timeouts[role],execution_deadline=deadline)
            metric=deepcopy(self.metric);metric.update(output=self.transport.container_path(directory/'metrics'),
                created_at=_now(),deadline_unix=deadline,selection_started_at=self.started_at,candidate_sha256=digest(settings),
                trial_cells=trial_cells)
            for key,role in [('references','reference'),('candidates','candidate')]:
                job=executed[role]
                metric[key]=[{'directory':job['directory'],'role':role,
                              'expected_spec_sha256':job['spec_sha256'],
                              'expected_completion_sha256':job['completion_sha256']}]
            scored=self.transport('metrics',metric,spec_path=directory/'metrics-spec.json',
                timeout_seconds=self.timeouts['metrics'],execution_deadline=deadline)
            if scored['completion'].get('cohort_id')!=context['cohort_id']:
                raise ValueError('development metric cohort binding failed')
            result=scored['completion']['result']
            if (any(result.get(k)!=v for k,v in expected.items() if k in ('quality_policy_sha256','identity_sha256')) or
                result.get('candidate_sha256')!=digest(settings) or
                result.get('screening_status') not in ('no_observed_breach','breach') or
                result.get('quality_assessed') is not False):raise ValueError('development metric result binding failed')
            a=_finite(result['reference_median_latency_ms'],'reference latency')
            b=_finite(result['candidate_median_latency_ms'],'candidate latency')
            ratio=b/a;_finite(ratio,'paired latency ratio')
            result={**result,'cohort_id':context['cohort_id'],'status':'ok',
                    'candidate_reference_latency_ratio':ratio,'proposal_id':proposal['proposal_id'],
                    'worker_exit_verified':True,'final_quality_pass':False,
                    'worker_receipts':{r:{k:v for k,v in job.items() if k!='completion'} for r,job in {**executed,'metrics':scored}.items()}}
            _write_new(directory/'completed.json',result);self.blocked=False
            return result
        except BaseException as exc:
            _write_new(directory/'failed.json',{'type':type(exc).__name__,'message':str(exc),
                       'attempt_charged':True,'automatic_retry':False,'completed_at':_now()})
            raise


EPISODE_KEYS={'schema_version','output','created_at','deadline_unix','session_policy','batch_template',
              'metric_template','transport','trial_timeouts','selector','max_turns'}
TRANSPORT_KEYS={'root','launcher','launcher_sha256','python','host_template','max_jobs'}

def _episode_schema(spec):
    _keys(spec,EPISODE_KEYS | ({'trial_schedule'} if 'trial_schedule' in spec else set()),'episode spec')
    transport=spec['transport']
    _keys(transport,TRANSPORT_KEYS | ({'package_root'} if 'package_root' in transport else set()),'transport spec')



def _validate_candidate_domain(policy):
    """Allow a prospectively pinned capability subset, never new tuning fields.

    Keep each fixed field explicit, with its reference value still admitted:
    the initial reference proposal is always available to the agent. Exact JSON
    scalar types prevent bool/int/float equality from widening the domain.
    """
    document=policy.to_dict(); allowed=document['allowed_config_values']
    if set(allowed)!=set(ALLOWED_VALUES):
        raise ValueError('candidate domain must retain exactly the fixed Vista fields')
    for field,values in allowed.items():
        if type(values) is not list or not values:
            raise ValueError('candidate domain needs a nonempty list for each field')
        if any(not any(type(value) is type(item) and value==item for item in ALLOWED_VALUES[field])
               for value in values):
            raise ValueError('candidate domain values must be typed subsets of fixed Vista choices')
        section,key=field.split('.')
        reference=document['reference_config'][section][key]
        if not any(type(reference) is type(value) and reference==value for value in values):
            raise ValueError('candidate domain must include each reference value')


def run_spec(path, *, expected_sha256):
    """Execute ONE policy session. Final submit locks selection; it does not score.

    Run separately for each of LLM/random/grid with equal policy budgets. Then
    use freeze_final_plan after closing all three sessions. The existing batch
    and metric worker CLIs execute that plan under the same outer resource guard.
    No API request occurs for random/grid; no implicit fallback policy exists.
    """
    path=Path(path).absolute();spec=_pinned(path,expected_sha256);_episode_schema(spec)
    if spec['schema_version']!=1:raise ValueError('unsupported episode schema')
    now=time.time();deadline=_finite(spec['deadline_unix'],'episode deadline')
    if now>=deadline:raise TimeoutError('episode deadline already expired')
    policy=SessionPolicy(**spec['session_policy'])
    if policy.max_profiles!=0 or policy.final_evaluations!=1:raise ValueError('configuration cohort has no profile jobs and one selection reservation')
    if policy.wall_seconds>=deadline-now:raise ValueError('session wall budget exceeds immutable episode deadline')
    _validate_candidate_domain(policy)
    if spec['transport']['max_jobs']!=3*(policy.max_evaluations-1):raise ValueError('exact paired-generation/metric job cap required')
    transport=ExistingBoundedLauncher(**spec['transport'])
    output=Path(spec['output']).absolute()
    transport.container_path(output)
    output.mkdir(exist_ok=False)
    _write_new(output/'spec.json',spec)
    selector_spec=spec['selector']
    with ExitStack() as stack:
        try:
            quality=QualityPolicy.from_dict(_read(transport.host_path(spec['metric_template']['quality_policy']['path'])))
            trial=PairedTrialExecutor(directory=output/'trials',transport=transport,
                batch_template=spec['batch_template'],metric_template=spec['metric_template'],quality_policy=quality,
                timeouts=spec['trial_timeouts'],max_trials=policy.max_evaluations-1,
                selection_started_at=_now(),trial_schedule=spec.get('trial_schedule'))
            session=stack.enter_context(AgentSession.create(output/'session',policy=policy,
                executor=selection_only_executor(trial)))
            if selector_spec.get('kind') in ('random','grid'):
                _keys(selector_spec,{'kind','seed'},'baseline selector')
                selector=SearchBaseline(mode=selector_spec['kind'],seed=selector_spec['seed'],
                    reference=policy.to_dict()['reference_config'],allowed_values=policy.to_dict()['allowed_config_values'])
            elif selector_spec.get('kind')=='openai-responses':
                _keys(selector_spec,{'kind','model','max_calls','max_output_tokens','request_timeout_seconds','cost_policy'},'LLM selector')
                # Optional SDK; uses OPENAI_API_KEY from the trusted operator's
                # environment, never from the episode JSON or public observation.
                from openai import OpenAI
                cost=APICostPolicy(**selector_spec['cost_policy'])
                if cost.model!=selector_spec['model'] or cost.max_output_tokens!=selector_spec['max_output_tokens']:
                    raise ValueError('selector and cost contract differ')
                ledger=stack.enter_context(APICostLedger.create(output/'api-budget',policy=cost))
                client=stack.enter_context(OpenAI(base_url='https://api.openai.com/v1',max_retries=0,
                    timeout=selector_spec['request_timeout_seconds']))
                selector=ResponsesActionPolicy(BudgetedResponsesClient(client,ledger),model=selector_spec['model'],
                    max_calls=selector_spec['max_calls'],max_output_tokens=selector_spec['max_output_tokens'],
                    request_timeout_seconds=selector_spec['request_timeout_seconds'])
            else:raise ValueError('explicit random/grid/openai-responses selector required')
            projector=public_observation(session)
            def observation(state,last):
                if trial.blocked or transport.blocked:raise SessionError('worker failed; remaining selector calls are not admitted')
                return projector(state,last)
            result=run_agent(session,selector,observation_builder=observation,max_turns=spec['max_turns'])
            if result['state']['status']!='completed':raise RuntimeError('policy did not lock a selection within budget')
            receipt={'schema_version':1,'status':'ok','spec_sha256':expected_sha256,'cohort_id':policy.cohort_id,
                'session_directory':str(output/'session'),'session_policy':policy.to_dict(),
                'agent_ledger_sha256':result['agent_ledger_sha256'],'jobs_used':transport.used,
                'selection_only':True,'final_evaluation_performed':False,'quality_preserving_claim':False,
                'selected_proposal_id':result['state']['selected_proposal_id'],'completed_at':_now()}
            _write_new(output/'completed.json',receipt);return receipt
        except BaseException as exc:
            _write_new(output/'failed.json',{'status':'failed','type':type(exc).__name__,'message':str(exc),
                'jobs_used':transport.used,'automatic_retry':False,'spec_sha256':expected_sha256,'completed_at':_now()})
            raise



def step_spec(path, *, expected_sha256, operation, action_path=None, action_sha256=None):
    """Initialize, observe or apply ONE operator-supplied JSON action.

    This is the recommended mode when the LLM/API ledger runs on the caller's
    laptop. The caller transmits the selected JSON action through its existing
    authorized transport; the host never receives an API credential. Observe
    returns only the explicit public projection. Every action and any failure
    remain in the protected host session. No loop, watcher or supervisor runs
    between calls, and session/allocation deadlines are never extended.
    """
    path=Path(path).absolute();spec=_pinned(path,expected_sha256);_episode_schema(spec)
    if spec['schema_version']!=1 or operation not in ('initialize','observe','act'):raise ValueError('invalid episode operation')
    if operation!='act' and (action_path is not None or action_sha256 is not None):raise ValueError('only act accepts an action')
    if operation=='act' and (action_path is None or action_sha256 is None):raise ValueError('pinned action required')
    policy=SessionPolicy(**spec['session_policy'])
    if policy.max_profiles!=0 or policy.final_evaluations!=1:raise ValueError('selection-only session required')
    _validate_candidate_domain(policy)
    if spec['transport']['max_jobs']!=3*(policy.max_evaluations-1):raise ValueError('fixed job cap required')
    if spec['selector']!={'kind':'external-json'}:raise ValueError('step mode explicitly requires external-json selector')
    transport=ExistingBoundedLauncher(**spec['transport']);output=Path(spec['output']).absolute();transport.container_path(output)
    if operation=='initialize':
        if time.time()+policy.wall_seconds>=spec['deadline_unix']:raise ValueError('session wall budget exceeds episode deadline')
        output.mkdir(exist_ok=False);_write_new(output/'spec.json',spec);(output/'operator-actions').mkdir()
    elif _read(output/'spec.json')!=spec:raise ValueError('immutable episode spec changed')
    quality=QualityPolicy.from_dict(_read(transport.host_path(spec['metric_template']['quality_policy']['path'])))
    manifest=None if operation=='initialize' else _read(output/'session/policy.json')
    started_at=_now() if manifest is None else datetime.fromtimestamp(manifest['created_at'],timezone.utc).isoformat()
    trial=PairedTrialExecutor(directory=output/'trials',transport=transport,
        batch_template=spec['batch_template'],metric_template=spec['metric_template'],quality_policy=quality,
        timeouts=spec['trial_timeouts'],max_trials=policy.max_evaluations-1,selection_started_at=started_at,
        resume=operation!='initialize',trial_schedule=spec.get('trial_schedule'))
    session=(AgentSession.create(output/'session',policy=policy,executor=selection_only_executor(trial))
             if operation=='initialize' else AgentSession.open(output/'session',expected_policy=policy,executor=selection_only_executor(trial)))
    with session:
        if session.state['deadline_unix']>spec['deadline_unix']:raise ValueError('persisted session exceeds episode deadline')
        # The durable session and the worker attempts must describe the same
        # completed trials. A replaced local receipt cannot reopen execution.
        pending=None;finished=[]
        for event_path in sorted((session.directory/'events').glob('*.json')):
            event=_read(event_path)
            if event['type']=='execution_started':pending=event['payload']
            elif event['type'] in ('tool_finished','tool_failed') and pending:
                if pending['kind']=='trial':finished.append((pending,event))
                pending=None
        if len(finished)!=trial.used:trial.blocked=True
        for index,(execution,event) in enumerate(finished):
            attempt=trial.directory/f'{index:04d}'
            if not attempt.is_dir():trial.blocked=True;continue
            intent=_read(attempt/'intent.json')
            if (intent['proposal_id']!=execution['proposal_id'] or
                intent['candidate_sha256']!=digest(session._proposal(execution['proposal_id'])['config'])):
                raise SessionError('trial intent differs from durable session')
            if event['type']=='tool_finished' and _read(attempt/'completed.json')!=event['payload']['result']:
                raise SessionError('trial receipt differs from durable session')
        outcome=None
        if operation=='act':
            if trial.blocked or transport.blocked:raise SessionError('interrupted or failed worker; new actions not admitted')
            expected_trials=session.state['evaluations_used']-(1 if session.state['selected_proposal_id'] else 0)
            if trial.used!=expected_trials:raise SessionError('trial files differ from durable charged attempts')
            action=validate_action(_pinned(action_path,action_sha256))
            log=output/'operator-actions';index=len(list(log.glob('*-started.json')))+1
            if any(not q.with_name(q.name.replace('-started','-finished')).is_file() for q in log.glob('*-started.json')):
                raise SessionError('operator action interrupted; no automatic retry')
            _write_new(log/f'{index:08d}-started.json',{'action':action,'action_sha256':action_sha256,'at':_now()})
            try:
                response=session.call(action['tool'],action['arguments']);outcome={'ok':True,'response':response}
            except Exception as exc:
                outcome={'ok':False,'error':{'type':type(exc).__name__,'message':str(exc)}}
            _write_new(log/f'{index:08d}-finished.json',{'outcome':outcome,'at':_now()})
        observation=public_observation(session)(session.state,outcome)
        observation['remaining_wall_seconds']=max(0.,session.state['deadline_unix']-session.controller_time())
        observation['execution_available']=(not(trial.blocked or transport.blocked) and
            session.state['status']=='active' and observation['remaining_wall_seconds']>0)
        if session.state['status']=='completed' and not (output/'completed.json').exists():
            _write_new(output/'completed.json',{'schema_version':1,'status':'ok','spec_sha256':expected_sha256,
                'cohort_id':policy.cohort_id,'selection_only':True,'final_evaluation_performed':False,
                'session_directory':str(output/'session'),'session_policy':policy.to_dict(),
                'selected_proposal_id':session.state['selected_proposal_id'],'jobs_used':transport.used,
                'quality_preserving_claim':False,'completed_at':_now()})
        return observation


def main(argv=None):
    parser=argparse.ArgumentParser(description='Run trusted configuration-search operations through an existing bounded host launcher; no provisioning or SSH.')
    parser.add_argument('--spec',required=True);parser.add_argument('--spec-sha256',required=True)
    parser.add_argument('--operation',choices=['run','initialize','observe','act'],default='run')
    parser.add_argument('--action');parser.add_argument('--action-sha256')
    args=parser.parse_args(argv)
    if args.operation=='run':
        if args.action or args.action_sha256:parser.error('run mode does not accept an action file')
        receipt=run_spec(args.spec,expected_sha256=args.spec_sha256)
        print(json.dumps({'status':receipt['status'],'selection_only':True,'final_evaluation_performed':False}))
    else:
        observation=step_spec(args.spec,expected_sha256=args.spec_sha256,operation=args.operation,
                              action_path=args.action,action_sha256=args.action_sha256)
        print(json.dumps(observation,allow_nan=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
