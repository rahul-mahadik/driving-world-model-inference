"""Bounded trusted calibration/trial/final metric worker. No inference or scheduling.

Run in a fresh process under an external timeout. All input artifacts, source,
LPIPS assets and package versions are operator-pinned. Derived controls use one
scratch array; successful controls retain regeneration recipes/hashes/metrics,
while failed scratch is preserved. Original generated arrays are never removed.
The synthetic fixture is explicitly not LPIPS and is forbidden for real batches.
"""
from __future__ import annotations

import argparse
import hashlib
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import socket
import sys
import time

import numpy as np

from . import agent_evidence as evidence, agent_metrics, agent_scoring, vista_agent, vista_batch

METRICS = ('mae', 'temporal_derivative_mae', 'lpips')
CONTROLS = ('freeze_future', 'black_future', 'reverse_future')
COMMON = {'schema_version', 'kind', 'synthetic', 'created_at', 'deadline_unix',
          'minimum_free_bytes', 'output', 'metric', 'references', 'sources', 'study_bindings'}
EXTRA = {'calibrate': {'quality_plan'},
         'trial': {'candidates', 'quality_policy', 'selection_started_at', 'candidate_sha256', 'trial_cells'},
         'final': {'candidates', 'quality_policy', 'selection_started_at', 'candidate_sha256', 'final_plan'}}


def _now(): return datetime.now(timezone.utc).isoformat()
def _write(path, value):
    from .agent_session import _write_new
    _write_new(Path(path), value)


def _pin(record):
    vista_batch._keys(record, {'path', 'sha256'}, 'artifact pin')
    path = Path(record['path'])
    if not path.is_absolute(): raise ValueError('absolute artifact path required')
    return evidence._read(path, record['sha256'])[0]


def required_source_paths():
    import drivebench
    # Receipt loading reconstructs inputs through the CPU core, and final joins
    # import the cohort/session validators. Pin both full Python packages so a
    # transitive checker change cannot preserve the metric definition identity.
    roots = (Path(__file__).resolve().parent, Path(drivebench.__file__).resolve().parent)
    return sorted({path.resolve() for root in roots for path in root.rglob('*.py')})


def _check_sources(sources):
    if not isinstance(sources, dict): raise ValueError('source hash mapping required')
    if not {str(p) for p in required_source_paths()}.issubset(sources):
        raise ValueError('metric worker source set incomplete')
    for filename, sha in sources.items():
        path = Path(filename)
        if not path.is_absolute() or path.is_symlink() or not path.is_file() or vista_agent.file_hash(path) != vista_batch._hash(sha, 'source hash'):
            raise ValueError('metric source drift')


def _batches(records):
    if not isinstance(records, list) or not records: raise ValueError('pinned batches required')
    result = []
    for record in records:
        vista_batch._keys(record, {'directory', 'role', 'expected_spec_sha256', 'expected_completion_sha256'}, 'batch pin')
        result.append(evidence.load_batch(record['directory'], **{k:v for k,v in record.items() if k != 'directory'}))
    return result


def _study_bindings(binding, batches, final=None):
    vista_batch._keys(binding, {'source_manifest', 'input_manifest'}, 'study binding')
    source = _pin(binding['source_manifest']); inputs = _pin(binding['input_manifest'])
    vista_batch._keys(source, {'source_files'}, 'generation source manifest')
    vista_batch._keys(inputs, {'manifests'}, 'input manifest bundle')
    if not isinstance(inputs['manifests'], list) or not inputs['manifests']:
        raise ValueError('nonempty input manifest bundle required')
    for batch in batches:
        spec = batch.to_dict()['spec']
        evidence._same(spec['source_files'], source['source_files'], 'aggregate source manifest')
        if spec['manifest'] not in inputs['manifests']: raise ValueError('batch input manifest not admitted')
    if final is not None:
        for key in ('source_manifest', 'input_manifest'):
            plan_key = 'inputs_manifest_sha256' if key == 'input_manifest' else 'source_manifest_sha256'
            evidence._same(final[plan_key], binding[key]['sha256'], 'final aggregate manifest pin')


@contextmanager
def _offline(torch):
    def denied(*args, **kwargs): raise RuntimeError('network forbidden in metric worker')
    previous = (socket.socket.connect, socket.socket.connect_ex, socket.create_connection, torch.hub.download_url_to_file)
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = torch.hub.download_url_to_file = denied
    try: yield
    finally:
        socket.socket.connect, socket.socket.connect_ex, socket.create_connection, torch.hub.download_url_to_file = previous


class _Fixture:
    def compare(self, reference, candidate): return float(np.abs(reference-candidate).mean())


@contextmanager
def _metric(config, synthetic):
    if config == {'kind': 'synthetic_fixture'}:
        if not synthetic: raise ValueError('synthetic perceptual fixture forbidden for real evidence')
        yield _Fixture(), {'kind': 'synthetic_fixture', 'is_lpips': False}
        return
    vista_batch._keys(config, {'kind', 'setup', 'device', 'versions'}, 'LPIPS metric config')
    if config['kind'] != 'lpips' or config['device'] not in ('cpu', 'cuda'):
        raise ValueError('explicit LPIPS evaluator required')
    if synthetic: raise ValueError('synthetic batch must use explicitly labeled fixture metric')
    from drivebench.evaluation.metrics import verify_perceptual_setup, PerceptualMetric
    _pin(config['setup'])
    setup = verify_perceptual_setup(config['setup']['path'])
    expected_assets={'wheel':'fd537af5828b69d2e6ffc0a397bd506dbc28ca183543617690844c08e102ec5e',
        'alexnet_cached_weight':'7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02',
        'lpips_learned_weight':'df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0'}
    if any(setup[key]['sha256']!=sha for key,sha in expected_assets.items()):
        raise ValueError('LPIPS assets differ from pinned AlexNet v0.1 definition')
    versions = config['versions']
    if (not isinstance(versions,dict) or set(versions) != {'torch', 'torchvision', 'numpy', 'scipy', 'lpips'} or
            any(not isinstance(value,str) or not value for value in versions.values()) or versions['lpips'] != '0.1.4'):
        raise ValueError('exact metric package versions required')
    if 'torch' in sys.modules or 'lpips' in sys.modules:
        raise ValueError('real metric initialization requires a fresh process')
    prior_path, prior_home = sys.path[:], os.environ.get('TORCH_HOME')
    prior_bytecode = sys.dont_write_bytecode
    target = Path(setup['package']['target']).resolve()
    # The staged wheel is the entire import root: reject extra executable files
    # and precompiled bytecode, which could otherwise bypass pinned source bytes.
    def check_package():
        paths = list(target.rglob('*'))
        if any(p.is_symlink() for p in paths) or {str(p.relative_to(target)) for p in paths if p.is_file()} != set(setup['package']['files']):
            raise ValueError('unlisted file or symlink in pinned LPIPS package')
        source_files = setup['cpu_verification']['source_files']
        if not {'torchvision_alexnet','lpips','lpips_model','lpips_trunk'}.issubset(source_files):
            raise ValueError('verified metric source bindings incomplete')
        for source in source_files.values():
            path = Path(source['path'])
            if (not path.is_absolute() or path.is_symlink() or not path.is_file() or
                    path.stat().st_size != source['bytes'] or vista_agent.file_hash(path) != source['sha256']):
                raise ValueError('verified metric source changed')
    check_package()
    os.environ['TORCH_HOME'] = str(Path(setup['alexnet_cached_weight']['path']).parent.parent.parent)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(target))
    try:
        import torch
        for name, expected in versions.items():
            if importlib.metadata.version(name) != expected: raise ValueError('metric package version drift')
        with _offline(torch):
            metric = PerceptualMetric(enabled=True, device=config['device'])
            if metric.status != 'available': raise ValueError('pinned LPIPS unavailable: ' + str(metric.reason))
            if not Path(metric.provenance['module_path']).is_relative_to(target):
                raise ValueError('LPIPS imported outside pinned package')
            imported_sources = {'lpips':metric.provenance['module_path'],
                                'torchvision_alexnet':importlib.import_module('torchvision.models.alexnet').__file__}
            if any(Path(path).resolve() != Path(setup['cpu_verification']['source_files'][key]['path']).resolve()
                   for key,path in imported_sources.items()):
                raise ValueError('metric dependency imported outside verified source')
            if metric.provenance['state_dict_sha256'] != setup['cpu_verification']['model_state_dict_sha256']:
                raise ValueError('loaded LPIPS weights differ from verified CPU setup')
            yield metric, {**metric.provenance, 'kind': 'lpips', 'is_lpips': True,
                           'setup_sha256': config['setup']['sha256'], 'versions': versions}
            verify_perceptual_setup(config['setup']['path'])
            check_package()
    finally:
        sys.path[:] = prior_path
        sys.dont_write_bytecode = prior_bytecode
        if prior_home is None: os.environ.pop('TORCH_HOME', None)
        else: os.environ['TORCH_HOME'] = prior_home


def run_spec(spec_path, *, expected_sha256):
    spec_path = Path(spec_path).absolute()
    spec, _ = evidence._read(spec_path, expected_sha256)
    if spec.get('kind') not in EXTRA: raise ValueError('unknown metric operation')
    vista_batch._keys(spec, COMMON | EXTRA[spec['kind']], 'metric specification')
    if spec['schema_version'] != 1 or type(spec['synthetic']) is not bool: raise ValueError('invalid metric schema')
    output = Path(spec['output'])
    if not output.is_absolute() or output.is_symlink(): raise ValueError('absolute new output directory required')
    output.mkdir(parents=True, exist_ok=False)
    _write(output/'spec.json', spec)
    started = time.time(); previous_time = started
    def check():
        nonlocal previous_time
        now = time.time()
        if (type(spec['deadline_unix']) not in (int,float) or not math.isfinite(spec['deadline_unix']) or
                now < previous_time or now >= spec['deadline_unix']): raise ValueError('metric deadline expired or clock moved backwards')
        previous_time = now
        if type(spec['minimum_free_bytes']) is not int or spec['minimum_free_bytes'] < 0:
            raise ValueError('nonnegative storage reserve required')
        if shutil.disk_usage(output).free < spec['minimum_free_bytes']: raise ValueError('insufficient metric storage reserve')
    try:
        check(); _check_sources(spec['sources'])
        if evidence._time(spec['created_at']) > started: raise ValueError('specification created in the future')
        refs = _batches(spec['references']); candidates = _batches(spec['candidates']) if spec['kind'] != 'calibrate' else []
        all_batches = refs+candidates
        if any(b.to_dict()['spec']['synthetic'] != spec['synthetic'] for b in all_batches): raise ValueError('mixed real/synthetic evidence')
        final = _pin(spec['final_plan']) if spec['kind'] == 'final' else None
        _study_bindings(spec['study_bindings'], all_batches, final)
        definition = {'metrics': agent_metrics.METRIC_DEFINITION, 'metric': spec['metric'], 'source_files': spec['sources']}
        identity = evidence.identity_for_batches(refs, metric_definition_sha256=vista_agent.digest(definition))
        with _metric(spec['metric'], spec['synthetic']) as (perceptual, provenance):
            if spec['metric'].get('device') == 'cuda':
                evidence._same(vista_agent.probe_runtime(synthetic=False),refs[0].to_dict()['spec']['expected_runtime'],'metric GPU/runtime')
            _write(output/'metric-provenance.json', {'identity':identity, 'definition':definition, 'runtime':provenance})
            def measure(left, right, **pins):
                check()
                result = agent_metrics.measure_pair(left, right, **pins, perceptual_metric=perceptual)
                check()
                return result
            if spec['kind'] == 'calibrate':
                rows = evidence.calibration_rows(refs, identity=identity, measure_pair=measure, metric_names=METRICS)
                _write(output/'calibration-records.json', {'records':rows})
                controls=[]; control_dir=output/'controls'; control_dir.mkdir()
                scratch=output/'derived-control.npy'
                for batch in refs:
                    data=batch.recheck()
                    for original in data['rows']:
                        for control in CONTROLS:
                            check()
                            if scratch.exists(): raise ValueError('unconsumed derived scratch array')
                            if shutil.disk_usage(output).free - Path(original['raw_path']).stat().st_size < spec['minimum_free_bytes']:
                                raise ValueError('insufficient derived-control scratch reserve')
                            raw=np.load(original['raw_path'], mmap_mode='r', allow_pickle=False)
                            with scratch.open('xb') as out:
                                np.save(out, agent_metrics.negative_control(raw, control), allow_pickle=False); out.flush(); os.fsync(out.fileno())
                            sha=vista_agent.file_hash(scratch)
                            row={**evidence._base(original,identity), 'record_id':vista_agent.canonical(['control', control, original['record_id']]),
                                 'control_id':control, 'reference_output_sha256':original['output_sha256'], 'output_sha256':sha,
                                 'observed_at':_now(), 'metrics':measure(original['raw_path'],str(scratch),
                                     reference_sha256=original['output_sha256'], candidate_sha256=sha, conditioning_frames=1)}
                            receipt={'row':row,'recipe':{'version':1,'control':control,'reference_path':original['raw_path'],
                                'reference_sha256':original['output_sha256'],'derived_sha256':sha,'conditioning_frames':1},
                                'derived_array_retention_policy':'retire_after_verified_receipt','reference_array_retained':True}
                            receipt_path=control_dir/f'{len(controls):04d}.json'
                            _write(receipt_path,receipt)
                            expected_receipt_hash=hashlib.sha256((vista_agent.canonical(receipt)+'\n').encode()).hexdigest()
                            saved,_=evidence._read(receipt_path,expected_receipt_hash)
                            evidence._same(saved,receipt,'permanent derived-control receipt')
                            if (vista_agent.file_hash(original['raw_path']) != original['output_sha256'] or
                                    vista_agent.file_hash(scratch) != sha):
                                raise ValueError('derived-control input/output changed before scratch retirement')
                            # Remove only this worker's completely recorded deterministic derivative.
                            scratch.unlink(); controls.append(row)
                plan=dict(spec['quality_plan'])
                forbidden={'identity','calibration_records','negative_controls','frozen_at'}
                if forbidden & set(plan): raise ValueError('quality plan contains evaluator-owned fields')
                if set(plan['metric_margins']) != set(METRICS): raise ValueError('all three proxy margins required')
                if plan.pop('noise_floor_method', None) != 'maximum_paired_repeat_relative_difference':
                    raise ValueError('explicit reference-repeat noise-floor procedure required')
                latency_cells={}
                for batch in refs:
                    for row in batch.to_dict()['rows']:
                        latency_cells.setdefault((row['scene_id'],row['seed']),[]).append(row['latency_ms'])
                observed_noise=max(max(values)/min(values)-1.0 for values in latency_cells.values())
                plan['timing_noise_floor']=max(plan.get('timing_noise_floor',0.0),observed_noise)
                _write(output/'timing-noise.json',{'method':'maximum_paired_repeat_relative_difference','observed_floor':observed_noise,
                    'frozen_floor':plan['timing_noise_floor'],'confidence_bound':False})
                policy=agent_scoring.freeze_quality_policy(identity=identity,calibration_records=rows,negative_controls=controls,frozen_at=_now(),**plan)
                _write(output/'quality-policy.json',policy.to_dict())
                result={'calibration_status':policy.to_dict()['calibration']['status'],'quality_policy_sha256':policy.sha256,
                        'quality_policy_file_sha256':vista_agent.file_hash(output/'quality-policy.json'),'calibration_records':len(rows),'control_records':len(controls)}
            else:
                policy=agent_scoring.QualityPolicy.from_dict(_pin(spec['quality_policy']))
                if policy.to_dict()['calibration']['status'] != 'ready': raise ValueError('calibration inconclusive; selection forbidden')
                evidence._same(policy.to_dict()['plan']['identity'],identity,'frozen metric identity')
                if evidence._time(spec['selection_started_at']) <= evidence._time(policy.to_dict()['plan']['frozen_at']):
                    raise ValueError('selection must follow quality freeze')
                paired=evidence.pair_rows(refs,candidates,identity=identity,candidate_sha256=spec['candidate_sha256'],measure_pair=measure,metric_names=METRICS,
                    **({'final_plan_path':spec['final_plan']['path'],'expected_final_plan_sha256':spec['final_plan']['sha256']} if final is not None else {}))
                _write(output/'paired-records.json',paired)
                if spec['kind'] == 'final':
                    evidence._same(final['quality_policy_sha256'],policy.sha256,'final quality policy')
                    quality=agent_scoring.assess_proxy_quality(policy,paired['quality_rows'],split='final',selection_started_at=spec['selection_started_at'])
                    score=agent_scoring.score_heldout_timing(policy,paired['timing_rows'],quality_assessment=quality,selection_started_at=spec['selection_started_at'])
                    _write(output/'quality-assessment.json',quality);_write(output/'timing-score.json',score)
                    result={'quality_status':quality['status'],'score':score,'quality_policy_sha256':policy.sha256,'candidate_sha256':spec['candidate_sha256']}
                else:
                    if any(evidence._time(b.to_dict()['manifest']['created_at']) <= evidence._time(spec['selection_started_at']) for b in all_batches):
                        raise ValueError('trial reference/candidate batch predates selection')
                    plan=policy.to_dict()['plan'];cells=spec['trial_cells']
                    if not isinstance(cells,list) or not cells: raise ValueError('explicit trial cells required')
                    for c in cells: vista_batch._keys(c,{'scene_id','seed','replicate'},'trial cell')
                    required={(c['scene_id'],c['seed'],c['replicate']) for c in cells}
                    if len(required)!=len(cells): raise ValueError('duplicate trial cells')
                    if any(s not in plan['development_scenes'] or seed not in plan['seeds'] or rep not in plan['replicates'] for s,seed,rep in required):
                        raise ValueError('trial cell outside frozen development plan')
                    if {evidence._cell(r) for r in paired['quality_rows']}!=required: raise ValueError('incomplete or unexpected trial cells')
                    if any(evidence._time(r['observed_at'])<=evidence._time(spec['selection_started_at']) for r in paired['quality_rows']):
                        raise ValueError('trial generation predates selection')
                    breaches=[{'record_id':r['record_id'],'metric':key,'actual':r['metrics'][key],'maximum':maximum}
                              for r in paired['quality_rows'] for key,maximum in plan['metric_margins'].items() if r['metrics'][key]>maximum]
                    maxima={key:max(r['metrics'][key] for r in paired['quality_rows']) for key in METRICS}
                    result={'screening_status':'breach' if breaches else 'no_observed_breach','quality_assessed':False,
                            'quality_policy_sha256':policy.sha256,'candidate_sha256':spec['candidate_sha256'],'identity_sha256':policy.identity_sha256,
                            'metrics_maxima':maxima,'breaches':breaches,'paired_cells':len(required),
                            'reference_median_latency_ms':float(np.median([r['latency_ms'] for b in refs for r in b.to_dict()['rows']])),
                            'candidate_median_latency_ms':float(np.median([r['latency_ms'] for b in candidates for r in b.to_dict()['rows']]))}
            check();_check_sources(spec['sources']);_study_bindings(spec['study_bindings'],all_batches,final)
            if spec['metric'].get('device') == 'cuda':
                evidence._same(vista_agent.probe_runtime(synthetic=False),refs[0].to_dict()['spec']['expected_runtime'],'metric GPU/runtime')
            for batch in all_batches:batch.recheck()
            for key in ('quality_policy','final_plan'):
                if key in spec:_pin(spec[key])
            evidence._read(spec_path,expected_sha256)
        check()
        complete={'schema_version':1,'status':'ok','kind':spec['kind'],'cohort_id':refs[0].to_dict()['cohort_id'],
            'spec_sha256':expected_sha256,'synthetic':spec['synthetic'],'real_lpips':not spec['synthetic'],
            'identity_sha256':vista_agent.digest(identity),'quality_preserving_claim':False,'completed_at':_now(),
            'external_worker_exit_verified':False,'result':result,
            'artifacts':{str(p.relative_to(output)):vista_agent.file_hash(p) for p in sorted(output.rglob('*')) if p.is_file()}}
        _write(output/'completed.json',complete);return complete
    except BaseException as error:
        _write(output/'failure.json',{'status':'failed','error_type':type(error).__name__,'error':str(error),'at':_now(),
             'spec_sha256':expected_sha256,'scratch_retained':(output/'derived-control.npy').exists(),'quality_preserving_claim':False})
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',required=True);parser.add_argument('--spec-sha256',required=True)
    args=parser.parse_args(argv);print(vista_agent.canonical(run_spec(args.spec,expected_sha256=args.spec_sha256)))


if __name__=='__main__':main()
