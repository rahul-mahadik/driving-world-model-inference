import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from drivebench.config import digest
from scripts import run_bootstrap_evaluation as wrapper


def test_import_is_stdlib_only_and_environment_precedes_numerical_imports(tmp_path):
    script = Path(wrapper.__file__).resolve()
    code = f'''
import importlib.util, os, sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('wrapper', {str(script)!r})
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
assert not any(x in sys.modules for x in ('numpy','torch','drivebench.evaluate','lpips'))
original=sys.path[:]
os.environ['OMP_NUM_THREADS']='2'
with w.isolated_environment(Path({str(tmp_path)!r})) as environment:
 assert os.environ['OMP_NUM_THREADS']=='4'
 assert sys.path[:2]==[str(Path({str(tmp_path)!r})/'outputs/evaluation-packages/lpips'),{str(tmp_path)!r}]
 assert os.environ['TORCH_HOME'].endswith('/evaluation-packages/torch-home')
 assert os.environ['DRIVEBENCH_EVALUATION_SETUP_RECEIPT'].endswith('/setup-receipt.json')
assert os.environ['OMP_NUM_THREADS']=='2' and sys.path==original
sys.modules['torch']=object()
try:
 with w.isolated_environment(Path({str(tmp_path)!r})): pass
except RuntimeError: pass
else: raise AssertionError('did not reject preloaded torch')
'''
    subprocess.run([sys.executable, '-c', code], check=True)


@pytest.mark.parametrize('name', ['upstream', 'compiled-default', 'compiled-conditioning'])
def test_host_source_and_actual_wrapper_command_binding(tmp_path, name):
    files = {'one.py': 'a' * 64}
    source = digest(files)
    args = ['python', '-m', 'drivebench.benchmark', '--config', 'configs/study_upstream_development.yaml']
    if name != 'upstream':
        args = ['python', 'scripts/run_compiled_bootstrap.py', '--hostspec',
                'configs/study_bootstrap_compiled_hostspec.json', '--variant', name]
    record = dict(status='completed', exit_code=0, finished_at='now', source_unchanged=True,
                  source_identity=source, source_files=files, argv=args)
    if name != 'upstream': record['inputs_unchanged'] = True
    path = tmp_path / 'host.json'; path.write_text(json.dumps(record))
    wrapper.validate_host_receipt(path, source, '/workspace/outputs/study-bootstrap/' + name)
    with pytest.raises(ValueError): wrapper.validate_host_receipt(path, 'b' * 64, name)
    record['source_files']['one.py'] = 'c' * 64; path.write_text(json.dumps(record))
    with pytest.raises(ValueError): wrapper.validate_host_receipt(path, source, name)


@pytest.fixture
def result_plan(tmp_path, monkeypatch):
    import drivebench.evaluation.io as io
    monkeypatch.setattr(io, 'artifact_sha256', lambda run: {'config.json': run.name})
    output = tmp_path / 'qualification-v1/compiled-default-vs-upstream.json'
    review = output.parent / 'compiled-default-vs-upstream-review'
    runs = {name: SimpleNamespace(name=name) for name in ('upstream', 'compiled-default')}
    hashes = {name: {(f'r{i}', 0, f'r{i}:c00', i): name + str(i) for i in range(32)} for name in runs}
    source = tmp_path / 'input.npy'; source.write_bytes(b'bound')
    plan = dict(variant='compiled-default', output=output, review=review,
                receipt=output.parent / 'compiled-default-evaluation-launch.json', runs=runs,
                bindings={str(source): wrapper.sha(source)}, pair_hashes=hashes, argv=['drivebench.evaluate'])
    pairs = [dict(request_id=f'r{i}', repeat=0, candidate_id=f'r{i}:c00', seed=i,
                  reference_output_sha256='upstream'+str(i), candidate_output_sha256='compiled-default'+str(i),
                  metrics={'lpips': .01}, recorded_ground_truth={'mae': .3}) for i in range(32)]
    result = dict(perceptual={'status':'available','provenance':{'device':'cuda','setup_receipt_sha256':wrapper.SETUP_SHA}},
                  reference_artifact_sha256={'config.json':'upstream'}, tolerances={'frozen':False},
                  review={'cases':4, 'index':str(review/'index.html')},
                  runs=[dict(run='compiled-default', complete_pairs=True, matched_pairs=32, pairs=pairs, errors=[],
                             gate={'quality_preserving':False}, artifact_sha256={'config.json':'compiled-default'})])
    return plan, result


@pytest.mark.parametrize('mutation', ['unavailable','cpu','partial','hash','duplicate','nan','no_gt','accepted','review'])
def test_incomplete_or_mismatched_evaluation_is_rejected(result_plan, mutation):
    plan, result = result_plan
    if mutation == 'unavailable': result['perceptual']['status']='unavailable'
    if mutation == 'cpu': result['perceptual']['provenance']['device']='cpu'
    if mutation == 'partial': result['runs'][0]['matched_pairs']=31
    if mutation == 'hash': result['runs'][0]['pairs'][0]['candidate_output_sha256']='changed'
    if mutation == 'duplicate': result['runs'][0]['pairs'][1]=result['runs'][0]['pairs'][0]
    if mutation == 'nan': result['runs'][0]['pairs'][0]['metrics']['lpips']=float('nan')
    if mutation == 'no_gt': result['runs'][0]['pairs'][0]['recorded_ground_truth']=None
    if mutation == 'accepted': result['runs'][0]['gate']['quality_preserving']=True
    if mutation == 'review': result['review']['cases']=3
    plan['output'].parent.mkdir(); plan['output'].write_text(json.dumps(result))
    with pytest.raises(ValueError): wrapper.validate_result(plan)


@pytest.mark.parametrize('failure', ['none','exception','input_drift','existing_output','existing_receipt'])
def test_execute_keeps_run_inputs_immutable_and_preserves_failure_receipt(result_plan, failure):
    plan, result = result_plan
    old_argv = sys.argv
    if failure in ('existing_output','existing_receipt'):
        plan['output'].parent.mkdir()
        plan['output' if failure=='existing_output' else 'receipt'].write_text('preserved')
    def evaluate():
        assert sys.argv == plan['argv']
        if failure == 'exception': raise RuntimeError('failed GPU evaluation')
        plan['review'].mkdir()
        (plan['review']/'index.html').write_text('review')
        plan['output'].write_text(json.dumps(result))
        if failure == 'input_drift': Path(next(iter(plan['bindings']))).write_bytes(b'changed')
    if failure == 'none':
        assert wrapper.execute(plan, {'OMP_NUM_THREADS':'4'}, evaluator_main=evaluate)==0
        receipt=json.loads(plan['receipt'].read_text())
        assert receipt['status']=='completed' and receipt['inputs_unchanged']
        assert receipt['quality_acceptance_claim'] is False
    else:
        with pytest.raises((RuntimeError, ValueError, FileExistsError)):
            wrapper.execute(plan, {}, evaluator_main=evaluate)
        if failure=='existing_receipt': assert plan['receipt'].read_text()=='preserved'
        else: assert json.loads(plan['receipt'].read_text())['status']=='failed'
        if failure=='existing_output': assert plan['output'].read_text()=='preserved'
    assert sys.argv is old_argv


def test_retained_array_path_escape_and_missing_fail_before_any_metrics(tmp_path):
    from drivebench.evaluation.io import output_path
    run=SimpleNamespace(path=tmp_path)
    for value in ('../elsewhere.npy', 'missing.npy'):
        with pytest.raises(ValueError): output_path(run,value)


@pytest.mark.parametrize('mutation', ['none','batch','fallback','shape','runtime','identity','input_hash'])
def test_saved_run_checks_real_batch_rng_identity_and_all_arrays(tmp_path,monkeypatch,mutation):
    import numpy as np
    import drivebench.data.manifests as manifests
    runpath=tmp_path/'run';runpath.mkdir()
    (runpath/'cold_start.json').write_text('{"status":"ready"}')
    frames=['observed.jpg',*[f'future{i}.jpg' for i in range(24)]]
    camera=tmp_path/'data/nuscenes';camera.mkdir(parents=True)
    for frame in frames:(camera/frame).write_bytes(b'frame')
    checks={frame:wrapper.sha(camera/frame) for frame in frames}
    records=[dict(request_id=f'r{i}',observed_frames=frames[:1],future_frames=frames[1:],frame_checksums=checks) for i in range(32)]
    monkeypatch.setattr(manifests,'inference_request',lambda record,*args,**kwargs:dict(
        scene_id=record['request_id'],candidate_ids=[record['request_id']+':c00'],seeds=[7],
        action_kinds=['recorded'],trajectories=[[0.0]]))
    rows=[]
    for record in records:
        path=runpath/(record['request_id']+'.npy');path.write_bytes(b'array')
        candidate=dict(candidate_id=record['request_id']+':c00',seed=7,action_kind='recorded',conditioning_frames=1,
                       trajectory_sha256=digest([0.0]),frames_path=path.name,
                       ground_truth_paths=[str(camera/frame) for frame in frames[1:]])
        rows.append(dict(request_id=record['request_id'],scene_id=record['request_id'],repeat=0,candidate_count=1,batch_size=1,
                         effective_config={'actual_batch_sizes':[1]},denoiser_evaluations=50,fallbacks=[],failures=[],
                         cache_stats={'compiler_runtime':{'passed':True,'restoration_complete':True,
                            'graphs_created_by_runtime':347,'kernels_created_by_runtime':1200,'after':{'unimplemented':{}}}},candidates=[candidate]))
    shape=(25,576,1024,3)
    if mutation=='batch':rows[0]['effective_config']['actual_batch_sizes']=[2]
    if mutation=='fallback':rows[0]['fallbacks']=['serial retry']
    if mutation=='shape':shape=(24,576,1024,3)
    if mutation=='runtime':rows[0]['cache_stats']['compiler_runtime']['restoration_complete']=False
    if mutation=='identity':rows[0]['candidates'][0]['seed']=8
    if mutation=='input_hash':(camera/frames[-1]).write_bytes(b'changed')
    monkeypatch.setattr(np,'load',lambda *args,**kwargs:SimpleNamespace(shape=shape,dtype=np.dtype('float32')))
    run=SimpleNamespace(path=runpath,measurements=rows,summary=dict(measured_requests=32,successful_requests=32,
        completed_futures=32,oom_attempts=0,profiled=False,recompiled_during_measurement=False,headline_eligible=False))
    if mutation=='none':
        bindings,pairs=wrapper.run_inputs(run,records,tmp_path,compiled=True)
        assert len(pairs)==32 and len([p for p in bindings if p.endswith('.npy')])==32
    else:
        with pytest.raises(ValueError):wrapper.run_inputs(run,records,tmp_path,compiled=True)
