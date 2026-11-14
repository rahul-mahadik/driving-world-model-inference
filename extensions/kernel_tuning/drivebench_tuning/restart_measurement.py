"""Fresh-process measurement and paired restart analysis, using the existing benchmark.

This is a single-worker entry point, not a resource manager. An external bounded
runner must own the GPU, enforce the deadline, and verify process exit/cleanup.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
import shutil
from pathlib import Path
import sys
import time
import uuid

import numpy as np

from drivebench.config import digest, file_sha256, load_config
from drivebench.experiments import balanced_order

ROLES = ('modern_eager', 'modern_compiled')
_CACHE_VARS = ('TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR')
_USED_PROCESS = False


def _write(path, value):
    with Path(path).open('x') as out:
        json.dump(value, out, indent=2, allow_nan=False, sort_keys=True)
        out.write('\n')


def _read(path):
    return json.loads(Path(path).read_text())


def _number(value, name, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f'{name} must be finite and ' + ('positive' if positive else 'nonnegative'))
    return float(value)


def _pin(path):
    path = Path(path)
    return {'sha256': file_sha256(path), 'bytes': path.stat().st_size}


def _effective(config):
    """Compare the fully expanded existing types, not just YAML overrides."""
    from drivebench.types import GenerationConfig, OptimizationConfig
    value = deepcopy(config)
    value['generation'] = GenerationConfig(**value.get('generation', {})).to_dict()
    value['optimizations'] = OptimizationConfig(**value.get('optimizations', {})).to_dict()
    value.pop('name', None); value.pop('output', None)
    return value


def make_plan(eager_config, compiled_config, *, blocks=12, seed=20261114,
              mode='paired', gpu_name='NVIDIA RTX A6000', minimum_memory_mib=45000,
              minimum_host_ram_bytes=32*2**30, mig_supported=False, compute_usd_per_hour=None):
    if mode not in ('paired', 'feasibility'):
        raise ValueError('choose paired or feasibility mode')
    if (mode == 'paired' and (type(blocks) is not int or blocks < 8 or blocks % 2)) or (mode == 'feasibility' and blocks != 1):
        raise ValueError('paired mode needs an even block count >=8; feasibility needs exactly one block')
    if not isinstance(gpu_name,str) or not gpu_name.strip() or type(minimum_memory_mib) is not int or minimum_memory_mib < 1:
        raise ValueError('explicit GPU name and positive minimum memory required')
    if type(minimum_host_ram_bytes) is not int or minimum_host_ram_bytes < 1 or type(mig_supported) is not bool:
        raise ValueError('positive host RAM and explicit MIG support policy required')
    if compute_usd_per_hour is not None: _number(compute_usd_per_hour, 'compute rate', positive=True)
    configs = {ROLES[0]: _effective(eager_config), ROLES[1]: _effective(compiled_config)}
    for role, config in configs.items():
        if config.get('model', {}).get('backend') != 'modern_vista':
            raise ValueError('explicit modern_vista backend required; never infer role from settings')
        generation = config['generation']
        if any(generation[key] != value for key, value in {'steps':50,'total_frames':25,'conditioning_frames':1,'height':576,'width':1024,'decode_chunk_size':14,'dtype':'fp16'}.items()):
            raise ValueError('Phase 1 fixes the full-resolution 50-step FP16 single-future workload')
        opts = config['optimizations']
        if opts['compile_denoiser'] is not (role == 'modern_compiled'):
            raise ValueError('role and compile flag disagree')
        if any((opts['compile_decoder'], opts['prefetch_depth'], opts['conditioning_cache_bytes'],
                opts['feature_cache_interval'] != 1, opts['max_batch_size'] != 1)):
            raise ValueError('Phase 1 fixes batching/cache/prefetch and compiles only the denoiser')
        bench = config.get('benchmark', {})
        if bench.get('headline', False) or bench.get('candidate_count', 1) != 1 or bench.get('input_condition', 'uncached') != 'uncached':
            raise ValueError('single uncached future, non-headline timing boundary required')
        if type(bench.get('warmup_requests',3)) is not int or type(bench.get('max_warmup_requests',12)) is not int or bench.get('warmup_requests', 3) < 3:
            raise ValueError('record at least three cold-prefix requests')
        if bench.get('max_warmup_requests', 12) < bench.get('warmup_requests', 3):
            raise ValueError('invalid warmup cap')
    reference = deepcopy(configs['modern_eager'])
    candidate = deepcopy(configs['modern_compiled'])
    candidate['optimizations']['compile_denoiser'] = False
    if candidate != reference:
        raise ValueError('modern roles must differ only in compile_denoiser')
    manifest = Path(reference['data']['manifest'])
    from drivebench.data import load_manifest, inference_request
    records = load_manifest(manifest)
    scenes = [row['scene_id'] for row in records]
    scene_count = 3 if mode == 'paired' else 1
    if len(records) != scene_count or len(set(scenes)) != scene_count:
        raise ValueError(f'prespecify exactly {scene_count} distinct scenes, one record per scene')
    if any(row.get('dataset') != 'nuscenes' or row.get('trajectory_metadata', {}).get('synthetic') for row in records):
        raise ValueError('real nuScenes manifest with verified trajectory semantics required')
    bindings = []
    for record in records:
        request = inference_request(record, reference['data']['root'], candidate_count=1)
        bindings.append({'request_id':request['request_id'],'scene_id':request['scene_id'],
                         'candidate_id':request['candidate_ids'][0],'seed':request['seeds'][0],
                         'action_kind':request['action_kinds'][0],
                         'trajectory_sha256':digest(np.asarray(request['trajectories'][0]).tolist())})
    requests = reference['benchmark'].get('requests', 3)
    if type(requests) is not int or requests < scene_count or requests % scene_count or (mode == 'feasibility' and requests != 1):
        raise ValueError('equal complete scene passes required; feasibility measures exactly one request per role')
    if reference['benchmark'].get('save_outputs') != requests:
        raise ValueError('preserve all measured raw outputs in Phase 1')
    order = balanced_order(list(ROLES), repeats=blocks, seed=seed) if mode == 'paired' else list(ROLES)
    schedule = [{'run_id': f'{i:03d}-{role}', 'role': role, 'block': i // 2,
                 'position': i % 2} for i, role in enumerate(order)]
    return {'schema_version': 1, 'kind': 'paired_process_restarts' if mode == 'paired' else 'full_workload_feasibility', 'created_at': datetime.now(timezone.utc).isoformat(),
            'resource_admitted': False, 'synthetic': False, 'blocks': blocks, 'seed': seed, 'mode': mode,
            'hardware': {'gpu_name':gpu_name,'minimum_memory_mib':minimum_memory_mib,
                         'minimum_host_ram_bytes':minimum_host_ram_bytes,'mig_supported':mig_supported},
            'compute_usd_per_hour_assumption':compute_usd_per_hour,
            'roles': list(ROLES), 'configs': configs, 'schedule': schedule,
            'input_manifest_sha256': file_sha256(manifest), 'scene_ids': scenes, 'request_bindings':bindings,
            'warm_requests_per_process': requests,
            'cache_condition': 'empty_process_owned_inductor_and_triton_directories',
            'legacy_eager': {'status': 'historical_separate_cohort_only', 'new_processes': 0},
            'statistical_unit': 'four-process AB/BA reversal group, conditional on the fixed scenes',
            'timing_boundary': 'input load/preprocess through host-ready frames; exports excluded',
            'requires': ['fresh dedicated declared GPU hardware admission', 'same GPU/driver/full modern runtime',
                         'external immutable deadline and process cleanup', 'modern compatibility and quality evidence']}


def _validate_plan(plan):
    rebuilt = make_plan(plan['configs']['modern_eager'], plan['configs']['modern_compiled'],
                        blocks=plan['blocks'], seed=plan['seed'], mode=plan['mode'],
                        gpu_name=plan['hardware']['gpu_name'], minimum_memory_mib=plan['hardware']['minimum_memory_mib'],
                        minimum_host_ram_bytes=plan['hardware']['minimum_host_ram_bytes'],mig_supported=plan['hardware']['mig_supported'],
                        compute_usd_per_hour=plan['compute_usd_per_hour_assumption'])
    for key in ('kind', 'roles', 'configs', 'schedule', 'input_manifest_sha256', 'scene_ids',
                'warm_requests_per_process', 'cache_condition', 'synthetic', 'hardware', 'request_bindings'):
        if plan.get(key) != rebuilt[key]:
            raise ValueError('plan binding changed: ' + key)


def _measurement_sources():
    root = Path(__file__).parent
    return {name:file_sha256(root/name) for name in
            ('restart_measurement.py','modern_profile.py','attribution.py','study_cost.py')}


def _modern_sources():
    from drivebench_tuning import modern_vista
    parent = Path(modern_vista.__file__).parent
    return {name:file_sha256(parent/name) for name in ('modern_vista.py','modern_attention.py')}


def _check_modern_proof(proof, hardware):
    """Require the new declared-device proof; never relabel an old H100 pass."""
    from drivebench_tuning.modern_verify_environment import validate_policy
    policy = proof.get('hardware_policy')
    validate_policy(policy)
    if (proof.get('proof_kind') != 'modern_declared_gpu_v1' or proof.get('passed') is not True
            or proof.get('legacy_h100_proof') is not False
            or any(policy.get(key) != value for key,value in hardware.items())
            or proof.get('profiling_proof',{}).get('modern_runtime_passed') is not True
            or proof.get('profiling_proof',{}).get('torch_cuda_execution_passed') is not True):
        raise ValueError('a passed modern proof for the exact declared GPU policy is required')


def _archive_modern_environment(config, hardware, destination):
    proof_path = Path(config['environment_verification'])
    lock_path = Path(config['environment_lock'])
    _check_modern_proof(_read(proof_path), hardware)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for source,name in ((proof_path,'modern_environment_verification.json'),(lock_path,'modern_runtime_lock.json')):
        target = destination / name
        with source.open('rb') as reader, target.open('xb') as writer: shutil.copyfileobj(reader,writer)
    return {**config, 'environment_verification':str(destination/'modern_environment_verification.json'),
            'environment_lock':str(destination/'modern_runtime_lock.json')}


def _verify_archived_environment(root, hardware, environment):
    from drivebench.benchmark import _validate_reference_environment
    root = Path(root)
    proof = _read(root/'modern_environment_verification.json')
    _check_modern_proof(proof,hardware)
    return _validate_reference_environment({'environment_verification':str(root/'modern_environment_verification.json'),
             'environment_lock':str(root/'modern_runtime_lock.json')}, environment)


def collect_restart(plan_path, expected_plan_sha256, run_id, directory, *, deadline_unix):
    """Exactly one call per interpreter; invoke with an existing bounded worker.

    Empty compiler caches do not imply cold OS, checkpoint, CUDA-driver caches,
    imports, or GPU context. Those conditions must remain recorded separately.
    """
    global _USED_PROCESS
    if _USED_PROCESS or 'torch' in sys.modules:
        raise RuntimeError('use a fresh interpreter before importing Torch')
    if file_sha256(plan_path) != expected_plan_sha256:
        raise ValueError('plan hash mismatch')
    _number(deadline_unix, 'deadline', positive=True)
    if time.time() >= deadline_unix:
        raise TimeoutError('immutable worker deadline expired')
    plan = _read(plan_path); _validate_plan(plan)
    jobs = [row for row in plan['schedule'] if row['run_id'] == run_id]
    if len(jobs) != 1:
        raise ValueError('unknown or duplicate run ID')
    job = jobs[0]
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    cache = directory / 'compiler-cache'; cache.mkdir()
    env_before = {name: os.environ.get(name) for name in _CACHE_VARS}
    for name, child in zip(_CACHE_VARS, ('inductor', 'triton')):
        target = cache / child; target.mkdir()
        os.environ[name] = str(target.resolve())
    # Disable optional remote artifact reuse; fresh directories alone would not.
    remote = ('TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE', 'TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE',
              'TORCHINDUCTOR_AOT_AUTOGRAD_REMOTE_CACHE')
    remote_before = {name: os.environ.get(name) for name in remote}
    for name in remote: os.environ[name] = '0'
    _USED_PROCESS = True
    identity = {'worker_id': uuid.uuid4().hex, 'pid': os.getpid(), 'run_id': run_id, **job,
                'plan_sha256': expected_plan_sha256, 'deadline_unix': deadline_unix,
                'started_at': datetime.now(timezone.utc).isoformat(), 'synthetic': False,
                'cache_condition': plan['cache_condition'],
                'compiler_cache_initially_empty': all(not list((cache / n).iterdir()) for n in ('inductor', 'triton')),
                'compiler_cache_paths': {name: os.environ[name] for name in _CACHE_VARS},
                'remote_cache_environment': {name: os.environ[name] for name in remote},
                'interpreter_import_cuda_context_startup_measured': False,
                'quality_preserving': False, 'headline_eligible': False}
    _write(directory / 'started.json', identity)
    started = time.perf_counter()
    try:
        from drivebench import benchmark
        from drivebench.benchmarks.environment import source_identity
        from drivebench.engine import InferenceEngine
        from drivebench_tuning.modern_vista import ModernVistaBackend
        legacy_source_before = source_identity()
        measurement_before = _measurement_sources()
        source_before = digest({'legacy':legacy_source_before,'measurement':measurement_before})
        backend_source_before = _modern_sources()
        original = benchmark.InferenceEngine
        config = deepcopy(plan['configs'][job['role']])
        actual_model = deepcopy(config['model'])
        # Only the legacy config validator sees its existing backend vocabulary.
        # The scoped factory below always instantiates the explicitly named new
        # backend, whose identity is retained in model_identity.json.
        config['model']['backend'] = 'vista'
        execution_config = _archive_modern_environment(config, plan['hardware'], directory / 'benchmark')
        def modern_engine(model, generation, optimizations):
            backend = ModernVistaBackend(actual_model, generation, optimizations)
            return InferenceEngine(actual_model, generation, optimizations, backend=backend)
        try:
            benchmark.InferenceEngine = modern_engine
            _, summary = benchmark.run_benchmark(execution_config, directory / 'benchmark')
        finally:
            benchmark.InferenceEngine = original
        if time.time() >= deadline_unix:
            raise TimeoutError('worker deadline reached; retain evidence but do not accept result')
        if (source_identity() != legacy_source_before or _measurement_sources() != measurement_before
                or _modern_sources() != backend_source_before):
            raise ValueError('measurement implementation changed during worker execution')
        if file_sha256(plan['configs'][job['role']]['data']['manifest']) != plan['input_manifest_sha256']:
            raise ValueError('input manifest changed during worker execution')
        if summary['failures'] or summary['recompiled_during_measurement']:
            raise ValueError('failed or recompiling steady-state run')
        names = ('config.json', 'environment.json', 'environment_gate.json', 'model_identity.json',
                 'cold_start.json', 'measurements.jsonl', 'summary.json',
                 'modern_environment_verification.json', 'modern_runtime_lock.json')
        raw_outputs = {}
        for line in (directory / 'benchmark' / 'measurements.jsonl').read_text().splitlines():
            for candidate in json.loads(line)['candidates']:
                relative = candidate['frames_path']
                target = _contained_output(directory / 'benchmark', relative)
                raw_outputs[relative] = _pin(target)
        artifact_pins = {name: _pin(directory / 'benchmark' / name) for name in names}
        if time.time() >= deadline_unix: raise TimeoutError('evidence hashing exceeded immutable deadline')
        complete = {**identity, 'status': 'completed', 'completed_at': datetime.now(timezone.utc).isoformat(),
                    'source_identity': source_before, 'legacy_source_identity':legacy_source_before,
                    'measurement_implementation_sha256':measurement_before,'source_unchanged': True, 'worker_wall_s': time.perf_counter() - started,
                    'original_modern_config_sha256': digest(plan['configs'][job['role']]),
                    'modern_backend_source_sha256': backend_source_before['modern_vista.py'],
                    'modern_implementation_sha256':backend_source_before,
                    'artifact_pins': artifact_pins, 'raw_output_pins':raw_outputs,
                    'compile_only_s': None,
                    'compile_only_note': 'Not directly instrumented. First request and cold-prefix premium include compilation and other initialization.',
                    'external_process_exit_and_cleanup_still_required': True}
        _write(directory / 'restart.json', complete)
        return complete
    except BaseException as error:
        _write(directory / 'restart-failure.json', {**identity, 'status': 'failed',
                'error_type': type(error).__name__, 'elapsed_s': time.perf_counter() - started})
        raise
    finally:
        for name, value in {**env_before, **remote_before}.items():
            if value is None: os.environ.pop(name, None)
            else: os.environ[name] = value


def _contained_output(root, relative):
    if not isinstance(relative,str) or not relative or Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError('safe relative raw output path required')
    root=Path(root).resolve(); path=root/relative
    if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
        raise ValueError('raw output missing, linked or outside attempt')
    return path


def read_restart(directory, plan, plan_sha):
    directory = Path(directory); evidence = _read(directory / 'restart.json')
    if evidence.get('status') != 'completed' or evidence.get('synthetic') is not False or evidence.get('plan_sha256') != plan_sha:
        raise ValueError('completed real plan-bound restart required')
    if evidence.get('source_unchanged') is not True:
        raise ValueError('source stability was not verified')
    measurement = evidence.get('measurement_implementation_sha256',{})
    if (set(measurement) != {'restart_measurement.py','modern_profile.py','attribution.py','study_cost.py'}
            or any(not isinstance(value,str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value) for value in measurement.values())
            or evidence.get('source_identity') != digest({'legacy':evidence.get('legacy_source_identity'),'measurement':measurement})):
        raise ValueError('complete measurement implementation identity required')
    if evidence.get('compiler_cache_initially_empty') is not True:
        raise ValueError('cold compiler cache condition is unverified')
    job = next((row for row in plan['schedule'] if row['run_id'] == evidence.get('run_id')), None)
    if job is None or any(evidence.get(key) != value for key, value in job.items()):
        raise ValueError('restart schedule identity changed')
    required_artifacts = {'config.json','environment.json','environment_gate.json','model_identity.json','cold_start.json','measurements.jsonl','summary.json','modern_environment_verification.json','modern_runtime_lock.json'}
    if set(evidence.get('artifact_pins', {})) != required_artifacts:
        raise ValueError('complete artifact pin set required')
    if evidence.get('original_modern_config_sha256') != digest(plan['configs'][job['role']]):
        raise ValueError('explicit original modern configuration binding changed')
    for name, pin in evidence['artifact_pins'].items():
        if name not in required_artifacts:
            raise ValueError('unknown artifact path')
        if _pin(directory / 'benchmark' / name) != pin:
            raise ValueError('restart artifact changed')
    root = directory / 'benchmark'; cold = _read(root / 'cold_start.json'); summary = _read(root / 'summary.json')
    rows = [json.loads(line) for line in (root / 'measurements.jsonl').read_text().splitlines() if line.strip()]
    warmups = cold.get('warmups', [])
    expected = plan['configs'][job['role']]
    bench = expected['benchmark']
    if cold.get('status') != 'ready' or not bench.get('warmup_requests',3) <= len(warmups) <= bench.get('max_warmup_requests',12) or len(rows) != plan['warm_requests_per_process']:
        raise ValueError('incomplete cold prefix or measurement count')
    effective = {**expected['generation'], 'candidate_count':1, 'requested_batch_size':1,
                 'actual_batch_sizes':[1], 'optimizations':expected['optimizations']}
    for sequence in (warmups, rows):
      for index,row in enumerate(sequence):
        binding = plan['request_bindings'][index % len(plan['request_bindings'])]
        if (row.get('request_id') != binding['request_id'] or row.get('scene_id') != binding['scene_id']
                or row.get('request_index') != index or row.get('repeat') != index // len(plan['scene_ids'])
                or row.get('candidate_count') != 1 or row.get('batch_size') != 1
                or row.get('input_condition') != 'uncached' or row.get('effective_config') != effective
                or row.get('denoiser_evaluations') != expected['generation']['steps']
                or row.get('cache_stats', {}).get('denoiser_evaluations_complete') is not True
                or row.get('cache_stats', {}).get('context_hit') is not False):
            raise ValueError('raw request workload, order or work counters differ from plan')
    for row in [*warmups, *rows]:
        if row.get('status') != 'ok' or row.get('failures') or row.get('fallbacks') or row.get('cache_stats', {}).get('oom_attempts'):
            raise ValueError('invalid, fallback or failed request')
        _number(row['latency_s'], 'latency', positive=True)
        for key in ('peak_allocated_bytes','peak_reserved_bytes'):
            if type(row.get(key)) is not int or row[key] < 0: raise ValueError('invalid raw memory counter')
    graph_before = summary.get('compiler_before',{}).get('stats',{}).get('unique_graphs',0)
    graph_after = summary.get('compiler_after',{}).get('stats',{}).get('unique_graphs',0)
    if type(graph_before) is not int or type(graph_after) is not int or graph_before < 0 or graph_after != graph_before:
        raise ValueError('steady graph counters are absent, invalid or changed')
    if summary.get('recompiled_during_measurement') is not False or summary.get('failures') != 0:
        raise ValueError('steady state failed or recompiled')
    expected_raw = {}
    for index,row in enumerate(rows):
        binding = plan['request_bindings'][index % len(plan['request_bindings'])]
        candidates = row.get('candidates')
        if not isinstance(candidates,list) or len(candidates) != 1: raise ValueError('exactly one completed future required')
        candidate = candidates[0]
        for key in ('candidate_id','seed','action_kind','trajectory_sha256'):
            if candidate.get(key) != binding[key]: raise ValueError('raw candidate seed/action identity differs')
        path = candidate.get('frames_path')
        if path in expected_raw: raise ValueError('raw output reused by another request')
        expected_raw[path] = _pin(_contained_output(root,path))
    if evidence.get('raw_output_pins') != expected_raw: raise ValueError('raw output hash or coverage differs')
    env = _read(root / 'environment.json')
    from drivebench.benchmarks.environment import reference_identity
    runtime = reference_identity(env)
    gpu = runtime.get('gpu')
    if not isinstance(gpu, dict) or not gpu.get('uuid') or gpu.get('name') != plan['hardware']['gpu_name']:
        raise ValueError('actual GPU differs from explicitly declared hardware')
    try: memory_mib = float(gpu['memory_total_mib'])
    except (KeyError,ValueError,TypeError): raise ValueError('recorded GPU memory required')
    if not math.isfinite(memory_mib) or memory_mib < plan['hardware']['minimum_memory_mib']:
        raise ValueError('actual GPU memory is below the declared minimum')
    gate = _verify_archived_environment(root, plan['hardware'], env)
    if _read(root / 'environment_gate.json') != gate:
        raise ValueError('fresh environment verification binding differs')
    config = _read(root / 'config.json')
    if config.get('profiled') is not False:
        raise ValueError('profiled runs cannot enter headline timing analysis')
    if config.get('data', {}).get('manifest_sha256') != plan['input_manifest_sha256']:
        raise ValueError('benchmark manifest hash differs from plan')
    identity = _read(root / 'model_identity.json')
    if (identity.get('backend') != 'modern_vista'
            or identity.get('modern_runtime') != expected['model'].get('modern_runtime')
            or identity.get('compilation') != {'enabled':job['role'] == 'modern_compiled','target':'denoiser','backend':'inductor','mode':'default','fullgraph':False,'dynamic':False,'compile_cost':'lazy_first_use_must_be_measured','fallback_free_proven':False}
            or identity.get('implementation_sha256') != evidence.get('modern_implementation_sha256')
            or set(identity.get('implementation_sha256',{})) != {'modern_vista.py','modern_attention.py'}
            or identity.get('implementation_sha256',{}).get('modern_vista.py') != evidence.get('modern_backend_source_sha256')):
        raise ValueError('actual modern backend/compile/source identity differs')
    if job['role'] == 'modern_compiled' and summary.get('compiler_before', {}).get('stats', {}).get('unique_graphs', 0) < 1:
        raise ValueError('positive graph capture evidence required for compiled role')
    expected_model = deepcopy(expected['model']); expected_model['backend'] = 'vista'
    if config.get('model') != expected_model:
        raise ValueError('benchmark model differs from explicit modern configuration')
    if config['generation'] != expected['generation'] or config['optimizations'] != expected['optimizations']:
        raise ValueError('effective settings changed')
    initialization = _number(cold['model_load_s'], 'model initialization')
    first = float(warmups[0]['latency_s']); steady = float(np.mean([row['latency_s'] for row in rows]))
    prefix = initialization + sum(float(row['latency_s']) for row in warmups)
    return {**job, 'worker_id': evidence['worker_id'], 'runtime': runtime,
            'source_identity': evidence['source_identity'], 'model_identity': identity,
            'initialization_s': initialization, 'first_request_s': first,
            'cold_prefix_s': prefix, 'cold_prefix_requests': len(warmups),
            'steady_mean_s': steady, 'steady_median_s': float(np.median([row['latency_s'] for row in rows])),
            'steady_p95_s': None, 'steady_p95_note': 'Too few within-process requests for a useful tail estimate.',
            'peak_allocated_bytes': max(row['peak_allocated_bytes'] for row in rows),
            'peak_reserved_bytes': max(row['peak_reserved_bytes'] for row in rows),
            'worker_wall_s': _number(evidence['worker_wall_s'],'worker wall',positive=True),
            'artifact': {'path': str(directory / 'restart.json'), **_pin(directory / 'restart.json')}}


def _break_even(intercept_delta, savings, minimum_requests):
    if savings < 0 or (savings == 0 and intercept_delta >= 0):
        return math.inf
    if savings == 0:  # Equal warm cost and strictly lower startup: already ahead.
        return minimum_requests
    return max(minimum_requests, math.floor(intercept_delta / savings) + 1)


def paired_summary(restarts, *, bootstrap_samples=4000, seed=23):
    """Resample intact AB/BA reversal groups, never individual request rows."""
    if type(bootstrap_samples) is not int or bootstrap_samples < 100:
        raise ValueError('at least 100 bootstrap draws required')
    blocks = sorted({row['block'] for row in restarts})
    if len(blocks) < 8 or len(blocks)%2 or blocks != list(range(len(blocks))) or len(restarts) != 2 * len(blocks):
        raise ValueError('at least four complete AB/BA reversal groups required')
    if len({row['worker_id'] for row in restarts}) != len(restarts):
        raise ValueError('process identity reused')
    runtime = restarts[0]['runtime']; source = restarts[0]['source_identity']
    if any(row['runtime'] != runtime or row['source_identity'] != source for row in restarts):
        raise ValueError('do not pool hardware, runtime or evaluator source')
    model_identity = deepcopy(restarts[0]['model_identity'])
    model_identity.pop('compilation', None)
    for row in restarts:
        identity = deepcopy(row['model_identity']); identity.pop('compilation', None)
        if identity != model_identity:
            raise ValueError('modern model implementation, weights or parameter policy differ')
    pairs = []
    for block in blocks:
        rows = {row['role']: row for row in restarts if row['block'] == block}
        if set(rows) != set(ROLES): raise ValueError('missing or duplicate paired role')
        pairs.append([rows[role] for role in ROLES])
    warm = np.array([[row['steady_mean_s'] for row in pair] for pair in pairs])
    prefix = np.array([[row['cold_prefix_s'] for row in pair] for pair in pairs])
    k = np.array([[row['cold_prefix_requests'] for row in pair] for pair in pairs])
    intercept = prefix - k * warm
    minimum = int(np.max(k))
    def estimates(indices):
        w = warm[indices].mean(axis=0); a = intercept[indices].mean(axis=0)
        return (float(w[0] / w[1] - 1), float(w[0] - w[1]),
                _break_even(float(a[1] - a[0]), float(w[0] - w[1]), minimum))
    point = estimates(np.arange(len(pairs)))
    rng = np.random.default_rng(seed)
    group_count = len(pairs)//2
    samples = []
    for _ in range(bootstrap_samples):
        groups = rng.integers(0,group_count,group_count)
        indices = np.column_stack((2*groups,2*groups+1)).reshape(-1)
        samples.append(estimates(indices))
    samples = np.array(samples)
    # Order-statistic quantiles preserve infinite crossing times; dropping them
    # would manufacture a finite upper bound when warm speedup is uncertain.
    crossings = np.sort(samples[:, 2])
    def extended(value): return None if math.isinf(float(value)) else int(value)
    bounds = [crossings[int(math.floor(q * (len(crossings) - 1)))] for q in (.025, .975)]
    table = []
    for role_index, role in enumerate(ROLES):
        rows = [pair[role_index] for pair in pairs]
        table.append({'baseline': role, 'cohort': 'fresh_paired_modern', 'process_restarts': len(rows),
                      'model_initialization_mean_s': float(np.mean([r['initialization_s'] for r in rows])),
                      'first_request_mean_s': float(np.mean([r['first_request_s'] for r in rows])),
                      'cold_prefix_mean_s': float(np.mean([r['cold_prefix_s'] for r in rows])),
                      'warm_mean_s': float(warm[:, role_index].mean()), 'compile_only_s': None,
                      'peak_allocated_bytes': max(r['peak_allocated_bytes'] for r in rows),
                      'peak_reserved_bytes': max(r['peak_reserved_bytes'] for r in rows)})
    return {'status': 'measured_conditional_runtime_comparison', 'synthetic': False,
            'statistical_unit': 'four-process AB/BA reversal group', 'paired_blocks': len(blocks), 'randomized_reversal_groups':group_count,
            'uncertainty_scope': 'runtime variation on the fixed scene subset; no scene-population or hardware generalization',
            'bootstrap': {'samples': bootstrap_samples, 'seed': seed, 'method': 'intact AB/BA reversal-group percentile bootstrap; draws are not additional independent observations'},
            'baseline_table': [{'baseline': 'legacy_eager', 'cohort': 'historical_separate', 'status': 'not_loaded',
                                'process_restarts': None, 'first_request_mean_s': None, 'warm_mean_s': None}, *table],
            'warm_throughput_gain': point[0], 'warm_throughput_gain_ci95': np.quantile(samples[:,0], [.025,.975]).tolist(),
            'paired_latency_saving_s': point[1], 'paired_latency_saving_ci95_s': np.quantile(samples[:,1], [.025,.975]).tolist(),
            'break_even': {'requests': extended(point[2]), 'finite': math.isfinite(point[2]),
                           'ci95_requests': [extended(x) for x in bounds],
                           'upper_unbounded': math.isinf(float(bounds[1])),
                           'no_finite_crossing_fraction': float(np.mean(np.isinf(samples[:,2]))),
                           'minimum_valid_horizon_requests': minimum,
                           'model': 'T(N)=initialization+sum(cold requests)+(N-cold_count)*mean(warm request); N>=max(cold_count)',
                           'extrapolated_not_observed_crossing': True,
                           'null_bound_means_no_finite_crossing': True},
            'compile_overhead_note': 'Cold-prefix premium includes compile, allocation, autotuning and other initialization; compile-only time is not measured.',
            'quality_preserving': False, 'headline_eligible': False,
            'external_process_exit_and_cleanup_verified': False,
            'acceptance_note': 'This analyzer checks worker evidence. Existing trusted host-runner process-exit, idle-GPU and cleanup receipts must be independently verified before a result is accepted.'}


def historical_legacy_row(summary_path, expected_sha256):
    if file_sha256(summary_path) != expected_sha256: raise ValueError('historical summary hash mismatch')
    summary = _read(summary_path); role = summary['roles']['upstream']
    if summary.get('timing_complete') is not True or role['measured_requests'] != 200:
        raise ValueError('expected completed historical H100 eager cohort')
    return {'baseline': 'legacy_eager', 'cohort': 'historical_2026_10_30_separate',
            'status': 'descriptive_only_not_matched_modern_comparison',
            'historical_summary': {'path': str(summary_path), **_pin(summary_path)},
            'runtime': summary['hardware'], 'warm_median_s': role['latency_s']['p50'],
            'warm_mean_s': role['throughput']['measurement_wall_s'] / role['measured_requests'],
            'throughput_futures_s': role['throughput']['futures_per_s'],
            'peak_allocated_bytes': role['cuda_memory']['peak_allocated_bytes'],
            'process_restarts': None, 'first_request_mean_s': None, 'model_initialization_mean_s': None,
            'compile_only_s': None, 'pooled': False, 'cross_runtime_speedup_claim': False,
            'quality_preserving': False}


def feasibility_summary(restarts, plan):
    if plan['mode'] != 'feasibility' or len(restarts)!=2 or {r['role'] for r in restarts}!=set(ROLES):
        raise ValueError('complete two-process feasibility evidence required')
    if restarts[0]['runtime'] != restarts[1]['runtime'] or restarts[0]['source_identity'] != restarts[1]['source_identity'] or restarts[0]['worker_id']==restarts[1]['worker_id']:
        raise ValueError('same hardware/runtime/source and distinct fresh workers required')
    identities=[]
    for row in restarts:
        identity=deepcopy(row['model_identity']);identity.pop('compilation',None);identities.append(identity)
    if identities[0]!=identities[1]: raise ValueError('feasibility model/weights/source identity differs')
    rate=plan['compute_usd_per_hour_assumption'];rows=[]
    for row in restarts:
        rows.append({key:row[key] for key in ('role','first_request_s','cold_prefix_s','steady_mean_s','peak_allocated_bytes','peak_reserved_bytes','worker_wall_s')})
        rows[-1]['illustrative_worker_compute_usd_per_completed_warm_request']=None if rate is None else rate*row['worker_wall_s']/3600
        rows[-1]['illustrative_warm_compute_usd_per_request']=None if rate is None else rate*row['steady_mean_s']/3600
    return {'status':'full_workload_execution_feasibility_only','hardware':plan['hardware'],'rows':rows,
            'cost_per_quality_valid_request':None,'quality_assessed':False,'quality_preserving':False,
            'noise_or_speedup_claim':False,'subsequent_full_restart_study_admitted':False,
            'cost_scope':'Declared compute rate times observed worker-body elapsed time; excludes interpreter/import prefix, plan validation/cache setup, allocation startup, idle time, storage, profiling and cleanup. Execution validity is not output quality.',
            'external_process_exit_and_cleanup_verified':False}


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('plan'); p.add_argument('--eager-config', required=True); p.add_argument('--compiled-config', required=True)
    p.add_argument('--blocks', type=int, default=12); p.add_argument('--output', required=True)
    p.add_argument('--mode',choices=['paired','feasibility'],default='paired');p.add_argument('--gpu-name',required=True)
    p.add_argument('--minimum-memory-mib',type=int,required=True);p.add_argument('--compute-usd-per-hour',type=float)
    p.add_argument('--minimum-host-ram-gib',type=int,default=32);p.add_argument('--mig-supported',action='store_true')
    r = sub.add_parser('run'); r.add_argument('--plan', required=True); r.add_argument('--plan-sha256', required=True)
    r.add_argument('--run-id', required=True); r.add_argument('--output', required=True); r.add_argument('--deadline-unix', type=float, required=True)
    a = sub.add_parser('analyze'); a.add_argument('--plan', required=True); a.add_argument('--plan-sha256', required=True)
    a.add_argument('--runs', nargs='+', required=True); a.add_argument('--output', required=True)
    a.add_argument('--legacy-summary'); a.add_argument('--legacy-summary-sha256')
    args = parser.parse_args()
    if args.command == 'plan':
        result = make_plan(load_config(args.eager_config), load_config(args.compiled_config), blocks=args.blocks,mode=args.mode,
                           gpu_name=args.gpu_name,minimum_memory_mib=args.minimum_memory_mib,
                           minimum_host_ram_bytes=args.minimum_host_ram_gib*2**30,mig_supported=args.mig_supported,
                           compute_usd_per_hour=args.compute_usd_per_hour)
    elif args.command == 'run':
        collect_restart(args.plan, args.plan_sha256, args.run_id, args.output, deadline_unix=args.deadline_unix); return
    else:
        if file_sha256(args.plan) != args.plan_sha256: raise ValueError('plan hash mismatch')
        plan = _read(args.plan); _validate_plan(plan)
        rows = [read_restart(path, plan, args.plan_sha256) for path in args.runs]
        if {r['run_id'] for r in rows} != {r['run_id'] for r in plan['schedule']} or len(rows) != len(plan['schedule']):
            raise ValueError('all scheduled restarts, including failures, must be accounted for; no incomplete-case estimate')
        result = paired_summary(rows) if plan['mode']=='paired' else feasibility_summary(rows,plan)
        result['plan_sha256'] = args.plan_sha256
        result['restart_evidence'] = [row['artifact'] for row in rows]
        if bool(args.legacy_summary) != bool(args.legacy_summary_sha256):
            raise ValueError('historical summary and hash must be supplied together')
        if args.legacy_summary and plan['mode']!='paired': raise ValueError('no historical comparison table in feasibility smoke')
        if args.legacy_summary:
            result['baseline_table'][0] = historical_legacy_row(args.legacy_summary, args.legacy_summary_sha256)
    _write(args.output, result)


if __name__ == '__main__': main()
