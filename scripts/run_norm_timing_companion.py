#!/usr/bin/env python3
"""Separate compiler norm-qualified timing, preserving all legacy strict flags.

The upstream variability pilot and old timing/report tools are unmodified.
This contract never turns a failed schema2 numerical proof into a passed one.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drivebench import calibrate_pilot as pilot, study
from drivebench.config import digest, file_sha256, validate_run_config
from drivebench.benchmarks.agreement import numerical_evidence
from drivebench.data.manifests import load_manifest
from drivebench.evaluation.io import artifact_sha256, load_run
from drivebench.regression import _bootstrap_ratios, _pairs, timing_signature

TYPE = 'compiler_norm_timing_companion_v1'
RESULT = 'compiler_norm_timing_result_v1'
SOURCE = 'd8a32813492c0939a0a91573a945c914938972419946065e6822f6209fa470cb'
HELPERS = ('run_timing_companion.py', 'qualify_compiler_numerics.py')


def module(name):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location('norm_timing_' + path.stem, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def read(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError('Expected JSON object')
    return value


def write_once(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError('Immutable completed evidence differs: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Core writer provides fsync; exclusive evidence is protected by execution
    # locks. Existing documents are never replaced with different contents.
    study.atomic_json(path, value)


def source_bindings():
    return {str(Path(__file__).resolve()): file_sha256(__file__),
            **{str(Path(__file__).with_name(name).resolve()): file_sha256(Path(__file__).with_name(name))
               for name in HELPERS}}


def check_hashes(values):
    for path, expected in values.items():
        if file_sha256(path) != expected:
            raise ValueError('Bound artifact changed: ' + str(path))


def serial_config(cfg):
    if (cfg['generation']['steps'] != 50 or cfg['optimizations']['max_batch_size'] != 1
            or cfg['optimizations']['feature_cache_interval'] != 1
            or cfg['model'].get('compiler_feature_boundary') is not None
            or cfg['benchmark']['candidate_count'] not in (1, 4)
            or cfg['benchmark']['requests'] != 50 or cfg['benchmark']['save_outputs'] != 0
            or cfg['benchmark'].get('headline') is not False
            or cfg['benchmark']['warmup_requests'] != 3 or cfg['benchmark']['max_warmup_requests'] != 5):
        raise ValueError('This contract is fixed50-step serial single-future/candidate4 timing only')


def strict_baseline(proof_path, cfg, calibration):
    proof = read(proof_path)
    if (cfg['model']['backend'] != 'upstream' or any(pilot._compiler_selectors(cfg).values())
            or proof.get('schema_version') != 2 or proof.get('passed') is not True
            or proof.get('atol') != 0 or proof.get('rtol') != 0
            or proof.get('implementation_sha256') != SOURCE
            or proof.get('runtime_identity') != calibration['identity']['runtime']):
        raise ValueError('Baseline must use a passed exact current-source upstream proof on the pilot GPU')
    if (not proof.get('checks') or 'upstream' not in proof.get('runs', {})
            or any(c.get('passed') is not True or c.get('atol', 0) != 0 or c.get('rtol', 0) != 0
                   for c in proof.get('checks', {}).values())):
        raise ValueError('Baseline proof contains a failed check')
    pilot._proof_compiler_identity(cfg, proof)
    for details in proof.get('runs', {}).values():
        outputs, hashes = details.get('outputs', {}), details.get('outputs_sha256', {})
        if not outputs or set(outputs) != set(hashes):
            raise ValueError('Baseline proof output coverage missing')
        check_hashes({path: hashes[key] for key, path in outputs.items()})
    generation, opts = validate_run_config(cfg)
    environment = read(cfg['environment_verification'])['environment']
    gate = numerical_evidence(cfg, generation, opts, environment, proof['runs']['upstream']['model_identity'],
                              load_manifest(cfg['data']['manifest']))
    if gate.get('passed') is not True:
        raise ValueError('Baseline strict proof does not qualify the selected workload: ' + '; '.join(gate.get('reasons', [])))
    return proof


def exact_pilot(calibration):
    cfg = calibration['configuration']; proof = read(cfg['numerical_validation'])
    if (cfg['model']['backend'] != 'upstream' or any(pilot._compiler_selectors(cfg).values())
            or proof.get('atol') != 0 or proof.get('rtol') != 0 or proof.get('passed') is not True):
        raise ValueError('This contract requires the unchanged strict-zero upstream pilot')


def norm_binding(path, cfg, calibration):
    report = module('qualify_compiler_numerics.py').verify(path)
    if (report.get('type') != 'compiler_norm_qualification_v1' or report.get('qualified') is not True
            or report.get('legacy_strict_passed') is not False
            or report.get('implementation_sha256') != SOURCE
            or report.get('runtime_identity') != calibration['identity']['runtime']
            or any(report.get(k, {}).get('passed') is not True
                   for k in ('exact_semantics', 'numerical', 'quality_development', 'localized_review'))):
        raise ValueError('Current full-runtime norm qualification and independent exact semantics are required')
    if report.get('generation') != cfg['generation'] or report.get('optimizations') != cfg['optimizations']:
        raise ValueError('Norm qualification generation or optimization settings differ')
    identity = report['model_identity']
    if cfg['model'].get('backend') != 'vista' or not cfg['optimizations']['compile_denoiser']:
        raise ValueError('Candidate must be the explicitly qualified compiled Vista implementation')
    for name in ('upstream_commit', 'checkpoint_sha256', 'backend'):
        if not identity.get(name) or identity[name] != cfg['model'].get(name):
            raise ValueError('Norm qualification model identity differs')
    if identity.get('inference_config_sha256') != file_sha256(cfg['model']['inference_config']):
        raise ValueError('Norm qualification inference config changed')
    for key, field in (('compiler_runtime', 'name'), ('compiler_feature_boundary', 'module')):
        selected, actual = cfg['model'].get(key), identity.get(key)
        if ((selected is None and actual is not None)
                or (selected is not None and (not isinstance(actual, dict) or actual.get(field) != selected))):
            raise ValueError('Norm qualification compiler selectors differ')
    legacy = read(report['legacy_proof']['path'])
    if (legacy.get('schema_version') != 2 or legacy.get('passed') is not False
            or legacy.get('atol') != 0 or legacy.get('rtol') != 0
            or len(legacy.get('input_provenance', {}).get('candidate_ids', [])) < 2):
        raise ValueError('Preserve a failed exact proof with multi-candidate semantic coverage')
    pilot._time(report['created_at']); pilot._time(report['policy']['frozen_at'])
    if file_sha256(report['legacy_proof']['path']) != report['legacy_proof']['sha256']:
        raise ValueError('Legacy strict proof changed')
    return report


def reconstructed_configs(parent_plan, selected, names, count, calibration, proof_path, qualification_path, runtime_config):
    old = module('run_timing_companion.py')
    overlay, overlay_binding = old._runtime_overlay(parent_plan, runtime_config)
    template = copy.deepcopy(parent_plan)
    template['timing'].update(warmups=3, blocks=4, requests_per_block=50, candidate_count=count)
    environment = read(calibration['configuration']['environment_verification'])['environment']
    configs = {}
    for name in names:
        cfg = study._variant_config(template, name, parent_plan['manifests']['evaluation']['path'], 50, count, quality=False)
        cfg.update(overlay); serial_config(cfg)
        old._matching_runtime(cfg, calibration['identity']['runtime'])
        if pilot._identity(cfg, environment) != calibration['identity']:
            raise ValueError('Timing workload/generation differs from its own upstream pilot')
        configs[name] = cfg
    baseline, candidate = names
    reference = configs[baseline]
    if (reference['model']['backend'] != calibration['reference_backend']
            or reference['optimizations'] != calibration['reference_optimizations']
            or pilot._compiler_selectors(reference) != pilot._reference_compiler_selectors(calibration)
            or reference['benchmark']['input_condition'] != calibration['reference_input_condition']):
        raise ValueError('Baseline implementation differs from upstream variability pilot')
    reference['numerical_validation'] = str(Path(proof_path).resolve())
    strict_baseline(proof_path, reference, calibration)
    qualification = norm_binding(qualification_path, configs[candidate], calibration)
    development = parent_plan['manifests']['development']
    if (qualification.get('development_manifest', {}).get('sha256') != development['sha256']
            or qualification.get('development_manifest', {}).get('scene_ids') != development['metadata']['scene_ids']):
        raise ValueError('Norm qualification must use the same frozen32-scene development partition')
    configs[candidate]['numerical_validation'] = qualification['legacy_proof']['path']
    return configs, qualification, overlay_binding


def stable_config(cfg):
    result = copy.deepcopy(cfg)
    result.get('study_provenance', {}).pop('recorded_at', None)
    return result


def plan_companion(study_path, output, *, baseline, candidate, candidate_count, pilot_policy,
                   baseline_proof, qualification, runtime_config=None):
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError('Use a fresh norm timing companion directory')
    if type(candidate_count) is not int or candidate_count not in (1, 4):
        raise ValueError('Choose single future1 or serial candidate4')
    parent, parent_plan = study._load_plan(study_path)
    selected = study._selection(parent, parent_plan)
    names = [baseline, candidate]
    if baseline != parent_plan['reference'] or baseline == candidate or not set(names) <= set(selected['variants']):
        raise ValueError('Choose the upstream reference and one distinct development-frozen compiler finalist')
    policy_path, proof_path, qualification_path = (Path(p).resolve() for p in (pilot_policy, baseline_proof, qualification))
    policy, calibration = pilot.validate_pilot_policy(policy_path)
    if digest(pilot._sources()) != SOURCE or calibration['source_identity'] != SOURCE:
        raise ValueError('Only frozen d8 source and its unchanged upstream pilot are supported')
    # The baseline pilot remains the original strict-zero workflow.
    exact_pilot(calibration)
    configs, qualified, overlay = reconstructed_configs(parent_plan, selected, names, candidate_count,
        calibration, proof_path, qualification_path, runtime_config)
    old = module('run_timing_companion.py')
    value = {'schema_version': 1, 'type': TYPE, 'created_at': study.now(), 'parent_study': str(parent),
        'parent_plan_sha256': file_sha256(parent / 'plan.json'), 'selection_sha256': file_sha256(parent / 'selection.json'),
        'quality_evidence_sha256': old._quality_bindings(selected), 'source_identity': SOURCE,
        'helper_sha256': source_bindings(), 'pilot_policy': {'path': str(policy_path), 'sha256': file_sha256(policy_path)},
        'baseline_proof': {'path': str(proof_path), 'sha256': file_sha256(proof_path)},
        'qualification_path': str(qualification_path), 'qualification': qualified, 'timing_runtime_overlay': overlay,
        'baseline': baseline, 'candidate': candidate, 'candidate_count': candidate_count,
        'runtime_identity': calibration['identity']['runtime'],
        'workload': 'single_future' if candidate_count == 1 else 'candidate_set',
        'schedule': study.timing_schedule(names, 4, 50), 'configs': configs,
        'legacy_strict_passed': {baseline: True, candidate: False}, 'save_outputs': 0,
        'combined_headline_claim': False, 'quality_generated_here': False,
        'scope': 'New norm-qualified timing contract only. Legacy compiler numerical/headline flags remain false; separate complete heldout quality, second-seed, action and accepted review evidence remains mandatory.'}
    destination.mkdir(parents=True)
    write_once(destination / 'plan.json', value)
    write_once(destination / 'plan-receipt.json', {'plan_sha256': file_sha256(destination / 'plan.json')})
    return value


def _load(directory):
    directory = Path(directory).resolve(); path = directory / 'plan.json'
    value = read(path)
    if (read(directory / 'plan-receipt.json').get('plan_sha256') != file_sha256(path)
            or value.get('type') != TYPE or value.get('source_identity') != SOURCE
            or value.get('helper_sha256') != source_bindings() or digest(pilot._sources()) != SOURCE):
        raise ValueError('Immutable norm timing plan, helper or source changed')
    parent, parent_plan = study._load_plan(value['parent_study']); selected = study._selection(parent, parent_plan)
    old = module('run_timing_companion.py')
    if (file_sha256(parent / 'plan.json') != value['parent_plan_sha256']
            or file_sha256(parent / 'selection.json') != value['selection_sha256']
            or old._quality_bindings(selected) != value['quality_evidence_sha256']):
        raise ValueError('Parent selection or frozen quality evidence changed')
    names = [value['baseline'], value['candidate']]; count = value['candidate_count']
    if (type(count) is not int or count not in (1, 4) or names[0] != parent_plan['reference']
            or names[0] == names[1] or not set(names) <= set(selected['variants'])
            or value['schedule'] != study.timing_schedule(names, 4, 50)
            or value['workload'] != ('single_future' if count == 1 else 'candidate_set')
            or value.get('save_outputs') != 0 or value.get('combined_headline_claim') is not False
            or value.get('legacy_strict_passed') != {names[0]: True, names[1]: False}):
        raise ValueError('Finalists, schedule, workload or legacy flags changed')
    for key in ('pilot_policy', 'baseline_proof'):
        check_hashes({value[key]['path']: value[key]['sha256']})
    policy, calibration = pilot.validate_pilot_policy(value['pilot_policy']['path'])
    if calibration['source_identity'] != SOURCE:
        raise ValueError('Pilot source differs')
    exact_pilot(calibration)
    saved = value['timing_runtime_overlay']
    configs, qualified, overlay = reconstructed_configs(parent_plan, selected, names, count, calibration,
        value['baseline_proof']['path'], value['qualification_path'], saved['path'] if saved else None)
    if (qualified != value['qualification'] or overlay != saved
            or value['runtime_identity'] != calibration['identity']['runtime']
            or {k: stable_config(v) for k, v in configs.items()} != {k: stable_config(v) for k, v in value['configs'].items()}):
        raise ValueError('Qualification, allocation or selected configuration changed')
    return directory, value, parent, policy, calibration


def artifacts(run):
    return {**artifact_sha256(run), **{name: file_sha256(run.path / name) for name in
        ('cold_start.json', 'environment_gate.json', 'numerical_gate.json', 'model_identity.json')}}


def compiler_row(row, identity, *, measured):
    runtime = row.get('cache_stats', {}).get('compiler_runtime', {})
    if (runtime.get('passed') is not True or runtime.get('restoration_complete') is not True
            or runtime.get('require_graphs') is not True
            or runtime.get('runtime_implementation_sha256') != identity['compiler_runtime']['implementation_sha256']
            or type(runtime.get('graphs_created_by_runtime')) is not int or runtime['graphs_created_by_runtime'] <= 0
            or type(runtime.get('kernels_created_by_runtime')) is not int or runtime['kernels_created_by_runtime'] <= 0):
        raise ValueError('Measured compiler runtime lacks positive codegen or complete restoration')
    before, after = runtime.get('before', {}), runtime.get('after', {})
    if after.get('unimplemented') or before.get('unimplemented'):
        raise ValueError('Compiler unimplemented fallback detected')
    if measured and any(before.get(k) != after.get(k) for k in ('unique_graphs', 'generated_kernels')):
        raise ValueError('Compiler changed during a measured request')


def validate_block(run, cfg, value, policy, calibration):
    records, metadata = pilot._manifest(cfg['data']['manifest'], 'held_out_evaluation')
    if len(records) != 128 or len(metadata['scene_ids']) < 64:
        raise ValueError('Timing requires the frozen128-clip heldout partition')
    if set(metadata['scene_ids']) & set(policy['evidence']['development_scenes']):
        raise ValueError('Heldout timing scenes overlap the upstream development pilot')
    study._validate_completed_run(run.path, cfg, records)
    pilot._valid_run(run, calibration, pilot=False, minimum_requests=50)
    if any(run.config.get(k) != cfg.get(k) for k in ('model', 'generation', 'optimizations')):
        raise ValueError('Block implementation differs from frozen selected configuration')
    if (run.config.get('numerical_validation') != cfg['numerical_validation']
            or any(run.config['benchmark'].get(k) != cfg['benchmark'].get(k) for k in pilot.PROTOCOL)
            or run.config['benchmark'].get('headline') is not False or run.summary.get('headline_eligible') is not False
            or run.config['data'].get('manifest_metadata') != metadata
            or run.config['data'].get('manifest_sha256') != file_sha256(cfg['data']['manifest'])):
        raise ValueError('Block protocol, legacy flags or partition provenance differs')
    start = pilot._time(run.config['started_at'])
    latest = max(pilot._time(policy['frozen_at']), pilot._time(value['created_at']),
                 pilot._time(value['qualification']['created_at']), pilot._time(value['qualification']['policy']['frozen_at']))
    if start <= latest:
        raise ValueError('Timing must begin after pilot, norm policy, qualification and plan freeze')
    if (run.summary.get('synthetic') is not False or run.summary.get('profiled') is not False
            or run.summary.get('recompiled_during_measurement') is not False
            or run.summary.get('measured_requests') != 50 or run.summary.get('successful_requests') != 50
            or run.summary.get('completed_futures') != 50*cfg['benchmark']['candidate_count']
            or run.summary.get('failures') != 0 or run.summary.get('oom_attempts') != 0):
        raise ValueError('Summary lacks explicit complete unprofiled execution evidence')
    environment_gate = read(run.path / 'environment_gate.json')
    if environment_gate.get('passed') is not True:
        raise ValueError('Block environment gate failed')
    identity = read(run.path / 'model_identity.json')
    generation, opts = validate_run_config(run.config)
    gate = read(run.path / 'numerical_gate.json')
    recomputed = numerical_evidence(run.config, generation, opts, run.environment, identity, records)
    if gate != recomputed or run.summary.get('numerical_validation') != gate:
        raise ValueError('Raw legacy numerical gate differs from actual strict evidence')
    baseline = cfg['model']['backend'] == 'upstream'
    if gate.get('passed') is not baseline:
        raise ValueError('Legacy strict pass/failure must remain visible and unchanged')
    if baseline:
        pilot._execution_gates(run, calibration)
    elif identity != value['qualification']['model_identity']:
        raise ValueError('Actual compiled model identity differs from norm qualification')
    cold = read(run.path / 'cold_start.json'); warmups = cold.get('warmups', [])
    if (cold.get('status') != 'ready' or cold.get('compilation_and_initialization_excluded') is not True
            or not 3 <= len(warmups) <= 5):
        raise ValueError('Each block needs3–5 independently completed warmups')
    count, steps = cfg['benchmark']['candidate_count'], cfg['generation']['steps']
    for measured, rows in ((False, warmups), (True, run.measurements)):
        for row in rows:
            if (row.get('status') != 'ok' or row.get('failures') or row.get('fallbacks')
                    or row.get('cache_stats', {}).get('oom_attempts') or row.get('oom_attempts')
                    or row.get('candidate_count') != count
                    or row.get('effective_config', {}).get('actual_batch_sizes') != [1]*count
                    or row.get('effective_config', {}).get('requested_batch_size') != 1
                    or row.get('effective_config', {}).get('optimizations') != cfg['optimizations']
                    or any(row.get('effective_config', {}).get(k) != v for k, v in cfg['generation'].items())
                    or row.get('denoiser_evaluations') != steps*count
                    or row.get('cache_stats', {}).get('denoiser_evaluations_complete') is not True):
                raise ValueError('Block has failed, fallback, OOM, missing work or nonserial execution')
            if not baseline:
                compiler_row(row, identity, measured=measured)
    warm_graphs = [r.get('compiler', {}).get('stats', {}).get('unique_graphs', 0) for r in warmups[-3:]]
    before, after = (run.summary.get(k, {}) for k in ('compiler_before', 'compiler_after'))
    if (len(set(warm_graphs)) != 1 or before.get('stats', {}).get('unique_graphs', 0) != after.get('stats', {}).get('unique_graphs', 0)
            or before.get('unimplemented') or after.get('unimplemented')):
        raise ValueError('Compiler graphs did not stabilize before all measured requests')
    return {'legacy_numerical_passed': gate['passed'], 'legacy_headline_eligible': run.summary['headline_eligible'],
            'norm_execution_eligible': True, 'artifact_sha256': artifacts(run)}


def block_evidence(directory, item, value, policy, calibration):
    block = directory / 'blocks' / f"{item['index']:03d}"
    receipt_path = block / 'norm-completed.json'; receipt = read(receipt_path)
    if (receipt.get('index') != item['index'] or receipt.get('variant') != item['variant']
            or receipt.get('variant_block') != item['variant_block']
            or receipt.get('plan_sha256') != file_sha256(directory / 'plan.json')
            or receipt.get('qualification_sha256') != value['qualification']['report_sha256']):
        raise ValueError('Block completion belongs to a different plan or qualification')
    run = load_run(receipt['path'])
    start_path = block / 'norm-started.json'; started = read(start_path)
    if (receipt.get('start_receipt_sha256') != file_sha256(start_path)
            or started.get('plan_sha256') != receipt['plan_sha256']
            or started.get('index') != item['index'] or started.get('variant') != item['variant']
            or started.get('source_identity') != SOURCE
            or not pilot._time(started['started_at']) <= pilot._time(run.config['started_at']) <= pilot._time(receipt['finished_at'])):
        raise ValueError('Block start or measured process chronology differs')
    expected = validate_block(run, value['configs'][item['variant']], value, policy, calibration)
    if receipt.get('execution') != expected or receipt.get('artifact_sha256') != artifacts(run):
        raise ValueError('Block raw artifacts or execution qualification changed')
    core = read(block / 'completed' / (item['variant'] + '.json'))
    if (core.get('adopted') is not False or core.get('path') != str(run.path) or core.get('outputs') != []
            or core.get('artifact_sha256') != artifact_sha256(run)
            or core.get('generation_source_evidence', {}).get('generation_sources') != study.generation_sources()):
        raise ValueError('Block must be an independently executed source-bound benchmark')
    process_path = run.path.parent / 'process.json'; process = read(process_path)
    if (process.get('exit_code') != 0 or process.get('config_sha256') != file_sha256(run.path.parent / 'config.yaml')
            or yaml.safe_load((run.path.parent / 'config.yaml').read_text()) != value['configs'][item['variant']]
            or not pilot._time(started['started_at']) <= pilot._time(process['finished_at']) <= pilot._time(receipt['finished_at'])):
        raise ValueError('Benchmark process failed or its launch config changed')
    return run, {**receipt, 'receipt_path': str(receipt_path), 'receipt_sha256': file_sha256(receipt_path),
                 'start_receipt': started, 'process_path': str(process_path), 'process_sha256': file_sha256(process_path)}


def statistics(rows_by_variant, names, policy):
    """Compute on validated observations without rewriting any legacy flag/row."""
    vectors, pairkeys, identities = {}, None, None
    for name in names:
        blocks = rows_by_variant[name]
        if len(blocks) != 4 or any(len(rows) != 50 for rows in blocks):
            raise ValueError('Four complete50-request blocks per variant required')
        keys, candidate_ids, values, scenes = [], [], [], []
        for index, rows in enumerate(blocks):
            for row in rows:
                latency = row['latency_s']
                if type(latency) not in (int, float) or not math.isfinite(latency) or latency <= 0:
                    raise ValueError('Finite positive raw latencies required')
                keys.append((index, row['request_id'], row['scene_id'], row.get('repeat', 0)))
                candidate_ids.append(pilot._candidate_inputs(row)); values.append(latency); scenes.append(row['scene_id'])
        if len(set(keys)) != 200 or (pairkeys is not None and (keys != pairkeys or candidate_ids != identities)):
            raise ValueError('Every request, action and seed must be paired in identical order')
        pairkeys, identities = keys, candidate_ids
        vectors[name] = (np.asarray(values), scenes)
    a, b = (vectors[name][0] for name in names); scenes = vectors[names[0]][1]
    metrics = {}; tolerance = policy['relative_latency_tolerance']
    for name, q in (('p50', .5), ('p95', .95)):
        point, ci = _bootstrap_ratios(a, b, scene_groups=scenes, quantile=q)
        metrics[name] = {'relative_latency_change': point, 'ci95': ci, 'tolerance': tolerance,
            'exceeds_tolerance_and_uncertainty': point > tolerance and ci[0] > tolerance,
            'improvement_exceeds_tolerance_and_uncertainty': point < -tolerance and ci[1] < -tolerance}
    exceeds = any(v['exceeds_tolerance_and_uncertainty'] for v in metrics.values())
    possible = any(v['relative_latency_change'] > tolerance for v in metrics.values())
    return {'status': 'rerun_required' if exceeds else 'inconclusive' if possible else 'pass',
        'metrics': metrics, 'paired_requests': 200, 'independent_scenes': len(set(scenes)),
        'block_medians_s': {n: [float(np.median([r['latency_s'] for r in b])) for b in rows_by_variant[n]] for n in names},
        'latency_s': {n: {'p50': float(np.quantile(vectors[n][0], .5)), 'p95': float(np.quantile(vectors[n][0], .95))} for n in names},
        'uncertainty': '4000 paired scene-cluster percentile bootstrap resamples; all repeated requests remain grouped. Eight interleaved processes; intervals do not guarantee stationarity.',
        'interpretation': 'Regression pass is not a speedup or quality claim. Apparent regressions require independent confirmation.',
        'quality_preserving': False, 'combined_headline_claim': False}


def reconstruct(directory, *, require_completed=True):
    directory, value, parent, policy, calibration = _load(directory)
    rows = {n: [] for n in (value['baseline'], value['candidate'])}; blocks = []; signatures = set(); previous_end = None
    for item in value['schedule']:
        run, evidence = block_evidence(directory, item, value, policy, calibration)
        signature = timing_signature(run)
        if signature in signatures or str(run.path) in {b['path'] for b in blocks}:
            raise ValueError('Copied timings or duplicate process paths cannot be independent blocks')
        signatures.add(signature)
        start = pilot._time(evidence['start_receipt']['started_at'])
        if previous_end is not None and start < previous_end:
            raise ValueError('Timing blocks overlap or violate the interleaved schedule')
        previous_end = pilot._time(evidence['finished_at'])
        rows[item['variant']].append(run.measurements); blocks.append(evidence)
    comparison = statistics(rows, list(rows), policy)
    legacy = {name: {b['execution']['legacy_numerical_passed'] for b in blocks if b['variant'] == name} for name in rows}
    if legacy != {value['baseline']: {True}, value['candidate']: {False}}:
        raise ValueError('Legacy strict flags changed')
    result = {'type': RESULT, 'timing_qualified': True, 'combined_headline_claim': False,
        'legacy_strict_passed': value['legacy_strict_passed'],
        'legacy_headline_eligible': {name: False for name in rows},
        'plan_path': str(directory / 'plan.json'), 'plan_sha256': file_sha256(directory / 'plan.json'),
        **{k: value[k] for k in ('parent_study', 'parent_plan_sha256', 'selection_sha256', 'baseline', 'candidate', 'configs',
            'qualification', 'pilot_policy', 'runtime_identity', 'source_identity', 'schedule', 'candidate_count', 'workload')},
        'blocks': blocks, 'comparison': comparison, 'quality_generated_here': False,
        'artifact_sha256': {str(directory / 'plan.json'): file_sha256(directory / 'plan.json'),
            **{b['receipt_path']: b['receipt_sha256'] for b in blocks},
            **{str(Path(b['path']) / key): val for b in blocks for key, val in b['artifact_sha256'].items()}}}
    if require_completed:
        completed = read(directory / 'completed.json')
        if completed != result:
            raise ValueError('Completed norm timing evidence differs from reconstructed raw blocks')
    _load(directory) # Revalidate qualification/proof/source after all block reads.
    return result


def verify_completed(directory):
    return reconstruct(directory, require_completed=True)


def confirm_completed(primary_directory, confirmation_directory):
    """Read-only independent confirmation; never schedules another measurement."""
    primary, confirmation = (verify_completed(path) for path in (primary_directory, confirmation_directory))
    if primary['comparison']['status'] != 'rerun_required':
        raise ValueError('Confirmation is reserved for an apparent regression')
    for key in ('parent_plan_sha256', 'selection_sha256', 'baseline', 'candidate', 'candidate_count',
                'workload', 'runtime_identity', 'source_identity', 'qualification', 'pilot_policy'):
        if primary[key] != confirmation[key]:
            raise ValueError('Confirmation identity or qualification differs')
    if {k: stable_config(v) for k, v in primary['configs'].items()} != {k: stable_config(v) for k, v in confirmation['configs'].items()}:
        raise ValueError('Confirmatory implementations differ')
    blocks = [*primary['blocks'], *confirmation['blocks']]
    if (len({b['path'] for b in blocks}) != 16
            or len({timing_signature(load_run(b['path'])) for b in blocks}) != 16
            or min(pilot._time(b['start_receipt']['started_at']) for b in confirmation['blocks'])
               <= max(pilot._time(b['finished_at']) for b in primary['blocks'])):
        raise ValueError('Confirmation must be separately collected after the primary measurement')
    repeated = [name for name in ('p50', 'p95')
                if primary['comparison']['metrics'][name]['exceeds_tolerance_and_uncertainty']
                and confirmation['comparison']['metrics'][name]['exceeds_tolerance_and_uncertainty']]
    return {'type': 'compiler_norm_timing_confirmation_v1',
            'status': 'regression_confirmed' if repeated else 'not_confirmed',
            'repeated_metrics': repeated, 'primary': primary, 'confirmation': confirmation,
            'combined_headline_claim': False, 'automatic_measurement_performed': False}


def run_companion(directory, *, max_new_blocks=None, deadline=None, block_budget_seconds=None,
                  cleanup_seconds=240):
    if max_new_blocks is not None and (type(max_new_blocks) is not int or max_new_blocks < 1):
        raise ValueError('max_new_blocks must be a positive integer')
    if (deadline is None) != (block_budget_seconds is None):
        raise ValueError('Deadline admission requires both an absolute deadline and a full-block budget')
    if (type(cleanup_seconds) not in (int, float) or not math.isfinite(cleanup_seconds) or cleanup_seconds < 240
            or (block_budget_seconds is not None and (type(block_budget_seconds) not in (int, float)
                or not math.isfinite(block_budget_seconds) or block_budget_seconds <= 0))):
        raise ValueError('Finite positive block budget and at least240-second cleanup margin required')
    deadline_time = pilot._time(deadline) if deadline is not None else None
    directory, value, parent, policy, calibration = _load(directory)
    with (parent / '.execution.lock').open('a') as parent_lock, (directory / '.execution.lock').open('a') as lock:
        fcntl.flock(parent_lock, fcntl.LOCK_EX | fcntl.LOCK_NB); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        new_blocks = 0
        for item in value['schedule']:
            _load(directory)
            block = directory / 'blocks' / f"{item['index']:03d}"
            if (block / 'norm-completed.json').exists():
                block_evidence(directory, item, value, policy, calibration); continue
            if block.exists() and any(block.iterdir()):
                raise ValueError('Incomplete started block is preserved; automatic retry is forbidden')
            if any((directory / 'blocks' / f"{later['index']:03d}").exists()
                   for later in value['schedule'][item['index'] + 1:]):
                raise ValueError('Noncontiguous block evidence violates the immutable schedule')
            reason = None
            if max_new_blocks is not None and new_blocks >= max_new_blocks:
                reason = 'max_new_blocks'
            if deadline_time is not None and (deadline_time - pilot._time(study.now())).total_seconds() < block_budget_seconds + cleanup_seconds:
                reason = 'deadline_admission'
            if reason:
                paused = {'type': 'compiler_norm_timing_pause_v1', 'status': 'paused', 'incomplete': True,
                    'timing_qualified': False, 'combined_headline_claim': False, 'reason': reason,
                    'created_at': study.now(), 'plan_sha256': file_sha256(directory / 'plan.json'),
                    'completed_blocks': item['index'], 'remaining_blocks': 8 - item['index'],
                    'new_blocks_this_invocation': new_blocks, 'next_block': item,
                    'legacy_strict_passed': value['legacy_strict_passed'],
                    'admission': {'deadline': deadline, 'block_budget_seconds': block_budget_seconds,
                                  'cleanup_seconds': cleanup_seconds, 'max_new_blocks': max_new_blocks},
                    'scope': 'Complete blocks only; the execution bound is an admission estimate, not a runtime guarantee.'}
                pauses = directory / 'pauses'; pauses.mkdir(exist_ok=True)
                write_once(pauses / f'{len(list(pauses.glob("*.json"))):03d}.json', paused)
                return paused
            block.mkdir(parents=True, exist_ok=True)
            started_path = block / 'norm-started.json'
            if not started_path.exists():
                write_once(started_path, {'started_at': study.now(), 'index': item['index'], 'variant': item['variant'],
                    'plan_sha256': file_sha256(directory / 'plan.json'), 'source_identity': SOURCE})
            cfg = copy.deepcopy(value['configs'][item['variant']]); records = load_manifest(cfg['data']['manifest'])
            receipt = study._execute_run(block, item['variant'], cfg, records)
            _load(directory)
            run = load_run(receipt['path']); execution = validate_block(run, cfg, value, policy, calibration)
            write_once(block / 'norm-completed.json', {'index': item['index'], 'variant': item['variant'],
                'variant_block': item['variant_block'], 'path': str(run.path), 'finished_at': study.now(),
                'plan_sha256': file_sha256(directory / 'plan.json'), 'start_receipt_sha256': file_sha256(started_path),
                'qualification_sha256': value['qualification']['report_sha256'],
                'artifact_sha256': artifacts(run), 'execution': execution})
            block_evidence(directory, item, value, policy, calibration)
            new_blocks += 1
        result = reconstruct(directory, require_completed=False)
        write_once(directory / 'completed.json', result)
        return verify_completed(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('plan')
    for field in ('study', 'output', 'baseline', 'candidate', 'pilot-policy', 'baseline-proof', 'qualification'):
        p.add_argument('--' + field, required=True)
    p.add_argument('--candidate-count', type=int, required=True); p.add_argument('--runtime-config')
    for name in ('run', 'verify'):
        p = sub.add_parser(name); p.add_argument('--companion', required=True)
        if name == 'run':
            p.add_argument('--max-new-blocks', type=int)
            p.add_argument('--deadline', help='Absolute timezone-aware allocation deadline')
            p.add_argument('--block-budget-seconds', type=float)
            p.add_argument('--cleanup-seconds', type=float, default=240)
    p = sub.add_parser('confirm'); p.add_argument('--primary', required=True); p.add_argument('--confirmation', required=True)
    a = parser.parse_args()
    if a.command == 'plan':
        result = plan_companion(a.study, a.output, baseline=a.baseline, candidate=a.candidate,
            candidate_count=a.candidate_count, pilot_policy=a.pilot_policy, baseline_proof=a.baseline_proof,
            qualification=a.qualification, runtime_config=a.runtime_config)
        print(json.dumps({'status': 'planned_only', 'slots': len(result['schedule']), 'combined_headline_claim': False}))
    elif a.command == 'confirm':
        result = confirm_completed(a.primary, a.confirmation)
        print(json.dumps({'status': result['status'], 'combined_headline_claim': False, 'automatic_measurement_performed': False}))
    else:
        result = (run_companion(a.companion, max_new_blocks=a.max_new_blocks, deadline=a.deadline,
                               block_budget_seconds=a.block_budget_seconds, cleanup_seconds=a.cleanup_seconds)
                  if a.command == 'run' else verify_completed(a.companion))
        print(json.dumps(result if result.get('status') == 'paused' else
            {'status': 'complete', 'timing_qualified': result['timing_qualified'],
             'legacy_strict_passed': result['legacy_strict_passed'],
             'performance_status': result['comparison']['status'], 'combined_headline_claim': False}))


if __name__ == '__main__':
    main()
