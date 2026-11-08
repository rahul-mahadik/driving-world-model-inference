#!/usr/bin/env python3
"""Prepared-only, bounded four-scene A100 47-step screening; no acceptance."""
from __future__ import annotations
import argparse
import importlib.util
import math
from pathlib import Path
import shutil
import signal
import statistics
import sys
import time
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
REL = 'outputs/host/a100-math-step47-v1'
RESULT = 'outputs/a100-math-step47-v1/runs/eager47'
QUALITY = 'outputs/host/a100-math-quality-v1/quality.py'
QUALITY_SHA = '233b18a2d0e64f937978de9b65db5a03c86f175fa04fa29e0ecc6c0885422724'
EVALUATOR = 'outputs/host/a100-math-quality-eval-v1/evaluate.py'
EVALUATOR_SHA = 'fd493adc303cd72f5742dac23f3285930c58563ac70cae2d185a66177e006b07'
SPEC_PINS = {
    'eager50': '4a6a7795641614442724f2da2e69bce6926e61eff6d742a1ee0d70d3a816411b',
    'eager40': '15b5ad07fbf1ee75c316d5469bad39c82323d6d9f98084a9524b6554bfc11b5c',
    'feature2': '64957bd5ccea793b5eca707ad8052fd51e5bc2a4f244bdac7d89047887add6cb',
}
OUTER, INNER, RESERVE = 1800, 1680, 1200
FREE_FLOOR, WORKING_MARGIN = 12 * 1024**3, 2 * 1024**3
ARRAY_FILE_BYTES = 176947328
DEADLINE = 1793168129.682277
SCOPE = ('Four fixed development smoke scenes, one recorded candidate and fixed seed each, '
         '47 steps and one warmup under the separate A100 math-SDPA runtime. '
         '47 steps is a screening hypothesis from four-scene IO-confounded timing medians only. '
         'No matched-latency, quality-preservation, optimized-baseline, H100 adoption, speedup, '
         'threshold-calibration or headline claim. Original H100 plan and all old evidence unchanged.')


class ScreeningTimeout(BaseException):
    """Escape request-level Exception handlers; the host enforces the outer bound."""


def load_pinned(name, relative, expected):
    import hashlib
    path = ROOT / relative
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Frozen dependency changed: ' + relative)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quality():
    return load_pinned('step47_frozen_quality', QUALITY, QUALITY_SHA)


def expected_config(original, q):
    cfg = q.expected_config(original, 'eager50')
    cfg['name'] = 'a100-math-step47-v1-eager47'
    cfg['output'] = '/workspace/' + RESULT
    cfg['generation']['steps'] = 47
    cfg['a100_pilot']['scope'] = SCOPE
    cfg['a100_quality_smoke'].update(variant='eager47', scope=SCOPE)
    cfg['a100_step47_screening'] = {'scope': SCOPE, 'automatic_followup': False,
        'prior_timings_may_include_archive_io': True, 'retire_outputs': False,
        'matched_latency_claimed': False, 'optimized_baseline_claimed': False}
    return cfg


def config_checked(q, context):
    from drivebench.config import load_config, validate_run_config
    path = q.project_file(REL + '/configs/eager47.json')
    cfg = load_config(path)
    if cfg != expected_config(context.original, q):
        raise ValueError('Exact four-scene 47-step eager screening configuration required')
    validate_run_config(cfg)
    return path, cfg


def hypothesis(q, items):
    if set(items) != set(SPEC_PINS):
        raise ValueError('All three completed smoke generations required')
    medians = {}
    for name, item in items.items():
        rows = item['run'].measurements
        if len(rows) != 4:
            raise ValueError('Exactly four measured requests required for hypothesis')
        medians[name] = statistics.median(q.positive(r['latency_s']) for r in rows)
    slope = (medians['eager50'] - medians['eager40']) / 10
    if not math.isfinite(slope) or slope <= 0:
        raise ValueError('Measured step medians do not support interpolation')
    return {'measured_medians_s': medians,
        'linear_interpolated_steps': 40 + (medians['feature2'] - medians['eager40']) / slope,
        'predicted_47_step_seconds': medians['eager40'] + 7 * slope,
        'selected_steps': 47, 'selection_scope': 'Unvalidated screening hypothesis only',
        'prior_timings_may_include_archive_io': True,
        'latency_matching_established': False, 'quality_claims_imported': False}


def estimate(q, items):
    maximum = max(q.positive(x['baseline']['result']['max_observed_request_s']) for x in items.values())
    load = max(q.positive(x['baseline']['result']['model_load_s']) for x in items.values())
    empirical = 5 * maximum + load + 900
    if not math.isfinite(empirical) or empirical >= INNER:
        raise ValueError('Measured prior work cannot support the fixed inner bound')
    return {'outer_timeout_seconds': OUTER, 'internal_timeout_seconds': INNER,
        'archive_evaluation_cleanup_reserve_seconds': RESERVE,
        'empirical_estimate_seconds': empirical, 'empirical_bound_guaranteed': False,
        'method': 'Five worst observed prior requests including warmups plus model load and900s checks; no assumed47-step speedup',
        'funding_check': 'Fresh provider quote, balance and unchanged active guard required before host launch.'}


def admission_limits(*, now=None, free_bytes=None):
    now = time.time() if now is None else now
    if type(now) not in (int, float) or not math.isfinite(now) or now + OUTER + RESERVE >= DEADLINE:
        raise ValueError('Fixed job and1200s reserve exceed immutable allocation deadline')
    needed = 4 * ARRAY_FILE_BYTES + FREE_FLOOR + WORKING_MARGIN
    free_bytes = shutil.disk_usage(ROOT / 'outputs').free if free_bytes is None else free_bytes
    if type(free_bytes) is not int or free_bytes < needed:
        raise ValueError('Require four arrays plus12GiB floor and2GiB working margin')
    return {'required_free_bytes': needed, 'observed_free_bytes': free_bytes,
        'array_file_bytes': ARRAY_FILE_BYTES, 'array_count': 4,
        'preserved_free_floor_bytes': FREE_FLOOR, 'working_margin_bytes': WORKING_MARGIN}


def prepare(expected_sha):
    q = quality()
    if q.sha(__file__) != expected_sha or q.DEADLINE != DEADLINE:
        raise ValueError('Reviewed screening helper or fixed deadline differs')
    e = load_pinned('step47_frozen_evaluator', EVALUATOR, EVALUATOR_SHA)
    context = q.dependencies(QUALITY_SHA)
    path, cfg = config_checked(q, context)
    records, wanted, inputs = q.smoke_inputs(cfg)
    admitted = context.verifier.gate(context.original, DEADLINE)
    strict = q.strict_prerequisite(context, admitted)
    _, base_cfg = q.config_checked(context, 'eager50')
    baseline = q.completed_baseline(context, admitted, strict, base_cfg, wanted)
    items = {}
    for name, digest in SPEC_PINS.items():
        spec = q.REL + '/' + name + '-run-01-spec.json'
        if q.sha(q.project_file(spec)) != digest:
            raise ValueError('Preserved smoke generation specification changed')
        items[name] = e.generation(q, context, admitted, strict, name, spec, items.get('eager50'))
        inputs.update(items[name]['bindings'])
    if items['eager50']['baseline'] != baseline:
        raise ValueError('Independent baseline provenance views disagree')
    inputs.update(strict['artifact_sha256'])
    inputs.update(q.bindings([__file__, QUALITY, EVALUATOR, path]))
    q.check_files(inputs)
    return NS(q=q, context=context, config=cfg, wanted=wanted, records=records,
        admitted=admitted, strict=strict, inputs=inputs,
        hypothesis=hypothesis(q, items), bound=estimate(q, items))


def run_artifacts(plan, output):
    from drivebench.evaluation.io import load_run, output_path
    q = plan.q
    result = q.run_artifacts(output, plan.config, plan.wanted, plan.admitted)
    run = load_run(output)
    if run.config.get('a100_step47_screening') != plan.config['a100_step47_screening']:
        raise ValueError('Screening scope changed in recorded config')
    paths = []
    for row in run.measurements:
        p = output_path(run, row['candidates'][0]['frames_path'])
        if p.is_symlink() or not p.resolve().is_relative_to(output.resolve()) or p.stat().st_size != ARRAY_FILE_BYTES:
            raise ValueError('Require four distinct project-owned complete arrays')
        paths.append(p.resolve())
    if len(set(paths)) != 4 or {p.resolve() for p in output.rglob('*.npy')} != set(paths):
        raise ValueError('Duplicated, extra or missing retained arrays')
    return result


def admission_fields(plan):
    return {'platform_gate': plan.admitted, 'strict_prerequisite': plan.strict,
        'input_sha256': plan.inputs, 'time_bound': plan.bound, 'hypothesis': plan.hypothesis,
        'automatic_launch_authorized': False, 'scope': SCOPE}


def admit(output, expected_sha):
    output = Path(output)
    if output.exists() or output.is_symlink() or output.resolve() != (ROOT / REL / 'admission-01.json').resolve():
        raise ValueError('Use exact fresh screening admission path')
    plan = prepare(expected_sha)
    value = {'type': 'a100_math_step47_admission_v1', 'created_at': plan.context.base.stamp(),
        'producer_sha256': expected_sha, 'source_identity': plan.q.SOURCE,
        'allocation_deadline_epoch': DEADLINE, 'disk': admission_limits(), **admission_fields(plan)}
    plan.q.check_files(plan.inputs)
    plan.context.base.save(output, value)
    print({'admission': str(output), 'sha256': plan.q.sha(output), 'automatic_launch_authorized': False})
    return 0


def run(admission_path, output, expected_sha):
    q = quality()
    admission_path = q.project_file(admission_path)
    admission = q.read(admission_path)
    if (admission_path.resolve() != (ROOT / REL / 'admission-01.json').resolve()
            or admission.get('type') != 'a100_math_step47_admission_v1'
            or admission.get('producer_sha256') != expected_sha or admission.get('source_identity') != q.SOURCE
            or admission.get('allocation_deadline_epoch') != DEADLINE
            or not 0 <= time.time() - q.epoch(admission.get('created_at')) <= 300):
        raise ValueError('Fresh unchanged screening admission required')
    output = Path(output)
    if output.resolve() != (ROOT / RESULT).resolve() or output.exists() or output.is_symlink():
        raise ValueError('Use exact fresh screening output path')
    def timeout(_signum, _frame):
        raise ScreeningTimeout('Fixed screening inner time bound expired')
    previous = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(INNER)
    value = None
    try:
        plan = prepare(expected_sha)
        if any(admission.get(k) != v for k, v in admission_fields(plan).items()):
            raise ValueError('Current proof, hypothesis, inputs or admission bound changed')
        disk = admission_limits()
        output.mkdir(parents=True)
        inputs = {**plan.inputs, **q.bindings([admission_path])}
        value = {'type': 'a100_math_step47_generation_v1', 'status': 'running', 'variant': 'eager47',
            'started_at': plan.context.base.stamp(), 'producer_sha256': expected_sha, 'source_identity': q.SOURCE,
            **admission_fields(plan), 'input_sha256': inputs,
            'algorithm_policy': plan.context.runtime.declaration(plan.context.base),
            'admission_path': str(admission_path), 'admission_sha256': q.sha(admission_path),
            'allocation_deadline_epoch': DEADLINE, 'disk_at_start': disk,
            'legacy_numerical_passed': False, 'headline_eligible': False, 'quality_preserving': False,
            'h100_reference_eligible': False, 'evaluation_pending': True,
            'matched_latency_claimed': False, 'optimized_baseline_claimed': False,
            'variant_exact_agreement_claimed': False, 'policy_execution': {}}
        plan.context.base.save(output / 'attempt.json', value)
        import torch
        from drivebench.benchmark import run_benchmark
        from drivebench.benchmarks.environment import collect_environment, reference_identity
        with plan.context.runtime.activate(torch, plan.context.ab, plan.context.base, value['policy_execution']):
            destination, _ = run_benchmark(plan.config, output, profiled=False)
            if Path(destination).resolve() != output.resolve():
                raise ValueError('Benchmark wrote an unexpected destination')
        q.lifecycle(value['policy_execution'], plan.context.ab.snapshot(torch, plan.context.base))
        if reference_identity(collect_environment()) != plan.admitted['runtime_identity']:
            raise ValueError('Original platform identity changed after generation')
        value['result'] = run_artifacts(plan, output)
        if q.strict_prerequisite(plan.context, plan.admitted) != plan.strict:
            raise ValueError('Strict provenance changed during generation')
        q.check_files(inputs)
        if shutil.disk_usage(ROOT / 'outputs').free < FREE_FLOOR:
            raise ValueError('Original12GiB free floor not preserved')
        value.update(status='completed', inputs_and_source_unchanged=True, platform_restored=True)
    except BaseException as error:
        if value is None:
            raise
        value.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
    finally:
        try:
            if value is not None:
                value['finished_at'] = plan.context.base.stamp()
                value['partial_artifact_sha256'] = {str(p.relative_to(output)): q.sha(p) for p in output.rglob('*') if p.is_file()}
                plan.context.base.save(output / 'completed.json', value)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
    print({'status': value['status'], 'output': str(output), 'headline_eligible': False})
    return 0 if value['status'] == 'completed' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['admit', 'run'])
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--admission')
    args = parser.parse_args()
    if args.command == 'run' and not args.admission:
        parser.error('run requires --admission')
    raise SystemExit(admit(args.output, args.expected_sha256) if args.command == 'admit'
        else run(args.admission, args.output, args.expected_sha256))
