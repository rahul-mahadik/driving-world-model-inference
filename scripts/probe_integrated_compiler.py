#!/usr/bin/env python3
"""Bounded opt-in compiler/feature/shape discovery through the real adapter.

The historical probe stays byte-for-byte unchanged. This wrapper supplies only
validated configs and comparison inputs; compiler behavior is owned by the
normal Vista backend. Zero tolerance remains a diagnostic, not acceptance.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import replace
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from drivebench.config import digest, file_sha256, load_config, validate_run_config
from drivebench.types import GenerationConfig, OptimizationConfig
from scripts import probe_vista_optimization as base
from scripts.probe_candidate_memory import config_files


def bound_files(path):
    result = config_files(path)
    for script in (__file__, base.__file__, 'scripts/probe_candidate_memory.py',
                   'scripts/validate_compiled_attention_boundary.py', 'scripts/validate_compiled_preflight.py'):
        value = Path(script).resolve()
        result[str(value)] = file_sha256(value)
    return result


def inspect_config(path):
    config = load_config(path)
    generation, options = validate_run_config(config)
    probe = config.get('probe', {})
    kind = probe.get('kind')
    if kind not in {'denoiser_default', 'shape_static', 'shape_dynamic'}:
        raise ValueError('Integrated probe supports default compiler or declared static/dynamic shape transition')
    boundary = config['model'].get('compiler_feature_boundary')
    interval = options.feature_cache_interval
    expected = replace(OptimizationConfig(), compile_denoiser=True, dynamic_shapes=kind == 'shape_dynamic')
    if boundary:
        if boundary != 'output_blocks.9' or kind != 'denoiser_default' or interval not in {1,2,3,4}:
            raise ValueError('Selected feature boundary requires default/static output9 at interval1/2/3/4')
        expected = replace(expected, feature_cache_module='output_blocks.9', feature_cache_interval=interval)
    if generation != GenerationConfig() or options != expected:
        raise ValueError('Integrated probe must retain declared independent primary settings')
    if (config['model'].get('compiler_runtime') != 'registered_einops_attention_v1'
            or config['model'].get('backend') != 'vista' or config.get('numerical_validation') is not None
            or config.get('benchmark', {}).get('headline') is not False
            or config.get('benchmark', {}).get('candidate_count') != 1
            or config.get('data', {}).get('verify_checksums') is not True):
        raise ValueError('Require explicit integrated runtime, one candidate, checksums and no headline/proof')
    if probe.get('strict_atol') != 0 or probe.get('strict_rtol') != 0 or probe.get('cuda_graph_trace', False):
        raise ValueError('Preserve zero tolerance and keep CUDA graph tracing a separate discovery')
    for key, lower, upper in [('request_index', 0,31), ('max_warmups',2,3), ('measurements',1,1)]:
        if type(probe.get(key)) is not int or not lower <= probe[key] <= upper:
            raise ValueError('Invalid bounded integrated probe ' + key)
    if boundary and not probe.get('comparison_probe'):
        raise ValueError('Boundary probes require a prior unmodified integrated compiled50 comparison probe')
    return config, generation, options


def compiled_reference(path, config, upstream, upstream_proof):
    """Bind an actual compiled50 output to the same verified upstream request."""
    from drivebench.benchmarks.environment import source_identity
    root = Path(path).resolve()
    report_path = root / 'probe.json'
    report = json.loads(report_path.read_text())
    prior_config = report['config']
    _, prior_options = validate_run_config(prior_config)
    expected = replace(OptimizationConfig(), compile_denoiser=True)
    if (prior_options != expected or prior_config['generation'] != GenerationConfig().to_dict()
            or prior_config['model'].get('compiler_runtime') != 'registered_einops_attention_v1'
            or prior_config['model'].get('compiler_feature_boundary') is not None
            or prior_config.get('probe', {}).get('kind') != 'denoiser_default'
            or prior_config.get('probe', {}).get('comparison_probe') is not None
            or report.get('status') not in {'execution_complete', 'execution_complete_strict_failed'}
            or report.get('exit_code') not in {0,2} or not report.get('positive_cuda_codegen_evidence')
            or not report.get('source_unchanged') or not report.get('launcher_and_config_unchanged')
            or not report.get('integrated_runtime', {}).get('bound_files_unchanged')
            or report.get('implementation_sha256') != source_identity()):
        raise ValueError('Compiled reference is not a completed current-source unmodified compiled50 probe')
    if report['reference'] != upstream_proof:
        raise ValueError('Compiled reference was not checked against the same saved upstream request and runtime')
    prior_bindings = report.get('integrated_runtime', {}).get('bound_file_sha256', {})
    if not prior_bindings or not all(Path(path).is_file() and file_sha256(path) == sha for path, sha in prior_bindings.items()):
        raise ValueError('Compiled reference helper or config changed after its probe')
    if report['config']['data'] != config['data'] or report['config']['probe']['request_index'] != config['probe']['request_index']:
        raise ValueError('Compiled reference input identity differs')
    for key in ['checkpoint_sha256', 'upstream_commit', 'inference_config_sha256']:
        if report['model_identity'].get(key) != upstream_proof['model_identity'].get(key):
            raise ValueError('Compiled reference model identity differs')
    rows = [row for row in report['executions'] if row['label'] == 'measured-0']
    if len(rows) != 1:
        raise ValueError('Compiled reference measured request missing or duplicated')
    row = rows[0]
    runtime = row.get('cache_stats', {}).get('compiler_runtime', {})
    if (row.get('status') != 'ok' or row.get('fallbacks') or row.get('cache_stats', {}).get('oom_attempts')
            or row.get('effective_config', {}).get('actual_batch_sizes') != [1]
            or row.get('denoiser_evaluations') != 50 or not runtime.get('passed')
            or not runtime.get('restoration_complete') or row.get('compiler', {}).get('unimplemented')):
        raise ValueError('Compiled reference did not complete the exact work without fallback')
    output = Path(row['output_path']).resolve()
    if not output.is_relative_to(root) or file_sha256(output) != row['output_sha256']:
        raise ValueError('Compiled reference output escaped directory or changed')
    array = np.load(output, mmap_mode='r', allow_pickle=False)
    if array.shape != upstream.shape or array.dtype != np.float32:
        raise ValueError('Compiled reference tensor layout differs')
    proof = dict(upstream_proof)
    proof.update(scope='Same-request integrated compiled50 output; upstream strict outcome retained separately',
                 upstream_comparison=report.get('strict_passed'), compiled_probe_report=str(report_path),
                 compiled_probe_report_sha256=file_sha256(report_path), output_path=str(output),
                 output_sha256=row['output_sha256'], model_identity=report['model_identity'])
    return array, proof


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    config, _, _ = inspect_config(args.config)
    bindings = bound_files(args.config)
    original_inspect, original_boundary, original_reference, original_save = (
        base.inspect_config, base.attention_boundary, base.load_reference, base.save)
    owned = False
    report_path = None
    declaration = {'kind': 'normal_adapter_registered_runtime', 'bound_file_sha256': bindings,
                   'compiler_runtime': config['model']['compiler_runtime'],
                   'compiler_feature_boundary': config['model'].get('compiler_feature_boundary'),
                   'global_patches_by_wrapper': False,
                   'active_cache_size_limit': 512, 'historical_probe_unchanged': True}

    def current():
        return all(Path(path).is_file() and file_sha256(path) == sha for path, sha in bindings.items())

    def save(path, value):
        nonlocal owned, report_path
        if path.name == 'probe.json':
            owned, report_path = True, path
            value['integrated_runtime'] = {**declaration, 'bound_files_unchanged': current()}
        original_save(path, value)

    def reference(config, request, record, generation):
        array, proof = original_reference(config, request, record, generation)
        prior = config['probe'].get('comparison_probe')
        return compiled_reference(prior, config, array, proof) if prior else (array, proof)

    try:
        base.inspect_config = inspect_config
        # The adapter alone owns registration, stable attention/boundary wrappers,
        # the execution lock, graph evidence and restoration.
        base.attention_boundary = lambda _enabled: contextlib.nullcontext(declaration)
        base.load_reference, base.save = reference, save
        command = ['--config', str(args.config)]
        if args.output:
            command += ['--output', str(args.output)]
        code = base.main(command)
        if not current():
            code = 1
            if owned:
                report = json.loads(report_path.read_text())
                report.update(status='integrated_probe_source_changed', exit_code=1)
                save(report_path, report)
        return code
    finally:
        base.inspect_config, base.attention_boundary, base.load_reference, base.save = (
            original_inspect, original_boundary, original_reference, original_save)


if __name__ == '__main__':
    raise SystemExit(main())
