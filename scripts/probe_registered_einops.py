#!/usr/bin/env python3
"""Separate strict compiler probe with the official pinned einops registration.

The original probe helper and historical configs remain untouched. Execute in a
fresh process under timeout --signal=TERM --kill-after=30s 600s. No tolerance
calibration or headline claim is performed here.
"""
from __future__ import annotations

import argparse
import contextlib
from copy import deepcopy
import inspect
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drivebench.config import digest, file_sha256
from scripts import probe_vista_optimization as base


HELPER_SHA256 = {
    'scripts/probe_vista_optimization.py': '7ba82bf7efe5ca86ad498867a25ba4fef92d4610bdb85fc2df37ac9156d168dd',
    'scripts/validate_compiled_attention_boundary.py': 'f266438c63ca8fa5b7a954906fe4c1ecdf88a7d969cf4d8e2693aa8f2c553854',
    'scripts/validate_compiled_preflight.py': '6c1aa61b6a8c656a4c541d6bf11a2dc62a9c6350769d52af36c1ebabc96a3334',
}
EINOPS_SHA256 = {
    '_torch_specific.py': '3b2e546ea034e446afbe60cb79450d271f6202b2db212c7a40f0753aa2278530',
    'einops.py': '6d5af8549c684fb0a4b2261e8207a18ffcb638102f7e2f52a4d1e101f0f722a6',
    'packing.py': 'cfd55f3fc49680d4ef8bbe9ab6662c970eb4f004e47fcbca64749351af6caeb2',
}
FUNCTIONS = ('rearrange', 'reduce', 'repeat', 'einsum', 'pack', 'unpack')
DECLARATION = {
    'kind': 'official_einops_registration_with_explicit_dynamo_cache_limit',
    'einops_version': '0.6.1',
    'registration_api': 'einops._torch_specific.allow_ops_in_compiled_graph',
    'registered_functions': list(FUNCTIONS),
    'cache_size_limit': 512,
    'native_attention_boundary': True,
    'suppress_errors': False,
}


def bound_files(config_path):
    """Bind full inherited config chain plus immutable reused helper sources."""
    import yaml
    root = Path(__file__).resolve().parents[1]
    result = {str(Path(__file__).resolve()): file_sha256(__file__)}
    for relative, expected in HELPER_SHA256.items():
        path = root / relative
        if file_sha256(path) != expected:
            raise ValueError('Frozen helper changed: ' + relative)
        result[str(path)] = expected
    path, seen = Path(config_path).resolve(), set()
    while True:
        if path in seen:
            raise ValueError('Config inheritance cycle')
        seen.add(path)
        result[str(path)] = file_sha256(path)
        value = yaml.safe_load(path.read_text())
        if not isinstance(value, dict):
            raise ValueError('Config must be a mapping')
        if not value.get('extends'):
            break
        path = (path.parent / value['extends']).resolve()
    return result


def inspect_config(path):
    config, generation, options = base.inspect_config(path)
    if config['probe']['kind'] not in {'denoiser_default', 'denoiser_dynamic', 'denoiser_reduce_overhead'}:
        raise ValueError('Registered-einops probes support only declared denoiser variants')
    if config.get('compiler_runtime') != DECLARATION:
        raise ValueError('Compiler runtime declaration must exactly select registration and cache limit512')
    return config, generation, options


def load_runtime():
    import torch
    import torch._dynamo
    import einops
    import einops.einops
    import einops.packing
    import einops._torch_specific
    if torch.__version__ != '2.0.1+cu118' or einops.__version__ != '0.6.1':
        raise ValueError('Registered compiler probe requires pinned Torch2.0.1+cu118/einops0.6.1')
    registration = einops._torch_specific.allow_ops_in_compiled_graph
    path = Path(inspect.getsourcefile(registration) or '')
    if registration.__module__ != 'einops._torch_specific' or registration.__name__ != 'allow_ops_in_compiled_graph':
        raise ValueError('Official registration API was replaced')
    files = {str(path.parent / name): file_sha256(path.parent / name) for name in EINOPS_SHA256}
    if {Path(name).name: value for name, value in files.items()} != EINOPS_SHA256:
        raise ValueError('Installed einops source differs from official0.6.1 wheel')
    functions = {}
    for name in FUNCTIONS:
        owner = einops.packing if name in {'pack', 'unpack'} else einops.einops
        function = getattr(einops, name)
        if function is not getattr(owner, name) or function.__module__ != owner.__name__:
            raise ValueError('A registered einops function was replaced: ' + name)
        functions[name] = function
    return torch._dynamo, registration, functions, files


@contextlib.contextmanager
def registered_runtime(state):
    dynamo, registration, functions, files = load_runtime()
    if dynamo.config.disable or dynamo.config.suppress_errors is not False:
        raise ValueError('Dynamo must be enabled with compiler errors unsuppressed')
    previous_limit = dynamo.config.cache_size_limit
    if previous_limit != 64:
        raise ValueError('Use a fresh pinned process with the original cache limit64')
    allowed = dynamo.allowed_functions._allowed_function_ids
    disallowed = dynamo.allowed_functions._disallowed_function_ids
    before = {name: {'allowed': id(fn) in allowed, 'disallowed': id(fn) in disallowed}
              for name, fn in functions.items()}
    if any(item['allowed'] or item['disallowed'] for item in before.values()):
        raise ValueError('Use a fresh process without prior einops graph-registration overrides')
    state.update(einops_source_files=files, registration_before=before,
                 cache_size_limit_before=previous_limit, restoration_complete=False)
    try:
        registration()  # The official API registers exactly these six functions.
        after = {name: {'allowed': id(fn) in allowed, 'disallowed': id(fn) in disallowed}
                 for name, fn in functions.items()}
        if not all(item['allowed'] and not item['disallowed'] for item in after.values()):
            raise RuntimeError('Official einops graph registration did not take effect')
        dynamo.config.cache_size_limit = 512
        state.update(registration_after=after, applied=True, cache_size_limit_active=512)
        yield
    finally:
        dynamo.config.cache_size_limit = previous_limit
        for name, fn in functions.items():
            for registry, key in ((allowed, 'allowed'), (disallowed, 'disallowed')):
                if before[name][key]:
                    registry.add(id(fn))
                else:
                    registry.remove(id(fn))
        state['restoration_complete'] = (dynamo.config.cache_size_limit == previous_limit and
            all((id(fn) in allowed) == before[name]['allowed'] and
                (id(fn) in disallowed) == before[name]['disallowed'] for name, fn in functions.items()))
        state['einops_source_unchanged'] = all(file_sha256(path) == value for path, value in files.items())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    config, _, _ = inspect_config(args.config)
    declared_output = args.output or Path(config['output'])
    if declared_output.exists() or declared_output.is_symlink():
        raise FileExistsError('Use a fresh attempt; preserve previous evidence')
    output = declared_output.resolve()
    files = bound_files(args.config)
    state = {'schema_version': 1, 'status': 'initializing', 'declaration': deepcopy(DECLARATION),
             'wrapper_sha256': file_sha256(__file__), 'bound_file_sha256': files,
             'headline_eligible': False, 'quality_validated': False,
             'scope': 'new registered-einops compiler probe; strict0 diagnostic, not tolerance calibration',
             'official_source': 'https://github.com/arogozhnikov/einops/blob/v0.6.1/einops/_torch_specific.py',
             'official_wheel_sha256': '99149e46cc808956b174932fe563d920db4d6e5dadb8c6ecdaa7483b7ef7cfc3'}
    original_save, original_boundary = base.save, base.attention_boundary
    report_path = output / 'probe.json'
    output_owned = False

    def save_with_provenance(path, value):
        nonlocal output_owned
        if Path(path) == report_path:
            output_owned = True  # The frozen helper successfully created the fresh directory.
            value = {**value, 'registered_compiler_wrapper': deepcopy(state)}
        return original_save(path, value)

    @contextlib.contextmanager
    def combined_boundary(enabled):
        if enabled is not True or state.get('applied') is not True:
            raise ValueError('Both explicit runtime changes and native attention boundary are required')
        with original_boundary(enabled) as native:
            yield {'kind': 'registered_einops_and_native_attention_boundary',
                   'native_attention_boundary': native, 'registration': deepcopy(DECLARATION),
                   'wrapper_sha256': state['wrapper_sha256'], 'bound_files_sha256': digest(files),
                   'einops_source_files': dict(state['einops_source_files'])}

    code = 1
    base.save, base.attention_boundary = save_with_provenance, combined_boundary
    try:
        with registered_runtime(state):
            state['status'] = 'running'
            code = base.main(['--config', str(args.config), '--output', str(output)])
            state['status'] = 'completed'
    except BaseException as error:
        state.update(status='failed', error={'type': type(error).__name__, 'message': str(error),
                                            'traceback': traceback.format_exc()})
        if isinstance(error, SystemExit) and isinstance(error.code, int):
            code = error.code or 1
        traceback.print_exc()
    finally:
        base.save, base.attention_boundary = original_save, original_boundary
        try:
            state['bound_files_unchanged'] = all(file_sha256(path) == value for path, value in files.items())
            if (not state['bound_files_unchanged'] or state.get('restoration_complete') is not True
                    or state.get('einops_source_unchanged') is not True):
                code = 1
                state['status'] = 'failed_runtime_or_source_check'
        except Exception as error:
            code = 1
            state.update(status='failed_finalization', finalization_error=str(error))
        if not output_owned:
            # A competing attempt must never be overwritten, even if setup failed.
            output.mkdir(parents=True, exist_ok=False)
        report = json.loads(report_path.read_text()) if report_path.is_file() else {
            'schema_version': 1, 'status': 'failed_before_base_probe', 'executions': [],
            'strict_atol': 0, 'strict_rtol': 0, 'headline_eligible': False, 'quality_validated': False}
        report['registered_compiler_wrapper'] = state
        if code == 1 and report.get('exit_code') in {0, 2}:
            report['status'] = 'registered_wrapper_failed'
        report['exit_code'] = code
        original_save(report_path, report)
        original_save(output / 'registered-runtime.json', {**state, 'probe_report_sha256': file_sha256(report_path), 'exit_code': code})
    print(json.dumps({'report': str(report_path), 'registered_runtime': str(output / 'registered-runtime.json'), 'exit_code': code}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
