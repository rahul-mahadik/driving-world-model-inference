#!/usr/bin/env python3
"""Run one frozen development bootstrap with fresh, declared compiler caches.

The external bounded launcher owns GPU serialization, timeout and allocation
cleanup. This wrapper adds no instrumentation or model patches; after verifying
the recipe it calls the ordinary drivebench.benchmark entry point.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml

from drivebench.benchmarks.environment import source_identity
from drivebench.config import digest, file_sha256, load_config, validate_run_config
from drivebench.types import GenerationConfig, OptimizationConfig
from dataclasses import replace

FROZEN_RECIPE_SHA256 = 'd60c904ab65f80120b00ba07e287d2cb8e0b758f829ef378337190bbee9e2d91'
FROZEN_SOURCE = 'd8a32813492c0939a0a91573a945c914938972419946065e6822f6209fa470cb'
VARIANTS = ('compiled-default', 'compiled-conditioning')
MIN_FREE_BYTES = 18 * 1024**3


def project_path(root, value):
    path = Path(value)
    if path.is_absolute():
        try:
            path = path.relative_to('/workspace')
        except ValueError as error:
            raise ValueError('Recipe absolute paths must be inside /workspace') from error
    if '..' in path.parts or not path.parts:
        raise ValueError('Invalid project path')
    result = root / path
    if not result.resolve().is_relative_to(root):
        raise ValueError('Project path escapes through a symlink')
    return result


def config_bindings(root, path):
    result = {}
    while path is not None:
        path = path.resolve()
        if not path.is_relative_to(root):
            raise ValueError('Config inheritance escaped project root')
        relative = str(path.relative_to(root))
        if relative in result:
            raise ValueError('Config inheritance cycle')
        result[relative] = file_sha256(path)
        value = yaml.safe_load(path.read_text())
        if not isinstance(value,dict):
            raise ValueError('Inherited config must be a mapping')
        parent = value.get('extends')
        path = path.parent / parent if parent else None
    return result


def validate_recipe(hostspec, variant, *, root=None):
    root = Path(root or Path(__file__).resolve().parents[1]).resolve()
    hostspec = Path(hostspec).resolve()
    if not hostspec.is_relative_to(root) or file_sha256(hostspec) != FROZEN_RECIPE_SHA256:
        raise ValueError('Host recipe differs from the frozen reviewed recipe')
    recipe = json.loads(hostspec.read_text())
    if (recipe.get('schema_version') != 1 or recipe.get('expected_source_identity') != FROZEN_SOURCE
            or source_identity() != FROZEN_SOURCE or variant not in VARIANTS):
        raise ValueError('Unknown variant or changed frozen package source')
    matches = [run for run in recipe['runs'] if run['adoption_variant'] == variant]
    if len(matches) != 1:
        raise ValueError('Recipe variant is absent or duplicated')
    run = matches[0]
    config_path = project_path(root, run['config'])
    bindings = config_bindings(root, config_path)
    if bindings != run['inherited_config_sha256'] or file_sha256(config_path) != run['config_sha256']:
        raise ValueError('Config or inheritance changed after declaration')
    config = load_config(config_path)
    generation, options = validate_run_config(config)
    cache_budget = 2 * 1024**3 if variant == 'compiled-conditioning' else 0
    if (digest(config) != run['resolved_config_digest'] or generation != GenerationConfig()
            or options != replace(OptimizationConfig(), compile_denoiser=True, conditioning_cache_bytes=cache_budget)
            or config['model'].get('backend') != 'vista'
            or config['model'].get('compiler_runtime') != 'registered_einops_attention_v1'
            or config['model'].get('compiler_feature_boundary') is not None
            or config.get('numerical_validation') is not None or config['benchmark'].get('headline') is not False):
        raise ValueError('Resolved config differs from the normal compiled bootstrap')
    for recipe_key, config_key in [('expected_requests','requests'),('candidate_count','candidate_count'),
                                   ('expected_saved_outputs','save_outputs'),('warmup_requests','warmup_requests'),
                                   ('max_warmup_requests','max_warmup_requests'),('input_condition','input_condition')]:
        if run[recipe_key] != config['benchmark'][config_key]:
            raise ValueError('Recipe/config workload differs: ' + recipe_key)
    expected_argv = ['/workspace/outputs/venvs/vista-upstream-cu118-sm90/bin/python','-m','drivebench.benchmark',
                     '--config','/workspace/' + run['config'],'--output',run['output']]
    if run['argv_in_container'] != expected_argv:
        raise ValueError('Recipe does not invoke the normal benchmark')
    output = project_path(root, run['output'])
    if output != project_path(root, config['output']):
        raise ValueError('Recipe/config output differs')
    manifest = project_path(root, config['data']['manifest'])
    metadata = manifest.with_suffix(manifest.suffix + '.meta.json')
    if (file_sha256(manifest) != recipe['expected_manifest_sha256']
            or file_sha256(metadata) != recipe['expected_manifest_metadata_sha256']):
        raise ValueError('Development manifest or metadata changed')
    meta = json.loads(metadata.read_text())
    if meta.get('partition_role') != 'development' or meta.get('synthetic') or sum(bool(line.strip()) for line in manifest.read_text().splitlines()) != 32:
        raise ValueError('Require the real32-request development selection')
    for path in (manifest,metadata):
        bindings[str(path.relative_to(root))] = file_sha256(path)
    env_bindings = recipe['expected_environment_artifact_sha256']
    required_env = {str(project_path(root, config[key]).relative_to(root)) for key in ('environment_verification','environment_lock')}
    if set(env_bindings) != required_env:
        raise ValueError('Environment recipe/config paths differ')
    for relative, sha in env_bindings.items():
        if file_sha256(project_path(root, relative)) != sha:
            raise ValueError('Environment proof or lock changed')
        bindings[relative] = sha
    if json.loads(project_path(root,config['environment_verification']).read_text()).get('passed') is not True:
        raise ValueError('Environment proof is not passing')
    environment = run['compiler_cache_environment']
    cache_root = root / 'outputs/study-bootstrap/compiler-caches' / variant
    if environment != {key:'/workspace/' + str((cache_root / leaf).relative_to(root)) for key,leaf in
                       [('TORCHINDUCTOR_CACHE_DIR','inductor'),('TRITON_CACHE_DIR','triton')]}:
        raise ValueError('Compiler cache paths differ from declared per-run isolation')
    cache_root = project_path(root, str(cache_root.relative_to(root)))
    for path in (output,cache_root):
        if path.exists() or path.is_symlink():
            raise FileExistsError('Output/cache must be fresh: ' + str(path))
    if os.environ.get('DRIVEBENCH_PROFILE') == '1' or os.environ.get('TORCH_COMPILE_DEBUG') == '1':
        raise ValueError('Bootstrap must run without profiling/debug instrumentation')
    minimum = recipe['minimum_free_bytes_before_each_run']
    if type(minimum) is not int or minimum < MIN_FREE_BYTES:
        raise ValueError('Require at least18GiB free before each run')
    free = {}
    for path in (output,cache_root):
        ancestor = path.parent
        while not ancestor.exists():
            ancestor = ancestor.parent
        available = shutil.disk_usage(ancestor).free
        if available < minimum:
            raise ValueError('Insufficient free disk space before bootstrap')
        free[str(ancestor)] = available
    bindings[str(hostspec.relative_to(root))] = FROZEN_RECIPE_SHA256
    helper=Path(__file__).resolve()
    bindings[str(helper.relative_to(root)) if helper.is_relative_to(root) else str(helper)] = file_sha256(__file__)
    return {'root':root,'hostspec':hostspec,'config_path':config_path,'output':output,'cache_root':cache_root,
            'variant':variant,'config':config,'bindings':bindings,'free_bytes':free,
            'environment':{key:str(project_path(root,value)) for key,value in environment.items()},
            'declared_argv':expected_argv}


def ensure_uninitialized():
    if any(name in sys.modules for name in ('torch','torch._inductor','triton','drivebench.benchmark')):
        raise RuntimeError('Use a fresh process: compiler cache environment must precede Torch/benchmark imports')


def execute(plan, *, benchmark_main=None):
    if benchmark_main is None:
        ensure_uninitialized()
    plan['output'].mkdir(parents=True,exist_ok=False)
    plan['cache_root'].mkdir(parents=True,exist_ok=False)
    for path in plan['environment'].values():
        Path(path).mkdir(exist_ok=False)
    receipt = {'schema_version':1,'status':'starting','variant':plan['variant'],
               'started_at':datetime.now(timezone.utc).isoformat(),'source_identity':FROZEN_SOURCE,
               'bound_file_sha256':plan['bindings'],'compiler_cache_environment':plan['environment'],
               'free_bytes_before_run':plan['free_bytes'],'normal_benchmark_argv':plan['declared_argv'],
               'instrumented':False,'headline_eligible':False,'model_patches_by_wrapper':False}
    receipt_path = plan['output'] / 'bootstrap-launch.json'
    def save():
        temporary=receipt_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(receipt,indent=2)+'\n');temporary.replace(receipt_path)
    save()
    previous = {key:os.environ.get(key) for key in plan['environment']}
    argv = sys.argv
    code=1
    try:
        os.environ.update(plan['environment'])
        if benchmark_main is None:
            benchmark_main=importlib.import_module('drivebench.benchmark').main
        sys.argv=['drivebench.benchmark','--config',str(plan['config_path']),'--output',str(plan['output'])]
        code=benchmark_main()
        if type(code) is not int:
            raise RuntimeError('Normal benchmark returned an invalid exit status')
        receipt['status']='completed' if code==0 else 'failed'
    except BaseException as error:
        receipt.update(status='failed',error={'type':type(error).__name__,'message':str(error)})
        code=error.code if isinstance(error,SystemExit) and type(error.code) is int and error.code else 1
        raise
    finally:
        receipt.update(exit_code=code,finished_at=datetime.now(timezone.utc).isoformat())
        try:
            save()
        finally:
            sys.argv=argv
            for key,value in previous.items():
                if value is None:os.environ.pop(key,None)
                else:os.environ[key]=value
    return code


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hostspec',type=Path,required=True)
    parser.add_argument('--variant',choices=VARIANTS,required=True)
    args=parser.parse_args(argv)
    ensure_uninitialized()
    return execute(validate_recipe(args.hostspec,args.variant))


if __name__=='__main__':raise SystemExit(main())
