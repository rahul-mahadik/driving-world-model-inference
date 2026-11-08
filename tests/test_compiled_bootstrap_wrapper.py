import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

from drivebench.config import file_sha256
from scripts import run_compiled_bootstrap as wrapper


@pytest.fixture
def recipe(tmp_path,monkeypatch):
    root=Path(__file__).resolve().parents[1]
    hostspec='configs/study_bootstrap_compiled_hostspec.json'
    spec=json.loads((root/hostspec).read_text())
    assert file_sha256(root/hostspec)==wrapper.FROZEN_RECIPE_SHA256
    paths={hostspec}
    for run in spec['runs']:paths.update(run['inherited_config_sha256'])
    for relative in paths:
        path=tmp_path/relative;path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/relative,path)
    # Contract fixtures only: no licensed records, real GPU identity, or actual
    # environment verification is required or produced by these unit tests.
    manifest=tmp_path/'data/manifests/development.jsonl'
    manifest.parent.mkdir(parents=True,exist_ok=True)
    manifest.write_text(''.join(json.dumps({'unit_test_record':i})+'\n' for i in range(32)))
    metadata=manifest.with_suffix('.jsonl.meta.json')
    metadata.write_text(json.dumps({'partition_role':'development','synthetic':False,'unit_test_fixture':True}))
    spec['expected_manifest_sha256']=file_sha256(manifest)
    spec['expected_manifest_metadata_sha256']=file_sha256(metadata)
    for relative in spec['expected_environment_artifact_sha256']:
        path=tmp_path/relative;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps({'passed':True,'unit_test_fixture':True}))
        spec['expected_environment_artifact_sha256'][relative]=file_sha256(path)
    (tmp_path/hostspec).write_text(json.dumps(spec,sort_keys=True)+'\n')
    monkeypatch.setattr(wrapper,'FROZEN_RECIPE_SHA256',file_sha256(tmp_path/hostspec))
    monkeypatch.setattr(wrapper.shutil,'disk_usage',lambda _:SimpleNamespace(free=20*1024**3))
    monkeypatch.delenv('DRIVEBENCH_PROFILE',raising=False)
    monkeypatch.delenv('TORCH_COMPILE_DEBUG',raising=False)
    return tmp_path,tmp_path/hostspec


@pytest.mark.parametrize('variant',wrapper.VARIANTS)
def test_frozen_recipe_resolves_normal_model_and_separate_fresh_caches(recipe,variant):
    root,hostspec=recipe
    plan=wrapper.validate_recipe(hostspec,variant,root=root)
    assert plan['config']['model']['compiler_runtime']=='registered_einops_attention_v1'
    assert plan['output']==root/'outputs/study-bootstrap'/variant
    assert plan['cache_root']==root/'outputs/study-bootstrap/compiler-caches'/variant
    assert plan['config']['benchmark']['requests']==32
    assert not plan['output'].exists() and not plan['cache_root'].exists()


@pytest.mark.parametrize('mutation',['hostspec','source','config','inheritance','manifest','metadata','environment','disk','output','cache','profile','symlink'])
def test_recipe_fails_closed_before_creation(recipe,monkeypatch,mutation,tmp_path):
    root,hostspec=recipe
    if mutation=='hostspec':hostspec.write_text(hostspec.read_text()+' ')
    if mutation=='source':monkeypatch.setattr(wrapper,'source_identity',lambda:'changed')
    if mutation=='config':(root/'configs/study_bootstrap_compiled_default.yaml').write_text('wrong')
    if mutation=='inheritance':(root/'configs/baseline.yaml').write_text('wrong')
    if mutation=='manifest':(root/'data/manifests/development.jsonl').write_text('wrong')
    if mutation=='metadata':(root/'data/manifests/development.jsonl.meta.json').write_text('wrong')
    if mutation=='environment':(root/'outputs/environment-study-01/reference-lock.json').write_text('wrong')
    if mutation=='disk':monkeypatch.setattr(wrapper.shutil,'disk_usage',lambda _:SimpleNamespace(free=18*1024**3-1))
    if mutation=='output':(root/'outputs/study-bootstrap/compiled-default').mkdir(parents=True)
    if mutation=='cache':(root/'outputs/study-bootstrap/compiler-caches/compiled-default').mkdir(parents=True)
    if mutation=='profile':monkeypatch.setenv('DRIVEBENCH_PROFILE','1')
    if mutation=='symlink':
        target=tmp_path.parent/(tmp_path.name+'-outside');target.mkdir()
        (root/'outputs/study-bootstrap').symlink_to(target,target_is_directory=True)
    with pytest.raises((ValueError,FileExistsError)):
        wrapper.validate_recipe(hostspec,'compiled-default',root=root)


@pytest.mark.parametrize('failure',[False,True])
def test_normal_benchmark_receives_cache_env_and_no_instrumentation_and_restores_process(recipe,monkeypatch,failure):
    root,hostspec=recipe
    plan=wrapper.validate_recipe(hostspec,'compiled-default',root=root)
    monkeypatch.setenv('TORCHINDUCTOR_CACHE_DIR','previous')
    monkeypatch.delenv('TRITON_CACHE_DIR',raising=False)
    old_argv=sys.argv
    def main():
        assert os.environ['TORCHINDUCTOR_CACHE_DIR']==plan['environment']['TORCHINDUCTOR_CACHE_DIR']
        assert os.environ['TRITON_CACHE_DIR']==plan['environment']['TRITON_CACHE_DIR']
        assert sys.argv==['drivebench.benchmark','--config',str(plan['config_path']),'--output',str(plan['output'])]
        assert all(Path(value).is_dir() for value in plan['environment'].values())
        receipt=json.loads((plan['output']/'bootstrap-launch.json').read_text())
        assert not receipt['instrumented'] and not receipt['model_patches_by_wrapper']
        if failure:raise RuntimeError('normal benchmark failed')
        return 0
    if failure:
        with pytest.raises(RuntimeError,match='normal benchmark failed'):wrapper.execute(plan,benchmark_main=main)
    else:assert wrapper.execute(plan,benchmark_main=main)==0
    assert os.environ['TORCHINDUCTOR_CACHE_DIR']=='previous' and 'TRITON_CACHE_DIR' not in os.environ
    assert sys.argv is old_argv
    receipt=json.loads((plan['output']/'bootstrap-launch.json').read_text())
    assert receipt['status']==('failed' if failure else 'completed')
    assert receipt['exit_code']==int(failure)
    assert file_sha256(hostspec)==wrapper.FROZEN_RECIPE_SHA256


@pytest.mark.parametrize('module',['torch','torch._inductor','triton','drivebench.benchmark'])
def test_compiler_import_before_environment_setup_is_rejected(monkeypatch,module):
    monkeypatch.setitem(sys.modules,module,SimpleNamespace())
    with pytest.raises(RuntimeError,match='fresh process'):wrapper.ensure_uninitialized()
