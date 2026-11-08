"""CPU-only contracts for the isolated registration experiment."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from drivebench.config import file_sha256, load_config
from scripts import probe_registered_einops as registered
from scripts import probe_vista_optimization as base


@pytest.mark.parametrize('name', ['default', 'dynamic', 'reduce_overhead'])
def test_configs_keep_original_probe_contract(name):
    path=Path(f'configs/study_registered_einops_{name}.yaml')
    config,generation,options=registered.inspect_config(path)
    assert generation.steps==50 and generation.total_frames==25
    assert options.compile_denoiser and not options.compile_decoder
    assert options.dynamic_shapes is (name=='dynamic')
    assert options.compile_mode==('reduce-overhead' if name=='reduce_overhead' else 'default')
    assert config['probe']['strict_atol']==config['probe']['strict_rtol']==0
    assert config['benchmark']['headline'] is False
    files=registered.bound_files(path)
    assert str(Path('configs/baseline.yaml').resolve()) in files
    assert files[str(Path('scripts/probe_vista_optimization.py').resolve())]==registered.HELPER_SHA256['scripts/probe_vista_optimization.py']


@pytest.mark.parametrize('key,value', [('cache_size_limit',1024),('suppress_errors',True),('native_attention_boundary',False),('einops_version','0.8.0')])
def test_runtime_declaration_cannot_drift(monkeypatch,key,value):
    config,generation,options=base.inspect_config('configs/study_registered_einops_default.yaml')
    config['compiler_runtime'][key]=value
    monkeypatch.setattr(base,'inspect_config',lambda _: (config,generation,options))
    with pytest.raises(ValueError,match='declaration'):
        registered.inspect_config('unused')


class Registry(set):
    def remove(self, value):
        self.discard(value)


@pytest.fixture
def runtime(tmp_path,monkeypatch):
    functions={name:(lambda x:x) for name in registered.FUNCTIONS}
    allowed,disallowed=Registry(),Registry()
    dynamo=SimpleNamespace(config=SimpleNamespace(disable=False,suppress_errors=False,cache_size_limit=64),
        allowed_functions=SimpleNamespace(_allowed_function_ids=allowed,_disallowed_function_ids=disallowed))
    path=tmp_path/'official.py';path.write_text('official source fixture')
    calls=[]
    def registration():
        calls.append('official_api')
        allowed.update(map(id,functions.values()))
    source={str(path):file_sha256(path)}
    monkeypatch.setattr(registered,'load_runtime',lambda:(dynamo,registration,functions,source))
    return dynamo,allowed,disallowed,functions,source,calls


@pytest.mark.parametrize('raise_inside',[False,True])
def test_registration_and_cache_limit_restore_after_success_or_exception(runtime,raise_inside):
    dynamo,allowed,disallowed,functions,files,calls=runtime
    state={}
    with pytest.raises(RuntimeError) if raise_inside else __import__('contextlib').nullcontext():
        with registered.registered_runtime(state):
            assert dynamo.config.cache_size_limit==512
            assert all(id(fn) in allowed for fn in functions.values())
            if raise_inside: raise RuntimeError('GPU/probe failure fixture')
    assert dynamo.config.cache_size_limit==64 and not allowed and not disallowed
    assert state['restoration_complete'] and state['einops_source_unchanged']
    assert calls==['official_api']


@pytest.mark.parametrize('change',['already_allowed','disallowed','wrong_limit','suppressed','disabled'])
def test_preexisting_runtime_overrides_rejected(runtime,change):
    dynamo,allowed,disallowed,functions,_,calls=runtime
    if change=='already_allowed': allowed.add(id(functions['rearrange']))
    elif change=='disallowed': disallowed.add(id(functions['reduce']))
    elif change=='wrong_limit': dynamo.config.cache_size_limit=1024
    elif change=='suppressed': dynamo.config.suppress_errors=True
    else: dynamo.config.disable=True
    with pytest.raises(ValueError):
        with registered.registered_runtime({}): pytest.fail('must fail before executing')
    assert calls==[]


def test_partial_registration_failure_restores_every_target(runtime,monkeypatch):
    dynamo,allowed,disallowed,functions,files,_=runtime
    def broken():
        allowed.add(id(functions['rearrange']))
        raise RuntimeError('registration failed midway')
    monkeypatch.setattr(registered,'load_runtime',lambda:(dynamo,broken,functions,files))
    state={}
    with pytest.raises(RuntimeError):
        with registered.registered_runtime(state):pass
    assert not allowed and state['restoration_complete']


def test_successful_api_must_actually_register_all_functions(runtime,monkeypatch):
    dynamo,allowed,disallowed,functions,files,_=runtime
    monkeypatch.setattr(registered,'load_runtime',lambda:(dynamo,lambda:None,functions,files))
    with pytest.raises(RuntimeError,match='did not take effect'):
        with registered.registered_runtime({}):pass
    assert not allowed and dynamo.config.cache_size_limit==64


@pytest.mark.parametrize('code',[0,1,2,143])
def test_wrapper_binds_all_changes_and_preserves_base_failure_codes(tmp_path,monkeypatch,runtime,code):
    dynamo,allowed,_,_,_,_=runtime
    original_save=base.save
    @contextmanager
    def native(enabled):
        assert enabled
        yield {'kind':'native_boundary','source_sha256':'pinned'}
    monkeypatch.setattr(base,'attention_boundary',native)
    def main(argv):
        output=Path(argv[argv.index('--output')+1]);output.mkdir()
        assert dynamo.config.cache_size_limit==512
        with base.attention_boundary(True) as patch:
            assert patch['native_attention_boundary']['kind']=='native_boundary'
            assert patch['registration']['cache_size_limit']==512
            report={'status':'execution_complete' if code==0 else 'failed', 'exit_code':code,
                    'runtime_patch':patch,'strict_passed':code==0,'headline_eligible':False,
                    'compiler_settings':{'cache_size_limit':dynamo.config.cache_size_limit}}
            base.save(output/'probe.json',report)
        return code
    monkeypatch.setattr(base,'main',main)
    output=tmp_path/'attempt'
    assert registered.main(['--config','configs/study_registered_einops_default.yaml','--output',str(output)])==code
    report=json.loads((output/'probe.json').read_text())
    evidence=json.loads((output/'registered-runtime.json').read_text())
    assert report['exit_code']==code and report['compiler_settings']['cache_size_limit']==512
    assert report['registered_compiler_wrapper']['restoration_complete'] is True
    assert evidence['probe_report_sha256']==file_sha256(output/'probe.json')
    assert base.save is original_save and base.attention_boundary is native
    assert not allowed and dynamo.config.cache_size_limit==64
    with pytest.raises(FileExistsError):
        registered.main(['--config','configs/study_registered_einops_default.yaml','--output',str(output)])


def test_setup_failure_is_preserved_without_model_call(tmp_path,monkeypatch):
    monkeypatch.setattr(registered,'load_runtime',lambda:(_ for _ in ()).throw(ValueError('source mismatch')))
    monkeypatch.setattr(base,'main',lambda *a:pytest.fail('must not call model helper'))
    output=tmp_path/'failed'
    assert registered.main(['--config','configs/study_registered_einops_default.yaml','--output',str(output)])==1
    report=json.loads((output/'probe.json').read_text())
    assert report['status']=='failed_before_base_probe'
    assert report['registered_compiler_wrapper']['error']['message']=='source mismatch'


def test_runtime_source_drift_cannot_exit_success(tmp_path,monkeypatch,runtime):
    _,_,_,_,files,_=runtime
    def main(argv):
        output=Path(argv[argv.index('--output')+1]);output.mkdir()
        base.save(output/'probe.json',{'status':'execution_complete','exit_code':0})
        Path(next(iter(files))).write_text('changed while running')
        return 0
    monkeypatch.setattr(base,'main',main)
    output=tmp_path/'changed'
    assert registered.main(['--config','configs/study_registered_einops_default.yaml','--output',str(output)])==1
    report=json.loads((output/'probe.json').read_text())
    assert report['status']=='registered_wrapper_failed'
    assert report['registered_compiler_wrapper']['einops_source_unchanged'] is False


def test_output_race_never_overwrites_other_attempt(tmp_path,monkeypatch,runtime):
    output=tmp_path/'raced'
    def main(argv):
        output.mkdir()
        (output/'probe.json').write_text('other attempt evidence')
        raise FileExistsError('another process created output')
    monkeypatch.setattr(base,'main',main)
    with pytest.raises(FileExistsError):
        registered.main(['--config','configs/study_registered_einops_default.yaml','--output',str(output)])
    assert (output/'probe.json').read_text()=='other attempt evidence'
