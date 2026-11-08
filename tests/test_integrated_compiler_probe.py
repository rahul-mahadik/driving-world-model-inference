import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from drivebench.config import file_sha256
from scripts import probe_integrated_compiler as probe


@pytest.mark.parametrize('name', ['compiler','feature1','feature2','feature3','feature4','shape_static','shape_dynamic'])
def test_integrated_probe_configs_are_explicit_and_bounded(name):
    config,generation,options=probe.inspect_config(Path(f'configs/study_integrated_{name}.yaml'))
    assert generation.steps==50 and generation.total_frames==25
    assert options.compile_denoiser and not options.fullgraph
    assert config['model']['compiler_runtime']=='registered_einops_attention_v1'
    assert config['probe']['strict_atol']==config['probe']['strict_rtol']==0
    assert options.dynamic_shapes==(name=='shape_dynamic')


@pytest.mark.parametrize('mutation', ['runtime','boundary','reference','steps','batch','tolerance','interval','kind'])
def test_integrated_config_rejects_unregistered_or_mixed_changes(tmp_path,mutation):
    config,_,_=probe.inspect_config(Path('configs/study_integrated_feature2.yaml'))
    if mutation=='runtime':config['model']['compiler_runtime']=None
    if mutation=='boundary':config['model']['compiler_feature_boundary']='middle_block'
    if mutation=='reference':config['probe']['comparison_probe']=None
    if mutation=='steps':config['generation']['steps']=47
    if mutation=='batch':config['optimizations']['max_batch_size']=2
    if mutation=='tolerance':config['probe']['strict_atol']=0.01
    if mutation=='interval':config['optimizations']['feature_cache_interval']=5
    if mutation=='kind':config['probe']['kind']='shape_dynamic'
    path=tmp_path/'bad.yaml';path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):probe.inspect_config(path)


def test_wrapper_leaves_historical_helper_unchanged_and_restores_on_error(monkeypatch,tmp_path):
    helper=Path(probe.base.__file__);before=file_sha256(helper)
    originals=probe.base.inspect_config,probe.base.attention_boundary,probe.base.load_reference,probe.base.save
    def fail(argv):
        assert probe.base.inspect_config is probe.inspect_config
        with probe.base.attention_boundary(True) as patch:
            assert patch['global_patches_by_wrapper'] is False
        raise RuntimeError('fixture')
    monkeypatch.setattr(probe.base,'main',fail)
    with pytest.raises(RuntimeError,match='fixture'):
        probe.main(['--config','configs/study_integrated_compiler.yaml','--output',str(tmp_path/'new')])
    assert originals==(probe.base.inspect_config,probe.base.attention_boundary,probe.base.load_reference,probe.base.save)
    assert file_sha256(helper)==before and not (tmp_path/'new').exists()


@pytest.mark.parametrize('mutation', [None,'core','unrestored','fallback','output','bindings','reference'])
def test_compiled_reference_requires_completed_same_source_same_inputs_and_bound_output(tmp_path,monkeypatch,mutation):
    from drivebench.benchmarks import environment
    monkeypatch.setattr(environment,'source_identity',lambda:'frozen-core')
    baseline,_,_=probe.inspect_config(Path('configs/study_integrated_compiler.yaml'))
    selected,_,_=probe.inspect_config(Path('configs/study_integrated_feature1.yaml'))
    array=np.zeros((2,2,3),np.float32);output=tmp_path/'measured-0.npy';np.save(output,array)
    source=tmp_path/'helper.py';source.write_text('frozen')
    upstream_proof={'model_identity':{'checkpoint_sha256':'a','upstream_commit':'b','inference_config_sha256':'c'},
                    'record_sha256':'record','runtime_identity':{'gpu':'H100'}}
    row={'label':'measured-0','status':'ok','fallbacks':[],
         'cache_stats':{'oom_attempts':[],'compiler_runtime':{'passed':True,'restoration_complete':True}},
         'effective_config':{'actual_batch_sizes':[1]},'denoiser_evaluations':50,'compiler':{},
         'output_path':str(output),'output_sha256':file_sha256(output)}
    report={'config':baseline,'status':'execution_complete_strict_failed','exit_code':2,
            'positive_cuda_codegen_evidence':True,'source_unchanged':True,'launcher_and_config_unchanged':True,
            'integrated_runtime':{'bound_files_unchanged':True,'bound_file_sha256':{str(source):file_sha256(source)}},
            'implementation_sha256':'frozen-core','reference':upstream_proof,'model_identity':upstream_proof['model_identity'],
            'executions':[row],'strict_passed':False}
    if mutation=='core':report['implementation_sha256']='old'
    if mutation=='unrestored':row['cache_stats']['compiler_runtime']['restoration_complete']=False
    if mutation=='fallback':row['fallbacks']=['oom']
    if mutation=='output':row['output_sha256']='wrong'
    if mutation=='bindings':source.write_text('changed')
    if mutation=='reference':report['reference']={}
    (tmp_path/'probe.json').write_text(json.dumps(report))
    if mutation:
        with pytest.raises(ValueError):probe.compiled_reference(tmp_path,selected,array,upstream_proof)
    else:
        result,proof=probe.compiled_reference(tmp_path,selected,array,upstream_proof)
        np.testing.assert_array_equal(result,array)
        assert proof['upstream_comparison'] is False
        assert proof['compiled_probe_report_sha256']==file_sha256(tmp_path/'probe.json')
