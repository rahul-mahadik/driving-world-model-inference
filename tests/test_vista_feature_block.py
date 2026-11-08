"""Full-block reuse contracts, independent of real-GPU quality validation."""
from types import SimpleNamespace

import numpy as np
import pytest

from drivebench.models.vista import VistaBackend
from drivebench.runtime.feature_cache import FixedIntervalFeatureCache
from drivebench.types import GenerationConfig, OptimizationConfig


class Feature(np.ndarray):
    def __new__(cls, value):
        return np.asarray(value).view(cls)
    def stride(self):
        return tuple(x // self.itemsize for x in self.strides)
    @property
    def device(self):
        return 'cpu'
    def detach(self):
        return self
    def clone(self):
        return Feature(self.copy())


def backend_fixture(module, interval=2):
    backend=VistaBackend.__new__(VistaBackend)
    backend.generation=GenerationConfig()
    backend.optimizations=OptimizationConfig(feature_cache_interval=interval,feature_cache_module=module)
    backend.feature_cache=FixedIntervalFeatureCache(interval)
    backend._active_feature_scope=False
    backend._feature_step=0
    calls=[]
    def block(name):
        def forward(x, *, offset=0):
            calls.append((name,backend._feature_step))
            return Feature(x + offset)
        return SimpleNamespace(forward=forward)
    network=SimpleNamespace(middle_block=block('middle_block'),output_blocks=[block(f'output_blocks.{i}') for i in range(12)])
    backend.model=SimpleNamespace(model=SimpleNamespace(diffusion_model=network))
    return backend,network,calls


@pytest.mark.parametrize('name', ['middle_block','model.diffusion_model.middle_block','output_blocks.9','model.diffusion_model.output_blocks.9'])
@pytest.mark.parametrize('interval', [2,3,4])
def test_selected_full_block_keeps_cfg_order_refreshes_and_clears_per_candidate(name,interval):
    backend,network,calls=backend_fixture(name,interval)
    originals=[network.middle_block.forward]+[x.forward for x in network.output_blocks]
    backend._install_feature_cache()
    target=network.middle_block if 'middle_block' in name else network.output_blocks[9]
    selected=0 if 'middle_block' in name else 10
    current=[network.middle_block.forward]+[x.forward for x in network.output_blocks]
    assert [i for i,(a,b) in enumerate(zip(originals,current)) if a is not b]==[selected]
    assert backend._feature_cache_module == ('middle_block' if selected==0 else 'output_blocks.9')
    x=Feature(np.r_[np.full((25,1,1,1),3),np.full((25,1,1,1),17)].astype(np.float32))
    # Outside sampling, the real block still runs and stores no features.
    np.testing.assert_array_equal(target.forward(x,offset=5),x+5)
    assert backend.feature_cache.entry_count==0
    calls.clear();backend._active_feature_scope=True
    with backend.feature_cache.candidate(('first',1)):
        for step in range(50):
            backend._feature_step=step
            out=target.forward(x,offset=step)
            expected=step-(step%interval)
            np.testing.assert_array_equal(out,x+expected)
            out[:] = -999  # Neither returned fresh nor reused arrays alias cached storage.
    assert backend.feature_cache.entry_count==0
    assert len(calls)==len(range(0,50,interval))
    assert backend.feature_cache.hits==50-len(calls)
    with backend.feature_cache.candidate(('second',2)):
        backend._feature_step=1
        np.testing.assert_array_equal(target.forward(x,offset=100),x+100)
    assert backend.feature_cache.entry_count==0


@pytest.mark.parametrize('shape', [(25,1,1,1),(100,1,1,1),(50,1,1)])
def test_output_block_rejects_cross_candidate_or_inner_temporal_layout(shape):
    backend,network,calls=backend_fixture('output_blocks.9')
    backend._install_feature_cache();backend._active_feature_scope=True
    with backend.feature_cache.candidate('a'),pytest.raises(RuntimeError,match='one candidate'):
        network.output_blocks[9].forward(Feature(np.zeros(shape,np.float32)))
    assert not calls and backend.feature_cache.entry_count==0


@pytest.mark.parametrize('name', ['output_blocks.8','output_blocks.10','output_blocks.9.0','input_blocks.2','anything'])
def test_unaudited_targets_rejected_before_model_loading(name):
    backend=VistaBackend.__new__(VistaBackend)
    backend.generation=GenerationConfig()
    backend.optimizations=OptimizationConfig(feature_cache_interval=2,feature_cache_module=name)
    backend.is_upstream=False
    with pytest.raises(ValueError,match='Only middle_block and output_blocks.9'):
        backend._validate_options()


def test_output_target_requires_pinned_topology_and_failure_does_not_cache():
    backend,network,_=backend_fixture('output_blocks.9')
    network.output_blocks.pop()
    with pytest.raises(ValueError,match='twelve-output-block'):
        backend._install_feature_cache()
    backend,network,_=backend_fixture('output_blocks.9')
    def fail(x):raise MemoryError('simulated OOM')
    network.output_blocks[9].forward=fail
    backend._install_feature_cache();backend._active_feature_scope=True
    with pytest.raises(MemoryError),backend.feature_cache.candidate('a'):
        network.output_blocks[9].forward(Feature(np.zeros((50,1,1,1),np.float32)))
    assert backend.feature_cache.entry_count==0


@pytest.mark.parametrize('interval', [1,2,3,4])
def test_compiled_boundary_is_stable_scoped_and_hides_cache_state(interval):
    backend,network,calls=backend_fixture('output_blocks.9',interval)
    backend._compiler_feature_boundary='output_blocks.9'
    backend.identity={}
    originals=[]
    def disable(fn):
        originals.append(fn)
        def wrapper(*args,**kwargs):return fn(*args,**kwargs)
        wrapper._torchdynamo_orig_callable=fn
        return wrapper
    backend.torch=SimpleNamespace(_dynamo=SimpleNamespace(disable=disable))
    target=network.output_blocks[9]
    original=target.forward
    backend._install_feature_cache()
    assert target.forward is original and len(originals)==1
    stable_wrapper=backend._compiled_feature_forward[2]
    backend._active_feature_scope=True
    x=Feature(np.zeros((50,1,1,1),np.float32))
    for candidate in ('a','b'):
        with backend._feature_forward_scope(),backend.feature_cache.candidate(candidate):
            assert target.forward is stable_wrapper
            for step in range(50):
                backend._feature_step=step
                expected=step if interval==1 else step-step%interval
                np.testing.assert_array_equal(target.forward(x,offset=step),x+expected)
        assert target.forward is original and backend.feature_cache.entry_count==0
    assert len(originals)==1
    assert len(calls)==2*len(range(0,50,interval))
    assert backend.identity['compiler_feature_boundary']['original_forward_restored_after_scope']
    if interval==1:
        assert backend.feature_cache.hits==backend.feature_cache.misses==0
    with pytest.raises(MemoryError),backend._feature_forward_scope():
        assert target.forward is stable_wrapper
        raise MemoryError('simulated OOM')
    assert target.forward is original
    target.forward=lambda *args:None
    with pytest.raises(RuntimeError,match='changed outside'):
        with backend._feature_forward_scope():pass


def test_compiled_scope_accepts_python_bound_method_before_first_install():
    backend=VistaBackend.__new__(VistaBackend)
    class Block:
        def forward(self,x):return x+1
    module=Block();original=module.forward
    wrapper=lambda x:x+2
    backend._compiled_feature_forward=(module,original,wrapper)
    assert module.forward is not original  # Descriptor returns a new method object.
    with backend._feature_forward_scope():
        assert module.forward is wrapper and module.forward(3)==5
    assert module.forward is original and module.forward(3)==4
