import numpy as np
import pytest
import sys
from types import ModuleType,SimpleNamespace

from scripts import probe_batch_permutation as probe
from scripts.probe_batch_permutation import cfg_permutation,compare_permuted


def test_b2_cfg_permutation_preserves_cfg_halves_and_temporal_blocks():
    permutation=cfg_permutation()
    values=np.arange(100*2).reshape(100,2)
    assert permutation[:25].tolist()==list(range(25,50))
    assert permutation[50:75].tolist()==list(range(75,100))
    np.testing.assert_array_equal(values[permutation][permutation],values)
    result=compare_permuted(values,values[permutation],permutation)
    assert result['passed_exact'] and result['different_values']==0


def test_permutation_comparison_reports_actual_deviation_not_false_reordering_error():
    values=np.zeros((100,2,3),np.float32);permuted=values[cfg_permutation()].copy()
    permuted[25,0,0]=3
    result=compare_permuted(values,permuted,cfg_permutation())
    assert result['max_absolute_error']==3 and result['different_values']==1
    assert result['rmse']==pytest.approx(np.sqrt(9/600))
    with pytest.raises(ValueError,match='layout'):compare_permuted(values,permuted[:50],cfg_permutation())
    permuted[25,0,0]=np.nan
    with pytest.raises(ValueError,match='Nonfinite'):compare_permuted(values,permuted,cfg_permutation())


def test_attention_and_block_wrappers_restore_even_if_cpu_budget_fails(monkeypatch):
    from drivebench.runtime.compiler import NATIVE_SOURCE_SHA256
    class Tensor(np.ndarray):
        def __new__(cls,value):return np.asarray(value).view(cls)
        def numel(self):return self.size
        def element_size(self):return self.itemsize
        def stride(self):return tuple(value//self.itemsize for value in self.strides)
        def detach(self):return self
        def contiguous(self):return self
        def cpu(self):return self
        def numpy(self):return np.asarray(self)
    ops=ModuleType('xformers.ops');root=ModuleType('xformers');root.ops=ops
    def native(q,k,v,**kwargs):return q
    ops.memory_efficient_attention=native
    monkeypatch.setitem(sys.modules,'xformers',root);monkeypatch.setitem(sys.modules,'xformers.ops',ops)
    monkeypatch.setattr(probe,'file_sha256',lambda _:NATIVE_SOURCE_SHA256)
    def forward(x):
        ops.memory_efficient_attention(x,x,x,op=None)
        return Tensor(x+1)
    network=SimpleNamespace(input_blocks=[SimpleNamespace(forward=forward)],
                            middle_block=SimpleNamespace(forward=forward),output_blocks=[SimpleNamespace(forward=forward)])
    backend=SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(diffusion_model=network)))
    value=Tensor(np.arange(100,dtype=np.float32).reshape(100,1,1,1))
    with probe.block_capture(backend) as state:
        network.input_blocks[0].forward(value)
        state['pass']='reversed'
        network.input_blocks[0].forward(value[cfg_permutation()])
        assert state['rows']['reversed'][0]['comparison']['passed_exact']
        assert state['attention']['normal'][0]['block']=='input_blocks.0'
    assert ops.memory_efficient_attention is native
    assert network.input_blocks[0].forward is forward
    with pytest.raises(MemoryError,match='CPU snapshot budget'),probe.block_capture(backend,memory_limit=1):
        network.middle_block.forward(value)
    assert ops.memory_efficient_attention is native
    assert network.middle_block.forward is forward
