import contextlib
from types import SimpleNamespace

import pytest

from drivebench.models.vista import VistaBackend
from scripts.probe_denoiser_bf16_scoped import precision_backend, bound_files


@pytest.mark.parametrize('fail', [False, True])
def test_retained_backend_reverts_to_original_denoiser_after_scope(monkeypatch, fail):
    calls=[];removed=[];originals=[]
    class Module:
        def register_forward_hook(self,hook):return SimpleNamespace(remove=lambda:removed.append(True))
    @contextlib.contextmanager
    def autocast(device,dtype):
        calls.append((device,dtype))
        try:yield
        finally:calls.append('exit')
    def initialize(self,*args,**kwargs):
        self.torch=SimpleNamespace(cuda=SimpleNamespace(is_bf16_supported=lambda:True),bfloat16='bf16',autocast=autocast)
        self.model=SimpleNamespace(model=SimpleNamespace(diffusion_model=SimpleNamespace(input_blocks=[[Module()]],out=[None,None,Module()])))
        self.identity={}
        self._denoiser=lambda value:value
        originals.append(self._denoiser)
    monkeypatch.setattr(VistaBackend,'__init__',initialize)
    state={};instances=[]
    try:
        with precision_backend(state):
            for index in range(2):
                backend=VistaBackend({},None,None);instances.append(backend)
                assert backend._denoiser is not originals[index]
                assert backend._denoiser(index)==index
            if fail:raise MemoryError('fixture OOM')
    except MemoryError:
        assert fail
    before=len(calls)
    for index,backend in enumerate(instances):
        assert backend._denoiser is originals[index]
        assert backend._denoiser(index)==index
    assert len(calls)==before
    assert VistaBackend.__init__ is initialize and len(removed)==4
    assert state['restoration_complete'] and state['retained_backend_restoration_complete']
    assert state['restored_backend_count']==2


def test_successor_binds_immutable_historical_probe():
    from pathlib import Path
    files=bound_files('configs/study_denoiser_bf16_probe.yaml')
    old=str(Path('scripts/probe_denoiser_bf16.py').resolve())
    assert files[old]=='c1225668a8430272d823b1166b63e1c5b9ffc28e188a45c69111cb5749fc2ab4'
    assert str(Path('scripts/probe_denoiser_bf16_scoped.py').resolve()) in files
