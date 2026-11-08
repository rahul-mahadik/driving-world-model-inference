import contextlib
from types import SimpleNamespace

import pytest

from scripts.probe_denoiser_bf16 import precision_backend
from drivebench.models.vista import VistaBackend


def test_precision_scope_preserves_outer_dtype_and_cleans_hooks(monkeypatch):
    events, removed, hooks = [], [], []

    class Module:
        def register_forward_hook(self, hook):
            hooks.append(hook)
            return SimpleNamespace(remove=lambda: removed.append(True))

    @contextlib.contextmanager
    def autocast(device, dtype):
        events.append((device, dtype))
        try:
            yield
        finally:
            events.append("restored")

    def original(self, *_args, **_kwargs):
        self.torch = SimpleNamespace(cuda=SimpleNamespace(is_bf16_supported=lambda: True),
                                     bfloat16="bf16", autocast=autocast)
        self.model = SimpleNamespace(model=SimpleNamespace(diffusion_model=SimpleNamespace(
            input_blocks=[[Module()]], out=[None, None, Module()])))
        self.identity = {}
        self._denoiser = lambda x: x

    monkeypatch.setattr(VistaBackend, "__init__", original)
    state = {}
    with pytest.raises(MemoryError), precision_backend(state):
        backend = VistaBackend({}, None, None)
        assert not events
        assert backend._denoiser("sentinel") == "sentinel"
        assert events == [("cuda", "bf16"), "restored"]
        for hook in hooks:
            hook(None, (), SimpleNamespace(dtype="torch.bfloat16"))
        raise MemoryError("simulated allocation failure")
    assert VistaBackend.__init__ is original
    assert removed == [True, True]
    assert state["restoration_complete"] is True
    assert all(v == {"torch.bfloat16": 1} for v in state["dtype_observations"].values())


@pytest.mark.parametrize("model,kwargs", [({"compiler_runtime": "enabled"}, {}), ({}, {"_upstream": True})])
def test_precision_probe_refuses_combined_or_upstream(monkeypatch, model, kwargs):
    def original(*args, **kw):
        raise AssertionError("must reject before model loading")
    monkeypatch.setattr(VistaBackend, "__init__", original)
    with precision_backend({}), pytest.raises(ValueError):
        VistaBackend(model, None, None, **kwargs)
    assert VistaBackend.__init__ is original
