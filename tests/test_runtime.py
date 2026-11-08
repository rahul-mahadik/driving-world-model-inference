import weakref

import numpy as np
import pytest

from drivebench.engine import InferenceEngine
from drivebench.models.synthetic import SyntheticBackend
from drivebench.runtime.cache import ContextCache, size_bytes
from drivebench.types import GenerationConfig, OptimizationConfig


def test_views_charge_complete_storage_once():
    allocation = np.zeros(1000, dtype=np.float32)
    views = {"a": allocation[:1], "b": allocation[-1:]}
    assert size_bytes(views) == allocation.nbytes
    cache = ContextCache(100)
    cache.put("context", views)
    assert cache.bytes == 0 and not cache.items


def test_torch_storage_protocol_deduplicates_aliases():
    class Storage:
        def data_ptr(self): return 100
        def nbytes(self): return 4096
    class Tensor:
        device = "cuda:0"
        def untyped_storage(self): return Storage()
    assert size_bytes([Tensor(), Tensor()]) == 4096


def test_oversize_replacement_does_not_serve_stale_context():
    cache = ContextCache(32)
    cache.put("a", np.ones(4))
    cache.put("a", np.ones(100))
    assert cache.get("a") is None and cache.bytes == 0


@pytest.mark.parametrize("settings", [
    {"compile_denoiser": "false"}, {"max_batch_size": 1.5}, {"prefetch_depth": True},
    {"conditioning_cache_bytes": -1}, {"feature_cache_module": ""},
])
def test_ambiguous_optimization_types_rejected(settings):
    with pytest.raises(ValueError):
        OptimizationConfig(**settings)


@pytest.mark.parametrize("value", [True, "2.5", float("nan")])
def test_nonfinite_or_ambiguous_generation_scalars_rejected(value):
    with pytest.raises(ValueError):
        GenerationConfig(guidance_scale=value)


def test_oom_traceback_released_before_backend_cleanup():
    generation = GenerationConfig(height=16, width=16, total_frames=5, steps=2, dtype="fp32")
    options = OptimizationConfig(max_batch_size=2)
    class Backend(SyntheticBackend):
        def generate_batch(self, payload, trajectories, seeds, config):
            if len(seeds) > 1:
                temporary = np.ones(256)
                self.reference = weakref.ref(temporary)
                raise MemoryError("simulated device allocation")
            return super().generate_batch(payload, trajectories, seeds, config)
        def release_after_oom(self):
            assert self.reference() is None, "OOM traceback still retains temporary allocation"
            self.released = True
    backend = Backend({}, generation, options)
    engine = InferenceEngine({"backend": "synthetic"}, generation, options, backend)
    context = engine.prepare_context(np.zeros((1, 3, 16, 16), np.float32))
    result = engine.generate(context, [np.zeros(8)] * 2, [1, 2])
    assert result.status == "ok" and backend.released
