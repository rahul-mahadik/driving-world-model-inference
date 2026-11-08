import numpy as np
import pytest

from drivebench.engine import InferenceEngine
from drivebench.models.synthetic import SyntheticBackend
from drivebench.runtime.cache import ContextCache
from drivebench.runtime.prefetch import prefetch
from drivebench.types import GenerationConfig, OptimizationConfig


def make_engine(batch=1, cache=0, checkpoint="a", backend=None):
    return InferenceEngine({"backend": "synthetic", "checkpoint": checkpoint},
                           GenerationConfig(height=16, width=24, steps=3, total_frames=5, dtype="fp32"),
                           OptimizationConfig(max_batch_size=batch, conditioning_cache_bytes=cache), backend=backend)


def test_candidate_streams_independent_of_order_and_batching():
    frames = np.zeros((1, 3, 16, 24), np.float32)
    actions = [np.arange(8), np.zeros(8), -np.arange(8)]
    outputs = []
    for batch in (1, 2, 3):
        engine = make_engine(batch)
        context = engine.prepare_context(frames)
        result = engine.generate(context, actions, [10, 20, 30], candidate_ids=["a", "b", "c"])
        reverse = engine.generate(context, actions[::-1], [30, 20, 10], candidate_ids=["c", "b", "a"])
        for key in result.frames:
            np.testing.assert_array_equal(result.frames[key], reverse.frames[key])
        outputs.append(result.frames)
    for key in outputs[0]:
        np.testing.assert_array_equal(outputs[0][key], outputs[-1][key])
    assert not np.array_equal(outputs[0]["a"], outputs[0]["b"])


def test_content_cache_invalidation_and_context_ownership():
    engine = make_engine(cache=100000)
    frames = np.zeros((1, 3, 16, 24), np.float32)
    a = engine.prepare_context(frames)
    assert not a.cache_hit
    assert engine.prepare_context(frames.copy()).cache_hit
    frames[0, 0, 0, 0] = .5
    b = engine.prepare_context(frames)
    assert not b.cache_hit and a.cache_key != b.cache_key
    other = make_engine(checkpoint="b")
    with pytest.raises(ValueError, match="different model"):
        other.generate(a, [np.zeros(8)], [1])


def test_cache_byte_budget_and_oversize():
    cache = ContextCache(32)
    cache.put("a", np.zeros(4, dtype=np.float32))
    cache.put("b", np.zeros(4, dtype=np.float32))
    assert cache.get("a") is not None
    cache.put("c", np.zeros(4, dtype=np.float32))
    assert cache.get("b") is None
    assert cache.bytes == 32
    cache.put("too-large", np.zeros(100, dtype=np.float32))
    assert cache.bytes == 32


def test_oom_batches_retry_without_changing_streams():
    template = make_engine(batch=4)
    class LimitedBackend(SyntheticBackend):
        def generate_batch(self, payload, actions, seeds, config):
            if len(seeds) > 1:
                raise MemoryError("test batch limit")
            return super().generate_batch(payload, actions, seeds, config)
    backend = LimitedBackend({}, template.generation, template.optimizations)
    engine = make_engine(batch=4, backend=backend)
    context = engine.prepare_context(np.zeros((1, 3, 16, 24), np.float32))
    result = engine.generate(context, [np.zeros(8)] * 4, [1, 2, 3, 4])
    assert result.status == "ok"
    assert result.effective_config["actual_batch_sizes"] == [1] * 4
    assert any("out of memory" in x for x in result.fallbacks)
    assert len(result.frames) == 4


def test_prefetch_propagates_errors_and_keeps_order():
    assert list(prefetch(range(8), lambda x: x*x, 2)) == [x*x for x in range(8)]
    def load(x):
        if x == 3:
            raise ValueError("bad image")
        return x
    with pytest.raises(ValueError, match="bad image"):
        list(prefetch(range(8), load, 2))


def test_invalid_candidates_fail_before_model():
    engine = make_engine()
    context = engine.prepare_context(np.zeros((1, 3, 16, 24), np.float32))
    with pytest.raises(ValueError, match="one seed"):
        engine.generate(context, [np.zeros(8)], [])
    with pytest.raises(ValueError, match="eight finite"):
        engine.generate(context, [np.array([np.nan] * 8)], [1])
    with pytest.raises(ValueError, match="unique"):
        engine.generate(context, [np.zeros(8)]*2, [1,2], candidate_ids=["a","a"])
