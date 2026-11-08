"""Exercise the actual adapter orchestration with CPU tensors and two RNGs.

These contracts detect temporal mixing and changed random draws. They do not
stand in for the required real H100 floating-point agreement experiment.
"""
from __future__ import annotations

import contextlib
from types import SimpleNamespace

import numpy as np
import pytest

from drivebench.engine import InferenceEngine
from drivebench.models.vista import VistaBackend
from drivebench.runtime.feature_cache import FixedIntervalFeatureCache
from drivebench.types import GenerationConfig, OptimizationConfig


class Tensor(np.ndarray):
    def __new__(cls, value):
        return np.asarray(value).view(cls)

    def float(self):
        return Tensor(self.astype(np.float32))

    def permute(self, *axes):
        return Tensor(self.transpose(axes))

    def contiguous(self):
        return Tensor(np.ascontiguousarray(self))

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)


@pytest.fixture
def backend_factory():
    def make(batch_size=2, cond_aug=0, batch_limit=16):
        backend = VistaBackend.__new__(VistaBackend)
        backend.generation = GenerationConfig(height=64, width=64, cond_aug=cond_aug)
        backend.optimizations = OptimizationConfig(max_batch_size=batch_size)
        backend.is_upstream = False
        backend.max_batch_size = batch_limit
        backend.device = SimpleNamespace(index=0)
        backend.dtype = np.float16
        backend._profile_enabled = False
        backend._active_feature_scope = False
        backend.feature_cache = FixedIntervalFeatureCache()
        backend._samplers = {}
        cpu, cuda = np.random.RandomState(123), np.random.RandomState(456)
        state = SimpleNamespace(cpu=cpu, cuda=cuda, draws=[], denoiser_shapes=[], decode_shapes=[],
                                conditioning_actions=[], depth=0, fail_joint=False, fail_decode=False)

        @contextlib.contextmanager
        def fork_rng(**_kwargs):
            states = cpu.get_state(), cuda.get_state()
            state.depth += 1
            try:
                yield
            finally:
                cpu.set_state(states[0])
                cuda.set_state(states[1])
                state.depth -= 1

        def randn_like(value):
            state.draws.append(("cuda", value.shape))
            return Tensor(cuda.randn(*value.shape).astype(np.float32))

        class Event:
            def record(self):
                pass

            def elapsed_time(self, _end):
                return 1

        backend.torch = SimpleNamespace(
            float32=np.float32, no_grad=contextlib.nullcontext,
            autocast=lambda *args, **kwargs: contextlib.nullcontext(),
            random=SimpleNamespace(fork_rng=fork_rng, default_generator=SimpleNamespace(manual_seed=cpu.seed)),
            cuda=SimpleNamespace(device=lambda *args: contextlib.nullcontext(), manual_seed=cuda.seed,
                                 Event=lambda **kwargs: Event(), synchronize=lambda *args: None,
                                 max_memory_allocated=lambda *args: 0, max_memory_reserved=lambda *args: 0,
                                 empty_cache=lambda: None),
            randn_like=randn_like,
            as_tensor=lambda value, **kwargs: Tensor(value),
            zeros=lambda shape, **kwargs: Tensor(np.zeros(shape, np.float32)),
            zeros_like=lambda value, dtype=None: Tensor(np.zeros_like(value, dtype=dtype)),
            cat=lambda values, dim=0: Tensor(np.concatenate(values, axis=dim)),
            clamp=lambda value, min, max: Tensor(np.clip(value, min, max)),
        )

        def encode(_images):
            parts = []
            for frames in (14, 11):
                state.draws.append(("cpu", (frames, 4, 8, 8)))
                parts.append(cpu.randn(frames, 4, 8, 8).astype(np.float32) * 0.01)
            return Tensor(np.concatenate(parts))

        def decode(value):
            state.decode_shapes.append(value.shape)
            assert value.shape == (25, 4, 8, 8), "Temporal decoder must never see adjacent candidates"
            assert value.dtype == np.float32, "Preserve upstream samples_z copy"
            if state.fail_decode:
                raise MemoryError("decoder fixture OOM")
            return Tensor(np.repeat(np.repeat(value[:, :3], 8, axis=2), 8, axis=3))

        backend.model = SimpleNamespace(conditioner=SimpleNamespace(embedders=[]),
                                        ema_scope=lambda _: contextlib.nullcontext(),
                                        en_and_decode_n_samples_a_time=14,
                                        encode_first_stage=encode, decode_first_stage=decode)

        def condition(_model, value, frames, zero_keys, device):
            action = value["trajectory"].copy()
            state.conditioning_actions.append(action)
            # Keep input action and augmentation visible to detect cross-clip
            # mistakes. Each frame block belongs to exactly one candidate.
            scalar = np.sum(action) * 0.001 + np.mean(value["cond_frames"]) * 0.01
            c = {key: Tensor(np.full((frames, 1), scalar, np.float32))
                 for key in ("crossattn", "vector", "concat")}
            return c, {key: Tensor(np.zeros_like(value)) for key, value in c.items()}

        def denoiser(x, sigma, conditions, mask):
            state.denoiser_shapes.append(x.shape)
            if state.fail_joint and x.shape[0] > 50:
                raise MemoryError("joint fixture OOM")
            # Each 25-frame temporal reduction must stay inside its candidate.
            shaped = x.reshape(-1, 25, 4, 8, 8)
            temporal = np.repeat(shaped.mean(axis=1, keepdims=True), 25, axis=1).reshape(x.shape)
            action = conditions["crossattn"][:, 0, None, None, None]
            return Tensor(x * 0.01 + temporal * 0.01 + action)

        backend._denoiser = denoiser

        class Sampler:
            s_churn = 0

            def __call__(self, denoiser, noise, *, cond, uc, cond_frame, cond_mask):
                count = noise.shape[0]
                assert count % 25 == 0
                expected = np.zeros(count, np.float32)
                expected[::25] = 1
                np.testing.assert_array_equal(cond_mask, expected)
                # Pinned VanillaCFG layout: all UC clips, then all C clips.
                cfg = {key: Tensor(np.concatenate([uc[key], cond[key]])) for key in cond}
                output = denoiser(Tensor(np.concatenate([noise, noise])), Tensor(np.ones(count * 2)),
                                  cfg, Tensor(np.concatenate([cond_mask, cond_mask])))
                uncond, conditioned = np.split(output, 2)
                result = Tensor(uncond + 2.5 * (conditioned - uncond))
                result[cond_mask.astype(bool)] = cond_frame[cond_mask.astype(bool)]
                return result

        state.sampler = Sampler()
        backend.reference = SimpleNamespace(set_lowvram_mode=lambda _: None,
                                            init_embedder_options=lambda _: {}, get_condition=condition,
                                            init_sampling=lambda **kwargs: state.sampler)
        payload = {"images": Tensor(np.zeros((25, 3, 64, 64), np.float32)),
                   "image_embeddings": {}, "posterior_parameters": []}
        backend.prepare_context = lambda _frames: payload
        return backend, payload, state
    return make


@pytest.mark.parametrize("cond_aug", [0, 0.25])
@pytest.mark.parametrize("batch_size", [2, 4, 8, 16])
def test_joint_batch_replays_candidate_rng_and_temporal_layout(backend_factory, cond_aug, batch_size):
    backend, payload, state = backend_factory(cond_aug=cond_aug, batch_size=batch_size)
    actions = [np.full((4, 2), index) for index in range(batch_size)]
    seeds = [42 + index * 49 for index in range(batch_size)]
    cpu_next = np.random.RandomState(123).randn()
    cuda_next = np.random.RandomState(456).randn()
    joint = backend.generate_batch(payload, actions, seeds, backend.generation)
    expected_draws = [("cuda", (1, 3, 64, 64)), ("cpu", (14, 4, 8, 8)),
                      ("cpu", (11, 4, 8, 8)), ("cuda", (25, 4, 8, 8))] * batch_size
    assert state.draws == expected_draws
    assert state.denoiser_shapes == [(batch_size * 50, 4, 8, 8)]
    assert state.decode_shapes == [(25, 4, 8, 8)] * batch_size
    assert joint.denoiser_evaluations == 1
    assert joint.cache_stats["joint_denoising_batches"] == 1
    assert joint.cache_stats["serial_denoising_batches"] == 0
    assert state.cpu.randn() == cpu_next and state.cuda.randn() == cuda_next
    serial = [backend.generate_batch(payload, [action], [seed], backend.generation).frames[0]
              for action, seed in zip(actions, seeds)]
    np.testing.assert_array_equal(joint.frames, np.stack(serial))
    assert state.denoiser_shapes[1:] == [(50, 4, 8, 8)] * batch_size
    assert state.depth == 0


@pytest.mark.parametrize("batch_size", [2, 4, 8, 16])
def test_joint_batch_order_and_other_candidate_action_are_isolated(backend_factory, batch_size):
    backend, payload, _ = backend_factory(batch_size=batch_size)
    actions = [np.full((4, 2), index) for index in range(batch_size)]
    seeds = [101 + index * 101 for index in range(batch_size)]
    original = backend.generate_batch(payload, actions, seeds, backend.generation).frames
    reverse = backend.generate_batch(payload, actions[::-1], seeds[::-1], backend.generation).frames
    np.testing.assert_array_equal(original, reverse[::-1])
    changed = backend.generate_batch(payload, [actions[0] + 1, *actions[1:]], seeds, backend.generation).frames
    np.testing.assert_array_equal(original[1:], changed[1:])
    assert not np.array_equal(original[0], changed[0])


@pytest.mark.parametrize("batch_size", [4, 5, 6, 7, 8, 16])
def test_larger_batches_preserve_remainder_order_and_report_oom_splits(backend_factory, batch_size):
    backend, _, state = backend_factory(batch_size=batch_size)
    engine = InferenceEngine(generation=backend.generation, optimizations=backend.optimizations, backend=backend)
    context = engine.prepare_context(np.zeros((1, 3, 64, 64), np.float32))
    actions = [np.full((4, 2), index) for index in range(batch_size + 1)]
    seeds = list(range(100, 101 + batch_size))
    normal = engine.generate(context, actions, seeds)
    assert normal.effective_config["actual_batch_sizes"] == [batch_size, 1]
    assert normal.denoiser_evaluations == 2
    releases = []
    def release():
        assert state.depth == 0 and not backend._active_feature_scope
        assert backend.model.en_and_decode_n_samples_a_time == 14
        releases.append(True)
    backend.release_after_oom = release
    state.fail_joint = True
    retried = engine.generate(context, actions, seeds)
    assert retried.status == "ok"
    assert retried.effective_config["actual_batch_sizes"] == [1] * (batch_size + 1)
    assert retried.cache_stats["oom_attempts"][0]["batch_size"] == batch_size
    assert len(releases) == len(retried.cache_stats["oom_attempts"])
    assert retried.cache_stats["denoiser_evaluations_complete"] is False
    for key in normal.frames:
        np.testing.assert_array_equal(normal.frames[key], retried.frames[key])
    assert state.depth == 0 and backend.feature_cache.entry_count == 0


def test_experimental_batch_bound_is_not_default_or_hardware_capacity():
    assert VistaBackend.max_batch_size == 16
    assert OptimizationConfig().max_batch_size == 1


@pytest.mark.parametrize("fail", [False, True])
def test_runtime_scope_encloses_gpu_synchronization_and_restores_on_oom(backend_factory, fail):
    backend,payload,state=backend_factory()
    events=[]
    class Runtime:
        active=False
        last_execution=None
        @contextlib.contextmanager
        def activate(self,*,require_graphs):
            assert require_graphs and not self.active
            self.active=True;events.append('enter')
            try:yield
            finally:
                self.active=False;events.append('exit')
                self.last_execution={'restoration_complete':True}
    runtime=Runtime();backend._compiler_runtime=runtime
    original=backend._denoiser
    def denoiser(*args):
        assert runtime.active
        if fail:raise MemoryError('compiled fixture OOM')
        return original(*args)
    backend._denoiser=denoiser
    def synchronize():
        assert runtime.active
        events.append('synchronize')
    backend.synchronize=synchronize
    if fail:
        with pytest.raises(MemoryError):
            backend.generate_batch(payload,[np.zeros((4,2))]*2,[1,2],backend.generation)
        assert events==['enter','exit']
    else:
        result=backend.generate_batch(payload,[np.zeros((4,2))]*2,[1,2],backend.generation)
        assert events==['enter','synchronize','exit']
        assert result.cache_stats['compiler_runtime']=={'restoration_complete':True}
    assert not runtime.active and state.depth==0


def test_real_adapter_oom_splits_deterministically_and_reports_actual_work(backend_factory):
    backend, _, state = backend_factory(batch_size=4, batch_limit=2)
    engine = InferenceEngine(generation=backend.generation, optimizations=backend.optimizations, backend=backend)
    context = engine.prepare_context(np.zeros((1, 3, 64, 64), np.float32))
    actions, seeds = [np.zeros((4, 2))] * 3, [31, 32, 33]
    normal = engine.generate(context, actions, seeds)
    assert normal.effective_config["actual_batch_sizes"] == [2, 1]
    assert normal.denoiser_evaluations == 2
    state.fail_joint = True
    retried = engine.generate(context, actions, seeds)
    assert retried.status == "ok"
    assert retried.effective_config["actual_batch_sizes"] == [1, 1, 1]
    assert retried.cache_stats["oom_attempts"][0]["batch_size"] == 2
    assert retried.cache_stats["denoiser_evaluations_complete"] is False
    for key in normal.frames:
        np.testing.assert_array_equal(normal.frames[key], retried.frames[key])
    assert state.depth == 0 and backend.feature_cache.entry_count == 0


def test_joint_decode_failure_restores_chunk_and_rng_scope(backend_factory):
    from dataclasses import replace
    backend, payload, state = backend_factory()
    state.fail_decode = True
    with pytest.raises(MemoryError, match="decoder fixture"):
        backend.generate_batch(payload, [np.zeros((4, 2))] * 2, [42, 43],
                               replace(backend.generation, decode_chunk_size=6))
    assert backend.model.en_and_decode_n_samples_a_time == 14 and state.depth == 0
    assert state.cpu.randn() == np.random.RandomState(123).randn()
    assert state.cuda.randn() == np.random.RandomState(456).randn()


def test_joint_batch_rejects_stochastic_sampler_or_oversized_call(backend_factory):
    backend, payload, state = backend_factory()
    state.sampler.s_churn = 0.1
    with pytest.raises(ValueError, match="s_churn=0"):
        backend.generate_batch(payload, [np.zeros((4, 2))] * 2, [1, 2], backend.generation)
    with pytest.raises(ValueError, match="configured backend limit"):
        backend.generate_batch(payload, [np.zeros((4, 2))] * 3, [1, 2, 3], backend.generation)
    assert not state.draws


@pytest.mark.parametrize("mutation,match", [("unknown_key", "layout"), ("one_frame", "25 frames"),
                                           ("different_shape", "shapes differ")])
def test_joint_conditioning_fails_closed_on_unexpected_vendor_layout(backend_factory, mutation, match):
    backend, _, _ = backend_factory()
    conditions = [{key: Tensor(np.zeros((25, 2), np.float32)) for key in ("vector", "concat", "crossattn")}
                  for _ in range(2)]
    if mutation == "unknown_key":
        conditions[0]["new_key"] = Tensor(np.zeros(25))
    elif mutation == "one_frame":
        conditions[0]["vector"] = Tensor(np.zeros((1, 2)))
    else:
        conditions[1]["vector"] = Tensor(np.zeros((25, 3)))
    with pytest.raises(ValueError, match=match):
        backend._join_candidate_conditions(conditions, 25)


def test_joint_batch_and_candidate_scoped_feature_cache_are_rejected_before_loading():
    backend = VistaBackend.__new__(VistaBackend)
    backend.generation = GenerationConfig()
    backend.optimizations = OptimizationConfig(max_batch_size=2, feature_cache_interval=2,
                                                feature_cache_module="middle_block")
    backend.is_upstream = False
    with pytest.raises(ValueError, match="scoped to one candidate"):
        backend._validate_options()


def test_validation_retains_effective_work_for_normal_reverse_and_action_change(backend_factory, monkeypatch):
    from drivebench import validate
    from drivebench.benchmarks.agreement import candidate_batching_coverage
    backend, _, _ = backend_factory()
    backend.identity = {"candidate_batch_limit": 2}
    engine = InferenceEngine(generation=backend.generation, optimizations=backend.optimizations, backend=backend)
    monkeypatch.setattr(validate, "InferenceEngine", lambda *args, **kwargs: engine)
    request = {"trajectories": [np.zeros((4, 2)), np.ones((4, 2))],
               "seeds": [51, 52], "candidate_ids": ["a", "b"]}
    outputs, details = validate._run({"model": {}}, backend.generation, backend.optimizations,
                                     request, np.zeros((1, 3, 64, 64), np.float32),
                                     backend="vista", isolation=True)
    assert set(details["executions"]) == {"normal", "reverse", "action_change"}
    assert set(outputs["other_candidates_after_action_change"]) == {"b"}
    for execution in details["executions"].values():
        assert execution["effective_config"]["actual_batch_sizes"] == [2]
        assert execution["cache_stats"]["oom_attempts"] == []
    # The fixture's one-step sampler validates orchestration only.
    assert candidate_batching_coverage(details, steps=1, requested_batch=2,
                                       required_executions=["normal", "reverse", "action_change"])["passed"]


@pytest.mark.parametrize('interval', [2, 3, 4])
def test_selected_output_block_reuse_reports_actual_module_and_nonadditive_policy(backend_factory, monkeypatch, interval):
    from dataclasses import replace
    backend,payload,state=backend_factory(batch_size=1)
    backend.optimizations=replace(backend.optimizations,feature_cache_interval=interval,feature_cache_module='output_blocks.9')
    backend.feature_cache=FixedIntervalFeatureCache(interval)
    monkeypatch.setattr(Tensor,'stride',lambda self:tuple(value//self.itemsize for value in self.strides),raising=False)
    monkeypatch.setattr(Tensor,'device',property(lambda self:'cpu'),raising=False)
    modules=[SimpleNamespace(forward=lambda *args:None) for _ in range(12)]
    modules[9].forward=backend._denoiser
    backend.model.model=SimpleNamespace(diffusion_model=SimpleNamespace(output_blocks=modules))
    backend._install_feature_cache()
    backend._denoiser=modules[9].forward
    original_sampler=state.sampler
    class FiftySteps:
        s_churn=0
        def __call__(self,*args,**kwargs):
            for _ in range(50):result=original_sampler(*args,**kwargs)
            return result
    backend.reference.init_sampling=lambda **kwargs:FiftySteps()
    engine=InferenceEngine(generation=backend.generation,optimizations=backend.optimizations,backend=backend)
    context=engine.prepare_context(np.zeros((1,3,64,64),np.float32))
    result=engine.generate(context,[np.zeros((4,2)),np.ones((4,2))],[11,22])
    assert result.status=='ok'
    refreshes=len(range(0,50,interval))*2
    assert result.cache_stats['feature_misses']==refreshes
    assert result.cache_stats['feature_hits']==100-refreshes
    assert len(state.denoiser_shapes)==refreshes
    assert result.cache_stats['feature_policy']=='approximate_fixed_interval_full_block_output'
    assert result.cache_stats['feature_cache_module']=='output_blocks.9'
    assert result.cache_stats['feature_cache_configuration']=={
        'module':'output_blocks.9','interval':interval,'candidate_scope':'single_candidate',
        'cfg_layout':'unconditional_T_then_conditional_T'}
    assert result.denoiser_evaluations==100 and backend.feature_cache.entry_count==0
