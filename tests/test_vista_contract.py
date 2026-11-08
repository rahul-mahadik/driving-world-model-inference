"""CPU contract checks. These do not establish Vista GPU output agreement."""
from __future__ import annotations

import contextlib
import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from drivebench.models.upstream import observed_only_padding, verify_checkpoint
from drivebench.models.vista import VistaBackend
from drivebench.runtime.feature_cache import FixedIntervalFeatureCache
from drivebench.types import GenerationConfig, OptimizationConfig


def test_model_modules_import_without_torch():
    code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise AssertionError('torch was imported at module import time')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import drivebench.models.vista
import drivebench.models.upstream
import drivebench.runtime.feature_cache
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


@pytest.fixture(params=[False, True], ids=["refactored", "upstream"])
def grad_contract_backend(request, monkeypatch, tmp_path):
    """Exercise the real adapter with CPU stand-ins, never importing Torch."""
    from drivebench.models import vista

    state = SimpleNamespace(no_grad_depth=0, no_grad_entries=0, sampling_entries=0)
    parameters = [
        SimpleNamespace(dtype="torch.float32", requires_grad=True,
                        numel=lambda: 6, element_size=lambda: 4),
        SimpleNamespace(dtype="torch.float32", requires_grad=False,
                        numel=lambda: 10, element_size=lambda: 4),
        SimpleNamespace(dtype="torch.float16", requires_grad=False,
                        numel=lambda: 3, element_size=lambda: 2),
    ]

    @contextlib.contextmanager
    def no_grad():
        state.no_grad_entries += 1
        state.no_grad_depth += 1
        try:
            yield
        finally:
            state.no_grad_depth -= 1

    def inference_mode(*args, **kwargs):
        raise AssertionError("The upstream contract uses no_grad, not inference_mode")

    @contextlib.contextmanager
    def ema_scope(_name):
        assert state.no_grad_depth > 0
        yield

    class Model:
        conditioner = SimpleNamespace(embedders=[])

        def eval(self):
            return self

        def requires_grad_(self, _enabled):
            raise AssertionError("Do not replace the loader's selective parameter flags")

        def parameters(self):
            return iter(parameters)

        def state_dict(self):
            return {"weight": None}

    model = Model()
    model.ema_scope = ema_scope

    class Tensor:
        def __init__(self, value):
            assert state.no_grad_depth > 0
            self.value = value.copy()

        def to(self, _device):
            assert state.no_grad_depth > 0
            return self

    class Event:
        def record(self):
            assert state.no_grad_depth > 0

    def seed(_value):
        assert state.no_grad_depth > 0

    fake_torch = SimpleNamespace(
        device=lambda *args: SimpleNamespace(index=0),
        float16="torch.float16", bfloat16="torch.bfloat16", float32="torch.float32",
        is_autocast_cache_enabled=lambda: True,
        no_grad=no_grad, inference_mode=inference_mode, from_numpy=Tensor,
        autocast=lambda *args, **kwargs: contextlib.nullcontext(),
        cuda=SimpleNamespace(
            is_available=lambda: True, current_device=lambda: 0,
            device=lambda *args: contextlib.nullcontext(), synchronize=lambda *args: None,
            manual_seed=seed, Event=lambda **kwargs: Event(),
        ),
        random=SimpleNamespace(fork_rng=lambda **kwargs: contextlib.nullcontext(),
                               default_generator=SimpleNamespace(manual_seed=seed)),
    )
    reference = SimpleNamespace(set_lowvram_mode=lambda enabled: None, init_model=lambda config: model)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "safetensors", SimpleNamespace(
        safe_open=lambda *args, **kwargs: contextlib.nullcontext({"weight": None})))
    monkeypatch.setattr(vista, "verify_checkout", lambda *args: {"upstream_commit": "fixture"})
    monkeypatch.setattr(vista, "verify_checkpoint", lambda path, sha: {"checkpoint": path})
    monkeypatch.setattr(vista, "sha256_file", lambda path: "a" * 64)
    monkeypatch.setattr(vista, "import_reference", lambda root: reference)
    backend = VistaBackend(
        {"upstream_dir": str(tmp_path), "checkpoint": str(tmp_path / "fixture.safetensors")},
        GenerationConfig(height=64, width=64), OptimizationConfig(), _upstream=request.param,
    )
    backend._profile_enabled = False
    return backend, state, parameters


def test_constructor_preserves_loader_parameter_flags_and_records_inventory(grad_contract_backend):
    backend, _, parameters = grad_contract_backend
    assert [parameter.requires_grad for parameter in parameters] == [True, False, False]
    assert backend.identity["parameter_dtypes"] == {
        "torch.float32": {"tensors": 2, "elements": 16, "bytes": 64,
                          "requires_grad_tensors": 1, "requires_grad_elements": 6},
        "torch.float16": {"tensors": 1, "elements": 3, "bytes": 6,
                          "requires_grad_tensors": 0, "requires_grad_elements": 0},
    }
    assert backend.identity["parameter_grad_policy"] == "preserve_upstream_loader_flags"
    assert backend.identity["execution_grad_mode"] == "no_grad"
    assert backend.identity["autocast_cache_enabled"] is True


def test_prepare_context_uses_no_grad_and_restores_context(grad_contract_backend):
    backend, state, _ = grad_contract_backend
    frames = np.zeros((1, 3, 64, 64), np.float32)
    payload = backend.prepare_context(frames)
    assert payload["images"].value.shape == (25, 3, 64, 64)
    assert payload["observed_count"] == 1
    assert state.no_grad_entries == 1 and state.no_grad_depth == 0


def test_candidate_enters_no_grad_and_restores_it_on_failure(grad_contract_backend):
    backend, state, _ = grad_contract_backend

    class ReachedSampling(Exception):
        pass

    def stop_at_sampling(_keys):
        assert state.no_grad_depth > 0
        state.sampling_entries += 1
        raise ReachedSampling

    backend.reference.init_embedder_options = stop_at_sampling
    with pytest.raises(ReachedSampling):
        backend.generate_batch({"image_embeddings": {}}, [np.zeros((4, 2))], [42], backend.generation)
    assert state.sampling_entries == 1
    assert state.no_grad_entries == 1 and state.no_grad_depth == 0
    assert backend.feature_cache.entry_count == 0


def test_padding_uses_only_observed_prefix_and_owns_memory():
    observed = np.stack([np.full((3, 8, 8), -0.5), np.full((3, 8, 8), 0.5)]).astype(np.float32)
    padded = observed_only_padding(observed, 25)
    np.testing.assert_array_equal(padded[:2], observed)
    np.testing.assert_array_equal(padded[2:], np.repeat(observed[-1:], 23, axis=0))
    padded[0] = 1
    assert (observed[0] == -0.5).all()


@pytest.mark.parametrize("frames", [np.zeros((25, 3, 8, 8), np.float32), np.zeros((0, 3, 8, 8), np.float32),
                                    np.full((1, 3, 8, 8), np.nan, np.float32)])
def test_padding_rejects_future_length_or_invalid_input(frames):
    with pytest.raises(ValueError):
        observed_only_padding(frames, 25)


def test_checkpoint_hash_is_checked_not_filled_in_automatically(tmp_path):
    checkpoint = tmp_path / "fixture.safetensors"
    checkpoint.write_bytes(b"contract fixture; not model weights")
    expected = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert verify_checkpoint(checkpoint, expected)["checkpoint_sha256"] == expected
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_checkpoint(checkpoint, "0" * 64)
    with pytest.raises(ValueError, match="independently recorded"):
        verify_checkpoint(checkpoint, "")


def test_features_are_scoped_by_candidate_cfg_shape_and_refresh_interval():
    cache = FixedIntervalFeatureCache(interval=3)
    calls = []
    def compute():
        calls.append(1)
        return np.array([len(calls)], np.float32)
    def evaluate(step, cfg="cfg-u-then-c", shape=(50, 4)):
        return cache.evaluate(block="middle", step=step, cfg_scope=cfg, signature=shape, compute=compute)
    with cache.candidate("a"):
        np.testing.assert_array_equal(evaluate(0), [1])
        reused = evaluate(1)
        reused[0] = -99
        np.testing.assert_array_equal(evaluate(2), [1])
        np.testing.assert_array_equal(evaluate(3), [2])
        np.testing.assert_array_equal(evaluate(4, cfg="conditional-only"), [3])
        np.testing.assert_array_equal(evaluate(4, shape=(25, 4)), [4])
    assert cache.entry_count == 0
    with cache.candidate("b"):
        np.testing.assert_array_equal(evaluate(1), [5])
    assert (cache.hits, cache.misses) == (2, 5)


def test_feature_scope_is_cleared_on_failure():
    cache = FixedIntervalFeatureCache(2)
    with pytest.raises(RuntimeError, match="simulated"):
        with cache.candidate("a"):
            cache.evaluate(block="middle", step=0, cfg_scope="cfg", signature=(), compute=lambda: np.ones(4))
            raise RuntimeError("simulated OOM")
    assert cache.entry_count == 0
    with cache.candidate("b"):
        assert cache.entry_count == 0


def test_cached_posterior_replays_original_chunk_order_and_random_draws():
    """A CPU stand-in checks orchestration, not real VAE numerical parity."""
    random = np.random.RandomState(100)
    calls = []
    def regularize(parameters):
        calls.append(parameters.shape[0])
        return parameters + random.randn(*parameters.shape), {}
    parameters = [np.ones((14, 4, 2, 2)), np.ones((11, 4, 2, 2))]
    def encode(_images):
        return np.concatenate([regularize(x)[0] for x in parameters]) * 0.18215
    backend = VistaBackend.__new__(VistaBackend)
    backend.torch = SimpleNamespace(autocast=lambda *a, **k: contextlib.nullcontext(), cat=lambda xs, dim: np.concatenate(xs, axis=dim))
    backend.model = SimpleNamespace(
        encode_first_stage=encode, disable_first_stage_autocast=True, scale_factor=0.18215,
        first_stage_model=SimpleNamespace(regularization=regularize),
    )
    uncached = backend._encode({"images": None, "posterior_parameters": []}, {})
    next_uncached_draw = random.randn()
    assert calls == [14, 11]
    calls.clear()
    random.seed(100)
    stats = {"posterior_parameter_hits": 0}
    cached = backend._encode({"posterior_parameters": parameters}, stats)
    np.testing.assert_array_equal(cached, uncached)
    assert random.randn() == next_uncached_draw
    assert calls == [14, 11] and stats["posterior_parameter_hits"] == 1


@pytest.mark.parametrize("generation,options,upstream,match", [
    (GenerationConfig(total_frames=24), OptimizationConfig(), False, "25-frame"),
    (GenerationConfig(dtype="fp32"), OptimizationConfig(), False, "fp16 or bf16"),
    (GenerationConfig(decode_chunk_size=4), OptimizationConfig(), False, "overlap=3"),
    (GenerationConfig(), OptimizationConfig(attention_backend="sdpa"), False, "xFormers"),
    (GenerationConfig(), OptimizationConfig(keep_model_resident=False), False, "offload"),
    (GenerationConfig(), OptimizationConfig(compile_denoiser=True, feature_cache_interval=2, feature_cache_module="middle_block"), False, "graph-break"),
    (GenerationConfig(dtype="bf16"), OptimizationConfig(), True, "untouched upstream"),
    (GenerationConfig(), OptimizationConfig(compile_decoder=True), True, "cannot enable"),
])
def test_unsupported_combinations_fail_before_model_loading(generation, options, upstream, match):
    backend = VistaBackend.__new__(VistaBackend)
    backend.generation, backend.optimizations, backend.is_upstream = generation, options, upstream
    with pytest.raises(ValueError, match=match):
        backend._validate_options()


def test_numerical_validation_requires_all_values_within_frozen_tolerance():
    from drivebench.validate import compare_outputs
    reference = {"a": np.array([0.0, 1.0], np.float32)}
    actual = {"a": np.array([0.01, 1.001], np.float32)}
    passed = compare_outputs(reference, actual, atol=0.02, rtol=0)
    failed = compare_outputs(reference, actual, atol=0.002, rtol=0)
    assert passed["passed"] is True
    assert failed["passed"] is False
    assert failed["candidates"]["a"]["values_outside_tolerance"] == 1
    assert failed["candidates"]["a"]["rmse"] > 0
    with pytest.raises(ValueError, match="Nonfinite"):
        compare_outputs(reference, {"a": np.array([np.nan, 1.0])}, atol=0.1, rtol=0)


def test_cached_embeddings_own_returned_storage_and_restore_on_failure():
    class Feature:
        def __init__(self):
            self.data = np.ones((25, 4), np.float32)

        def clone(self):
            return self.data.copy()

    original = lambda _: "original"
    embedder = SimpleNamespace(forward=original)
    backend = VistaBackend.__new__(VistaBackend)
    backend.model = SimpleNamespace(conditioner=SimpleNamespace(embedders=[embedder]))
    feature = Feature()
    stats = {"conditioning_embedding_hits": 0}
    with pytest.raises(RuntimeError, match="simulated conditioning failure"):
        with backend._cached_embeddings({"image_embeddings": {0: feature}}, stats):
            first = embedder.forward(None)
            first[:] = 0
            np.testing.assert_array_equal(embedder.forward(None), np.ones((25, 4)))
            raise RuntimeError("simulated conditioning failure")
    assert embedder.forward is original
    assert stats["conditioning_embedding_hits"] == 2


def test_equal_outputs_without_exercising_cache_are_not_reuse_evidence():
    from drivebench.validate import cache_coverage
    assert not cache_coverage({"cache_stats": {}})["passed"]
    stats = {"conditioning_embedding_hits": 4, "posterior_parameter_hits": 2,
             "conditioning_cached_embedders": ["cond_frames_without_noise"]}
    assert not cache_coverage({"cache_stats": stats})["passed"]
    stats["conditioning_cached_embedders"].append("cond_frames")
    assert cache_coverage({"cache_stats": stats})["passed"]


def test_validation_provenance_binds_actual_inputs_and_ignores_future_files(tmp_path):
    from drivebench.data.manifests import write_manifest
    from drivebench.validate import input_provenance
    observed = tmp_path / "observed.png"
    observed.write_bytes(b"encoded observed image")
    record = {"request_id": "a", "scene_id": "scene-a", "source_split": "val",
              "future_frames": [str(tmp_path / "must-never-be-opened.png")]}
    manifest = tmp_path / "development.jsonl"
    write_manifest(manifest, [record], {"partition_role": "development", "synthetic": False})
    frames = np.zeros((1, 3, 8, 8), np.float32)
    request = {"request_id": "a", "scene_id": "scene-a", "candidate_ids": ["a:0"], "seeds": [42],
               "observed_frames": [str(observed)], "trajectories": [np.zeros((4, 2), np.float32)]}

    def fingerprint():
        return input_provenance(manifest, [record], record, request, frames, request_index=0)

    original = fingerprint()
    assert original["partition_role"] == "development"
    assert original["manifest_scene_ids"] == ["scene-a"]
    assert original["observed_file_sha256"] == [hashlib.sha256(observed.read_bytes()).hexdigest()]
    frames.flat[0] = 0.5
    assert fingerprint()["observed_sha256"] != original["observed_sha256"]
    request["trajectories"][0].flat[1] = 1
    assert fingerprint()["actions_sha256"] != original["actions_sha256"]
    metadata_path = manifest.with_suffix(".jsonl.meta.json")
    metadata = json.loads(metadata_path.read_text())
    metadata["scene_ids"] = ["unrelated-scene"]
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="scene IDs"):
        fingerprint()


def test_validation_report_tracks_configured_cache_budget_and_artifact_integrity(tmp_path, monkeypatch):
    """The proof-producing orchestration is tested without claiming GPU parity."""
    from drivebench import validate
    from drivebench.data.manifests import write_manifest
    observed = tmp_path / "observed.png"
    observed.write_bytes(b"input fixture")
    record = {"request_id": "a", "scene_id": "scene-a", "source_split": "val"}
    manifest = tmp_path / "development.jsonl"
    write_manifest(manifest, [record], {"partition_role": "development", "synthetic": False})
    request = {"request_id": "a", "scene_id": "scene-a", "candidate_ids": ["a:0", "a:1"], "seeds": [42, 43],
               "observed_frames": [str(observed)], "trajectories": [np.zeros((4, 2), np.float32)] * 2}
    config = {"model": {"backend": "vista"}, "data": {"root": str(tmp_path), "manifest": str(manifest),
              "verify_checksums": False}, "optimizations": {"conditioning_cache_bytes": 512}}
    monkeypatch.setattr(validate, "load_config", lambda _: config)
    monkeypatch.setattr(validate, "inference_request", lambda *a, **k: request)
    monkeypatch.setattr(validate, "read_preprocess", lambda *a: np.zeros((1, 3, 8, 8), np.float32))
    monkeypatch.setattr(validate, "collect_environment", lambda: {"python": "CPU orchestration fixture"})
    monkeypatch.setattr(validate, "_cleanup_gpu", lambda: None)
    called_options = []

    def fake_run(config, generation, options, request, frames, **kwargs):
        called_options.append(options)
        normal = {key: np.ones(4, np.float32) for key in request["candidate_ids"]}
        details = {"model_identity": {"backend": kwargs["backend"]}, "effective_config": generation.to_dict(),
                   "cache_stats": {"conditioning_embedding_hits": 4, "posterior_parameter_hits": 2,
                                   "conditioning_cached_embedders": ["cond_frames", "cond_frames_without_noise"]}}
        outputs = {"normal": normal}
        if kwargs.get("isolation"):
            outputs.update(reverse=normal, other_candidates_after_action_change={"a:1": normal["a:1"]})
        if kwargs.get("cache_check"):
            details["cache_key_checks"] = {"unchanged_input_hit": True, "unchanged_key_equal": True,
                                           "changed_input_missed": True, "changed_key_differs": True}
        return outputs, details

    monkeypatch.setattr(validate, "_run", fake_run)
    destination = tmp_path / "validation.json"
    assert validate.main(["--config", "fixture.yaml", "--mode", "all", "--atol", "0", "--rtol", "0",
                          "--output", str(destination)]) == 0
    report = json.loads(destination.read_text())
    assert report["schema_version"] == 2 and report["validated_request_count"] == 1
    assert report["mode"] == "all" and report["passed"] and not report["quality_validated"]
    assert called_options[-1].conditioning_cache_bytes == 512
    assert report["optimizations"]["conditioning_cache_bytes"] == 512
    assert report["input_provenance"]["scene_id"] == "scene-a"
    for run in report["runs"].values():
        for kind, path in run["outputs"].items():
            assert run["outputs_sha256"][kind] == validate.file_sha256(path)
