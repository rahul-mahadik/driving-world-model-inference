"""CPU checks for bounded discovery/provenance; no GPU performance claims."""
from __future__ import annotations

import contextlib
from copy import deepcopy
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest

from drivebench.config import digest, file_sha256, load_config
from drivebench.types import GenerationConfig, OptimizationConfig
from scripts import probe_vista_optimization as probe


@pytest.mark.parametrize("path", sorted(Path("configs").glob("study_probe_*.yaml")))
def test_declared_probe_configs_are_independent_and_bounded(path):
    config, _, _ = probe.inspect_config(path)
    assert config["probe"]["request_index"] == 28
    assert config["probe"]["measurements"] == 1


@pytest.mark.parametrize("mutation", ["steps", "cache", "tolerance", "warmups", "headline"])
def test_probe_rejects_undeclared_changes(monkeypatch, mutation):
    config = load_config("configs/study_probe_decoder_compile.yaml")
    if mutation == "steps":
        config["generation"]["steps"] = 20
    elif mutation == "cache":
        config["optimizations"]["conditioning_cache_bytes"] = 1024
    elif mutation == "tolerance":
        config["probe"]["strict_atol"] = 0.01
    elif mutation == "warmups":
        config["probe"]["max_warmups"] = 1
    else:
        config["benchmark"]["headline"] = True
    monkeypatch.setattr(probe, "load_config", lambda _: config)
    with pytest.raises(ValueError):
        probe.inspect_config("fixture")


def reference_fixture(tmp_path):
    root = tmp_path / "upstream"
    root.mkdir()
    manifest = tmp_path / "development.jsonl"
    manifest.write_text('{"request_id":"request"}\n')
    config = {"probe": {"reference_run": str(root), "request_index": 28},
              "data": {"manifest": str(manifest)},
              "model": {"checkpoint_sha256": "checkpoint", "upstream_commit": "commit"}}
    request = {"request_id": "request", "candidate_ids": ["candidate"], "seeds": [12],
               "trajectories": [np.zeros((4, 2), np.float32)]}
    identity = {"checkpoint_sha256": "checkpoint", "upstream_commit": "commit",
                "parameter_grad_policy": "preserve_upstream_loader_flags"}
    snapshot = {"model": {"backend": "upstream"}, "generation": GenerationConfig().to_dict(),
                "optimizations": OptimizationConfig().to_dict(),
                "data": {"manifest_sha256": file_sha256(manifest),
                         "manifest_metadata": {"partition_role": "development"}}}
    row = {"request_index": 28, "request_id": "request", "scene_id": "scene", "status": "ok",
           "candidates": [{"candidate_id": "candidate", "seed": 12,
                           "trajectory_sha256": digest(request["trajectories"][0].tolist()), "frames_path": "output.npy"}]}
    for name, value in (("config.json", snapshot), ("model_identity.json", identity), ("environment.json", {}),
                        ("summary.json", {"successful_requests": 32, "failures": 0, "oom_attempts": 0})):
        (root / name).write_text(json.dumps(value))
    (root / "measurements.jsonl").write_text(json.dumps(row) + "\n")
    (root / "output.npy").write_bytes(b"fixture array read mocked separately")
    return root, config, request, row


@pytest.mark.parametrize("mutation", [None, "seed", "action", "incomplete", "traversal"])
def test_saved_reference_requires_matching_completed_upstream_request(tmp_path, monkeypatch, mutation):
    root, config, request, row = reference_fixture(tmp_path)
    monkeypatch.setattr(probe.np, "load", lambda *args, **kwargs: SimpleNamespace(shape=(25, 576, 1024, 3), dtype=np.float32))
    if mutation == "seed":
        row["candidates"][0]["seed"] = 13
    elif mutation == "action":
        row["candidates"][0]["trajectory_sha256"] = "changed"
    elif mutation == "incomplete":
        (root / "summary.json").write_text(json.dumps({"successful_requests": 31, "failures": 0, "oom_attempts": 0}))
    elif mutation == "traversal":
        row["candidates"][0]["frames_path"] = "../outside.npy"
    (root / "measurements.jsonl").write_text(json.dumps(row) + "\n")
    if mutation:
        with pytest.raises(ValueError):
            probe.load_reference(config, request, {"scene_id": "scene"}, GenerationConfig())
    else:
        _, evidence = probe.load_reference(config, request, {"scene_id": "scene"}, GenerationConfig())
        assert evidence["output_sha256"] == file_sha256(root / "output.npy")
        assert len(evidence["artifact_sha256"]) == 5


@pytest.mark.parametrize("fail", [False, True])
def test_block_events_restore_original_forward_after_success_or_failure(fail):
    class Event:
        def record(self):
            pass
        def elapsed_time(self, _other):
            return 2
    def middle(value):
        if fail:
            raise RuntimeError("block failure")
        return value + 1
    first, middle_module, last = (SimpleNamespace(forward=fn) for fn in (lambda x: x + 1, middle, lambda x: x + 1))
    originals = [first.forward, middle_module.forward, last.forward]
    backend = SimpleNamespace(torch=SimpleNamespace(cuda=SimpleNamespace(Event=lambda **kwargs: Event())),
                              model=SimpleNamespace(model=SimpleNamespace(diffusion_model=SimpleNamespace(
                                  input_blocks=[first], middle_block=middle_module, output_blocks=[last]))))
    with pytest.raises(RuntimeError) if fail else contextlib.nullcontext():
        with probe.block_events(backend) as records:
            first.forward(0)
            middle_module.forward(0)
            last.forward(0)
    assert [first.forward, middle_module.forward, last.forward] == originals
    assert len(records) == (2 if fail else 3)
    if not fail:
        summary = probe.summarize_blocks(records, 0.01)
        assert summary["blocks"]["middle_block"]["fraction_of_instrumented_denoising"] == 0.2


def test_cuda_graph_claim_requires_runtime_launch_event(tmp_path):
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({"traceEvents": [
        {"cat": "cpu_op", "name": "cudaGraphLaunch"},
        {"cat": "cuda_runtime", "name": "cudaGraphInstantiate"}]}))
    assert probe.cuda_graph_evidence(trace)["cuda_graph_replay_observed"] is False
    trace.write_text(json.dumps({"traceEvents": [{"cat": "cuda_runtime", "name": "cudaGraphLaunch_v10000"}]}))
    assert probe.cuda_graph_evidence(trace)["cuda_graph_launches"] == 1


@pytest.mark.parametrize("mode", ["exact", "numerical_difference", "compiler_fallback"])
def test_probe_preserves_outputs_and_separates_numerical_failure_from_execution(tmp_path, monkeypatch, mode):
    import drivebench.benchmark as benchmark
    import drivebench.benchmarks.environment as environment
    import drivebench.data as data
    import drivebench.engine as engine_module
    import drivebench.validate as validate
    path = Path("configs/study_probe_decoder_compile.yaml" if mode == "compiler_fallback" else "configs/study_probe_bf16.yaml")
    config, generation, options = probe.inspect_config(path)
    state = {"calls": 0}
    torch, dynamo, inductor, inductor_config = (ModuleType(name) for name in
        ("torch", "torch._dynamo", "torch._inductor", "torch._inductor.config"))
    torch.__version__ = "2.0.1+cu118"
    dynamo.config = SimpleNamespace(disable=False, cache_size_limit=8)
    inductor_config.trace = SimpleNamespace(enabled=False)
    torch._dynamo, torch._inductor, inductor.config = dynamo, inductor, inductor_config
    for module in (torch, dynamo, inductor, inductor_config):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(benchmark, "_validate_reference_environment", lambda *args: {"passed": True})
    monkeypatch.setattr(environment, "collect_environment", lambda: {})
    monkeypatch.setattr(environment, "reference_identity", lambda _: {})
    monkeypatch.setattr(environment, "source_identity", lambda: "frozen-source")
    record = {"request_id": "r", "scene_id": "s"}
    monkeypatch.setattr(data, "load_manifest", lambda _: [record] * 32)
    monkeypatch.setattr(data, "verify_input_checksums", lambda *args, **kwargs: None)
    request = {"request_id": "r", "candidate_ids": ["a"], "seeds": [1], "trajectories": [np.zeros((4, 2))], "observed_frames": []}
    monkeypatch.setattr(data, "inference_request", lambda *args, **kwargs: request)
    monkeypatch.setattr(data, "read_preprocess", lambda *args: np.zeros((1, 3, 8, 8), np.float32))
    monkeypatch.setattr(validate, "input_provenance", lambda *args, **kwargs: {"request_id": "r"})
    monkeypatch.setattr(probe, "load_reference", lambda *args: (np.zeros((25, 8, 8, 3), np.float32),
        {"runtime_identity": {}, "model_identity": {"inference_config_sha256": "config"}}))
    monkeypatch.setattr(probe, "compiler_state", lambda: {"stats": {"unique_graphs": 1},
        "unimplemented": {"cache_size_limit reached": 1} if mode == "compiler_fallback" and state["calls"] else {}})
    monkeypatch.setattr(probe, "generated_records", lambda *args: ([{"contains_triton_source": True}], {}, ["fixture.cubin"]))
    class Engine:
        def __init__(self, *args, **kwargs):
            self.generation = generation
            self.backend = SimpleNamespace(identity={"inference_config_sha256": "config"})
            self.cache = SimpleNamespace(clear=lambda: None)
        def synchronize(self):
            pass
        def reset_peak_memory(self):
            pass
        def prepare_context(self, _):
            return None
        def generate(self, *args, **kwargs):
            state["calls"] += 1
            value = 0.01 if mode == "numerical_difference" else 0
            return SimpleNamespace(status="ok", effective_config=generation.to_dict(), stage_timings_s={},
                peak_allocated_bytes=1, peak_reserved_bytes=2, denoiser_evaluations=50,
                cache_stats={"oom_attempts": []}, fallbacks=[], frames={"a": np.full((25, 8, 8, 3), value, np.float32)})
    monkeypatch.setattr(engine_module, "InferenceEngine", Engine)
    output = tmp_path / "probe"
    code = probe.main(["--config", str(path), "--output", str(output)])
    report = json.loads((output / "probe.json").read_text())
    assert code == {"exact": 0, "numerical_difference": 2, "compiler_fallback": 1}[mode]
    assert (output / "cold.npy").is_file()
    assert report["headline_eligible"] is False and report["quality_validated"] is False
    assert report["strict_passed"] is (mode != "numerical_difference")
    if mode == "compiler_fallback":
        assert state["calls"] == 1 and "eager fallback" in report["error"]["message"]
    else:
        assert [row["label"] for row in report["executions"]] == ["cold", "warmup-0", "measured-0"]
    with pytest.raises(FileExistsError):
        probe.main(["--config", str(path), "--output", str(output)])
