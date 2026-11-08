import json

import pytest

from drivebench.benchmarks.statistics import summarize
from drivebench.benchmarks.environment import reference_identity
from drivebench.config import load_config, validate_run_config
from drivebench.experiments import balanced_order, experiment_matrix
from drivebench.verify_environment import has_cuda_kernel_trace, has_hardware_counters


def test_hardware_counter_gate_requires_real_numeric_counter():
    assert not has_hardware_counters("Nsight Compute installed successfully")
    assert not has_hardware_counters('"Metric Name","Metric Value"\n"gpu__time_duration.sum","123"')
    assert not has_hardware_counters('"Metric Name","Metric Value"\n"sm__cycles_elapsed.avg","N/A"')
    assert has_hardware_counters('==PROF== connected\n"Metric Name","Metric Value"\n"sm__cycles_elapsed.avg","1,234"')


def test_reference_identity_binds_cuda_module_and_jit_cache_settings():
    first = {"cuda_runtime_environment": {"CUDA_MODULE_LOADING": "LAZY", "CUDA_CACHE_DISABLE": "0"}}
    eager = {"cuda_runtime_environment": {"CUDA_MODULE_LOADING": "EAGER", "CUDA_CACHE_DISABLE": "0"}}
    no_cache = {"cuda_runtime_environment": {"CUDA_MODULE_LOADING": "LAZY", "CUDA_CACHE_DISABLE": "1"}}
    assert reference_identity(first) != reference_identity(eager)
    assert reference_identity(first) != reference_identity(no_cache)


def test_reference_identity_distinguishes_native_xformers_build(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from drivebench.benchmarks import environment
    build = tmp_path / "cpp_lib.json"
    build.write_text('{"env":{"TORCH_CUDA_ARCH_LIST":"9.0"}}')
    provenance = {"archive_info": {"hashes": {"sha256": "a" * 64}}}
    distribution = SimpleNamespace(version="0.0.20", locate_file=lambda _: build,
                                   read_text=lambda _: json.dumps(provenance))
    monkeypatch.setattr(environment.importlib.metadata, "distribution", lambda _: distribution)
    native = environment.xformers_build_identity()
    assert native["installed_wheel_sha256"] == "a" * 64
    assert native["cpp_lib_sha256"]
    provenance["archive_info"] = {"hash": "sha256=" + "b" * 64}
    official = environment.xformers_build_identity()
    assert official["version"] == native["version"]
    assert reference_identity({"xformers_build": native}) != reference_identity({"xformers_build": official})


def test_hardware_counter_gate_accepts_ncu_2025_wide_kernel_rows():
    # Reduced columns from the real H100 / NCU 2025.2.1 preflight capture.
    header = '"ID","Kernel Name","gpu__time_duration.sum","sm__cycles_elapsed.avg"\n'
    units = '"","","ns","cycle"\n'
    kernel = '"0","drivebench_smoke(float *, int)","9408","14933.98"\n'
    assert has_hardware_counters("==PROF== Disconnected\n" + header + units + kernel)
    assert not has_hardware_counters(header + units)
    assert not has_hardware_counters(header + '"0","","9408","14933.98"\n')
    assert not has_hardware_counters(header + '"","drivebench_smoke","9408","14933.98"\n')
    assert not has_hardware_counters('"ID","Kernel Name","gpu__time_duration.sum"\n"0","smoke","9408"')


@pytest.mark.parametrize("value", ["0", "-1", "N/A", "nan", "inf", "-inf", "cycle", ""])
def test_hardware_counter_gate_rejects_invalid_wide_counter(value):
    assert not has_hardware_counters(
        '"ID","Kernel Name","sm__cycles_elapsed.avg"\n"0","drivebench_smoke","' + value + '"\n'
    )


def test_small_or_synthetic_runs_do_not_get_headline_claims():
    rows = [{"status": "ok", "latency_s": .1, "scene_id": str(i), "candidates": [{"id": 1}]} for i in range(8)]
    summary = summarize(rows, .8, "synthetic", headline=True)
    assert not summary["headline_eligible"]
    assert summary["latency_p95_s"] is None
    assert not summary["quality_preserving"]


def test_config_inheritance_and_matrix():
    config = load_config("configs/compiled.yaml")
    generation, optimizations = validate_run_config(config)
    assert generation.total_frames == 25 and generation.steps == 50
    assert optimizations.compile_denoiser
    matrix = experiment_matrix(load_config("configs/baseline.yaml"))
    assert {matrix[f"steps-{n}"]["generation"]["steps"] for n in (20,30,40,50)} == {20,30,40,50}
    assert not matrix["feature-reuse-2"]["optimizations"]["compile_denoiser"]


def test_interleaved_order_balanced_and_seeded():
    order = balanced_order(["a", "b", "c"], 4, 1)
    assert order == balanced_order(["a", "b", "c"], 4, 1)
    assert order[:3] == order[3:6][::-1]
    assert all(order.count(x) == 4 for x in ["a", "b", "c"])
    with pytest.raises(ValueError):
        balanced_order(["a", "b"], 3)


def test_headline_request_count_enforced():
    config = load_config("configs/baseline.yaml")
    config["benchmark"]["requests"] = 2
    with pytest.raises(ValueError, match="200"):
        validate_run_config(config)


def test_failed_experiments_preserve_configuration_and_reason(tmp_path):
    from drivebench.benchmark import run_benchmark
    config = load_config("configs/demo.yaml")
    config["data"]["manifest"] = str(tmp_path / "missing.jsonl")
    output = tmp_path / "run"
    with pytest.raises(FileNotFoundError):
        run_benchmark(config, output)
    assert json.loads((output / "config.json").read_text())["model"]["backend"] == "synthetic"
    assert json.loads((output / "failure.json").read_text())["type"] == "FileNotFoundError"
    assert json.loads((output / "summary.json").read_text())["headline_eligible"] is False
    before = (output / "summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_benchmark(config, output)
    assert (output / "summary.json").read_bytes() == before


def test_real_benchmark_rejects_synthetic_manifest_before_model(tmp_path):
    from drivebench.benchmark import run_benchmark
    path = tmp_path / "fixture.jsonl"
    path.write_text(json.dumps({"request_id": "x", "dataset": "synthetic_fixture"}) + "\n")
    config = load_config("configs/baseline.yaml")
    config["data"]["manifest"] = str(path)
    with pytest.raises(ValueError, match="real nuScenes"):
        run_benchmark(config, tmp_path / "run")


def test_nsys_proof_requires_positive_kernel_execution():
    assert not has_cuda_kernel_trace("Report created successfully")
    header = '"Time (%)","Total Time (ns)","Instances","Name"\n'
    assert not has_cuda_kernel_trace(header + '100,0,1,drivebench_smoke')
    assert not has_cuda_kernel_trace(header + '100,100,0,drivebench_smoke')
    assert not has_cuda_kernel_trace(header + '100,100,1,unrelated', kernel="drivebench_smoke")
    assert has_cuda_kernel_trace(header + '100,100,1,"void drivebench_smoke(float *, int)"', kernel="drivebench_smoke")


def _small_benchmark(tmp_path, monkeypatch, prefetch_depth=0):
    import numpy as np
    import drivebench.benchmark as module
    records = [{"request_id": f"request-{i}", "scene_id": f"scene-{i}", "dataset": "synthetic_fixture"}
               for i in range(2)]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(record) + "\n" for record in records))
    config = {"model": {"backend": "synthetic"},
              "data": {"manifest": str(manifest), "root": str(tmp_path), "verify_checksums": False},
              "generation": {"height": 16, "width": 24, "total_frames": 5, "steps": 2, "dtype": "fp32"},
              "optimizations": {"prefetch_depth": prefetch_depth},
              "benchmark": {"requests": 3, "warmup_requests": 2, "save_outputs": 0}}
    monkeypatch.setattr(module, "collect_environment", lambda: {})
    monkeypatch.setattr(module, "read_preprocess", lambda *args: np.zeros((1, 3, 16, 24), dtype=np.float32))
    monkeypatch.setattr(module, "inference_request", lambda record, root, candidate_count: {
        "request_id": record["request_id"], "scene_id": record["scene_id"], "observed_frames": [],
        "trajectories": [np.zeros(8)], "seeds": [1], "candidate_ids": ["a"], "action_kinds": ["alternative"]})
    return config, module


@pytest.mark.parametrize("prefetch_depth", [0, 2])
def test_failed_context_keeps_raw_attempt_and_throughput_time(tmp_path, monkeypatch, prefetch_depth):
    config, module = _small_benchmark(tmp_path, monkeypatch, prefetch_depth)
    original = module.InferenceEngine.prepare_context
    original_prefetch = module.prefetch
    closed = []
    def tracked_prefetch(*args):
        try:
            yield from original_prefetch(*args)
        finally:
            closed.append(True)
    monkeypatch.setattr(module, "prefetch", tracked_prefetch)
    count = 0
    def prepare(self, frames):
        nonlocal count
        count += 1
        if count == 4:  # Two warmups, one success, then a measured failure.
            raise MemoryError("context allocation failed")
        return original(self, frames)
    monkeypatch.setattr(module.InferenceEngine, "prepare_context", prepare)
    output = tmp_path / "run"
    with pytest.raises(MemoryError, match="context allocation"):
        module.run_benchmark(config, output)
    assert closed == [True]
    rows = [json.loads(line) for line in (output / "measurements.jsonl").read_text().splitlines()]
    assert [row["status"] for row in rows] == ["ok", "out_of_memory"]
    assert rows[-1]["failure_stage"] == "prepare_context"
    assert rows[-1]["latency_s"] > 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["failures"] == 1 and summary["completed_futures"] == 1
    assert summary["measurement_wall_s"] > rows[-1]["latency_s"]
    assert summary["throughput_futures_s"] == 1 / summary["measurement_wall_s"]
    assert not summary["headline_eligible"]


def test_lazy_compile_failure_preserves_warmup_context(tmp_path, monkeypatch):
    config, module = _small_benchmark(tmp_path, monkeypatch)
    def compile_failure(*args, **kwargs):
        raise RuntimeError("compiler backend failed")
    monkeypatch.setattr(module.InferenceEngine, "generate", compile_failure)
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="compiler backend"):
        module.run_benchmark(config, output)
    cold = json.loads((output / "cold_start.json").read_text())
    assert cold["model_load_s"] >= 0
    assert len(cold["warmups"]) == 1
    assert cold["warmups"][0]["failure_stage"] == "generate"
    assert cold["warmups"][0]["failures"][0]["error"] == "compiler backend failed"


def test_reference_proof_rejects_different_runtime_and_empty_trace(tmp_path):
    from drivebench.benchmark import _validate_reference_environment
    env = {"gpu": {"uuid": "gpu1"}, "torch": {"version": "original"}, "python": "python", "packages": {}, "tools": {}}
    path = tmp_path / "verification.json"
    proof = {"passed": True, "environment": env, "profiling_proof": {
        "hardware_counters_collected": True, "nsys_trace_collected": True,
        "ncu": {"returncode": 0, "stdout": '"Metric Name","Metric Value"\n"sm__cycles_elapsed.avg","1"'},
        "nsys_stats": {"returncode": 0, "stdout": '"Total Time (ns)","Instances","Name"\n1,1,drivebench_smoke'}}}
    path.write_text(json.dumps(proof))
    config = {"model": {"backend": "vista"}, "environment_verification": str(path)}
    assert _validate_reference_environment(config, env)["passed"]
    changed = {**env, "torch": {"version": "changed"}}
    with pytest.raises(ValueError, match="runtime"):
        _validate_reference_environment(config, changed)
    proof["profiling_proof"]["nsys_stats"]["stdout"] = "empty report"
    path.write_text(json.dumps(proof))
    with pytest.raises(ValueError, match="Nsight"):
        _validate_reference_environment(config, env)


def test_ncu_success_without_counters_is_failure_and_preserves_logs(tmp_path, monkeypatch):
    import subprocess
    import drivebench.profile as module
    destination = tmp_path / "profile"
    def fake_run(command, **kwargs):
        kwargs["stdout"].write("No kernels were profiled")
        (destination / "kernels.ncu-rep").write_bytes(b"empty")
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="required CUDA data"):
        module.profile("configs/baseline.yaml", tool="ncu", output=destination)
    proof = json.loads((destination / "capture_verification.json").read_text())
    assert not proof["passed"] and not proof["hardware_counters_collected"]
    assert (destination / "failure.json").is_file()
    assert (destination / "profiler.stdout.txt").read_text() == "No kernels were profiled"


def test_nsys_scope_handles_torch_unregistered_nvtx_strings(tmp_path):
    from drivebench.profile import profile
    profile("configs/baseline.yaml", tool="nsys", output=tmp_path, dry_run=True)
    argv = json.loads((tmp_path / "command.json").read_text())["argv"]
    assert "--capture-range=nvtx" in argv
    assert "--nvtx-capture=drivebench_measured_request" in argv
    assert "NSYS_NVTX_PROFILER_REGISTER_ONLY=0" in argv


def _numerical_fixture(tmp_path):
    from datetime import datetime, timezone
    from drivebench.benchmarks.environment import reference_identity, source_identity
    from drivebench.config import digest, file_sha256
    from drivebench.data import write_manifest
    from drivebench.types import GenerationConfig, OptimizationConfig
    frame = tmp_path / "observed.jpg"
    frame.write_bytes(b"observed fixture bytes")
    records = [{"request_id": "request", "scene_id": "development-scene", "dataset": "nuscenes",
                "observed_frames": ["observed.jpg"], "frame_checksums": {"observed.jpg": file_sha256(frame)}}]
    manifest = tmp_path / "development.jsonl"
    write_manifest(manifest, records, {"partition_role": "development"})
    metadata = manifest.with_suffix(".jsonl.meta.json")
    generation, options = GenerationConfig(), OptimizationConfig()
    environment = {"gpu": {"uuid": "h100-1"}, "torch": {"version": "frozen"}}
    identity = {"backend": "vista", "upstream_commit": "upstream-commit",
                "checkpoint_sha256": "checkpoint-hash", "inference_config_sha256": "inference-hash"}
    output = tmp_path / "normal.npz"
    output.write_bytes(b"fixed raw outputs")
    details = {"model_identity": identity, "effective_config": {**generation.to_dict(), "optimizations": options.to_dict()},
               "outputs": {"normal": str(output)}, "outputs_sha256": {"normal": file_sha256(output)}}
    provenance = {"manifest_path": str(manifest), "manifest_sha256": file_sha256(manifest),
                  "manifest_metadata_sha256": file_sha256(metadata), "partition_role": "development",
                  "manifest_scene_ids": ["development-scene"], "request_index": 0, "request_id": "request",
                  "scene_id": "development-scene", "record_sha256": digest(records[0]), "candidate_ids": ["a", "b"]}
    proof = {"schema_version": 2, "passed": True, "created_at": datetime.now(timezone.utc).isoformat(),
             "implementation_sha256": source_identity(), "runtime_identity": reference_identity(environment),
             "generation": generation.to_dict(), "optimizations": options.to_dict(), "input_provenance": provenance,
             "input_provenance_sha256": digest(provenance), "validated_request_count": 1,
             "config": {"data": {"root": str(tmp_path)}},
             "runs": {name: details for name in ("upstream", "refactored", "isolation")},
             "checks": {name: {"passed": True}
                        for name in ("upstream_agreement", "candidate_order", "unrelated_candidate_action_change")}}
    path = tmp_path / "numerical.json"
    path.write_text(json.dumps(proof))
    config = {"model": {"backend": "vista"}, "data": {"manifest": str(manifest)}, "numerical_validation": str(path)}
    return config, generation, options, environment, identity, records, proof, path


def test_numerical_gate_binds_configuration_and_saved_arrays(tmp_path):
    from dataclasses import replace
    from drivebench.benchmarks.agreement import numerical_evidence
    config, generation, options, env, identity, records, proof, path = _numerical_fixture(tmp_path)
    assert numerical_evidence(config, generation, options, env, identity, records)["passed"]
    assert not numerical_evidence({}, generation, options, env, identity, records)["passed"]
    gate = numerical_evidence(config, generation, replace(options, compile_denoiser=True), env, identity, records)
    assert not gate["passed"] and any("optimization settings differ" in reason for reason in gate["reasons"])
    (tmp_path / "normal.npz").write_bytes(b"changed raw outputs")
    gate = numerical_evidence(config, generation, options, env, identity, records)
    assert not gate["passed"] and any("output changed" in reason for reason in gate["reasons"])


def test_numerical_gate_allows_disjoint_heldout_and_rejects_scene_overlap(tmp_path):
    from drivebench.benchmarks.agreement import numerical_evidence
    from drivebench.data import write_manifest
    config, generation, options, env, identity, records, proof, path = _numerical_fixture(tmp_path)
    evaluation = tmp_path / "evaluation.jsonl"
    heldout = [{**records[0], "request_id": "other-request", "scene_id": "heldout-scene"}]
    write_manifest(evaluation, heldout, {"partition_role": "held_out_evaluation"})
    config["data"]["manifest"] = str(evaluation)
    assert numerical_evidence(config, generation, options, env, identity, heldout)["passed"]
    heldout[0]["scene_id"] = "development-scene"
    write_manifest(evaluation, heldout, {"partition_role": "held_out_evaluation"})
    gate = numerical_evidence(config, generation, options, env, identity, heldout)
    assert not gate["passed"] and any("scenes overlap" in reason for reason in gate["reasons"])


def test_numerical_gate_rejects_source_change_and_missing_isolation(tmp_path):
    from drivebench.benchmarks.agreement import numerical_evidence
    config, generation, options, env, identity, records, proof, path = _numerical_fixture(tmp_path)
    proof["implementation_sha256"] = "old source"
    del proof["checks"]["candidate_order"]
    path.write_text(json.dumps(proof))
    gate = numerical_evidence(config, generation, options, env, identity, records)
    assert not gate["passed"]
    assert any("different engine source" in reason for reason in gate["reasons"])
    assert any("candidate_order" in reason for reason in gate["reasons"])


def test_numerical_gate_rejects_cache_proof_for_a_different_budget(tmp_path):
    import copy
    from dataclasses import replace
    from drivebench.benchmarks.agreement import numerical_evidence
    config, generation, options, env, identity, records, proof, path = _numerical_fixture(tmp_path)
    options = replace(options, conditioning_cache_bytes=1)
    for name in ("refactored", "isolation"):
        proof["runs"][name] = copy.deepcopy(proof["runs"][name])
        proof["runs"][name]["effective_config"]["optimizations"] = options.to_dict()
    proof["runs"]["cached"] = copy.deepcopy(proof["runs"]["refactored"])
    proof["runs"]["cached"]["effective_config"]["optimizations"]["conditioning_cache_bytes"] = 2**31
    proof["runs"]["uncached"] = copy.deepcopy(proof["runs"]["upstream"])
    for name in ("conditioning_cache_agreement", "conditioning_cache_keys", "conditioning_cache_exercised"):
        proof["checks"][name] = {"passed": True}
    path.write_text(json.dumps(proof))
    gate = numerical_evidence(config, generation, options, env, identity, records)
    assert not gate["passed"]
    assert any("cached uses different optimization settings" in reason for reason in gate["reasons"])


@pytest.mark.parametrize("mutation", [None, "missing_reverse", "serial_fallback", "oom", "count", "settings"])
def test_numerical_gate_requires_actual_joint_work_for_every_isolation_variant(tmp_path, mutation):
    import copy
    from dataclasses import replace
    from drivebench.benchmarks.agreement import numerical_evidence
    config, generation, options, env, identity, records, proof, path = _numerical_fixture(tmp_path)
    options = replace(options, max_batch_size=2)
    identity["candidate_batch_limit"] = 2
    for name in ("refactored", "isolation"):
        details = copy.deepcopy(proof["runs"][name])
        details["candidate_count"] = 2
        details["model_identity"]["candidate_batch_limit"] = 2
        details["effective_config"].update(optimizations=options.to_dict(), candidate_count=2,
                                           requested_batch_size=2, actual_batch_sizes=[2])
        execution = {"status": "ok", "effective_config": details["effective_config"],
                     "cache_stats": {"oom_attempts": [], "denoiser_evaluations_complete": True},
                     "denoiser_evaluations": generation.steps}
        variants = ["normal", "reverse", "action_change"] if name == "isolation" else ["normal"]
        details["executions"] = {variant: copy.deepcopy(execution) for variant in variants}
        proof["runs"][name] = details
    executions = proof["runs"]["isolation"]["executions"]
    if mutation == "missing_reverse":
        del executions["reverse"]
    elif mutation == "serial_fallback":
        executions["reverse"]["effective_config"]["actual_batch_sizes"] = [1, 1]
    elif mutation == "oom":
        executions["action_change"]["cache_stats"]["oom_attempts"] = [{"batch_size": 2}]
    elif mutation == "count":
        executions["action_change"]["denoiser_evaluations"] = generation.steps * 2
    elif mutation == "settings":
        executions["reverse"]["effective_config"]["optimizations"]["max_batch_size"] = 1
    path.write_text(json.dumps(proof))
    gate = numerical_evidence(config, generation, options, env, identity, records)
    assert gate["passed"] is (mutation is None), gate["reasons"]
    if mutation:
        assert any("isolation:" in reason for reason in gate["reasons"])
