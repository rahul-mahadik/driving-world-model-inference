"""Reproducible end-to-end requests, from manifest resolution to host frames."""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .benchmarks.environment import collect_environment, reference_identity
from .benchmarks.agreement import numerical_evidence
from .benchmarks.statistics import summarize
from .config import digest, file_sha256, load_config, validate_run_config
from .data import inference_request, load_manifest, read_preprocess, verify_input_checksums
from .engine import InferenceEngine
from .runtime.prefetch import prefetch


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def _compiler_state():
    if "torch" not in sys.modules:
        return {}
    try:
        from torch._dynamo.utils import counters
        return {category: dict(counts) for category, counts in counters.items()}
    except (ImportError, AttributeError):
        return {}


def _compilation_count(state):
    return state.get("stats", {}).get("unique_graphs", 0)


def _validate_reference_environment(config, environment):
    if config.get("model", {}).get("backend", "vista") == "synthetic":
        return {"passed": False, "reason": "synthetic CPU test double"}
    verification = Path(config.get("environment_verification", "outputs/environment/verification.json"))
    if not verification.is_file():
        raise ValueError("Run python -m drivebench.verify_environment --dedicated before the real workload")
    proof = json.loads(verification.read_text())
    from .verify_environment import has_cuda_kernel_trace, has_hardware_counters
    profiling = proof.get("profiling_proof", {})
    if (not proof.get("passed") or not proof.get("profiling_proof", {}).get("hardware_counters_collected")
            or not proof.get("profiling_proof", {}).get("nsys_trace_collected")
            or profiling.get("ncu", {}).get("returncode") != 0
            or not has_hardware_counters(profiling.get("ncu", {}).get("stdout", ""))
            or profiling.get("nsys_stats", {}).get("returncode") != 0
            or not has_cuda_kernel_trace(profiling.get("nsys_stats", {}).get("stdout", ""), kernel="drivebench_smoke")):
        raise ValueError("H100 reference environment and Nsight hardware-counter verification must pass")
    # Check identity again; yesterday's successful run on another host is not proof.
    if reference_identity(environment) != reference_identity(proof.get("environment", {})):
        raise ValueError("current runtime or GPU configuration differs from profiling verification; rerun verification")
    lock_path = config.get("environment_lock")
    if lock_path:
        expected = json.loads(Path(lock_path).read_text())
        if reference_identity(environment) != expected:
            raise ValueError("runtime differs from the frozen environment lock")
    return {"passed": True, "verification_sha256": file_sha256(verification)}


def _run_benchmark(config, output=None, profiled=False):
    generation, optimizations = validate_run_config(config)
    bench = config.get("benchmark", {})
    model = config.get("model", {"backend": "vista"})
    backend_name = model.get("backend", "vista")
    manifest_path = Path(config["data"]["manifest"])
    records = load_manifest(manifest_path)
    if not records:
        raise ValueError("manifest is empty")
    root = Path(config["data"]["root"])
    count = bench.get("candidate_count", 1)
    condition = bench.get("input_condition", "uncached")
    if condition == "reused_conditioning" and not optimizations.conditioning_cache_bytes:
        raise ValueError("reused_conditioning workload requires a positive conditioning cache budget")
    if backend_name != "synthetic" and any(r.get("dataset") != "nuscenes" for r in records):
        raise ValueError("real Vista benchmark requires a real nuScenes manifest")
    # Verify file identity before measured iterations. This intentionally warms OS
    # cache; uncached below means no decoded/context reuse, not drop_caches.
    if config["data"].get("verify_checksums", True):
        for record in records:
            verify_input_checksums(record, root)
    destination = Path(output or config.get("output", "outputs/run"))
    if (destination / "measurements.jsonl").exists():
        raise FileExistsError(f"Refusing to overwrite a previous run: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    snapshot = dict(config)
    snapshot["generation"] = generation.to_dict()
    snapshot["optimizations"] = optimizations.to_dict()
    snapshot["data"] = {**config["data"], "manifest_sha256": file_sha256(manifest_path)}
    manifest_metadata = manifest_path.with_suffix(manifest_path.suffix + ".meta.json")
    if manifest_metadata.is_file():
        snapshot["data"]["manifest_metadata"] = json.loads(manifest_metadata.read_text())
    snapshot["profiled"] = profiled
    snapshot["benchmark"] = {**bench, "requests": bench.get("requests", 200),
                             "save_outputs": bench.get("save_outputs", min(bench.get("requests", 200), len(records)))}
    snapshot["started_at"] = datetime.now(timezone.utc).isoformat()
    write_json(destination / "config.json", snapshot)
    environment = collect_environment()
    write_json(destination / "environment.json", environment)
    write_json(destination / "environment_gate.json", _validate_reference_environment(config, environment))
    started = time.perf_counter()
    try:
        engine = InferenceEngine(model, generation, optimizations)
        engine.synchronize()
    except Exception as error:
        write_json(destination / "cold_start.json", {"model_load_s": time.perf_counter() - started,
                   "warmups": [], "status": "failed", "failure_stage": "model_initialization",
                   "error_type": type(error).__name__, "error": str(error)})
        raise
    model_load_s = time.perf_counter() - started
    write_json(destination / "model_identity.json", getattr(engine.backend, "identity", {}))
    numerical_gate = numerical_evidence(snapshot, generation, optimizations, environment,
                                        getattr(engine.backend, "identity", {}), records)
    write_json(destination / "numerical_gate.json", numerical_gate)

    def load(item):
        index, record = item
        began = time.perf_counter()
        request = frames = error = None
        try:
            request = inference_request(record, root, candidate_count=count)
            frames = read_preprocess(request["observed_frames"], generation.height, generation.width)
        except Exception as caught:
            error = caught
        return index, record, request, frames, began, time.perf_counter() - began, error

    def execute(loaded):
        index, record, request, frames, began, load_s, load_error = loaded
        execution_started = time.perf_counter()
        queue_wait = max(0.0, execution_started - began - load_s)
        stage = "input_load_preprocess"
        context = None
        try:
            if load_error is not None:
                raise load_error
            stage = "prepare_context"
            if condition != "reused_conditioning":
                engine.cache.clear()
            engine.reset_peak_memory()
            context = engine.prepare_context(frames)
            stage = "generate"
            result = engine.generate(context, request["trajectories"], request["seeds"], generation, request["candidate_ids"])
            stage = "host_frames_ready"
            engine.synchronize()
        except Exception as error:
            # Include any queued device work from the failed attempt. CUDA may
            # itself remain in an error state; preserve the initiating error.
            synchronization_error = None
            try:
                engine.synchronize()
            except Exception as sync_error:
                synchronization_error = str(sync_error)
            elapsed = time.perf_counter() - began
            oom = isinstance(error, MemoryError) or type(error).__name__ in {"OutOfMemoryError", "CUDAOutOfMemoryError"}
            row = {"request_id": record.get("request_id", record.get("id")), "scene_id": record.get("scene_id"),
                   "repeat": index // len(records), "request_index": index, "latency_s": elapsed,
                   "status": "out_of_memory" if oom else "failed", "candidate_count": count,
                   "batch_size": 0, "input_condition": condition, "candidates": [], "cache_stats": {},
                   "stage_timings_s": {"input_load_preprocess": load_s, "queue_wait": queue_wait},
                   "failure_stage": stage, "failures": [{"type": type(error).__name__, "error": str(error), "stage": stage}],
                   "fatal_error": True,
                   "effective_config": {**generation.to_dict(), "optimizations": optimizations.to_dict(),
                                        "candidate_count": count, "actual_batch_sizes": []}}
            if oom:
                row["cache_stats"]["oom_attempts"] = [{"stage": stage, "error": str(error), "retried": False}]
            if synchronization_error is not None:
                row["failure_synchronization_error"] = synchronization_error
            if context is not None:
                row["stage_timings_s"]["prepare_context"] = context.preparation_s
            return row, None, request, record, error
        elapsed = time.perf_counter() - began
        row = {"request_id": request["request_id"], "scene_id": request["scene_id"], "repeat": index // len(records),
               "request_index": index, "latency_s": elapsed, "status": result.status,
               "candidate_count": count, "batch_size": max(result.effective_config["actual_batch_sizes"], default=0),
               "input_condition": condition, "stage_timings_s": {"input_load_preprocess": load_s, **result.stage_timings_s},
               "gpu_inference_s": result.gpu_inference_s,
               "peak_allocated_bytes": result.peak_allocated_bytes, "peak_reserved_bytes": result.peak_reserved_bytes,
               "denoiser_evaluations": result.denoiser_evaluations, "cache_stats": result.cache_stats,
               "effective_config": result.effective_config, "fallbacks": result.fallbacks, "failures": result.failures,
               "categories": record.get("categories", []), "candidates": []}
        row["stage_timings_s"]["queue_wait"] = queue_wait
        return row, result, request, record, None

    if condition == "warm_filesystem":
        for record in records:
            for frame in record["observed_frames"]:
                with (root / frame).open("rb") as f:
                    while f.read(1024 * 1024):
                        pass
    warmups = []
    minimum = bench.get("warmup_requests", 3)
    stable = 0
    limit = max(minimum, bench.get("max_warmup_requests", max(10, minimum)))
    previous_graphs = _compilation_count(_compiler_state())
    for index in range(limit):
        row, _, _, _, error = execute(load((index, records[index % len(records)])))
        state = _compiler_state()
        graphs = _compilation_count(state)
        stable = stable + 1 if graphs == previous_graphs else 0
        previous_graphs = graphs
        warmups.append({**row, "compiler": state})
        write_json(destination / "cold_start.json", {"model_load_s": model_load_s, "warmups": warmups,
                   "compilation_and_initialization_excluded": True, "status": "warming_up",
                   "note": "First warmups include lazy compilation; no inferred compile-only duration."})
        if error is not None:
            raise error
        if row["status"] != "ok":
            raise RuntimeError("warmup failed; inspect memory settings before a measured run")
        if index + 1 >= minimum and stable >= 2:
            break
    if (optimizations.compile_denoiser or optimizations.compile_decoder) and stable < 2:
        raise RuntimeError("compiler did not stabilize before max_warmup_requests")
    write_json(destination / "cold_start.json", {"model_load_s": model_load_s, "warmups": warmups,
                                               "compilation_and_initialization_excluded": True, "status": "ready",
                                               "note": "First warmups include lazy compilation; no inferred compile-only duration."})
    requests = bench.get("requests", 200)
    saves = bench.get("save_outputs", min(requests, len(records)))
    rows, export_s = [], 0.0
    engine.synchronize()
    started = time.perf_counter()
    compiler_before = _compiler_state()
    loader = prefetch(((i, records[i % len(records)]) for i in range(requests)), load, optimizations.prefetch_depth)
    write_json(destination / "measurement_state.json", {"started_perf_counter_s": started, "elapsed_wall_s": 0})
    with contextlib.closing(loader), (destination / "measurements.jsonl").open("w") as f:
        for loaded in loader:
            row, result, request, record, error = execute(loaded)
            if error is not None:
                rows.append(row)
                f.write(json.dumps(row, allow_nan=False) + "\n")
                f.flush()
                write_json(destination / "measurement_state.json", {"started_perf_counter_s": started,
                                                                   "elapsed_wall_s": time.perf_counter() - started})
                raise error
            export_start = time.perf_counter()
            for j, candidate_id in enumerate(result.candidate_ids):
                if candidate_id not in result.frames:
                    continue
                relative = None
                if row["request_index"] < saves:
                    relative = Path("outputs") / f"request-{row['request_index']:05d}" / f"candidate-{j:03d}.npy"
                    (destination / relative).parent.mkdir(parents=True, exist_ok=True)
                    np.save(destination / relative, result.frames[candidate_id], allow_pickle=False)
                candidate = {"candidate_id": candidate_id, "seed": result.seeds[j], "action_kind": request["action_kinds"][j],
                             "frames_path": str(relative) if relative else None, "conditioning_frames": generation.conditioning_frames,
                             "trajectory_sha256": digest(np.asarray(request["trajectories"][j]).tolist())}
                if candidate["action_kind"] == "recorded":
                    candidate["ground_truth_paths"] = [str((root / name).resolve()) for name in record.get("future_frames", [])]
                row["candidates"].append(candidate)
            row["export_s"] = time.perf_counter() - export_start
            export_s += row["export_s"]
            rows.append(row)
            f.write(json.dumps(row, allow_nan=False) + "\n")
            f.flush()
            write_json(destination / "measurement_state.json", {"started_perf_counter_s": started,
                                                               "elapsed_wall_s": time.perf_counter() - started})
    engine.synchronize()
    total_wall = time.perf_counter() - started
    compiler_after = _compiler_state()
    summary = summarize(rows, total_wall, backend_name, bench.get("headline", False), profiled)
    summary["numerical_validation"] = numerical_gate
    if not numerical_gate["passed"]:
        summary["headline_eligible"] = False
    summary.update({"export_s": export_s, "throughput_includes_export": True,
                    "throughput_measurement_wall_s": total_wall,
                    "summed_request_latency_futures_s": summary["completed_futures"] / sum(r["latency_s"] for r in rows),
                    "input_cache_semantics": "uncached means no decoded/conditioning reuse; filesystem cache is OS-controlled",
                    "compiler_before": compiler_before, "compiler_after": compiler_after,
                    "recompiled_during_measurement": _compilation_count(compiler_after) > _compilation_count(compiler_before)})
    if summary["recompiled_during_measurement"]:
        summary["headline_eligible"] = False
    # The authoritative throughput excludes synchronous export for non-prefetch
    # runs. With prefetch, subtracting overlapping export is invalid: require
    # save_outputs=0 for a clean continuous-throughput experiment.
    if optimizations.prefetch_depth == 0:
        measure_wall = sum(r["latency_s"] for r in rows)
        summary["throughput_futures_s"] = summary["completed_futures"] / measure_wall
        summary["throughput_measurement_wall_s"] = measure_wall
        summary["throughput_includes_export"] = False
    elif saves:
        summary["headline_eligible"] = False
        summary["throughput_note"] = "prefetch + export overlap: rerun with save_outputs=0 for headline throughput"
    write_json(destination / "summary.json", summary)
    return destination, summary


def run_benchmark(config, output=None, profiled=False):
    destination = Path(output or config.get("output", "outputs/run"))
    if any((destination / name).exists() for name in ("measurements.jsonl", "summary.json", "failure.json")):
        raise FileExistsError(f"Refusing to overwrite a previous run: {destination}")
    try:
        return _run_benchmark(config, output, profiled)
    except Exception as error:
        failed_at = time.perf_counter()
        destination.mkdir(parents=True, exist_ok=True)
        if not (destination / "config.json").exists():
            write_json(destination / "config.json", config)
        if not (destination / "environment.json").exists():
            write_json(destination / "environment.json", collect_environment())
        write_json(destination / "failure.json", {"type": type(error).__name__, "message": str(error),
                                                  "status": "failed", "headline_eligible": False})
        cold_path = destination / "cold_start.json"
        if cold_path.exists():
            cold = json.loads(cold_path.read_text())
            if cold.get("status") == "warming_up":
                cold.update(status="failed", failure_stage="warmup", error_type=type(error).__name__, error=str(error))
                write_json(cold_path, cold)
        raw_path = destination / "measurements.jsonl"
        rows = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()] if raw_path.exists() else []
        raw_path.touch(exist_ok=True)
        state_path = destination / "measurement_state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        elapsed = max(state.get("elapsed_wall_s", 0), failed_at - state.get("started_perf_counter_s", failed_at))
        summary = summarize(rows, elapsed, config.get("model", {}).get("backend", "vista"), False, profiled)
        summary.update(status="failed", run_failure=str(error),
                       failures=summary["failures"] + (0 if rows and rows[-1].get("fatal_error") else 1),
                       throughput_measurement_wall_s=elapsed, throughput_includes_export=True,
                       throughput_note="Incomplete run: elapsed wall includes all recorded failed attempts and any exports")
        write_json(destination / "summary.json", summary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--profiled", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        destination, summary = run_benchmark(load_config(args.config), args.output, args.profiled)
    except (ValueError, TypeError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        parser.exit(1, f"drivebench: {error}\n")
    print(json.dumps({"output": str(destination), **summary}, indent=2))
    return 0 if not summary["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
