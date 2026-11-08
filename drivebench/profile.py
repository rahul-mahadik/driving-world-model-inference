"""Collect profiler artifacts in a separate run, never headline speed results."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from .benchmark import _compilation_count, _compiler_state, _validate_reference_environment, write_json
from .benchmarks.environment import collect_environment
from .config import load_config, validate_run_config
from .data import inference_request, load_manifest, verify_input_checksums
from .engine import InferenceEngine
from .verify_environment import has_cuda_kernel_trace, has_hardware_counters


def _profile(config_path, tool="torch", output="outputs/profile", kernel=None, dry_run=False):
    config = load_config(config_path)
    generation, optimizations = validate_run_config(config)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    if tool in {"nsys", "ncu"}:
        inner = [sys.executable, "-m", "drivebench.profile", "--config", str(Path(config_path).resolve()),
                 "--tool", "workload", "--output", str((destination / "workload").resolve())]
        if tool == "nsys":
            command = ["nsys", "profile", "--trace=cuda,nvtx,osrt", "--sample=none", "--force-overwrite=true",
                       "--capture-range=nvtx", "--nvtx-capture=drivebench_measured_request", "--capture-range-end=stop",
                       "-e", "NSYS_NVTX_PROFILER_REGISTER_ONLY=0",
                       "-o", str((destination / "timeline").resolve()), *inner]
        else:
            command = ["ncu", "--set", "full", "--target-processes", "all", "--launch-count", "10",
                       "--metrics", "sm__cycles_elapsed.avg,gpu__time_duration.sum", "--csv", "--page", "raw",
                       "--nvtx", "--nvtx-include", "drivebench_measured_request/",
                       "--force-overwrite", "-o", str((destination / "kernels").resolve())]
            if kernel:
                command += ["--kernel-name", f"regex:{kernel}"]
            command += inner
        write_json(destination / "command.json", {"argv": command, "profiled": True})
        if dry_run:
            print(shlex.join(command))
            return
        with (destination / "profiler.stdout.txt").open("w") as stdout, (destination / "profiler.stderr.txt").open("w") as stderr:
            completed = subprocess.run(command, stdout=stdout, stderr=stderr, text=True,
                                       env={**os.environ, "DRIVEBENCH_PROFILE": "1"})
        proof = {"profiled": True, "headline_eligible": False, "tool": tool,
                 "returncode": completed.returncode, "hardware_counters_collected": False,
                 "cuda_trace_collected": False}
        if tool == "ncu":
            proof["hardware_counters_collected"] = (completed.returncode == 0
                and (destination / "kernels.ncu-rep").is_file()
                and has_hardware_counters((destination / "profiler.stdout.txt").read_text()))
            passed = proof["hardware_counters_collected"]
        else:
            if completed.returncode == 0 and (destination / "timeline.nsys-rep").is_file():
                stats = subprocess.run(["nsys", "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv",
                                        str((destination / "timeline.nsys-rep").resolve())], text=True, capture_output=True)
                (destination / "kernel-summary.csv").write_text(stats.stdout)
                (destination / "stats.stderr.txt").write_text(stats.stderr)
                proof["cuda_trace_collected"] = stats.returncode == 0 and has_cuda_kernel_trace(stats.stdout)
            passed = proof["cuda_trace_collected"]
        proof["passed"] = passed
        write_json(destination / "capture_verification.json", proof)
        if not passed:
            raise RuntimeError(f"{tool} did not capture the required CUDA data; inspect {destination}")
        return
    if dry_run:
        print("PyTorch operator profiling: one initialized request; trace and operator table")
        return
    os.environ["DRIVEBENCH_PROFILE"] = "1"
    env = collect_environment()
    _validate_reference_environment(config, env)
    write_json(destination / "environment.json", env)
    write_json(destination / "config.json", config)
    if config.get("model", {}).get("backend") == "synthetic":
        raise ValueError("A synthetic test double is not useful for profiling Vista; use the H100 environment")
    engine = InferenceEngine(config["model"], generation, optimizations)
    records = load_manifest(config["data"]["manifest"])
    if not records or any(record.get("dataset") != "nuscenes" for record in records):
        raise ValueError("Real profiling requires a nonempty nuScenes manifest")
    record = records[0]
    if config["data"].get("verify_checksums", True):
        verify_input_checksums(record, config["data"]["root"])
    request = inference_request(record, config["data"]["root"], config.get("benchmark", {}).get("candidate_count", 1))
    def workload():
        if config.get("benchmark", {}).get("input_condition", "uncached") != "reused_conditioning":
            engine.cache.clear()
        engine.reset_peak_memory()
        context = engine.prepare_context(request["observed_frames"])
        result = engine.generate(context, request["trajectories"], request["seeds"], generation, request["candidate_ids"])
        engine.synchronize()
        if result.status != "ok":
            raise RuntimeError(f"profiling workload failed: {result.failures}")
        return result
    bench = config.get("benchmark", {})
    minimum = bench.get("warmup_requests", 3)
    limit = bench.get("max_warmup_requests", max(10, minimum))
    graphs, stable = _compilation_count(_compiler_state()), 0
    warmups = []
    for index in range(limit):
        workload()
        state = _compiler_state()
        current = _compilation_count(state)
        stable = stable + 1 if current == graphs else 0
        graphs = current
        warmups.append(state)
        if index + 1 >= minimum and stable >= 2:
            break
    write_json(destination / "warmup.json", {"compiler_states": warmups, "stable_iterations": stable})
    if (optimizations.compile_denoiser or optimizations.compile_decoder) and stable < 2:
        raise RuntimeError("compiler did not stabilize before profiling")
    import torch
    if tool == "torch":
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True, profile_memory=True, with_stack=True) as profiler:
            result = workload()
        profiler.export_chrome_trace(str(destination / "torch-trace.json"))
        (destination / "operators.txt").write_text(profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=80))
    else:
        # Nsight captures only this annotated request after compilation/warmup.
        with torch.cuda.nvtx.range("drivebench_measured_request"):
            result = workload()
    write_json(destination / "profile_metadata.json", {"profiled": True, "headline_eligible": False,
               "stage_timings_s": result.stage_timings_s, "peak_allocated_bytes": result.peak_allocated_bytes,
               "peak_reserved_bytes": result.peak_reserved_bytes, "denoiser_evaluations": result.denoiser_evaluations,
               "bottleneck_classification": "requires timeline/operator/kernel analysis"})


def profile(config_path, tool="torch", output="outputs/profile", kernel=None, dry_run=False):
    destination = Path(output)
    if not dry_run and any((destination / name).exists() for name in
                           ("profile_metadata.json", "capture_verification.json", "failure.json", "profiler.stdout.txt")):
        raise FileExistsError(f"Refusing to overwrite profiler evidence: {destination}")
    previous = os.environ.get("DRIVEBENCH_PROFILE")
    try:
        return _profile(config_path, tool, output, kernel, dry_run)
    except Exception as error:
        destination.mkdir(parents=True, exist_ok=True)
        write_json(destination / "failure.json", {"type": type(error).__name__, "message": str(error),
                   "profiled": True, "headline_eligible": False})
        raise
    finally:
        if previous is None:
            os.environ.pop("DRIVEBENCH_PROFILE", None)
        else:
            os.environ["DRIVEBENCH_PROFILE"] = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tool", choices=["torch", "nsys", "ncu", "workload"], default="torch")
    parser.add_argument("--output", default="outputs/profile")
    parser.add_argument("--kernel", help="Nsight Compute kernel regex selected from a prior timeline")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        profile(args.config, args.tool, args.output, args.kernel, args.dry_run)
    except (ValueError, RuntimeError, FileNotFoundError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"drivebench profile: {error}\n")


if __name__ == "__main__":
    main()
