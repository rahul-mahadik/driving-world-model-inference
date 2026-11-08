#!/usr/bin/env python3
"""Bounded development discovery; never a headline or quality-acceptance report.

Run each config in a fresh process under timeout --kill-after=30s 600s. Exact
comparison is always reported, including failures. Separately calibrated future
tolerances cannot rewrite this zero-tolerance diagnostic.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import replace
from datetime import datetime, timezone
import inspect
import json
import logging
import os
from pathlib import Path
import signal
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from drivebench.config import digest, file_sha256, load_config, validate_run_config
from drivebench.types import GenerationConfig, OptimizationConfig
from scripts.validate_compiled_attention_boundary import NATIVE_SOURCE_SHA256, generated_records


KINDS = {"eager", "decoder_compile", "denoiser_default", "denoiser_reduce_overhead",
         "denoiser_dynamic", "bf16", "decode_chunk6", "block_timing", "shape_static", "shape_dynamic"}


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def inspect_config(path):
    config = load_config(path)
    generation, options = validate_run_config(config)
    probe = config.get("probe", {})
    kind = probe.get("kind")
    if kind not in KINDS:
        raise ValueError("Select a declared independent probe kind")
    expected_generation, expected_options = GenerationConfig(), OptimizationConfig()
    if kind == "bf16":
        expected_generation = replace(expected_generation, dtype="bf16")
    elif kind == "decode_chunk6":
        expected_generation = replace(expected_generation, decode_chunk_size=6)
    if kind == "decoder_compile":
        expected_options = replace(expected_options, compile_decoder=True)
    elif kind.startswith("denoiser_") or kind.startswith("shape_"):
        expected_options = replace(expected_options, compile_denoiser=True,
            compile_mode="reduce-overhead" if kind == "denoiser_reduce_overhead" else "default",
            dynamic_shapes=kind in {"denoiser_dynamic", "shape_dynamic"})
    if generation != expected_generation or options != expected_options:
        raise ValueError("Probe must change only its declared independent option")
    if (config["model"].get("backend") != "vista" or config.get("numerical_validation") is not None
            or config.get("benchmark", {}).get("headline") is not False
            or config.get("benchmark", {}).get("candidate_count") != 1):
        raise ValueError("Probes require one Vista candidate, no inherited proof, and headline=false")
    if config["data"].get("verify_checksums") is not True:
        raise ValueError("Probe input checksums must be verified")
    if probe.get("strict_atol") != 0 or probe.get("strict_rtol") != 0:
        raise ValueError("Preserve the original zero-tolerance diagnostic; calibrate acceptance separately")
    for key, lower, upper in (("request_index", 0, 31), ("max_warmups", 0, 4), ("measurements", 1, 2)):
        if type(probe.get(key)) is not int or not lower <= probe[key] <= upper:
            raise ValueError(f"Invalid bounded probe {key}")
    if (options.compile_denoiser or options.compile_decoder) and probe["max_warmups"] < 2:
        raise ValueError("Compiled probes need up to two stable warm requests")
    if probe.get("cuda_graph_trace", False) and kind != "denoiser_reduce_overhead":
        raise ValueError("CUDA graph discovery belongs to the explicit reduce-overhead probe")
    return config, generation, options


def load_reference(config, request, record, generation):
    """Bind a saved upstream request to these inputs before reusing its output."""
    from drivebench.benchmarks.environment import reference_identity
    root = Path(config["probe"]["reference_run"]).resolve()
    snapshot = json.loads((root / "config.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    identity = json.loads((root / "model_identity.json").read_text())
    environment = json.loads((root / "environment.json").read_text())
    if (snapshot["model"].get("backend") != "upstream" or snapshot["generation"] != GenerationConfig().to_dict()
            or snapshot["optimizations"] != OptimizationConfig().to_dict()
            or summary.get("successful_requests", 0) != 32 or summary.get("failures") != 0
            or summary.get("oom_attempts") != 0):
        raise ValueError("Reference must be the completed unchanged 32-request upstream development run")
    manifest = Path(config["data"]["manifest"])
    if (snapshot["data"].get("manifest_sha256") != file_sha256(manifest)
            or snapshot["data"].get("manifest_metadata", {}).get("partition_role") != "development"):
        raise ValueError("Probe/reference development selection differs")
    for key in ("checkpoint_sha256", "upstream_commit"):
        if identity.get(key) != config["model"].get(key):
            raise ValueError(f"Reference model differs: {key}")
    if identity.get("parameter_grad_policy") != "preserve_upstream_loader_flags":
        raise ValueError("Reference must use corrected upstream parameter flags")
    rows = [json.loads(line) for line in (root / "measurements.jsonl").read_text().splitlines() if line.strip()]
    matches = [row for row in rows if row.get("request_index") == config["probe"]["request_index"]]
    if len(matches) != 1:
        raise ValueError("Reference request index is absent or duplicated")
    row = matches[0]
    if (row.get("request_id") != request["request_id"] or row.get("scene_id") != record["scene_id"]
            or row.get("status") != "ok" or len(row.get("candidates", [])) != 1):
        raise ValueError("Reference request identity/status differs")
    candidate = row["candidates"][0]
    if (candidate.get("candidate_id") != request["candidate_ids"][0]
            or candidate.get("seed") != request["seeds"][0]
            or candidate.get("trajectory_sha256") != digest(np.asarray(request["trajectories"][0]).tolist())):
        raise ValueError("Reference candidate action or seed differs")
    relative = Path(candidate["frames_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Reference output must stay inside the reference run")
    path = root / relative
    frames = np.load(path, mmap_mode="r", allow_pickle=False)
    if frames.shape != (25, 576, 1024, 3) or frames.dtype != np.float32:
        raise ValueError("Reference tensor layout differs")
    proof = {"run": str(root), "request_index": row["request_index"], "request_id": row["request_id"],
             "output_path": str(path), "output_sha256": file_sha256(path),
             "artifact_sha256": {name: file_sha256(root / name) for name in
                 ("config.json", "summary.json", "model_identity.json", "environment.json", "measurements.jsonl")},
             "runtime_identity": reference_identity(environment), "model_identity": identity,
             "record_sha256": digest(record),
             "scope": "saved upstream development output; no new quality acceptance"}
    return frames, proof


def compiler_state():
    from torch._dynamo.utils import counters
    return {str(category): dict(values) for category, values in counters.items()}


def graph_count(state):
    return state.get("stats", {}).get("unique_graphs", 0)


def unexpected_compiler_fallbacks(state):
    # In Torch2.0.1 these frames stay eager even with suppress_errors=False.
    return {key: value for key, value in state.get("unimplemented", {}).items() if value}


@contextlib.contextmanager
def attention_boundary(enabled):
    if not enabled:
        yield None
        return
    import torch._dynamo
    import xformers.ops
    native = xformers.ops.memory_efficient_attention
    source = Path(inspect.getsourcefile(native) or "")
    if (native.__module__ != "xformers.ops.fmha" or native.__name__ != "memory_efficient_attention"
            or hasattr(native, "_torchdynamo_orig_callable") or file_sha256(source) != NATIVE_SOURCE_SHA256):
        raise ValueError("Native attention source differs from the inspected pinned build")
    wrapper = torch._dynamo.disable(native)
    if getattr(wrapper, "_torchdynamo_orig_callable", None) is not native:
        raise RuntimeError("Dynamo did not preserve the declared native attention callable")
    xformers.ops.memory_efficient_attention = wrapper
    try:
        yield {"kind": "explicit_native_attention_dynamo_boundary", "source_path": str(source),
               "source_sha256": NATIVE_SOURCE_SHA256, "native_operator_unchanged": True}
    finally:
        xformers.ops.memory_efficient_attention = native


@contextlib.contextmanager
def block_events(backend):
    """Nonoverlapping UNet block events; wrappers restore even on exceptions."""
    network = backend.model.model.diffusion_model
    modules = [(f"input_blocks.{i}", module) for i, module in enumerate(network.input_blocks)]
    modules += [("middle_block", network.middle_block)]
    modules += [(f"output_blocks.{i}", module) for i, module in enumerate(network.output_blocks)]
    records, originals = [], []
    try:
        for name, module in modules:
            original = module.forward
            originals.append((module, original))
            def measured(*args, _name=name, _forward=original, **kwargs):
                begin, end = (backend.torch.cuda.Event(enable_timing=True) for _ in range(2))
                begin.record()
                try:
                    return _forward(*args, **kwargs)
                finally:
                    end.record()
                    records.append((_name, begin, end))
            module.forward = measured
        yield records
    finally:
        for module, original in originals:
            module.forward = original


def summarize_blocks(records, denoising_s):
    values = {}
    for name, begin, end in records:
        entry = values.setdefault(name, {"calls": 0, "cuda_event_s": 0.0})
        entry["calls"] += 1
        entry["cuda_event_s"] += begin.elapsed_time(end) / 1000
    for entry in values.values():
        entry["fraction_of_instrumented_denoising"] = entry["cuda_event_s"] / denoising_s
    return {"instrumented": True, "blocks": values, "denoising_gpu_s": denoising_s,
            "scope": "exclusive block CUDA-event spans; hooks add overhead; not an unprofiled benchmark",
            "middle_block_upper_bound_note": "Even perfect middle-block elimination saves at most its measured fraction; reuse saves less."}


def cuda_graph_evidence(trace):
    import ijson
    launches, captures = 0, 0
    with Path(trace).open("rb") as stream:
        for event in ijson.items(stream, "traceEvents.item", use_float=True):
            if event.get("cat") != "cuda_runtime":
                continue
            name = event.get("name", "")
            launches += int(name.startswith("cudaGraphLaunch"))
            captures += int(name.startswith("cudaGraphInstantiate"))
    return {"cuda_graph_launches": launches, "cuda_graph_instantiations": captures,
            "cuda_graph_replay_observed": launches > 0, "trace_sha256": file_sha256(trace),
            "scope": "separate instrumented warm request; compile mode alone does not establish graph replay"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Fresh attempt directory; never overwrite an earlier result")
    args = parser.parse_args(argv)
    config, generation, options = inspect_config(args.config)
    declared_output = args.output or Path(config["output"])
    if declared_output.is_symlink():
        raise FileExistsError("Refusing an output-directory symlink")
    output = declared_output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "status": "running", "started_at": now(), "config": config,
              "config_sha256": file_sha256(args.config), "helper_sha256": file_sha256(__file__),
              "scope": "one development request discovery; no held-out or quality-acceptance claim",
              "headline_eligible": False, "quality_validated": False, "executions": [],
              "timing_scope": "synchronized context preparation + generation; file decode and output export excluded",
              "strict_atol": 0, "strict_rtol": 0, "external_timeout_seconds": 600}
    report_path = output / "probe.json"
    save(report_path, report)
    roots = [output / "cache/torchinductor", output / "cache/triton", output / "compiler-debug"]
    old_env = {key: os.environ.get(key) for key in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "TORCH_COMPILE_DEBUG")}
    os.environ.update(TORCHINDUCTOR_CACHE_DIR=str(roots[0]), TRITON_CACHE_DIR=str(roots[1]), TORCH_COMPILE_DEBUG="1")
    handlers = {}
    def interrupted(signum, _frame):
        report["termination_signal"] = signum
        raise SystemExit(128 + signum)
    for sig in (signal.SIGTERM, signal.SIGINT):
        handlers[sig] = signal.signal(sig, interrupted)
    compiled = options.compile_denoiser or options.compile_decoder
    exit_code = 1
    try:
        from drivebench.benchmark import _validate_reference_environment
        from drivebench.benchmarks.environment import collect_environment, reference_identity, source_identity
        from drivebench.data import inference_request, load_manifest, read_preprocess, verify_input_checksums
        from drivebench.engine import InferenceEngine
        from drivebench.validate import compare_outputs, input_provenance
        import torch
        import torch._dynamo
        import torch._inductor.config
        if torch.__version__ != "2.0.1+cu118":
            raise ValueError("Probe is audited for the pinned Torch2.0.1+cu118 runtime")
        if torch._dynamo.config.disable:
            raise ValueError("Dynamo is disabled")
        torch._dynamo.config.suppress_errors = False
        torch._dynamo.config.verbose = True
        torch._dynamo.config.print_graph_breaks = True
        torch._dynamo.config.log_level = logging.INFO
        torch._dynamo.config.debug_dir_root = str(roots[2])
        torch._inductor.config.trace.enabled = True
        environment = collect_environment()
        save(output / "environment.json", environment)
        report["environment_gate"] = _validate_reference_environment(config, environment)
        report["implementation_sha256"] = source_identity()
        records = load_manifest(config["data"]["manifest"])
        record = records[config["probe"]["request_index"]]
        verify_input_checksums(record, config["data"]["root"], include_future=False)
        request = inference_request(record, config["data"]["root"], candidate_count=1)
        frames = read_preprocess(request["observed_frames"], generation.height, generation.width)
        reference, report["reference"] = load_reference(config, request, record, generation)
        if report["reference"]["runtime_identity"] != reference_identity(environment):
            raise ValueError("Probe/reference GPU or runtime differs")
        report["input_provenance"] = input_provenance(config["data"]["manifest"], records, record, request, frames,
                                                     request_index=config["probe"]["request_index"])
        report["compiler_before"] = compiler_state()
        report["compiler_settings"] = {"suppress_errors": False, "cache_size_limit": torch._dynamo.config.cache_size_limit,
                                       "fullgraph": options.fullgraph, "dynamic_shapes": options.dynamic_shapes,
                                       "mode": options.compile_mode}
        with attention_boundary(options.compile_denoiser) as patch:
            report["runtime_patch"] = patch
            report["runtime_implementation_sha256"] = digest({"source": report["implementation_sha256"],
                                                               "helper": report["helper_sha256"], "patch": patch})
            start = time.perf_counter()
            engine = InferenceEngine(config["model"], generation, options)
            engine.synchronize()
            report["model_load_s"] = time.perf_counter() - start
            report["model_identity"] = engine.backend.identity
            if engine.backend.identity["inference_config_sha256"] != report["reference"]["model_identity"]["inference_config_sha256"]:
                raise ValueError("Reference model inference config differs")

            def execute(label, *, instrumented=False, selected_generation=None):
                selected_generation = selected_generation or generation
                nonlocal engine
                # Shape-bucket discovery reuses this exact compiled callable and
                # model, but gives each shape a fresh context/cache identity.
                if engine.generation != selected_generation:
                    engine.backend.generation = selected_generation
                    engine = InferenceEngine(config["model"], selected_generation, options, backend=engine.backend)
                observed = read_preprocess(request["observed_frames"], selected_generation.height, selected_generation.width)
                engine.cache.clear()
                engine.reset_peak_memory()
                engine.synchronize()
                begin = time.perf_counter()
                context = engine.prepare_context(observed)
                result = engine.generate(context, request["trajectories"], request["seeds"], candidate_ids=request["candidate_ids"])
                engine.synchronize()
                elapsed = time.perf_counter() - begin
                row = {"label": label, "wall_s_excluding_file_decode_and_export": elapsed,
                       "instrumented": instrumented, "status": result.status,
                       "effective_config": result.effective_config, "stage_timings_s": result.stage_timings_s,
                       "peak_allocated_bytes": result.peak_allocated_bytes, "peak_reserved_bytes": result.peak_reserved_bytes,
                       "denoiser_evaluations": result.denoiser_evaluations, "cache_stats": result.cache_stats,
                       "fallbacks": result.fallbacks, "compiler": compiler_state()}
                report["executions"].append(row)
                save(report_path, report)
                if result.status != "ok" or result.cache_stats.get("oom_attempts"):
                    raise RuntimeError("Probe failed or retried after OOM; preserve this attempt")
                path = output / f"{label}.npy"
                np.save(path, result.frames[request["candidate_ids"][0]], allow_pickle=False)
                row.update(output_path=str(path), output_sha256=file_sha256(path))
                if (selected_generation.height, selected_generation.width) == (576, 1024):
                    row["strict_comparison"] = compare_outputs({request["candidate_ids"][0]: reference}, result.frames, atol=0, rtol=0)
                else:
                    row["strict_comparison"] = None
                    row["scope"] = "non-primary spatial bucket discovery; no matching reference/quality claim"
                save(report_path, report)
                if compiled and unexpected_compiler_fallbacks(row["compiler"]):
                    raise RuntimeError("Additional eager fallback frames observed; see unimplemented compiler counters")
                return row

            cold = execute("cold")
            report["cold_note"] = "First request includes lazy compilation; model load is separate; not compile-only time"
            if config["probe"]["kind"].startswith("shape_"):
                report["shape_bucket_policy"] = {"sizes": [[576, 1024], [512, 896], [576, 1024]],
                                                  "compiled_callable_reused": True, "primary_generation_unchanged": True}
                execute("bucket-512x896", selected_generation=replace(generation, height=512, width=896))
                execute("bucket-primary-return", selected_generation=generation)
            else:
                stable, previous = 0, graph_count(cold["compiler"])
                for i in range(config["probe"]["max_warmups"]):
                    warm = execute(f"warmup-{i}")
                    current = graph_count(warm["compiler"])
                    stable = stable + 1 if current == previous else 0
                    previous = current
                    if not compiled or stable >= 2:
                        break
                if compiled and stable < 2:
                    raise RuntimeError("Compilation did not stabilize within the declared warmup budget")
                for i in range(config["probe"]["measurements"]):
                    if config["probe"]["kind"] == "block_timing":
                        with block_events(engine.backend) as events:
                            row = execute(f"blocks-{i}", instrumented=True)
                        report["block_profile"] = summarize_blocks(events, row["stage_timings_s"]["denoising_gpu"])
                        if any(value["calls"] != generation.steps for value in report["block_profile"]["blocks"].values()):
                            raise RuntimeError("Block event call coverage differs from the sampler step count")
                    else:
                        row = execute(f"measured-{i}")
                        if compiled and graph_count(row["compiler"]) != previous:
                            raise RuntimeError("Compilation occurred during the measured request")
                if config["probe"].get("cuda_graph_trace"):
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profiler:
                        execute("cuda-graph-instrumented", instrumented=True)
                    trace = output / "cuda-graph-trace.json"
                    profiler.export_chrome_trace(str(trace))
                    report["cuda_graph_evidence"] = cuda_graph_evidence(trace)
        report["status"] = "execution_complete"
        exit_code = 0
    except BaseException as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        if isinstance(error, SystemExit) and isinstance(error.code, int):
            exit_code = error.code or 1
        traceback.print_exc()
    finally:
        # Persist termination before slower cache hashing; timeout may kill after
        # the external grace period. Existing logs/outputs are never deleted.
        save(report_path, report)
        try:
            if "torch._dynamo" in sys.modules:
                report["compiler_after"] = compiler_state()
            generated, artifacts, binaries = generated_records(roots, {})
            report.update(generated_code=generated, compiler_artifacts=artifacts, cuda_binaries=binaries)
            report["new_compiled_graphs"] = graph_count(report.get("compiler_after", {})) - graph_count(report.get("compiler_before", {}))
            report["positive_cuda_codegen_evidence"] = (report["new_compiled_graphs"] > 0 and bool(binaries)
                                                        and any(x["contains_triton_source"] for x in generated))
            if compiled and not report["positive_cuda_codegen_evidence"] and exit_code == 0:
                report["status"], exit_code = "no_positive_cuda_codegen", 1
            comparisons = [row["strict_comparison"] for row in report["executions"] if row.get("strict_comparison") is not None]
            report["strict_passed"] = bool(comparisons) and all(item["passed"] for item in comparisons)
            if exit_code == 0 and not report["strict_passed"]:
                report["status"], exit_code = "execution_complete_strict_failed", 2
            if "implementation_sha256" in report:
                from drivebench.benchmarks.environment import source_identity
                report["source_unchanged"] = report["implementation_sha256"] == source_identity()
                report["launcher_and_config_unchanged"] = (report["helper_sha256"] == file_sha256(__file__)
                                                            and report["config_sha256"] == file_sha256(args.config)
                                                            and report["config"] == load_config(args.config))
                if not report["source_unchanged"] or not report["launcher_and_config_unchanged"]:
                    report["status"], exit_code = "source_changed_during_probe", 1
        except BaseException as error:
            report["finalization_error"] = {"type": type(error).__name__, "message": str(error)}
            exit_code = 1
        report.update(finished_at=now(), exit_code=exit_code)
        save(report_path, report)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print(json.dumps({"status": report["status"], "exit_code": exit_code, "report": str(report_path)}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
