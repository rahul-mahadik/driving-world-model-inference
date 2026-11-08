#!/usr/bin/env python3
"""Bounded strict compiled-denoiser diagnostic against the corrected eager baseline.

Run in the verified container under `timeout --kill-after=30s 600s ...`.
This helper never widens tolerances, retries, or turns off compiler errors.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import signal
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drivebench.config import digest, file_sha256, load_config, validate_run_config


def now():
    return datetime.now(timezone.utc).isoformat()


def inspect_config(path, base_path=None):
    path = Path(path).resolve()
    base_path = Path(base_path).resolve() if base_path else path.with_name("preflight_scene0103_corrected.yaml")
    # Reuse the fixed scene/checkpoint/selection/tolerance contract. A different
    # base location is explicit; execution additionally requires corrected
    # gradient-mode evidence, so an old frozen-weight report cannot qualify.
    from scripts.preflight_real_clip import read_config
    base, config = read_config(base_path), load_config(path)
    baseline_generation, baseline_options = validate_run_config(base)
    generation, options = validate_run_config(config)
    from drivebench.types import GenerationConfig, OptimizationConfig
    if baseline_generation != GenerationConfig() or baseline_options != OptimizationConfig():
        raise ValueError("The corrected preflight generation/options have changed")
    if generation != baseline_generation or options != replace(baseline_options, compile_denoiser=True):
        raise ValueError("Only compile_denoiser=True may change; retain default/static/partial capture")
    for key in ("model", "data", "environment_lock", "environment_verification"):
        if config.get(key) != base.get(key):
            raise ValueError(f"Compiled diagnostic changed the inherited {key}")
    if config.get("numerical_validation") is not None:
        raise ValueError("This first compile diagnostic cannot reuse eager numerical evidence")
    expected_benchmark = {**base["benchmark"], "headline": False, "requests": 1,
                          "warmup_requests": 2, "max_warmup_requests": 3}
    if config.get("benchmark") != expected_benchmark:
        raise ValueError("Compiled diagnostic workload/warmup differs from the bounded declaration")
    expected_preflight = {**base["preflight"],
        "scope": config["preflight"].get("scope"),
        "validation_output": config["preflight"].get("validation_output")}
    if config.get("preflight") != expected_preflight:
        raise ValueError("Compiled diagnostic changed frozen clip identity or strict tolerances")
    if config["preflight"].get("atol") != 0 or config["preflight"].get("rtol") != 0:
        raise ValueError("First compile comparison requires atol=rtol=0")
    declared_output = Path(config["output"]).absolute()
    if declared_output.is_symlink():
        raise FileExistsError("Refusing an output-directory symlink")
    output = declared_output.resolve()
    report = Path(config["preflight"]["validation_output"]).resolve()
    if report != output / "strict-agreement.json" or output == Path(base["output"]).resolve():
        raise ValueError("Use a separate output directory and its strict-agreement.json")
    return config, base_path, output, report


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def compiler_state():
    from torch._dynamo.utils import counters
    return {str(category): dict(counts) for category, counts in counters.items()}


def codegen_evidence(root):
    records = []
    for path in sorted(root.rglob("output_code.py")):
        content = path.read_text()
        records.append({"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size,
                        "contains_triton_kernel": "async_compile.triton(" in content or "@triton.jit" in content})
    return records


def baseline_input_evidence(config, base_path):
    """Bind this compile diagnostic to the already-passed real clip, without arrays."""
    from drivebench.data.manifest import load_manifest
    base = load_config(base_path)
    prior_path = Path(base["preflight"]["validation_output"])
    prior = json.loads(prior_path.read_text())
    expected = prior.get("input_provenance", {})
    declaration = config["preflight"]
    if (prior.get("passed") is not True or prior.get("mode") != "agreement"
            or prior.get("atol") != 0 or prior.get("rtol") != 0
            or prior.get("config") != base
            or prior.get("request_id") != declaration["request_id"]
            or expected.get("request_id") != declaration["request_id"]
            or expected.get("scene_id") != declaration["scene_id"]
            or expected.get("synthetic") is not False
            or len(expected.get("candidate_ids", [])) != 1
            or prior.get("input_provenance_sha256") != digest(expected)):
        raise ValueError("Corrected strict real-clip baseline evidence differs or did not pass")
    if (prior.get("schema_version") != 2 or prior.get("validated_request_count") != 1
            or prior.get("generation") != base["generation"]
            or prior.get("optimizations") != validate_run_config(base)[1].to_dict()
            or prior.get("checks", {}).get("upstream_agreement", {}).get("passed") is not True):
        raise ValueError("Corrected baseline lacks the declared strict agreement check")
    identities = {}
    for label, backend in (("upstream", "upstream"), ("refactored", "vista")):
        run = prior.get("runs", {}).get(label, {})
        identity = run.get("model_identity", {})
        inventory = identity.get("parameter_dtypes", {}).get("torch.float32", {})
        if (identity.get("backend") != backend
                or identity.get("parameter_grad_policy") != "preserve_upstream_loader_flags"
                or identity.get("execution_grad_mode") != "no_grad"
                or identity.get("autocast_cache_enabled") is not True
                or not 0 < inventory.get("requires_grad_tensors", 0) < inventory.get("tensors", 0)
                or not 0 < inventory.get("requires_grad_elements", 0) < inventory.get("elements", 0)
                or run.get("candidate_count") != 1
                or run.get("denoiser_evaluations") != base["generation"]["steps"]):
            raise ValueError("Baseline must preserve upstream parameter flags and no_grad autocast caching")
        identities[label] = identity
    for key in ("parameter_dtypes", "parameter_grad_policy", "execution_grad_mode", "autocast_cache_enabled"):
        if identities["upstream"][key] != identities["refactored"][key]:
            raise ValueError("Corrected baseline parameter/execution inventories differ")
    manifest = Path(config["data"]["manifest"])
    metadata_path = manifest.with_suffix(manifest.suffix + ".meta.json")
    records = load_manifest(manifest)
    metadata = json.loads(metadata_path.read_text())
    if (len(records) != 1 or records[0].get("request_id") != declaration["request_id"]
            or records[0].get("scene_id") != declaration["scene_id"]
            or records[0].get("preflight_only") is not True
            or digest(records[0]) != expected.get("record_sha256")
            or file_sha256(manifest) != expected.get("manifest_sha256")
            or file_sha256(metadata_path) != expected.get("manifest_metadata_sha256")
            or metadata.get("preflight_config_sha256") != file_sha256(base_path)):
        raise ValueError("Scene-0103 manifest changed after the strict baseline")
    return {"path": str(prior_path), "sha256": file_sha256(prior_path),
            "input_provenance": expected, "runtime_identity": prior["runtime_identity"],
            "implementation_sha256": prior["implementation_sha256"],
            "corrected_model_identities": identities}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/preflight_scene0103_corrected_compiled.yaml"))
    parser.add_argument("--base-config", type=Path,
                        help="Corrected eager config; defaults to preflight_scene0103_corrected.yaml beside --config")
    args = parser.parse_args(argv)
    config, base_path, output, report_path = inspect_config(args.config, args.base_config)
    # Atomic creation rejects existing files, directories and symlinks, including
    # an interrupted attempt. Never remove them automatically for another try.
    output.mkdir(parents=True, exist_ok=False)
    attempt_path = output / "attempt.json"
    debug_root = output / "compiler-debug"
    validation_args = ["--config", str(args.config.resolve()), "--mode", "agreement",
                       "--request-index", "0", "--candidates", "1", "--atol", "0", "--rtol", "0",
                       "--output", str(report_path)]
    attempt = {"schema_version": 1, "status": "running", "started_at": now(), "pid": os.getpid(),
               "config_sha256": file_sha256(args.config), "base_config": str(base_path),
               "base_config_sha256": file_sha256(base_path),
               "helper_sha256": file_sha256(__file__), "validation_argv": validation_args,
               "headline_eligible": False, "quality_validated": False,
               "external_timeout_required": {"seconds": 600, "kill_after_seconds": 30},
               "compiler_debug_root": str(debug_root),
               "compile_cache_environment": {key: os.environ.get(key) for key in
                   ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")}}
    save(attempt_path, attempt)
    old_handlers = {}

    def interrupted(signum, _frame):
        attempt["termination_signal"] = signum
        raise SystemExit(128 + signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        old_handlers[signum] = signal.signal(signum, interrupted)
    exit_code = 1
    try:
        attempt["baseline_evidence"] = baseline_input_evidence(config, base_path)
        save(attempt_path, attempt)
        # These switches exist in the pinned PyTorch 2.0.1 source. Set the env
        # before importing its compiler modules so Inductor retains output code.
        os.environ["TORCH_COMPILE_DEBUG"] = "1"
        import torch
        import torch._dynamo
        import torch._inductor.config
        from drivebench.benchmark import _validate_reference_environment
        from drivebench.benchmarks.environment import collect_environment, reference_identity, source_identity
        from drivebench.validate import main as validate

        if torch.__version__ != "2.0.1+cu118":
            raise ValueError("This diagnostic launcher is audited for Torch2.0.1+cu118 only")
        if torch._dynamo.config.disable:
            raise ValueError("TorchDynamo is disabled; refusing an eager-only compile diagnostic")
        torch._dynamo.config.suppress_errors = False
        torch._dynamo.config.verbose = True
        torch._dynamo.config.print_graph_breaks = True
        torch._dynamo.config.log_level = logging.INFO
        torch._dynamo.config.debug_dir_root = str(debug_root)
        torch._inductor.config.trace.enabled = True
        if torch._dynamo.config.suppress_errors is not False:
            raise RuntimeError("Compiler errors must remain visible")
        environment = collect_environment()
        save(output / "environment-before.json", environment)
        attempt["environment_gate"] = _validate_reference_environment(config, environment)
        attempt["runtime_before"] = reference_identity(environment)
        attempt["source_before"] = source_identity()
        if (attempt["runtime_before"] != attempt["baseline_evidence"]["runtime_identity"]
                or attempt["source_before"] != attempt["baseline_evidence"]["implementation_sha256"]):
            raise ValueError("Runtime or implementation changed after the strict eager baseline")
        attempt["compiler_before"] = compiler_state()
        attempt["compiler_settings"] = {"suppress_errors": torch._dynamo.config.suppress_errors,
            "disable": torch._dynamo.config.disable, "debug_dir_root": torch._dynamo.config.debug_dir_root,
            "inductor_trace_enabled": torch._inductor.config.trace.enabled}
        save(attempt_path, attempt)
        exit_code = validate(validation_args)
        attempt["validation_exit_code"] = exit_code
    except BaseException as error:
        exit_code = error.code if isinstance(error, SystemExit) and isinstance(error.code, int) else 1
        if not exit_code:
            exit_code = 1
        attempt["error"] = {"type": type(error).__name__, "message": str(error),
                            "traceback": traceback.format_exc()}
        attempt.update(status="failed", exit_code=exit_code)
        save(attempt_path, attempt)
        traceback.print_exc()
    finally:
        try:
            if "torch._dynamo" in sys.modules:
                attempt["compiler_after"] = compiler_state()
            # Persist fast evidence before any subprocess query. TERM's grace
            # period may be shorter than a stalled environment collector.
            attempt["status"] = "failed" if exit_code else "finalizing"
            save(attempt_path, attempt)
            if "source_before" in attempt and "termination_signal" not in attempt:
                from drivebench.benchmarks.environment import collect_environment, reference_identity, source_identity
                after = collect_environment()
                save(output / "environment-after.json", after)
                attempt["runtime_after"] = reference_identity(after)
                attempt["source_after"] = source_identity()
                attempt["source_and_runtime_unchanged"] = (
                    attempt["source_before"] == attempt["source_after"]
                    and attempt["runtime_before"] == attempt["runtime_after"]
                    and attempt["config_sha256"] == file_sha256(args.config)
                    and attempt["base_config_sha256"] == file_sha256(base_path)
                    and attempt["helper_sha256"] == file_sha256(__file__)
                    and attempt["baseline_evidence"]["sha256"] == file_sha256(attempt["baseline_evidence"]["path"]))
            generated = codegen_evidence(debug_root)
            attempt["generated_code"] = generated
            before_graphs = attempt.get("compiler_before", {}).get("stats", {}).get("unique_graphs", 0)
            after_graphs = attempt.get("compiler_after", {}).get("stats", {}).get("unique_graphs", 0)
            attempt["new_compiled_graphs"] = after_graphs - before_graphs
            attempt["positive_cuda_codegen_evidence"] = (
                after_graphs > before_graphs and any(item["contains_triton_kernel"] for item in generated))
            report = json.loads(report_path.read_text()) if report_path.is_file() else {}
            attempt["numerical_passed"] = (report.get("passed") is True
                and report.get("atol") == 0 and report.get("rtol") == 0
                and report.get("request_id") == config["preflight"]["request_id"]
                and report.get("input_provenance") == attempt.get("baseline_evidence", {}).get("input_provenance")
                and report.get("implementation_sha256") == attempt.get("source_before")
                and report.get("runtime_identity") == attempt.get("runtime_before"))
            if report_path.is_file():
                attempt["numerical_report_sha256"] = file_sha256(report_path)
            passed = (exit_code == 0 and attempt["numerical_passed"]
                      and attempt.get("source_and_runtime_unchanged") is True
                      and attempt["positive_cuda_codegen_evidence"])
            if not passed and exit_code == 0:
                exit_code = 1
            attempt["status"] = "passed" if passed else "failed"
        except BaseException as error:
            exit_code = 1
            attempt["status"] = "failed"
            attempt["finalization_error"] = {"type": type(error).__name__, "message": str(error)}
            traceback.print_exc()
        attempt.update(finished_at=now(), exit_code=exit_code)
        save(attempt_path, attempt)
        for signum, previous in old_handlers.items():
            signal.signal(signum, previous)
    print(json.dumps({"status": attempt["status"], "exit_code": exit_code, "attempt": str(attempt_path)}, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
