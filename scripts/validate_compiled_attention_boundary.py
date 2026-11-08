#!/usr/bin/env python3
"""Strict bounded experiment: compile the denoiser around native xFormers attention.

Use timeout --signal=TERM --kill-after=30s 600s with a fresh output directory.
This launcher changes only a process-local Dynamo boundary. It preserves the
native operator, gradient policy, source package, and zero numerical tolerances.
"""
from __future__ import annotations

import argparse
import copy
import inspect
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drivebench.config import digest, file_sha256
from scripts import validate_compiled_preflight as base


TARGET = "xformers.ops.memory_efficient_attention"
NATIVE_SOURCE_SHA256 = "6ee2ed49dfd70c3ccce4e046c3d15c2da35e5963f77eaa591627883a11f08fb9"
SCOPE = "Dynamo disabled only during native memory_efficient_attention calls; upstream and refactored diagnostic paths"


def patch_declaration():
    return {"schema_version": 1, "target": TARGET, "kind": "explicit_dynamo_attention_boundary",
            "scope": SCOPE, "launcher_sha256": file_sha256(__file__),
            "native_source_sha256": NATIVE_SOURCE_SHA256,
            "prior_upstream_values_must_match": True, "suppress_errors": False}


def cache_snapshot(roots):
    result = {}
    for root in roots:
        for path in sorted(root.rglob("*")) if root.exists() else []:
            if path.is_symlink():
                raise ValueError("Compiler cache must not contain symlinks")
            if path.is_file():
                result[str(path)] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
    return result


def generated_records(roots, before):
    after = cache_snapshot(roots)
    changed = {path: info for path, info in after.items() if before.get(path) != info}
    binaries = [path for path, info in changed.items() if info["bytes"] and Path(path).suffix in {".cubin", ".ptx"}]
    records = []
    for name, info in changed.items():
        path = Path(name)
        if path.suffix != ".py":
            continue
        source = path.read_text()
        marker = "async_compile.triton(" in source or "@triton.jit" in source
        records.append({"path": name, **info, "contains_triton_source": marker,
                        "contains_triton_kernel": marker and bool(binaries)})
    return records, after, binaries


def verify_prior_outputs(prior):
    run = prior["runs"]["upstream"]
    path = Path(run["outputs"]["normal"])
    expected = run["outputs_sha256"]["normal"]
    if file_sha256(path) != expected:
        raise ValueError("Corrected prior upstream array archive changed")
    return path, expected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/preflight_scene0103_corrected_compiled_attention_boundary.yaml"))
    parser.add_argument("--base-config", type=Path, default=Path("configs/preflight_scene0103_corrected.yaml"))
    args = parser.parse_args(argv)
    config, base_path, output, report_path = base.inspect_config(args.config, args.base_config)
    declaration = patch_declaration()
    if config.get("runtime_patch") != declaration:
        raise ValueError("Runtime patch declaration or launcher/native source hash differs")
    if output.exists() or output.is_symlink():
        raise FileExistsError("Preserve the previous attempt; use a fresh declared output")
    roots = [output / "cache/torchinductor", output / "cache/triton"]
    # No cache reuse: Torch 2.0 moves completed .debug folders into its code cache.
    # Nest both fresh caches under this attempt so final scanning follows them.
    cache_vars = dict(zip(("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"), map(str, roots)))
    previous_env = {name: os.environ.get(name) for name in cache_vars}
    os.environ.update(cache_vars)
    from drivebench import validate
    from drivebench.benchmarks.environment import source_identity
    original_main, original_run = validate.main, validate._run
    original_scan = base.codegen_evidence
    original_save = base.save
    state = {"schema_version": 1, "runtime_patch": declaration,
             "unpatched_package_sha256": source_identity(), "cache_roots": list(map(str, roots)),
             "cache_before": {}, "status": "waiting_for_environment_gate"}
    state["runtime_implementation_sha256"] = digest({
        "unpatched_package_sha256": state["unpatched_package_sha256"], "runtime_patch": declaration})
    receipt = output / "runtime-patch.json"

    def bound_save(path, value):
        if Path(path) == output / "attempt.json":
            value = {**value, "runtime_patch": declaration,
                     "runtime_implementation_sha256": state["runtime_implementation_sha256"],
                     "unpatched_package_sha256": state["unpatched_package_sha256"]}
            if receipt.is_file():
                value["runtime_patch_receipt_sha256"] = file_sha256(receipt)
        return original_save(path, value)

    def save_state():
        base.save(receipt, state)

    def codegen_scan(debug_root):
        records, after, binaries = generated_records([debug_root, *roots], state["cache_before"])
        state.update(cache_after=after, cuda_compiler_artifacts=binaries,
                     generated_code=records)
        try:
            from torch._dynamo.utils import compile_times
            state["compiler_phase_timings"] = compile_times()
            state["compiler_timing_note"] = "Torch-reported phase timings may overlap; not end-to-end request latency"
        except (ImportError, AttributeError):
            state["compiler_phase_timings"] = None
        save_state()
        return records

    def guarded_validation(validation_args):
        # The existing launcher calls this only after environment and source
        # identity match the corrected strict baseline. No GPU job starts above.
        import numpy as np
        import torch._dynamo
        import xformers.ops
        baseline = base.baseline_input_evidence(config, base_path)
        prior = json.loads(Path(baseline["path"]).read_text())
        prior_array, prior_sha = verify_prior_outputs(prior)
        native = xformers.ops.memory_efficient_attention
        native_file = Path(inspect.getsourcefile(native) or "")
        if (native.__module__ != "xformers.ops.fmha" or native.__name__ != "memory_efficient_attention"
                or file_sha256(native_file) != NATIVE_SOURCE_SHA256
                or hasattr(native, "_torchdynamo_orig_callable")):
            raise ValueError("Native attention callable is not the pinned unpatched implementation")
        if torch._dynamo.config.suppress_errors is not False:
            raise ValueError("Compiler errors must remain enabled")
        boundary = torch._dynamo.disable(native)
        if getattr(boundary, "_torchdynamo_orig_callable", None) is not native:
            raise RuntimeError("Dynamo boundary did not preserve the native callable")
        state.update(status="running", native_source_path=str(native_file),
                     baseline_report_sha256=baseline["sha256"], prior_upstream_archive=str(prior_array),
                     prior_upstream_archive_sha256=prior_sha,
                     cache_before=cache_snapshot([output / "compiler-debug", *roots]))
        save_state()
        started = time.perf_counter()

        def patched_run(*run_args, **run_kwargs):
            outputs, details = original_run(*run_args, **run_kwargs)
            details["model_identity"] = {**details["model_identity"],
                "runtime_patch": copy.deepcopy(declaration),
                "runtime_implementation_sha256": state["runtime_implementation_sha256"]}
            if run_kwargs.get("backend") == "upstream":
                with np.load(prior_array, allow_pickle=False) as saved:
                    agreement = validate.compare_outputs(saved, outputs["normal"], atol=0, rtol=0)
                state["patched_upstream_matches_corrected_prior"] = agreement
                save_state()
                if not agreement["passed"]:
                    raise RuntimeError("Patched upstream differs from corrected prior; refusing compiled comparison")
            return outputs, details

        validate._run = patched_run
        xformers.ops.memory_efficient_attention = boundary
        try:
            result = original_main(validation_args)
            report = json.loads(report_path.read_text())
            report.update(runtime_patch=copy.deepcopy(declaration),
                runtime_implementation_sha256=state["runtime_implementation_sha256"],
                unpatched_package_sha256=state["unpatched_package_sha256"],
                patched_upstream_matches_corrected_prior=state["patched_upstream_matches_corrected_prior"],
                headline_eligible=False,
                scope="one-clip explicit native-attention Dynamo boundary experiment; strict numerical checks only")
            base.save(report_path, report)
            state["status"] = "numerical_passed" if result == 0 else "numerical_failed"
            return result
        except BaseException as error:
            state.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
            raise
        finally:
            xformers.ops.memory_efficient_attention = native
            validate._run = original_run
            state["cold_validation_wall_s"] = time.perf_counter() - started
            state["cold_validation_note"] = "Includes model loading, compilation and strict diagnostic generation; not pure compilation time"
            state["native_callable_restored"] = xformers.ops.memory_efficient_attention is native
            save_state()
            if (file_sha256(__file__) != declaration["launcher_sha256"]
                    or file_sha256(native_file) != NATIVE_SOURCE_SHA256):
                raise RuntimeError("Runtime patch launcher or native source changed during validation")

    validate.main, base.codegen_evidence, base.save = guarded_validation, codegen_scan, bound_save
    try:
        code = base.main(["--config", str(args.config), "--base-config", str(base_path)])
        attempt_path = output / "attempt.json"
        attempt = json.loads(attempt_path.read_text())
        attempt.update(runtime_patch=declaration, runtime_patch_receipt_sha256=file_sha256(receipt),
                       runtime_implementation_sha256=state["runtime_implementation_sha256"],
                       unpatched_package_sha256=state["unpatched_package_sha256"])
        base.save(attempt_path, attempt)
        return code
    finally:
        validate.main, validate._run, base.codegen_evidence, base.save = original_main, original_run, original_scan, original_save
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


if __name__ == "__main__":
    raise SystemExit(main())
