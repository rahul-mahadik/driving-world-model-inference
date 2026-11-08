"""Opt-in GPU checks for upstream agreement, isolation, and exact cache reuse.

Numerical tolerances are mandatory inputs, not defaults selected after seeing
evaluation results. A passing check does not establish generation quality.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path

import numpy as np

from drivebench.benchmarks.environment import collect_environment, reference_identity, source_identity
from drivebench.benchmarks.agreement import candidate_batching_coverage
from drivebench.config import digest, file_sha256, load_config, validate_run_config
from drivebench.data.manifest import inference_request, load_manifest, verify_input_checksums
from drivebench.data.preprocess import read_preprocess
from drivebench.engine import InferenceEngine
from drivebench.types import OptimizationConfig


def _array_sha256(value) -> str:
    """Hash the values actually sent to inference, including their layout contract."""
    array = np.ascontiguousarray(value)
    header = json.dumps({"dtype": array.dtype.str, "shape": array.shape}, sort_keys=True).encode()
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def input_provenance(manifest_path, records, record, request, frames, *, request_index):
    """Record observed-only inference inputs without reading evaluation frames."""
    manifest_path = Path(manifest_path).resolve()
    metadata_path = manifest_path.with_suffix(manifest_path.suffix + ".meta.json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    manifest_hash = file_sha256(manifest_path)
    scenes = sorted({item["scene_id"] for item in records})
    if metadata and metadata.get("manifest_sha256") != manifest_hash:
        raise ValueError("Validation manifest metadata checksum does not match the manifest")
    if metadata and metadata.get("scene_ids") != scenes:
        raise ValueError("Validation manifest metadata scene IDs do not match the manifest")
    return {
        "manifest_path": str(manifest_path), "manifest_sha256": manifest_hash,
        "manifest_metadata_sha256": file_sha256(metadata_path) if metadata else None,
        "manifest_scene_ids": scenes, "partition_role": metadata.get("partition_role"),
        "split": metadata.get("split"), "source_split": record.get("source_split"),
        "synthetic": metadata.get("synthetic", record.get("trajectory_metadata", {}).get("synthetic", False)),
        "request_index": request_index, "request_id": request["request_id"], "scene_id": request["scene_id"],
        "record_sha256": digest(record),
        "observed_sha256": _array_sha256(frames),
        "observed_file_sha256": [file_sha256(path) for path in request["observed_frames"]],
        "actions_sha256": [_array_sha256(np.asarray(action, dtype=np.float32).reshape(4, 2))
                           for action in request["trajectories"]],
        "candidate_ids": request["candidate_ids"], "seeds": request["seeds"],
    }


def cache_coverage(details):
    """Equal outputs establish reuse only when the advertised cache paths ran."""
    stats = details["cache_stats"]
    checks = {
        "image_embeddings_reused": stats.get("conditioning_embedding_hits", 0) > 0,
        "posterior_parameters_reused": stats.get("posterior_parameter_hits", 0) > 0,
        "cached_both_image_encoders": set(stats.get("conditioning_cached_embedders", []))
        == {"cond_frames", "cond_frames_without_noise"},
    }
    return {**checks, "passed": all(checks.values())}


def compare_outputs(reference, actual, *, atol: float, rtol: float) -> dict:
    """Bounded-memory allclose/max-absolute/RMSE diagnostics for host outputs."""
    if not np.isfinite([atol, rtol]).all() or min(atol, rtol) < 0:
        raise ValueError("Numerical tolerances must be finite and nonnegative")
    if set(reference) != set(actual) or not reference:
        raise ValueError("Reference and actual candidate IDs must match and be nonempty")
    comparisons = {}
    for key in reference:
        left, right = np.asarray(reference[key]), np.asarray(actual[key])
        if left.shape != right.shape or not left.size:
            raise ValueError(f"Output shape mismatch or empty output for {key}")
        left, right = left.reshape(-1), right.reshape(-1)
        maximum, squares, outside = 0.0, 0.0, 0
        for start in range(0, left.size, 1_000_000):
            a = left[start:start + 1_000_000].astype(np.float64)
            b = right[start:start + 1_000_000].astype(np.float64)
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError(f"Nonfinite output for {key}")
            error = np.abs(a - b)
            maximum = max(maximum, float(error.max()))
            squares += float(np.square(error).sum())
            outside += int(np.count_nonzero(error > atol + rtol * np.abs(a)))
        comparisons[key] = {"max_abs_error": maximum, "rmse": float(np.sqrt(squares / left.size)),
                            "values_outside_tolerance": outside, "values": int(left.size), "passed": outside == 0}
    return {"atol": atol, "rtol": rtol, "passed": all(x["passed"] for x in comparisons.values()),
            "candidates": comparisons}


def _cleanup_gpu():
    gc.collect()
    import torch
    torch.cuda.empty_cache()


def _run(config, generation, options, request, frames, *, backend, isolation=False, cache_check=False):
    model_config = {**config["model"], "backend": backend}
    if backend == "upstream":
        # Reference execution explicitly retains the original process runtime.
        model_config.pop("compiler_runtime", None)
        model_config.pop("compiler_feature_boundary", None)
    engine = InferenceEngine(model_config, generation, options)
    context = engine.prepare_context(frames)
    result = engine.generate(context, request["trajectories"], request["seeds"], candidate_ids=request["candidate_ids"])
    if result.status != "ok":
        raise RuntimeError(f"Validation generation failed: {result.failures}")
    details = {"model_identity": engine.backend.identity, "effective_config": result.effective_config,
               "cache_stats": result.cache_stats, "fallbacks": result.fallbacks,
               "denoiser_evaluations": result.denoiser_evaluations,
               "candidate_count": len(request["candidate_ids"])}
    def execution_evidence(value):
        return {"status": value.status, "effective_config": value.effective_config,
                "cache_stats": value.cache_stats, "fallbacks": value.fallbacks,
                "denoiser_evaluations": value.denoiser_evaluations}
    details["executions"] = {"normal": execution_evidence(result)}
    outputs = {"normal": result.frames}
    if isolation:
        reverse = engine.generate(context, list(reversed(request["trajectories"])), list(reversed(request["seeds"])),
                                  candidate_ids=list(reversed(request["candidate_ids"])))
        if reverse.status != "ok":
            raise RuntimeError(f"Reversed candidate generation failed: {reverse.failures}")
        outputs["reverse"] = reverse.frames
        details["executions"]["reverse"] = execution_evidence(reverse)
        # Perturb one candidate while retaining every other seed/action pair.
        altered = [np.asarray(x).copy() for x in request["trajectories"]]
        altered[0].reshape(-1)[0] += np.float32(0.125)
        changed = engine.generate(context, altered, request["seeds"], candidate_ids=request["candidate_ids"])
        if changed.status != "ok":
            raise RuntimeError(f"Changed-candidate generation failed: {changed.failures}")
        outputs["other_candidates_after_action_change"] = {
            key: value for key, value in changed.frames.items() if key != request["candidate_ids"][0]
        }
        details["executions"]["action_change"] = execution_evidence(changed)
    if cache_check:
        again = engine.prepare_context(frames.copy())
        changed_frames = frames.copy()
        changed_frames.flat[0] = -1.0 if frames.flat[0] > 0 else 1.0
        changed = engine.prepare_context(changed_frames)
        details["cache_key_checks"] = {
            "unchanged_input_hit": again.cache_hit,
            "unchanged_key_equal": again.cache_key == context.cache_key,
            "changed_input_missed": not changed.cache_hit,
            "changed_key_differs": changed.cache_key != context.cache_key,
        }
    return outputs, details


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["agreement", "isolation", "cache", "all"], default="all")
    parser.add_argument("--atol", type=float, required=True, help="Frozen development-selected absolute tolerance")
    parser.add_argument("--rtol", type=float, required=True, help="Frozen development-selected relative tolerance")
    parser.add_argument("--request-index", type=int, default=0)
    parser.add_argument("--candidates", type=int, default=2)
    parser.add_argument("--cache-bytes", type=int, help="Reuse check budget; defaults to configured budget or 2 GiB")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if not np.isfinite([args.atol, args.rtol]).all() or min(args.atol, args.rtol) < 0:
        parser.error("--atol and --rtol must be finite and nonnegative")
    if args.candidates < 2 and args.mode in {"isolation", "all"}:
        parser.error("candidate isolation requires at least two candidates")
    if args.candidates < 1 or (args.cache_bytes is not None and args.cache_bytes < 1):
        parser.error("candidate count and cache budget must be positive")
    config = load_config(args.config)
    generation, options = validate_run_config(config)
    cache_bytes = args.cache_bytes if args.cache_bytes is not None else options.conditioning_cache_bytes or 2 * 1024**3
    if config["model"].get("backend") == "synthetic":
        parser.error("this command validates real Vista outputs, not the synthetic fixture")
    if args.mode in {"agreement", "all"} and (generation.dtype != "fp16" or generation.decode_chunk_size != 14):
        parser.error("upstream refactoring agreement requires original FP16 and decode_chunk_size=14")
    if args.mode in {"agreement", "all"} and options.feature_cache_interval > 1:
        parser.error("upstream refactoring agreement requires approximate feature reuse to be disabled")
    if args.mode in {"cache", "all"} and generation.cond_aug:
        parser.error("cross-request cache checks require cond_aug=0")
    records = load_manifest(config["data"]["manifest"])
    if not 0 <= args.request_index < len(records):
        parser.error("--request-index is outside the manifest")
    record = records[args.request_index]
    root = config["data"]["root"]
    if config["data"].get("verify_checksums", True):
        verify_input_checksums(record, root, include_future=False)
    request = inference_request(record, root, candidate_count=args.candidates)
    frames = read_preprocess(request["observed_frames"], generation.height, generation.width)
    provenance = input_provenance(config["data"]["manifest"], records, record, request, frames,
                                  request_index=args.request_index)
    if provenance["synthetic"]:
        parser.error("real Vista numerical validation requires real observed inputs, not synthetic fixtures")
    environment = collect_environment()
    report_path = Path(args.output)
    artifact_dir = report_path.parent / (report_path.stem + "-outputs")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 2, "mode": args.mode, "created_at": datetime.now(timezone.utc).isoformat(),
              "request_id": request["request_id"], "config": config,
              "generation": generation.to_dict(), "optimizations": options.to_dict(),
              "environment": environment, "runtime_identity": reference_identity(environment),
              "implementation_sha256": source_identity(),
              "input_provenance": provenance, "input_provenance_sha256": digest(provenance),
              "validated_request_count": 1,
              "seeds": request["seeds"], "atol": args.atol, "rtol": args.rtol,
              "quality_validated": False, "scope": "selected request numerical checks only", "checks": {}, "runs": {}}
    def run(label, selected_options, backend="vista", **kwargs):
        outputs, details = _run(config, generation, selected_options, request, frames, backend=backend, **kwargs)
        _cleanup_gpu()  # _run's engine and GPU contexts are out of scope now.
        for kind, arrays in outputs.items():
            path = artifact_dir / f"{label}-{kind}.npz"
            np.savez(path, **arrays)
            details.setdefault("outputs", {})[kind] = str(path.resolve())
            details.setdefault("outputs_sha256", {})[kind] = file_sha256(path)
        report["runs"][label] = details
        return outputs
    if args.mode in {"agreement", "all"}:
        reference = run("upstream", OptimizationConfig(), backend="upstream")["normal"]
        actual = run("refactored", options)["normal"]
        report["checks"]["upstream_agreement"] = compare_outputs(reference, actual, atol=args.atol, rtol=args.rtol)
        del reference, actual
    if args.mode in {"isolation", "all"}:
        outputs = run("isolation", options, isolation=True)
        report["checks"]["candidate_order"] = compare_outputs(outputs["normal"], outputs["reverse"], atol=args.atol, rtol=args.rtol)
        others = {key: value for key, value in outputs["normal"].items() if key != request["candidate_ids"][0]}
        report["checks"]["unrelated_candidate_action_change"] = compare_outputs(
            others, outputs["other_candidates_after_action_change"], atol=args.atol, rtol=args.rtol)
        del outputs, others
    if args.mode in {"cache", "all"}:
        exact_options = replace(options, feature_cache_interval=1, feature_cache_module=None)
        reference = run("uncached", replace(exact_options, conditioning_cache_bytes=0))["normal"]
        actual = run("cached", replace(exact_options, conditioning_cache_bytes=cache_bytes), cache_check=True)["normal"]
        report["checks"]["conditioning_cache_agreement"] = compare_outputs(reference, actual, atol=args.atol, rtol=args.rtol)
        key_checks = report["runs"]["cached"]["cache_key_checks"]
        report["checks"]["conditioning_cache_keys"] = {**key_checks, "passed": all(key_checks.values())}
        report["checks"]["conditioning_cache_exercised"] = cache_coverage(report["runs"]["cached"])
    if options.max_batch_size > 1:
        coverage = {}
        for name, details in report["runs"].items():
            if name == "upstream":
                continue
            coverage[name] = candidate_batching_coverage(
                details, steps=generation.steps, requested_batch=options.max_batch_size,
                required_executions=["normal", "reverse", "action_change"] if name == "isolation" else ["normal"],
            )
        report["checks"]["candidate_batching_exercised"] = {
            "passed": bool(coverage) and all(item["passed"] for item in coverage.values()), "runs": coverage,
        }
    report["passed"] = all(check["passed"] for check in report["checks"].values())
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": report["passed"], "report": str(report_path.resolve()), "quality_validated": False}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
