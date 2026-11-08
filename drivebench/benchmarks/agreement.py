"""Bind headline eligibility to reproducible, scoped numerical evidence."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from drivebench.config import digest, file_sha256
from drivebench.data import load_manifest, verify_input_checksums
from .environment import reference_identity, source_identity


def candidate_batching_coverage(details, *, steps, requested_batch, required_executions):
    """A requested batch size is not evidence that a joint batch actually ran."""
    reasons = []
    limit = details.get("model_identity", {}).get("candidate_batch_limit")
    count = details.get("candidate_count")
    if type(limit) is not int or limit < 2 or type(count) is not int or count < 2:
        return {"passed": False, "reasons": ["Joint batching requires an explicit backend limit and at least two candidates"]}
    batch = min(requested_batch, limit)
    expected = [min(batch, count - start) for start in range(0, count, batch)]
    for name in required_executions:
        execution = details.get("executions", {}).get(name, {})
        effective, stats = execution.get("effective_config", {}), execution.get("cache_stats", {})
        if (execution.get("status") != "ok" or effective.get("candidate_count") != count
                or effective.get("actual_batch_sizes") != expected or max(expected) < 2):
            reasons.append(f"{name} did not execute the expected joint candidate batches {expected}")
        if (effective != details.get("effective_config")
                or effective.get("requested_batch_size") != requested_batch):
            reasons.append(f"{name} effective settings differ from the selected candidate-batching run")
        if stats.get("oom_attempts") != [] or stats.get("denoiser_evaluations_complete") is not True:
            reasons.append(f"{name} has missing or failed OOM/completed-work evidence")
        if execution.get("denoiser_evaluations") != steps * len(expected):
            reasons.append(f"{name} denoiser invocation count does not match joint batching")
    return {"passed": not reasons, "reasons": reasons, "expected_batch_sizes": expected,
            "required_executions": list(required_executions)}


def numerical_evidence(config, generation, optimizations, environment, model_identity, records):
    """Return a failed gate for absent/stale evidence; exploratory runs can proceed.

    This proves only the selected request's checks. It does not establish
    held-out generation quality or numerical agreement on every measured scene.
    """
    reasons = []
    result = {"passed": False, "reasons": reasons, "quality_validated": False,
              "scope": "selected development request numerical checks only"}
    location = config.get("numerical_validation")
    if not location:
        reasons.append("No numerical_validation report configured")
        return result
    try:
        path = Path(location)
        proof = json.loads(path.read_text())
        result.update(report_path=str(path.resolve()), report_sha256=file_sha256(path))
        if proof.get("schema_version") != 2 or proof.get("passed") is not True:
            reasons.append("Numerical validation schema or pass status is invalid")
        created = datetime.fromisoformat(str(proof.get("created_at")).replace("Z", "+00:00"))
        started = datetime.fromisoformat(config["started_at"]) if config.get("started_at") else datetime.now(timezone.utc)
        if created.tzinfo is None or started.tzinfo is None or created > started:
            reasons.append("Numerical validation must precede the benchmark")
        if proof.get("implementation_sha256") != source_identity():
            reasons.append("Numerical validation uses different engine source")
        if proof.get("runtime_identity") != reference_identity(environment):
            reasons.append("Numerical validation uses a different GPU or runtime")
        if proof.get("generation") != generation.to_dict():
            reasons.append("Numerical validation generation settings differ")
        backend = config.get("model", {}).get("backend", "vista")
        selected = "upstream" if backend == "upstream" else "refactored"
        run = proof.get("runs", {}).get(selected, {})
        tested_options = run.get("effective_config", {}).get("optimizations")
        if tested_options != optimizations.to_dict():
            reasons.append("Numerical validation optimization settings differ")
        identity = run.get("model_identity", {})
        selected_runtime = config.get("model", {}).get("compiler_runtime")
        actual_runtime = model_identity.get("compiler_runtime")
        if ((selected_runtime is not None and (not isinstance(actual_runtime, dict) or actual_runtime.get("name") != selected_runtime))
                or (selected_runtime is None and actual_runtime is not None)):
            reasons.append("Declared compiler runtime differs from the active model identity")
        if identity.get("compiler_runtime") != actual_runtime:
            reasons.append("Numerical validation compiler runtime identity differs")
        selected_boundary = config.get("model", {}).get("compiler_feature_boundary")
        actual_boundary = model_identity.get("compiler_feature_boundary")
        if ((selected_boundary is not None and (not isinstance(actual_boundary, dict) or actual_boundary.get("module") != selected_boundary))
                or (selected_boundary is None and actual_boundary is not None)
                or identity.get("compiler_feature_boundary") != actual_boundary):
            reasons.append("Numerical validation compiler feature boundary identity differs")
        if any(proof.get("runs", {}).get("upstream", {}).get("model_identity", {}).get(key) is not None
               for key in ("compiler_runtime", "compiler_feature_boundary")):
            reasons.append("Upstream numerical reference enabled a compiler runtime or feature boundary")
        for key in ("upstream_commit", "checkpoint_sha256", "inference_config_sha256", "backend"):
            if not identity.get(key) or identity.get(key) != model_identity.get(key):
                reasons.append(f"Numerical validation model identity differs: {key}")
        matched_runs = [selected]
        if backend != "upstream":
            matched_runs.append("isolation")
            if optimizations.conditioning_cache_bytes:
                matched_runs.append("cached")
        for name in matched_runs:
            details = proof.get("runs", {}).get(name, {})
            effective = details.get("effective_config", {})
            if effective.get("optimizations") != optimizations.to_dict():
                reasons.append(f"Numerical validation {name} uses different optimization settings")
            if any(effective.get(key) != value for key, value in generation.to_dict().items()):
                reasons.append(f"Numerical validation {name} uses different generation settings")
            tested_identity = details.get("model_identity", {})
            if tested_identity.get("compiler_runtime") != actual_runtime:
                reasons.append(f"Numerical validation {name} uses a different compiler runtime")
            if tested_identity.get("compiler_feature_boundary") != actual_boundary:
                reasons.append(f"Numerical validation {name} uses a different compiler feature boundary")
            if any(tested_identity.get(key) != model_identity.get(key)
                   for key in ("upstream_commit", "checkpoint_sha256", "inference_config_sha256", "backend")):
                reasons.append(f"Numerical validation {name} uses a different model")
            if backend != "upstream" and optimizations.max_batch_size > 1:
                if tested_identity.get("candidate_batch_limit") != model_identity.get("candidate_batch_limit"):
                    reasons.append(f"Numerical validation {name} has a different backend batch limit")
                variants = ["normal", "reverse", "action_change"] if name == "isolation" else ["normal"]
                coverage = candidate_batching_coverage(details, steps=generation.steps,
                                                       requested_batch=optimizations.max_batch_size,
                                                       required_executions=variants)
                reasons.extend(f"Numerical validation {name}: {reason}" for reason in coverage["reasons"])
        required = ["upstream_agreement", "candidate_order", "unrelated_candidate_action_change"]
        if optimizations.conditioning_cache_bytes:
            required += ["conditioning_cache_agreement", "conditioning_cache_keys", "conditioning_cache_exercised"]
        checks = proof.get("checks", {})
        for name in required:
            if checks.get(name, {}).get("passed") is not True:
                reasons.append(f"Required numerical check did not pass: {name}")
        provenance = proof.get("input_provenance", {})
        if digest(provenance) != proof.get("input_provenance_sha256"):
            reasons.append("Numerical validation input provenance digest differs")
        source_manifest = Path(provenance["manifest_path"])
        source_records = load_manifest(source_manifest)
        metadata_path = source_manifest.with_suffix(source_manifest.suffix + ".meta.json")
        metadata = json.loads(metadata_path.read_text())
        source_hash = file_sha256(source_manifest)
        if (source_hash != provenance.get("manifest_sha256")
                or file_sha256(metadata_path) != provenance.get("manifest_metadata_sha256")):
            reasons.append("Numerical validation manifest evidence changed")
        if provenance.get("partition_role") != "development" or metadata.get("partition_role") != "development":
            reasons.append("Numerical validation must use the development manifest")
        source_scenes = sorted({r["scene_id"] for r in source_records})
        if source_scenes != provenance.get("manifest_scene_ids") or source_scenes != metadata.get("scene_ids"):
            reasons.append("Numerical validation development scene provenance differs")
        if any(r.get("dataset") != "nuscenes" for r in source_records):
            reasons.append("Numerical validation requires real nuScenes inputs")
        record = source_records[provenance["request_index"]]
        if (digest(record) != provenance.get("record_sha256") or record["scene_id"] != provenance.get("scene_id")
                or record["request_id"] != provenance.get("request_id")):
            reasons.append("Numerical validation request content differs")
        validation_data = proof.get("config", {}).get("data", {})
        if validation_data.get("verify_checksums", True) is not True:
            reasons.append("Numerical validation must verify observed input checksums")
        verify_input_checksums(record, validation_data["root"], include_future=False)
        if len(provenance.get("candidate_ids", [])) < 2:
            reasons.append("Numerical isolation evidence requires at least two candidates")
        if config.get("data", {}).get("verify_checksums", True) is not True:
            reasons.append("Headline benchmarks require verified input checksums")
        current_manifest = Path(config["data"]["manifest"])
        current_hash = file_sha256(current_manifest)
        if current_hash != source_hash:
            current_metadata = json.loads(current_manifest.with_suffix(current_manifest.suffix + ".meta.json").read_text())
            current_scenes = sorted({r["scene_id"] for r in records})
            if current_metadata.get("partition_role") not in {"held_out_evaluation", "held_out_evaluation_second_seed"}:
                reasons.append("A different benchmark manifest must be explicitly held out")
            if current_scenes != current_metadata.get("scene_ids") or set(current_scenes) & set(source_scenes):
                reasons.append("Benchmark evaluation scenes overlap or lack verified provenance")
        # Retain raw outputs as evidence; a stale report with replaced arrays
        # must not promote an unrelated measurement.
        required_runs = ["upstream", "refactored", "isolation"]
        if optimizations.conditioning_cache_bytes:
            required_runs += ["uncached", "cached"]
        for name in required_runs:
            if name not in proof.get("runs", {}):
                reasons.append(f"Numerical validation run is missing: {name}")
        for name, details in proof.get("runs", {}).items():
            outputs = details.get("outputs", {})
            if not outputs:
                reasons.append(f"Numerical validation raw outputs are missing: {name}")
            for kind, output in outputs.items():
                if file_sha256(output) != details.get("outputs_sha256", {}).get(kind):
                    reasons.append(f"Numerical validation output changed: {name}/{kind}")
        if backend == "synthetic":
            reasons.append("Synthetic checks do not establish Vista numerical agreement")
        result["validated_request_count"] = proof.get("validated_request_count", 0)
        if not result["validated_request_count"]:
            reasons.append("Numerical validation did not exercise any requests")
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
        reasons.append(f"Numerical evidence is unavailable or invalid: {error}")
    result["passed"] = not reasons
    return result
