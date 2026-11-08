#!/usr/bin/env python3
"""Recheck retained numerical-proof arrays at exact zero, without GPU execution.

Upstream agreement may use a development-frozen tolerance. Candidate isolation
and conditioning reuse must remain exact; this sidecar cannot relax them.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drivebench.config import digest, file_sha256
from drivebench.benchmarks.environment import source_identity
from drivebench.validate import compare_outputs

TYPE = "exact_numerical_semantics_v1"


def requires_sidecar(proof):
    """Missing/invalid tolerances fail closed, including on each check."""
    values = [proof.get("atol"), proof.get("rtol")]
    for name in ("upstream_agreement", "candidate_order", "unrelated_candidate_action_change",
                 "conditioning_cache_agreement"):
        check = proof.get("checks", {}).get(name)
        if check is not None:
            values.extend([check.get("atol"), check.get("rtol")])
    if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("Numerical proof needs explicit finite nonnegative tolerances")
    return any(x != 0 for x in values)


def _arrays(path, expected_ids, expected_shape):
    with np.load(path, allow_pickle=False) as value:
        if len(value.files) != len(set(value.files)) or set(value.files) != set(expected_ids):
            raise ValueError("Saved output candidate IDs differ from the proof")
        result = {key: value[key] for key in expected_ids}
    for array in result.values():
        if array.dtype != np.float32 or array.shape != expected_shape:
            raise ValueError("Saved outputs must match generation float32 [frames,height,width,3]")
    return result


def inspect(proof_path):
    """Recompute evidence; never use declared check pass flags as exact evidence."""
    path = Path(proof_path).resolve()
    proof_hash = file_sha256(path)
    proof = json.loads(path.read_text())
    requires_sidecar(proof)
    if (proof.get("schema_version") != 2 or proof.get("passed") is not True
            or proof.get("implementation_sha256") != source_identity()
            or not isinstance(proof.get("runtime_identity"), dict)
            or not proof["runtime_identity"].get("gpu", {}).get("uuid")):
        raise ValueError("Require a passed current-source proof with a full GPU identity")
    provenance = proof.get("input_provenance", {})
    ids = provenance.get("candidate_ids", [])
    if (len(ids) < 2 or len(set(ids)) != len(ids) or any(not isinstance(key, str) for key in ids)
            or digest(provenance) != proof.get("input_provenance_sha256")):
        raise ValueError("Require unchanged multi-candidate input provenance")
    runs = proof.get("runs", {})
    shape = tuple(proof.get("generation", {}).get(key) for key in ("total_frames", "height", "width"))
    if any(type(size) is not int or size < 1 for size in shape):
        raise ValueError("Explicit positive generation output dimensions are required")
    shape += (3,)
    if not {"upstream", "refactored", "isolation"} <= set(runs):
        raise ValueError("A complete agreement and isolation proof is required")
    hashes = {}
    for name, details in runs.items():
        outputs = details.get("outputs", {})
        if not outputs or set(outputs) != set(details.get("outputs_sha256", {})):
            raise ValueError(f"Missing output/hash coverage for {name}")
        for kind, location in outputs.items():
            location = str(Path(location).resolve())
            actual = file_sha256(location)
            if actual != details["outputs_sha256"][kind]:
                raise ValueError(f"Saved output checksum changed: {name}/{kind}")
            hashes[location] = actual
    checks = {}
    def compare(label, left_run, left_kind, right_run, right_kind, right_ids=None):
        selected = ids if right_ids is None else right_ids
        left = _arrays(runs[left_run]["outputs"][left_kind], ids, shape)
        right = _arrays(runs[right_run]["outputs"][right_kind], selected, shape)
        checks[label] = compare_outputs({key: left[key] for key in selected}, right, atol=0, rtol=0)
    try:
        compare("candidate_order", "isolation", "normal", "isolation", "reverse")
        compare("unrelated_candidate_action_change", "isolation", "normal", "isolation",
                "other_candidates_after_action_change", ids[1:])
        cache_needed = bool(proof.get("optimizations", {}).get("conditioning_cache_bytes"))
        cache_present = bool({"cached", "uncached"} & set(runs))
        if cache_needed or cache_present:
            compare("conditioning_cache_agreement", "uncached", "normal", "cached", "normal")
            keys = runs["cached"].get("cache_key_checks", {})
            if set(keys) != {"unchanged_input_hit", "unchanged_key_equal", "changed_input_missed", "changed_key_differs"} or any(v is not True for v in keys.values()):
                raise ValueError("Conditioning cache key checks are missing or failed")
            from drivebench.validate import cache_coverage
            checks["conditioning_cache_exercised"] = cache_coverage(runs["cached"])
            checks["conditioning_cache_keys"] = {**keys, "passed": True}
    except KeyError as error:
        raise ValueError("Missing retained isolation/cache pair") from error
    # Hash twice to reject replacement during the read/comparison window.
    if (file_sha256(path) != proof_hash or any(file_sha256(p) != h for p, h in hashes.items())
            or source_identity() != proof["implementation_sha256"]):
        raise ValueError("Numerical proof or arrays changed during verification")
    return {"type": TYPE, "proof_path": str(path), "proof_sha256": proof_hash,
            "implementation_sha256": proof["implementation_sha256"],
            "runtime_identity": proof["runtime_identity"], "runtime_identity_sha256": digest(proof["runtime_identity"]),
            "configuration_sha256": digest({key: proof.get(key) for key in ("config", "generation", "optimizations")}),
            "input_provenance_sha256": proof["input_provenance_sha256"],
            "outputs_sha256": hashes, "verifier_sha256": file_sha256(__file__),
            "checks": checks, "passed": all(check["passed"] for check in checks.values()),
            "atol": 0, "rtol": 0, "quality_validated": False,
            "scope": "Exact selected-request isolation and conditioning-cache semantics only; upstream tolerance unchanged"}


def create(proof_path, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Semantic sidecars are immutable; choose a fresh output")
    value = inspect(proof_path)
    value["created_at"] = datetime.now(timezone.utc).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2); stream.write("\n")
    return value


def verify(proof_path, sidecar_path):
    path = Path(sidecar_path).resolve()
    checksum = file_sha256(path)
    stored = json.loads(path.read_text())
    created = stored.pop("created_at", None)
    if not isinstance(created, str) or datetime.fromisoformat(created).tzinfo is None:
        raise ValueError("Sidecar creation time is missing or unzoned")
    actual = inspect(proof_path)
    if stored != actual or actual["passed"] is not True or file_sha256(path) != checksum:
        raise ValueError("Exact semantic sidecar differs, failed, or changed")
    return {"path": str(path), "sha256": checksum, "proof_path": actual["proof_path"],
            "proof_sha256": actual["proof_sha256"], "verifier_sha256": actual["verifier_sha256"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["create", "check"])
    parser.add_argument("--proof", required=True)
    parser.add_argument("--sidecar", required=True)
    args = parser.parse_args(argv)
    value = create(args.proof, args.sidecar) if args.command == "create" else verify(args.proof, args.sidecar)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if value.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
