#!/usr/bin/env python3
"""Strict comparison of saved outputs from the actual benchmark input pipeline."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from drivebench.benchmarks.environment import reference_identity
from drivebench.config import file_sha256
from drivebench.validate import compare_outputs


def read_run(path):
    root = Path(path).resolve()
    values = {name: json.loads((root / f"{name}.json").read_text())
              for name in ("config", "environment", "model_identity", "summary")}
    rows = [json.loads(line) for line in (root / "measurements.jsonl").read_text().splitlines() if line.strip()]
    if not rows or values["config"].get("data", {}).get("manifest_metadata", {}).get("partition_role") != "development_subset":
        raise ValueError("This check requires nonempty exploratory development-subset benchmark outputs")
    if values["config"]["data"].get("verify_checksums", True) is not True:
        raise ValueError("Benchmark input checksums must have been verified")
    indexed = {}
    for row in rows:
        if row.get("status") != "ok" or row.get("cache_stats", {}).get("oom_attempts"):
            raise ValueError("Input-pipeline check cannot accept failures or OOM fallback")
        for candidate in row.get("candidates", []):
            key = (row["request_index"], row["request_id"], candidate["candidate_id"])
            relative = Path(candidate.get("frames_path") or "")
            if (key in indexed or not candidate.get("frames_path") or relative.is_absolute()
                    or ".." in relative.parts or not (root / relative).is_file()):
                raise ValueError("Missing, duplicate, or unsafe saved candidate output")
            resolved = (root / relative).resolve()
            if not resolved.is_relative_to(root):
                raise ValueError("Saved output escaped the benchmark directory")
            indexed[key] = (row, candidate, resolved)
    if not indexed:
        raise ValueError("No saved outputs to compare")
    values.update(root=root, indexed=indexed,
                  artifact_sha256={name: file_sha256(root / name) for name in
                      ("config.json", "environment.json", "model_identity.json", "summary.json", "measurements.jsonl")})
    return values


def check(reference_path, candidate_path):
    reference, candidate = read_run(reference_path), read_run(candidate_path)
    if set(reference["indexed"]) != set(candidate["indexed"]):
        raise ValueError("Benchmark request/candidate identities differ")
    for key in ("generation",):
        if reference["config"].get(key) != candidate["config"].get(key):
            raise ValueError(f"Benchmark {key} settings differ")
    if reference["config"]["data"]["manifest_sha256"] != candidate["config"]["data"]["manifest_sha256"]:
        raise ValueError("Benchmark source manifests differ")
    for key in ("checkpoint_sha256", "inference_config_sha256", "upstream_commit", "parameter_grad_policy"):
        if not reference["model_identity"].get(key) or reference["model_identity"][key] != candidate["model_identity"].get(key):
            raise ValueError(f"Benchmark model identity differs: {key}")
    if reference_identity(reference["environment"]) != reference_identity(candidate["environment"]):
        raise ValueError("Benchmark GPU or runtime differs")
    comparisons = []
    for key in reference["indexed"]:
        left_row, left_candidate, left_path = reference["indexed"][key]
        right_row, right_candidate, right_path = candidate["indexed"][key]
        if left_row["scene_id"] != right_row["scene_id"] or any(
            left_candidate.get(field) != right_candidate.get(field)
            for field in ("seed", "trajectory_sha256", "action_kind")):
            raise ValueError("Paired scene, seed, or action differs")
        if not isinstance(left_candidate.get("trajectory_sha256"), str) or len(left_candidate["trajectory_sha256"]) != 64:
            raise ValueError("Paired action fingerprint missing")
        left, right = np.load(left_path, mmap_mode="r", allow_pickle=False), np.load(right_path, mmap_mode="r", allow_pickle=False)
        generation = reference["config"]["generation"]
        expected = (generation["total_frames"], generation["height"], generation["width"], 3)
        if left.shape != expected or right.shape != expected or left.dtype != np.float32 or right.dtype != np.float32:
            raise ValueError("Saved output dtype or dimensions differ from the declared host-frame contract")
        comparison = compare_outputs({key[2]: left}, {key[2]: right}, atol=0, rtol=0)
        comparisons.append({"request_index": key[0], "request_id": key[1], "candidate_id": key[2],
                            "reference_sha256": file_sha256(left_path), "candidate_sha256": file_sha256(right_path),
                            **comparison})
    return {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
            "passed": all(row["passed"] for row in comparisons), "atol": 0, "rtol": 0,
            "reference": str(reference["root"]), "candidate": str(candidate["root"]),
            "reference_artifact_sha256": reference["artifact_sha256"], "candidate_artifact_sha256": candidate["artifact_sha256"],
            "comparisons": comparisons, "headline_eligible": False, "quality_validated": False,
            "scope": "saved benchmark-path outputs only; does not validate unsaved requests or a latency claim"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Preserve the previous proof; choose a fresh output")
    report = check(args.reference, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"passed": report["passed"], "output": str(args.output)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
