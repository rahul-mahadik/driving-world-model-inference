"""Freeze an explicitly selected performance tolerance against repeated baselines.

Four or more independent, eligible baseline runs are required. The user chooses
the tolerance; this workflow verifies that it covers observed baseline drift.
No generated video arrays are opened by calibration or evidence verification.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import lru_cache
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import yaml

from .evaluation.io import Run, artifact_sha256, load_run
from .regression import _bootstrap_ratios, _comparison_identity, _pairs, timing_signature


RULE = "maximum upper 95% paired-scene bootstrap bound across p50/p95 and both directions of every baseline pair"


def _timestamp(value):
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Calibration and run timestamps must include a timezone")
    return result


def _started_at(run):
    return run.config.get("started_at") or run.environment.get("timestamp_utc")


def calibration_identity(run: Run) -> dict:
    identity = _comparison_identity(run)
    identity["optimizations"] = run.config.get("optimizations")
    selectors = {key: run.config.get("model", {}).get(key) for key in ("compiler_runtime", "compiler_feature_boundary")}
    if any(value is not None for value in selectors.values()):
        identity["compiler_implementation"] = selectors
    return identity


def _candidate_inputs(row):
    entries, count = row.get("candidates", []), row.get("candidate_count")
    if type(count) is not int or count < 1 or len(entries) != count:
        raise ValueError("Calibration requires every declared candidate input identity")
    values = []
    for item in entries:
        action = item.get("trajectory_sha256", item.get("action_sha256"))
        if (not isinstance(item.get("candidate_id"), str) or not item["candidate_id"]
                or type(item.get("seed")) is not int or not item.get("action_kind")
                or not isinstance(action, str) or len(action) != 64):
            raise ValueError("Calibration candidate seed, action kind, and trajectory checksum are required")
        values.append((item["candidate_id"], item["seed"], item["action_kind"], action))
    if len({item[0] for item in values}) != count:
        raise ValueError("Calibration candidate IDs must be unique")
    return sorted(values)


def _checked_baselines(paths):
    runs = [load_run(path) for path in paths]
    if len(runs) < 4 or len({run.path for run in runs}) != len(runs):
        raise ValueError("At least four independent baseline run directories are required")
    identity = calibration_identity(runs[0])
    for key, value in identity.items():
        if key == "optimizations":
            if not isinstance(value, dict):
                raise ValueError("Calibration requires an explicit optimization configuration")
        elif value in (None, {}, []):
            raise ValueError(f"Calibration comparison identity is missing: {key}")
    if any(not identity["runtime"].get(key) for key in ("python", "packages", "tools")):
        raise ValueError("Calibration requires Python, package, and CUDA/profiler runtime versions")
    expected = None
    expected_order = None
    observed_latency_vectors = set()
    for run in runs:
        if (run.synthetic or run.config.get("profiled") or run.summary.get("profiled")
                or run.summary.get("headline_eligible") is not True):
            raise ValueError("Calibration requires real, unprofiled, headline-eligible baseline runs")
        if calibration_identity(run) != identity:
            raise ValueError("Calibration baseline model, runtime, generation, workload, and optimizations must match")
        if (run.summary.get("status", "ok") not in {"ok", "success"}
                or run.summary.get("run_failure") or run.summary.get("failures")
                or run.summary.get("oom_attempts") or run.summary.get("recompiled_during_measurement")):
            raise ValueError("Failed, OOM, or recompiled baseline runs cannot calibrate performance")
        planned = run.config.get("benchmark", {}).get("requests")
        if type(planned) is not int or planned < 200 or len(run.measurements) != planned:
            raise ValueError("Every calibration run requires at least 200 complete planned requests")
        pairs = _pairs(run)
        if expected is None:
            expected = pairs
            expected_order = list(pairs)
        if pairs.keys() != expected.keys() or len({key[1] for key in pairs}) < 2:
            raise ValueError("Calibration requires matched request keys from at least two independent scenes")
        if list(pairs) != expected_order:
            raise ValueError("Calibration baseline request order must match")
        latencies = []
        for key in sorted(pairs):
            row = pairs[key]
            cache = row.get("cache_stats", {})
            if (row.get("status", "ok") not in {"ok", "success"} or row.get("oom_attempts")
                    or cache.get("oom_attempts") or cache.get("denoiser_evaluations_complete") is False):
                raise ValueError("Failed requests or successful OOM retries invalidate calibration")
            if _candidate_inputs(row) != _candidate_inputs(expected[key]):
                raise ValueError("Calibration request candidate inputs must match")
            latency = row.get("latency_s")
            if type(latency) not in (int, float) or not np.isfinite(latency) or latency <= 0:
                raise ValueError("Calibration latencies must be finite and positive")
            latencies.append(float(latency))
        vector = tuple(latencies)
        if vector in observed_latency_vectors:
            raise ValueError("Calibration baseline timing measurements must be independently collected; duplicate vectors found")
        observed_latency_vectors.add(vector)
    return runs, identity


@lru_cache(maxsize=16)
def _measured_variability(fingerprints: tuple[tuple[str, str], ...]):
    # Callers verify current hashes before reaching this cache. The key includes
    # those hashes, so edited files cannot reuse previously verified statistics.
    runs = [load_run(path) for path, _ in fingerprints]
    records = []
    required = 0.0
    for left_index, right_index in combinations(range(len(runs)), 2):
        left, right = _pairs(runs[left_index]), _pairs(runs[right_index])
        keys = sorted(left)
        before = np.array([left[key]["latency_s"] for key in keys], dtype=float)
        after = np.array([right[key]["latency_s"] for key in keys], dtype=float)
        scenes = [key[1] for key in keys]
        metrics = {}
        for name, quantile in (("p50", 0.5), ("p95", 0.95)):
            point, ci = _bootstrap_ratios(before, after, scene_groups=scenes, quantile=quantile)
            # Invert the same paired draws to cover the reverse comparison too.
            reverse_point = 1 / (1 + point) - 1
            reverse_ci = [1 / (1 + ci[1]) - 1, 1 / (1 + ci[0]) - 1]
            bound = max(0.0, point, reverse_point, ci[1], reverse_ci[1])
            required = max(required, bound)
            metrics[name] = {"relative_latency_change": point, "ci95": ci,
                             "reverse_relative_latency_change": reverse_point,
                             "reverse_ci95": reverse_ci, "required_tolerance": bound}
        records.append({"baseline_indices": [left_index, right_index], "metrics": metrics,
                        "paired_requests": len(keys), "independent_scenes": len(set(scenes))})
    return records, required


def _fingerprints(runs):
    return [{"path": str(run.path), "artifact_sha256": artifact_sha256(run)} for run in runs]


def _variability(evidence):
    return _measured_variability(tuple((entry["path"], json.dumps(entry["artifact_sha256"], sort_keys=True)) for entry in evidence))


def calibrate_performance(run_paths, relative_latency_tolerance, output):
    destination = Path(output)
    if destination.exists():
        raise FileExistsError("Frozen performance policies cannot be overwritten; choose a new versioned file")
    if (type(relative_latency_tolerance) not in (int, float)
            or not np.isfinite(relative_latency_tolerance) or relative_latency_tolerance < 0):
        raise ValueError("Select a finite nonnegative relative latency tolerance explicitly")
    runs, identity = _checked_baselines(run_paths)
    evidence = _fingerprints(runs)
    comparisons, required = _variability(evidence)
    if relative_latency_tolerance < required:
        raise ValueError(f"Selected tolerance {relative_latency_tolerance:.6g} does not cover measured variability bound {required:.6g}")
    frozen = datetime.now(timezone.utc)
    for run in runs:
        if _started_at(run) and _timestamp(_started_at(run)) > frozen:
            raise ValueError("Calibration baseline run timestamp is after policy freeze time")
    policy = {"schema_version": 1, "policy_type": "performance_regression", "performance": {
        "calibrated": True, "frozen": True, "frozen_at": frozen.isoformat(),
        "relative_latency_tolerance": float(relative_latency_tolerance), "minimum_requests": 200,
        "require_confirmatory_rerun": True, "calibration_evidence": {
            "schema_version": 1, "identity": identity, "runs": evidence,
            "comparisons": comparisons, "required_tolerance": required,
            "rule": RULE, "bootstrap_resamples": 4000, "bootstrap_seed": 0,
            "calibration_timestamps_available": all(bool(_started_at(run)) for run in runs),
            "selection": "explicit user tolerance; never fitted or relaxed automatically"}}}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        yaml.safe_dump(policy, stream, sort_keys=False)
    return policy


def validate_performance_policy(policy, baseline: Run, candidate: Run):
    settings = policy.get("performance", {})
    evidence = settings.get("calibration_evidence", {})
    if (policy.get("schema_version") != 1 or policy.get("policy_type") != "performance_regression"
            or settings.get("calibrated") is not True or settings.get("frozen") is not True
            or settings.get("require_confirmatory_rerun") is not True
            or type(settings.get("minimum_requests")) is not int or settings["minimum_requests"] < 200
            or evidence.get("schema_version") != 1 or evidence.get("rule") != RULE
            or evidence.get("bootstrap_resamples") != 4000 or evidence.get("bootstrap_seed") != 0):
        raise ValueError("A frozen performance policy with verified repeated-baseline calibration evidence is required")
    submitted = evidence.get("runs", [])
    runs, identity = _checked_baselines([entry["path"] for entry in submitted])
    actual = _fingerprints(runs)
    if actual != submitted:
        raise ValueError("Performance calibration artifacts changed after the policy was frozen")
    if identity != evidence.get("identity") or calibration_identity(baseline) != identity:
        raise ValueError("Performance calibration identity does not match the evaluated baseline workload")
    comparisons, required = _variability(actual)
    tolerance = settings.get("relative_latency_tolerance")
    if (comparisons != evidence.get("comparisons") or required != evidence.get("required_tolerance")
            or type(tolerance) not in (int, float) or not np.isfinite(tolerance) or tolerance < required):
        raise ValueError("Performance tolerance evidence does not cover verified baseline variability")
    frozen = _timestamp(settings.get("frozen_at"))
    for run in runs:
        if _started_at(run) and _timestamp(_started_at(run)) > frozen:
            raise ValueError("Performance calibration baselines must precede the policy freeze")
    calibration_measurements = {entry["artifact_sha256"]["measurements.jsonl"] for entry in actual}
    calibration_timings = {timing_signature(run) for run in runs}
    for run in (baseline, candidate):
        if (run.path in {source.path for source in runs}
                or artifact_sha256(run)["measurements.jsonl"] in calibration_measurements
                or timing_signature(run) in calibration_timings):
            raise ValueError("Regression comparisons must be independently collected after calibration")
        if _started_at(run) and _timestamp(_started_at(run)) < frozen:
            raise ValueError("Performance policy must be frozen before comparison runs start")
    return {"source_artifacts_verified": True, "variability_recomputed": True,
            "calibration_timestamps_available": all(bool(_started_at(run)) for run in runs),
            "comparison_timestamps_available": all(bool(_started_at(run)) for run in (baseline, candidate)),
            "frozen_at": frozen.isoformat(), "required_tolerance": required}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="Four or more independent repeated baseline directories")
    parser.add_argument("--relative-latency-tolerance", type=float, required=True)
    parser.add_argument("--output", required=True, help="New versioned performance policy YAML")
    args = parser.parse_args()
    try:
        policy = calibrate_performance(args.runs, args.relative_latency_tolerance, args.output)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(1, f"drivebench: {exc}\n")
    print(json.dumps({"output": str(Path(args.output).resolve()),
                      "relative_latency_tolerance": policy["performance"]["relative_latency_tolerance"],
                      "measured_variability_bound": policy["performance"]["calibration_evidence"]["required_tolerance"]}, indent=2))


if __name__ == "__main__":
    main()
