"""Paired performance regression checks with uncertainty and confirmatory reruns.

A possible slowdown cannot reject a change until an independent confirmatory
run also exceeds the calibrated tolerance and its bootstrap lower bound does.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import json

import numpy as np
import yaml

from .evaluation.io import Run, experiment_identity, load_run, write_json
from .benchmarks.environment import reference_identity


def _pairs(run: Run) -> dict[tuple, dict]:
    result = {}
    for row in run.measurements:
        key = (str(row["request_id"]), str(row["scene_id"]), int(row.get("repeat", 0)))
        if key in result:
            raise ValueError(f"Duplicate request key {key}")
        result[key] = row
    return result


def timing_signature(run: Run) -> tuple:
    """Ignore filenames/notes when identifying copied timing observations."""
    return tuple((key, float(row["latency_s"])) for key, row in sorted(_pairs(run).items()))


def _comparison_identity(run: Run) -> dict[str, Any]:
    identity = experiment_identity(run)
    # Regression applies to the same generated computation; approximate step
    # sweeps belong on the quality/performance curve, not in a fixed-workload CI.
    identity["generation"] = run.config.get("generation", {})
    identity["torch"] = run.environment.get("torch")
    runtime = reference_identity(run.environment)
    identity["runtime"] = {key: runtime.get(key) for key in ("python", "packages", "tools")}
    # Export and warmup policy can change queue residence and cache state even
    # when the generated tensor shapes and candidate counts are unchanged.
    identity["measurement_protocol"] = run.config.get("benchmark", {})
    return identity


def _bootstrap_ratios(baseline: np.ndarray, candidate: np.ndarray, *, scene_groups: list[str], quantile: float,
                      resamples: int = 4000, seed: int = 0) -> tuple[float, list[float]]:
    """Resample paired scene clusters, retaining every repeat within each scene."""
    groups = np.asarray(scene_groups)
    if len(groups) != len(baseline) or len(baseline) != len(candidate):
        raise ValueError("Scene groups must align with paired latency measurements")
    names = sorted(set(scene_groups))
    if len(names) < 2:
        raise ValueError("Scene-cluster uncertainty requires at least two independent scenes")
    clusters = [np.flatnonzero(groups == name) for name in names]
    rng = np.random.default_rng(seed)
    point = float(np.quantile(candidate, quantile) / np.quantile(baseline, quantile) - 1)
    draws = np.empty(resamples)
    for index in range(resamples):
        selected = rng.integers(0, len(clusters), len(clusters))
        sample = np.concatenate([clusters[cluster] for cluster in selected])
        draws[index] = (np.quantile(candidate[sample], quantile)
                        / np.quantile(baseline[sample], quantile) - 1)
    return point, [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def _compare_once(baseline: Run, candidate: Run, tolerances: dict[str, Any]) -> dict[str, Any]:
    baseline_identity, candidate_identity = _comparison_identity(baseline), _comparison_identity(candidate)
    issues = []
    for key in baseline_identity:
        before, after = baseline_identity[key], candidate_identity[key]
        if before in (None, {}, []) or after in (None, {}, []):
            issues.append(f"Missing comparison identity: {key}")
        elif before != after:
            issues.append(f"Comparison identity mismatch: {key}")
    if baseline.synthetic or candidate.synthetic:
        issues.append("Synthetic runs cannot establish hardware performance regressions")
    if any(run.config.get("profiled") or run.summary.get("profiled") for run in (baseline, candidate)):
        issues.append("Profiled runs cannot establish performance regressions")
    if issues:
        return {"status": "incompatible", "reasons": issues, "baseline_identity": baseline_identity,
                "candidate_identity": candidate_identity}
    invalid_measurements = []
    for run in (baseline, candidate):
        if run.summary.get("headline_eligible") is not True:
            invalid_measurements.append(f"{run.name}: run is not eligible for a performance conclusion")
        if run.summary.get("status", "ok") not in {"ok", "success"} or run.summary.get("run_failure") or run.summary.get("failures"):
            invalid_measurements.append(f"{run.name}: run failed or contains failed requests")
        if run.summary.get("recompiled_during_measurement"):
            invalid_measurements.append(f"{run.name}: compilation occurred during measured requests")
        planned = run.config.get("benchmark", {}).get("requests")
        if planned is not None and (not isinstance(planned, int) or isinstance(planned, bool)
                                    or len(run.measurements) != planned):
            invalid_measurements.append(f"{run.name}: saved measurements do not cover the planned requests")
        if run.summary.get("oom_attempts") or any(
            row.get("cache_stats", {}).get("oom_attempts") or row.get("oom_attempts")
            or row.get("cache_stats", {}).get("denoiser_evaluations_complete") is False
            for row in run.measurements
        ):
            invalid_measurements.append(f"{run.name}: OOM attempts or incomplete execution counters invalidate the run, including successful retries")
    if invalid_measurements:
        return {"status": "incomplete", "reasons": invalid_measurements}
    settings = tolerances.get("performance", {})
    tolerance = settings.get("relative_latency_tolerance")
    if (not settings.get("calibrated") or not isinstance(tolerance, (int, float))
            or isinstance(tolerance, bool) or not np.isfinite(tolerance) or tolerance < 0):
        return {"status": "uncalibrated", "reasons": ["Performance tolerance must be calibrated using measured variability"]}
    original, current = _pairs(baseline), _pairs(candidate)
    if original.keys() != current.keys() or not original:
        return {"status": "incomplete", "reasons": ["Complete matched request keys are required"]}
    if list(original) != list(current):
        return {"status": "incompatible", "reasons": ["Paired request order differs between the baseline and candidate"]}
    before, after, scene_groups = [], [], []
    for key in sorted(original):
        left, right = original[key], current[key]
        scene_groups.append(key[1])
        if left.get("status", "ok") not in {"ok", "success"} or right.get("status", "ok") not in {"ok", "success"}:
            return {"status": "incomplete", "reasons": ["Failures/OOMs invalidate a successful-request-only comparison"]}
        if left.get("candidate_count") != right.get("candidate_count"):
            return {"status": "incompatible", "reasons": ["Candidate workload differs between paired requests"]}
        def candidate_inputs(row):
            return sorted((str(c["candidate_id"]), int(c["seed"]), c.get("action_kind"),
                           c.get("trajectory_sha256", c.get("action_sha256"))) for c in row.get("candidates", []))
        for row in (left, right):
            declared, entries = row.get("candidate_count"), row.get("candidates", [])
            if (not isinstance(declared, int) or isinstance(declared, bool) or declared < 1
                    or len(entries) != declared or len({c.get("candidate_id") for c in entries}) != declared):
                return {"status": "incomplete", "reasons": ["Every declared candidate requires a unique input identity"]}
            for item in entries:
                action_hash = item.get("trajectory_sha256", item.get("action_sha256"))
                if (not isinstance(item.get("candidate_id"), str) or not item["candidate_id"]
                        or not isinstance(item.get("seed"), int) or isinstance(item["seed"], bool)
                        or not isinstance(item.get("action_kind"), str) or not item["action_kind"]
                        or not isinstance(action_hash, str) or len(action_hash) != 64):
                    return {"status": "incomplete", "reasons": ["Candidate seed, action kind, and trajectory checksum are required"]}
        if candidate_inputs(left) != candidate_inputs(right):
            return {"status": "incompatible", "reasons": ["Candidate seeds/actions differ between paired requests"]}
        for row, values in ((left, before), (right, after)):
            value = row.get("latency_s")
            if not isinstance(value, (float, int)) or not np.isfinite(value) or value <= 0:
                return {"status": "incomplete", "reasons": ["Latency measurements must be finite and positive"]}
            values.append(value)
    minimum = max(200, int(settings.get("minimum_requests", 200)))
    if len(before) < minimum:
        return {"status": "insufficient_samples", "reasons": [f"Need at least {minimum} paired requests"], "pairs": len(before)}
    if len(set(scene_groups)) < 2:
        return {"status": "insufficient_samples", "reasons": ["Scene-cluster uncertainty requires at least two independent scenes"],
                "pairs": len(before), "independent_scenes": len(set(scene_groups))}
    if baseline.path == candidate.path or timing_signature(baseline) == timing_signature(candidate):
        return {"status": "incomplete", "reasons": ["Baseline and candidate require independent timing measurements"]}
    # Flags alone are not calibration. Verify source artifacts, measured drift,
    # configuration identity, and freeze timing before making a speed decision.
    from .calibrate_performance import validate_performance_policy
    try:
        calibration_verification = validate_performance_policy(tolerances, baseline, candidate)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return {"status": "uncalibrated", "reasons": [str(exc)]}
    results = {}
    exceeds, possible = False, False
    for name, quantile in (("p50", 0.5), ("p95", 0.95)):
        point, ci = _bootstrap_ratios(np.array(before), np.array(after), scene_groups=scene_groups, quantile=quantile)
        confident = point > tolerance and ci[0] > tolerance
        exceeds |= confident
        possible |= point > tolerance
        results[name] = {"relative_latency_change": point, "ci95": ci,
                         "tolerance": tolerance, "exceeds_tolerance_and_uncertainty": confident}
    return {"status": "rerun_required" if exceeds else "inconclusive" if possible else "pass",
            "metrics": results, "pairs": len(before), "independent_scenes": len(set(scene_groups)),
            "calibration_verification": calibration_verification,
            "uncertainty_method": "paired scene-cluster percentile bootstrap, 4000 resamples; repeats remain grouped; p50 and p95 latency ratios",
            "reasons": ["Independent confirmatory rerun is required before rejecting a change"] if exceeds else []}


def compare_runs(baseline_path: str | Path, candidate_path: str | Path, tolerances: dict[str, Any], *,
                 confirm_baseline_path: str | Path | None = None,
                 confirm_candidate_path: str | Path | None = None) -> dict[str, Any]:
    baseline, candidate = load_run(baseline_path), load_run(candidate_path)
    result = _compare_once(baseline, candidate, tolerances)
    result["baseline"], result["candidate"] = str(baseline.path), str(candidate.path)
    if (confirm_baseline_path is None) != (confirm_candidate_path is None):
        raise ValueError("Provide both --confirm-baseline and --confirm-candidate")
    if result["status"] == "rerun_required" and confirm_baseline_path is not None:
        confirm_baseline, confirm_candidate = load_run(confirm_baseline_path), load_run(confirm_candidate_path)
        # Reusing the same saved measurements is not an independent rerun.
        if ({confirm_baseline.path, confirm_candidate.path} & {baseline.path, candidate.path}
                or confirm_baseline.path == confirm_candidate.path
                or len({timing_signature(run) for run in
                        (baseline, candidate, confirm_baseline, confirm_candidate)}) != 4):
            result["reasons"].append("Confirmatory artifacts must contain independent measurements")
            return result
        if (_comparison_identity(confirm_baseline) != _comparison_identity(baseline)
                or _comparison_identity(confirm_candidate) != _comparison_identity(candidate)
                or confirm_baseline.config.get("optimizations") != baseline.config.get("optimizations")
                or confirm_candidate.config.get("optimizations") != candidate.config.get("optimizations")
                or any(confirm.config.get("model", {}).get(key) != original.config.get("model", {}).get(key)
                       for confirm, original in ((confirm_baseline, baseline), (confirm_candidate, candidate))
                       for key in ("compiler_runtime", "compiler_feature_boundary"))):
            result["reasons"].append("Confirmatory runs have incompatible GPU, model, or workload identity")
            return result
        confirmation = _compare_once(confirm_baseline, confirm_candidate, tolerances)
        result["confirmation"] = confirmation
        # The same statistic must regress twice; do not combine an isolated p50
        # signal in one run with a different isolated p95 signal in another.
        repeated = any(result["metrics"][metric]["exceeds_tolerance_and_uncertainty"]
                       and confirmation.get("metrics", {}).get(metric, {}).get("exceeds_tolerance_and_uncertainty")
                       for metric in ("p50", "p95"))
        result["status"] = "regression_confirmed" if repeated else "not_confirmed"
        result["reasons"] = (["Same latency statistic exceeds calibrated tolerance and uncertainty in independent reruns"]
                             if repeated else ["Apparent regression did not repeat; do not reject the change"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--tolerances", default="configs/tolerances.yaml")
    parser.add_argument("--confirm-baseline")
    parser.add_argument("--confirm-candidate")
    parser.add_argument("--output")
    args = parser.parse_args()
    with Path(args.tolerances).open() as stream:
        tolerances = yaml.safe_load(stream)
    result = compare_runs(args.baseline, args.candidate, tolerances,
                          confirm_baseline_path=args.confirm_baseline, confirm_candidate_path=args.confirm_candidate)
    if args.output:
        write_json(Path(args.output), result)
    print(json.dumps(result, indent=2))
    raise SystemExit(1 if result["status"] == "regression_confirmed" else 0 if result["status"] == "pass" else 2)


if __name__ == "__main__":
    main()
