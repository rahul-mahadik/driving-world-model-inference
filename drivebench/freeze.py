"""Freeze explicitly selected quality tolerances with development provenance.

Threshold values come from the supplied draft; this command never chooses them
from evaluation outcomes. Synthetic fixtures cannot create a release policy.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import yaml

from .config import digest, file_sha256
from .evaluation.io import load_run


METRICS = {"mae", "rmse", "max_abs_error", "lpips", "temporal_change_error",
           "freeze_fraction_increase", "temporal_derivative_mae"}
ARTIFACTS = ("config.json", "environment.json", "measurements.jsonl", "summary.json")


def _checked_run(path, expected_hashes, minimum_scenes):
    run = load_run(path)
    if run.synthetic or run.summary.get("status") == "failed" or run.summary.get("failures"):
        raise ValueError("Tolerance calibration requires successful real Vista development runs")
    hashes = {name: file_sha256(run.path / name) for name in ARTIFACTS}
    if hashes != expected_hashes:
        raise ValueError(f"Development artifacts changed after evaluation or lack fingerprints: {run.path}")
    data = run.config.get("data", {})
    meta = data.get("manifest_metadata", {})
    if meta.get("partition_role") != "development":
        raise ValueError("Only the development partition may select final quality tolerances")
    manifest_hash = data.get("manifest_sha256")
    if not isinstance(manifest_hash, str) or len(manifest_hash) != 64 or meta.get("manifest_sha256") != manifest_hash:
        raise ValueError("Development manifest checksum evidence is missing or inconsistent")
    declared = meta.get("scene_ids", [])
    measured = {row.get("scene_id") for row in run.measurements if row.get("status") == "ok"}
    if not declared or None in measured or measured != set(declared) or len(measured) < minimum_scenes:
        raise ValueError(f"Development measurements must cover the declared scenes (minimum {minimum_scenes})")
    requests = {row.get("request_id") for row in run.measurements}
    clip_count = meta.get("clip_count")
    if (type(clip_count) is not int or clip_count < len(declared)
            or any(not isinstance(item, str) or not item for item in requests)
            or len(requests) != clip_count
            or any(row.get("status") != "ok" for row in run.measurements)):
        raise ValueError("Development measurements must cover every declared clip without failures")
    if meta.get("scene_ids_sha256") != digest(sorted(declared)):
        raise ValueError("Development scene list does not match its checksum")
    return run, manifest_hash, sorted(measured), hashes


def freeze_tolerances(evaluation_path, thresholds_path, output, *, minimum_scenes=32):
    if type(minimum_scenes) is not int or minimum_scenes < 2:
        raise ValueError("minimum_scenes must be an integer of at least two")
    destination = Path(output)
    if destination.exists():
        raise FileExistsError("Frozen tolerances cannot be overwritten; choose a new versioned file")
    evaluation_path, thresholds_path = Path(evaluation_path), Path(thresholds_path)
    evaluation = json.loads(evaluation_path.read_text())
    draft = yaml.safe_load(thresholds_path.read_text())
    if not isinstance(draft, dict) or not isinstance(draft.get("metrics"), dict):
        raise ValueError("Draft must contain explicitly selected metric thresholds")
    metrics = draft["metrics"]
    exact = draft.get("policy_type") == "exact_output_equivalence"
    if exact and metrics != exact_metrics():
        raise ValueError("Exact equivalence requires the predefined zero-error constraints")
    if not ({"mae", "rmse", "max_abs_error"} & metrics.keys()) or ("lpips" not in metrics and not exact) or not ({"temporal_change_error", "freeze_fraction_increase", "temporal_derivative_mae"} & metrics.keys()):
        raise ValueError("Select numerical, LPIPS, and temporal diagnostic constraints before freezing")
    for name, setting in metrics.items():
        if (name not in METRICS or not isinstance(setting, dict)
                or setting.get("aggregation", "mean") not in {"mean", "max"}
                or type(setting.get("maximum")) not in (int, float)
                or not math.isfinite(setting["maximum"]) or setting["maximum"] < 0):
            raise ValueError(f"Invalid explicit threshold: {name}")
    if evaluation.get("perceptual", {}).get("status") != "available" and not exact:
        raise ValueError("Development LPIPS measurements must actually be available")
    if draft.get("performance", {}).get("calibrated"):
        raise ValueError("Quality freezing does not calibrate performance variability; use a separate performance policy")
    reference, manifest_hash, scenes, ref_hashes = _checked_run(
        evaluation["reference"], evaluation.get("reference_artifact_sha256"), minimum_scenes)
    entries = evaluation.get("runs", [])
    if exact:
        selected = draft.get("exact_runs", [])
        if not selected or len(set(selected)) != len(selected) or set(selected) - {entry["run"] for entry in entries}:
            raise ValueError("Select distinct existing development runs for exact equivalence")
        entries = [entry for entry in entries if entry["run"] in selected]
    if not entries:
        raise ValueError("At least one real development comparison is required")
    calibration_runs = []
    for entry in entries:
        if entry.get("synthetic") or not entry.get("complete_pairs") or entry.get("errors"):
            raise ValueError("Development comparisons must contain complete, valid real output pairs")
        run, run_manifest_hash, run_scenes, hashes = _checked_run(entry["run_path"], entry.get("artifact_sha256"), minimum_scenes)
        if run_manifest_hash != manifest_hash or run_scenes != scenes:
            raise ValueError("All calibration runs must use the same development manifest")
        pairs = entry.get("pairs", [])
        paired_scenes = {pair["scene_id"] for pair in pairs}
        if paired_scenes != set(scenes):
            raise ValueError("Saved development output pairs must cover every development scene")
        paired_requests = {pair["request_id"] for pair in pairs}
        run_requests = {row["request_id"] for row in run.measurements}
        reference_requests = {row["request_id"] for row in reference.measurements}
        if (paired_requests != run_requests or paired_requests != reference_requests
                or len(pairs) != entry.get("matched_pairs") or not pairs):
            raise ValueError("Saved development output pairs must cover every declared clip")
        observed = {}
        for name in metrics:
            summary = entry.get("aggregates", {}).get(name, {})
            value = summary.get(metrics[name].get("aggregation", "mean"))
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or summary.get("pairs") != entry.get("matched_pairs")):
                raise ValueError(f"Development metric {name} is missing pairs or is not finite")
            observed[name] = value
            if exact and (value != 0 or any(pair["metrics"].get(name) != 0 for pair in pairs)):
                raise ValueError(f"Development run {run.name} does not preserve exact outputs")
        calibration_runs.append({"path": str(run.path), "artifact_sha256": hashes,
                                 "observed_metrics": observed, "matched_pairs": entry["matched_pairs"]})
    policy = {"schema_version": 1, "calibrated": True, "frozen": True,
              "selected_on": "development", "development_manifest_sha256": manifest_hash,
              "development_scene_ids": scenes, "frozen_at": datetime.now(timezone.utc).isoformat(),
              "metrics": metrics,
              "policy_type": "exact_output_equivalence" if exact else "reviewed_metric_tolerances",
              "performance": {"calibrated": False, "relative_latency_tolerance": None,
                              "minimum_requests": 200, "require_confirmatory_rerun": True},
              "calibration_evidence": {"evaluation_path": str(evaluation_path.resolve()),
                                       "evaluation_sha256": file_sha256(evaluation_path),
                                       "threshold_draft_sha256": file_sha256(thresholds_path),
                                       "minimum_development_scenes": minimum_scenes,
                                       "reference": str(reference.path), "reference_artifact_sha256": ref_hashes,
                                       "runs": calibration_runs,
                                       "threshold_selection": "explicit draft; no thresholds inferred or adjusted"}}
    if exact:
        policy["calibration_evidence"].update(
            exact_runs=draft["exact_runs"],
            threshold_selection="Fixed zero-error equivalence tested on every development output; no fitted acceptability limits",
            quality_rationale="Identical newly generated RGB values preserve reference outputs exactly; a subjective perceptual threshold is unnecessary. This does not establish reference driving validity.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also prevents concurrent calibration jobs from replacing
    # the policy after an evaluation has started using it.
    with destination.open("x") as stream:
        yaml.safe_dump(policy, stream, sort_keys=False)
    return policy


def exact_metrics():
    return {"max_abs_error": {"maximum": 0.0, "aggregation": "max"},
            "rmse": {"maximum": 0.0, "aggregation": "mean"},
            "temporal_derivative_mae": {"maximum": 0.0, "aggregation": "mean"}}


def freeze_exact_tolerances(evaluation_path, runs, output, *, minimum_scenes=32):
    """Freeze a zero-error policy only after complete real development proof."""
    destination = Path(output)
    if destination.exists():
        raise FileExistsError("Frozen tolerances cannot be overwritten")
    draft = destination.with_suffix(destination.suffix + ".zero-draft.yaml")
    destination.parent.mkdir(parents=True, exist_ok=True)
    value = {"policy_type": "exact_output_equivalence", "exact_runs": list(runs), "metrics": exact_metrics()}
    if draft.exists():
        if yaml.safe_load(draft.read_text()) != value:
            raise ValueError("Existing exact-policy draft differs")
    else:
        with draft.open("x") as stream:
            yaml.safe_dump(value, stream)
    return freeze_tolerances(evaluation_path, draft, destination, minimum_scenes=minimum_scenes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-evaluation", required=True)
    parser.add_argument("--thresholds", required=True, help="YAML draft with explicitly selected metric maxima")
    parser.add_argument("--output", required=True, help="new versioned frozen YAML file")
    parser.add_argument("--minimum-scenes", type=int, default=32)
    args = parser.parse_args()
    try:
        policy = freeze_tolerances(args.development_evaluation, args.thresholds, args.output,
                                   minimum_scenes=args.minimum_scenes)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"drivebench freeze: {error}\n")
    print(json.dumps({"output": args.output, "frozen_at": policy["frozen_at"],
                      "development_scenes": len(policy["development_scene_ids"]),
                      "performance_calibrated": False}, indent=2))


if __name__ == "__main__":
    main()
