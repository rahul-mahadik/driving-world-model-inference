"""Bind separately frozen nonzero tolerances to one study reference.

This verifies metric calibration provenance, not acceptance of blinded reviews.
The downstream nonzero quality gate still requires separately verified review.
"""
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import yaml

from ..config import file_sha256
from ..freeze import METRICS, _checked_run


def _time(value):
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Quality policy timestamps require a timezone")
    return result


def reviewed_policy_binding(destination, plan, reference, path):
    destination, path = Path(destination), Path(path).resolve()
    if reference not in {plan["reference"], plan["optimized_reference"]}:
        raise ValueError("Reviewed policy must name an actual study reference")
    policy = yaml.safe_load(path.read_text())
    if (policy.get("policy_type") != "reviewed_metric_tolerances" or policy.get("calibrated") is not True
            or policy.get("frozen") is not True or policy.get("selected_on") != "development"):
        raise ValueError("Use a separately frozen development metric policy; exact policy remains separate")
    if policy.get("performance", {}).get("calibrated"):
        raise ValueError("Reviewed quality policy cannot calibrate timing variability")
    frozen = _time(policy.get("frozen_at"))
    if frozen > datetime.now(timezone.utc):
        raise ValueError("Reviewed policy freeze cannot be in the future")
    metrics = policy.get("metrics", {})
    if (not ({"mae", "rmse", "max_abs_error"} & metrics.keys()) or "lpips" not in metrics
            or not ({"temporal_change_error", "freeze_fraction_increase", "temporal_derivative_mae"} & metrics.keys())):
        raise ValueError("Reviewed policy requires numerical, perceptual and temporal limits")
    for name, setting in metrics.items():
        value = setting.get("maximum")
        if (name not in METRICS or setting.get("aggregation", "mean") not in {"mean", "max"}
                or type(value) not in (int, float) or not math.isfinite(value) or value < 0):
            raise ValueError("Invalid reviewed metric threshold")
    evidence = policy.get("calibration_evidence", {})
    evaluation = (destination / "quality/development/aggregate" / f"evaluation-vs-{reference}.json").resolve()
    if (Path(evidence.get("evaluation_path", "")).resolve() != evaluation
            or evidence.get("evaluation_sha256") != file_sha256(evaluation)):
        raise ValueError("Reviewed policy is not bound to this reference's unchanged development evaluation")
    document = json.loads(evaluation.read_text())
    if document.get("perceptual", {}).get("status") != "available":
        raise ValueError("Reviewed policy requires measured development LPIPS")
    if document.get("created_at") and _time(document["created_at"]) > frozen:
        raise ValueError("Reviewed policy predates its calibration evaluation")
    manifest = plan["manifests"]["development"]
    scenes = sorted(manifest["metadata"]["scene_ids"])
    if (len(scenes) != 32 or policy.get("development_manifest_sha256") != manifest["sha256"]
            or sorted(policy.get("development_scene_ids", [])) != scenes
            or evidence.get("minimum_development_scenes", 0) < 32):
        raise ValueError("Reviewed policy must cover the study's same 32 development scenes")
    if (Path(evidence.get("reference", "")).resolve() != Path(document["reference"]).resolve()
            or evidence.get("reference_artifact_sha256") != document["reference_artifact_sha256"]):
        raise ValueError("Reviewed policy reference provenance differs")
    ref, manifest_hash, ref_scenes, _ = _checked_run(document["reference"], document["reference_artifact_sha256"], 32)
    if manifest_hash != manifest["sha256"] or ref_scenes != scenes:
        raise ValueError("Reviewed policy reference uses a different development partition")
    entries = {str(Path(row["run_path"]).resolve()): row for row in document["runs"]}
    calibrated, sources = [], [{"path": str(ref.path), "artifact_sha256": document["reference_artifact_sha256"]}]
    for row in evidence.get("runs", []):
        key = str(Path(row["path"]).resolve())
        entry = entries.get(key)
        if (not entry or entry.get("artifact_sha256") != row["artifact_sha256"] or entry.get("synthetic")
                or not entry.get("complete_pairs") or entry.get("errors")):
            raise ValueError("Reviewed policy calibration run is absent or incomplete")
        run, run_manifest, run_scenes, _ = _checked_run(key, row["artifact_sha256"], 32)
        pairs = entry.get("pairs", [])
        if (run_manifest != manifest["sha256"] or run_scenes != scenes
                or {p["request_id"] for p in pairs} != {r["request_id"] for r in run.measurements}
                or {p["scene_id"] for p in pairs} != set(scenes)
                or len(pairs) != row["matched_pairs"] or len(pairs) != entry.get("matched_pairs")):
            raise ValueError("Reviewed policy calibration does not cover every development request")
        for name, setting in metrics.items():
            aggregate = entry.get("aggregates", {}).get(name, {})
            if (aggregate.get(setting.get("aggregation", "mean")) != row.get("observed_metrics", {}).get(name)
                    or aggregate.get("pairs") != len(pairs)):
                raise ValueError("Reviewed policy observed calibration metrics changed")
        calibrated.append(entry["run"])
        sources.append({"path": key, "artifact_sha256": row["artifact_sha256"]})
    if not calibrated or len(set(calibrated)) != len(calibrated):
        raise ValueError("Reviewed policy needs distinct calibrated variants")
    # Small metadata only; no generated video arrays are read.
    for phase in ("evaluation", "evaluation_second_seed", "evaluation_actions"):
        for config in (destination / "quality" / phase).rglob("config.json"):
            started = json.loads(config.read_text()).get("started_at")
            if not started or _time(started) <= frozen:
                raise ValueError("Reviewed policy must be frozen before held-out generation")
    return {"path": str(path), "sha256": file_sha256(path), "reference": reference,
            "frozen_at": policy["frozen_at"], "evaluation_path": str(evaluation),
            "evaluation_sha256": file_sha256(evaluation), "calibrated_variants": sorted(calibrated),
            "source_runs": sources, "review_acceptance_established": False,
            "scope": "Reference-specific frozen metric limits; accepted blinded review remains a separate required gate"}


def policy_for_comparison(selection, reference, variant):
    if variant not in selection.get("exact_runs", []):
        binding = selection.get("reviewed_quality_policies", {}).get(reference)
        if binding and variant in binding["calibrated_variants"]:
            return binding["path"]
    return selection["quality_policy"]
