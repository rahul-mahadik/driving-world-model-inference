"""Fabricated metadata tests policy provenance; no GPU/quality result is claimed."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest
import yaml

from drivebench.config import digest, file_sha256
from drivebench.evaluation.study_policy import policy_for_comparison, reviewed_policy_binding
from drivebench.freeze import freeze_tolerances


def fixture(tmp_path):
    study = tmp_path / "study"
    scenes = [f"development-{i:02d}" for i in range(32)]
    metadata = {"partition_role": "development", "manifest_sha256": "d" * 64, "clip_count": 32,
                "scene_ids": scenes, "scene_ids_sha256": digest(scenes)}
    artifacts = {}
    paths = {}
    for name in ("upstream", "compiled"):
        path = study / "quality/development/aggregate/runs" / name
        path.mkdir(parents=True)
        for file, value in {"config.json": {"model": {"backend": "vista"}, "data": {
                "manifest_sha256": "d" * 64, "manifest_metadata": metadata}},
                "environment.json": {}, "summary.json": {"backend": "vista", "failures": 0}}.items():
            (path / file).write_text(json.dumps(value))
        (path / "measurements.jsonl").write_text("".join(json.dumps({"status": "ok", "request_id": s, "scene_id": s}) + "\n" for s in scenes))
        paths[name] = str(path)
        artifacts[name] = {file: file_sha256(path / file) for file in ("config.json", "environment.json", "summary.json", "measurements.jsonl")}
    draft = {"metrics": {"rmse": {"maximum": .02}, "lpips": {"maximum": .05}, "temporal_derivative_mae": {"maximum": .03}}}
    evaluation = {"reference": paths["upstream"], "reference_artifact_sha256": artifacts["upstream"],
        "perceptual": {"status": "available"}, "runs": [{"run": "compiled", "run_path": paths["compiled"],
        "artifact_sha256": artifacts["compiled"], "synthetic": False, "complete_pairs": True, "errors": [],
        "matched_pairs": 32, "pairs": [{"scene_id": s, "request_id": s} for s in scenes],
        "aggregates": {name: {"mean": .01, "pairs": 32} for name in draft["metrics"]}}]}
    evaluation_path = study / "quality/development/aggregate/evaluation-vs-upstream.json"
    evaluation_path.write_text(json.dumps(evaluation))
    draft_path = tmp_path / "explicit-draft.yaml"; draft_path.write_text(yaml.safe_dump(draft))
    policy = tmp_path / "reviewed-v1.yaml"
    freeze_tolerances(evaluation_path, draft_path, policy)
    plan = {"reference": "upstream", "optimized_reference": "conditioning-reuse",
            "manifests": {"development": {"sha256": "d" * 64, "metadata": metadata}}}
    return study, plan, policy, evaluation_path


def test_binding_preserves_exact_policy_and_is_reference_specific(tmp_path):
    study, plan, policy, _ = fixture(tmp_path)
    binding = reviewed_policy_binding(study, plan, "upstream", policy)
    assert binding["calibrated_variants"] == ["compiled"]
    assert binding["review_acceptance_established"] is False
    selection = {"quality_policy": "exact-zero.yaml", "exact_runs": ["conditioning-reuse"],
                 "reviewed_quality_policies": {"upstream": binding}}
    assert policy_for_comparison(selection, "upstream", "compiled") == str(policy.resolve())
    assert policy_for_comparison(selection, "conditioning-reuse", "compiled") == "exact-zero.yaml"
    assert policy_for_comparison(selection, "upstream", "conditioning-reuse") == "exact-zero.yaml"
    with pytest.raises(ValueError, match="this reference"):
        reviewed_policy_binding(study, plan, "conditioning-reuse", policy)


@pytest.mark.parametrize("mutation", ["scene_subset", "evaluation", "run_artifact", "observed_metrics", "future", "old_policy_type"])
def test_reviewed_policy_rejects_changed_or_incomplete_provenance(tmp_path, mutation):
    study, plan, policy, evaluation = fixture(tmp_path)
    value = yaml.safe_load(policy.read_text())
    if mutation == "scene_subset":
        value["development_scene_ids"] = value["development_scene_ids"][:16]
    elif mutation == "evaluation":
        evaluation.write_text(evaluation.read_text() + "\n")
    elif mutation == "run_artifact":
        source = Path(value["calibration_evidence"]["runs"][0]["path"]) / "summary.json"
        source.write_text(source.read_text() + "\n")
    elif mutation == "observed_metrics":
        value["calibration_evidence"]["runs"][0]["observed_metrics"]["rmse"] = .009
    elif mutation == "future":
        value["frozen_at"] = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    else:
        value["policy_type"] = "exact_output_equivalence"
    policy.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError):
        reviewed_policy_binding(study, plan, "upstream", policy)


def test_policy_must_precede_every_heldout_attempt(tmp_path):
    study, plan, policy, _ = fixture(tmp_path)
    frozen = datetime.fromisoformat(yaml.safe_load(policy.read_text())["frozen_at"])
    path = study / "quality/evaluation/shards/000/attempts/compiled/attempt-000/compiled/config.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"started_at": (frozen - timedelta(seconds=1)).isoformat()}))
    with pytest.raises(ValueError, match="before held-out"):
        reviewed_policy_binding(study, plan, "upstream", policy)
    path.write_text(json.dumps({"started_at": (frozen + timedelta(seconds=1)).isoformat()}))
    assert reviewed_policy_binding(study, plan, "upstream", policy)["reference"] == "upstream"
