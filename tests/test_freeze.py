"""Artifact-policy tests use miniature fabricated records, never GPU claims."""
import json
from pathlib import Path

import pytest
import yaml

from drivebench.config import digest, file_sha256
from drivebench.freeze import ARTIFACTS, freeze_tolerances


def fixture(tmp_path):
    scenes = ["dev-a", "dev-b"]
    metadata = {"partition_role": "development", "manifest_sha256": "d" * 64, "clip_count": 2,
                "scene_ids": scenes, "scene_ids_sha256": digest(scenes)}
    paths, hashes = [], []
    for name in ("reference", "candidate"):
        path = tmp_path / name
        path.mkdir()
        (path / "config.json").write_text(json.dumps({"model": {"backend": "vista"},
            "data": {"manifest_sha256": "d" * 64, "manifest_metadata": metadata}}))
        (path / "environment.json").write_text('{}')
        (path / "summary.json").write_text(json.dumps({"backend": "vista", "failures": 0}))
        (path / "measurements.jsonl").write_text(''.join(json.dumps({"status": "ok", "scene_id": scene, "request_id": scene}) + '\n' for scene in scenes))
        paths.append(path)
        hashes.append({name: file_sha256(path / name) for name in ARTIFACTS})
    draft = {"metrics": {"rmse": {"maximum": 0.02}, "lpips": {"maximum": 0.05},
                         "temporal_derivative_mae": {"maximum": 0.03}}}
    entry = {"run_path": str(paths[1]), "artifact_sha256": hashes[1], "complete_pairs": True,
             "synthetic": False, "errors": [], "matched_pairs": 2,
             "pairs": [{"scene_id": scene, "request_id": scene} for scene in scenes],
             "aggregates": {name: {"mean": 0.01, "pairs": 2} for name in draft["metrics"]}}
    evaluation = {"reference": str(paths[0]), "reference_artifact_sha256": hashes[0],
                  "perceptual": {"status": "available"}, "runs": [entry]}
    eval_path, draft_path = tmp_path / "evaluation.json", tmp_path / "thresholds.yaml"
    eval_path.write_text(json.dumps(evaluation))
    draft_path.write_text(yaml.safe_dump(draft))
    return eval_path, draft_path, evaluation, draft


def test_freeze_records_evidence_without_changing_selected_thresholds(tmp_path):
    evaluation, thresholds, _, draft = fixture(tmp_path)
    output = tmp_path / "frozen.yaml"
    result = freeze_tolerances(evaluation, thresholds, output, minimum_scenes=2)
    assert result["metrics"] == draft["metrics"]
    assert result["calibrated"] and result["frozen"]
    assert not result["performance"]["calibrated"]
    assert result["calibration_evidence"]["evaluation_sha256"] == file_sha256(evaluation)
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        freeze_tolerances(evaluation, thresholds, output, minimum_scenes=2)
    assert output.read_bytes() == original


def test_stale_evaluation_cannot_freeze(tmp_path):
    evaluation, thresholds, data, _ = fixture(tmp_path)
    path = Path(data["runs"][0]["run_path"]) / "measurements.jsonl"
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match="changed after evaluation"):
        freeze_tolerances(evaluation, thresholds, tmp_path / "frozen.yaml", minimum_scenes=2)


@pytest.mark.parametrize("change,error", [
    ("synthetic", "successful real"), ("evaluation", "Only the development"),
    ("missing_scene", "cover the declared"),
])
def test_unqualified_runs_rejected(tmp_path, change, error):
    evaluation, thresholds, data, _ = fixture(tmp_path)
    path = Path(data["reference"])
    config = json.loads((path / "config.json").read_text())
    if change == "synthetic":
        (path / "summary.json").write_text(json.dumps({"backend": "synthetic"}))
    elif change == "evaluation":
        config["data"]["manifest_metadata"]["partition_role"] = "held_out_evaluation"
        (path / "config.json").write_text(json.dumps(config))
    else:
        (path / "measurements.jsonl").write_text(json.dumps({"scene_id": "dev-a", "status": "ok"}) + '\n')
    data["reference_artifact_sha256"] = {name: file_sha256(path / name) for name in ARTIFACTS}
    evaluation.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=error):
        freeze_tolerances(evaluation, thresholds, tmp_path / "frozen.yaml", minimum_scenes=2)


def test_metric_coverage_and_minimum_scene_requirement(tmp_path):
    evaluation, thresholds, data, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="minimum 32"):
        freeze_tolerances(evaluation, thresholds, tmp_path / "frozen.yaml")
    data["runs"][0]["aggregates"]["lpips"]["pairs"] = 1
    evaluation.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="missing pairs"):
        freeze_tolerances(evaluation, thresholds, tmp_path / "frozen.yaml", minimum_scenes=2)


def test_missing_perceptual_values_and_uncalibrated_thresholds(tmp_path):
    evaluation, thresholds, data, draft = fixture(tmp_path)
    data["perceptual"]["status"] = "unavailable"
    evaluation.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="actually be available"):
        freeze_tolerances(evaluation, thresholds, tmp_path / "frozen.yaml", minimum_scenes=2)
    draft["metrics"]["lpips"]["maximum"] = float("nan")
    thresholds.write_text(yaml.safe_dump(draft))
    with pytest.raises(ValueError, match="Invalid explicit threshold"):
        freeze_tolerances(evaluation, thresholds, tmp_path / "frozen.yaml", minimum_scenes=2)


def test_saved_subset_cannot_freeze_whole_development_policy(tmp_path):
    evaluation, thresholds, data, _ = fixture(tmp_path)
    # Keep both scenes represented while substituting a different clip identity.
    data["runs"][0]["pairs"][1]["request_id"] = "unsaved-development-clip"
    evaluation.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="every declared clip"):
        freeze_tolerances(evaluation, thresholds, tmp_path / "frozen.yaml", minimum_scenes=2)
