"""Prepared selection checks use synthetic metadata and tiny byte fixtures."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

from drivebench.data.manifests import write_manifest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_prepared_selection.py"
spec = importlib.util.spec_from_file_location("verify_prepared_selection", SCRIPT)
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)
A, B = "samples/CAM_FRONT/a.jpg", "sweeps/CAM_FRONT/b.jpg"


@pytest.fixture
def selection(tmp_path):
    tmp_path = tmp_path.resolve()
    plan, prepared, root, annotations = [tmp_path / name for name in ("plan", "manifests", "data", "annotations")]
    for directory in (plan, prepared, root, annotations):
        directory.mkdir()
    annotation = annotations / "nuScenes_val.json"
    annotation.write_text("[]\n")
    for name, data in ((A, b"observed bytes"), (B, b"future bytes")):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    identities = {name: verify.base.file_identity(root / name) for name in (A, B)}
    required = plan / "required.txt"
    required.write_text(A + "\n" + B + "\n")
    list_sha = verify.sha(required)
    (plan / "plan.json").write_text(json.dumps({"metadata_only_plan": True,
        "required_camera_file_list_sha256": list_sha, "unique_camera_files_required": 2,
        "clips_per_manifest": {name: 2 for name in verify.ROLES}}))
    acquisition = tmp_path / "merged.json"
    acquisition.write_text(json.dumps({"status": "complete", "missing_files": [],
        "required_file_list_sha256": list_sha,
        "verified_archives": {"1": {"status": "verified", "files": identities}}}))
    for name, role in verify.ROLES.items():
        rows = [{"request_id": f"{name}-{index}", "scene_id": f"scene-{index}",
                 "annotation_source": {"path": "/old-machine/nuScenes_val.json",
                                       "sha256": verify.sha(annotation), "index": index},
                 "observed_frames": [A], "future_frames": [B], "seeds": [100 + index, 200 + index],
                 "trajectory": [[0.0, 1.0], [0.0, 2.0]], "future_timestamps_us": [123]}
                for index in range(2)]
        (plan / f"{name}.planned.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        for row in rows:
            row["annotation_source"]["path"] = "/new-machine/nuScenes_val.json"
            row["frame_checksums"] = {key: value["sha256"] for key, value in identities.items()}
        meta = {"dataset": "nuscenes", "synthetic": False, "split": name, "partition_role": role,
                "data_root": "/new-machine/data/nuscenes"}
        if name in verify.PARENTS:
            parent = verify.PARENTS[name]
            meta.update(parent_manifest_role=parent, parent_manifest_sha256=verify.sha(prepared / f"{parent}.jsonl"),
                        selected_parent_request_ids=[row["request_id"] for row in rows])
        write_manifest(prepared / f"{name}.jsonl", rows, meta)
    return plan, prepared, root, annotations, acquisition, required


def rewrite(selection, mutate, name="development"):
    path = selection[1] / f"{name}.jsonl"
    rows = verify.records(path)
    mutate(rows)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_exact_plan_plus_checksums_and_verified_path_relocation_passes(selection):
    result = verify.verify_selection(*selection)
    assert result["status"] == "passed" and result["verified_unique_frame_count"] == 2
    assert set(result["manifests"]) == set(verify.ROLES)
    assert result["verified_data_root"] == str(selection[2])


@pytest.mark.parametrize("field,value", [("seeds", [999, 998]), ("trajectory", [[1, 2]]),
                                          ("scene_id", "different"), ("future_timestamps_us", [999])])
def test_changed_selection_action_seed_or_timing_is_rejected(selection, field, value):
    rewrite(selection, lambda rows: rows[0].update({field: value}))
    with pytest.raises(verify.base.IntegrityError, match="Planned record changed"):
        verify.verify_selection(*selection)


def test_changed_record_order_is_rejected(selection):
    rewrite(selection, lambda rows: rows.reverse())
    with pytest.raises(verify.base.IntegrityError, match="Planned record changed"):
        verify.verify_selection(*selection)


def test_missing_frame_checksum_is_rejected(selection):
    rewrite(selection, lambda rows: rows[0]["frame_checksums"].pop(B))
    with pytest.raises(verify.base.IntegrityError, match="checksum coverage"):
        verify.verify_selection(*selection)


def test_frame_hash_not_matching_acquisition_is_rejected(selection):
    rewrite(selection, lambda rows: rows[0]["frame_checksums"].update({B: "0" * 64}))
    with pytest.raises(verify.base.IntegrityError, match="verified acquisition"):
        verify.verify_selection(*selection)


def test_changed_actual_frame_bytes_are_rejected(selection):
    (selection[2] / B).write_bytes(b"corrupted in transfer")
    with pytest.raises(verify.base.IntegrityError, match="Image bytes differ"):
        verify.verify_selection(*selection)


def test_annotation_relocation_cannot_change_content(selection):
    (selection[3] / "nuScenes_val.json").write_text("[1]\n")
    with pytest.raises(verify.base.IntegrityError, match="Annotation content differs"):
        verify.verify_selection(*selection)


def test_incomplete_acquisition_is_rejected(selection):
    path = selection[4]
    report = json.loads(path.read_text())
    report["status"] = "incomplete"
    path.write_text(json.dumps(report))
    with pytest.raises(verify.base.IntegrityError, match="not complete"):
        verify.verify_selection(*selection)


def test_wrong_partition_role_in_sidecar_is_rejected(selection):
    path = selection[1] / "evaluation.jsonl.meta.json"
    meta = json.loads(path.read_text())
    meta["partition_role"] = "development"
    path.write_text(json.dumps(meta))
    with pytest.raises(verify.base.IntegrityError, match="metadata identity/role"):
        verify.verify_selection(*selection)
