#!/usr/bin/env python3
"""Verify prepared nuScenes manifests against the frozen metadata-only plan.

No image decoding, inference or GPU dependencies. Every record must be equal in
order and content after adding frame_checksums and relocating annotation paths;
annotation content hashes, indices and filenames must remain identical. Check
all selected image bytes against both prepared manifests and merged acquisition
evidence. Local verification may use relocated data without editing manifests.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import download_nuscenes_subset as base
from merge_nuscenes_subsets import checked_directory, create_report

ROLES = {"development": "development", "evaluation": "held_out_evaluation",
         "calibration": "training_calibration", "smoke": "development_subset",
         "evaluation_second_seed": "held_out_evaluation_second_seed"}
PARENTS = {"smoke": "development", "evaluation_second_seed": "evaluation"}


def sha(path):
    return base.file_identity(path)["sha256"]


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def records(path):
    result = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [record["request_id"] for record in result]
    if len(ids) != len(set(ids)):
        raise base.IntegrityError("Duplicate request IDs")
    return result


def normalized(record, annotation_root, annotation_hashes, *, prepared):
    value = copy.deepcopy(record)
    checksums = value.pop("frame_checksums", None)
    if prepared and not isinstance(checksums, dict):
        raise base.IntegrityError("Prepared record lacks frame checksums")
    if not prepared and checksums is not None:
        raise base.IntegrityError("Expected a metadata-only selection plan")
    annotation = value["annotation_source"]
    name = Path(annotation["path"]).name
    if name not in {"nuScenes.json", "nuScenes_val.json"}:
        raise base.IntegrityError("Unknown annotation filename")
    if name not in annotation_hashes:
        annotation_hashes[name] = sha(annotation_root / name)
    if annotation["sha256"] != annotation_hashes[name]:
        raise base.IntegrityError("Annotation content differs from planned identity")
    annotation["path"] = name
    return value, checksums


def verify_selection(plan_dir, manifest_dir, root, annotation_root, acquisition_path, required_path):
    plan_dir, manifest_dir = Path(plan_dir), Path(manifest_dir)
    root, annotation_root = checked_directory(root), checked_directory(annotation_root)
    required, list_hash = base.required_files(required_path)
    plan = json.loads((plan_dir / "plan.json").read_text())
    if (plan.get("required_camera_file_list_sha256") != list_hash
            or plan.get("unique_camera_files_required") != len(required)
            or plan.get("metadata_only_plan") is not True):
        raise base.IntegrityError("Plan required-file binding differs")
    acquisition = json.loads(Path(acquisition_path).read_text())
    if (acquisition.get("status") != "complete" or acquisition.get("missing_files") != []
            or acquisition.get("required_file_list_sha256") != list_hash):
        raise base.IntegrityError("Acquisition is not complete for this selection")
    acquired = {}
    for entry in acquisition["verified_archives"].values():
        if entry.get("status") != "verified":
            raise base.IntegrityError("Acquisition contains an unverified archive")
        for name, identity in entry["files"].items():
            if name in acquired and acquired[name] != identity:
                raise base.IntegrityError("Acquisition sources disagree")
            acquired[name] = identity
    if set(acquired) != required:
        raise base.IntegrityError("Acquisition does not cover the exact required file set")
    all_frames, annotation_hashes, verified = {}, {}, {}
    for name, role in ROLES.items():
        planned_path, prepared_path = plan_dir / f"{name}.planned.jsonl", manifest_dir / f"{name}.jsonl"
        planned, prepared = records(planned_path), records(prepared_path)
        if len(planned) != len(prepared) or len(planned) != plan["clips_per_manifest"][name]:
            raise base.IntegrityError(f"Selection count changed: {name}")
        for index, (before, after) in enumerate(zip(planned, prepared)):
            expected, _ = normalized(before, annotation_root, annotation_hashes, prepared=False)
            actual, checksums = normalized(after, annotation_root, annotation_hashes, prepared=True)
            if expected != actual:
                changed = sorted(key for key in set(expected) | set(actual) if expected.get(key) != actual.get(key))
                raise base.IntegrityError(f"Planned record changed: {name}[{index}], fields={changed}")
            paths = set(after["observed_frames"] + after["future_frames"])
            if set(checksums) != paths:
                raise base.IntegrityError("Record checksum coverage differs from its frame paths")
            for path, digest in checksums.items():
                if path not in acquired or acquired[path]["sha256"] != digest:
                    raise base.IntegrityError("Manifest frame hash differs from verified acquisition")
                if path in all_frames and all_frames[path] != digest:
                    raise base.IntegrityError("Manifest frame hashes disagree")
                all_frames[path] = digest
        meta_path = prepared_path.with_suffix(prepared_path.suffix + ".meta.json")
        meta = json.loads(meta_path.read_text())
        scenes = sorted({record["scene_id"] for record in prepared})
        scenes_sha = hashlib.sha256(json.dumps(scenes, separators=(",", ":")).encode()).hexdigest()
        if (meta.get("manifest_sha256") != sha(prepared_path) or meta.get("clip_count") != len(prepared)
                or meta.get("scene_ids") != scenes or meta.get("scene_count") != len(scenes)
                or meta.get("scene_ids_sha256") != scenes_sha or meta.get("partition_role") != role
                or meta.get("split") != name or meta.get("dataset") != "nuscenes"
                or meta.get("synthetic") is not False):
            raise base.IntegrityError(f"Manifest metadata identity/role differs: {name}")
        if name in PARENTS:
            parent = PARENTS[name]
            parent_ids = [record.get("parent_request_id", record["request_id"]) for record in prepared]
            if (meta.get("parent_manifest_role") != parent
                    or meta.get("parent_manifest_sha256") != sha(manifest_dir / f"{parent}.jsonl")
                    or meta.get("selected_parent_request_ids") != parent_ids):
                raise base.IntegrityError("Derivative manifest parent provenance differs")
        verified[name] = {"plan_sha256": sha(planned_path), "manifest_sha256": sha(prepared_path),
                          "metadata_sha256": sha(meta_path), "clip_count": len(prepared),
                          "scene_count": len(scenes), "partition_role": role,
                          "ordered_request_ids_sha256": json_digest([record["request_id"] for record in prepared]),
                          "recorded_data_root": meta.get("data_root")}
    if set(all_frames) != required:
        raise base.IntegrityError("Prepared manifests do not cover the exact required file set")
    for name, identity in acquired.items():
        if base.camera_path(name) != name:
            raise base.IntegrityError("Unsafe acquired frame path")
        checked_directory((root / name).parent)
        if base.file_identity(root / name) != identity:
            raise base.IntegrityError("Image bytes differ from verified acquisition")
    return {"schema_version": 1, "status": "passed", "created_at": datetime.now(timezone.utc).isoformat(),
            "scope": "Metadata equality and selected file integrity only; no model quality or benchmark claim",
            "allowed_record_changes": ["frame_checksums added", "annotation_source.path relocated with same basename, SHA256 and index"],
            "verified_data_root": str(root), "plan_sha256": sha(plan_dir / "plan.json"),
            "acquisition_report_sha256": sha(acquisition_path), "required_file_list_sha256": list_hash,
            "verified_unique_frame_count": len(all_frames), "verified_frame_bytes": sum(item["size_bytes"] for item in acquired.values()),
            "frame_identities_sha256": json_digest(acquired), "annotation_sha256": annotation_hashes,
            "manifests": verified}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", type=Path, default=Path("data/selection-plan"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--root", type=Path, default=Path("data/nuscenes"))
    parser.add_argument("--annotation-root", type=Path, default=Path("data/annotations"))
    parser.add_argument("--acquisition-report", type=Path, default=Path("outputs/nuscenes-merged.json"))
    parser.add_argument("--required-file-list", type=Path, default=Path("data/selection-plan/required-camera-files.txt"))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.report.exists():
            raise base.IntegrityError("Output receipt already exists")
        result = verify_selection(args.plan_dir, args.manifest_dir, args.root, args.annotation_root,
                                  args.acquisition_report, args.required_file_list)
        create_report(args.report, result)
        base.emit("prepared_selection_verified", unique_frames=result["verified_unique_frame_count"], report=str(args.report))
    except (base.IntegrityError, OSError, ValueError, KeyError, TypeError) as exc:
        base.emit("prepared_selection_failed", error=type(exc).__name__, message=str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
