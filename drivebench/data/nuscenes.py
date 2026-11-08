"""Adapter for Vista translated annotations plus official nuScenes metadata.

The adapter deliberately does not reconstruct or rotate actions: upstream's
translation is external to the pinned repository. Verified action semantics must
be supplied alongside the translated annotations before alternative actions run.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import ijson
import numpy as np

from .manifests import resolve_frame, sha256_file
from .trajectories import candidate_seed, upstream_trajectory, validate_trajectory_metadata


def _read_table(directory: Path, name: str) -> Iterator[dict[str, Any]]:
    """Yield one metadata object at a time without loading the full JSON array.

    nuScenes sample_data is much larger than the selected experiment subset.
    The caller can discard unrelated objects as they arrive. Binary input keeps
    parser buffering bounded; use_float preserves json.load's numeric types for
    timestamps, calibration values, and later JSON manifest serialization.
    """
    path = directory / f"{name}.json"
    with path.open("rb") as stream:
        # ijson.items(..., 'item') alone silently yields no rows for an object
        # root. Check the first non-whitespace token before using the fast parser.
        # Whitespace may span buffers, so never read the whole prefix at once.
        while True:
            chunk = stream.read(64 * 1024)
            if not chunk:
                raise ValueError(f"JSON table is empty: {path}")
            prefix = chunk.lstrip(b" \t\r\n")
            if prefix:
                if prefix[:1] != b"[":
                    raise ValueError(f"JSON table must contain an array of objects: {path}")
                break
        stream.seek(0)
        try:
            for index, row in enumerate(ijson.items(stream, "item", use_float=True)):
                if not isinstance(row, dict):
                    raise ValueError(f"JSON table row {index} must be an object: {path}")
                yield row
        except ijson.JSONError as exc:
            raise ValueError(f"Malformed JSON table {path}: {exc}") from exc


def _index(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["token"]: row for row in rows}


def _categories(description: str, annotation: dict[str, Any]) -> list[str]:
    text = description.lower()
    categories = set(annotation.get("categories", []))
    # Description-derived labels describe the scene, not frame-level detections.
    for word, label in [("night", "lighting:night"), ("rain", "weather:rain"),
                        ("turn", "motion:turning"), ("pedestrian", "scene:pedestrians"),
                        ("vehicle", "scene:vehicles"), ("traffic", "scene:traffic")]:
        if word in text:
            categories.add(label)
    if not any(category.startswith("lighting:") for category in categories):
        categories.add("lighting:unknown")
    if not categories:
        categories.add("unknown")
    return sorted(categories)


def checksum_record_frames(records: list[dict[str, Any]], root: str | Path) -> None:
    """Hash only a selected subset, reading each shared image at most once."""
    checksums: dict[str, str] = {}
    for record in records:
        paths = record["observed_frames"] + record["future_frames"]
        for frame in paths:
            if frame not in checksums:
                checksums[frame] = sha256_file(resolve_frame(root, frame))
        record["frame_checksums"] = {frame: checksums[frame] for frame in paths}


def _lidar_trajectory_timing(first_camera: dict[str, Any], scene_id: str,
                             samples: dict[str, dict[str, Any]],
                             lidar_keyframes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Resolve translator waypoint times without transforming its coordinates."""
    if first_camera.get("is_key_frame") is not True:
        raise ValueError("Derived trajectory timestamps require the first CAM_FRONT frame to be a keyframe")
    sample_tokens = [first_camera["sample_token"]]
    for _ in range(4):
        current = samples.get(sample_tokens[-1])
        following = current.get("next") if current else None
        if not following or following not in samples:
            raise ValueError("Derived trajectory timestamps require four available future samples")
        if following in sample_tokens:
            raise ValueError("Trajectory sample chain contains a cycle")
        if samples[following].get("scene_token") != scene_id:
            raise ValueError("Trajectory sample chain crosses a scene boundary")
        sample_tokens.append(following)
    missing = [token for token in sample_tokens if token not in lidar_keyframes]
    if missing:
        raise ValueError(f"Missing LIDAR_TOP keyframe for trajectory samples: {missing}")
    lidar = [lidar_keyframes[token] for token in sample_tokens]
    lidar_times = [int(row["timestamp"]) for row in lidar]
    if not np.all(np.diff(lidar_times) > 0):
        raise ValueError("Trajectory LIDAR_TOP keyframe timestamps must strictly increase")
    camera_timestamp = int(first_camera["timestamp"])
    return {"reference_timestamp_us": camera_timestamp,
            "spatial_reference_timestamp_us": lidar_times[0],
            "spatial_reference_sample_token": sample_tokens[0],
            "spatial_reference_lidar_token": lidar[0]["token"],
            "action_sample_tokens": sample_tokens[1:],
            "action_lidar_tokens": [row["token"] for row in lidar[1:]],
            "action_timestamps_us": lidar_times[1:],
            "timestamps_seconds": [(timestamp - camera_timestamp) / 1_000_000 for timestamp in lidar_times[1:]]}


def load_nuscenes_records(config: dict[str, Any], *, hash_frames: bool = True) -> list[dict[str, Any]]:
    """Join annotation metadata; preparation defers image I/O until selection."""
    root = Path(config["data_root"]).resolve()
    tables = root / config.get("version", "v1.0-trainval")
    frame_count = int(config.get("num_frames", 25))
    observed_count = int(config.get("conditioning_frames", 1))
    if not 1 <= observed_count < frame_count:
        raise ValueError("conditioning_frames must be positive and smaller than num_frames")
    metadata = dict(config.get("trajectory_metadata") or {})
    validate_trajectory_metadata(metadata, allow_derived_timestamps=True)
    derive_times = metadata.get("timestamp_source") == "nuscenes_lidar_top_keyframes"
    annotations = []
    for source in config.get("annotation_sources", []):
        split = source["split"]
        if split not in {"train", "val"}:
            raise ValueError("Annotation source split must be train or val")
        path = Path(source["path"])
        checksum = sha256_file(path)
        expected_checksum = source.get("sha256")
        if expected_checksum is not None and (not isinstance(expected_checksum, str)
                                              or checksum != expected_checksum.lower()):
            raise ValueError(f"Annotation source checksum mismatch: {path}")
        with path.open() as stream:
            rows = json.load(stream)
        annotations.extend((row, split, str(path.resolve()), checksum, index) for index, row in enumerate(rows))
    if not annotations:
        raise ValueError("Provide Vista translated train and validation annotation_sources")
    needed_paths = {str(Path(frame)) for row, *_ in annotations for frame in row["frames"][:frame_count]}
    samples = _index(_read_table(tables, "sample"))
    scenes = _index(_read_table(tables, "scene"))
    calibrations = _index(_read_table(tables, "calibrated_sensor"))
    sensors = _index(_read_table(tables, "sensor"))
    lidar_calibrations = {token for token, row in calibrations.items()
                          if sensors.get(row["sensor_token"], {}).get("channel") == "LIDAR_TOP"} if derive_times else set()
    if derive_times and not lidar_calibrations:
        raise ValueError("LIDAR_TOP sensor calibration is required for derived trajectory timestamps")
    data_by_filename, lidar_keyframes = {}, {}
    # Scan the large table once. Keep requested camera rows and only the compact
    # keyframe identities/timestamps needed to resolve the four action waypoints.
    for row in _read_table(tables, "sample_data"):
        filename = str(Path(row["filename"]))
        if filename in needed_paths:
            data_by_filename[filename] = row
        if (derive_times and row.get("is_key_frame") is True
                and row.get("calibrated_sensor_token") in lidar_calibrations):
            sample_token = row["sample_token"]
            if sample_token in lidar_keyframes:
                raise ValueError(f"Multiple LIDAR_TOP keyframes for sample {sample_token}")
            lidar_keyframes[sample_token] = {"token": row["token"], "timestamp": row["timestamp"]}
    official_splits = None
    if config.get("official_splits_json"):
        official_splits = json.loads(Path(config["official_splits_json"]).read_text())
        if (not isinstance(official_splits, dict)
                or any(not isinstance(official_splits.get(split), list)
                       or any(not isinstance(scene, str) or not scene for scene in official_splits[split])
                       for split in ("train", "val"))):
            raise ValueError("Official split JSON must contain train and val scene-name lists")
        if set(official_splits["train"]) & set(official_splits["val"]):
            raise ValueError("Official train and val scene names must be disjoint")
    seen = {}
    records = []
    for annotation, split, source_path, source_checksum, source_index in annotations:
        frames = [str(Path(frame)) for frame in annotation["frames"][:frame_count]]
        if len(frames) != frame_count:
            raise ValueError(f"Annotation {source_index} has fewer than {frame_count} frames")
        missing = [filename for filename in frames if filename not in data_by_filename]
        if missing:
            raise ValueError(f"Annotation {source_index} references frames absent from sample_data: {missing[:3]}")
        data = [data_by_filename[filename] for filename in frames]
        scene_ids = {samples[row["sample_token"]]["scene_token"] for row in data}
        if len(scene_ids) != 1:
            raise ValueError("Clip crosses scene boundary")
        scene_id = next(iter(scene_ids))
        scene = scenes[scene_id]
        if official_splits is not None and scene["name"] not in official_splits[split]:
            raise ValueError(f"Scene {scene['name']} does not belong to official {split} split")
        timestamps = [int(row["timestamp"]) for row in data]
        if not np.all(np.diff(timestamps) > 0):
            raise ValueError("Clip timestamps must strictly increase")
        calibration = []
        for row in data:
            cal = calibrations[row["calibrated_sensor_token"]]
            sensor = sensors[cal["sensor_token"]]
            if sensor["channel"] != "CAM_FRONT":
                raise ValueError("Only CAM_FRONT is supported")
            calibration.append({key: cal[key] for key in ("token", "translation", "rotation", "camera_intrinsic")})
        request_id = f"nuscenes-{scene_id}-{data[0]['token']}"
        trajectory = upstream_trajectory(annotation)
        signature = (split, frames, trajectory)
        if request_id in seen:
            if seen[request_id][0] != split:
                raise ValueError("Duplicated clip appears in train and validation annotations")
            if seen[request_id] != signature:
                raise ValueError("Duplicated clip has conflicting frames or trajectory annotations")
            continue  # Translated training files may contain duplicated/balanced samples.
        seen[request_id] = signature
        action_metadata = dict(metadata)
        action_metadata.update({"reference_timestamp_us": timestamps[0], "feature_source": "Vista annotation traj[2:]",
                                "annotation_sha256": source_checksum})
        if derive_times:
            action_metadata.update(_lidar_trajectory_timing(data[0], scene_id, samples, lidar_keyframes))
        # Null config placeholders never enter a manifest or the inference path.
        validate_trajectory_metadata(action_metadata)
        record = {"schema_version": 1, "request_id": request_id, "id": request_id,
                  "dataset": "nuscenes", "scene_id": scene_id, "scene_name": scene["name"], "source_split": split,
                  "observed_frames": frames[:observed_count], "future_frames": frames[observed_count:],
                  "observed_timestamps_us": timestamps[:observed_count], "future_timestamps_us": timestamps[observed_count:],
                  "camera_calibration": calibration[:observed_count],
                  "evaluation_camera_calibration": calibration[observed_count:],
                  "trajectory": trajectory, "trajectory_metadata": action_metadata,
                  "conditioning_frame_count": observed_count, "total_frame_count": frame_count,
                  "seeds": [candidate_seed(request_id, index, config.get("seed", 23)) for index in range(16)],
                  "second_seed": candidate_seed(request_id, 1000, config.get("seed", 23)),
                  "base_seed": config.get("seed", 23), "categories": _categories(scene.get("description", ""), annotation),
                  "category_provenance": "supplied labels and scene description keyword audit; not per-frame detections",
                  "official_source_split_verified": official_splits is not None,
                  "perturbation_bounds": config.get("perturbation_bounds", {"max_lateral_m": 0.5, "max_longitudinal_m": 1.0}),
                  "annotation_source": {"path": source_path, "sha256": source_checksum, "index": source_index}}
        records.append(record)
    if hash_frames:
        checksum_record_frames(records, root)
    return records
