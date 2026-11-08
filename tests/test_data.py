"""Tests target leakage, scene isolation, exact crop semantics and input integrity."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from drivebench.data import (alternative_trajectories, audit_manifests, inference_request, load_manifest,
                            read_preprocess, second_seed_subset, select_manifests, verify_input_checksums, write_manifest)
from drivebench.data.nuscenes import _read_table, load_nuscenes_records
from drivebench.data.synthetic import synthetic_records
from drivebench.data.trajectories import upstream_trajectory, validate_trajectory_metadata
from drivebench.prepare import prepare


def metadata():
    return {"synthetic": True, "units": "meters", "coordinate_frame": "fixture_ego", "axis_order": ["lateral", "longitudinal"],
            "timestamps_seconds": [0.5, 1.0, 1.5, 2.0], "provenance": "synthetic fixture"}


def simple_record(scene, clip=0, split="val"):
    return {"request_id": f"{scene}-{clip}", "scene_id": scene, "source_split": split,
            "observed_frames": [f"{scene}/{clip}/observed.png"], "future_frames": [f"{scene}/{clip}/future.png"],
            "trajectory": [0, 2, 0, 4, 0, 6, 0, 8], "trajectory_metadata": metadata(), "categories": ["synthetic"]}


def test_scene_disjoint_selection_and_smoke_subset():
    records = [simple_record(f"val-{scene}", clip) for scene in range(96) for clip in range(2)]
    records += [simple_record(f"train-{scene}", split="train") for scene in range(16)]
    manifests = select_manifests(records)
    assert [len(manifests[name]) for name in ["smoke", "development", "evaluation", "calibration"]] == [4, 32, 128, 16]
    assert len({record["scene_id"] for record in manifests["evaluation"]}) == 64
    assert manifests["smoke"] == manifests["development"][:4]
    assert manifests == select_manifests(list(reversed(records)))
    audit = audit_manifests(manifests, evaluation_min_scenes=64)
    assert audit["scene_disjoint"]
    manifests["evaluation"][0] = manifests["development"][0]
    with pytest.raises(ValueError, match="leakage"):
        audit_manifests(manifests)


def test_insufficient_data_fails_instead_of_silently_shrinking():
    with pytest.raises(ValueError, match="validation scenes"):
        select_manifests([simple_record("one")])


def test_future_fields_cannot_enter_inference(tmp_path):
    record = simple_record("scene")
    first = inference_request(record, tmp_path, candidate_count=4)
    record.update(future_frames=["/not/a/real/future.png"], future_timestamps_us="malicious payload", evaluation_camera_calibration=None)
    second = inference_request(record, tmp_path, candidate_count=4)
    assert first.keys() == {"request_id", "scene_id", "observed_frames", "trajectories", "candidate_ids", "seeds", "action_kinds"}
    for key in first:
        if key == "trajectories":
            np.testing.assert_array_equal(first[key], second[key])
        else:
            assert first[key] == second[key]


def test_candidate_seed_and_action_are_independent_of_candidate_count(tmp_path):
    record = simple_record("scene")
    one = inference_request(record, tmp_path, candidate_count=1)
    many = inference_request(record, tmp_path, candidate_count=16)
    assert one["seeds"][0] == many["seeds"][0]
    assert len(set(many["seeds"])) == 16
    np.testing.assert_array_equal(one["trajectories"][0], many["trajectories"][0])


def test_alternatives_bounds_and_axes():
    trajectory = np.arange(8, dtype=np.float32)
    candidates, kinds = alternative_trajectories(trajectory, metadata(), 16)
    assert kinds[:4] == ["recorded", "deceleration_offset", "lateral_positive", "lateral_negative"]
    np.testing.assert_array_equal(candidates[0], trajectory)
    offsets = np.stack(candidates).reshape(16, 4, 2) - trajectory.reshape(4, 2)
    assert np.abs(offsets[:, :, 0]).max() <= 0.500001
    assert np.abs(offsets[:, :, 1]).max() <= 1.000001
    assert (offsets[1, :, 1] < 0).all()
    assert (offsets[2, :, 0] > 0).all()
    assert (offsets[3, :, 0] < 0).all()
    np.testing.assert_array_equal(candidates, alternative_trajectories(trajectory, metadata(), 16)[0])
    reversed_axes = metadata()
    reversed_axes["axis_order"] = ["longitudinal", "lateral"]
    reversed_candidates, _ = alternative_trajectories(trajectory, reversed_axes, 2)
    assert (reversed_candidates[1].reshape(4, 2)[:, 0] < trajectory.reshape(4, 2)[:, 0]).all()
    np.testing.assert_array_equal(reversed_candidates[1].reshape(4, 2)[:, 1], trajectory.reshape(4, 2)[:, 1])


def test_unclear_real_trajectory_semantics_fail_closed():
    data = metadata()
    data.pop("synthetic")
    with pytest.raises(ValueError, match="Confirm trajectory semantics"):
        validate_trajectory_metadata(data)
    data["axis_order"] = None
    with pytest.raises(ValueError, match="axis_order"):
        validate_trajectory_metadata(data)
    np.testing.assert_array_equal(upstream_trajectory({"traj": list(range(10))}), np.arange(2, 10))
    with pytest.raises(ValueError, match="eight"):
        upstream_trajectory({"traj": list(range(12))})


def test_vista_center_crop_rounding_and_normalization(tmp_path):
    # A wider-than-target odd-width image exercises Vista's exact integer crop.
    raw = np.arange(7 * 4 * 3, dtype=np.uint8).reshape(4, 7, 3)
    path = tmp_path / "input.png"
    Image.fromarray(raw).save(path)
    result = read_preprocess([path], 4, 4)
    expected = raw[:, 1:5].transpose(2, 0, 1).astype(np.float32) / 255 * 2 - 1
    assert result.shape == (1, 3, 4, 4) and result.dtype == np.float32 and result.flags.c_contiguous
    np.testing.assert_array_equal(result[0], expected)
    assert -1 <= result.min() <= result.max() <= 1


def test_checksums_detect_changed_manifest_and_observed_image(tmp_path):
    records = synthetic_records(tmp_path / "frames", validation_scenes=1, training_scenes=0, clips_per_scene=1, frame_count=3)
    path = tmp_path / "test.jsonl"
    write_manifest(path, records)
    assert load_manifest(path) == records
    verify_input_checksums(records[0], tmp_path / "frames")
    observed_path = tmp_path / "frames" / records[0]["observed_frames"][0]
    observed_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Frame checksum"):
        verify_input_checksums(records[0], tmp_path / "frames")
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="Manifest checksum"):
        load_manifest(path)


def test_root_escape_is_rejected(tmp_path):
    record = simple_record("scene")
    record["observed_frames"] = ["../elsewhere.png"]
    with pytest.raises(ValueError, match="escapes"):
        inference_request(record, tmp_path)


def test_prepare_small_synthetic_fixture(tmp_path):
    config = {"dataset": "synthetic", "output_dir": str(tmp_path), "num_frames": 3,
              "splits": {"smoke": 1, "development": 2, "evaluation": 4, "evaluation_min_scenes": 2, "calibration_scenes": 1}}
    audit = prepare(config)
    assert audit["synthetic"]
    assert audit["coverage"]["evaluation"]["scenes"] == 2
    manifest = load_manifest(tmp_path / "smoke.jsonl")
    request = inference_request(manifest[0], audit["data_root"], 4)
    assert read_preprocess(request["observed_frames"], 64, 64).shape == (1, 3, 64, 64)
    assert (tmp_path / "audit.json").exists()
    evaluation_meta = json.loads((tmp_path / "evaluation.jsonl.meta.json").read_text())
    second_meta = json.loads((tmp_path / "evaluation_second_seed.jsonl.meta.json").read_text())
    assert second_meta["partition_role"] == "held_out_evaluation_second_seed"
    assert second_meta["parent_manifest_sha256"] == evaluation_meta["manifest_sha256"]
    assert len(evaluation_meta["scene_ids"]) == 2 and len(evaluation_meta["scene_ids_sha256"]) == 64


def test_second_seed_subset_reproducible_with_identical_inputs(tmp_path):
    evaluation = [simple_record(f"scene-{index}") for index in range(20)]
    original_request = inference_request(evaluation[0], tmp_path, 16)
    second = second_seed_subset(evaluation)
    assert len(second) == 16 and second == second_seed_subset(evaluation)
    assert second[0]["request_id"] == evaluation[0]["request_id"] + "-seed2"
    assert second[0]["parent_request_id"] == evaluation[0]["request_id"]
    assert "parent_request_id" not in evaluation[0]
    request = inference_request(second[0], tmp_path, 16)
    np.testing.assert_array_equal(request["trajectories"], original_request["trajectories"])
    assert request["observed_frames"] == original_request["observed_frames"]
    assert not set(request["seeds"]) & set(original_request["seeds"])
    assert second[0]["future_frames"] == evaluation[0]["future_frames"]


def nuscenes_fixture(tmp_path):
    tables = tmp_path / "v1.0-trainval"
    tables.mkdir()
    paths = []
    for index in range(3):
        path = tmp_path / f"frame{index}.png"
        Image.new("RGB", (8, 8), "red").save(path)
        paths.append(path.name)
    table_data = {
        "sample_data": [{"filename": paths[i], "token": f"data{i}", "sample_token": "sample1", "timestamp": 1000000 + i * 100000,
                         "calibrated_sensor_token": "cal1"} for i in range(3)],
        "sample": [{"token": "sample1", "scene_token": "scene1"}],
        "scene": [{"token": "scene1", "name": "scene-0001", "description": "Night turning with pedestrians"}],
        "calibrated_sensor": [{"token": "cal1", "sensor_token": "sensor1", "translation": [0, 0, 0],
                               "rotation": [1, 0, 0, 0], "camera_intrinsic": np.eye(3).tolist()}],
        "sensor": [{"token": "sensor1", "channel": "CAM_FRONT"}],
    }
    for name, rows in table_data.items():
        (tables / f"{name}.json").write_text(json.dumps(rows))
    annos = tmp_path / "annos.json"
    annos.write_text(json.dumps([{"frames": paths, "traj": list(range(10))}]))
    action_metadata = metadata()
    action_metadata.pop("synthetic")
    action_metadata["verified_against_annotation_source"] = True
    config = {"data_root": str(tmp_path), "num_frames": 3, "trajectory_metadata": action_metadata,
              "annotation_sources": [{"path": str(annos), "split": "val"}]}
    return config


def test_nuscenes_metadata_join_and_camera_guard(tmp_path):
    config = nuscenes_fixture(tmp_path)
    records = load_nuscenes_records(config)
    assert records[0]["trajectory"] == list(range(2, 10))
    assert records[0]["observed_timestamps_us"] == [1000000]
    assert records[0]["future_timestamps_us"] == [1100000, 1200000]
    assert "lighting:night" in records[0]["categories"]
    assert len(records[0]["camera_calibration"]) == 1
    (tmp_path / "v1.0-trainval/sensor.json").write_text(json.dumps([{"token": "sensor1", "channel": "CAM_BACK"}]))
    with pytest.raises(ValueError, match="CAM_FRONT"):
        load_nuscenes_records(config)


def test_nuscenes_metadata_selection_does_not_read_images(tmp_path):
    config = nuscenes_fixture(tmp_path)
    for path in tmp_path.glob("frame*.png"):
        path.unlink()
    records = load_nuscenes_records(config, hash_frames=False)
    assert len(records) == 1 and "frame_checksums" not in records[0]
    with pytest.raises(FileNotFoundError):
        load_nuscenes_records(config)


def test_prepare_hashes_only_selected_real_frames(tmp_path, monkeypatch):
    import importlib
    module = importlib.import_module("drivebench.prepare")
    records = [simple_record(f"val-{i}") for i in range(8)] + [simple_record("train", split="train")]
    options = {"smoke_clips": 1, "development_clips": 1, "evaluation_clips": 1,
               "evaluation_min_scenes": 1, "calibration_scenes": 1}
    selected = {record["request_id"]: record for group in select_manifests(records, **options).values() for record in group}
    for record in selected.values():
        for relative in record["observed_frames"] + record["future_frames"]:
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"selected-image-fixture")
    def loader(config, *, hash_frames):
        assert hash_frames is False
        return records
    monkeypatch.setattr(module, "load_nuscenes_records", loader)
    result = prepare({"dataset": "nuscenes", "data_root": str(tmp_path), "output_dir": str(tmp_path / "manifests"),
                      "splits": {"smoke": 1, "development": 1, "evaluation": 1,
                                 "evaluation_min_scenes": 1, "calibration_scenes": 1}})
    assert result["coverage"]["evaluation"]["clips"] == 1
    assert all(record.get("frame_checksums") for record in selected.values())
    assert all("frame_checksums" not in record for record in records if record["request_id"] not in selected)


def test_nuscenes_rejects_conflicting_duplicate_annotations(tmp_path):
    config = nuscenes_fixture(tmp_path)
    path = Path(config["annotation_sources"][0]["path"])
    rows = json.loads(path.read_text())
    duplicate = {**rows[0], "traj": list(range(10, 20))}
    path.write_text(json.dumps(rows + [duplicate]))
    with pytest.raises(ValueError, match="conflicting"):
        load_nuscenes_records(config, hash_frames=False)


def test_nuscenes_rejects_overlapping_official_source_splits(tmp_path):
    config = nuscenes_fixture(tmp_path)
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"train": ["scene-0001"], "val": ["scene-0001"]}))
    config["official_splits_json"] = str(splits)
    with pytest.raises(ValueError, match="disjoint"):
        load_nuscenes_records(config, hash_frames=False)


def test_audit_rejects_calibration_from_validation_even_when_disjoint():
    manifests = {"development": [simple_record("dev")], "evaluation": [simple_record("eval")],
                 "calibration": [simple_record("calibration")], "smoke": []}
    with pytest.raises(ValueError, match="only train"):
        audit_manifests(manifests)


def test_nuscenes_table_stream_preserves_json_values(tmp_path):
    rows = [{"token": "one", "timestamp": 1532402927647951,
             "camera_intrinsic": [[1266.4172, 0.0, 816.267], [0.0, 1266.4172, 491.507], [0, 0, 1]],
             "nested": {"enabled": True, "missing": None}, "description": "rue piétonne"},
            {"token": "two", "translation": [1.25, -0.005, 0]}]
    (tmp_path / "table.json").write_text(" \n\t" + json.dumps(rows))
    result = list(_read_table(tmp_path, "table"))
    assert result == rows
    assert isinstance(result[0]["timestamp"], int)
    assert isinstance(result[0]["camera_intrinsic"][0][0], float)
    assert json.loads(json.dumps(result)) == rows


@pytest.mark.parametrize("contents, error", [
    ("", "empty"), (" \t\r\n", "empty"),
    ("{}", "array of objects"), ('{"item": {"token": "wrong-root"}}', "array of objects"),
    ("null", "array of objects"), ("42", "array of objects"),
    ("[1]", "row 0 must be an object"), ("[null]", "row 0 must be an object"),
    ("[[]]", "row 0 must be an object"), ('[{"token":"ok"},false]', "row 1 must be an object"),
    ("[", "Malformed"), ('[{"token":', "Malformed"), ('[{"token":"ok"}', "Malformed"),
    ('[{"token":"ok"},]', "Malformed"), ("[] {}", "Malformed"),
])
def test_nuscenes_table_stream_rejects_invalid_shape_or_json(tmp_path, contents, error):
    (tmp_path / "table.json").write_text(contents)
    with pytest.raises(ValueError, match=error):
        list(_read_table(tmp_path, "table"))


def test_nuscenes_table_stream_accepts_empty_array_and_long_whitespace(tmp_path):
    (tmp_path / "table.json").write_text(" " * (128 * 1024) + "[]\n")
    assert list(_read_table(tmp_path, "table")) == []


def test_nuscenes_table_stream_does_not_read_entire_file_before_first_row(tmp_path, monkeypatch):
    import io
    payload = b'[{"token":"first"},' + b' {"token":"unused","padding":"' + b"x" * (2 * 1024 * 1024) + b'"}]'
    class TrackedStream(io.BytesIO):
        def __init__(self, data):
            super().__init__(data)
            self.bytes_read = 0
        def read(self, size=-1):
            assert 0 <= size <= 64 * 1024, "table parser must use bounded reads"
            result = super().read(size)
            self.bytes_read += len(result)
            return result
        def readinto(self, buffer):
            assert len(buffer) <= 64 * 1024, "table parser must use bounded buffers"
            count = super().readinto(buffer)
            self.bytes_read += count
            return count
    stream = TrackedStream(payload)
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: stream)
    iterator = _read_table(tmp_path, "table")
    assert stream.bytes_read == 0
    assert next(iterator) == {"token": "first"}
    assert stream.bytes_read < len(payload) // 2
    iterator.close()
    assert stream.closed


def derived_nuscenes_fixture(tmp_path):
    config = nuscenes_fixture(tmp_path)
    config["trajectory_metadata"].update(timestamp_source="nuscenes_lidar_top_keyframes", timestamps_seconds=None)
    directory = tmp_path / "v1.0-trainval"
    lidar_timestamps = [980000, 1490000, 1985000, 2495000, 2990000]
    samples = [{"token": f"sample{index + 1}", "scene_token": "scene1", "timestamp": timestamp,
                "next": f"sample{index + 2}" if index < 4 else ""}
               for index, timestamp in enumerate(lidar_timestamps)]
    (directory / "sample.json").write_text(json.dumps(samples))
    sensors = json.loads((directory / "sensor.json").read_text())
    sensors.append({"token": "lidar-sensor", "channel": "LIDAR_TOP", "modality": "lidar"})
    (directory / "sensor.json").write_text(json.dumps(sensors))
    calibrations = json.loads((directory / "calibrated_sensor.json").read_text())
    calibrations.append({"token": "lidar-cal", "sensor_token": "lidar-sensor"})
    (directory / "calibrated_sensor.json").write_text(json.dumps(calibrations))
    rows = json.loads((directory / "sample_data.json").read_text())
    for index, row in enumerate(rows):
        row["is_key_frame"] = index == 0
    rows.extend({"token": f"lidar{index}", "sample_token": f"sample{index + 1}",
                 "timestamp": timestamp, "calibrated_sensor_token": "lidar-cal", "is_key_frame": True,
                 "filename": f"samples/LIDAR_TOP/lidar{index}.bin"}
                for index, timestamp in enumerate(lidar_timestamps))
    (directory / "sample_data.json").write_text(json.dumps(rows))
    return config


def test_derived_timestamps_are_only_deferred_in_adapter_configuration():
    value = metadata()
    value.pop("synthetic")
    value.update(verified_against_annotation_source=True, timestamp_source="nuscenes_lidar_top_keyframes",
                 timestamps_seconds=None)
    validate_trajectory_metadata(value, allow_derived_timestamps=True)
    with pytest.raises(ValueError, match="trajectory times"):
        validate_trajectory_metadata(value)
    with pytest.raises(ValueError, match="trajectory times"):
        alternative_trajectories(list(range(8)), value, 1)
    for source in (None, "unsupported_source"):
        value["timestamp_source"] = source
        with pytest.raises(ValueError, match="trajectory times"):
            validate_trajectory_metadata(value, allow_derived_timestamps=True)


def test_nuscenes_derives_exact_camera_relative_lidar_timing_in_one_scan(tmp_path, monkeypatch):
    import drivebench.data.nuscenes as adapter
    config = derived_nuscenes_fixture(tmp_path)
    calls = []
    read_table = adapter._read_table
    def recording_read(directory, name):
        calls.append(name)
        yield from read_table(directory, name)
    monkeypatch.setattr(adapter, "_read_table", recording_read)
    record = load_nuscenes_records(config, hash_frames=False)[0]
    derived = record["trajectory_metadata"]
    assert calls.count("sample_data") == 1
    assert derived["timestamps_seconds"] == [0.49, 0.985, 1.495, 1.99]
    assert derived["reference_timestamp_us"] == 1000000
    assert derived["spatial_reference_timestamp_us"] == 980000
    assert derived["spatial_reference_lidar_token"] == "lidar0"
    assert derived["spatial_reference_sample_token"] == "sample1"
    assert derived["action_sample_tokens"] == ["sample2", "sample3", "sample4", "sample5"]
    assert derived["action_lidar_tokens"] == ["lidar1", "lidar2", "lidar3", "lidar4"]
    assert derived["action_timestamps_us"] == [1490000, 1985000, 2495000, 2990000]
    assert config["trajectory_metadata"]["timestamps_seconds"] is None
    # Timing provenance does not rotate, translate, or otherwise alter features.
    assert record["trajectory"] == list(range(2, 10))
    validate_trajectory_metadata(derived)
    assert inference_request(record, tmp_path, 4)["trajectories"][0].tolist() == list(range(2, 10))


@pytest.mark.parametrize("fault, error", [
    ("not_camera_keyframe", "first CAM_FRONT frame"),
    ("missing_lidar_channel", "LIDAR_TOP sensor calibration"),
    ("missing_initial_lidar", "Missing LIDAR_TOP keyframe"),
    ("missing_future_lidar", "Missing LIDAR_TOP keyframe"),
    ("not_lidar_keyframe", "Missing LIDAR_TOP keyframe"),
    ("duplicate_lidar_keyframe", "Multiple LIDAR_TOP keyframes"),
    ("scene_crossing", "crosses a scene boundary"),
    ("missing_next", "four available future samples"),
    ("missing_sample", "four available future samples"),
    ("cycle", "cycle"),
    ("nonincreasing_lidar_time", "strictly increase"),
    ("nonpositive_action_time", "trajectory times"),
])
def test_nuscenes_derived_timestamps_reject_invalid_provenance(tmp_path, fault, error):
    config = derived_nuscenes_fixture(tmp_path)
    directory = tmp_path / "v1.0-trainval"
    if fault in {"scene_crossing", "missing_next", "missing_sample", "cycle"}:
        path = directory / "sample.json"
        rows = json.loads(path.read_text())
        if fault == "scene_crossing":
            rows[1]["scene_token"] = "different-scene"
        elif fault == "missing_next":
            rows[1]["next"] = ""
        elif fault == "missing_sample":
            rows.pop()
        else:
            rows[1]["next"] = "sample1"
    elif fault == "missing_lidar_channel":
        path = directory / "sensor.json"
        rows = json.loads(path.read_text())
        rows[1]["channel"] = "LIDAR_OTHER"
    else:
        path = directory / "sample_data.json"
        rows = json.loads(path.read_text())
        if fault == "not_camera_keyframe":
            rows[0]["is_key_frame"] = False
        elif fault == "missing_initial_lidar":
            rows = [row for row in rows if row["token"] != "lidar0"]
        elif fault == "missing_future_lidar":
            rows = [row for row in rows if row["token"] != "lidar2"]
        elif fault == "not_lidar_keyframe":
            rows[-1]["is_key_frame"] = False
        elif fault == "duplicate_lidar_keyframe":
            rows.append({**rows[-1], "token": "duplicate-lidar"})
        elif fault == "nonincreasing_lidar_time":
            rows[-1]["timestamp"] = rows[-2]["timestamp"]
        else:
            rows[4]["timestamp"] = 990000
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match=error):
        load_nuscenes_records(config, hash_frames=False)


def test_nuscenes_annotation_source_checksum_is_checked_before_parsing(tmp_path):
    from drivebench.data.manifests import sha256_file
    config = nuscenes_fixture(tmp_path)
    source = config["annotation_sources"][0]
    path = Path(source["path"])
    source["sha256"] = sha256_file(path)
    assert len(load_nuscenes_records(config, hash_frames=False)) == 1
    path.write_text("not valid JSON; changed after pinning")
    with pytest.raises(ValueError, match="Annotation source checksum mismatch"):
        load_nuscenes_records(config, hash_frames=False)
