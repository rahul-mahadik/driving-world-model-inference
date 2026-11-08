#!/usr/bin/env python3
"""Prepare and run the predeclared scene-0103 diagnostic without changing selections.

prepare verifies all 25 selected frame hashes against the verified mini download
report and writes an isolated, explicitly non-headline development subset.
check-imports imports the pinned inference dependency graph without loading weights.
validate enforces the same-runtime hardware proof, then runs exact upstream versus
refactored agreement with one candidate. It never adjusts numerical tolerances.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drivebench.config import digest, file_sha256, load_config, validate_run_config
from drivebench.data.manifest import inference_request, load_manifest, verify_input_checksums, write_manifest
from drivebench.types import GenerationConfig, OptimizationConfig

REQUEST_ID = "nuscenes-fcbccedd61424f1b85dcbf8f897f9754-054c2431ba324912bb4799b9217f5b7b"
SCENE_ID = "fcbccedd61424f1b85dcbf8f897f9754"
CHECKPOINT_SHA256 = "1b267bb753aa5c5e4d363f6e4c864dd4e343996dedc18b5ab20af6d20be34b98"
PARENT_SHA256 = "f33ebf5dffa1ea1d27cb468112ddea6c4a5044def7595f026309d918491b68b1"
RECORD_SHA256 = "3adc76e9721f962712c6daff1c9a3ab3808a091453767816fc37aa5e51510685"
PLAN_SHA256 = "0af23b939648a7dd8945fef97aeb06ef6c83ee2f368e4de1b049cce008821b53"
LIST_SHA256 = "ebba788ff9991b254a705d3550bcb45400148f8aac7c8c5650affdb8eeb74b7d"
NUMERICS = {"mode": "agreement", "candidates": 1, "atol": 0.0, "rtol": 0.0}


def read_config(path):
    config = load_config(path)
    generation, optimizations = validate_run_config(config)
    if generation != GenerationConfig() or optimizations != OptimizationConfig():
        raise ValueError("First diagnostic requires unchanged baseline generation and optimization defaults")
    preflight = config["preflight"]
    expected = {"request_id": REQUEST_ID, "scene_id": SCENE_ID, "scene_name": "scene-0103", "parent_row_index": 28,
                "parent_manifest_sha256": PARENT_SHA256, "parent_record_sha256": RECORD_SHA256,
                "selection_plan_sha256": PLAN_SHA256, "required_file_list_sha256": LIST_SHA256,
                "archive_key": "public/v1.0/v1.0-mini.tgz", "archive_size_bytes": 4168148189,
                "archive_etag": "b62e28f60d4229ec0e76def01e84dc5f-497",
                "validation_mode": "agreement", "candidate_count": 1, "atol": 0, "rtol": 0,
                "headline_eligible": False, "quality_validated": False}
    if any(preflight.get(key) != value for key, value in expected.items()):
        raise ValueError("First diagnostic selection, integrity pins, or strict tolerances changed")
    if config["benchmark"].get("headline") is not False or config.get("numerical_validation") is not None:
        raise ValueError("Preflight must remain explicitly ineligible for headline claims")
    if config["model"].get("backend") != "vista" or config["model"].get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise ValueError("First diagnostic requires the pinned real Vista checkpoint")
    if config["data"].get("verify_checksums") is not True:
        raise ValueError("Preflight requires input checksum verification")
    if not config.get("environment_verification") or not config.get("environment_lock"):
        raise ValueError("Preflight requires upstream environment verification and lock")
    return config


def check_hash(path, expected, description):
    if file_sha256(path) != expected:
        raise ValueError(description + " checksum mismatch")


def selected_paths(record):
    observed, future = record["observed_frames"], record["future_frames"]
    paths = observed + future
    if (len(observed), len(future), len(set(paths))) != (1, 24, 25):
        raise ValueError("Preflight requires one observed and 24 distinct future frames")
    if any(not re.fullmatch(r"(?:samples|sweeps)/CAM_FRONT/[A-Za-z0-9_.-]+\.jpg", path) for path in paths):
        raise ValueError("Unexpected selected camera path")
    return paths


def verify_download(config, record, required_path, download_report_path):
    pin = config["preflight"]
    check_hash(required_path, LIST_SHA256, "Preflight required-file list")
    paths = selected_paths(record)
    if set(Path(required_path).read_text().splitlines()) != set(paths):
        raise ValueError("Required-file list differs from the selected clip")
    report = json.loads(Path(download_report_path).read_text())
    if (report.get("schema_version") != 1 or report.get("status") != "complete" or report.get("errors")
            or report.get("missing_files") or report.get("required_file_list_sha256") != LIST_SHA256
            or report.get("required_count") != 25 or report.get("verified_file_count") != 25):
        raise ValueError("Mini downloader must report all 25 frames complete and verified")
    provenance = report.get("provenance", {})
    if provenance.get("bucket") != "motional-nuscenes" or provenance.get("region") != "ap-northeast-1":
        raise ValueError("Download provenance is not the official nuScenes bucket")
    entry = report["verified_archives"]["mini"]
    archive, integrity = entry["archive"], entry["integrity"]
    if (entry.get("status") != "verified" or archive != report["archive_catalog"]["mini"]
            or archive.get("key") != pin["archive_key"] or archive.get("etag") != pin["archive_etag"]
            or archive.get("size_bytes") != pin["archive_size_bytes"]
            or integrity.get("computed_etag") != pin["archive_etag"]
            or integrity.get("get_etag", "").strip('"') != pin["archive_etag"]
            or integrity.get("compressed_size_bytes") != pin["archive_size_bytes"]
            or integrity.get("multipart_part_size_bytes") != 8 * 1024**2
            or not re.fullmatch(r"[0-9a-f]{64}", integrity.get("compressed_sha256", ""))):
        raise ValueError("Mini archive lacks the expected current multipart integrity evidence")
    if set(entry["files"]) != set(paths):
        raise ValueError("Mini verification report does not contain exactly the selected 25 frames")
    root = Path(config["data"]["root"]).resolve()
    checksums = {}
    for name in paths:
        source = root / name
        if source.is_symlink() or not source.resolve().is_relative_to(root) or not source.is_file():
            raise ValueError("Selected frame is missing or not a safe regular file: " + name)
        expected = entry["files"][name]
        actual = file_sha256(source)
        if source.stat().st_size != expected.get("size_bytes") or actual != expected.get("sha256"):
            raise ValueError("Selected frame differs from verified archive: " + name)
        checksums[name] = actual
    return checksums, report


def prepare(config_path, parent_manifest, selection_plan, required_path, download_report_path):
    config = read_config(config_path)
    check_hash(parent_manifest, PARENT_SHA256, "Planned development manifest")
    check_hash(selection_plan, PLAN_SHA256, "Metadata-only selection plan")
    plan = json.loads(Path(selection_plan).read_text())
    if plan.get("metadata_only_plan") is not True or plan.get("ready_for_inference") is not False:
        raise ValueError("Parent provenance must describe the original metadata-only selection plan")
    records = load_manifest(parent_manifest)
    original = records[28]
    if (digest(original) != RECORD_SHA256 or original.get("request_id") != REQUEST_ID
            or original.get("scene_id") != SCENE_ID or original.get("scene_name") != "scene-0103"
            or original.get("source_split") != "val" or original.get("official_source_split_verified") is not True):
        raise ValueError("Planned row 28 no longer identifies the preselected official validation scene")
    checksums, report = verify_download(config, original, required_path, download_report_path)
    record = copy.deepcopy(original)
    record.update(frame_checksums=checksums, preflight_only=True)
    verify_input_checksums(record, config["data"]["root"], include_future=True)
    request = inference_request(record, config["data"]["root"], candidate_count=1)
    if len(request["observed_frames"]) != 1 or set(request) & {"future_frames", "future_timestamps_us", "evaluation_camera_calibration"}:
        raise ValueError("Inference whitelist includes evaluation-only data")
    metadata = {"partition_role": "development_subset", "split": "preflight", "source_split": "val",
                "synthetic": False, "preflight_only": True, "headline_eligible": False,
                "quality_validated": False, "scope": "one planned development request; no full-development or evaluation coverage",
                "parent_selection": {"path": str(Path(parent_manifest).resolve()), "sha256": PARENT_SHA256,
                                     "metadata_only": True, "role": "development", "row_index": 28,
                                     "record_sha256": RECORD_SHA256, "planned_clip_count": len(records),
                                     "selection_plan_sha256": PLAN_SHA256},
                "download_report": {"path": str(Path(download_report_path).resolve()), "sha256": file_sha256(download_report_path),
                                    "archive_integrity": report["verified_archives"]["mini"]["integrity"]},
                "all_25_frame_checksums_verified": True, "predeclared_numerics": NUMERICS,
                "preflight_config_sha256": file_sha256(config_path),
                "created_at": datetime.now(timezone.utc).isoformat()}
    output = Path(config["data"]["manifest"])
    sidecar = output.with_suffix(output.suffix + ".meta.json")
    if output.exists() or sidecar.exists() or output.is_symlink() or sidecar.is_symlink():
        raise FileExistsError("Preflight manifest already exists; refusing to overwrite first diagnostic evidence")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".preflight-") as temporary:
        staged = Path(temporary) / output.name
        write_manifest(staged, [record], metadata)
        os.link(staged, output)
        os.link(staged.with_suffix(staged.suffix + ".meta.json"), sidecar)
    return {"manifest": str(output.resolve()), "manifest_sha256": file_sha256(output), "verified_frames": 25,
            "request_id": REQUEST_ID, "headline_eligible": False, "predeclared_numerics": NUMERICS}


def check_imports(config):
    from drivebench.models.upstream import import_reference, verify_checkout
    identity = verify_checkout(config["model"]["upstream_dir"], config["model"]["upstream_commit"])
    import_reference(config["model"]["upstream_dir"])
    for module in ("vwm.models.diffusion", "vwm.modules.encoders.modules", "xformers.ops"):
        importlib.import_module(module)
    return {"passed": True, "scope": "dependency imports only; no model weights loaded", **identity}


def validate(config_path):
    config = read_config(config_path)
    path = Path(config["data"]["manifest"])
    records = load_manifest(path)
    metadata = json.loads(path.with_suffix(path.suffix + ".meta.json").read_text())
    if (len(records) != 1 or records[0]["request_id"] != REQUEST_ID or records[0].get("preflight_only") is not True
            or metadata.get("partition_role") != "development_subset" or metadata.get("headline_eligible") is not False
            or metadata.get("predeclared_numerics") != NUMERICS
            or metadata.get("preflight_config_sha256") != file_sha256(config_path)):
        raise ValueError("Preflight manifest or frozen diagnostic configuration differs")
    selected_paths(records[0])
    verify_input_checksums(records[0], config["data"]["root"], include_future=True)
    from drivebench.benchmark import _validate_reference_environment
    from drivebench.benchmarks.environment import collect_environment
    environment = collect_environment()
    gate = _validate_reference_environment(config, environment)
    proof_path = Path(config["environment_verification"])
    proof = json.loads(proof_path.read_text())
    if not proof.get("dedicated_attestation") or proof.get("failures"):
        raise ValueError("Upstream hardware proof lacks successful dedicated verification")
    for filename in ("smoke-systems.nsys-rep", "smoke-compute.ncu-rep"):
        artifact = proof_path.parent / filename
        if not artifact.is_file() or not artifact.stat().st_size:
            raise ValueError("Missing nonempty upstream profiling artifact: " + filename)
    check_imports(config)
    output = Path(config["preflight"]["validation_output"])
    attempt_path = output.with_suffix(".attempt.json")
    artifact_dir = output.parent / (output.stem + "-outputs")
    if output.exists() or attempt_path.exists() or artifact_dir.exists():
        raise FileExistsError("First diagnostic output already exists; preserve it before another attempt")
    output.parent.mkdir(parents=True, exist_ok=True)
    argv = ["--config", str(Path(config_path).resolve()), "--mode", "agreement", "--candidates", "1",
            "--request-index", "0", "--atol", "0", "--rtol", "0", "--output", str(output)]
    attempt = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "argv": argv,
               "config_sha256": file_sha256(config_path), "predeclared_numerics": NUMERICS,
               "manifest_sha256": file_sha256(path), "environment_gate": gate, "headline_eligible": False}
    attempt_path.write_text(json.dumps(attempt, indent=2) + "\n")
    try:
        from drivebench.validate import main as run_validation
        code = run_validation(argv)
        attempt.update(status="completed", exit_code=code)
        return code
    except BaseException as exc:
        attempt.update(status="failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        attempt["finished_at"] = datetime.now(timezone.utc).isoformat()
        attempt_path.write_text(json.dumps(attempt, indent=2) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check-imports", "validate"))
    parser.add_argument("--config", type=Path, default=Path("configs/preflight_scene0103.yaml"))
    parser.add_argument("--parent-manifest", type=Path, default=Path("data/selection-plan/development.planned.jsonl"))
    parser.add_argument("--selection-plan", type=Path, default=Path("data/selection-plan/plan.json"))
    parser.add_argument("--required-file-list", type=Path, default=Path("data/selection-plan/preflight-mini-camera-files.txt"))
    parser.add_argument("--download-report", type=Path, default=Path("outputs/nuscenes-mini-preflight.json"))
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.config, args.parent_manifest, args.selection_plan, args.required_file_list, args.download_report)
    elif args.command == "check-imports":
        result = check_imports(read_config(args.config))
    else:
        return validate(args.config)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
