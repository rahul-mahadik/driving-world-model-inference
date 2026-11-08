#!/usr/bin/env python3
"""CPU-only paired review and limited pixel/temporal diagnostics for scene-0103."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from preflight_real_clip import (PARENT_SHA256, RECORD_SHA256, check_hash, read_config,
                                selected_paths, verify_download)
from drivebench.config import digest, file_sha256
from drivebench.data.manifest import load_manifest
from drivebench.data.preprocessing import PREPROCESS_VERSION, read_preprocess
from drivebench.evaluation.metrics import compare_videos
from drivebench.validate import _array_sha256


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/preflight_scene0103.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data/nuscenes"))
    parser.add_argument("--parent-manifest", type=Path, default=Path("data/selection-plan/development.planned.jsonl"))
    parser.add_argument("--required-file-list", type=Path, default=Path("data/selection-plan/preflight-mini-camera-files.txt"))
    parser.add_argument("--download-report", type=Path, default=Path("outputs/gpu-session-20261026/nuscenes-mini-preflight.json"))
    parser.add_argument("--directory", type=Path, default=Path("outputs/preflight-scene0103"))
    parser.add_argument("--report-sha256", required=True)
    args = parser.parse_args(argv)
    strict_path = args.directory / "strict-agreement.json"
    check_hash(strict_path, args.report_sha256, "Strict numerical report")
    strict = json.loads(strict_path.read_text())
    require(strict["passed"] and strict["atol"] == strict["rtol"] == 0, "Require successful strict comparison")
    config = read_config(args.config)
    config["data"]["root"] = str(args.data_root.resolve())
    check_hash(args.parent_manifest, PARENT_SHA256, "Parent development manifest")
    record = load_manifest(args.parent_manifest)[28]
    require(digest(record) == RECORD_SHA256, "Preselected record differs")
    checksums, _ = verify_download(config, record, args.required_file_list, args.download_report)
    effective_record = copy.deepcopy(record)
    effective_record.update(frame_checksums=checksums, preflight_only=True)
    require(digest(effective_record) == strict["input_provenance"]["record_sha256"], "Recording differs from inference manifest")
    paths = selected_paths(record)
    preprocessed = read_preprocess([args.data_root / path for path in paths], 576, 1024)
    require(_array_sha256(preprocessed[:1]) == strict["input_provenance"]["observed_sha256"],
            "Local preprocessing differs from verified inference input")
    reference = np.ascontiguousarray(((preprocessed + 1) / 2).transpose(0, 2, 3, 1))
    del preprocessed
    source = args.directory / "strict-agreement-outputs/refactored-normal.npz"
    check_hash(source, strict["runs"]["refactored"]["outputs_sha256"]["normal"], "Generated array")
    candidate_id = strict["input_provenance"]["candidate_ids"][0]
    with np.load(source, allow_pickle=False) as archive:
        require(archive.files == [candidate_id], "Unexpected generated candidates")
        generated = archive[candidate_id]
    require(generated.shape == reference.shape == (25, 576, 1024, 3), "Unexpected clip shape")
    output = args.directory / "recorded-vs-generated-contact-sheet.jpg"
    metrics_path = args.directory / "recorded-vs-generated-diagnostics.json"
    receipt_path = args.directory / "recorded-vs-generated-receipt.json"
    require(not any(path.exists() for path in (output, metrics_path, receipt_path)), "Review artifacts already exist")
    indices = (5, 10, 15, 20, 24)
    timestamps = record["observed_timestamps_us"] + record["future_timestamps_us"]
    require(len(timestamps) == 25 and all(a < b for a, b in zip(timestamps, timestamps[1:])), "Invalid recording timeline")
    metrics = compare_videos(reference[1:], generated[1:])
    diagnosis = {"schema_version": 1, "scope": "one recorded-trajectory development clip; 24 future frames only",
                 "conditioning_frame_excluded": True, "frame_count": 24,
                 "metrics_api": "drivebench.evaluation.metrics.compare_videos",
                 "metrics": metrics, "reference_timestamps_us": timestamps[1:],
                 "reference_offsets_seconds": [(value - timestamps[0]) / 1e6 for value in timestamps[1:]],
                 "limitations": ["Pixel error is not a quality score for a stochastic possible future.",
                                 "Temporal metrics are frame-index differences, not optical flow or time-normalized motion.",
                                 "Reference camera intervals are irregular; pairing uses clip index, not validated generated timestamps.",
                                 "No quality, safety, trajectory-adherence, or generalization conclusion is established."]}
    with metrics_path.open("x") as stream:
        json.dump(diagnosis, stream, indent=2, allow_nan=False)
        stream.write("\n")
    sheet = Image.new("RGB", (1056, 1860), "#111827")
    draw = ImageDraw.Draw(sheet)
    font = None
    for name in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(name).exists():
            font = ImageFont.truetype(name, 17)
            break
    font = font or ImageFont.load_default()
    draw.text((16, 14), "scene-0103 | generated vs reference recording | recorded trajectory", fill="white", font=font)
    draw.text((16, 40), "Paired by clip index. Recording times below are measured; generated timing is not established.", fill="#cbd5e1", font=font)
    draw.text((16, 66), "Identical Vista RGB / center crop / Lanczos preprocessing. Visual review only; no quality claim.", fill="#cbd5e1", font=font)
    for row, index in enumerate(indices):
        top = 104 + row * 348
        offset = (timestamps[index] - timestamps[0]) / 1e6
        draw.text((16, top), f"GENERATED | frame {index:02d}", fill="white", font=font)
        draw.text((536, top), f"REFERENCE RECORDING | frame {index:02d} | +{offset:.6f} s", fill="white", font=font)
        draw.text((536, top + 23), f"Camera timestamp: {timestamps[index]} us", fill="#cbd5e1", font=font)
        for left, video in ((16, generated), (536, reference)):
            image = Image.fromarray(np.rint(video[index] * 255).astype(np.uint8))
            sheet.paste(image.resize((504, 284), Image.Resampling.LANCZOS), (left, top + 51))
    sheet.save(output, quality=95)
    receipt = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
               "strict_report_sha256": args.report_sha256, "generated_npz_sha256": file_sha256(source),
               "parent_manifest_sha256": PARENT_SHA256, "effective_record_sha256": digest(effective_record),
               "download_report_sha256": file_sha256(args.download_report),
               "all_25_image_hashes_verified": True, "source_frame_sha256": checksums,
               "preprocess_version": PREPROCESS_VERSION,
               "observed_preprocessing_matches_inference": True,
               "preprocess_source_sha256": file_sha256(Path("drivebench/data/preprocessing.py")),
               "metric_source_sha256": file_sha256(Path("drivebench/evaluation/metrics.py")),
               "helper_sha256": file_sha256(__file__), "contact_sheet_frames": list(indices),
               "contact_sheet_source": "original float32 model output and preprocessed recording; not decoded MP4",
               "artifacts": {path.name: {"sha256": file_sha256(path), "size_bytes": path.stat().st_size}
                             for path in (output, metrics_path)}, "gpu_used": False,
               "quality_validated": False, "trajectory_adherence_validated": False}
    with receipt_path.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"contact_sheet": str(output), "diagnostics": str(metrics_path),
                      "receipt": str(receipt_path), "receipt_sha256": file_sha256(receipt_path), "metrics": metrics}))


if __name__ == "__main__":
    main()
