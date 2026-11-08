#!/usr/bin/env python3
"""CPU-only export of the hash-verified strict preflight clip using pinned Vista."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--report-sha256", required=True)
    args = parser.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    require(sha256_file(args.report) == args.report_sha256, "Strict report SHA256 differs")
    report = json.loads(args.report.read_text())
    require(report["passed"] and report["mode"] == "agreement" and report["atol"] == report["rtol"] == 0,
            "Require successful strict agreement")
    require(report["checks"]["upstream_agreement"]["passed"], "Missing upstream agreement")
    candidates = report["input_provenance"]["candidate_ids"]
    require(len(candidates) == 1, "Expected exactly one strict candidate")
    sources = {}
    for backend in ("upstream", "refactored"):
        run = report["runs"][backend]
        source = Path(run["outputs"]["normal"])
        require(sha256_file(source) == run["outputs_sha256"]["normal"], f"{backend} NPZ SHA256 differs")
        sources[backend] = {"path": str(source), "sha256": run["outputs_sha256"]["normal"],
                            "size_bytes": source.stat().st_size}
    output = args.report.parent / "first-real-clip.mp4"
    contact = args.report.parent / "first-real-clip-contact-sheet.jpg"
    receipt_path = args.report.parent / "first-real-clip-export.json"
    require(not any(path.exists() for path in (output, contact, receipt_path)), "Export artifacts already exist")
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from drivebench.models.upstream import import_reference, verify_checkout

    model = report["config"]["model"]
    identity = verify_checkout(model["upstream_dir"], model["upstream_commit"])
    reference = import_reference(model["upstream_dir"])
    with np.load(sources["refactored"]["path"], allow_pickle=False) as archive:
        require(archive.files == candidates, "NPZ candidate differs from report")
        frames = archive[candidates[0]]
    require(frames.dtype == np.float32 and frames.shape == (25, 576, 1024, 3), "Unexpected clip layout")
    require(np.isfinite(frames).all() and frames.min() >= 0 and frames.max() <= 1, "Invalid frame values")
    pixels = np.rint(frames * 255).astype(np.uint8)
    del frames
    reference.save_img_seq_to_video(str(output), pixels, fps=10)
    indices, selected, count = (0, 5, 10, 15, 20, 24), {}, 0
    with imageio.get_reader(str(output)) as reader:
        metadata = reader.get_meta_data()
        require(float(metadata["fps"]) == 10, "Readback FPS differs")
        for index, frame in enumerate(reader):
            require(frame.shape == (576, 1024, 3), "Readback dimensions differ")
            if index in indices:
                selected[index] = frame.copy()
            count += 1
    require(count == 25 and set(selected) == set(indices), "Readback frame count differs")
    sheet = Image.new("RGB", (1568, 742), "#101827")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    draw.text((16, 12), "scene-0103 | Vista refactored | recorded trajectory | 10 fps display", fill="white", font=font)
    draw.text((16, 39), "1 conditioning frame (VAE reconstruction) + 24 generated frames | single development clip", fill="#cbd5e1", font=font)
    for position, index in enumerate(indices):
        left, top = 16 + (position % 3) * 520, 78 + (position // 3) * 328
        label = f"Frame {index:02d} | " + ("conditioning" if index == 0 else "generated")
        draw.text((left, top), label, fill="white", font=font)
        thumb = Image.fromarray(selected[index]).resize((512, 288), Image.Resampling.LANCZOS)
        sheet.paste(thumb, (left, top + 27))
    sheet.save(contact, quality=95)
    import torch
    require(not torch.cuda.is_initialized(), "Unexpected CUDA initialization during CPU export")
    receipt = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
               "report": str(args.report), "report_sha256": args.report_sha256, "sources": sources,
               "candidate_id": candidates[0], "seed": report["input_provenance"]["seeds"][0],
               "generation": report["generation"], "fps": 10, "frame_count": 25,
               "conditioning_frames": 1, "generated_frames": 24, "height": 576, "width": 1024,
               "frame_zero_note": "VAE-decoded observed conditioning; not raw input image",
               "display_timing_note": "10 fps visualization; not the irregular original camera timestamps",
               "pixel_conversion": "round(float32 RGB [0,1] *255) to uint8; MP4 is lossy",
               "readback_verified": True, "cuda_initialized": False,
               "upstream": identity, "export_api": "pinned Vista save_img_seq_to_video",
               "exporter_sha256": sha256_file(Path(model["upstream_dir"]) / "train.py"),
               "helper_sha256": sha256_file(__file__), "contact_sheet_frames": list(indices),
               "contact_sheet_source": "decoded MP4 frames",
               "artifacts": {path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
                             for path in (output, contact)},
               "scope": "single development clip; exact implementation agreement; quality not established"}
    with receipt_path.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"video": str(output), "contact_sheet": str(contact), "receipt": str(receipt_path),
                      "receipt_sha256": sha256_file(receipt_path), "readback_verified": True}))


if __name__ == "__main__":
    main()
