#!/usr/bin/env python3
"""Export explicit metric-selected cases as a separate blind review supplement.

Never opens an existing review key. Fresh mappings stay in a private directory;
stdout contains only public artifact paths/hashes and opaque frame highlights.
This is not the fixed review subset and cannot change an acceptance gate.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drivebench.config import file_sha256
from drivebench.evaluation.io import candidates, load_run, output_path, artifact_sha256
from drivebench.evaluation.metrics import as_video
from drivebench.evaluation.review import export_review


def native_helper():
    path = Path(__file__).with_name("run_quality_bounded.py")
    spec = importlib.util.spec_from_file_location("supplement_native_export", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def export(evaluation, requests, output, *, expected_future_frames=24):
    evaluation, output = Path(evaluation).resolve(), Path(output).resolve()
    if (not 1 <= len(requests) <= 4 or len(set(requests)) != len(requests)
            or type(expected_future_frames) is not int or expected_future_frames < 1):
        raise ValueError("Require one to four distinct requests and a positive future-frame count")
    if output.exists():
        raise FileExistsError("Supplement is immutable; choose a fresh output directory")
    metric_sha = file_sha256(evaluation)
    result = json.loads(evaluation.read_text())
    if len(result["runs"]) != 1 or result["runs"][0]["complete_pairs"] is not True:
        raise ValueError("Require one complete comparison before selecting review cases")
    entry = result["runs"][0]
    reference, candidate = load_run(result["reference"]), load_run(entry["run_path"])
    if (artifact_sha256(reference) != result["reference_artifact_sha256"]
            or artifact_sha256(candidate) != entry["artifact_sha256"]):
        raise ValueError("Completed evaluation run metadata changed")
    ref_index, new_index = candidates(reference), candidates(candidate)
    selected = [p for p in entry["pairs"] if p["request_id"] in set(requests)]
    if len(selected) != len(requests) or {p["request_id"] for p in selected} != set(requests):
        raise ValueError("Each selected request must identify exactly one evaluated pair")
    pairs, registry, errors = [], {}, {}
    for pair in selected:
        key = (pair["request_id"], pair["repeat"], pair["candidate_id"], pair["seed"])
        ref_row, ref_candidate = ref_index[key]
        row, new_candidate = new_index[key]
        count = new_candidate.get("conditioning_frames", candidate.config["generation"]["conditioning_frames"])
        if count != ref_candidate.get("conditioning_frames", reference.config["generation"]["conditioning_frames"]):
            raise ValueError("Conditioning counts differ")
        paths = [output_path(reference, ref_candidate["frames_path"]), output_path(candidate, new_candidate["frames_path"])]
        checksums = [pair["reference_output_sha256"], pair["candidate_output_sha256"]]
        for path, checksum in zip(paths, checksums):
            if file_sha256(path) != checksum:
                raise ValueError("Selected saved array differs from completed metric evidence")
            registry[str(path.resolve())] = checksum
        arrays = [np.load(p, mmap_mode="r", allow_pickle=False) for p in paths]
        if arrays[0].shape != arrays[1].shape or len(arrays[0]) - count != expected_future_frames:
            raise ValueError("Selected videos do not match the required future-frame contract")
        frame_errors = []
        for frame in range(count, len(arrays[0])):
            a, b = [as_video(v[frame:frame + 1]).astype(np.float64) for v in arrays]
            frame_errors.append(float(np.abs(a - b).max()))
        errors[pair["request_id"]] = {"greatest_error_future_frame_1based": int(np.argmax(frame_errors)) + 1,
                                      "selection_statistic": "maximum absolute RGB error per future frame",
                                      "frame_max_abs_errors": frame_errors}
        pairs.append({**{k: pair.get(k) for k in ("request_id", "scene_id", "repeat", "candidate_id", "seed",
                      "action_kind", "trajectory_sha256")}, "run": candidate.name, "reference_run": reference.name,
                      "reference_path": str(paths[0]), "candidate_path": str(paths[1]), "conditioning_frames": count})
    output.mkdir(parents=True)
    private = output / "private"
    private.mkdir(mode=0o700)
    helper = native_helper()
    # This creates a fresh private mapping; it does not inspect the old mapping.
    review = export_review(pairs, private / "preview", limit=len(pairs), seed=20261027)
    derived = copy.deepcopy(result)
    derived["review"] = review
    derived["supplemental_review"] = {"selection": "explicit metric-selected cases, not the fixed blinded subset",
        "source_evaluation": str(evaluation), "source_evaluation_sha256": metric_sha,
        "selected_request_ids": requests, "fixed_reviews_unchanged": True,
        "acceptance_decision": False, "headline_claim": False}
    derived_path, proof_path = private / "source-comparison-with-supplement-review.json", private / "export-input-receipt.json"
    helper.private_json(derived_path, derived)
    helper.private_json(proof_path, {"evaluation_sha256": file_sha256(derived_path),
        "original_evaluation": {"path": str(evaluation), "sha256": metric_sha},
        "review_artifact_sha256": {str(p): file_sha256(p) for p in (private / "preview").rglob("*") if p.is_file()}})
    bundle = helper.export_comparison(derived_path, proof_path, registry, output / "native")
    atlas_dir = output / "public-atlases"
    atlas_dir.mkdir()
    # Only the newly generated mapping is read inside this exporter. Never print it.
    cases = json.loads(Path(review["private_key"]).read_text())["cases"]
    highlights, atlas_links = [], []
    for case in cases:
        case_id = case["case_id"]
        gif = private / "preview" / f"{case_id}.gif"
        with Image.open(gif) as frames:
            if frames.n_frames != expected_future_frames:
                raise ValueError("Preview dropped an ordered future frame")
            w, h = frames.size
            atlas = Image.new("RGB", (4 * w, math_ceil_div(frames.n_frames, 4) * h), "#111827")
            for i in range(frames.n_frames):
                frames.seek(i)
                atlas.paste(frames.convert("RGB"), ((i % 4) * w, (i // 4) * h))
        atlas_path = atlas_dir / f"{case_id}-all-frames.png"
        atlas.save(atlas_path)
        frame = errors[case["request_id"]]["greatest_error_future_frame_1based"]
        native = bundle / "public" / f"{case_id}-frame-{frame:04d}.png"
        highlights.append({"case_id": case_id, "greatest_error_future_frame_1based": frame,
                           "native_frame_pair": str(native), "sha256": file_sha256(native)})
        atlas_links.append(f'<article><h2>{case_id}</h2><p>Greatest absolute-error future frame: {frame}.</p>'
                           f'<img src="{atlas_path.name}" alt="All ordered A/B future frames"></article>')
    (atlas_dir / "index.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<title>Metric-selected blind supplement</title><h1>Metric-selected blind supplement</h1>'
        '<p>Selected from completed numerical metrics; these cases do not replace the fixed blinded subset. '
        'Ordered static GIF frames are downscaled and palette-quantized, arranged left-to-right then top-to-bottom. '
        'Use the separately exported native PNG pairs for spatial defects. This is not real-time playback.</p>'
        + ''.join(atlas_links) + '</html>')
    files = {str(p): file_sha256(p) for parent in (bundle / "public", atlas_dir) for p in parent.rglob("*") if p.is_file()}
    if file_sha256(evaluation) != metric_sha or any(file_sha256(p) != checksum for p, checksum in registry.items()):
        raise ValueError("Original metrics or arrays changed during supplement export")
    helper.private_json(private / "supplement-receipt.json", {"schema_version": 1,
        "receipt_type": "metric_selected_blind_supplement_v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "selection": derived["supplemental_review"], "source_arrays_sha256": registry,
        "frame_errors_by_request": errors, "public_artifact_sha256": files,
        "native_receipt_sha256": file_sha256(bundle / "private-receipt.json"),
        "exporter_sha256": file_sha256(__file__), "native_exporter_sha256": file_sha256(helper.__file__),
        "original_private_key_read": False, "fixed_review_or_seal_modified": False,
        "quality_acceptance_claim": False, "headline_claim": False})
    return {"status": "exported_metric_selected_supplement", "native_index": str(bundle / "public/index.html"),
            "ordered_atlas_index": str(atlas_dir / "index.html"), "public_artifact_sha256": files,
            "opaque_case_highlights": highlights, "quality_acceptance_claim": False}


def math_ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--request", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-future-frames", type=int, default=24)
    args = parser.parse_args()
    print(json.dumps(export(args.evaluation, args.request, args.output,
                            expected_future_frames=args.expected_future_frames), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
