"""Export a fixed, reproducible blinded review subset and a private answer key."""
from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import json
import os
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .metrics import as_video
from ..data.manifests import sha256_file


def contact_sheet(a: np.ndarray, b: np.ndarray, path: Path, *, frame_count: int = 6) -> None:
    a, b = as_video(a), as_video(b)
    if a.shape != b.shape:
        raise ValueError("Contact sheet requires matching videos")
    count = min(frame_count, len(a))
    indices = np.linspace(0, len(a) - 1, count, dtype=int)
    width = 240
    height = max(1, round(width * a.shape[1] / a.shape[2]))
    canvas = Image.new("RGB", (count * width + 48, 2 * (height + 26)), "#111827")
    draw = ImageDraw.Draw(canvas)
    for row, video in enumerate((a, b)):
        y = row * (height + 26)
        draw.text((12, y + 32), "A" if row == 0 else "B", fill="white")
        for column, index in enumerate(indices):
            draw.text((48 + column * width + 6, y + 6), f"Future frame {index + 1}", fill="white")
            frame = Image.fromarray(np.rint(video[index] * 255).astype(np.uint8))
            frame = frame.resize((width, height), Image.Resampling.BILINEAR)
            canvas.paste(frame, (48 + column * width, y + 26))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def motion_preview(a: np.ndarray, b: np.ndarray, path: Path, *, fps: int = 10) -> None:
    """Render every future frame with fixed A/B mapping at a declared frame rate."""
    a, b = as_video(a), as_video(b)
    if a.shape != b.shape:
        raise ValueError("Motion preview requires matching videos")
    width = 240
    height = max(1, round(width * a.shape[1] / a.shape[2]))
    frames = []
    for index, (left, right) in enumerate(zip(a, b)):
        canvas = Image.new("RGB", (2 * width, height + 28), "#111827")
        draw = ImageDraw.Draw(canvas)
        for column, (label, frame) in enumerate((("A", left), ("B", right))):
            draw.text((column * width + 8, 8), f"{label} | Future frame {index + 1}/{len(a)}", fill="white")
            image = Image.fromarray(np.rint(frame * 255).astype(np.uint8))
            canvas.paste(image.resize((width, height), Image.Resampling.BILINEAR), (column * width, 28))
        # Changing frame numbers retain all timesteps, even for frozen futures.
        frames.append(canvas.quantize(colors=256, method=Image.Quantize.FASTOCTREE))
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=round(1000 / fps),
                   loop=0, optimize=False, disposal=2)


def export_review(pairs: list[dict[str, Any]], destination: Path, *, limit: int = 12,
                  seed: int = 2026) -> dict[str, Any]:
    """Pairs contain source array paths plus identities; public files reveal neither."""
    if limit < 1:
        raise ValueError("Review limit must be positive")
    destination.mkdir(parents=True, exist_ok=True)
    # Choose by scene/candidate hash independent of performance or quality scores.
    unique = {}
    for pair in sorted(pairs, key=lambda p: p["repeat"]):
        key = (pair["scene_id"], pair["candidate_id"], pair["seed"], pair["run"])
        unique.setdefault(key, pair)
    # Every variant receives the same scene/candidate/seed subset. The limit is
    # per comparison, not a cap that could accidentally omit a whole variant.
    experimental_units = {(pair["scene_id"], pair["candidate_id"], pair["seed"]) for pair in unique.values()}
    selected = set(sorted(experimental_units, key=lambda unit: hashlib.sha256(
        f"{seed}:{unit[0]}:{unit[1]}:{unit[2]}".encode()).digest())[:limit])
    ordered = sorted((pair for pair in unique.values()
                      if (pair["scene_id"], pair["candidate_id"], pair["seed"]) in selected),
                     key=lambda pair: hashlib.sha256(
                         f"{seed}:{pair['scene_id']}:{pair['candidate_id']}:{pair['seed']}:{pair['run']}".encode()).digest())
    key_rows, public_rows, panels = [], [], []
    for pair in ordered:
        # Bind reviewer votes to this exact pair and these exact saved arrays.
        # Sequential IDs allowed old votes to approve unrelated regenerated cases.
        identity = {key: pair.get(key) for key in
                    ("request_id", "scene_id", "repeat", "candidate_id", "seed", "run", "reference_run",
                     "action_kind", "trajectory_sha256", "conditioning_frames")}
        identity.update(reference_sha256=sha256_file(pair["reference_path"]),
                        candidate_sha256=sha256_file(pair["candidate_path"]), review_seed=seed)
        case_id = "case-" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
        swap = bool(hashlib.sha256(f"{seed}:{case_id}".encode()).digest()[0] & 1)
        reference = np.load(pair["reference_path"], allow_pickle=False)[pair["conditioning_frames"]:]
        candidate = np.load(pair["candidate_path"], allow_pickle=False)[pair["conditioning_frames"]:]
        left, right = (candidate, reference) if swap else (reference, candidate)
        filename = f"{case_id}.png"
        contact_sheet(left, right, destination / filename)
        motion_filename = f"{case_id}.gif"
        motion_preview(left, right, destination / motion_filename)
        key_rows.append({"case_id": case_id, "A": pair["run"] if swap else pair["reference_run"],
                         "B": pair["reference_run"] if swap else pair["run"], **pair,
                         "reference_sha256": identity["reference_sha256"],
                         "candidate_sha256": identity["candidate_sha256"]})
        public_rows.append({"case_id": case_id, "preferred_panel_A_B_tie": "", "acceptable_A": "", "acceptable_B": "", "object_disappearance": "",
                            "object_duplication": "", "freezing": "", "geometry_change": "",
                            "action_response_notes": "", "reviewer_notes": ""})
        panels.append(f'<article><h2>{case_id}</h2><img class="motion" loading="lazy" '
                      f'alt="{case_id}: all future frames, panels A and B, 10 fps" src="{motion_filename}">'
                      f'<p><a href="{motion_filename}">Open full-motion A/B preview (10 fps)</a></p>'
                      f'<img loading="lazy" alt="{case_id}: sampled panels A and B" src="{filename}"></article>')
    fields = list(public_rows[0]) if public_rows else ["case_id", "preferred_panel_A_B_tie", "reviewer_notes"]
    review_path = destination / "review.csv"
    if review_path.exists():
        previous = review_path.read_text()
        previous_rows = list(csv.DictReader(previous.splitlines()))
        if {row.get("case_id") for row in previous_rows} != {row["case_id"] for row in public_rows}:
            # Preserve completed reviews when a regenerated export changes its
            # cases. The old votes remain inspectable but cannot pass a new gate.
            backup = destination / f"review.previous-{hashlib.sha256(previous.encode()).hexdigest()[:16]}.csv"
            if not backup.exists():
                backup.write_text(previous)
    with review_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(public_rows)
    private_path = destination / "private_key.json"
    # Create privately from the outset, including when the default umask is permissive.
    descriptor = os.open(private_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.chmod(private_path, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump({"seed": seed, "cases": key_rows}, stream, indent=2)
    (destination / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Blinded driving futures review</title>'
        '<style>body{font:16px system-ui;background:#111827;color:#eee;max-width:1500px;margin:32px auto}'
        'img{width:100%}.motion{width:720px;max-width:100%}article{margin:36px 0}a{color:#93c5fd}</style>'
        '<h1>Blinded driving futures review</h1><p>Review A and B without consulting the private key. '
        'Record disappearance, duplication, freezing, geometry changes, and qualitative action response. '
        'Animated A/B previews include every newly generated frame at 10 fps; contact sheets sample frames. '
        'GIF previews are downscaled and palette-quantized for review; original arrays remain authoritative for metrics. '
        'These observations do not establish driving safety or counterfactual ground truth.</p>'
        '<p><a href="review.csv">Download review CSV</a></p>' + "".join(panels) + '</html>')
    return {"cases": len(key_rows), "index": str(destination / "index.html"),
            "csv": str(destination / "review.csv"), "private_key": str(private_path),
            "motion_preview_fps": 10, "motion_preview_format": "GIF", "motion_preview_includes_all_future_frames": True,
            "selection": "deterministic scene/candidate hash; no metric-based selection"}
