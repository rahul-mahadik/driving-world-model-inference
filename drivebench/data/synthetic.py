"""Small deterministic image sequences for CPU plumbing tests, never claims."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .manifests import sha256_file
from .trajectories import candidate_seed


def synthetic_records(root: str | Path, *, validation_scenes: int = 96, training_scenes: int = 16,
                      clips_per_scene: int = 2, frame_count: int = 25, observed_count: int = 1,
                      height: int = 64, width: int = 112, seed: int = 23) -> list[dict[str, Any]]:
    """Generate artificial road diagrams explicitly tagged synthetic in every row."""
    if not 0 < observed_count < frame_count or clips_per_scene < 1 or min(height, width) < 8:
        raise ValueError("Invalid synthetic fixture dimensions or frame counts")
    root = Path(root)
    records = []
    for split, scene_count in [("val", validation_scenes), ("train", training_scenes)]:
        for scene_index in range(scene_count):
            scene_id = f"synthetic-{split}-{scene_index:04d}"
            for clip_index in range(clips_per_scene):
                request_id = f"{scene_id}-clip-{clip_index:02d}"
                directory = root / scene_id / f"clip-{clip_index:02d}"
                directory.mkdir(parents=True, exist_ok=True)
                paths, checksums = [], {}
                for frame in range(frame_count):
                    image = Image.new("RGB", (width, height), (30 + scene_index % 80, 70, 110))
                    draw = ImageDraw.Draw(image)
                    draw.polygon([(width // 3, height // 2), (2 * width // 3, height // 2), (width, height), (0, height)], fill=(55, 55, 60))
                    lane_x = width // 2 + int(np.sin((scene_index + frame) / 10) * 4)
                    draw.line([(width // 2, height // 2), (lane_x, height)], fill=(240, 210, 80), width=2)
                    x = (scene_index * 7 + clip_index * 3 + frame) % max(width - 10, 1)
                    draw.rectangle((x, height * 2 // 3, x + 8, height * 2 // 3 + 5), fill=(210, 80, 60))
                    path = directory / f"{frame:03d}.png"
                    image.save(path)
                    relative = str(path.relative_to(root))
                    paths.append(relative)
                    checksums[relative] = sha256_file(path)
                t0 = (scene_index + (100000 if split == "train" else 0)) * 100000000 + clip_index * 3000000
                timestamps = [t0 + frame * 100000 for frame in range(frame_count)]
                lateral = np.arange(1, 5) ** 2 * ((scene_index % 3) - 1) * 0.01
                longitudinal = np.arange(1, 5) * (2.0 + scene_index % 4)
                trajectory = np.stack([lateral, longitudinal], axis=1).astype(np.float32).reshape(8).tolist()
                records.append({"schema_version": 1, "request_id": request_id, "id": request_id,
                    "dataset": "synthetic_fixture", "synthetic": True, "scene_id": scene_id, "source_split": split,
                    "observed_frames": paths[:observed_count], "future_frames": paths[observed_count:],
                    "observed_timestamps_us": timestamps[:observed_count], "future_timestamps_us": timestamps[observed_count:],
                    "camera_calibration": [{"synthetic": True, "camera_intrinsic": [[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]],
                                            "rotation": [1, 0, 0, 0], "translation": [0, 0, 0]} for _ in range(observed_count)],
                    "trajectory": trajectory, "trajectory_metadata": {
                        "synthetic": True, "units": "meters", "coordinate_frame": "synthetic_ego_origin_first_observed",
                        "axis_order": ["lateral", "longitudinal"], "timestamps_seconds": [0.5, 1.0, 1.5, 2.0],
                        "reference_timestamp_us": t0, "provenance": "artificial CPU fixture; not Vista action validation",
                        "feature_source": "synthetic generated numeric inputs"},
                    "conditioning_frame_count": observed_count, "total_frame_count": frame_count,
                    "base_seed": seed, "seeds": [candidate_seed(request_id, index, seed) for index in range(16)],
                    "second_seed": candidate_seed(request_id, 1000, seed),
                    "categories": ["synthetic", f"motion:{'turning' if scene_index % 3 else 'straight'}", "lighting:synthetic"],
                    "frame_checksums": checksums,
                    "perturbation_bounds": {"max_lateral_m": 0.5, "max_longitudinal_m": 1.0}})
    return records
