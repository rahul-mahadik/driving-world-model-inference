"""Explicit trajectory contracts; no inferred nuScenes coordinate conversions."""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def validate_trajectory_metadata(metadata: dict[str, Any], *, synthetic: bool = False,
                                 allow_derived_timestamps: bool = False) -> None:
    required = {"units", "coordinate_frame", "axis_order", "timestamps_seconds", "provenance"}
    missing = required - metadata.keys()
    if missing:
        raise ValueError(f"Trajectory metadata missing {sorted(missing)}")
    if metadata["units"] != "meters":
        raise ValueError("Bounded alternatives currently require trajectory units='meters'")
    if not isinstance(metadata["axis_order"], list) or sorted(metadata["axis_order"]) != ["lateral", "longitudinal"]:
        raise ValueError("axis_order must specify lateral and longitudinal in upstream feature order")
    derive_times = (allow_derived_timestamps and metadata["timestamps_seconds"] is None
                    and metadata.get("timestamp_source") == "nuscenes_lidar_top_keyframes")
    if not derive_times:
        times = np.asarray(metadata["timestamps_seconds"], dtype=float)
        if times.shape != (4,) or not np.isfinite(times).all() or not np.all(np.diff(times) > 0) or times[0] <= 0:
            raise ValueError("Four strictly increasing positive trajectory times relative to first observed frame required")
    if not metadata["coordinate_frame"] or not metadata["provenance"]:
        raise ValueError("Trajectory coordinate frame and provenance cannot be empty")
    if not synthetic and metadata.get("verified_against_annotation_source") is not True:
        raise ValueError("Confirm trajectory semantics against the translated annotation source; upstream code does not document axes/time units")


def upstream_trajectory(annotation: dict[str, Any]) -> list[float]:
    """Apply precisely Vista sample.py's traj[2:] with no coordinate transform."""
    trajectory = np.asarray(annotation["traj"][2:], dtype=np.float32)
    if trajectory.size != 8 or trajectory.ndim != 1 or not np.isfinite(trajectory).all():
        raise ValueError("Vista translated traj[2:] must contain exactly eight finite scalar features")
    return trajectory.tolist()


def candidate_seed(request_id: str, candidate_index: int, base_seed: int = 23) -> int:
    digest = hashlib.sha256(f"{base_seed}:{request_id}:{candidate_index}".encode()).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def alternative_trajectories(recorded: list[float] | np.ndarray, metadata: dict[str, Any], count: int,
                             max_lateral_m: float = 0.5, max_longitudinal_m: float = 1.0
                             ) -> tuple[list[np.ndarray], list[str]]:
    """Recorded input, then bounded quintic perturbations in documented axes.

    The continuous offset has zero initial/final first and second derivatives.
    Bounds apply to displacement from the recorded trajectory, not road legality
    or vehicle dynamics. Recorded positions are never interpreted as headings.
    """
    if count < 1 or not np.isfinite([max_lateral_m, max_longitudinal_m]).all() or min(max_lateral_m, max_longitudinal_m) < 0:
        raise ValueError("Candidate count must be positive and displacement bounds finite/nonnegative")
    validate_trajectory_metadata(metadata, synthetic=metadata.get("synthetic", False))
    original = np.asarray(recorded, dtype=np.float32)
    if original.shape != (8,) or not np.isfinite(original).all():
        raise ValueError("Expected eight finite trajectory features")
    positions = original.reshape(4, 2)
    times = np.asarray(metadata["timestamps_seconds"], dtype=np.float64)
    u = times / times[-1]
    smooth = (10 * u**3 - 15 * u**4 + 6 * u**5).astype(np.float32)
    lateral = metadata["axis_order"].index("lateral")
    longitudinal = metadata["axis_order"].index("longitudinal")
    candidates, kinds = [original.copy()], ["recorded"]
    # Low-discrepancy deterministic amplitudes include both lateral directions.
    for index in range(1, count):
        changed = positions.copy()
        if index == 1:
            lat, lon, kind = 0.0, -max_longitudinal_m, "deceleration_offset"
        elif index == 2:
            lat, lon, kind = max_lateral_m, 0.0, "lateral_positive"
        elif index == 3:
            lat, lon, kind = -max_lateral_m, 0.0, "lateral_negative"
        else:
            angle = index * np.pi * (3 - np.sqrt(5))
            lat = float(np.sin(angle)) * max_lateral_m
            lon = -0.5 * (float(np.cos(angle)) + 1) * max_longitudinal_m
            kind = "smooth_bounded_offset"
        changed[:, lateral] += smooth * lat
        changed[:, longitudinal] += smooth * lon
        candidates.append(changed.reshape(8))
        kinds.append(kind)
    return candidates, kinds
