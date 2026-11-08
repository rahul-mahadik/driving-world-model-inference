"""Deterministic manifests, disjoint scene selection and inference whitelist."""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .trajectories import alternative_trajectories, candidate_seed


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_frame(root: str | Path, path: str) -> Path:
    root = Path(root).resolve()
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Frame path escapes dataset root: {path}")
    return resolved


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())
        if sha256_file(path) != metadata["manifest_sha256"]:
            raise ValueError(f"Manifest checksum mismatch: {path}")
    with path.open() as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    ids = [record.get("request_id", record.get("id")) for record in records]
    if len(ids) != len(set(ids)) or None in ids:
        raise ValueError("Manifest request IDs must be present and unique")
    return records


def write_manifest(path: str | Path, records: list[dict[str, Any]], metadata: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for record in records))
    summary = dict(metadata or {})
    scene_ids = sorted({record["scene_id"] for record in records})
    summary.update({"schema_version": 1, "manifest_sha256": sha256_file(path), "clip_count": len(records),
                    "scene_count": len(scene_ids), "scene_ids": scene_ids,
                    "scene_ids_sha256": hashlib.sha256(json.dumps(scene_ids, separators=(",", ":")).encode()).hexdigest()})
    path.with_suffix(path.suffix + ".meta.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def second_seed_subset(evaluation: list[dict[str, Any]], count: int = 16) -> list[dict[str, Any]]:
    """Fixed evaluation prefix with identical scene/action content and new streams."""
    if count < 0 or count > len(evaluation):
        raise ValueError("Second-seed subset count must fit within evaluation")
    subset = []
    for original in evaluation[:count]:
        record = copy.deepcopy(original)
        parent_id = original["request_id"]
        record.update(request_id=f"{parent_id}-seed2", id=f"{parent_id}-seed2", parent_request_id=parent_id,
                      parent_manifest_role="evaluation", seed_variant=2)
        previous_seeds = original.get("seeds") or [candidate_seed(parent_id, i, original.get("base_seed", 23)) for i in range(16)]
        used = set(previous_seeds)
        seeds = []
        for index in range(len(previous_seeds)):
            seed = candidate_seed(parent_id, 1000 + index, original.get("base_seed", 23))
            while seed in used:
                seed = (seed + 1) & 0x7FFFFFFF
            used.add(seed)
            seeds.append(seed)
        record["seeds"] = seeds
        record["second_seed"] = seeds[0]
        subset.append(record)
    return subset


def inference_request(record: dict[str, Any], root: str | Path, candidate_count: int = 1) -> dict[str, Any]:
    """Whitelist inputs. Does not open or inspect future images, timestamps or labels."""
    request_id = record.get("request_id", record.get("id"))
    if not request_id:
        raise ValueError("Missing request ID")
    trajectories, action_kinds = alternative_trajectories(record["trajectory"], record["trajectory_metadata"], candidate_count,
        max_lateral_m=record.get("perturbation_bounds", {}).get("max_lateral_m", 0.5),
        max_longitudinal_m=record.get("perturbation_bounds", {}).get("max_longitudinal_m", 1.0))
    stored_seeds = record.get("seeds", [])
    seeds = [int(stored_seeds[i]) if i < len(stored_seeds) else candidate_seed(request_id, i, record.get("base_seed", 23))
             for i in range(candidate_count)]
    return {"request_id": request_id, "scene_id": record["scene_id"],
            "observed_frames": [str(resolve_frame(root, path)) for path in record["observed_frames"]],
            "trajectories": trajectories, "candidate_ids": [f"{request_id}:c{i:02d}" for i in range(candidate_count)],
            "seeds": seeds, "action_kinds": action_kinds}


def _scene_order(scene: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{scene}".encode()).hexdigest()


def select_manifests(records: list[dict[str, Any]], *, smoke_clips: int = 4, development_clips: int = 32,
                     evaluation_clips: int = 128, evaluation_min_scenes: int = 64,
                     calibration_scenes: int = 16, seed: int = 23) -> dict[str, list[dict[str, Any]]]:
    """Select val development/evaluation and separate train calibration by scene.

    Smoke is a subset of development, as required by the project specification.
    Evaluation uses remaining scenes in round-robin order to maximize coverage.
    """
    if min(smoke_clips, development_clips, evaluation_clips, evaluation_min_scenes, calibration_scenes) < 0:
        raise ValueError("Split sizes must be nonnegative")
    if smoke_clips > development_clips or evaluation_min_scenes > evaluation_clips:
        raise ValueError("Inconsistent split sizes")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scene_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        groups[record["scene_id"]].append(record)
        scene_splits[record["scene_id"]].add(record["source_split"])
    if any(len(splits) != 1 for splits in scene_splits.values()):
        raise ValueError("A scene appears in multiple source splits")
    for clips in groups.values():
        clips.sort(key=lambda clip: clip["request_id"])
    validation = sorted((scene for scene in groups if scene_splits[scene] == {"val"}), key=lambda x: _scene_order(x, seed))
    training = sorted((scene for scene in groups if scene_splits[scene] == {"train"}), key=lambda x: _scene_order(x, seed))
    if len(validation) < development_clips + evaluation_min_scenes:
        raise ValueError(f"Need {development_clips + evaluation_min_scenes} validation scenes; found {len(validation)}")
    if len(training) < calibration_scenes:
        raise ValueError(f"Need {calibration_scenes} separate training calibration scenes; found {len(training)}")
    dev_scenes = validation[:development_clips]
    eval_scenes = validation[development_clips:]
    development = [groups[scene][0] for scene in dev_scenes]
    evaluation = []
    depth = 0
    while len(evaluation) < evaluation_clips:
        added = 0
        for scene in eval_scenes:
            if depth < len(groups[scene]):
                evaluation.append(groups[scene][depth])
                added += 1
                if len(evaluation) == evaluation_clips:
                    break
        if added == 0:
            raise ValueError(f"Only {len(evaluation)} held-out clips available; requested {evaluation_clips}")
        depth += 1
    result = {"smoke": development[:smoke_clips], "development": development, "evaluation": evaluation,
              "calibration": [groups[scene][0] for scene in training[:calibration_scenes]]}
    audit_manifests(result, evaluation_min_scenes=evaluation_min_scenes)
    return result


def audit_manifests(manifests: dict[str, list[dict[str, Any]]], *, evaluation_min_scenes: int = 0) -> dict[str, Any]:
    sets = {name: {record["scene_id"] for record in records} for name, records in manifests.items()}
    for name, records in manifests.items():
        ids = [record["request_id"] for record in records]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate request IDs in {name}")
        expected_split = "train" if name == "calibration" else "val"
        if any(record["source_split"] != expected_split for record in records):
            raise ValueError(f"{name} must contain only {expected_split} source scenes")
    for left, right in [("development", "evaluation"), ("calibration", "development"), ("calibration", "evaluation")]:
        if sets.get(left, set()) & sets.get(right, set()):
            raise ValueError(f"Scene leakage between {left} and {right}")
    development_ids = {record["request_id"] for record in manifests.get("development", [])}
    if any(record["request_id"] not in development_ids for record in manifests.get("smoke", [])):
        raise ValueError("Smoke clips must be drawn from development")
    if len(sets.get("evaluation", set())) < evaluation_min_scenes:
        raise ValueError("Evaluation scene coverage is below the configured minimum")
    coverage = {}
    for name, records in manifests.items():
        categories: Counter[str] = Counter()
        for record in records:
            categories.update(set(record.get("categories", ["unknown"])))
            observed, future = set(record["observed_frames"]), set(record.get("future_frames", []))
            if observed & future:
                raise ValueError("Observed and evaluation-only future frames overlap")
            if record.get("conditioning_frame_count", len(observed)) != len(record["observed_frames"]):
                raise ValueError("Conditioning count differs from observed frame count")
        coverage[name] = {"clips": len(records), "scenes": len(sets[name]), "categories": dict(sorted(categories.items()))}
    return {"schema_version": 1, "scene_disjoint": True, "smoke_is_development_subset": True,
            "ground_truth_fields_excluded_by_inference_whitelist": True, "coverage": coverage}


def verify_input_checksums(record: dict[str, Any], root: str | Path, *, include_future: bool = False) -> None:
    paths = list(record["observed_frames"])
    if include_future:
        paths += record.get("future_frames", [])
    for path in paths:
        expected = record["frame_checksums"][path]
        if sha256_file(resolve_frame(root, path)) != expected:
            raise ValueError(f"Frame checksum mismatch: {path}")
