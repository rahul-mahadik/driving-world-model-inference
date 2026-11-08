"""Prepare scene-disjoint nuScenes manifests or explicitly synthetic fixtures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from .data.manifests import audit_manifests, second_seed_subset, select_manifests, sha256_file, write_manifest
from .data.nuscenes import checksum_record_frames, load_nuscenes_records
from .data.preprocessing import PREPROCESS_VERSION
from .data.synthetic import synthetic_records


def prepare(config: dict[str, Any], *, synthetic: bool = False, output: str | None = None) -> dict[str, Any]:
    config = dict(config.get("data", config))
    output_dir = Path(output or config.get("output_dir", "data/manifests"))
    sizes = config.get("splits", {})
    options = {"smoke_clips": sizes.get("smoke", 4), "development_clips": sizes.get("development", 32),
               "evaluation_clips": sizes.get("evaluation", 128), "evaluation_min_scenes": sizes.get("evaluation_min_scenes", 64),
               "calibration_scenes": sizes.get("calibration_scenes", 16), "seed": config.get("seed", 23)}
    synthetic = synthetic or config.get("dataset") == "synthetic"
    if synthetic:
        root = output_dir / "synthetic_frames"
        validation_scenes = options["development_clips"] + options["evaluation_min_scenes"]
        clips_per_scene = max(1, -(-options["evaluation_clips"] // max(1, options["evaluation_min_scenes"])))
        records = synthetic_records(root, validation_scenes=validation_scenes, training_scenes=options["calibration_scenes"],
            clips_per_scene=clips_per_scene, frame_count=config.get("num_frames", 25),
            observed_count=config.get("conditioning_frames", 1), seed=config.get("seed", 23))
    else:
        root = Path(config["data_root"])
        records = load_nuscenes_records(config, hash_frames=False)
    manifests = select_manifests(records, **options)
    if not synthetic:
        # Selection needs metadata only. Avoid reading the entire trainval image
        # corpus when the actual experiment uses a small scene-disjoint subset.
        selected = {record["request_id"]: record for group in manifests.values() for record in group}
        checksum_record_frames(list(selected.values()), root)
    manifests["evaluation_second_seed"] = second_seed_subset(manifests["evaluation"],
        count=sizes.get("evaluation_second_seed", min(16, len(manifests["evaluation"]))))
    audit = audit_manifests(manifests, evaluation_min_scenes=options["evaluation_min_scenes"])
    metadata = {"dataset": "synthetic_fixture" if synthetic else "nuscenes", "synthetic": synthetic,
                "data_root": str(root.resolve()), "preprocessing": PREPROCESS_VERSION,
                "num_frames": config.get("num_frames", 25), "conditioning_frames": config.get("conditioning_frames", 1),
                "synthetic_results_are_not_model_or_gpu_measurements": synthetic,
                "split_seed": config.get("seed", 23)}
    if not synthetic:
        official_split_path = config.get("official_splits_json")
        metadata["source_split_verification"] = "official_scene_names" if official_split_path else "annotation_source_labels_only"
        metadata["official_splits_sha256"] = sha256_file(official_split_path) if official_split_path else None
    output_dir.mkdir(parents=True, exist_ok=True)
    roles = {"smoke": "development_subset", "development": "development", "evaluation": "held_out_evaluation",
             "calibration": "training_calibration", "evaluation_second_seed": "held_out_evaluation_second_seed"}
    parent_roles = {"smoke": "development", "evaluation_second_seed": "evaluation"}
    # Parent manifests are written before derivative manifests for exact provenance.
    for name in ["development", "evaluation", "calibration", "smoke", "evaluation_second_seed"]:
        selected = manifests[name]
        extra = {"split": name, "partition_role": roles[name], "shared_audit_path": "audit.json"}
        if name in parent_roles:
            parent = parent_roles[name]
            extra.update(parent_manifest_role=parent, parent_manifest_sha256=sha256_file(output_dir / f"{parent}.jsonl"),
                         selected_parent_request_ids=[record.get("parent_request_id", record["request_id"]) for record in selected])
        write_manifest(output_dir / f"{name}.jsonl", selected, {**metadata, **extra})
    audit.update(metadata)
    audit["manifests"] = {name: {"path": str((output_dir / f"{name}.jsonl").resolve()), "partition_role": roles[name],
                                 "sha256": sha256_file(output_dir / f"{name}.jsonl"),
                                 "scene_ids": sorted({record["scene_id"] for record in selected})}
                          for name, selected in manifests.items()}
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    (output_dir / "prepare_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--synthetic", action="store_true", help="Create CPU test fixtures; no real model or GPU measurements")
    parser.add_argument("--output", help="Override manifest output directory")
    args = parser.parse_args()
    with Path(args.config).open() as stream:
        config = yaml.safe_load(stream)
    result = prepare(config, synthetic=args.synthetic, output=args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
