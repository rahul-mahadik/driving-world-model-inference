#!/usr/bin/env python3
"""Write a bounded development screening plan; never launches GPU work.

The plan keeps exact diagnostic gates, output-comparison runs, export-free
throughput runs, and small candidate-scaling observations separate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from drivebench.config import digest, file_sha256, load_config, merge, validate_run_config
from drivebench.data import load_manifest, write_manifest
from drivebench.types import GenerationConfig, OptimizationConfig

PROJECT = Path(__file__).resolve().parents[1]
PYTHON = "/workspace/outputs/venvs/vista-upstream-cu118-sm90/bin/python"
CACHE = 2 * 1024**3


def estimate(futures, models=1, seconds_per_future=57.5, model_load_s=30):
    return {"serial_equivalent_futures": futures,
            "serial_reference_seconds": futures * seconds_per_future + models * model_load_s,
            "assumption": "57.5 s per serial future plus 30 s per model load; no assumed batching/cache speedup; not an ETA"}


def prepare(base_path, manifest_path, output, *, runtime_project=None):
    base = load_config(base_path)
    generation, options = validate_run_config(base)
    if generation != GenerationConfig() or options != OptimizationConfig() or base["model"].get("backend") != "vista":
        raise ValueError("Use the unchanged eager primary baseline as the screen parent")
    manifest = Path(manifest_path).resolve()
    records = load_manifest(manifest)
    metadata_path = manifest.with_suffix(manifest.suffix + ".meta.json")
    metadata = json.loads(metadata_path.read_text())
    scenes = sorted({row["scene_id"] for row in records})
    if (len(records) != 32 or len(scenes) != 32 or metadata.get("partition_role") != "development"
            or metadata.get("manifest_sha256") != file_sha256(manifest) or metadata.get("scene_ids") != scenes):
        raise ValueError("Require the verified 32-scene development selection")
    output = Path(output).absolute()
    if output.is_symlink():
        raise FileExistsError("Screen destination cannot be a symlink")
    output = output.resolve()
    def runtime(path):
        path = Path(path).resolve()
        return str(Path(runtime_project) / path.relative_to(PROJECT)) if runtime_project else str(path)
    runtime(output)  # Validate mapping before creating anything.
    output.mkdir(parents=True, exist_ok=False)
    (output / "configs").mkdir()
    (output / "manifests").mkdir()
    manifests = {}
    for name, indices in (("four-scenes", [0, 8, 16, 24]), ("repeated-scene", [28])):
        path = output / "manifests" / f"{name}.jsonl"
        write_manifest(path, [records[index] for index in indices], {
            "partition_role": "development_subset", "synthetic": False,
            "parent_manifest_sha256": file_sha256(manifest), "parent_manifest_metadata_sha256": file_sha256(metadata_path),
            "parent_indices": indices, "scope": "exploratory execution screen; not held-out evaluation"})
        manifests[name] = runtime(path)
    configs, commands = {}, []
    result_root = output / "runs"

    def add(name, changes, *, manifest_name="four-scenes", requests=4, warmups=1, saves=0, workload="independent_scenes"):
        config = merge(base, {"name": name, "output": runtime(result_root / name), "numerical_validation": None,
                             "data": {"manifest": manifests[manifest_name]},
                             "benchmark": {"workload": workload, "candidate_count": 1, "requests": requests,
                                           "warmup_requests": warmups, "max_warmup_requests": warmups,
                                           "input_condition": "uncached", "save_outputs": saves, "headline": False}})
        config = merge(config, changes)
        config["screening_scope"] = "development only; small sample; no headline or quality-acceptance claim"
        validate_run_config(config)
        path = output / "configs" / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        configs[name] = {"path": runtime(path), "sha256": file_sha256(path), "config_digest": digest(config),
                         "output": config["output"]}
        return config

    def command(name, config, priority, phase, prerequisites=()):
        futures = config["benchmark"]["candidate_count"] * (config["benchmark"]["requests"] + config["benchmark"]["warmup_requests"])
        commands.append({"id": name, "priority": priority, "phase": phase, "prerequisites": list(prerequisites),
                         "argv": [PYTHON, "-m", "drivebench.benchmark", "--config", configs[name]["path"]],
                         "config": configs[name], **estimate(futures)})

    variants = {
        "eager": {},
        "batch2": {"optimizations": {"max_batch_size": 2}},
        "conditioning-b1": {"optimizations": {"conditioning_cache_bytes": CACHE}},
        "conditioning-b2": {"optimizations": {"conditioning_cache_bytes": CACHE, "max_batch_size": 2}},
        "pinned": {"optimizations": {"pinned_memory": True}},
        "pinned-async": {"optimizations": {"pinned_memory": True, "async_transfer": True}},
        "prefetch": {"optimizations": {"prefetch_depth": 2}},
        "pipeline": {"optimizations": {"pinned_memory": True, "async_transfer": True, "prefetch_depth": 2}},
    }
    gates = {}
    for name, changes in variants.items():
        gate = add(f"gate-{name}", merge(changes, {"data": {"manifest": base["data"]["manifest"]}}), requests=1, warmups=0)
        proof = runtime(output / "proofs" / f"{name}-all-strict.json")
        argv = [PYTHON, "-m", "drivebench.validate", "--config", configs[f"gate-{name}"]["path"],
                "--request-index", "28", "--candidates", "2", "--atol", "0", "--rtol", "0", "--output", proof]
        gates[name] = {"all_argv": argv + ["--mode", "all"], "report": proof,
                       "priority": "only finalists after screening", **estimate(14, models=5),
                       "scope": "all-mode agreement/order/unrelated-action/cache checks; two candidates on one development request"}
        if name == "batch2":
            for mode, futures in (("agreement", 4), ("isolation", 6)):
                cmd = list(argv)
                cmd[cmd.index("--output") + 1] = runtime(output / "proofs" / f"batch2-{mode}-strict.json")
                commands.append({"id": f"batch2-{mode}-strict", "priority": 0 if mode == "agreement" else 1,
                                 "phase": "initial_b2_gate", "prerequisites": [] if mode == "agreement" else ["batch2-agreement-strict"],
                                 "argv": cmd + ["--mode", mode], **estimate(futures, models=2 if mode == "agreement" else 1),
                                 "acceptance": "Inspect actual_batch_sizes=[2], 50 calls/request, no OOM; preserve strict failure without changing this proof"})

    # These are real tested batch sizes, not requested sizes above the cap.
    for batch in (1, 2):
        name = f"memory-b{batch}"
        config = add(name, {"optimizations": {"max_batch_size": batch}, "benchmark": {"candidate_count": 2}},
                     requests=1, warmups=0, workload="memory_scaling")
        command(name, config, 1, "memory", ["batch2-agreement-strict"] if batch == 2 else [])

    input_variants = {
        "uncached": ({}, "four-scenes", "uncached"),
        "warm-filesystem": ({}, "four-scenes", "warm_filesystem"),
        "pinned": (variants["pinned"], "four-scenes", "uncached"),
        "pinned-async": (variants["pinned-async"], "four-scenes", "uncached"),
        "prefetch": (variants["prefetch"], "four-scenes", "uncached"),
        "pipeline": (variants["pipeline"], "four-scenes", "uncached"),
        "conditioning-fresh": (variants["conditioning-b1"], "four-scenes", "uncached"),
        "repeated-uncached": ({}, "repeated-scene", "uncached"),
        "repeated-conditioning": (variants["conditioning-b1"], "repeated-scene", "reused_conditioning"),
    }
    for variant, (changes, manifest_name, condition) in input_variants.items():
        for phase, requests, saves in (("correctness", 2, 2), ("throughput", 4, 0)):
            name = f"input-{variant}-{phase}"
            config = add(name, merge(changes, {"benchmark": {"input_condition": condition}}),
                         manifest_name=manifest_name, requests=requests, saves=saves)
            prerequisite = [] if phase == "correctness" else [f"input-{variant}-correctness" if variant in {"uncached", "repeated-uncached"}
                                                               else f"input-{variant}-strict-output-check"]
            command(name, config, 2, f"input_{phase}", prerequisite)

    checks = []
    for variant in input_variants:
        reference = "repeated-uncached" if variant.startswith("repeated-") else "uncached"
        if variant == reference:
            continue
        checks.append({"id": f"input-{variant}-strict-output-check", "argv": [PYTHON, "scripts/check_screening_outputs.py",
            "--reference", configs[f"input-{reference}-correctness"]["output"],
            "--candidate", configs[f"input-{variant}-correctness"]["output"],
            "--output", runtime(output / "proofs" / f"input-{variant}-outputs-strict.json")],
            "scope": "Actual benchmark loading/prefetch path output check; not replaced by validate all"})

    for count in (4, 8, 16):
        for variant in ("eager", "batch2", "conditioning-b1", "conditioning-b2"):
            name = f"candidates-{count}-{variant}"
            config = add(name, merge(variants[variant], {"benchmark": {"candidate_count": count}}),
                         requests=2, warmups=1, saves=1, workload="candidate_set")
            priority = 3 if count == 4 else 4 if variant in {"eager", "conditioning-b2"} else 5
            command(name, config, priority, "candidate_scaling", [f"accepted-all-gate:{variant}"])

    plan = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
            "base_config_sha256": file_sha256(base_path), "helper_sha256": file_sha256(__file__),
            "source_manifest_sha256": file_sha256(manifest), "source_metadata_sha256": file_sha256(metadata_path),
            "runtime_source_manifest": base["data"]["manifest"], "generation": generation.to_dict(),
            "configs": configs, "commands": sorted(commands, key=lambda x: x["priority"]),
            "input_output_checks": checks, "full_numerical_gates": gates,
            "scope": "Bounded, optional development screening. Commands are a plan, not an automatic queue.",
            "notes": [
                "Strict B2 agreement is first; a failed zero gate is retained. Any later nonzero development calibration needs a separately frozen policy and new proof path.",
                "Candidate count4/8/16 and actual batch1/2 are independent. Backend cap2 means largest tested batch, not largest physically feasible batch.",
                "Uncached clears prepared context; checksum verification warms OS pages. No cold-filesystem claim.",
                "Fresh conditioning reuse includes prepare_context and clears cross-request cache. Repeated-conditioning uses the same observed scene after warmup.",
                "Pinned-async compares against pinned-only; prefetch compares against uncached. Pipeline is their explicitly combined experiment.",
                "Current async staging synchronizes before inference; only CPU input prefetch is expected to overlap subsequent GPU generation.",
                "Input correctness writes arrays; throughput reruns write no arrays. Prefetch+export timing cannot establish clean continuous throughput.",
                "Two scaling requests do not support p95, a production distribution, or headline speedup. Preserve complete-set latencies, stage times, peak memory and OOM/fallback details.",
                "All-mode validates engine options but not benchmark prefetch execution; require the separate benchmark output check.",
                "Timing scope for these benchmarks includes manifest resolution/file decode through host frames; export is separate.",
            ]}
    plan["cost_by_priority"] = {str(priority): {
        "commands": len([row for row in commands if row["priority"] == priority]),
        "serial_reference_hours": sum(row["serial_reference_seconds"] for row in commands if row["priority"] == priority) / 3600,
    } for priority in sorted({row["priority"] for row in commands})}
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default="configs/study_eager_development.yaml")
    parser.add_argument("--manifest", default="data/manifests/development.jsonl")
    parser.add_argument("--output", default="outputs/execution-screen-plan")
    parser.add_argument("--runtime-project", help="Map generated paths from this project into its runtime mount, e.g. /workspace")
    args = parser.parse_args(argv)
    plan = prepare(args.base_config, args.manifest, args.output, runtime_project=args.runtime_project)
    print(json.dumps({"plan": str(Path(args.output).resolve() / "plan.json"), "cost_by_priority": plan["cost_by_priority"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
