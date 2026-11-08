"""Resumable, disk-bounded study orchestration; generation stays in benchmark.

Quality shards commit paired metrics, output hashes and review previews before
deleting explicitly disposable raw arrays. Timing blocks never export videos.
No held-out generation starts until a development selection is frozen.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import yaml

from .config import digest, file_sha256, load_config, merge, validate_run_config
from .data.manifests import load_manifest, write_manifest
from .evaluation.io import artifact_sha256, experiment_identity, load_run, write_json
from .evaluation.study_policy import policy_for_comparison, reviewed_policy_binding
from .evaluate import aggregate_pairs, evaluate_runs, tolerance_gate
from .experiments import balanced_order
from .freeze import exact_metrics, freeze_exact_tolerances


PHASES = ("development", "evaluation", "evaluation_second_seed", "development_actions", "evaluation_actions")
ROLES = {"development": "development", "evaluation": "held_out_evaluation",
         "evaluation_second_seed": "held_out_evaluation_second_seed"}


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def sync_files(paths):
    """Persist receipt dependencies before committing the receipt or deleting raw data."""
    parents = set()
    for value in paths:
        path = Path(value)
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        parents.add(path.parent)
    for parent in sorted(parents):
        descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def generation_sources():
    """Exclude evaluation/report code so its development cannot rewrite history."""
    package = Path(__file__).parent
    files = [package / name for name in ("engine.py", "config.py", "benchmark.py", "types.py")]
    for directory in ("models", "runtime", "data", "benchmarks"):
        files.extend((package / directory).rglob("*.py"))
    return {str(path.relative_to(package)): file_sha256(path) for path in sorted(files)}


def _manifest_info(path):
    path = Path(path).resolve()
    records = load_manifest(path)
    meta = json.loads(path.with_suffix(path.suffix + ".meta.json").read_text())
    scenes = sorted({row["scene_id"] for row in records})
    if (meta.get("manifest_sha256") != file_sha256(path) or meta.get("scene_ids") != scenes
            or meta.get("clip_count") != len(records) or meta.get("scene_ids_sha256") != digest(scenes)):
        raise ValueError("Manifest metadata does not match its full contents")
    return records, meta


def create_plan(spec_path, output):
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError("Study plans are immutable; resume the existing plan or choose a new directory")
    spec = yaml.safe_load(Path(spec_path).read_text())
    if spec.get("schema_version") != 1:
        raise ValueError("Unsupported study specification")
    base = load_config(spec["base_config"])
    variants = spec["variants"]
    reference, optimized = spec.get("reference", "upstream"), spec.get("optimized_reference", "conditioning-reuse")
    if (not isinstance(variants, dict) or reference not in variants or optimized not in variants
            or any(not name or Path(name).name != name or name.startswith(".") for name in variants)):
        raise ValueError("Named reference and optimized variants must be present and safe path components")
    for changes in variants.values():
        validate_run_config(merge(base, changes))
    quality = {"shard_clips": 32, "candidate_count": 1, "action_candidate_count": 2,
               "action_subset_clips": 16, "retain_raw_clips": 2, "review_pairs_per_shard": 4,
               "warmups": 1, "minimum_free_bytes": 8 * 1024**3, "lpips": True, "lpips_device": "cuda",
               "evaluation_python_path": None, "evaluation_threads": 4,
               "evaluation_torch_home": None, "evaluation_setup_receipt": None,
               **spec.get("quality", {})}
    for name in ("shard_clips", "candidate_count", "action_candidate_count", "action_subset_clips", "review_pairs_per_shard"):
        if type(quality[name]) is not int or quality[name] < 1:
            raise ValueError(f"quality.{name} must be positive")
    if quality["action_candidate_count"] < 2:
        raise ValueError("Action subsets require at least two candidates")
    if quality["retain_raw_clips"] < 0 or quality["warmups"] < 0 or quality["minimum_free_bytes"] < 0:
        raise ValueError("Retention, warmups and disk reserve must be nonnegative")
    source = {}
    for phase in ROLES:
        records, meta = _manifest_info(spec["manifests"][phase])
        if meta.get("partition_role") != ROLES[phase]:
            raise ValueError(f"Incorrect manifest role for {phase}")
        source[phase] = (records, meta)
    expected = {"development": 32, "evaluation": 128, "evaluation_second_seed": 16}
    if any(len(source[name][0]) != size for name, size in expected.items()):
        raise ValueError("Study requires32 development,128 evaluation and16 second-seed clips")
    if len(source["development"][1]["scene_ids"]) != 32 or len(source["evaluation"][1]["scene_ids"]) < 64:
        raise ValueError("Development requires32 distinct scenes; evaluation requires at least64")
    dev_scenes = set(source["development"][1]["scene_ids"])
    if any(dev_scenes & set(source[phase][1]["scene_ids"]) for phase in ("evaluation", "evaluation_second_seed")):
        raise ValueError("Development and evaluation scenes overlap")
    parents = {r["request_id"]: r for r in source["evaluation"][0]}
    for row in source["evaluation_second_seed"][0]:
        parent = parents.get(row.get("parent_request_id"))
        if not parent or any(row.get(key) != parent.get(key) for key in
                             ("scene_id", "observed_frames", "future_frames", "trajectory", "trajectory_metadata")):
            raise ValueError("Second-seed subset must preserve parent scene, frames and actions")
        if not row.get("seeds") or set(row["seeds"]) & set(parent.get("seeds", [])):
            raise ValueError("Second-seed streams must be different from their parents")
    destination.mkdir(parents=True)
    manifests = {}
    for phase, (records, meta) in source.items():
        path = destination / "manifests" / f"{phase}.jsonl"
        write_manifest(path, records, meta)
        manifests[phase] = {"path": str(path), "sha256": file_sha256(path), "metadata": _manifest_info(path)[1]}
    for phase in ("development", "evaluation"):
        records = source[phase][0]
        # Fixed selection independent of generated values, timing or quality.
        chosen = copy.deepcopy(sorted(records, key=lambda r: digest({"action_subset": 23, "request_id": r["request_id"]}))[:quality["action_subset_clips"]])
        for row in chosen:
            from .data.trajectories import candidate_seed
            seed = (row.get("seeds") or [candidate_seed(row["request_id"], 0, row.get("base_seed", 23))])[0]
            row["seeds"] = [seed] * quality["action_candidate_count"]
            row["action_response_seed_policy"] = "same seed across recorded and perturbed actions"
        name = phase + "_actions"
        path = destination / "manifests" / f"{name}.jsonl"
        write_manifest(path, chosen, {"partition_role": ROLES[phase] + "_action_subset",
                                     "parent_manifest_sha256": manifests[phase]["sha256"],
                                     "selection": "fixed request-id hash before generation",
                                     "action_response_seed_policy": "same seed across recorded and perturbed actions"})
        manifests[name] = {"path": str(path), "sha256": file_sha256(path), "metadata": _manifest_info(path)[1]}
    plan = {"schema_version": 1, "created_at": now(), "specification_sha256": file_sha256(spec_path),
            "base_config": base, "variants": variants, "reference": reference, "optimized_reference": optimized,
            "manifests": manifests, "quality": quality, "timing": {"blocks": 4, "requests_per_block": 50,
             "warmups": 3, "candidate_count": 1, **spec.get("timing", {})},
            "generation_sources_at_plan": generation_sources(),
            "raw_output_policy": {"schema_version": 1, "disposable": "unselected quality arrays only",
             "requirements_before_deletion": ["complete paired metrics", "SHA256 of each original NPY", "review previews", "durable artifact receipt"],
             "retained": "all configs/environment/raw timings/metrics/hashes; fixed selected float arrays and blinded previews",
             "limitation": "Deleted videos cannot be reevaluated with a new metric without regeneration; hashes alone cannot reconstruct them"}}
    if plan["timing"]["blocks"] < 4 or plan["timing"]["blocks"] % 2 or plan["timing"]["blocks"] * plan["timing"]["requests_per_block"] < 200:
        raise ValueError("Timing requires even balanced blocks and at least200 total requests per finalist")
    package = Path(__file__).parent
    for name, expected in plan["generation_sources_at_plan"].items():
        target = destination / "source-snapshot" / "drivebench" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(package / name, target)
        if file_sha256(target) != expected:
            raise ValueError("Generation source changed during plan snapshot")
    atomic_json(destination / "plan.json", plan)
    atomic_json(destination / "unfrozen-tolerances.json", {"calibrated": False, "frozen": False, "metrics": {}})
    return plan


def _load_plan(destination):
    destination = Path(destination).resolve()
    plan = json.loads((destination / "plan.json").read_text())
    if generation_sources() != plan["generation_sources_at_plan"]:
        raise ValueError("Generation sources changed after planning; use a new versioned study")
    for entry in plan["manifests"].values():
        if file_sha256(entry["path"]) != entry["sha256"] or _manifest_info(entry["path"])[1] != entry["metadata"]:
            raise ValueError("Frozen study manifest changed")
    return destination, plan


def _selection(destination, plan):
    path = destination / "selection.json"
    selected = json.loads(path.read_text())
    if selected["plan_sha256"] != file_sha256(destination / "plan.json"):
        raise ValueError("Study plan changed after selection")
    for key in ("development_evaluation", "quality_policy"):
        if file_sha256(selected[key]) != selected[key + "_sha256"]:
            raise ValueError("Frozen development evidence or quality policy changed")
    for reference, binding in selected.get("reviewed_quality_policies", {}).items():
        if reviewed_policy_binding(destination, plan, reference, binding["path"]) != binding:
            raise ValueError("Frozen reviewed quality policy or calibration evidence changed")
    return selected


def _variant_config(plan, name, manifest, requests, candidates, *, quality=True):
    cfg = merge(plan["base_config"], plan["variants"][name])
    cfg["data"]["manifest"] = str(manifest)
    cfg["benchmark"].update(requests=requests, candidate_count=candidates,
         workload="single_future" if candidates == 1 else "candidate_set", headline=False,
         save_outputs=requests if quality else 0,
         warmup_requests=plan["quality" if quality else "timing"]["warmups"],
         max_warmup_requests=max(3, plan["quality" if quality else "timing"]["warmups"] + 2))
    cfg["study_provenance"] = {"generation_sources": generation_sources(), "recorded_at": now(),
                               "phase": "quality" if quality else "timing"}
    validate_run_config(cfg)
    return cfg


def _validate_completed_run(path, expected, records):
    run = load_run(path)
    if (run.summary.get("failures") or run.summary.get("run_failure") or run.summary.get("status", "ok") not in {"ok", "success"}
            or len(run.measurements) != expected["benchmark"]["requests"]
            or any(row.get("status") != "ok" for row in run.measurements)):
        raise ValueError("Only fully completed benchmark runs can be adopted")
    generation, optimizations = validate_run_config(expected)
    if run.config.get("generation") != generation.to_dict() or run.config.get("optimizations") != optimizations.to_dict():
        raise ValueError("Adopted generation or optimization settings differ")
    for key in ("backend", "upstream_commit", "checkpoint_sha256", "compiler_runtime", "compiler_feature_boundary"):
        if run.config.get("model", {}).get(key) != expected.get("model", {}).get(key):
            raise ValueError("Adopted model identity differs")
    if (run.config["data"].get("manifest_sha256") != file_sha256(expected["data"]["manifest"])
            or any(run.config["benchmark"].get(key) != expected["benchmark"].get(key)
                   for key in ("candidate_count", "input_condition", "save_outputs"))):
        raise ValueError("Adopted inputs or output-retention workload differ")
    if not run.synthetic and run.config["data"].get("verify_checksums") is not True:
        raise ValueError("Real study runs must verify input checksums")
    wanted = [row["request_id"] for row in records]
    if [row["request_id"] for row in run.measurements] != [wanted[i % len(wanted)] for i in range(len(run.measurements))]:
        raise ValueError("Benchmark request order differs from its declared shard")
    return run


def _execute_run(directory, name, cfg, records, adopted=None, adoption_source=None):
    complete = directory / "completed" / f"{name}.json"
    if complete.exists():
        receipt = json.loads(complete.read_text())
        if artifact_sha256(load_run(receipt["path"])) != receipt["artifact_sha256"]:
            raise ValueError("Completed run artifacts changed; refusing resume")
        source = receipt.get("generation_source_evidence") or {}
        if source.get("path") and file_sha256(source["path"]) != source.get("sha256"):
            raise ValueError("Adopted source receipt changed")
        return receipt
    attempts = directory / "attempts" / name
    attempts.mkdir(parents=True, exist_ok=True)
    if adopted:
        path = Path(adopted).resolve()
    else:
        attempt = attempts / f"attempt-{len(list(attempts.glob('attempt-*'))):03d}"
        attempt.mkdir()
        path = attempt / name
        cfg_path = attempt / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        before_sources = generation_sources()
        with (attempt / "process.log").open("wb") as log:
            result = subprocess.run([sys.executable, "-m", "drivebench.benchmark", "--config", str(cfg_path), "--output", str(path)], stdout=log, stderr=subprocess.STDOUT)
        atomic_json(attempt / "process.json", {"exit_code": result.returncode, "finished_at": now(),
                                               "config_sha256": file_sha256(cfg_path)})
        if generation_sources() != before_sources:
            raise ValueError("Generation sources changed during subprocess execution")
        if result.returncode:
            raise RuntimeError(f"Benchmark failed; evidence preserved in {attempt}")
    run = _validate_completed_run(path, cfg, records)
    source_evidence = None
    if adopted and not run.synthetic:
        if not adoption_source:
            raise ValueError("A real adopted run requires its completed source-identity receipt")
        source_path = Path(adoption_source).resolve()
        source = json.loads(source_path.read_text())
        if (source.get("source_unchanged") is not True or source.get("exit_code") != 0
                or not source.get("finished_at") or not source.get("source_files")
                or not source.get("source_identity")):
            raise ValueError("Adopted source receipt must prove unchanged sources through successful completion")
        source_evidence = {"path": str(source_path), "sha256": file_sha256(source_path), **source}
    outputs = []
    for row in run.measurements:
        for candidate in row["candidates"]:
            if candidate.get("frames_path"):
                p = (run.path / candidate["frames_path"]).resolve()
                if not p.is_relative_to(run.path) or p.is_symlink() or not p.is_file():
                    raise ValueError("Saved output is missing or escapes its benchmark run")
                outputs.append({"path": str(p), "sha256": file_sha256(p), "bytes": p.stat().st_size,
                                "request_id": row["request_id"], "candidate_id": candidate["candidate_id"]})
    receipt = {"path": str(run.path), "artifact_sha256": artifact_sha256(run), "outputs": outputs,
               "adopted": bool(adopted), "generation_source_evidence": source_evidence or run.config.get("study_provenance"),
               "source_note": "Historical implementations may differ; original artifacts and direct output comparisons are authoritative.",
               "finished_at": now()}
    atomic_json(complete, receipt)
    return receipt


def _retire_outputs(receipt, keep_requests, retained, *, permitted):
    if not permitted:
        return
    for item in receipt["outputs"]:
        path = Path(item["path"])
        if not path.exists():
            if item["request_id"] in keep_requests:
                keep = retained / (digest({"request": item["request_id"], "candidate": item["candidate_id"]}) + ".npy")
                if not keep.is_file() or file_sha256(keep) != item["sha256"]:
                    raise ValueError("Selected retained raw output is missing or changed")
            continue  # A previously committed retirement may have been interrupted.
        if file_sha256(path) != item["sha256"]:
            raise ValueError("Raw output changed before retirement")
        if item["request_id"] in keep_requests:
            keep = retained / (digest({"request": item["request_id"], "candidate": item["candidate_id"]}) + ".npy")
            keep.parent.mkdir(parents=True, exist_ok=True)
            if keep.exists():
                if file_sha256(keep) != item["sha256"]:
                    raise ValueError("Retained output differs")
            else:
                shutil.copyfile(path, keep)
                if file_sha256(keep) != item["sha256"]:
                    raise ValueError("Retained output copy failed integrity verification")
            sync_files([keep])
        path.unlink()


def _verify_metric_receipt(metric, proof, reference, candidate, plan):
    if file_sha256(metric) != proof["evaluation_sha256"]:
        raise ValueError("Committed paired metrics changed")
    document = json.loads(Path(metric).read_text())
    if (proof["reference_artifact_sha256"] != reference["artifact_sha256"]
            or proof["candidate_artifact_sha256"] != candidate["artifact_sha256"]
            or document["reference_artifact_sha256"] != reference["artifact_sha256"]
            or document["runs"][0]["artifact_sha256"] != candidate["artifact_sha256"]
            or proof["evaluator_source_sha256"] != document["evaluator_source_sha256"]
            or proof["raw_output_policy"] != plan["raw_output_policy"]
            or file_sha256(document["tolerances_path"]) != proof["tolerances_sha256"]):
        raise ValueError("Paired metric receipt identity or policy differs")
    for path, expected in proof["review_artifact_sha256"].items():
        if file_sha256(path) != expected:
            raise ValueError("Committed blinded review artifacts changed")


def _verify_shard_reviews(shard, evidence, plan):
    for path in Path(shard).glob("metrics/*.receipt.json"):
        metric = path.with_name(path.name.replace(".receipt.json", ".json"))
        reference, name = metric.stem.split("--", 1)
        _verify_metric_receipt(metric, json.loads(path.read_text()), evidence["runs"][reference], evidence["runs"][name], plan)


def quality_runtime_identity(environment):
    """Quality can span identical replacements; never use this for timing."""
    from .benchmarks.environment import reference_identity
    identity = copy.deepcopy(reference_identity(environment))
    if isinstance(identity.get("gpu"), dict):
        for name in ("uuid", "pci_bus_id"):
            identity["gpu"].pop(name, None)
    return identity


def quality_runtime_overlay(plan, path):
    if path is None:
        return {}
    from .benchmarks.environment import reference_identity
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict) or set(value) != {"environment_verification", "environment_lock"}:
        raise ValueError("Quality runtime overlay may only replace environment verification and lock paths")
    original = json.loads(Path(plan["base_config"]["environment_verification"]).read_text())
    proof = json.loads(Path(value["environment_verification"]).read_text())
    lock = json.loads(Path(value["environment_lock"]).read_text())
    if (proof.get("passed") is not True or lock != reference_identity(proof.get("environment", {}))
            or quality_runtime_identity(original["environment"]) != quality_runtime_identity(proof["environment"])):
        raise ValueError("Replacement quality runtime must pass verification and match all configuration except GPU UUID/bus address")
    return {**value, "study_runtime_overlay": {"path": str(Path(path).resolve()), "sha256": file_sha256(path),
            "artifact_sha256": {key: file_sha256(p) for key, p in value.items()},
            "scope": "quality only; replacement GPU UUID/bus address allowed, all remaining runtime fields identical"}}


def execute_quality(destination, phase, *, adopted=None, adoption_sources=None, max_shards=None, runtime_config=None):
    destination, plan = _load_plan(destination)
    overlay = quality_runtime_overlay(plan, runtime_config)
    if phase not in PHASES:
        raise ValueError("Unknown study phase")
    selection = _selection(destination, plan) if phase != "development" else None
    names = selection["variants"] if selection else list(plan["variants"])
    refs = [plan["reference"], plan["optimized_reference"]]
    names = list(dict.fromkeys([*refs, *names]))
    records, metadata = _manifest_info(plan["manifests"][phase]["path"])
    count = plan["quality"]["action_candidate_count"] if phase.endswith("_actions") else plan["quality"]["candidate_count"]
    keep = {r["request_id"] for r in sorted(records, key=lambda r: digest({"retain": 23, "id": r["request_id"]}))[:plan["quality"]["retain_raw_clips"]]}
    directory = destination / "quality" / phase
    directory.mkdir(parents=True, exist_ok=True)
    with (destination / ".execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        size = plan["quality"]["shard_clips"]
        for shard_index, start in enumerate(range(0, len(records), size)):
            if max_shards is not None and shard_index >= max_shards:
                break
            shard = directory / "shards" / f"{shard_index:03d}"
            done = shard / "complete.json"
            if done.exists():
                evidence = json.loads(done.read_text())
                for path, expected in evidence["metric_artifacts"].items():
                    if file_sha256(path) != expected:
                        raise ValueError("Completed shard metrics changed")
                _verify_shard_reviews(shard, evidence, plan)
                for name, receipt in evidence["runs"].items():
                    if artifact_sha256(load_run(receipt["path"])) != receipt["artifact_sha256"]:
                        raise ValueError("Completed shard run changed")
                    _retire_outputs(receipt, keep, directory / "retained" / name, permitted=not receipt["adopted"])
                continue
            shard_records = records[start:start + size]
            manifest = shard / "manifest.jsonl"
            if not manifest.exists():
                write_manifest(manifest, shard_records, {"partition_role": metadata["partition_role"] + "_quality_shard",
                    "parent_manifest_sha256": plan["manifests"][phase]["sha256"], "shard_index": shard_index})
            full_manifest = plan["manifests"][phase]["path"]
            # Whole-partition first shards can adopt existing baseline runs.
            use_manifest = full_manifest if len(shard_records) == len(records) else str(manifest)
            receipts, metrics = {}, {}
            for name in names:
                needed = (len(shard_records) * count * plan["base_config"]["generation"]["total_frames"]
                          * plan["base_config"]["generation"]["height"] * plan["base_config"]["generation"]["width"] * 3 * 4)
                if shutil.disk_usage(directory).free < needed + plan["quality"]["minimum_free_bytes"]:
                    raise RuntimeError("Disk reserve reached before generation; resume after preserving space")
                cfg = _variant_config(plan, name, use_manifest, len(shard_records), count)
                cfg.update(overlay)
                cfg["study_provenance"].update(plan_sha256=file_sha256(destination / "plan.json"), phase=phase, shard=shard_index)
                receipts[name] = _execute_run(shard, name, cfg, shard_records,
                    (adopted or {}).get(name) if len(shard_records) == len(records) else None,
                    (adoption_sources or {}).get(name))
                for reference in refs:
                    if reference == name or reference not in receipts:
                        continue
                    metric = shard / "metrics" / f"{reference}--{name}.json"
                    receipt_path = metric.with_suffix(".receipt.json")
                    if receipt_path.exists():
                        proof = json.loads(receipt_path.read_text())
                        _verify_metric_receipt(metric, proof, receipts[reference], receipts[name], plan)
                    else:
                        previous = [metric, metric.with_suffix(".process.log"), metric.parent / f"{metric.stem}-review"]
                        if any(path.exists() for path in previous):
                            failed = shard / "failed-evaluations" / f"{metric.stem}-{len(list((shard / 'failed-evaluations').glob('*'))):03d}"
                            failed.mkdir(parents=True)
                            for path in previous:
                                if path.exists():
                                    path.rename(failed / path.name)
                        metric.parent.mkdir(parents=True, exist_ok=True)
                        command = [sys.executable, "-m", "drivebench.evaluate", "--runs", receipts[name]["path"],
                            "--reference", receipts[reference]["path"], "--tolerances",
                            str(policy_for_comparison(selection, reference, name) if selection else destination / "unfrozen-tolerances.json"),
                            "--lpips-device", plan["quality"]["lpips_device"], "--review-limit",
                            str(plan["quality"]["review_pairs_per_shard"]), "--output", str(metric)]
                        if phase.endswith("_actions"):
                            command.append("--action-response")
                        if plan["quality"]["lpips"]:
                            command.append("--lpips")
                        environment = os.environ.copy()
                        evaluation_path = plan["quality"]["evaluation_python_path"]
                        if evaluation_path:
                            environment["PYTHONPATH"] = str(evaluation_path) + os.pathsep + environment.get("PYTHONPATH", "")
                        if plan["quality"]["evaluation_torch_home"]:
                            environment["TORCH_HOME"] = str(plan["quality"]["evaluation_torch_home"])
                        setup = plan["quality"]["evaluation_setup_receipt"]
                        if setup:
                            environment["DRIVEBENCH_EVALUATION_SETUP_RECEIPT"] = str(setup)
                            if not Path(setup).is_file():
                                raise ValueError("Prepare the isolated perceptual evaluator before quality metrics")
                        for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                            environment[variable] = str(plan["quality"]["evaluation_threads"])
                        with metric.with_suffix(".process.log").open("wb") as log:
                            process = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
                        if process.returncode:
                            raise RuntimeError(f"Metric subprocess failed; raw outputs retained: {metric}")
                        result = json.loads(metric.read_text())
                        if len(result["runs"]) != 1 or not result["runs"][0]["complete_pairs"]:
                            raise ValueError("Incomplete paired quality outputs; raw evidence retained")
                        if plan["quality"]["lpips"] and result["perceptual"]["status"] != "available":
                            raise ValueError("Requested LPIPS unavailable; preserve raw evidence and repair evaluator")
                        sync_files([metric, *[p for p in Path(result["review"]["index"]).parent.rglob("*") if p.is_file()]])
                        atomic_json(receipt_path, {"evaluation_sha256": file_sha256(metric),
                            "reference_artifact_sha256": receipts[reference]["artifact_sha256"],
                            "candidate_artifact_sha256": receipts[name]["artifact_sha256"], "committed_at": now(),
                            "raw_output_policy": plan["raw_output_policy"],
                            "evaluator_source_sha256": result["evaluator_source_sha256"],
                            "tolerances_sha256": file_sha256(result["tolerances_path"]),
                            "evaluation_process": {"argv": command, "pythonpath_addition": evaluation_path,
                                                   "threads": plan["quality"]["evaluation_threads"], "exit_code": process.returncode},
                            "review_artifact_sha256": {str(p): file_sha256(p) for p in Path(result["review"]["index"]).parent.rglob("*") if p.is_file()}})
                    _verify_metric_receipt(metric, json.loads(receipt_path.read_text()), receipts[reference], receipts[name], plan)
                    metrics[str(metric)] = file_sha256(metric)
                    metrics[str(receipt_path)] = file_sha256(receipt_path)
                if name not in refs:
                    _retire_outputs(receipts[name], keep, directory / "retained" / name, permitted=not receipts[name]["adopted"])
            atomic_json(done, {"completed_at": now(), "request_ids": [r["request_id"] for r in shard_records],
                              "runs": receipts, "metric_artifacts": metrics, "raw_output_policy": plan["raw_output_policy"]})
            for name in refs:
                _retire_outputs(receipts[name], keep, directory / "retained" / name, permitted=not receipts[name]["adopted"])
    return {"phase": phase, "completed_shards": len(list((directory / "shards").glob("*/complete.json")))}


def _merge_runs(destination, plan, phase, names, completed):
    from .benchmarks.environment import reference_identity
    records, metadata = _manifest_info(plan["manifests"][phase]["path"])
    index = {r["request_id"]: i for i, r in enumerate(records)}
    outputs = {}
    for name in names:
        parts = [load_run(item["runs"][name]["path"]) for item in completed]
        first = parts[0]
        for run, receipt in zip(parts, completed):
            if artifact_sha256(run) != receipt["runs"][name]["artifact_sha256"]:
                raise ValueError("Source benchmark artifacts changed before aggregation")
            if (quality_runtime_identity(run.environment) != quality_runtime_identity(first.environment)
                    or any(run.config.get(k) != first.config.get(k) for k in ("generation", "optimizations", "model"))):
                raise ValueError("Quality shards use different runtime, generation, optimization or model identities")
        rows = [copy.deepcopy(row) for run in parts for row in run.measurements]
        if len(rows) != len(records) or {row["request_id"] for row in rows} != set(index):
            raise ValueError("Quality shards must cover each partition request exactly once")
        for row in rows:
            row["request_index"] = index[row["request_id"]]
            for candidate in row["candidates"]:
                candidate["original_frames_path"] = candidate.pop("frames_path", None)
                candidate["frames_path"] = None
                candidate["retention_policy"] = "see paired metric hashes and shard raw-output receipts"
        rows.sort(key=lambda row: row["request_index"])
        path = destination / "quality" / phase / "aggregate" / "runs" / name
        cfg = copy.deepcopy(first.config)
        cfg["data"].update(manifest=plan["manifests"][phase]["path"], manifest_sha256=plan["manifests"][phase]["sha256"], manifest_metadata=metadata)
        cfg["benchmark"].update(requests=len(rows), save_outputs=len(rows), headline=False)
        cfg["started_at"] = min(run.config["started_at"] for run in parts)
        cfg["study_aggregation"] = {"quality_runtime_identities": [reference_identity(run.environment) for run in parts],
                                     "timing_runtime_identity_is_not_merged": True, "source_runs": [item["runs"][name] for item in completed],
                                     "raw_output_policy": plan["raw_output_policy"]}
        summary = {"backend": first.summary["backend"], "status": "ok", "failures": 0,
                   "headline_eligible": False, "quality_only": True,
                   "latency_p50_s": float(np.median([r["latency_s"] for r in rows])),
                   "note": "Quality shard timings are exploratory; separately interleaved timing blocks are required for headlines"}
        path.mkdir(parents=True, exist_ok=False)
        for filename, obj in (("config.json", cfg), ("environment.json", first.environment), ("summary.json", summary)):
            write_json(path / filename, obj)
        (path / "measurements.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        outputs[name] = load_run(path)
    return outputs


def aggregate_quality(destination, phase):
    destination, plan = _load_plan(destination)
    selection = _selection(destination, plan) if phase != "development" else None
    names = list(dict.fromkeys([plan["reference"], plan["optimized_reference"], *(selection["variants"] if selection else plan["variants"])]))
    records, metadata = _manifest_info(plan["manifests"][phase]["path"])
    directory = destination / "quality" / phase
    shards = sorted((directory / "shards").glob("*/complete.json"))
    completed = [json.loads(path.read_text()) for path in shards]
    if not completed or [r for item in completed for r in item["request_ids"]] != [r["request_id"] for r in records]:
        raise ValueError("Cannot aggregate an incomplete quality phase")
    for shard, item in zip(shards, completed):
        _verify_shard_reviews(shard.parent, item, plan)
        for path, expected in item["metric_artifacts"].items():
            if file_sha256(path) != expected:
                raise ValueError("Committed metric evidence changed")
    receipt_path = directory / "aggregation-receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        for path, expected in receipt["artifact_sha256"].items():
            if file_sha256(path) != expected:
                raise ValueError("Aggregated quality evidence changed")
        return receipt["results"]
    aggregate_dir = directory / "aggregate"
    if aggregate_dir.exists():
        aggregate_dir.rename(directory / f"aggregate-incomplete-{len(list(directory.glob('aggregate-incomplete-*'))):03d}")
    outputs = _merge_runs(destination, plan, phase, names, completed)
    results = {}
    for reference in (plan["reference"], plan["optimized_reference"]):
        entries, evaluator, evaluator_sources, reviews, comparison_policies = [], None, None, [], {}
        for name in names:
            if name == reference or (reference == plan["optimized_reference"] and name == plan["reference"]):
                continue
            comparisons = []
            for shard in shards:
                metric = shard.parent / "metrics" / f"{reference}--{name}.json"
                document = json.loads(metric.read_text())
                if evaluator is not None and evaluator != document["perceptual"]:
                    raise ValueError("Quality evaluator provenance differs across shards")
                if evaluator_sources is not None and evaluator_sources != document["evaluator_source_sha256"]:
                    raise ValueError("Quality evaluator code changed across shards")
                evaluator = document["perceptual"]
                evaluator_sources = document["evaluator_source_sha256"]
                comparisons.append(document)
                reviews.append(document["review"])
            pairs = [pair for document in comparisons for pair in document["runs"][0]["pairs"]]
            keys = [(p["request_id"], p["repeat"], p["candidate_id"], p["seed"]) for p in pairs]
            count = plan["quality"]["action_candidate_count"] if phase.endswith("_actions") else plan["quality"]["candidate_count"]
            if len(keys) != len(set(keys)) or len(pairs) != len(records) * count:
                raise ValueError("Missing or duplicate quality pairs")
            aggregates = aggregate_pairs(pairs)
            run = outputs[name]
            policy_path = policy_for_comparison(selection, reference, name) if selection else destination / "unfrozen-tolerances.json"
            policies = yaml.safe_load(Path(policy_path).read_text())
            comparison_policies[name] = {"path": str(policy_path), "sha256": file_sha256(policy_path),
                                         "policy": policies, "reference": reference}
            entry = {"run": name, "run_path": str(run.path), "reference": reference,
                     "artifact_sha256": artifact_sha256(run), "synthetic": run.synthetic,
                     "complete_pairs": True, "matched_pairs": len(pairs), "pairs": pairs,
                     "aggregates": aggregates, "errors": [], "manifest_metadata": metadata,
                     "action_response": [row for document in comparisons for row in document["runs"][0].get("action_response", [])],
                     "run_started_at": run.config["started_at"],
                     "quality_policy": comparison_policies[name],
                     "identity": experiment_identity(run),
                     "by_action_kind": {kind: aggregate_pairs([p for p in pairs if p["action_kind"] == kind]) for kind in sorted({p["action_kind"] for p in pairs})},
                     "by_category": {kind: aggregate_pairs([p for p in pairs if kind in p["categories"]]) for kind in sorted({c for p in pairs for c in p["categories"]})},
                     "metric_artifacts": [{"path": str(shard.parent / "metrics" / f"{reference}--{name}.json"),
                        "sha256": file_sha256(shard.parent / "metrics" / f"{reference}--{name}.json")} for shard in shards]}
            entry["gate"] = tolerance_gate(aggregates, policies, complete_pairs=True, synthetic=run.synthetic or outputs[reference].synthetic,
                evaluation_manifest_sha256=plan["manifests"][phase]["sha256"], evaluation_manifest_metadata=metadata,
                run_started_at=run.config["started_at"], reviewed=False,
                evaluated_scene_ids=metadata["scene_ids"], evaluated_request_ids=[r["request_id"] for r in records], evaluated_pair_count=len(pairs))
            entries.append(entry)
        result = {"schema_version": 1, "reference": str(outputs[reference].path),
                  "reference_artifact_sha256": artifact_sha256(outputs[reference]), "runs": entries,
                  "perceptual": evaluator, "tolerances": next(iter(comparison_policies.values()))["policy"] if comparison_policies and
                      len({p["sha256"] for p in comparison_policies.values()}) == 1 else None,
                  "tolerance_policies_by_run": comparison_policies, "raw_output_policy": plan["raw_output_policy"],
                  "evaluator_source_sha256": evaluator_sources,
                  "streamed_evaluation": True, "phase": phase, "created_at": now(),
                  "reviews": reviews,
                  "notes": ["Paired metrics were computed before raw retirement; source hashes and receipts bind every pair.",
                            "Blinded review remains an explicitly agent-assisted or human diagnostic, never implied by metric completion."]}
        target = directory / "aggregate" / f"evaluation-vs-{reference}.json"
        write_json(target, result)
        results[reference] = str(target)
    atomic_json(receipt_path, {"results": results, "artifact_sha256": {
        str(path): file_sha256(path) for path in aggregate_dir.rglob("*") if path.is_file()}, "completed_at": now()})
    return results


def freeze_selection(destination, *, exact_runs, variants, numerical_proofs=None, timing_candidates=None, timing_workload=None, reviewed_policies=None):
    destination, plan = _load_plan(destination)
    path = destination / "selection.json"
    if path.exists():
        raise FileExistsError("Finalists were already frozen")
    if set(variants) - plan["variants"].keys() or not set(exact_runs) <= set(variants):
        raise ValueError("Finalists and exact runs must be existing study variants")
    evaluation = destination / "quality/development/aggregate" / f"evaluation-vs-{plan['reference']}.json"
    policy_path = destination / "quality-policy-v1.yaml"
    entries = json.loads(evaluation.read_text())["runs"]
    medians = {entry["run"]: load_run(entry["run_path"]).summary["latency_p50_s"] for entry in entries}
    reference_steps = merge(plan["base_config"], plan["variants"][plan["optimized_reference"]])["generation"]["steps"]
    steps = {name: value for name, value in medians.items() if name.startswith("steps-")
             and merge(plan["base_config"], plan["variants"][name])["generation"]["steps"] < reference_steps}
    matched = []
    for name, value in medians.items():
        if name.startswith("feature-") and steps:
            nearest = min(steps, key=lambda step: abs(math.log(value / steps[step])))
            matched.append({"feature_variant": name, "step_variant": nearest,
                            "feature_median_s": value, "step_median_s": steps[nearest],
                            "relative_latency_gap": value / steps[nearest] - 1,
                            "note": "Nearest measured genuinely reduced-step development runtime; baseline50 is excluded as a reduced-step control. Proximity is reported, not assumed"})
    proof_paths = {name: {"path": str(Path(p).resolve()), "sha256": file_sha256(p)} for name, p in (numerical_proofs or {}).items()}
    timing = copy.deepcopy(plan["timing"])
    if timing_candidates is not None:
        if type(timing_candidates) is not int or timing_candidates < 1:
            raise ValueError("Timing candidate count must be positive")
        timing["candidate_count"] = timing_candidates
    timing["workload"] = timing_workload or ("single_future" if timing["candidate_count"] == 1 else "candidate_set")
    if timing["workload"] not in {"single_future", "candidate_set", "independent_scenes"}:
        raise ValueError("Unsupported headline timing workload")
    if timing["workload"] == "single_future" and timing["candidate_count"] != 1:
        raise ValueError("Single-future timing must contain exactly one candidate")
    reviewed = {reference: reviewed_policy_binding(destination, plan, reference, path)
                for reference, path in (reviewed_policies or {}).items()}
    if policy_path.exists():
        policy = yaml.safe_load(policy_path.read_text())
        evidence = policy.get("calibration_evidence", {})
        if (evidence.get("evaluation_sha256") != file_sha256(evaluation) or evidence.get("exact_runs") != list(exact_runs)
                or policy.get("policy_type") != "exact_output_equivalence" or policy.get("metrics") != exact_metrics()
                or policy.get("calibrated") is not True or policy.get("frozen") is not True):
            raise ValueError("Existing frozen policy differs from requested development selection")
        for source in [{"path": evidence["reference"], "artifact_sha256": evidence["reference_artifact_sha256"]}, *evidence["runs"]]:
            if artifact_sha256(load_run(source["path"])) != source["artifact_sha256"]:
                raise ValueError("Previously frozen development artifacts changed")
    else:
        freeze_exact_tolerances(evaluation, exact_runs, policy_path)
    selected = {"schema_version": 1, "frozen_at": now(), "plan_sha256": file_sha256(destination / "plan.json"),
                "variants": list(dict.fromkeys([plan["reference"], plan["optimized_reference"], *variants])),
                "exact_runs": exact_runs, "development_evaluation": str(evaluation),
                "development_evaluation_sha256": file_sha256(evaluation), "quality_policy": str(policy_path),
                "quality_policy_sha256": file_sha256(policy_path), "matched_runtime_comparisons": matched,
                "numerical_proofs": proof_paths,
                "timing": timing,
                "reviewed_quality_policies": reviewed,
                "approximate_claim_policy": "Keep all out-of-zero-tolerance outcomes visible; no quality-preserving claim without a separately frozen reviewed policy"}
    atomic_json(path, selected)
    return selected


def timing_schedule(names, blocks=4, requests_per_block=50):
    if blocks < 4 or blocks % 2 or blocks * requests_per_block < 200:
        raise ValueError("Require at least four balanced blocks and200 measured requests")
    counts = {name: 0 for name in names}
    result = []
    for index, name in enumerate(balanced_order(names, blocks)):
        result.append({"index": index, "variant": name, "variant_block": counts[name],
                       "requests": requests_per_block})
        counts[name] += 1
    return result


def aggregate_timing_blocks(paths, output, *, expected_blocks=4, expected_requests_per_block=50):
    """Treat interleaved subprocess blocks as one explicitly blocked200-request run."""
    from .benchmarks.environment import reference_identity
    from .benchmarks.statistics import summarize
    runs = [load_run(path) for path in paths]
    if len(runs) != expected_blocks or expected_blocks < 4 or expected_blocks % 2:
        raise ValueError("All planned independent timing blocks are required")
    if len({str(run.path) for run in runs}) != len(runs) or len({file_sha256(run.path / "measurements.jsonl") for run in runs}) != len(runs):
        raise ValueError("Timing blocks must be independent executions, not duplicate paths or copied measurements")
    first = runs[0]
    identity = experiment_identity(first)
    rows, block_metrics, eligible_reasons = [], [], []
    expected_keys = None
    for block, run in enumerate(runs):
        if (experiment_identity(run) != identity
                or reference_identity(run.environment) != reference_identity(first.environment)
                or run.config["generation"] != first.config["generation"]
                or run.config["optimizations"] != first.config["optimizations"]
                or any(run.config.get("model", {}).get(key) != first.config.get("model", {}).get(key)
                       for key in ("compiler_runtime", "compiler_feature_boundary"))):
            raise ValueError("Timing blocks must use identical workload, generation, optimizations and hardware/runtime")
        if (len(run.measurements) != expected_requests_per_block
                or run.config["benchmark"]["requests"] != expected_requests_per_block
                or run.config["benchmark"].get("save_outputs") != 0
                or any(row.get("status") != "ok" for row in run.measurements)
                or run.summary.get("failures") or run.summary.get("run_failure")):
            raise ValueError("Timing block is incomplete, failed, or exported videos")
        keys = [(row["request_id"], row["scene_id"], row.get("repeat", 0),
                 tuple((c["candidate_id"], c["seed"], c.get("trajectory_sha256")) for c in row["candidates"])) for row in run.measurements]
        if expected_keys is not None and keys != expected_keys:
            raise ValueError("Timing block request/action/seed order differs")
        expected_keys = keys
        if run.synthetic:
            eligible_reasons.append("Synthetic execution")
        if run.config.get("profiled") or run.summary.get("profiled"):
            eligible_reasons.append("Profiled block")
        if run.summary.get("recompiled_during_measurement"):
            eligible_reasons.append("Measured recompilation")
        if (run.summary.get("oom_attempts") or any(row.get("oom_attempts") or row.get("cache_stats", {}).get("oom_attempts") for row in run.measurements)):
            eligible_reasons.append("OOM retry")
        if run.summary.get("numerical_validation", {}).get("passed") is not True:
            eligible_reasons.append("Missing passed scoped numerical validation")
        gate = run.path / "environment_gate.json"
        if not gate.exists() or json.loads(gate.read_text()).get("passed") is not True:
            eligible_reasons.append("Missing passed environment gate")
        repetitions = max(row.get("repeat", 0) for row in run.measurements) + 1
        for original in run.measurements:
            row = copy.deepcopy(original)
            row.update(request_index=len(rows), repeat=block * repetitions + row.get("repeat", 0),
                       timing_block=block, original_request_index=original.get("request_index"))
            rows.append(row)
        block_metrics.append({"block": block, "path": str(run.path), "artifact_sha256": artifact_sha256(run),
                              "median_s": float(np.median([r["latency_s"] for r in run.measurements])),
                              "started_at": run.config.get("started_at")})
    if len(rows) < 200:
        raise ValueError("Headline timing aggregation requires at least200 measured requests")
    wall = sum(run.summary.get("throughput_measurement_wall_s", sum(row["latency_s"] for row in run.measurements)) for run in runs)
    summary = summarize(rows, wall, first.summary["backend"], True, False)
    summary["headline_eligible"] = summary["headline_eligible"] and not eligible_reasons
    summary.update(eligibility_reasons=sorted(set(eligible_reasons)), timing_blocks=block_metrics,
                   blocked_interleaved_measurement=True, unique_requests=len({row["request_id"] for row in rows}),
                   unique_scenes=len({row["scene_id"] for row in rows}), block_median_range_s=[min(x["median_s"] for x in block_metrics), max(x["median_s"] for x in block_metrics)],
                   cold_start_and_warmups="Recorded separately for every subprocess block; excluded from measured requests",
                   performance_regression_policy_calibrated=False,
                   note="200 total requests across four independently warmed interleaved blocks; not four200-request calibration runs")
    cfg = copy.deepcopy(first.config)
    cfg["benchmark"].update(requests=len(rows), headline=True, save_outputs=0)
    cfg["study_timing_blocks"] = block_metrics
    cfg["started_at"] = min(run.config["started_at"] for run in runs)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    for name, obj in (("config.json", cfg), ("environment.json", first.environment), ("summary.json", summary)):
        write_json(destination / name, obj)
    (destination / "measurements.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return load_run(destination)


def compare_timing_blocks(reference, candidate):
    """Paired scene-cluster uncertainty plus observed block drift; no tuned threshold."""
    from .regression import _bootstrap_ratios
    before, after = load_run(reference), load_run(candidate)
    def keyed(run):
        return {(row["request_id"], row["repeat"]): row for row in run.measurements}
    left, right = keyed(before), keyed(after)
    if left.keys() != right.keys() or len(left) != len(before.measurements) or len(right) != len(after.measurements):
        raise ValueError("Interleaved timing comparisons require complete unique paired requests")
    keys = sorted(left)
    for key in keys:
        a, b = left[key], right[key]
        if (a["scene_id"] != b["scene_id"] or a["candidates"] != b["candidates"]):
            raise ValueError("Timing comparison action or seed identities differ")
    result = {"reference": str(before.path), "candidate": str(after.path), "paired_requests": len(keys),
              "independent_scenes": len({left[key]["scene_id"] for key in keys}),
              "headline_eligible": bool(before.summary["headline_eligible"] and after.summary["headline_eligible"]),
              "reference_artifact_sha256": artifact_sha256(before), "candidate_artifact_sha256": artifact_sha256(after),
              "quality_preserving": False, "quality_note": "Join the separately frozen held-out quality result before making a combined claim",
              "performance_regression_policy_calibrated": False,
              "uncertainty": "Paired percentile bootstrap with scenes as clusters; all repeated blocks stay within their scene", "metrics": {}}
    x = np.asarray([left[key]["latency_s"] for key in keys])
    y = np.asarray([right[key]["latency_s"] for key in keys])
    scenes = [left[key]["scene_id"] for key in keys]
    for name, q in (("p50", .5), ("p95", .95)):
        point, ci = _bootstrap_ratios(x, y, scene_groups=scenes, quantile=q)
        result["metrics"][name] = {"relative_latency_change": point, "ci95": ci,
                                   "speedup": 1 / (1 + point), "speedup_ci95": [1 / (1 + ci[1]), 1 / (1 + ci[0])]}
    result["block_medians"] = {"reference": before.summary["timing_blocks"], "candidate": after.summary["timing_blocks"]}
    return result


def execute_timing(destination, names):
    destination, plan = _load_plan(destination)
    selected = _selection(destination, plan)
    if not set(names) <= set(selected["variants"]):
        raise ValueError("Only development-frozen finalists can enter timing")
    timing = selected["timing"]
    schedule = timing_schedule(names, timing["blocks"], timing["requests_per_block"])
    directory = destination / "timing"
    directory.mkdir(exist_ok=True)
    schedule_path = directory / "schedule.json"
    if schedule_path.exists():
        if json.loads(schedule_path.read_text()) != schedule:
            raise ValueError("Cannot change a started interleaved schedule")
    else:
        atomic_json(schedule_path, schedule)
    manifest = plan["manifests"]["evaluation"]["path"]
    records = load_manifest(manifest)
    paths = {name: [] for name in names}
    with (destination / ".execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for item in schedule:
            name = item["variant"]
            cfg = _variant_config(plan, name, manifest, item["requests"], timing["candidate_count"], quality=False)
            cfg["benchmark"]["workload"] = timing["workload"]
            proof = selected["numerical_proofs"].get(name)
            if not proof or file_sha256(proof["path"]) != proof["sha256"]:
                raise ValueError(f"Finalist {name} requires an unchanged numerical proof frozen with selection")
            cfg["numerical_validation"] = proof["path"]
            receipt = _execute_run(directory / "blocks" / f"{item['index']:03d}", name, cfg, records)
            paths[name].append(receipt["path"])
    aggregates = {}
    for name in names:
        output = directory / "aggregate" / name
        if not output.exists():
            run = aggregate_timing_blocks(paths[name], output, expected_blocks=timing["blocks"], expected_requests_per_block=timing["requests_per_block"])
        else:
            run = load_run(output)
            if [block["artifact_sha256"] for block in run.summary["timing_blocks"]] != [artifact_sha256(load_run(path)) for path in paths[name]]:
                raise ValueError("Timing source blocks changed after aggregation")
        aggregates[name] = str(run.path)
    comparisons = [compare_timing_blocks(aggregates[names[0]], aggregates[name]) for name in names[1:]]
    atomic_json(directory / "comparisons.json", comparisons)
    return {"aggregate_runs": aggregates, "comparisons": comparisons}


def estimate_study(destination, finalists=None, *, baseline_seconds=57.5, hourly_price=4.6608, timing_candidates=None):
    """Transparent planning estimate; feature speedup and LPIPS time are unmeasured."""
    _, plan = _load_plan(destination)
    names = finalists or list(plan["variants"])
    if set(names) - plan["variants"].keys():
        raise ValueError("Unknown finalist")
    def factor(name):
        cfg = merge(plan["base_config"], plan["variants"][name])
        return cfg["generation"]["steps"] / 50
    development = 32 * plan["quality"]["candidate_count"] * sum(factor(n) for n in plan["variants"])
    held_out = (128 + 16) * plan["quality"]["candidate_count"] * sum(factor(n) for n in names)
    actions = 2 * plan["quality"]["action_subset_clips"] * plan["quality"]["action_candidate_count"] * sum(factor(n) for n in names)
    count = timing_candidates or plan["timing"]["candidate_count"]
    timing = plan["timing"]["blocks"] * plan["timing"]["requests_per_block"] * count * 2
    seconds = {"development_recorded_s": development * baseline_seconds,
               "held_out_and_second_seed_s": held_out * baseline_seconds,
               "development_and_evaluation_action_subsets_s": actions * baseline_seconds,
               "two_exact_finalist_timing_s": timing * baseline_seconds}
    hours = sum(seconds.values()) / 3600
    generation = plan["base_config"]["generation"]
    bytes_per_future = generation["total_frames"] * generation["height"] * generation["width"] * 3 * 4
    return {"baseline_seconds_per_future": baseline_seconds, "assumptions": [
        "Serial generation estimate; no assumed batching/cache speedup", "Reduced-step cost scales linearly for planning only",
        "Excludes loading/warmups, metrics, discovery failures, transfers and allocation overhead", "Two exact headline finalists; all declared finalists receive held-out quality"],
        "phase_seconds": seconds, "generation_hours": hours, "generation_cost_at_hourly_price": hours * hourly_price,
        "hourly_price": hourly_price, "headline_candidate_count": count,
        "peak_raw_shard_bytes_upper": 3 * plan["quality"]["shard_clips"] * max(plan["quality"]["action_candidate_count"], plan["quality"]["candidate_count"]) * bytes_per_future,
        "disk_note": "Plus fixed selected raw arrays, previews and the configured free-space reserve; adopted external arrays are never deleted"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan"); p.add_argument("--spec", required=True); p.add_argument("--output", required=True)
    p = sub.add_parser("quality"); p.add_argument("--study", required=True); p.add_argument("--phase", choices=PHASES, required=True)
    p.add_argument("--adopt", action="append", default=[], metavar="VARIANT=RUN"); p.add_argument("--max-shards", type=int)
    p.add_argument("--runtime-config", help="Quality-only YAML replacing environment proof/lock for an identical replacement GPU")
    p.add_argument("--adopt-source", action="append", default=[], metavar="VARIANT=SOURCE_RECEIPT")
    p = sub.add_parser("aggregate"); p.add_argument("--study", required=True); p.add_argument("--phase", choices=PHASES, required=True)
    p = sub.add_parser("freeze"); p.add_argument("--study", required=True); p.add_argument("--exact-runs", nargs="+", required=True)
    p.add_argument("--variants", nargs="+", required=True); p.add_argument("--numerical-proof", action="append", default=[], metavar="VARIANT=REPORT")
    p.add_argument("--timing-candidates", type=int); p.add_argument("--timing-workload", choices=("single_future", "candidate_set", "independent_scenes"))
    p.add_argument("--reviewed-policy", action="append", default=[], metavar="REFERENCE=FROZEN_POLICY",
                   help="Optional separately frozen nonzero development limits attributed only to this reference")
    p = sub.add_parser("timing"); p.add_argument("--study", required=True); p.add_argument("--variants", nargs="+", required=True)
    p = sub.add_parser("estimate"); p.add_argument("--study", required=True); p.add_argument("--finalists", nargs="+")
    p.add_argument("--baseline-seconds", type=float, default=57.5); p.add_argument("--hourly-price", type=float, default=4.6608); p.add_argument("--timing-candidates", type=int)
    args = parser.parse_args(argv)
    if args.command == "plan":
        result = create_plan(args.spec, args.output)
    elif args.command == "quality":
        result = execute_quality(args.study, args.phase, adopted=dict(x.split("=", 1) for x in args.adopt),
                                 adoption_sources=dict(x.split("=", 1) for x in args.adopt_source), max_shards=args.max_shards, runtime_config=args.runtime_config)
    elif args.command == "aggregate":
        result = aggregate_quality(args.study, args.phase)
    elif args.command == "freeze":
        result = freeze_selection(args.study, exact_runs=args.exact_runs, variants=args.variants,
                                  numerical_proofs=dict(x.split("=", 1) for x in args.numerical_proof),
                                  timing_candidates=args.timing_candidates, timing_workload=args.timing_workload,
                                  reviewed_policies=dict(x.split("=", 1) for x in args.reviewed_policy))
    elif args.command == "timing":
        result = execute_timing(args.study, args.variants)
    else:
        result = estimate_study(args.study, args.finalists, baseline_seconds=args.baseline_seconds,
                                hourly_price=args.hourly_price, timing_candidates=args.timing_candidates)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
