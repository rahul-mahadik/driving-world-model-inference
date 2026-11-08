#!/usr/bin/env python3
"""Audit the separate norm-qualified timing contract and sealed native reviews.

This supplementary report never rewrites legacy numerical/headline flags. Public
output is an allowlisted numeric projection. The authoritative workspace must be
available at its recorded paths. No inference, provider action or source edit is
performed. Numerical proof arrays may be read by the companion verifier.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import html
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drivebench.config import file_sha256
from drivebench.evaluate import aggregate_pairs, tolerance_gate
from drivebench.evaluation.study_policy import policy_for_comparison
from drivebench import study as study_module
from drivebench.benchmarks.statistics import bootstrap_quantile

TYPE = "compiler_norm_timing_report_v1"
SEAL = "native_blind_ratings_v1"
PIN = "native_blind_ratings_external_pin_v1"
REGISTRY = "norm_timing_sealed_review_registry_v1"
FIELDS = ["case_id", "preferred_panel_A_B_tie", "acceptable_A", "acceptable_B",
          "object_disappearance", "object_duplication", "freezing", "geometry_change",
          "action_response_notes", "reviewer_notes"]
PHASES = {"development": 32, "evaluation": 128, "evaluation_second_seed": 16,
          "development_actions": 16, "evaluation_actions": 16}


def module(name):
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location("norm_report_" + name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def now():
    return datetime.now(timezone.utc).isoformat()


def stamp(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset().total_seconds() != 0:
        raise ValueError("Evidence timestamps must explicitly use UTC")
    return result


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    value = json.loads(Path(path).read_text())
    require(isinstance(value, dict), "Expected a JSON object")
    return value


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def checked(path, expected):
    path = Path(path)
    require(not path.is_symlink() and path.is_file(), "Missing or symlinked evidence")
    require(isinstance(expected, str) and re.fullmatch(r"[a-f0-9]{64}", expected), "Explicit SHA256 required")
    require(file_sha256(path) == expected, "Bound evidence changed")
    return path.resolve()


def public_inventory(directory):
    directory = Path(directory).resolve()
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    require(files and not any(p.is_symlink() for p in directory.rglob("*")), "Empty or symlinked public review bundle")
    return {str(p.relative_to(directory)): file_sha256(p) for p in files}


def votes(path, wanted):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == FIELDS, "Ratings must retain the original complete header")
        rows = list(reader)
    indexed = {r["case_id"]: r for r in rows}
    require(len(indexed) == len(rows) and set(indexed) == set(wanted), "Ratings have missing or duplicate cases")
    for row in rows:
        require(row["preferred_panel_A_B_tie"] in {"A", "B", "tie", "uncertain"}
                and all(row[k] in {"yes", "no", "uncertain"} for k in ("acceptable_A", "acceptable_B"))
                and any(row[k].strip() for k in FIELDS[4:]), "Incomplete ratings or observations")
    return indexed


def public_cases(directory, inventory):
    directory = Path(directory).resolve()
    require({"index.html", "manifest.json", "review.csv"} <= set(inventory), "Incomplete native review bundle")
    manifest = read(directory / "manifest.json")
    require(manifest.get("receipt_type") == "blind_native_frame_export_v1", "Wrong native export type")
    cases = manifest["cases"]
    ids = [r["case_id"] for r in cases]
    require(ids and len(set(ids)) == len(ids) and all(re.fullmatch(r"case-[a-f0-9]{24}", key) for key in ids),
            "Invalid opaque case identities")
    expected_frames = []
    for row in cases:
        require(row.get("future_frames") == 24 and len(row.get("frames", [])) == 24,
                "Every native case must include all 24 future frames")
        expected = [f'{row["case_id"]}-frame-{number:04d}.png' for number in range(1, 25)]
        require(row["frames"] == expected, "Native frame order differs")
        require(type(row.get("panel_width")) is int and type(row.get("panel_height")) is int
                and row["panel_width"] == 1024 and row["panel_height"] == 576,
                "Native review resolution differs")
        require(row.get("pages") and all(p in inventory for p in row["pages"]), "Missing native review pages")
        expected_frames.extend(expected)
    require(len(set(expected_frames)) == len(expected_frames)
            and set(manifest["frame_sha256"]) == set(expected_frames)
            and all(inventory.get(k) == v for k, v in manifest["frame_sha256"].items()), "Native frame hashes differ")
    return ids


def seal_review(public, ratings, protocol, output, *, original_index_sha256,
                started_at, reviewer_kind="agent_assisted"):
    """Blind operation: reads public pixels/ratings/rubric, never private mappings."""
    public, ratings, protocol = map(lambda p: Path(p).resolve(), (public, ratings, protocol))
    require(reviewer_kind in {"agent_assisted", "human"}, "Declare the actual reviewer kind")
    inventory = public_inventory(public); cases = public_cases(public, inventory)
    require(not ratings.is_relative_to(public), "Keep ratings separate from the immutable public bundle")
    require(not Path(output).resolve().is_relative_to(public), "Keep the seal outside the immutable public bundle")
    rows = votes(ratings, cases)
    created = now()
    require(stamp(read(protocol)["created_at"]) <= stamp(started_at) <= stamp(created), "Invalid review chronology")
    require(isinstance(original_index_sha256, str) and re.fullmatch(r"[a-f0-9]{64}", original_index_sha256),
            "Coordinator must provide the original anonymous review-index SHA256")
    value = {"schema_version": 1, "receipt_type": SEAL, "created_at": created,
        "review_started_at": started_at, "reviewer_kind": reviewer_kind,
        "protocol": {"path": str(protocol), "sha256": file_sha256(protocol)},
        "review_index": {"sha256": original_index_sha256},
        "public_directory": str(public), "public_artifact_sha256": inventory,
        "ratings": {"path": str(ratings), "sha256": file_sha256(ratings)},
        "reviewed_case_ids": cases, "reviewed_frames_per_case": {key: 24 for key in cases},
        "uncertain_case_ids": sorted(k for k, row in rows.items() if "uncertain" in
            (row["preferred_panel_A_B_tie"], row["acceptable_A"], row["acceptable_B"])),
        "all_panels_reviewed": True, "private_keys_read_before_seal": False,
        "helper_sha256": file_sha256(__file__),
        "attestation_scope": "Reviewer declaration plus bound artifacts; does not prove an observer never saw identities elsewhere"}
    write_once(output, value)
    return value


def pin_seal(seal, expected_sha256, output):
    path = checked(seal, expected_sha256); value = read(path)
    require(value.get("receipt_type") == SEAL, "Wrong review seal type")
    pinned = now(); require(stamp(value["created_at"]) <= stamp(pinned), "Seal lies in the future")
    result = {"schema_version": 1, "receipt_type": PIN, "created_at": pinned,
        "seal": {"path": str(path), "sha256": expected_sha256},
        "private_keys_read_before_pin": False,
        "scope": "Coordinator pins the externally supplied seal hash before attribution"}
    write_once(output, result)
    return result


def expected_metrics(evidence, study, candidate, reference):
    result = {}
    for phase in PHASES:
        path = study / "quality" / phase / "aggregate" / f"evaluation-vs-{reference}.json"
        document = evidence.read(path)
        entries = [r for r in document["runs"] if r["run"] == candidate]
        require(len(entries) == 1, "Missing or duplicated selected quality comparison")
        for artifact in entries[0]["metric_artifacts"]:
            metric = evidence.path(artifact["path"])
            require(metric not in result and evidence.check(metric, artifact["sha256"]), "Metric artifact changed or reused across phases")
            result[metric] = (phase, artifact["sha256"])
    return result


def verify_review_registry(evidence, registry_path, expected):
    """Validate every seal before any private key is parsed or attributed."""
    registry = evidence.read(registry_path)
    require(registry.get("schema_version") == 1 and registry.get("receipt_type") == REGISTRY,
            "A sealed native-review registry is required")
    submitted = registry.get("reviews", [])
    indexed = {evidence.path(r["metric"]): r for r in submitted}
    require(len(indexed) == len(submitted) and set(indexed) == set(expected), "Review registry must exactly cover selected comparisons in all five phases")
    ratings, bindings, kinds = [], {}, set()
    for metric, item in indexed.items():
        require(item["metric_sha256"] == expected[metric][1], "Registry metric binding differs")
        checked(metric, item["metric_sha256"]); doc = evidence.read(metric)
        native_path = checked(evidence.path(item["native_receipt"]), item["native_receipt_sha256"])
        native = evidence.read(native_path)
        require(native.get("receipt_type") == "blind_native_frame_export_v1", "Wrong native receipt")
        binding = native["binding"]
        require(evidence.path(binding["metric"]) == metric and binding["metric_sha256"] == item["metric_sha256"],
                "Native export belongs to a different comparison")
        metric_receipt = checked(evidence.path(binding["metric_receipt"]), binding["metric_receipt_sha256"])
        receipt = evidence.read(metric_receipt)
        require(receipt["evaluation_sha256"] == item["metric_sha256"]
                and receipt["review_artifact_sha256"] == binding["review_artifact_sha256"]
                and binding["wrapper_sha256"] == file_sha256(Path(__file__).with_name("run_quality_bounded.py")),
                "Native source receipt or exporter differs")
        seal_path = checked(evidence.path(item["seal"]), item["seal_sha256"]); seal = evidence.read(seal_path)
        pin_path = checked(evidence.path(item["pin"]), item["pin_sha256"]); pin = evidence.read(pin_path)
        require(seal.get("receipt_type") == SEAL and seal.get("schema_version") == 1
                and seal.get("reviewer_kind") in {"human", "agent_assisted"}
                and seal.get("private_keys_read_before_seal") is False and seal.get("all_panels_reviewed") is True,
                "Missing explicit blinded-review declarations")
        require(pin.get("receipt_type") == PIN and pin.get("private_keys_read_before_pin") is False
                and evidence.path(pin["seal"]["path"]) == seal_path and pin["seal"]["sha256"] == item["seal_sha256"],
                "External seal pin differs")
        public = evidence.path(seal["public_directory"])
        require(public == native_path.parent / "public", "Seal points at a different native export")
        inventory = public_inventory(public)
        require(inventory == native["public_artifact_sha256"] == seal["public_artifact_sha256"], "Viewed native artifacts changed")
        cases = public_cases(public, inventory)
        require(len(cases) == doc["review"]["cases"] == 4
                and seal["reviewed_case_ids"] == cases and seal["reviewed_frames_per_case"] == {k: 24 for k in cases},
                "Incomplete native frame review")
        protocol = checked(evidence.path(seal["protocol"]["path"]), seal["protocol"]["sha256"])
        require(stamp(evidence.read(protocol)["created_at"]) <= stamp(native["created_at"])
                <= stamp(seal["review_started_at"]) <= stamp(seal["created_at"])
                <= stamp(pin["created_at"]) <= stamp(now()), "Review/export/pin chronology is invalid")
        index = checked(evidence.path(doc["review"]["index"]), seal["review_index"]["sha256"])
        require(binding["review_artifact_sha256"].get(str(index)) == seal["review_index"]["sha256"], "Original review index differs")
        csv_path = checked(evidence.path(seal["ratings"]["path"]), seal["ratings"]["sha256"])
        require(csv_path != evidence.path(doc["review"]["csv"]) and not csv_path.is_relative_to(public), "Ratings overwrite the original blank CSV")
        rows = votes(csv_path, cases)
        uncertain = sorted(k for k, row in rows.items() if "uncertain" in
            (row["preferred_panel_A_B_tie"], row["acceptable_A"], row["acceptable_B"]))
        require(seal["uncertain_case_ids"] == uncertain, "Uncertainty declaration differs")
        ratings.append({"index": str(index), "index_sha256": seal["review_index"]["sha256"],
                        "csv": str(csv_path), "csv_sha256": seal["ratings"]["sha256"]})
        kinds.add(seal["reviewer_kind"])
        for path, checksum in ((metric, item["metric_sha256"]), (native_path, item["native_receipt_sha256"]),
                               (metric_receipt, binding["metric_receipt_sha256"]), (seal_path, item["seal_sha256"]),
                               (pin_path, item["pin_sha256"]), (protocol, seal["protocol"]["sha256"]),
                               (csv_path, seal["ratings"]["sha256"]), (index, seal["review_index"]["sha256"])):
            bindings[str(path)] = checksum
        bindings.update({str(public / p): checksum for p, checksum in inventory.items()})
    require(len(kinds) == 1, "Use a registry with one accurately declared reviewer kind")
    bindings[str(evidence.path(registry_path))] = file_sha256(evidence.path(registry_path))
    return {"reviewer_kind": kinds.pop(), "reviews": ratings}, bindings


def action_integrity(entry, count):
    pairs = entry["pairs"]; indexed = {}
    for pair in pairs:
        key = (pair["request_id"], pair.get("repeat", 0))
        indexed.setdefault(key, []).append(pair)
    actions = entry.get("action_response", [])
    require(len(indexed) == count and len(actions) == count and len({(r["request_id"], r.get("repeat", 0)) for r in actions}) == count,
            "Missing or duplicate controlled action responses")
    for action in actions:
        candidates = indexed[(action["request_id"], action.get("repeat", 0))]
        ordered = [next(p for p in candidates if p["candidate_id"] == action[key])
                   for key in ("recorded_candidate_id", "alternative_candidate_id")]
        trajectories = [p["trajectory_sha256"] for p in ordered]
        outputs = [p["reference_output_sha256"] for p in ordered] + [p["candidate_output_sha256"] for p in ordered]
        require(len(candidates) == 2 and ordered[0] is not ordered[1]
                and ordered[0]["action_kind"] == "recorded" and ordered[1]["action_kind"] != "recorded"
                and all(p["seed"] == action["seed"] and p["scene_id"] == action["scene_id"] for p in ordered)
                and len(set(trajectories)) == 2 and trajectories == action["trajectory_sha256"]
                and outputs == action["output_sha256"], "Action diagnostic is not paired at a fixed seed with different trajectories")
        for value in action["metrics"].values():
            require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "Invalid action response error")
        require({"reference_response_mae", "candidate_response_mae", "response_error_mae", "response_error_rmse", "response_error_max_abs"}
                <= set(action["metrics"]), "Missing action response metrics")
    return actions


def quality_bindings(evidence, study, phase):
    """Record verified small metadata and review hashes, never retired raw arrays."""
    folder = study / "quality" / phase
    aggregation = folder / "aggregation-receipt.json"
    values = {str(aggregation): file_sha256(aggregation)}
    values.update(evidence.read(aggregation)["artifact_sha256"])
    for path in sorted((folder / "shards").glob("*/complete.json")):
        item = evidence.read(path); values[str(path)] = file_sha256(path)
        values.update(item["metric_artifacts"])
        for run in item["runs"].values():
            values.update({str(evidence.path(run["path"]) / key): sha for key, sha in run["artifact_sha256"].items()})
        for metric in item["metric_artifacts"]:
            if str(metric).endswith(".receipt.json"):
                values.update(evidence.read(metric)["review_artifact_sha256"])
    return values


def phase_quality(old, evidence, study, phase, expected_count, value, ratings, selected):
    reference, candidate = value["baseline"], value["candidate"]
    path = study / "quality" / phase / "aggregate" / f"evaluation-vs-{reference}.json"
    document = evidence.read(path); entry = next(r for r in document["runs"] if r["run"] == candidate)
    require(old.verify_phase(evidence, study / "quality" / phase), "Quality phase receipts are incomplete or stale")
    require(evidence.check_run(document["reference"], document["reference_artifact_sha256"])
            and evidence.check_run(entry["run_path"], entry["artifact_sha256"]), "Quality run artifacts changed")
    plan = evidence.read(study / "plan.json"); manifest = plan["manifests"][phase]
    config, summary, _ = evidence.run(entry["run_path"])
    ref_config, ref_summary, _ = evidence.run(document["reference"])
    require(summary.get("synthetic") is False and ref_summary.get("synthetic") is False,
            "Synthetic quality runs cannot qualify a real timing claim")
    expected_candidates = 2 if phase.endswith("_actions") else 1
    pairs = entry["pairs"]; metadata = entry["manifest_metadata"]
    require(config["data"]["manifest_sha256"] == manifest["sha256"]
            and metadata == manifest["metadata"] and metadata["clip_count"] == expected_count
            and len({p["request_id"] for p in pairs}) == expected_count
            and {p["scene_id"] for p in pairs} == set(metadata["scene_ids"])
            and len(pairs) == entry["matched_pairs"] == expected_count * expected_candidates
            and len({(p["request_id"], p.get("repeat", 0), p["candidate_id"], p["seed"]) for p in pairs}) == len(pairs)
            and config["benchmark"]["candidate_count"] == expected_candidates
            and entry.get("complete_pairs") is True and entry.get("synthetic") is False and not entry.get("errors"),
            "Incomplete real quality coverage")
    for name, cfg, run_path in ((candidate, config, entry["run_path"]), (reference, ref_config, document["reference"])):
        require(all(cfg.get(k) == value["configs"][name].get(k) for k in ("model", "generation", "optimizations")),
                "Quality implementation differs from measured timing")
        # Historical upstream development adoption is already bound by the frozen
        # development policy. Held-out and action sources must match the plan.
        if not (phase == "development" and name == reference):
            require(old.quality_implementation_matches(evidence, run_path, cfg, value["configs"][name], plan, phase),
                    "Quality source, runtime or partition differs")
    perceptual = document.get("perceptual", {})
    setup_path = evidence.path(plan["quality"]["evaluation_setup_receipt"])
    setup = evidence.read(setup_path); provenance = perceptual.get("provenance", {})
    require(perceptual.get("status") == "available" and setup.get("passed") is True
            and setup.get("cpu_verification", {}).get("passed") is True
            and provenance.get("setup_receipt_sha256") == file_sha256(setup_path)
            and provenance.get("state_dict_sha256") == setup["cpu_verification"]["model_state_dict_sha256"]
            and provenance.get("device") == plan["quality"]["lpips_device"],
            "Actual perceptual evaluator and verified weights must match")
    aggregates = aggregate_pairs(pairs)
    require(aggregates == entry["aggregates"], "Quality aggregate differs from recorded pairs")
    policy_path = policy_for_comparison(selected, reference, candidate)
    binding = entry.get("quality_policy") or document["tolerance_policies_by_run"][candidate]
    # Development is evaluated before selection, so its metric artifact correctly
    # retains an unfrozen policy. The selected policy binds this exact aggregate.
    if phase != "development":
        require(binding.get("reference") == reference and evidence.path(binding["path"]) == evidence.path(policy_path)
                and evidence.check(policy_path, binding["sha256"]), "Held-out comparison policy differs")
    policy = yaml.safe_load(evidence.path(policy_path).read_text())
    accepted, cases, hashes = old.accepted_variant_reviews(evidence, entry, ratings, reference)
    checks = {name: bool(aggregates.get(name, {}).get("pairs") == len(pairs)
              and type(aggregates.get(name, {}).get(setting["aggregation"])) in (int, float)
              and aggregates[name][setting["aggregation"]] <= setting["maximum"])
              for name, setting in policy["metrics"].items()}
    metric_pass = bool(checks and all(checks.values()))
    actions = action_integrity(entry, expected_count) if phase.endswith("_actions") else []
    if phase in {"evaluation", "evaluation_second_seed"}:
        gate = tolerance_gate(aggregates, policy, complete_pairs=True, synthetic=False,
            evaluation_manifest_sha256=config["data"]["manifest_sha256"], evaluation_manifest_metadata=metadata,
            run_started_at=config["started_at"], reviewed=accepted, evaluated_scene_ids=sorted({p["scene_id"] for p in pairs}),
            evaluated_request_ids=sorted({p["request_id"] for p in pairs}), evaluated_pair_count=len(pairs))
        phase_accepted = bool(accepted and gate["quality_preserving"])
    elif phase.endswith("_actions"):
        phase_accepted = bool(accepted and metric_pass and len(actions) == expected_count)
    else:
        phase_accepted = bool(accepted and metric_pass)
    return {"phase": phase, "status": "verified", "accepted": phase_accepted,
        "clips": expected_count, "scenes": len({p["scene_id"] for p in pairs}), "pairs": len(pairs),
        "candidate_count": expected_candidates, "accepted_blind_cases": cases if accepted else 0,
        "blinded_review_accepted": accepted, "frozen_metric_limits_met": metric_pass,
        "metrics": old.summary_metrics(pairs), "recorded_ground_truth": old.summary_metrics(pairs, field="recorded_ground_truth"),
        "action_response": {"controlled_pairs": len(actions), "same_seed_and_distinct_actions_verified": bool(actions),
            "metrics": old.summary_metrics(actions, names=old.ACTION_METRICS),
            "scope": "Response errors are diagnostic; no calibrated response-error threshold, trajectory adherence or driving safety claim"},
        "policy_sha256": file_sha256(evidence.path(policy_path)), "metric_artifact_sha256": [a["sha256"] for a in entry["metric_artifacts"]],
        "lpips_state_sha256": old.sha(provenance.get("state_dict_sha256")),
        "review_artifact_sha256": hashes,
        "raw_action_response": [{"request_id": old.identifier(row["request_id"]), "scene_id": old.identifier(row["scene_id"]),
            "seed": old.integer(row["seed"]), "trajectory_sha256": [old.sha(v) for v in row["trajectory_sha256"]],
            "output_sha256": [old.sha(v) for v in row["output_sha256"]],
            "metrics": old.numeric_metrics(row["metrics"], old.ACTION_METRICS)} for row in actions],
        "raw_pairs": [{"request_id": old.identifier(p["request_id"]), "scene_id": old.identifier(p["scene_id"]),
            "candidate_id": old.identifier(p["candidate_id"]), "seed": old.integer(p["seed"]),
            "metrics": old.numeric_metrics(p["metrics"], old.METRICS)} for p in pairs]}


def timing_projection(old, evidence, result):
    names = (result["baseline"], result["candidate"]); runs = []
    for name in names:
        rows, blocks, values, scenes = [], [], [], []
        selected = [b for b in result["blocks"] if b["variant"] == name]
        require(len(selected) == 4, "Four timing blocks per variant required")
        for block in selected:
            cfg, summary, measured = evidence.run(block["path"])
            require(len(measured) == 50, "Each timing block needs exactly 50 requests")
            for row in measured:
                latency = old.number(row.get("latency_s"))
                require(latency is not None and latency > 0, "Invalid raw timing")
                values.append(latency); scenes.append(row["scene_id"])
                rows.append({"block": block["variant_block"], "request_id": old.identifier(row["request_id"]),
                    "scene_id": old.identifier(row["scene_id"]), "repeat": old.integer(row.get("repeat", 0)),
                    "latency_s": latency, "candidate_count": len(row["candidates"]),
                    "actual_batch_size": old.integer(row.get("batch_size")),
                    "denoiser_evaluations": old.integer(row.get("denoiser_evaluations")),
                    "peak_allocated_bytes": old.number(row.get("peak_allocated_bytes")),
                    "peak_reserved_bytes": old.number(row.get("peak_reserved_bytes")),
                    "stage_timings_s": {old.identifier(k): old.number(v) for k, v in row.get("stage_timings_s", {}).items()},
                    "candidates": [{"candidate_id": old.identifier(c["candidate_id"]), "seed": old.integer(c.get("seed")),
                        "trajectory_sha256": old.sha(c.get("trajectory_sha256"))} for c in row["candidates"]],
                    "status": "ok" if row.get("status") == "ok" else "failed"})
            cold_path = evidence.path(block["path"]) / "cold_start.json"
            cold = evidence.read(cold_path) if cold_path.exists() else {}
            blocks.append({"index": block["index"], "variant_block": block["variant_block"],
                "requests": len(measured), "legacy_headline_eligible": summary.get("headline_eligible") is True,
                "legacy_numerical_passed": summary.get("numerical_validation", {}).get("passed") is True,
                "norm_execution_eligible": block.get("execution", {}).get("norm_execution_eligible") is True,
                "started_at": old.timestamp(cfg.get("started_at")), "finished_at": old.timestamp(block.get("finished_at")),
                "measured_wall_s": old.number(summary.get("throughput_measurement_wall_s")),
                "cold_start": {"model_load_s": old.number(cold.get("model_load_s")),
                    "warmup_latencies_s": [old.number(row.get("latency_s")) for row in cold.get("warmups", [])],
                    "compilation_and_initialization_excluded": cold.get("compilation_and_initialization_excluded") is True,
                    "compile_only_duration_measured": False},
                "artifact_sha256": {old.identifier(k): old.sha(v) for k, v in block["artifact_sha256"].items()},
                "receipt_sha256": old.sha(block["receipt_sha256"]), "process_sha256": old.sha(block["process_sha256"])})
        require(len(rows) == 200, "Two hundred observations required per variant")
        runs.append({"variant": old.identifier(name), "requests": 200, "unique_clips": len({r["request_id"] for r in rows}),
            "unique_scenes": len(set(scenes)), "candidate_count": result["candidate_count"], "workload": old.identifier(result["workload"]),
            "p50_s": float(np.quantile(values, .5)), "p95_s": float(np.quantile(values, .95)),
            "p50_ci95_s": bootstrap_quantile(values, .5, groups=scenes), "p95_ci95_s": bootstrap_quantile(values, .95, groups=scenes),
            "peak_allocated_bytes": max((r["peak_allocated_bytes"] for r in rows if r["peak_allocated_bytes"] is not None), default=None),
            "peak_reserved_bytes": max((r["peak_reserved_bytes"] for r in rows if r["peak_reserved_bytes"] is not None), default=None),
            "blocks": blocks, "raw_measurements": rows})
    comparison = result["comparison"]
    metrics = {key: {"relative_latency_change": old.number(row.get("relative_latency_change")),
        "ci95": old.interval(row.get("ci95")), "tolerance": old.number(row.get("tolerance")),
        "exceeds_tolerance_and_uncertainty": row.get("exceeds_tolerance_and_uncertainty") is True,
        "improvement_exceeds_tolerance_and_uncertainty": row.get("improvement_exceeds_tolerance_and_uncertainty") is True}
        for key, row in comparison["metrics"].items() if key in {"p50", "p95"}}
    require(set(metrics) == {"p50", "p95"} and all(m["ci95"] is not None and m["tolerance"] is not None
            and m["relative_latency_change"] is not None for m in metrics.values()), "Incomplete timing comparison")
    improved = comparison.get("status") == "pass" and all(m["improvement_exceeds_tolerance_and_uncertainty"]
        and m["relative_latency_change"] < -m["tolerance"] and m["ci95"][1] < -m["tolerance"] for m in metrics.values())
    return {"runs": runs, "comparison": {"status": old.identifier(comparison["status"]), "metrics": metrics,
        "paired_requests": old.integer(comparison.get("paired_requests")), "independent_scenes": old.integer(comparison.get("independent_scenes")),
        "improvement_demonstrated": bool(improved), "interpretation": "A no-regression pass alone is not evidence of speedup"}}


def numerical_projection(old, qualification):
    def errors(value):
        histogram = value.get("absolute_error_histogram", {})
        return {"rmse": old.number(value.get("rmse")), "max_abs_error": old.number(value.get("max_abs_error")),
            "values": old.integer(value.get("values")), "passed": value.get("passed") is True,
            "conditioning_frames_excluded": old.integer(value.get("conditioning_frames_excluded")),
            "absolute_error_histogram": {"edges": [old.number(v) for v in histogram.get("edges", [])],
                                         "counts": [old.integer(v) for v in histogram.get("counts", [])]},
            "per_frame": [{"frame": old.integer(r.get("frame")), "rmse": old.number(r.get("rmse")),
                "max_abs_error": old.number(r.get("max_abs_error")),
                "p50_p95_p99_abs_error": [old.number(v) for v in r.get("p50_p95_p99_abs_error", [])]}
                for r in value.get("per_frame", [])]}
    numerical = qualification.get("numerical", {}); contract = numerical.get("contract", {})
    development = qualification.get("quality_development", {}); semantics = qualification.get("exact_semantics", {})
    return {"report_sha256": old.sha(qualification.get("report_sha256")), "contract_type": "compiler_norm_qualification_v1",
        "legacy_strict_reinterpreted": False,
        "numerical_contract": {"metric": old.identifier(contract.get("metric", "unavailable")),
            "maximum": old.number(contract.get("maximum")), "aggregation": old.identifier(contract.get("aggregation", "unavailable")),
            "conditioning_frames": old.integer(contract.get("conditioning_frames")), "total_frames": old.integer(contract.get("total_frames")),
            "normalization": "RGB values in [0,1]"},
        "proof_candidates": {old.identifier(k): errors(v) for k, v in numerical.get("candidates", {}).items()},
        "development_errors": {old.identifier(k): errors(v) for k, v in development.get("numerical_by_request", {}).items()},
        "development_metric_maxima": old.numeric_metrics(development.get("metrics_maximum", {}), old.METRICS),
        "development_reference_source_identities": [old.sha(s) for s in development.get("reference_source_identities", [])],
        "exact_semantics": {"passed": semantics.get("passed") is True, "atol": old.number(semantics.get("atol")),
            "rtol": old.number(semantics.get("rtol")), "checks": {key: row.get("passed") is True for key, row in semantics.get("checks", {}).items()
                if key in {"candidate_order", "unrelated_candidate_action_change", "conditioning_cache_agreement", "conditioning_cache_keys", "conditioning_cache_exercised"}}}}


def render(data):
    esc = html.escape
    lines = ['<!doctype html><meta charset="utf-8"><title>Norm-qualified timing</title>',
        '<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}table{border-collapse:collapse}code{overflow-wrap:anywhere}</style>',
        '<h1>Norm-qualified timing supplement</h1>', f'<p>{esc(data["status"])}</p>',
        '<p>This separate qualification preserves the failed legacy strict compiler comparison. It does not establish bitwise equivalence, trajectory adherence or driving safety.</p>',
        '<p><a href="results.json">Sanitized raw timings, quality diagnostics and evidence hashes</a></p>']
    for result in data["companions"]:
        lines += [f'<h2>{esc(result["name"])}</h2>', f'<p>Status: {esc(result["status"])}. Combined claim: {result["combined_headline_claim"]}. Legacy compiler strict pass: false.</p>']
        if result.get("timing"):
            lines.append('<table><tr><th>Variant</th><th>Requests / unique clips</th><th>p50 seconds</th><th>p95 seconds</th></tr>')
            for run in result["timing"]["runs"]:
                lines.append(f'<tr><td>{esc(run["variant"])}</td><td>{run["requests"]} / {run["unique_clips"]}</td><td>{run["p50_s"]:.5f}</td><td>{run["p95_s"]:.5f}</td></tr>')
            lines.append('</table>')
        for phase in result.get("quality", []):
            lines.append(f'<p>{esc(phase["phase"])}: {esc(phase["status"])}; accepted {phase["accepted"]}.</p>')
    return "\n".join(lines) + "\n"


def build(study, companions, output, *, review_registry=None, workspace_root=None):
    old = module("report_full_study"); verifier = module("run_norm_timing_companion")
    evidence = old.Evidence(workspace_root or Path.cwd())
    study, output = evidence.path(study), Path(output).resolve()
    require(not output.exists(), "Use a fresh immutable report directory")
    output.mkdir(parents=True, mode=0o700)
    plan = evidence.read(study / "plan.json"); selected = study_module._selection(study, plan)
    data = {"schema_version": 1, "type": TYPE, "created_at": now(), "status": "incomplete: individual gates remain explicit",
        "parent_plan_sha256": file_sha256(study / "plan.json"), "selection_sha256": file_sha256(study / "selection.json"),
        "companions": [], "headline_claims": [], "public_projection": {"paths": False, "gpu_uuids": False,
        "accounts": False, "raw_configs": False, "private_keys": False, "frames": False, "reviewer_notes": False},
        "qualification_distinction": "New norm-qualified contract; original strict-zero and legacy headline flags are preserved"}
    private = {"created_at": now(), "input_sha256": {str(study / "plan.json"): data["parent_plan_sha256"],
        str(study / "selection.json"): data["selection_sha256"], str(Path(__file__).resolve()): file_sha256(__file__)}, "errors": []}
    for name in ("report_full_study", "run_norm_timing_companion", "run_quality_bounded"):
        path = Path(__file__).with_name(name + ".py").resolve()
        private["input_sha256"][str(path)] = file_sha256(path)
    for name, path in companions.items():
        result = {"name": old.identifier(name), "status": "missing_or_invalid_timing", "combined_headline_claim": False,
                  "timing": None, "quality": [], "legacy_strict_passed": {"baseline": True, "candidate": False}}
        data["companions"].append(result)
        try:
            value = verifier.verify_completed(evidence.path(path))
            require(value.get("type") == "compiler_norm_timing_result_v1" and value.get("timing_qualified") is True
                    and value.get("combined_headline_claim") is False and evidence.path(value["parent_study"]) == study
                    and value["parent_plan_sha256"] == data["parent_plan_sha256"]
                    and value["selection_sha256"] == data["selection_sha256"]
                    and value["legacy_strict_passed"] == {value["baseline"]: True, value["candidate"]: False},
                    "Companion contract or parent binding differs")
            result.update(status="timing_verified_quality_pending", timing=timing_projection(old, evidence, value),
                candidate_count=value["candidate_count"], workload=old.identifier(value["workload"]),
                qualification=numerical_projection(old, value["qualification"]),
                source_identity=old.sha(value["source_identity"]), plan_sha256=old.sha(value["plan_sha256"]),
                pilot_policy_sha256=old.sha(value["pilot_policy"].get("sha256")),
                legacy_headline_eligible={old.identifier(k): v is True for k, v in value["legacy_headline_eligible"].items()})
            private["input_sha256"].update(value["artifact_sha256"])
            for item in (value["pilot_policy"], value["qualification"].get("policy", {})):
                if item.get("path"):
                    private["input_sha256"][str(evidence.path(item["path"]))] = item["sha256"]
            if value["qualification"].get("report_path"):
                private["input_sha256"][str(evidence.path(value["qualification"]["report_path"]))] = value["qualification"]["report_sha256"]
            require(review_registry is not None, "Sealed native review registry is missing")
            expected = expected_metrics(evidence, study, value["candidate"], value["baseline"])
            ratings, review_bindings = verify_review_registry(evidence, review_registry, expected)
            private["input_sha256"].update(review_bindings)
            # Every required seal and external pin is validated and committed
            # before old review helpers can parse a private A/B key.
            attribution_path = output / f"attribution-start-{len(data['companions']):03d}.json"
            write_once(attribution_path, {"type": "sealed_review_attribution_start_v1", "created_at": now(),
                "registry_sha256": file_sha256(evidence.path(review_registry)), "verified_input_sha256": review_bindings,
                "all_required_seals_verified_before_any_private_key_read": True, "helper_sha256": file_sha256(__file__)})
            private["input_sha256"][str(attribution_path)] = file_sha256(attribution_path)
            result["review_chronology"] = {"verified": True, "reviewer_kind": ratings["reviewer_kind"],
                "attribution_receipt_sha256": file_sha256(attribution_path), "scope": "Bound sealed declarations and actual report read order; not proof of observers' prior knowledge"}
            result["quality"] = [phase_quality(old, evidence, study, phase, count, value, ratings, selected)
                                 for phase, count in PHASES.items()]
            for phase in PHASES:
                private["input_sha256"].update(quality_bindings(evidence, study, phase))
            setup = evidence.path(plan["quality"]["evaluation_setup_receipt"])
            private["input_sha256"][str(setup)] = file_sha256(setup)
            quality_accepted = all(row["accepted"] for row in result["quality"])
            result["combined_headline_claim"] = bool(quality_accepted and result["timing"]["comparison"]["improvement_demonstrated"])
            result["status"] = "verified_with_quality_acceptance" if quality_accepted else "verified_metrics_or_reviews_not_accepted"
            if result["combined_headline_claim"]:
                data["headline_claims"].append({"companion": result["name"], "candidate_count": result["candidate_count"],
                    "scope": "Norm-qualified serial implementation; accepted recorded quality and fixed review subset, with completed controlled action diagnostics; not bitwise equivalent"})
        except (OSError, ValueError, KeyError, TypeError, StopIteration, RuntimeError) as exc:
            private["errors"].append({"companion": str(name), "error_type": type(exc).__name__, "message": str(exc)})
            result["combined_headline_claim"] = False
            if result["timing"] is not None:
                result["status"] = "timing_verified_quality_missing_or_invalid"
    for path, checksum in private["input_sha256"].items():
        checked(path, checksum)
    if data["companions"] and all(r["status"] == "verified_with_quality_acceptance" for r in data["companions"]):
        data["status"] = "requested companions and quality acceptance verified; improvement gates remain separate"
    public = output / "public"; public.mkdir()
    encoded = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    for folder in (output, public):
        (folder / "results.json").write_text(encoded)
        (folder / "index.html").write_text(render(data))
    write_once(output / "private-audit.json", private)
    write_once(public / "projection-manifest.json", {"schema_version": 1,
        "files": {p.name: {"sha256": file_sha256(p), "bytes": p.stat().st_size} for p in public.iterdir() if p.is_file()},
        "excluded": ["private_paths", "account_identifiers", "gpu_uuids", "frames", "weights", "private_keys", "reviewer_notes"]})
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("seal")
    for field in ("public", "ratings", "protocol", "output", "original-index-sha256", "started-at"):
        p.add_argument("--" + field, required=True)
    p.add_argument("--reviewer-kind", choices=("agent_assisted", "human"), default="agent_assisted")
    p = sub.add_parser("pin")
    for field in ("seal", "sha256", "output"):
        p.add_argument("--" + field, required=True)
    p = sub.add_parser("report"); p.add_argument("--study", required=True); p.add_argument("--output", required=True)
    p.add_argument("--companion", required=True, action="append", metavar="NAME=PATH")
    p.add_argument("--review-registry"); p.add_argument("--workspace-root", default=str(Path.cwd()))
    args = parser.parse_args(argv)
    if args.command == "seal":
        result = seal_review(args.public, args.ratings, args.protocol, args.output,
            original_index_sha256=args.original_index_sha256, started_at=args.started_at, reviewer_kind=args.reviewer_kind)
    elif args.command == "pin":
        result = pin_seal(args.seal, args.sha256, args.output)
    else:
        pairs = [r.split("=", 1) for r in args.companion]
        require(all(len(r) == 2 for r in pairs) and len(dict(pairs)) == len(pairs), "Use distinct NAME=PATH companions")
        result = build(args.study, dict(pairs), args.output, review_registry=args.review_registry, workspace_root=args.workspace_root)
    print(json.dumps({"status": result.get("status", "saved"), "type": result.get("type", result.get("receipt_type")),
                      "headline_claim_count": len(result.get("headline_claims", []))}))


if __name__ == "__main__":
    main()
