#!/usr/bin/env python3
"""Build an audited local study report and a sanitized, text/plot-only public copy.

No inference, array loading, cloud access or source-package mutation occurs.
Public results use an allowlist and omit paths, UUIDs, accounts, raw configurations,
commands, environment variables, weights, images, and freeform reviewer notes.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
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
from drivebench.evaluation.metrics import bootstrap_mean
from drivebench.benchmarks.environment import reference_identity
from drivebench.benchmarks.statistics import bootstrap_quantile
from drivebench.regression import _bootstrap_ratios

PHASES = ("development", "evaluation", "evaluation_second_seed", "development_actions", "evaluation_actions")
METRICS = ("mae", "rmse", "max_abs_error", "lpips", "temporal_change_error", "freeze_fraction_increase", "temporal_derivative_mae")
ACTION_METRICS = ("reference_response_mae", "candidate_response_mae", "response_error_mae", "response_error_rmse", "response_error_max_abs")
CORE = ("config.json", "environment.json", "measurements.jsonl", "summary.json")
COVERAGE_NOTE = ("Description labels mean literal scene-description mentions, not measured clip lighting/weather. "
                 "Nearby objects are annotation-center to ego-pose XY distances <=30m in all directions, not front-camera visibility. "
                 "Slow/lateral labels describe recorded endpoint displacement, not validated driving maneuvers.")
RUNTIME_MATCHING_CRITERION = {
    "schema_version": 1, "criterion": "development_paired_runtime_equivalence_v1",
    "predeclared_at": "2026-10-27T00:15:29+00:00",
    "scope": "Development runtime matching only; not a generation quality threshold or headline timing gate",
    "selection": "Nearest absolute log median-latency ratio among predeclared reduced-step controls with steps<50",
    "ratio": "feature median request latency / reduced-step median request latency",
    "ratio_bounds_inclusive": [0.98, 1.02], "confidence_level": 0.95,
    "bootstrap": "Paired scene clusters retaining all repeats", "resamples": 2000, "seed": 0,
    "minimum_real_development_scenes": 32,
    "decision": "Entire paired scene-bootstrap95% ratio interval inside inclusive bounds",
    "otherwise": "nearest_measured_control",
}
RUNTIME_MATCHING_CRITERION_SHA256 = hashlib.sha256(json.dumps(
    RUNTIME_MATCHING_CRITERION, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def identifier(value):
    text = str(value)
    private_shape = (re.search(r"(?:^GPU-|[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12})", text)
                     or re.fullmatch(r"(?:[0-9]{1,3}\.){3}[0-9]{1,3}", text)
                     or any(word in text.lower() for word in ("secret", "credential", "password", "account", "client_key")))
    return text if not private_shape and re.fullmatch(r"[A-Za-z0-9_.:-]{1,180}", text) else "id-" + hashlib.sha256(text.encode()).hexdigest()[:16]


def sha(value):
    return value if isinstance(value, str) and re.fullmatch(r"[a-fA-F0-9]{40}|[a-fA-F0-9]{64}", value) else None


def number(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def integer(value):
    return value if type(value) is int else None


def interval(value):
    return [number(v) for v in value] if isinstance(value, list) and len(value) == 2 and all(number(v) is not None for v in value) else None


def timestamp(value):
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.isoformat() if parsed.tzinfo is not None else None
    except (ValueError, TypeError):
        return None


def numeric_metrics(values, names):
    return {name: number(values.get(name)) for name in names}


def summary_metrics(pairs, field="metrics", names=METRICS):
    result = {}
    for name in names:
        scenes = defaultdict(list)
        values = []
        for pair in pairs:
            source = pair.get(field) or {}
            value = number(source.get(name))
            if value is not None:
                scenes[pair["scene_id"]].append(value)
                values.append(value)
        result[name] = ({**bootstrap_mean([float(np.mean(v)) for v in scenes.values()]),
                         "max": max(values), "pairs": len(values), "ci_unit": "scene"}
                        if values else {"mean": None, "max": None, "ci95": None, "units": 0, "pairs": 0, "ci_unit": "scene"})
    return result


class Evidence:
    def __init__(self, workspace):
        self.workspace = Path(workspace).resolve()
        self.errors = []
        self.links = []

    def path(self, value):
        path = Path(value)
        if path.is_absolute() and path.is_relative_to("/workspace"):
            path = self.workspace / path.relative_to("/workspace")
        elif not path.is_absolute():
            path = self.workspace / path
        path = path.resolve()
        if not path.is_relative_to(self.workspace):
            raise ValueError("Evidence escapes the declared workspace")
        return path

    def error(self, code, path):
        self.errors.append({"code": code, "artifact": identifier(Path(path).name)})

    def read(self, path):
        path = self.path(path)
        return json.loads(path.read_text())

    def hashes(self, directory):
        directory = self.path(directory)
        return {name: file_sha256(directory / name) for name in CORE}

    def check(self, value, expected):
        try:
            path = self.path(value)
            if file_sha256(path) != expected:
                raise ValueError("hash mismatch")
            return True
        except (OSError, ValueError, TypeError):
            self.error("missing_or_changed_artifact", value)
            return False

    def check_run(self, path, expected):
        try:
            if self.hashes(path) != expected:
                raise ValueError("hash mismatch")
            return True
        except (OSError, ValueError, TypeError):
            self.error("missing_or_changed_run", path)
            return False

    def run(self, directory):
        path = self.path(directory)
        return (self.read(path / "config.json"), self.read(path / "summary.json"),
                [json.loads(line) for line in (path / "measurements.jsonl").read_text().splitlines() if line.strip()])

    def link(self, kind, path, label):
        try:
            resolved = self.path(path)
            if resolved.is_file():
                self.links.append({"kind": kind, "label": str(label), "path": str(resolved)})
        except (OSError, ValueError, TypeError):
            self.error("invalid_local_link", path)


def review_status(evidence, reviews, ratings):
    """Ratings must be separate from the immutable exported blank CSV."""
    seen, complete, count = set(), 0, 0
    submitted = {str(evidence.path(row["index"])): row for row in (ratings or {}).get("reviews", [])}
    kind = (ratings or {}).get("reviewer_kind")
    if kind not in {"agent_assisted", "human"}:
        kind = None
    for review in reviews:
        index = evidence.path(review["index"])
        if index in seen:
            continue
        seen.add(index)
        key = evidence.read(review["private_key"])["cases"]
        wanted = {row["case_id"] for row in key}
        count += len(wanted)
        evidence.link("review", index, f"Blinded review {index.parent.name}")
        row = submitted.get(str(index))
        if not row or not kind:
            continue
        if (not evidence.check(index, row.get("index_sha256"))
                or not evidence.check(row["csv"], row.get("csv_sha256"))):
            continue
        path = evidence.path(row["csv"])
        if path == evidence.path(review["csv"]):
            evidence.error("ratings_must_be_separate_from_export", path)
            continue
        votes = list(csv.DictReader(path.read_text().splitlines()))
        if len(votes) != len(wanted) or {v.get("case_id") for v in votes} != wanted:
            evidence.error("review_case_mismatch", path)
            continue
        if all(v.get("acceptable_A", "").strip().lower() in {"yes", "no"}
               and v.get("acceptable_B", "").strip().lower() in {"yes", "no"} for v in votes):
            complete += len(wanted)
    return {"status": "complete" if count and complete == count else "pending", "cases": count,
            "rated_cases": complete, "reviewer_kind": kind,
            "note": "Ratings describe the predeclared blinded subset; they do not by themselves establish quality or trajectory adherence."}


def verify_phase(evidence, folder):
    receipt_path = folder / "aggregation-receipt.json"
    if not receipt_path.exists():
        return False
    valid = True
    receipt = evidence.read(receipt_path)
    for path, expected in receipt.get("artifact_sha256", {}).items():
        valid = evidence.check(path, expected) and valid
    for completed in sorted((folder / "shards").glob("*/complete.json")):
        shard = evidence.read(completed)
        for path, expected in shard.get("metric_artifacts", {}).items():
            valid = evidence.check(path, expected) and valid
            if str(path).endswith(".receipt.json"):
                try:
                    proof = evidence.read(path)
                    metric = evidence.read(str(path).replace(".receipt.json", ".json"))
                    valid = evidence.check(metric["tolerances_path"], proof["tolerances_sha256"]) and valid
                    valid = (valid and metric["evaluator_source_sha256"] == proof["evaluator_source_sha256"]
                             and metric["reference_artifact_sha256"] == proof["reference_artifact_sha256"]
                             and metric["runs"][0]["artifact_sha256"] == proof["candidate_artifact_sha256"])
                    for review_path, checksum in proof["review_artifact_sha256"].items():
                        valid = evidence.check(review_path, checksum) and valid
                except (OSError, KeyError, ValueError):
                    evidence.error("missing_metric_receipt", path)
                    valid = False
        for run in shard.get("runs", {}).values():
            valid = evidence.check_run(run["path"], run["artifact_sha256"]) and valid
    return valid and bool(receipt.get("artifact_sha256"))


def condition_labels(evidence, coverage, phase, metadata):
    base = phase.removesuffix("_actions")
    part = (coverage or {}).get("partitions", {}).get(base, {})
    expected = metadata.get("parent_manifest_sha256") if phase.endswith("_actions") else metadata.get("manifest_sha256")
    if not part or part.get("manifest_sha256") != expected:
        return {}, "unavailable_or_manifest_mismatch"
    return {row["request_id"]: row["labels"] for row in part["records"]}, "matched_offline_sidecar"


def quality_phase(evidence, study, phase, coverage, ratings):
    folder = study / "quality" / phase
    files = sorted((folder / "aggregate").glob("evaluation-vs-*.json"))
    if not files:
        return {"phase": phase, "status": "not_aggregated", "completed_shards": len(list((folder / "shards").glob("*/complete.json"))), "comparisons": []}
    verified = verify_phase(evidence, folder)
    comparisons, reviews = [], []
    for path in files:
        doc = evidence.read(path)
        valid_reference = evidence.check_run(doc["reference"], doc["reference_artifact_sha256"])
        reference_name = identifier(Path(doc["reference"]).name)
        reviews += doc.get("reviews", [doc["review"]] if doc.get("review") else [])
        evidence.link("quality_metrics", path, f"{phase}: quality vs {reference_name}")
        for entry in doc["runs"]:
            valid = verified and valid_reference and evidence.check_run(entry["run_path"], entry["artifact_sha256"])
            cfg, summary, rows = evidence.run(entry["run_path"])
            pairs = entry.get("pairs", [])
            pairkeys = [(p["request_id"], p.get("repeat", 0), p["candidate_id"], p["seed"]) for p in pairs]
            metadata = entry.get("manifest_metadata", {})
            count = cfg.get("benchmark", {}).get("candidate_count")
            covered = (len({p["request_id"] for p in pairs}) == metadata.get("clip_count")
                       and {p["scene_id"] for p in pairs} == set(metadata.get("scene_ids", []))
                       and type(count) is int and len(pairs) == count * metadata.get("clip_count", 0))
            complete = (covered and entry.get("complete_pairs") is True and len(pairs) == entry.get("matched_pairs")
                        and len(pairkeys) == len(set(pairkeys)) and not entry.get("errors"))
            labels, label_status = condition_labels(evidence, coverage, phase, entry.get("manifest_metadata", {}))
            by_condition = {identifier(label): summary_metrics([p for p in pairs if label in labels.get(p["request_id"], [])])
                            for label in sorted({label for p in pairs for label in labels.get(p["request_id"], [])})}
            action_rows = entry.get("action_response", [])
            actions = summary_metrics(action_rows, names=ACTION_METRICS)
            latencies = [number(row.get("latency_s")) for row in rows]
            latencies = [value for value in latencies if value is not None and value > 0]
            future_count = sum(len(row.get("candidates", [])) for row in rows)
            gate = entry.get("gate", {})
            verified_quality = bool(valid and complete and not entry.get("synthetic") and gate.get("quality_preserving"))
            provenance = doc.get("perceptual", {}).get("provenance", {})
            comparisons.append({"variant": identifier(entry["run"]), "reference": reference_name,
                "status": "verified" if valid and complete else "stale_or_incomplete", "synthetic": bool(entry.get("synthetic")),
                "pairs": len(pairs), "clips": len({p["request_id"] for p in pairs}), "scenes": len({p["scene_id"] for p in pairs}),
                "candidate_count": integer(cfg.get("benchmark", {}).get("candidate_count")), "steps": integer(cfg.get("generation", {}).get("steps")),
                "request_median_s": float(np.median(latencies)) if latencies else None,
                "request_p95_s": float(np.quantile(latencies, .95)) if len(latencies) >= 20 else None,
                "measured_requests": len(latencies), "measured_futures": future_count,
                "timing_scope": "quality generation; exploratory, not headline timing",
                "workload": identifier(cfg.get("benchmark", {}).get("workload", "unknown")),
                "raw_request_timings": [{"request_id": identifier(row["request_id"]), "scene_id": identifier(row["scene_id"]),
                    "repeat": integer(row.get("repeat", 0)), "latency_s": number(row.get("latency_s"))} for row in rows],
                "metrics": summary_metrics(pairs), "recorded_ground_truth": summary_metrics(pairs, field="recorded_ground_truth"),
                "condition_status": label_status, "by_condition": by_condition,
                "action_response": {"pairs": len(action_rows), "same_seed_control": bool(action_rows), "metrics": actions,
                                    "diagnostic_only": True},
                "quality_preserving": verified_quality, "within_frozen_metrics": bool(valid and complete and gate.get("within_frozen_metric_tolerances")),
                "metric_gates": {identifier(name): {"actual": number(value.get("actual")), "maximum": number(value.get("maximum")),
                                                   "passed": value.get("passed") is True} for name, value in gate.get("checks", {}).items()},
                "lpips_status": identifier(doc.get("perceptual", {}).get("status", "unknown")),
                "lpips_state_sha256": sha(provenance.get("state_dict_sha256")),
                "artifact_sha256": {name: sha(value) for name, value in entry["artifact_sha256"].items() if name in CORE},
                "raw_pairs": [{"request_id": identifier(p["request_id"]), "scene_id": identifier(p["scene_id"]),
                    "candidate_id": identifier(p["candidate_id"]), "repeat": integer(p.get("repeat", 0)), "seed": integer(p["seed"]),
                    "action_kind": identifier(p.get("action_kind", "unknown")), "trajectory_sha256": sha(p.get("trajectory_sha256")),
                    "reference_output_sha256": sha(p.get("reference_output_sha256")), "candidate_output_sha256": sha(p.get("candidate_output_sha256")),
                    "metrics": numeric_metrics(p.get("metrics", {}), METRICS),
                    "recorded_ground_truth": numeric_metrics(p.get("recorded_ground_truth") or {}, METRICS)} for p in pairs],
                "raw_action_response": [{"request_id": identifier(p["request_id"]), "scene_id": identifier(p["scene_id"]),
                    "seed": integer(p["seed"]), "recorded_candidate_id": identifier(p["recorded_candidate_id"]),
                    "alternative_candidate_id": identifier(p["alternative_candidate_id"]),
                    "trajectory_sha256": [sha(v) for v in p.get("trajectory_sha256", [])],
                    "output_sha256": [sha(v) for v in p.get("output_sha256", [])],
                    "metrics": numeric_metrics(p["metrics"], ACTION_METRICS)} for p in action_rows]})
    try:
        reviewed = review_status(evidence, reviews, ratings)
    except (OSError, KeyError, ValueError, TypeError):
        reviewed = {"status": "missing_review_artifacts", "cases": None, "rated_cases": 0, "reviewer_kind": None}
        evidence.error("missing_review_artifacts", folder)
    return {"phase": phase, "status": "verified" if verified and comparisons and all(c["status"] == "verified" for c in comparisons) else "stale_or_incomplete",
            "review": reviewed, "comparisons": comparisons}


def timing_phase(evidence, study, *, folder=None):
    folder = folder or study / "timing"
    runs, comparisons = [], []
    for path in sorted((folder / "aggregate").glob("*/summary.json")):
        directory = path.parent
        cfg, summary, rows = evidence.run(directory)
        valid = (summary.get("headline_eligible") is True and summary.get("backend") in {"vista", "upstream"}
                 and not summary.get("profiled") and len(rows) >= 200 and all(r.get("status") == "ok" for r in rows))
        blocks, raw, reconstructed, block_paths, block_vectors, source_ids = [], [], [], [], [], []
        runtime = reference_identity(evidence.read(directory / "environment.json"))
        valid = valid and bool((runtime.get("gpu") or {}).get("uuid")) and bool(runtime.get("torch"))
        for block in summary.get("timing_blocks", []):
            current = evidence.check_run(block["path"], block["artifact_sha256"])
            bp = evidence.path(block["path"])
            bc, bs, br = evidence.run(bp)
            cold = evidence.read(bp / "cold_start.json")
            numerical_gate = evidence.read(bp / "numerical_gate.json")
            environment_gate = evidence.read(bp / "environment_gate.json")
            gates = [numerical_gate.get("passed") is True, environment_gate.get("passed") is True]
            try:
                numerical = evidence.read(numerical_gate["report_path"])
                gates.append(evidence.check(numerical_gate["report_path"], numerical_gate["report_sha256"]))
                gates.append(numerical.get("passed") is True and numerical.get("runtime_identity") == runtime)
                source_ids.append(numerical.get("implementation_sha256"))
                gates.append(evidence.check(bc["environment_verification"], environment_gate["verification_sha256"]))
            except (OSError, KeyError, ValueError, TypeError):
                gates.append(False)
                evidence.error("missing_current_timing_gate_provenance", bp)
            warm = cold.get("warmups", [])
            gates.append(cold.get("status") == "ready" and 3 <= len(warm) <= 5
                         and all(r.get("status") == "ok" for r in warm)
                         and len({r.get("compiler", {}).get("stats", {}).get("unique_graphs", 0) for r in warm[-3:]}) == 1)
            current = (current and all(gates) and not bs.get("failures") and not bs.get("oom_attempts")
                       and not bs.get("recompiled_during_measurement") and not bs.get("profiled")
                       and reference_identity(evidence.read(bp / "environment.json")) == runtime
                       and bc.get("generation") == cfg.get("generation") and bc.get("optimizations") == cfg.get("optimizations")
                       and bc.get("model") == cfg.get("model") and bc.get("data") == cfg.get("data")
                       and all(bc["benchmark"].get(k) == cfg["benchmark"].get(k) for k in ("workload", "candidate_count", "input_condition"))
                       and len(br) == 50 and all(r.get("status") == "ok" and number(r.get("latency_s")) is not None
                                               and r["latency_s"] > 0 and not r.get("cache_stats", {}).get("oom_attempts") for r in br))
            repetitions = max(r.get("repeat", 0) for r in br) + 1
            for original in br:
                row = dict(original)
                row.update(request_index=len(reconstructed), repeat=len(blocks) * repetitions + original.get("repeat", 0),
                           timing_block=len(blocks), original_request_index=original.get("request_index"))
                reconstructed.append(row)
            block_paths.append(str(bp))
            block_vectors.append(hashlib.sha256(json.dumps(br, sort_keys=True).encode()).hexdigest())
            valid = valid and current
            process = evidence.read(bp.parent / "process.json") if (bp.parent / "process.json").exists() else {}
            warmups = [number(r.get("latency_s")) for r in cold.get("warmups", [])]
            blocks.append({"block": integer(block["block"]), "verified": bool(current), "benchmark_started_at": timestamp(bc.get("started_at")),
                "process_finished_at": timestamp(process.get("finished_at")), "model_load_s": number(cold.get("model_load_s")),
                "warmup_request_s": warmups, "measured_wall_s": number(bs.get("throughput_measurement_wall_s")),
                "median_s": number(block.get("median_s")), "requests": len(br),
                "artifact_sha256": {name: sha(value) for name, value in block["artifact_sha256"].items() if name in CORE}})
            for name in (*CORE, "cold_start.json", "environment_gate.json", "numerical_gate.json"):
                evidence.link("timing_evidence", bp / name, f"{directory.name} block {block['block']}: {name}")
        valid = (valid and reconstructed == rows and len(block_paths) == 4 and len(set(block_paths)) == 4
                 and len(set(block_vectors)) == 4 and len(source_ids) == 4 and len(set(source_ids)) == 1 and bool(sha(source_ids[0])))
        for row in rows:
            raw.append({"request_id": identifier(row["request_id"]), "scene_id": identifier(row["scene_id"]),
                        "repeat": integer(row.get("repeat", 0)), "block": integer(row.get("timing_block")), "latency_s": number(row.get("latency_s")),
                        "candidate_count": integer(row.get("candidate_count")), "actual_batch_size": integer(row.get("batch_size")),
                        "denoiser_evaluations": integer(row.get("denoiser_evaluations")),
                        "peak_allocated_bytes": number(row.get("peak_allocated_bytes")), "peak_reserved_bytes": number(row.get("peak_reserved_bytes")),
                        "stage_timings_s": {identifier(k): number(v) for k, v in row.get("stage_timings_s", {}).items()},
                        "candidates": [{"candidate_id": identifier(c["candidate_id"]), "seed": integer(c["seed"]),
                                        "trajectory_sha256": sha(c.get("trajectory_sha256"))} for c in row.get("candidates", [])]})
        environment = evidence.read(directory / "environment.json")
        gpu_name = str((environment.get("gpu") or {}).get("name", "unavailable"))
        if gpu_name not in {"NVIDIA H100 80GB HBM3", "NVIDIA H100 PCIe", "NVIDIA H100 NVL", "NVIDIA H100 SXM", "unavailable"}:
            gpu_name = "redacted"
        runs.append({"variant": identifier(directory.name), "status": "verified" if valid and len(blocks) >= 4 else "ineligible_or_incomplete",
            "headline_timing_eligible": bool(valid and len(blocks) >= 4), "synthetic": summary.get("backend") not in {"vista", "upstream"},
            "gpu_model": gpu_name[:100], "candidate_count": integer(cfg.get("benchmark", {}).get("candidate_count")),
            "workload": identifier(cfg.get("benchmark", {}).get("workload", "unknown")),
            "requests": len(rows), "unique_clips": len({r["request_id"] for r in rows}), "unique_scenes": len({r["scene_id"] for r in rows}),
            "p50_s": number(summary.get("latency_p50_s")), "p95_s": number(summary.get("latency_p95_s")),
            "p50_ci95_s": interval(summary.get("latency_p50_ci95_s")), "p95_ci95_s": interval(summary.get("latency_p95_ci95_s")),
            "futures_per_second": number(summary.get("throughput_futures_s")), "blocks": blocks, "raw_measurements": raw,
            "artifact_sha256": evidence.hashes(directory)})
    file = folder / "comparisons.json"
    if file.exists():
        for row in evidence.read(file):
            valid = evidence.check_run(row["reference"], row["reference_artifact_sha256"])
            valid = evidence.check_run(row["candidate"], row["candidate_artifact_sha256"]) and valid
            comparisons.append({"reference": identifier(Path(row["reference"]).name), "variant": identifier(Path(row["candidate"]).name),
                "status": "verified" if valid else "stale", "paired_requests": integer(row.get("paired_requests")),
                "independent_scenes": integer(row.get("independent_scenes")),
                "metrics": {name: {k: interval(value.get(k)) if k.endswith("ci95") or k == "ci95" else number(value.get(k)) for k in ("relative_latency_change", "ci95", "speedup", "speedup_ci95")}
                            for name, value in row.get("metrics", {}).items() if name in {"p50", "p95"}},
                "headline_combined_claim": False,
                "note": "Combine verified timing, frozen quality and development variability evidence before any headline claim."})
        evidence.link("timing_comparison", file, "Paired scene-cluster timing comparisons")
    return {"status": "complete" if len(runs) >= 2 and all(r["headline_timing_eligible"] for r in runs) else "pending_or_ineligible",
            "runs": runs, "comparisons": comparisons}


def companion_helper():
    path = Path(__file__).with_name("run_timing_companion.py")
    spec = importlib.util.spec_from_file_location("report_verified_timing_companion", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def accepted_variant_reviews(evidence, entry, ratings, reference):
    """Bind actual yes votes to each immutable shard's blinded candidate panel."""
    if (ratings or {}).get("reviewer_kind") not in {"human", "agent_assisted"}:
        return False, 0, []
    submitted = {str(evidence.path(r["index"])): r for r in ratings.get("reviews", [])}
    if len(submitted) != len(ratings.get("reviews", [])):
        return False, 0, []
    count, hashes = 0, []
    for artifact in entry.get("metric_artifacts", []):
        if not evidence.check(artifact["path"], artifact["sha256"]):
            return False, count, hashes
        metric = evidence.read(artifact["path"])
        review = metric["review"]
        index = evidence.path(review["index"])
        rating = submitted.get(str(index))
        if (not rating or evidence.path(rating["csv"]) == evidence.path(review["csv"])
                or not evidence.check(index, rating["index_sha256"])
                or not evidence.check(rating["csv"], rating["csv_sha256"])):
            return False, count, hashes
        cases = evidence.read(review["private_key"])["cases"]
        votes = list(csv.DictReader(evidence.path(rating["csv"]).read_text().splitlines()))
        indexed = {r.get("case_id"): r for r in votes}
        wanted = {r["case_id"] for r in cases}
        if len(indexed) != len(votes) or len(wanted) != len(cases) or set(indexed) != wanted:
            return False, count, hashes
        selected = [r for r in cases if r.get("run") == entry["run"] and r.get("reference_run") == reference]
        if not selected:
            return False, count, hashes
        for case in selected:
            if {case.get("A"), case.get("B")} != {entry["run"], reference}:
                return False, count, hashes
            panel = "A" if case["A"] == entry["run"] else "B"
            if indexed[case["case_id"]].get(f"acceptable_{panel}", "").strip().lower() != "yes":
                return False, count, hashes
        count += len(selected)
        hashes.extend([rating["index_sha256"], rating["csv_sha256"], file_sha256(evidence.path(review["private_key"]))])
    return count > 0, count, hashes


def quality_implementation_matches(evidence, run_path, config, timing_config, parent_plan, phase):
    """Bind every quality shard to this implementation and immutable partition."""
    from drivebench.study import quality_runtime_identity
    expected_sources = parent_plan.get("generation_sources_at_plan")
    manifest = parent_plan["manifests"][phase]
    if (not expected_sources or config["data"].get("manifest_sha256") != manifest["sha256"]
            or config["data"].get("manifest_metadata") != manifest["metadata"]):
        return False
    timing_proof = evidence.read(timing_config["environment_verification"])
    expected_runtime = quality_runtime_identity(timing_proof["environment"])
    if timing_proof.get("passed") is not True:
        return False
    sources = config.get("study_aggregation", {}).get("source_runs", [])
    if not sources or len({row["path"] for row in sources}) != len(sources):
        return False
    _, _, aggregate_rows = evidence.run(run_path)
    source_requests = []
    for receipt in sources:
        if not evidence.check_run(receipt["path"], receipt["artifact_sha256"]):
            return False
        source_config, _, rows = evidence.run(receipt["path"])
        source_requests.extend(row["request_id"] for row in rows)
        if any(source_config.get(key) != timing_config.get(key) for key in ("model", "generation", "optimizations")):
            return False
        data = source_config["data"]
        if (data.get("manifest_sha256") != manifest["sha256"]
                and data.get("manifest_metadata", {}).get("parent_manifest_sha256") != manifest["sha256"]):
            return False
        provenance = source_config.get("study_provenance", {})
        if receipt.get("adopted"):
            source = receipt.get("generation_source_evidence", {})
            if not evidence.check(source["path"], source["sha256"]):
                return False
            proof = evidence.read(source["path"])
            if (proof.get("source_unchanged") is not True or proof.get("exit_code") != 0
                    or not proof.get("finished_at")
                    or any(proof.get("source_files", {}).get(k) != v for k, v in expected_sources.items())):
                return False
        elif provenance.get("generation_sources") != expected_sources:
            return False
        environment = evidence.read(evidence.path(receipt["path"]) / "environment.json")
        if quality_runtime_identity(environment) != expected_runtime:
            return False
    return (len(source_requests) == len(set(source_requests)) == len(aggregate_rows)
            and set(source_requests) == {row["request_id"] for row in aggregate_rows}
            and quality_runtime_identity(evidence.read(evidence.path(run_path) / "environment.json")) == expected_runtime)


def companion_quality(evidence, study, value, helper, quality, ratings):
    """Recompute frozen metric gates; never treat review completion as acceptance."""
    from drivebench.evaluate import aggregate_pairs, tolerance_gate
    from drivebench.evaluation.study_policy import policy_for_comparison
    parent_plan = evidence.read(study / "plan.json")
    selected = helper.study._selection(study, parent_plan)
    reference, candidate = value["baseline"], value["candidate"]
    results = []
    for phase, expected_clips in (("evaluation", 128), ("evaluation_second_seed", 16)):
        visible = next((p for p in quality if p["phase"] == phase), {})
        comparison = next((c for c in visible.get("comparisons", [])
                           if c["variant"] == identifier(candidate) and c["reference"] == identifier(reference)), {})
        result = {"phase": phase, "accepted": False, "status": "missing_or_unaccepted_quality", "rated_cases": 0}
        results.append(result)
        if (visible.get("status") != "verified" or comparison.get("status") != "verified"
                or comparison.get("synthetic") or comparison.get("clips") != expected_clips):
            continue
        document = evidence.read(study / "quality" / phase / "aggregate" / f"evaluation-vs-{reference}.json")
        entry = next(r for r in document["runs"] if r["run"] == candidate)
        cfg, _, _ = evidence.run(entry["run_path"])
        ref_cfg, _, _ = evidence.run(document["reference"])
        # Workload candidate counts may differ; implementation and generation may not.
        if any(quality_cfg.get(key) != value["configs"][name].get(key)
               for name, quality_cfg in ((reference, ref_cfg), (candidate, cfg))
               for key in ("model", "generation", "optimizations")):
            result["status"] = "quality_implementation_differs"
            continue
        if not all(quality_implementation_matches(evidence, path, quality_cfg, value["configs"][name], parent_plan, phase)
                   for name, path, quality_cfg in ((reference, document["reference"], ref_cfg), (candidate, entry["run_path"], cfg))):
            result["status"] = "quality_source_runtime_or_partition_differs"
            continue
        policy_path = policy_for_comparison(selected, reference, candidate)
        binding = entry.get("quality_policy") or document.get("tolerance_policies_by_run", {}).get(candidate, {})
        if (binding.get("reference") != reference or evidence.path(binding.get("path", ".")) != evidence.path(policy_path)
                or not evidence.check(policy_path, binding.get("sha256"))):
            result["status"] = "quality_policy_not_bound"
            continue
        policy = yaml.safe_load(evidence.path(policy_path).read_text())
        pairs = entry["pairs"]
        aggregates = aggregate_pairs(pairs)
        if aggregates != entry["aggregates"]:
            result["status"] = "quality_aggregate_differs_from_pairs"
            continue
        accepted, rated, review_hashes = accepted_variant_reviews(evidence, entry, ratings, reference)
        gate = tolerance_gate(aggregates, policy, complete_pairs=entry.get("complete_pairs") is True,
            synthetic=entry.get("synthetic") is not False,
            evaluation_manifest_sha256=cfg["data"]["manifest_sha256"],
            evaluation_manifest_metadata=entry["manifest_metadata"], run_started_at=cfg["started_at"],
            reviewed=accepted, evaluated_scene_ids=sorted({p["scene_id"] for p in pairs}),
            evaluated_request_ids=sorted({p["request_id"] for p in pairs}), evaluated_pair_count=len(pairs))
        result.update(accepted=bool(accepted and gate["quality_preserving"]), rated_cases=rated,
            status="accepted" if accepted and gate["quality_preserving"] else "metrics_or_review_not_accepted",
            policy_sha256=file_sha256(evidence.path(policy_path)), review_artifact_sha256=[sha(h) for h in review_hashes],
            candidate_count=integer(cfg["benchmark"].get("candidate_count")),
            metric_acceptance=bool(gate["within_frozen_metric_tolerances"]))
        comparison.update(quality_preserving=result["accepted"], reviewed_acceptance_joined=accepted,
                          within_frozen_metrics=bool(gate["within_frozen_metric_tolerances"]))
    return {"accepted": all(r["accepted"] for r in results), "phases": results,
        "scope": "Frozen metric limits and accepted blinded subset on 128 recorded clips plus 16 second-seed clips. Does not establish all-candidate quality, trajectory adherence or driving safety."}


def timing_companion(evidence, study, name, path, quality, ratings):
    directory = evidence.path(path)
    result = {"name": identifier(name), "status": "unavailable_or_invalid", "runs": [], "comparisons": [],
              "headline_combined_claim": False, "quality": {"accepted": False, "status": "pending"}}
    try:
        helper = companion_helper()
        # Authoritative checks intentionally require the canonical /workspace tree.
        # Never monkeypatch core filesystem functions to reinterpret frozen evidence.
        _, value, parent = helper._load(directory)
        if Path(parent).resolve() != study:
            raise ValueError("Companion belongs to a different study")
        completion_path = directory / "completed.json"
        completion_hash = file_sha256(completion_path)
        completed = evidence.read(completion_path)
        names = [value["baseline"], value["candidate"]]
        if (completed.get("status") != "complete" or completed.get("quality_generated_here") is not False
                or completed.get("combined_headline_claim") is not False
                or completed.get("plan_sha256") != file_sha256(directory / "plan.json")
                or set(completed.get("aggregate_runs", {})) != set(names)
                or set(completed.get("artifact_sha256", {})) != set(names)
                or not evidence.check(directory / "comparison.json", completed.get("comparison_sha256"))):
            raise ValueError("Companion completion bindings differ")
        for variant in names:
            if (evidence.path(completed["aggregate_runs"][variant]) != directory / "aggregate" / variant
                    or not evidence.check_run(completed["aggregate_runs"][variant], completed["artifact_sha256"][variant])):
                raise ValueError("Companion aggregate binding differs")
        measured = timing_phase(evidence, study, folder=directory)
        if (measured["status"] != "complete" or len(measured["runs"]) != 2
                or any(r["requests"] != 200 or r["candidate_count"] != value["candidate_count"]
                       or r["workload"] != value["workload"] for r in measured["runs"])):
            raise ValueError("Companion source-block measurements are incomplete")
        for run in measured["runs"]:
            rows = run["raw_measurements"]
            values, scenes = [r["latency_s"] for r in rows], [r["scene_id"] for r in rows]
            p50, p95 = float(np.quantile(values, .5)), float(np.quantile(values, .95))
            if run["p50_s"] != p50 or run["p95_s"] != p95:
                raise ValueError("Companion summary quantiles differ from raw requests")
            run.update(p50_ci95_s=bootstrap_quantile(values, .5, groups=scenes),
                       p95_ci95_s=bootstrap_quantile(values, .95, groups=scenes),
                       uncertainty="Recomputed 2000 scene-cluster bootstrap resamples; repeated blocks stay within scene")
        comparison = evidence.read(directory / "comparison.json")
        # Recheck pilot provenance, chronology, every raw block and full same-GPU
        # identity; recompute the saved paired confidence intervals without videos.
        actual = helper.pilot.compare_pilot(value["pilot_policy"],
            completed["aggregate_runs"][names[0]], completed["aggregate_runs"][names[1]])
        if actual != comparison or completed.get("performance_status") != actual["status"]:
            raise ValueError("Companion saved comparison differs from verified evidence")
        if file_sha256(completion_path) != completion_hash:
            raise ValueError("Companion completion changed during reporting")
        result.update(status="verified", runs=measured["runs"], candidate_count=integer(value["candidate_count"]),
            workload=identifier(value["workload"]), performance_status=identifier(actual["status"]),
            finished_at=timestamp(completed.get("finished_at")),
            provenance={"plan_sha256": file_sha256(directory / "plan.json"), "completed_sha256": completion_hash,
                "comparison_sha256": file_sha256(directory / "comparison.json"), "helper_sha256": sha(value["helper_sha256"]),
                "parent_plan_sha256": sha(value["parent_plan_sha256"]), "selection_sha256": sha(value["selection_sha256"]),
                "pilot_policy_sha256": sha(value["pilot_policy_sha256"]), "source_identity": sha(value["source_identity"]),
                "quality_artifact_sha256": sorted(sha(v) for v in value["quality_evidence_sha256"].values()),
                "numerical_proof_sha256": sorted(sha(v) for v in value["numerical_proof_sha256"].values()),
                "runtime_overlay_present": value.get("timing_runtime_overlay") is not None,
                "runtime_overlay_artifact_sha256": sorted(sha(v) for v in (value.get("timing_runtime_overlay") or {}).get("bound_files_sha256", {}).values())},
            comparisons=[{"reference": identifier(names[0]), "variant": identifier(names[1]),
                "paired_requests": integer(actual.get("paired_requests")), "independent_scenes": integer(actual.get("independent_scenes")),
                "pilot_verified": actual.get("pilot_policy_verified") is True,
                "metrics": {metric: {"relative_latency_change": number(row.get("relative_latency_change")),
                    "ci95": interval(row.get("ci95")), "tolerance": number(row.get("tolerance")),
                    "exceeds_tolerance_and_uncertainty": row.get("exceeds_tolerance_and_uncertainty") is True}
                    for metric, row in actual.get("metrics", {}).items() if metric in {"p50", "p95"}},
                "interpretation": "Pilot pass means no detected regression beyond calibrated variability; it does not establish improvement."}])
        try:
            result["quality"] = companion_quality(evidence, study, value, helper, quality, ratings)
        except (OSError, ValueError, KeyError, TypeError, StopIteration):
            evidence.error("companion_quality_join_unavailable", name)
        result["headline_combined_claim"] = bool(result["quality"].get("accepted") and actual.get("status") == "pass"
            and actual.get("pilot_policy_verified") is True and set(actual.get("metrics", {})) == {"p50", "p95"}
            and all(interval(m.get("ci95")) is not None and m["ci95"][1] < 0 for m in actual["metrics"].values()))
        for filename in ("plan.json", "plan-receipt.json", "completed.json", "comparison.json"):
            evidence.link("timing_companion", directory / filename, f"{identifier(name)}: {filename}")
    except (OSError, ValueError, KeyError, TypeError, StopIteration, RuntimeError):
        evidence.error("companion_authoritative_validation_failed", name)
    return result


def runtime_matches(phases):
    dev = next((p for p in phases if p["phase"] == "development"), {})
    by_reference = defaultdict(dict)
    for row in dev.get("comparisons", []):
        if row.get("status") == "verified" and row.get("request_median_s"):
            by_reference[row["reference"]][row["variant"]] = row
    result, uncertainty = [], {}
    for reference, values in by_reference.items():
        reduced = {name: row for name, row in values.items() if name.startswith("steps-") and 0 < (row.get("steps") or 0) < 50}
        for name, row in values.items():
            if not name.startswith("feature-") or not reduced:
                continue
            target = min(reduced.values(), key=lambda x: abs(math.log(row["request_median_s"] / x["request_median_s"])))
            cache_key = (name, target["variant"])
            if cache_key not in uncertainty:
                def paired(values):
                    return {(v["request_id"], v["scene_id"], v["repeat"]): v["latency_s"] for v in values}
                left, right = paired(target["raw_request_timings"]), paired(row["raw_request_timings"])
                ci, count, scenes = None, 0, 0
                if (left.keys() == right.keys() and len(left) == len(target["raw_request_timings"])
                        and len(right) == len(row["raw_request_timings"])
                        and row["candidate_count"] == target["candidate_count"] and row["workload"] == target["workload"]
                        and len({key[1] for key in left}) >= 2
                        and all(v is not None and v > 0 for v in [*left.values(), *right.values()])):
                    keys = sorted(left)
                    _, ci = _bootstrap_ratios(np.asarray([left[k] for k in keys]), np.asarray([right[k] for k in keys]),
                                               scene_groups=[k[1] for k in keys], quantile=.5, resamples=2000)
                    count = len(keys)
                    scenes = len({key[1] for key in keys})
                uncertainty[cache_key] = (ci, count, scenes)
            ci, count, scenes = uncertainty[cache_key]
            ratio_ci = [1 + value for value in ci] if ci else None
            within_margin = bool(ratio_ci and ratio_ci[0] >= .98 and ratio_ci[1] <= 1.02)
            matched = bool(within_margin and scenes >= 32 and not row["synthetic"] and not target["synthetic"])
            result.append({"reference": reference, "feature_variant": name, "step_variant": target["variant"],
                "feature_median_s": row["request_median_s"], "step_median_s": target["request_median_s"],
                "relative_latency_gap": row["request_median_s"] / target["request_median_s"] - 1,
                "paired_scene_ci95_relative_gap": ci, "paired_requests": count,
                "paired_scene_ci95_latency_ratio": ratio_ci, "independent_scenes": scenes,
                "gap_resolved_at_95_percent": bool(ci and (ci[0] > 0 or ci[1] < 0)),
                "runtime_matching_criterion_sha256": RUNTIME_MATCHING_CRITERION_SHA256,
                "within_predeclared_runtime_margin": within_margin,
                "runtime_matching_status": "matched_runtime" if matched else "nearest_measured_control",
                "runtime_matched": matched,
                "steps": target["steps"], "equal_runtime_established": False,
                "note": "Runtime matched requires all 32 real development scenes and the entire paired 95% ratio interval inside[0.98,1.02]; exact equality and generation quality are not established."})
    return result


def render(data, output, links=(), *, public=False):
    esc = html.escape
    def fmt(v):
        return "unavailable" if v is None else ("yes" if v else "no") if isinstance(v, bool) else f"{v:.6g}" if isinstance(v, (float, int)) else esc(str(v))
    def table(headers, rows):
        return '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{esc(h)}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{fmt(c)}</td>' for c in row) + '</tr>' for row in rows) + '</tbody></table></div>'
    body = [f'<h1>Driving inference study</h1><p class="status">{esc(data["status"])}</p>',
            '<p>Measured results only. Missing or stale evidence stays visible. Numerical and image diagnostics do not establish driving safety or trajectory adherence.</p>',
            '<p><a href="results.json">Download sanitized paired metrics and raw request timings</a></p>']
    if any(r.get("synthetic") for phase in data["quality"] for r in phase["comparisons"]) or any(r.get("synthetic") for r in data["timing"]["runs"]):
        body.append('<p class="status"><strong>SYNTHETIC FUNCTIONAL FIXTURE DATA PRESENT.</strong> These rows and orange plot points do not measure Vista or GPU performance.</p>')
    if public:
        body.append('<p>This public projection omits private paths, accounts, GPU UUIDs, model weights, image/video files and freeform reviewer notes.</p>')
    for phase in data["quality"]:
        body.append(f'<h2>{esc(phase["phase"])}</h2><p>Status: {esc(phase["status"])}; review: {esc(phase.get("review", {}).get("status", "pending"))}</p>')
        body.append(table(["Variant", "Reference", "Pairs / clips / scenes", "Steps", "Median s (exploratory)", "LPIPS", "RMSE", "Temporal derivative MAE", "Quality gate"],
            [(r["variant"], r["reference"], f'{r["pairs"]} / {r["clips"]} / {r["scenes"]}', r["steps"], r["request_median_s"], r["metrics"]["lpips"]["mean"],
              r["metrics"]["rmse"]["mean"], r["metrics"]["temporal_derivative_mae"]["mean"], "passed" if r["quality_preserving"] else r["status"] + "; not established") for r in phase["comparisons"]]))
        for r in phase["comparisons"]:
            if r["action_response"]["pairs"]:
                a = r["action_response"]
                body.append(f'<p>{esc(r["variant"])} vs {esc(r["reference"])}: {a["pairs"]} same-seed action-response pairs; response-error RMSE {fmt(a["metrics"]["response_error_rmse"]["mean"])}. Diagnostic only.</p>')
        for name in data.get("plots", {}).get("files", []):
            if name.startswith(phase["phase"] + "--"):
                body.append(f'<img class="plot" alt="Measured performance quality curve" src="plots/{esc(name, quote=True)}">')
    body.append('<h2>Reduced-step runtime controls</h2>')
    body.append(table(["Reference", "Feature", "Reduced-step control", "Feature s", "Control s", "Relative gap", "Paired scene95% ratio CI", "Runtime criterion"],
                      [(r["reference"], r["feature_variant"], r["step_variant"], r["feature_median_s"], r["step_median_s"], r["relative_latency_gap"],
                        r["paired_scene_ci95_latency_ratio"], r["runtime_matching_status"]) for r in data["runtime_matches"]]))
    body.append('<p>The 50-step baseline is not a reduced-step control. Runtime matched means the entire paired scene95% ratio interval lies inside the predeclared[0.98,1.02] margin on all 32 real development scenes. This is separate from generation quality and does not mean exactly equal runtime. Criterion SHA256: ' + RUNTIME_MATCHING_CRITERION_SHA256 + '.</p>')
    body.append('<h2>Headline timing evidence</h2>')
    body.append(table(["Variant", "Requests", "Unique clips/scenes", "Candidates", "p50 s", "p95 s", "Eligible"],
        [(r["variant"], r["requests"], f'{r["unique_clips"]}/{r["unique_scenes"]}', r["candidate_count"], r["p50_s"], r["p95_s"], r["headline_timing_eligible"]) for r in data["timing"]["runs"]]))
    for r in data["timing"]["runs"]:
        body.append(f'<h3>{esc(r["variant"])} cold starts and blocks</h3>')
        body.append(table(["Block", "Benchmark started", "Process finished", "Model load s", "Warmup request seconds", "Measured wall s", "Median s"],
            [(b["block"], b["benchmark_started_at"], b["process_finished_at"], b["model_load_s"], ", ".join(fmt(v) for v in b["warmup_request_s"]), b["measured_wall_s"], b["median_s"]) for b in r["blocks"]]))
    for companion in data.get("timing_companions", []):
        body.append(f'<h2>Timing companion: {esc(companion["name"])}</h2><p>Status: {esc(companion["status"])}; '
                    f'quality acceptance: {fmt(companion["quality"].get("accepted"))}; combined claim: {fmt(companion["headline_combined_claim"])}.</p>')
        body.append(table(["Variant", "Workload", "Candidates", "Requests", "Unique clips/scenes", "p50 s", "p95 s", "Timing eligible"],
            [(r["variant"], r["workload"], r["candidate_count"], r["requests"], f'{r["unique_clips"]}/{r["unique_scenes"]}',
              r["p50_s"], r["p95_s"], r["headline_timing_eligible"]) for r in companion["runs"]]))
        for comparison in companion["comparisons"]:
            body.append(table(["Metric", "Relative latency change", "Paired scene 95% CI", "Pilot tolerance", "Exceeds tolerance and uncertainty"],
                [(name, row["relative_latency_change"], row["ci95"], row["tolerance"], row["exceeds_tolerance_and_uncertainty"])
                 for name, row in comparison["metrics"].items()]))
        for r in companion["runs"]:
            body.append(f'<h3>{esc(companion["name"])} / {esc(r["variant"])} cold starts and blocks</h3>')
            body.append(table(["Block", "Benchmark started", "Process finished", "Model load s", "Warmup request seconds", "Measured wall s", "Median s"],
                [(b["block"], b["benchmark_started_at"], b["process_finished_at"], b["model_load_s"], ", ".join(fmt(v) for v in b["warmup_request_s"]),
                  b["measured_wall_s"], b["median_s"]) for b in r["blocks"]]))
    body.append('<p>Scene-cluster paired p50/p95 uncertainty and every raw request are in results.json. Four 50-request blocks repeat the first 50 held-out clips; 200 requests do not imply200 distinct scenes. Combined headline claims remain false until separate quality and variability-policy gates are joined.</p>')
    body.append('<h2>Condition breakdowns and interpretation</h2><p>' + esc(COVERAGE_NOTE) + '</p><p>Per-condition scene-cluster intervals, recorded-ground-truth errors, and second-seed results are in results.json. Sparse groups are descriptive and their sample counts are explicit.</p>')
    body.append('<h2>Evidence status</h2>' + table(["Issue", "Artifact"], [(e["code"], e["artifact"]) for e in data["evidence_issues"]]))
    if links:
        body.append('<h2>Local raw evidence and blinded reviews</h2><ul>')
        for link in links:
            target = os.path.relpath(link["path"], Path(output).parent)
            body.append(f'<li><a href="{esc(target, quote=True)}">{esc(link["label"])}</a></li>')
        body.append('</ul>')
    text = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Driving inference study</title><style>body{font:15px/1.55 system-ui;margin:32px auto;padding:0 24px;max-width:1500px;background:#0b1220;color:#e2e8f0}h1{font-size:32px}h2{margin-top:36px}.status{padding:14px;background:#223047}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;border-bottom:1px solid #334155;text-align:left}a{color:#7dd3fc}.plot{max-width:850px;width:100%;background:white}</style><body>' + ''.join(body) + '</body></html>'
    Path(output).write_text(text)


def plots(data, destination):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {"status": "matplotlib_unavailable", "files": [], "note": "Complete numeric curves remain in results.json; no environment packages were installed."}
    destination.mkdir(parents=True, exist_ok=True)
    files = []
    for phase in data["quality"]:
        refs = sorted({r["reference"] for r in phase["comparisons"]})
        for reference in refs:
            for metric in ("lpips", "rmse"):
                points = [r for r in phase["comparisons"] if r["reference"] == reference and r["status"] == "verified"
                          and r["request_median_s"] is not None and r["metrics"][metric]["mean"] is not None]
                if not points:
                    continue
                fig, ax = plt.subplots(figsize=(9.3, 4.8), layout="constrained")
                markers = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "*", "8")
                for index, r in enumerate(points):
                    x, y, ci = r["request_median_s"], r["metrics"][metric]["mean"], r["metrics"][metric]["ci95"]
                    ax.scatter([x], [y], color="#b45309" if r["synthetic"] else "#047857" if r["quality_preserving"] else "#2563eb",
                               s=55, marker=markers[index % len(markers)], label=r["variant"], zorder=3)
                    if ci:
                        ax.errorbar(x, y, yerr=[[max(0., y-ci[0])], [max(0., ci[1]-y)]], color="#64748b", capsize=3, fmt="none")
                ax.set(xlabel="Median request latency (s; exploratory quality run)", ylabel=f"{metric.upper()} from reference (lower is closer)",
                       title=f'{phase["phase"]} — reference: {reference}')
                ax.grid(alpha=.2)
                ax.margins(x=.12, y=.12)
                ax.legend(title="Variant", loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8, frameon=False)
                name = f'{phase["phase"]}--{reference}--{metric}.svg'
                fig.savefig(destination / name, metadata={"Creator": "report_full_study", "Date": None})
                plt.close(fig)
                files.append(name)
    return {"status": "rendered", "files": files, "note": "Vertical intervals bootstrap scene means; timing points are exploratory quality runs."}


def build(study, output, *, workspace_root=None, coverage_path=None, review_ratings=None, replace=False, make_plots=True, timing_companions=None):
    workspace = Path(workspace_root or Path.cwd()).resolve()
    evidence = Evidence(workspace)
    study, output = evidence.path(study), Path(output).resolve()
    if output.exists() and not replace:
        raise FileExistsError("Choose a fresh report directory or explicitly pass --replace")
    output.mkdir(parents=True, exist_ok=True)
    plan = evidence.read(study / "plan.json")
    coverage = evidence.read(coverage_path) if coverage_path else None
    ratings = evidence.read(review_ratings) if review_ratings else None
    quality = []
    for phase in PHASES:
        try:
            quality.append(quality_phase(evidence, study, phase, coverage, ratings))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            evidence.error("unreadable_phase_evidence", phase)
            quality.append({"phase": phase, "status": "unreadable_or_incomplete", "comparisons": []})
    try:
        timing = timing_phase(evidence, study)
    except (OSError, ValueError, KeyError, TypeError):
        evidence.error("unreadable_timing_evidence", "timing")
        timing = {"status": "unreadable_or_incomplete", "runs": [], "comparisons": []}
    companions = [timing_companion(evidence, study, name, path, quality, ratings)
                  for name, path in (timing_companions or {}).items()]
    model = plan.get("base_config", {}).get("model", {})
    data = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "incomplete: see individual stages and gates", "study_plan_sha256": file_sha256(study / "plan.json"),
            "source_identity": {"model_commit": sha(model.get("upstream_commit")), "checkpoint_sha256": sha(model.get("checkpoint_sha256")),
                                "generation_files": {identifier(k): sha(v) for k, v in plan.get("generation_sources_at_plan", {}).items()}},
            "quality": quality, "timing": timing, "timing_companions": companions, "runtime_matches": runtime_matches(quality),
            "runtime_matching_criterion": RUNTIME_MATCHING_CRITERION,
            "runtime_matching_criterion_sha256": RUNTIME_MATCHING_CRITERION_SHA256,
            "coverage": {"status": "provided" if coverage else "not_provided", "sha256": file_sha256(evidence.path(coverage_path)) if coverage_path else None,
                         "interpretation": COVERAGE_NOTE},
            "raw_output_policy": {"unselected_arrays_retired_after_metric_receipts": True,
                                  "retained_float_clips_per_phase": integer(plan.get("quality", {}).get("retain_raw_clips")),
                                  "limitation": "Output hashes cannot recreate deleted videos; new metrics require regeneration.",
                                  "public_driving_images_included": False, "public_weights_included": False},
            "evidence_issues": evidence.errors, "headline_claims": [{"timing_companion": c["name"],
                "candidate_count": c["candidate_count"], "workload": c["workload"], "quality_scope": c["quality"]["scope"],
                "comparison": c["comparisons"][0], "provenance": c["provenance"]}
                for c in companions if c["headline_combined_claim"]],
            "public_projection": {"allowlist": True, "absolute_paths": False, "accounts": False, "gpu_uuids": False,
                                  "raw_configs": False, "commands_or_environment": False, "weights": False, "driving_images": False,
                                  "freeform_review_notes": False}}
    timing_complete = timing["status"] == "complete" or (companions and all(c["status"] == "verified" for c in companions)
        and {c["candidate_count"] for c in companions} >= {1, 4})
    if all(p["status"] == "verified" and p.get("review", {}).get("status") == "complete" for p in quality) and timing_complete and not evidence.errors:
        data["status"] = "measurements and planned reviews complete; combined claim gates remain separate"
    public = output / "public"
    public.mkdir(exist_ok=True)
    prior_manifest = public / "projection-manifest.json"
    prior = json.loads(prior_manifest.read_text()).get("files", {}) if prior_manifest.exists() else {}
    existing = {str(p.relative_to(public)) for p in public.rglob("*") if p.is_file() and p != prior_manifest}
    if existing - set(prior):
        raise ValueError("Public destination contains files not owned by the prior projection manifest")
    for name, record in prior.items():
        path = (public / name).resolve()
        if not path.is_relative_to(public) or (path.exists() and file_sha256(path) != record["sha256"]):
            raise ValueError("Previous public projection changed; preserve it and choose a fresh destination")
        if path.is_file() and name.startswith("plots/"):
            path.unlink()
    data["plots"] = plots(data, public / "plots") if make_plots else {"status": "disabled", "files": []}
    # The same sanitized numerical content appears locally and publicly. Only the
    # local index adds links to private artifacts; no raw evidence is copied.
    encoded = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    (output / "results.json").write_text(encoded)
    (public / "results.json").write_text(encoded)
    render(data, public / "index.html", public=True)
    local_data = json.loads(encoded)
    local_data["plots"]["files"] = []  # Public plot copies are linked separately below.
    evidence.links.append({"kind": "public", "label": "Sanitized public report with curves", "path": str(public / "index.html")})
    render(local_data, output / "index.html", evidence.links)
    produced = [public / "index.html", public / "results.json", *[public / "plots" / name for name in data["plots"]["files"]]]
    manifest = {str(p.relative_to(public)): {"bytes": p.stat().st_size, "sha256": file_sha256(p)} for p in sorted(produced)}
    (public / "projection-manifest.json").write_text(json.dumps({"schema_version": 1, "files": manifest,
        "excluded_by_default": ["accounts", "private_paths", "GPU_UUIDs", "weights", "driving_images", "videos", "private_review_keys", "reviewer_notes"]}, indent=2) + "\n")
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workspace-root", default=str(Path.cwd()), help="Map recorded /workspace paths into this local mirror")
    parser.add_argument("--coverage", help="Offline reports/scene-coverage.json sidecar")
    parser.add_argument("--review-ratings", help="JSON binding separate rated CSVs to exported review indexes and their SHA256s")
    parser.add_argument("--timing-companion", action="append", default=[], metavar="NAME=PATH",
                        help="Verified standalone timing companion; repeat for single/candidate4. Canonical evidence paths must be readable.")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    companions = {}
    for item in args.timing_companion:
        name, separator, path = item.partition("=")
        if not separator or not name or not path or name in companions or identifier(name) != name:
            parser.error("Use a unique nonprivate --timing-companion NAME=PATH label")
        companions[name] = path
    result = build(args.study, args.output, workspace_root=args.workspace_root, coverage_path=args.coverage,
                   review_ratings=args.review_ratings, replace=args.replace, make_plots=not args.no_plots,
                   timing_companions=companions)
    print(json.dumps({"status": result["status"], "evidence_issues": len(result["evidence_issues"]),
                      "public_projection": str(Path(args.output) / "public")}, indent=2))


if __name__ == "__main__":
    main()
