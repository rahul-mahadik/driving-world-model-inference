"""Development-only variability pilot, separate from headline timing calibration.

Four fresh processes each measure 50 requests after their own warmup. A floor
and derivation rule are committed before measurement. The resulting tolerance
is frozen before separate held-out runs; pilot blocks are never headline runs.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from itertools import combinations
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from .benchmarks.environment import reference_identity
from .calibrate_performance import _candidate_inputs
from .config import digest, file_sha256, load_config, validate_run_config
from .data.manifests import load_manifest
from .evaluation.io import artifact_sha256, load_run
from .regression import _bootstrap_ratios, _pairs, timing_signature


TYPE = "development_variability_pilot_v1"
RULE = "max(predeclared_floor, every bidirectional p50/p95 block ratio and upper95 scene-bootstrap bound)"
PROTOCOL = ("workload", "candidate_count", "warmup_requests", "max_warmup_requests", "save_outputs")
COMPILER_SELECTORS = ("compiler_runtime", "compiler_feature_boundary")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _time(value):
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timestamps must include a timezone")
    return result


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sources():
    root = Path(__file__).parent
    return {str(p.relative_to(root)): file_sha256(p) for p in sorted(root.rglob("*.py"))}


def _manifest(path, role):
    path = Path(path).resolve()
    records = load_manifest(path)
    metadata = json.loads(path.with_suffix(path.suffix + ".meta.json").read_text())
    scenes = sorted({r["scene_id"] for r in records})
    if (metadata.get("partition_role") != role or metadata.get("manifest_sha256") != file_sha256(path)
            or metadata.get("scene_ids") != scenes or metadata.get("scene_ids_sha256") != digest(scenes)
            or metadata.get("clip_count") != len(records)):
        raise ValueError("Manifest role or full-content provenance differs")
    return records, metadata


def _runtime(environment):
    identity = reference_identity(environment)
    if (not identity.get("gpu", {}).get("uuid") or not identity.get("torch")
            or not identity.get("python") or not identity.get("packages") or not identity.get("tools")):
        raise ValueError("Full runtime, GPU UUID and tool identity are required")
    return identity


def _identity(config, environment):
    if any(not config["model"].get(k) for k in ("upstream_commit", "checkpoint_sha256")):
        raise ValueError("Pinned model commit and checkpoint hash required")
    return {"runtime": _runtime(environment), "generation": config["generation"],
            "model": {k: config["model"].get(k) for k in ("upstream_commit", "checkpoint_sha256")},
            "protocol": {k: config["benchmark"].get(k) for k in PROTOCOL}}


def _compiler_selectors(config):
    """Implementation choices stay separate from common comparison semantics."""
    return {key: config.get("model", {}).get(key) for key in COMPILER_SELECTORS}


def _reference_compiler_selectors(plan):
    return plan.get("reference_compiler_selectors", _compiler_selectors(plan["configuration"]))


def _proof_compiler_identity(config, proof, actual_identity=None):
    selected = "upstream" if config["model"].get("backend") == "upstream" else "refactored"
    tested = proof.get("runs", {}).get(selected, {}).get("model_identity", {})
    selectors = _compiler_selectors(config)
    for key, identity_key in (("compiler_runtime", "name"), ("compiler_feature_boundary", "module")):
        expected, identity = selectors[key], tested.get(key)
        if ((expected is None and identity is not None)
                or (expected is not None and (not isinstance(identity, dict) or identity.get(identity_key) != expected))):
            raise ValueError("Numerical proof compiler selectors differ from the measured implementation")
        if actual_identity is not None and identity != actual_identity.get(key):
            raise ValueError("Numerical proof compiler identity differs from the saved model identity")


def plan_pilot(config_path, output, *, relative_tolerance_floor):
    if (type(relative_tolerance_floor) not in (int, float) or not math.isfinite(relative_tolerance_floor)
            or relative_tolerance_floor < 0):
        raise ValueError("Predeclare a finite nonnegative relative tolerance floor")
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError("Pilot plans are immutable; resume or use a new directory")
    cfg = load_config(config_path)
    generation, optimizations = validate_run_config(cfg)
    if cfg.get("model", {}).get("backend") not in {"upstream", "vista"} or cfg.get("profiled"):
        raise ValueError("Pilot requires an unprofiled real backend")
    records, metadata = _manifest(cfg["data"]["manifest"], "development")
    if len(records) != 32 or len(metadata["scene_ids"]) != 32:
        raise ValueError("Pilot requires all32 distinct development scenes")
    cfg["generation"], cfg["optimizations"] = generation.to_dict(), optimizations.to_dict()
    cfg["benchmark"].update(requests=50, warmup_requests=3, max_warmup_requests=5,
                            save_outputs=0, headline=False)
    if cfg["benchmark"].get("workload") not in {"single_future", "candidate_set", "independent_scenes"}:
        raise ValueError("Explicit pilot workload required")
    environment_path = Path(cfg["environment_verification"]).resolve()
    environment_proof = json.loads(environment_path.read_text())
    if environment_proof.get("passed") is not True:
        raise ValueError("Passed dedicated runtime verification required")
    numerical_path = Path(cfg["numerical_validation"]).resolve()
    numerical = json.loads(numerical_path.read_text())
    sources = _sources()
    runtime = _runtime(environment_proof["environment"])
    if (numerical.get("passed") is not True or numerical.get("implementation_sha256") != digest(sources)
            or numerical.get("runtime_identity") != runtime):
        raise ValueError("Current-source passed numerical validation required before pilot planning")
    _proof_compiler_identity(cfg, numerical)
    lock_path = Path(cfg["environment_lock"]).resolve()
    if json.loads(lock_path.read_text()) != runtime:
        raise ValueError("Pilot runtime lock differs from verified environment")
    plan = {"schema_version": 1, "policy_type": TYPE, "created_at": _now(),
            "predeclared_floor": float(relative_tolerance_floor), "rule": RULE,
            "blocks": 4, "requests_per_block": 50, "bootstrap_resamples": 4000, "bootstrap_seed": 0,
            "configuration": cfg, "identity": _identity(cfg, environment_proof["environment"]),
            "reference_backend": cfg["model"]["backend"], "reference_optimizations": cfg["optimizations"],
            "reference_compiler_selectors": _compiler_selectors(cfg),
            "reference_input_condition": cfg["benchmark"].get("input_condition"),
            "development_manifest": {"path": str(Path(cfg["data"]["manifest"]).resolve()),
                                      "sha256": metadata["manifest_sha256"], "metadata": metadata},
            "source_files": sources, "source_identity": digest(sources),
            "prerequisites": {str(p): file_sha256(p) for p in (environment_path, lock_path, numerical_path)},
            "scope": "Four development-only50-request blocks; no headline speedup or quality conclusion"}
    destination.mkdir(parents=True)
    for name, expected in sources.items():
        target = destination / "source-snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((Path(__file__).parent / name).read_bytes())
        if file_sha256(target) != expected:
            raise ValueError("Source changed while snapshotting pilot")
    _write(destination / "plan.json", plan)
    return plan


def _load_plan(directory, *, current_source=False):
    directory = Path(directory).resolve()
    path = directory / "plan.json"
    plan = json.loads(path.read_text())
    if (plan.get("policy_type") != TYPE or plan.get("rule") != RULE or plan.get("blocks") != 4
            or plan.get("requests_per_block") != 50 or plan.get("bootstrap_resamples") != 4000
            or plan.get("bootstrap_seed") != 0 or digest(plan["source_files"]) != plan["source_identity"]
            or type(plan.get("predeclared_floor")) not in (int, float)
            or not math.isfinite(plan["predeclared_floor"]) or plan["predeclared_floor"] < 0):
        raise ValueError("Pilot plan type, source identity or fixed derivation rule differs")
    if current_source and _sources() != plan["source_files"]:
        raise ValueError("Source changed after pilot planning; create a new versioned pilot")
    if _reference_compiler_selectors(plan) != _compiler_selectors(plan["configuration"]):
        raise ValueError("Pilot reference compiler selectors differ from the frozen configuration")
    for name, expected in plan["source_files"].items():
        if file_sha256(directory / "source-snapshot" / name) != expected:
            raise ValueError("Pilot source snapshot changed")
    entry = plan["development_manifest"]
    _, metadata = _manifest(entry["path"], "development")
    if metadata != entry["metadata"] or file_sha256(entry["path"]) != entry["sha256"]:
        raise ValueError("Pilot development manifest changed")
    for item, expected in plan["prerequisites"].items():
        if file_sha256(item) != expected:
            raise ValueError("Pilot runtime or numerical prerequisite changed")
    return directory, plan


def _run_artifacts(run):
    names = ["cold_start.json", "environment_gate.json", "numerical_gate.json"]
    # Older eager receipts fingerprinted only the original fields. Preserve that
    # shape while binding the extra runtime identity for every opt-in execution.
    if (run.path / "model_identity.json").exists() and any(v is not None for v in _compiler_selectors(run.config).values()):
        names.append("model_identity.json")
    return {**artifact_sha256(run), **{name: file_sha256(run.path / name) for name in names}}


def _valid_run(run, plan, *, pilot, minimum_requests=200):
    expected = 50 if pilot else run.config["benchmark"]["requests"]
    if (run.synthetic or run.config.get("profiled") or run.summary.get("profiled")
            or run.summary.get("status", "ok") not in {"ok", "success"}
            or run.summary.get("failures") or run.summary.get("run_failure")
            or run.summary.get("recompiled_during_measurement") or run.summary.get("oom_attempts")
            or len(run.measurements) != expected or expected < (50 if pilot else minimum_requests)
            or run.config["benchmark"].get("save_outputs") != 0
            or _identity(run.config, run.environment) != plan["identity"]):
        raise ValueError("Pilot/comparison run is incomplete or differs in runtime/workload/generation")
    if pilot and (run.config["benchmark"].get("requests") != 50 or run.config["benchmark"].get("headline") is not False
                  or run.summary.get("headline_eligible") is not False):
        raise ValueError("Pilot blocks must remain explicitly non-headline50-request runs")
    for row in run.measurements:
        value = row.get("latency_s")
        if (row.get("status") != "ok" or type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                or row.get("oom_attempts") or row.get("cache_stats", {}).get("oom_attempts")
                or row.get("cache_stats", {}).get("denoiser_evaluations_complete") is False
                or row.get("fallbacks") or row.get("candidate_count") != run.config["benchmark"]["candidate_count"]
                or any(c.get("frames_path") for c in row["candidates"])):
            raise ValueError("Invalid measured request, OOM retry, fallback or exported output")
        _candidate_inputs(row)
    _pairs(run)
    _time(run.config["started_at"])


def _execution_gates(run, plan):
    gate = json.loads((run.path / "numerical_gate.json").read_text())
    if gate.get("passed") is not True or json.loads((run.path / "environment_gate.json").read_text()).get("passed") is not True:
        raise ValueError("Passed numerical and environment gates required for every process")
    numerical = Path(gate["report_path"])
    proof = json.loads(numerical.read_text())
    if (file_sha256(numerical) != gate["report_sha256"] or proof.get("passed") is not True
            or proof.get("implementation_sha256") != plan["source_identity"]
            or proof.get("runtime_identity") != plan["identity"]["runtime"]):
        raise ValueError("Numerical proof source/runtime differs from pilot")
    identity_path = run.path / "model_identity.json"
    if any(value is not None for value in _compiler_selectors(run.config).values()) and not identity_path.is_file():
        raise ValueError("Opt-in compiler execution requires its saved model identity")
    actual = json.loads(identity_path.read_text()) if identity_path.is_file() else None
    _proof_compiler_identity(run.config, proof, actual)
    cold = json.loads((run.path / "cold_start.json").read_text())
    warmups = cold.get("warmups", [])
    if (cold.get("status") != "ready" or len(warmups) < 3 or len(warmups) > 5
            or cold.get("compilation_and_initialization_excluded") is not True
            or any(r.get("status") != "ok" or r.get("fallbacks") or r.get("failures")
                   or r.get("cache_stats", {}).get("oom_attempts") for r in warmups)
            or len({r.get("compiler", {}).get("stats", {}).get("unique_graphs", 0) for r in warmups[-3:]}) != 1):
        raise ValueError("Every process needs independently completed stable warmup")


def run_pilot(directory):
    """Sequential fresh subprocesses; this command alone performs GPU inference."""
    import fcntl
    directory, plan = _load_plan(directory, current_source=True)
    with (directory / ".execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for index in range(4):
            block = directory / "blocks" / f"{index:02d}"
            receipt = block / "completed.json"
            if receipt.exists():
                _checked_block(receipt, plan, directory / "plan.json", index)
                continue
            attempts = block / "attempts"
            attempts.mkdir(parents=True, exist_ok=True)
            attempt = attempts / f"{len(list(attempts.iterdir())):03d}"
            attempt.mkdir()
            cfg = copy.deepcopy(plan["configuration"])
            cfg["pilot_provenance"] = {"plan_sha256": file_sha256(directory / "plan.json"),
                                       "block": index, "source_identity": plan["source_identity"]}
            config = attempt / "config.json"
            _write(config, cfg)
            output = attempt / "run"
            before = _sources()
            if before != plan["source_files"]:
                raise ValueError("Source changed before pilot subprocess launch")
            started = _now()
            with (attempt / "process.log").open("wb") as log:
                process = subprocess.run([sys.executable, "-m", "drivebench.benchmark", "--config", str(config),
                                          "--output", str(output)], stdout=log, stderr=subprocess.STDOUT)
            evidence = {"started_at": started, "finished_at": _now(), "exit_code": process.returncode,
                        "block": index, "plan_sha256": file_sha256(directory / "plan.json"),
                        "source_identity": digest(before), "source_unchanged": _sources() == before == plan["source_files"],
                        "path": str(output)}
            _write(attempt / "process.json", evidence)
            if process.returncode or not evidence["source_unchanged"]:
                raise ValueError(f"Pilot process failed or source changed; evidence retained in {attempt}")
            run = load_run(output)
            _valid_run(run, plan, pilot=True)
            _execution_gates(run, plan)
            evidence["artifact_sha256"] = _run_artifacts(run)
            _write(receipt, evidence)
    return {"completed_blocks": 4, "measured_requests": 200, "headline_eligible": False}


def _checked_block(path, plan, plan_path, index):
    receipt = json.loads(Path(path).read_text())
    if (receipt.get("exit_code") != 0 or receipt.get("source_unchanged") is not True
            or receipt.get("source_identity") != plan["source_identity"] or receipt.get("block") != index
            or receipt.get("plan_sha256") != file_sha256(plan_path)
            or _time(receipt["started_at"]) < _time(plan["created_at"])
            or _time(receipt["finished_at"]) < _time(receipt["started_at"])):
        raise ValueError("Pilot process provenance or chronology differs")
    run = load_run(receipt["path"])
    _valid_run(run, plan, pilot=True)
    _execution_gates(run, plan)
    if (_run_artifacts(run) != receipt["artifact_sha256"]
            or not _time(receipt["started_at"]) <= _time(run.config["started_at"]) <= _time(receipt["finished_at"])):
        raise ValueError("Pilot run changed or falls outside its process interval")
    if (run.config["model"].get("backend") != plan["reference_backend"]
            or run.config["optimizations"] != plan["reference_optimizations"]
            or _compiler_selectors(run.config) != _reference_compiler_selectors(plan)
            or run.config["benchmark"].get("input_condition") != plan["reference_input_condition"]
            or run.config["data"].get("manifest_sha256") != plan["development_manifest"]["sha256"]):
        raise ValueError("Pilot baseline implementation or manifest differs")
    return run, receipt


def _pilot_evidence(directory):
    directory, plan = _load_plan(directory)
    runs, receipts, evidence = [], [], []
    expected = None
    for index in range(4):
        path = directory / "blocks" / f"{index:02d}" / "completed.json"
        run, receipt = _checked_block(path, plan, directory / "plan.json", index)
        keys = [(key, _candidate_inputs(row)) for key, row in _pairs(run).items()]
        if expected is not None and keys != expected:
            raise ValueError("Pilot request order, actions or seeds differ across blocks")
        expected = keys
        if receipts and _time(receipt["started_at"]) < _time(receipts[-1]["finished_at"]):
            raise ValueError("Pilot processes overlap; independent sequential blocks required")
        runs.append(run)
        receipts.append(receipt)
        evidence.append({"path": str(path), "sha256": file_sha256(path), "run": receipt})
    if len({r.path for r in runs}) != 4 or len({timing_signature(r) for r in runs}) != 4:
        raise ValueError("Pilot requires four independent measurements, not copied timing observations")
    records = load_manifest(plan["development_manifest"]["path"])
    wanted = [(r["request_id"], r["scene_id"]) for r in records]
    if [(r["request_id"], r["scene_id"]) for r in runs[0].measurements] != [wanted[i % len(wanted)] for i in range(50)]:
        raise ValueError("Pilot must cycle the fixed full development manifest in order")
    comparisons, bound = [], 0.
    for a, b in combinations(range(4), 2):
        x, y = _pairs(runs[a]), _pairs(runs[b])
        keys = sorted(x)
        before, after = (np.array([items[k]["latency_s"] for k in keys]) for items in (x, y))
        metrics = {}
        for name, quantile in (("p50", .5), ("p95", .95)):
            point, ci = _bootstrap_ratios(before, after, scene_groups=[k[1] for k in keys], quantile=quantile)
            reverse, reverse_ci = 1 / (1 + point) - 1, [1 / (1 + ci[1]) - 1, 1 / (1 + ci[0]) - 1]
            required = max(0., point, reverse, ci[1], reverse_ci[1])
            bound = max(bound, required)
            metrics[name] = {"relative_change": point, "ci95": ci, "reverse_relative_change": reverse,
                             "reverse_ci95": reverse_ci, "required_tolerance": required}
        comparisons.append({"blocks": [a, b], "paired_requests": 50, "independent_scenes": 32, "metrics": metrics})
    medians = [float(np.median([r["latency_s"] for r in run.measurements])) for run in runs]
    return plan, {"receipts": evidence, "comparisons": comparisons, "measured_variability_bound": bound,
                  "block_medians_s": medians, "relative_median_range": max(medians) / min(medians) - 1,
                  "development_scenes": plan["development_manifest"]["metadata"]["scene_ids"]}


def freeze_pilot(directory, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Frozen pilot policies cannot be overwritten")
    _load_plan(directory, current_source=True)
    plan, evidence = _pilot_evidence(directory)
    frozen = _now()
    if any(_time(item["run"]["finished_at"]) > _time(frozen) for item in evidence["receipts"]):
        raise ValueError("Pilot measurements must finish before policy freezing")
    policy = {"schema_version": 1, "policy_type": TYPE, "frozen_at": frozen,
              "pilot_directory": str(Path(directory).resolve()), "plan_sha256": file_sha256(Path(directory) / "plan.json"),
              "rule": RULE, "predeclared_floor": plan["predeclared_floor"], "evidence": evidence,
              "relative_latency_tolerance": max(plan["predeclared_floor"], evidence["measured_variability_bound"]),
              "minimum_headline_requests": 200, "require_confirmatory_rerun": True,
              "pilot_headline_eligible": False, "quality_preserving": False,
              "limitation": "32 development scenes, four50-request blocks on one GPU; scene-bootstrap bounds do not guarantee stationarity or fleet-wide error control"}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(policy, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return policy


def validate_pilot_policy(path):
    policy = json.loads(Path(path).read_text())
    plan, evidence = _pilot_evidence(policy["pilot_directory"])
    if (policy.get("policy_type") != TYPE or policy.get("rule") != RULE
            or policy.get("plan_sha256") != file_sha256(Path(policy["pilot_directory"]) / "plan.json")
            or policy.get("evidence") != evidence or policy.get("predeclared_floor") != plan["predeclared_floor"]
            or policy.get("relative_latency_tolerance") != max(plan["predeclared_floor"], evidence["measured_variability_bound"])
            or policy.get("minimum_headline_requests") != 200 or policy.get("require_confirmatory_rerun") is not True
            or any(_time(r["run"]["finished_at"]) > _time(policy["frozen_at"]) for r in evidence["receipts"])):
        raise ValueError("Pilot policy derivation, chronology or evidence changed")
    return policy, plan


def _headline(run, policy, plan, *, reference):
    _valid_run(run, plan, pilot=False)
    if run.summary.get("headline_eligible") is not True or _time(run.config["started_at"]) <= _time(policy["frozen_at"]):
        raise ValueError("Eligible headline measurements must start after pilot policy freeze")
    if reference and (run.config["optimizations"] != plan["reference_optimizations"]
                      or run.config["model"].get("backend") != plan["reference_backend"]
                      or _compiler_selectors(run.config) != _reference_compiler_selectors(plan)
                      or run.config["benchmark"].get("input_condition") != plan["reference_input_condition"]):
        raise ValueError("Headline reference implementation differs from calibrated baseline")
    if set(r["scene_id"] for r in run.measurements) & set(policy["evidence"]["development_scenes"]):
        raise ValueError("Headline scenes overlap development pilot")
    metadata = run.config["data"].get("manifest_metadata", {})
    records, actual_metadata = _manifest(run.config["data"]["manifest"], "held_out_evaluation")
    if (metadata != actual_metadata or len(records) != 128 or len(metadata["scene_ids"]) < 64
            or run.config["data"].get("manifest_sha256") != metadata["manifest_sha256"]):
        raise ValueError("Headline comparison requires held-out evaluation manifest provenance")
    # Aggregates inherit their first block's configuration; verify every actual
    # subprocess gate and its preserved artifacts, not the aggregate flag alone.
    blocks = run.summary.get("timing_blocks")
    if blocks:
        if len(blocks) != 4 or sum(len(load_run(b["path"]).measurements) for b in blocks) != len(run.measurements):
            raise ValueError("Headline aggregate block coverage differs")
        sources = [load_run(block["path"]) for block in blocks]
        if len({source.path for source in sources}) != 4 or len({timing_signature(source) for source in sources}) != 4:
            raise ValueError("Headline aggregate needs independently collected source blocks")
        reconstructed = []
        for index, block in enumerate(blocks):
            source = load_run(block["path"])
            _valid_run(source, plan, pilot=False, minimum_requests=50)
            if artifact_sha256(source) != block["artifact_sha256"] or _time(source.config["started_at"]) <= _time(policy["frozen_at"]):
                raise ValueError("Headline source block changed or predates freeze")
            if _identity(source.config, source.environment) != plan["identity"]:
                raise ValueError("Headline source block identity differs from pilot")
            if (source.config["generation"] != run.config["generation"]
                    or source.config["optimizations"] != run.config["optimizations"]
                    or source.config["model"] != run.config["model"]
                    or source.config["benchmark"].get("input_condition") != run.config["benchmark"].get("input_condition")
                    or source.config["data"] != run.config["data"]):
                raise ValueError("Headline aggregate and source implementation differ")
            _execution_gates(source, plan)
            repetitions = max(r.get("repeat", 0) for r in source.measurements) + 1
            for original in source.measurements:
                row = copy.deepcopy(original)
                row.update(request_index=len(reconstructed), repeat=index * repetitions + original.get("repeat", 0),
                           timing_block=index, original_request_index=original.get("request_index"))
                reconstructed.append(row)
        if reconstructed != run.measurements:
            raise ValueError("Headline aggregate observations differ from preserved source blocks")
    else:
        _execution_gates(run, plan)
    allowed_requests = {(r["request_id"], r["scene_id"]) for r in records}
    if any((r["request_id"], r["scene_id"]) not in allowed_requests for r in run.measurements):
        raise ValueError("Headline observations fall outside the frozen held-out manifest")


def _comparison(baseline, candidate, policy, plan):
    _headline(baseline, policy, plan, reference=True)
    _headline(candidate, policy, plan, reference=False)
    left, right = _pairs(baseline), _pairs(candidate)
    if (list(left) != list(right) or baseline.path == candidate.path
            or timing_signature(baseline) == timing_signature(candidate)):
        raise ValueError("Headline comparisons require matched independent requests in the same order")
    for key in left:
        if _candidate_inputs(left[key]) != _candidate_inputs(right[key]):
            raise ValueError("Headline candidate action or seed identities differ")
    if baseline.config["data"].get("manifest_sha256") != candidate.config["data"].get("manifest_sha256"):
        raise ValueError("Headline manifests differ")
    keys = sorted(left)
    values = [np.array([items[k]["latency_s"] for k in keys]) for items in (left, right)]
    metrics = {}
    for name, quantile in (("p50", .5), ("p95", .95)):
        point, ci = _bootstrap_ratios(*values, scene_groups=[k[1] for k in keys], quantile=quantile)
        threshold = policy["relative_latency_tolerance"]
        metrics[name] = {"relative_latency_change": point, "ci95": ci, "tolerance": threshold,
                         "exceeds_tolerance_and_uncertainty": point > threshold and ci[0] > threshold}
    exceeds = any(m["exceeds_tolerance_and_uncertainty"] for m in metrics.values())
    possible = any(m["relative_latency_change"] > m["tolerance"] for m in metrics.values())
    return {"status": "rerun_required" if exceeds else "inconclusive" if possible else "pass",
            "metrics": metrics, "paired_requests": len(keys), "independent_scenes": len({k[1] for k in keys}),
            "baseline": str(baseline.path), "candidate": str(candidate.path),
            "baseline_artifact_sha256": artifact_sha256(baseline), "candidate_artifact_sha256": artifact_sha256(candidate),
            "optimization_changes": {"baseline": baseline.config["optimizations"], "candidate": candidate.config["optimizations"]},
            "compiler_implementation_changes": {"baseline": _compiler_selectors(baseline.config),
                                                "candidate": _compiler_selectors(candidate.config)},
            "input_conditions": {"baseline": baseline.config["benchmark"].get("input_condition"),
                                 "candidate": candidate.config["benchmark"].get("input_condition")},
            "quality_preserving": False, "quality_note": "Requires separate frozen held-out quality evidence"}


def compare_pilot(policy_path, baseline_path, candidate_path, *, confirm_baseline=None, confirm_candidate=None):
    policy, plan = validate_pilot_policy(policy_path)
    baseline, candidate = load_run(baseline_path), load_run(candidate_path)
    result = _comparison(baseline, candidate, policy, plan)
    result["policy_sha256"] = file_sha256(policy_path)
    result.update(policy_type=TYPE, pilot_policy_verified=True,
                  interpretation="Pass means no detected regression beyond this development-pilot tolerance; it is not a speedup or quality claim")
    if (confirm_baseline is None) != (confirm_candidate is None):
        raise ValueError("Provide both confirmatory runs")
    if result["status"] == "rerun_required" and confirm_baseline is not None:
        a, b = load_run(confirm_baseline), load_run(confirm_candidate)
        all_runs = (baseline, candidate, a, b)
        if len({r.path for r in all_runs}) != 4 or len({timing_signature(r) for r in all_runs}) != 4:
            raise ValueError("Confirmation requires independently collected measurements")
        if (a.config["optimizations"] != baseline.config["optimizations"]
                or b.config["optimizations"] != candidate.config["optimizations"]
                or _compiler_selectors(a.config) != _compiler_selectors(baseline.config)
                or _compiler_selectors(b.config) != _compiler_selectors(candidate.config)
                or b.config["model"].get("backend") != candidate.config["model"].get("backend")
                or b.config["benchmark"].get("input_condition") != candidate.config["benchmark"].get("input_condition")):
            raise ValueError("Confirmatory implementation differs")
        confirmation = _comparison(a, b, policy, plan)
        result["confirmation"] = confirmation
        repeated = any(result["metrics"][m]["exceeds_tolerance_and_uncertainty"] and
                       confirmation["metrics"][m]["exceeds_tolerance_and_uncertainty"] for m in ("p50", "p95"))
        result["status"] = "regression_confirmed" if repeated else "not_confirmed"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan"); p.add_argument("--config", required=True); p.add_argument("--output", required=True)
    p.add_argument("--relative-tolerance-floor", type=float, required=True)
    p = sub.add_parser("run"); p.add_argument("--pilot", required=True)
    p = sub.add_parser("freeze"); p.add_argument("--pilot", required=True); p.add_argument("--output", required=True)
    p = sub.add_parser("compare"); p.add_argument("--policy", required=True); p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", required=True); p.add_argument("--confirm-baseline"); p.add_argument("--confirm-candidate")
    p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        result = plan_pilot(args.config, args.output, relative_tolerance_floor=args.relative_tolerance_floor)
    elif args.command == "run":
        result = run_pilot(args.pilot)
    elif args.command == "freeze":
        result = freeze_pilot(args.pilot, args.output)
    else:
        result = compare_pilot(args.policy, args.baseline, args.candidate,
                               confirm_baseline=args.confirm_baseline, confirm_candidate=args.confirm_candidate)
        _write(args.output, result)
    print(json.dumps(result, indent=2))
    if args.command == "compare":
        raise SystemExit(1 if result["status"] == "regression_confirmed" else 0 if result["status"] == "pass" else 2)


if __name__ == "__main__":
    main()
