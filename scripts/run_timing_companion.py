#!/usr/bin/env python3
"""Run a separate timing workload using an existing immutable study selection.

This never generates quality outputs or revises quality policies. Each workload
needs its own completed development pilot and explicit numerical-proof bindings.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drivebench import calibrate_pilot as pilot
from drivebench import study
from drivebench.config import digest, file_sha256
from drivebench.data.manifests import load_manifest
from drivebench.evaluation.io import artifact_sha256, load_run


TYPE = "selection_bound_timing_companion_v1"


def _semantics_helper():
    path = Path(__file__).with_name("verify_numerical_semantics.py")
    spec = importlib.util.spec_from_file_location("companion_exact_semantics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _semantic_binding(proof_path, sidecar=None):
    helper = _semantics_helper()
    proof = json.loads(Path(proof_path).read_text())
    if sidecar is None:
        if helper.requires_sidecar(proof):
            raise ValueError("Nonzero numerical tolerances require a recomputed exact semantic sidecar")
        return None
    return helper.verify(proof_path, sidecar)


def _check_semantic_binding(proof_path, binding):
    actual = _semantic_binding(proof_path, binding["path"] if binding else None)
    if actual != binding:
        raise ValueError("Exact semantic sidecar binding changed")
    return actual


def _pilot_policy_semantics(policy_path, calibration):
    """Nonzero pilot prerequisites must have passed the guarded pilot workflow."""
    proof_path = calibration["configuration"].get("numerical_validation")
    if proof_path is None:
        raise ValueError("Pilot calibration lacks its numerical prerequisite")
    proof = json.loads(Path(proof_path).read_text())
    if not _semantics_helper().requires_sidecar(proof):
        return None
    policy = json.loads(Path(policy_path).read_text())
    root = Path(policy["pilot_directory"])
    _pilot_guard(root)
    receipt = json.loads((root / "companion-frozen-guard.json").read_text())
    if (receipt.get("policy_path") != str(Path(policy_path).resolve())
            or receipt.get("policy_sha256") != file_sha256(policy_path)
            or receipt.get("run_guard_sha256") != file_sha256(root / "companion-run-guard.json")
            or receipt.get("plan_guard_sha256") != file_sha256(root / "companion-guard.json")
            or receipt.get("helper_sha256") != file_sha256(__file__)):
        raise ValueError("Pilot policy lacks unchanged guarded semantic qualification")
    return {"path": str((root / "companion-frozen-guard.json").resolve()),
            "sha256": file_sha256(root / "companion-frozen-guard.json")}


def _runtime_overlay(plan, path):
    """Reuse compatibility validation, then bind a new timing allocation explicitly."""
    if path is None:
        return {}, None
    path = Path(path).resolve()
    original = {key: str(Path(plan["base_config"][key]).resolve())
                for key in ("environment_verification", "environment_lock")}
    original_proof = json.loads(Path(original["environment_verification"]).read_text())
    original_identity = pilot._runtime(original_proof.get("environment", {}))
    if (original_proof.get("passed") is not True
            or json.loads(Path(original["environment_lock"]).read_text()) != original_identity):
        raise ValueError("Original study environment proof and lock are inconsistent")
    # This permits only UUID/PCI replacement; its quality-only provenance is
    # deliberately discarded rather than being treated as timing authorization.
    validated = study.quality_runtime_overlay(plan, path)
    replacement = {key: str(Path(validated[key]).resolve()) for key in original}
    replacement_proof = json.loads(Path(replacement["environment_verification"]).read_text())
    pilot._runtime(replacement_proof.get("environment", {}))
    paths = [str(path), *original.values(), *replacement.values()]
    binding = {"path": str(path), "sha256": file_sha256(path),
        "original_environment_paths": original, "replacement_environment_paths": replacement,
        "bound_files_sha256": {name: file_sha256(name) for name in paths},
        "scope": "New timing allocation; compatibility with prior quality permits UUID/PCI replacement only. All pilot, proof and timing identities must use the same full new GPU UUID. Never merge timing blocks across GPU UUIDs."}
    return {**replacement, "timing_runtime_overlay": binding}, binding


def _matching_runtime(config, expected):
    proof = json.loads(Path(config["environment_verification"]).read_text())
    lock = json.loads(Path(config["environment_lock"]).read_text())
    actual = pilot._runtime(proof.get("environment", {}))
    if proof.get("passed") is not True or actual != expected or lock != actual:
        raise ValueError("Timing environment proof/lock differs from the full pilot GPU/runtime identity")


def _quality_bindings(selection):
    bindings = {}
    for key in ("development_evaluation", "quality_policy"):
        path = selection.get(key)
        if path is None:
            continue
        expected = selection.get(key + "_sha256")
        if not isinstance(path, str) or file_sha256(path) != expected:
            raise ValueError("Selected development or quality evidence changed")
        bindings[str(Path(path).resolve())] = expected
    for reviewed in selection.get("reviewed_quality_policies", {}).values():
        for path_key, hash_key in (("path", "sha256"), ("evaluation_path", "evaluation_sha256")):
            path, expected = reviewed[path_key], reviewed[hash_key]
            if file_sha256(path) != expected:
                raise ValueError("Selected reviewed quality evidence changed")
            bindings[str(Path(path).resolve())] = expected
        for source in reviewed["source_runs"]:
            run = load_run(source["path"])
            actual = artifact_sha256(run)
            if actual != source["artifact_sha256"]:
                raise ValueError("Selected reviewed quality source run changed")
            for name, expected in actual.items():
                bindings[str(run.path / name)] = expected
    if not bindings:
        raise ValueError("Timing companion needs immutable development and quality evidence")
    return bindings


def prepare_pilot_config(study_path, output, *, baseline, candidate_count, numerical_proof, runtime_config=None, semantic_sidecar=None):
    """Materialize the selected baseline's development workload without running it."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Pilot configurations cannot be overwritten")
    if type(candidate_count) is not int or not 1 <= candidate_count <= 16:
        raise ValueError("Pilot candidate count must be an integer from 1 to 16")
    parent, plan = study._load_plan(study_path)
    selected = study._selection(parent, plan)
    if baseline not in selected["variants"]:
        raise ValueError("Pilot baseline is not a frozen finalist")
    template = copy.deepcopy(plan)
    template["timing"]["warmups"] = 3
    cfg = study._variant_config(template, baseline, plan["manifests"]["development"]["path"], 50, candidate_count, quality=False)
    overlay, overlay_binding = _runtime_overlay(plan, runtime_config)
    cfg.update(overlay)
    proof_path = Path(numerical_proof).resolve()
    proof = json.loads(proof_path.read_text())
    if proof.get("passed") is not True or proof.get("implementation_sha256") != digest(pilot._sources()):
        raise ValueError("Pilot needs a passed current-source numerical proof")
    pilot._proof_compiler_identity(cfg, proof)
    _matching_runtime(cfg, proof.get("runtime_identity"))
    semantic_binding = _semantic_binding(proof_path, semantic_sidecar)
    cfg["numerical_validation"] = str(proof_path)
    quality_bindings = _quality_bindings(selected)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(cfg, stream, indent=2, sort_keys=True)
        stream.write("\n")
    receipt = {"config_sha256": file_sha256(output), "parent_plan_sha256": file_sha256(parent / "plan.json"),
        "selection_sha256": file_sha256(parent / "selection.json"), "quality_evidence_sha256": quality_bindings,
        "numerical_proof_sha256": file_sha256(proof_path), "helper_sha256": file_sha256(__file__),
        "exact_semantics": semantic_binding,
        "timing_runtime_overlay": overlay_binding,
        "quality_generated_here": False, "gpu_execution_performed": False}
    study.atomic_json(output.with_suffix(output.suffix + ".receipt.json"), receipt)
    return receipt


def plan_companion(study_path, output, *, baseline, candidate, candidate_count, pilot_policy, numerical_proofs, runtime_config=None, semantic_sidecars=None):
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError("Companion plans cannot be overwritten; resume or choose a new directory")
    if type(candidate_count) is not int or not 1 <= candidate_count <= 16:
        raise ValueError("Companion candidate count must be an integer from 1 to 16")
    parent, parent_plan = study._load_plan(study_path)
    selected = study._selection(parent, parent_plan)
    overlay, overlay_binding = _runtime_overlay(parent_plan, runtime_config)
    names = [baseline, candidate]
    if baseline == candidate or not set(names) <= set(selected["variants"]):
        raise ValueError("Choose two distinct development-frozen finalists")
    if set(numerical_proofs) != set(names):
        raise ValueError("Bind exactly one explicit numerical proof for each companion finalist")
    semantic_sidecars = semantic_sidecars or {}
    if not set(semantic_sidecars) <= set(names):
        raise ValueError("Semantic sidecars must name selected finalists")
    policy_path = Path(pilot_policy).resolve()
    policy, calibration = pilot.validate_pilot_policy(policy_path)
    pilot_semantics = _pilot_policy_semantics(policy_path, calibration)
    if digest(pilot._sources()) != calibration["source_identity"]:
        raise ValueError("Companion source differs from the workload pilot")
    template = copy.deepcopy(parent_plan)
    template["timing"].update(warmups=3, blocks=4, requests_per_block=50, candidate_count=candidate_count)
    environment = json.loads(Path(calibration["configuration"]["environment_verification"]).read_text())["environment"]
    configs, proofs, semantics = {}, {}, {}
    for name in names:
        cfg = study._variant_config(template, name, parent_plan["manifests"]["evaluation"]["path"], 50, candidate_count, quality=False)
        cfg.update(overlay)
        _matching_runtime(cfg, calibration["identity"]["runtime"])
        proof_path = Path(numerical_proofs[name]).resolve()
        proof = json.loads(proof_path.read_text())
        if (proof.get("passed") is not True or proof.get("implementation_sha256") != calibration["source_identity"]
                or proof.get("runtime_identity") != calibration["identity"]["runtime"]):
            raise ValueError("Companion numerical proof source, runtime or pass status differs")
        pilot._proof_compiler_identity(cfg, proof)
        semantics[name] = _semantic_binding(proof_path, semantic_sidecars.get(name))
        if pilot._identity(cfg, environment) != calibration["identity"]:
            raise ValueError("Companion workload/generation differs from its own calibration pilot")
        cfg["numerical_validation"] = str(proof_path)
        configs[name] = cfg
        proofs[str(proof_path)] = file_sha256(proof_path)
    reference = configs[baseline]
    if (reference["model"].get("backend") != calibration["reference_backend"]
            or reference["optimizations"] != calibration["reference_optimizations"]
            or pilot._compiler_selectors(reference) != pilot._reference_compiler_selectors(calibration)
            or reference["benchmark"].get("input_condition") != calibration["reference_input_condition"]):
        raise ValueError("Companion baseline implementation differs from its own pilot")
    value = {"schema_version": 1, "type": TYPE, "created_at": study.now(),
        "parent_study": str(parent), "parent_plan_sha256": file_sha256(parent / "plan.json"),
        "selection_sha256": file_sha256(parent / "selection.json"), "quality_evidence_sha256": _quality_bindings(selected),
        "pilot_policy": str(policy_path), "pilot_policy_sha256": file_sha256(policy_path),
        "pilot_exact_semantics": pilot_semantics,
        "source_identity": calibration["source_identity"], "helper_sha256": file_sha256(__file__),
        "timing_runtime_overlay": overlay_binding,
        "baseline": baseline, "candidate": candidate, "candidate_count": candidate_count,
        "workload": "single_future" if candidate_count == 1 else "candidate_set",
        "schedule": study.timing_schedule(names, 4, 50), "configs": configs,
        "numerical_proof_sha256": proofs, "save_outputs": 0,
        "exact_semantics": semantics,
        "scope": "Timing only, four independently warmed 50-request blocks per finalist; reuses the original selection and quality evidence.",
        "quality_generated_here": False, "combined_headline_claim": False}
    destination.mkdir(parents=True)
    study.atomic_json(destination / "plan.json", value)
    study.atomic_json(destination / "plan-receipt.json", {"plan_sha256": file_sha256(destination / "plan.json")})
    return value


def _load(directory):
    directory = Path(directory).resolve()
    receipt = json.loads((directory / "plan-receipt.json").read_text())
    if receipt.get("plan_sha256") != file_sha256(directory / "plan.json"):
        raise ValueError("Companion immutable plan changed")
    value = json.loads((directory / "plan.json").read_text())
    if value.get("type") != TYPE or value.get("helper_sha256") != file_sha256(__file__):
        raise ValueError("Companion type or helper changed")
    parent, parent_plan = study._load_plan(value["parent_study"])
    selected = study._selection(parent, parent_plan)
    saved_overlay = value.get("timing_runtime_overlay")
    overlay, overlay_binding = _runtime_overlay(parent_plan, saved_overlay["path"] if saved_overlay else None)
    if overlay_binding != saved_overlay:
        raise ValueError("Timing allocation overlay or original/replacement evidence changed")
    if (file_sha256(parent / "plan.json") != value["parent_plan_sha256"]
            or file_sha256(parent / "selection.json") != value["selection_sha256"]
            or _quality_bindings(selected) != value["quality_evidence_sha256"]
            or file_sha256(value["pilot_policy"]) != value["pilot_policy_sha256"]
            or digest(pilot._sources()) != value["source_identity"]):
        raise ValueError("Companion selection, quality, pilot or source binding changed")
    for path, expected in value["numerical_proof_sha256"].items():
        if file_sha256(path) != expected:
            raise ValueError("Companion numerical proof changed")
    _, calibration = pilot.validate_pilot_policy(value["pilot_policy"])
    if _pilot_policy_semantics(value["pilot_policy"], calibration) != value.get("pilot_exact_semantics"):
        raise ValueError("Pilot exact semantic guard changed")
    names = [value["baseline"], value["candidate"]]
    if value["schedule"] != study.timing_schedule(names, 4, 50) or set(value["configs"]) != set(names):
        raise ValueError("Companion schedule differs from four balanced 50-request blocks")
    if (names[0] == names[1] or not set(names) <= set(selected["variants"])
            or type(value["candidate_count"]) is not int or not 1 <= value["candidate_count"] <= 16
            or value["workload"] != ("single_future" if value["candidate_count"] == 1 else "candidate_set")):
        raise ValueError("Companion finalists or workload changed")
    template = copy.deepcopy(parent_plan)
    template["timing"].update(warmups=3, blocks=4, requests_per_block=50, candidate_count=value["candidate_count"])
    for name in names:
        expected = study._variant_config(template, name, parent_plan["manifests"]["evaluation"]["path"], 50, value["candidate_count"], quality=False)
        expected.update(overlay)
        actual = copy.deepcopy(value["configs"][name])
        numerical = actual.get("numerical_validation")
        if numerical not in value["numerical_proof_sha256"]:
            raise ValueError("Companion config numerical proof is not bound")
        expected["numerical_validation"] = numerical
        _check_semantic_binding(numerical, value.get("exact_semantics", {}).get(name))
        # The original timestamp is evidence; reconstruct only stable fields.
        for cfg in (actual, expected):
            cfg.get("study_provenance", {}).pop("recorded_at", None)
        if actual != expected:
            raise ValueError("Companion config differs from its selected variant")
    return directory, value, parent


def run_companion(directory):
    directory, value, parent = _load(directory)
    _, calibration = pilot.validate_pilot_policy(value["pilot_policy"])
    names = [value["baseline"], value["candidate"]]
    paths = {name: [] for name in names}
    # Share the original execution lock so a companion cannot overlap its study.
    with (parent / ".execution.lock").open("a") as parent_lock, (directory / ".execution.lock").open("a") as own_lock:
        fcntl.flock(parent_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for item in value["schedule"]:
            _load(directory)
            name, cfg = item["variant"], copy.deepcopy(value["configs"][item["variant"]])
            records = load_manifest(cfg["data"]["manifest"])
            receipt = study._execute_run(directory / "blocks" / f"{item['index']:03d}", name, cfg, records)
            _load(directory)
            run = study._validate_completed_run(receipt["path"], cfg, records)
            pilot._valid_run(run, calibration, pilot=False, minimum_requests=50)
            pilot._execution_gates(run, calibration)
            paths[name].append(receipt["path"])
    aggregates = {}
    for name in names:
        _load(directory)
        output = directory / "aggregate" / name
        if not output.exists():
            run = study.aggregate_timing_blocks(paths[name], output)
        else:
            run = load_run(output)
            if [b["artifact_sha256"] for b in run.summary["timing_blocks"]] != [artifact_sha256(load_run(p)) for p in paths[name]]:
                raise ValueError("Companion aggregate source blocks changed")
        aggregates[name] = str(run.path)
    _load(directory)
    result = pilot.compare_pilot(value["pilot_policy"], aggregates[names[0]], aggregates[names[1]])
    study.atomic_json(directory / "comparison.json", result)
    completion = {"status": "complete", "plan_sha256": file_sha256(directory / "plan.json"),
        "finished_at": study.now(), "aggregate_runs": aggregates,
        "artifact_sha256": {name: artifact_sha256(load_run(path)) for name, path in aggregates.items()},
        "comparison_sha256": file_sha256(directory / "comparison.json"),
        "performance_status": result["status"], "quality_generated_here": False, "combined_headline_claim": False}
    study.atomic_json(directory / "completed.json", completion)
    return completion


def _pilot_config_receipt(config_path):
    """Guard the standalone pilot workflow without changing frozen core APIs."""
    path = Path(config_path).resolve()
    receipt = json.loads(path.with_suffix(path.suffix + ".receipt.json").read_text())
    config = json.loads(path.read_text())
    if receipt.get("config_sha256") != file_sha256(path) or receipt.get("helper_sha256") != file_sha256(__file__):
        raise ValueError("Prepared pilot config or companion helper changed")
    proof_path = config["numerical_validation"]
    if receipt.get("numerical_proof_sha256") != file_sha256(proof_path):
        raise ValueError("Prepared pilot numerical proof changed")
    _check_semantic_binding(proof_path, receipt.get("exact_semantics"))
    return path, receipt


def guarded_pilot_plan(config_path, output, *, relative_tolerance_floor):
    path, receipt = _pilot_config_receipt(config_path)
    value = pilot.plan_pilot(path, output, relative_tolerance_floor=relative_tolerance_floor)
    _pilot_config_receipt(path)
    study.atomic_json(Path(output) / "companion-guard.json", {
        "config": str(path), "config_receipt_sha256": file_sha256(path.with_suffix(path.suffix + ".receipt.json")),
        "pilot_plan_sha256": file_sha256(Path(output) / "plan.json"),
        "exact_semantics": receipt.get("exact_semantics"), "helper_sha256": file_sha256(__file__)})
    return value


def _pilot_guard(directory):
    root = Path(directory).resolve()
    value = json.loads((root / "companion-guard.json").read_text())
    path, receipt = _pilot_config_receipt(value["config"])
    if (value.get("helper_sha256") != file_sha256(__file__)
            or value.get("pilot_plan_sha256") != file_sha256(root / "plan.json")
            or value.get("config_receipt_sha256") != file_sha256(path.with_suffix(path.suffix + ".receipt.json"))
            or value.get("exact_semantics") != receipt.get("exact_semantics")):
        raise ValueError("Standalone pilot semantic guard changed")
    return value


def guarded_pilot_run(directory):
    _pilot_guard(directory)
    result = pilot.run_pilot(directory)
    _pilot_guard(directory)
    study.atomic_json(Path(directory) / "companion-run-guard.json", {
        "plan_guard_sha256": file_sha256(Path(directory) / "companion-guard.json"),
        "helper_sha256": file_sha256(__file__), "result": result})
    return result


def guarded_pilot_freeze(directory, output):
    _pilot_guard(directory)
    root = Path(directory)
    run_guard = json.loads((root / "companion-run-guard.json").read_text())
    if (run_guard.get("plan_guard_sha256") != file_sha256(root / "companion-guard.json")
            or run_guard.get("helper_sha256") != file_sha256(__file__)):
        raise ValueError("Pilot must complete the guarded semantic run before freezing")
    result = pilot.freeze_pilot(directory, output)
    _pilot_guard(directory)
    study.atomic_json(root / "companion-frozen-guard.json", {"policy_path": str(Path(output).resolve()),
        "policy_sha256": file_sha256(output), "run_guard_sha256": file_sha256(root / "companion-run-guard.json"),
        "plan_guard_sha256": file_sha256(root / "companion-guard.json"), "helper_sha256": file_sha256(__file__)})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("prepare-pilot")
    command.add_argument("--study", required=True); command.add_argument("--output", required=True)
    command.add_argument("--baseline", required=True); command.add_argument("--candidate-count", type=int, default=1)
    command.add_argument("--numerical-proof", required=True)
    command.add_argument("--semantic-sidecar", help="Required exact-zero retained-output sidecar for any nonzero proof tolerance")
    command.add_argument("--runtime-config", help="Explicit compatible replacement environment proof/lock paths for this timing allocation")
    command = commands.add_parser("plan")
    command.add_argument("--study", required=True); command.add_argument("--output", required=True)
    command.add_argument("--baseline", required=True); command.add_argument("--candidate", required=True)
    command.add_argument("--candidate-count", type=int, default=1); command.add_argument("--pilot-policy", required=True)
    command.add_argument("--proof", action="append", required=True, help="VARIANT=NUMERICAL_REPORT")
    command.add_argument("--semantic-sidecar", action="append", default=[], help="VARIANT=EXACT_SEMANTIC_SIDECAR")
    command.add_argument("--runtime-config", help="Same timing allocation overlay used for the workload pilot")
    command = commands.add_parser("run"); command.add_argument("--companion", required=True)
    command = commands.add_parser("pilot-plan"); command.add_argument("--config", required=True)
    command.add_argument("--output", required=True); command.add_argument("--relative-tolerance-floor", type=float, required=True)
    command = commands.add_parser("pilot-run"); command.add_argument("--pilot", required=True)
    command = commands.add_parser("pilot-freeze"); command.add_argument("--pilot", required=True); command.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-pilot":
        print(json.dumps(prepare_pilot_config(args.study, args.output, baseline=args.baseline,
              candidate_count=args.candidate_count, numerical_proof=args.numerical_proof, runtime_config=args.runtime_config,
              semantic_sidecar=args.semantic_sidecar), indent=2))
    elif args.command == "plan":
        proofs = {}
        for item in args.proof:
            name, separator, path = item.partition("=")
            if not separator or not name or not path or name in proofs:
                parser.error("Use one unique --proof VARIANT=PATH for each finalist")
            proofs[name] = path
        sidecars = {}
        for item in args.semantic_sidecar:
            name, separator, path = item.partition("=")
            if not separator or not name or not path or name in sidecars:
                parser.error("Use one unique --semantic-sidecar VARIANT=PATH for each nonzero proof")
            sidecars[name] = path
        result = plan_companion(args.study, args.output, baseline=args.baseline, candidate=args.candidate,
                                candidate_count=args.candidate_count, pilot_policy=args.pilot_policy, numerical_proofs=proofs,
                                runtime_config=args.runtime_config, semantic_sidecars=sidecars)
        print(json.dumps({"status": "planned", "workload": result["workload"], "candidate_count": result["candidate_count"],
                          "blocks": len(result["schedule"]), "quality_generated_here": False}))
    elif args.command == "pilot-plan":
        print(json.dumps(guarded_pilot_plan(args.config, args.output, relative_tolerance_floor=args.relative_tolerance_floor), indent=2))
    elif args.command == "pilot-run":
        print(json.dumps(guarded_pilot_run(args.pilot), indent=2))
    elif args.command == "pilot-freeze":
        print(json.dumps(guarded_pilot_freeze(args.pilot, args.output), indent=2))
    else:
        print(json.dumps(run_companion(args.companion), indent=2))


if __name__ == "__main__":
    main()
