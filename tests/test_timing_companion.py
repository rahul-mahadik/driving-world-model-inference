"""CPU-only orchestration fixtures; fabricated timings never represent a GPU."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from drivebench.config import digest, file_sha256, validate_run_config
from drivebench.data.manifests import write_manifest

SPEC = importlib.util.spec_from_file_location("timing_companion", Path(__file__).resolve().parents[1] / "scripts/run_timing_companion.py")
companion = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(companion)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path, monkeypatch):
    parent = tmp_path / "study"
    parent.mkdir()
    manifest = parent / "evaluation.jsonl"
    write_manifest(manifest, [{"request_id": f"e{i}", "scene_id": f"s{i}"} for i in range(128)], {"partition_role": "held_out_evaluation"})
    environment = {"gpu": {"uuid": "GPU-CPU-FIXTURE"}, "torch": {"version": "fixture"},
        "python": "fixture", "packages": {"torch": "fixture"}, "tools": {"ncu": {"stdout": "fixture"}}}
    verification = tmp_path / "environment.json"
    put(verification, {"passed": True, "environment": environment})
    lock = tmp_path / "environment-lock.json"
    put(lock, companion.pilot._runtime(environment))
    cfg = {"model": {"backend": "upstream", "upstream_commit": "a" * 40, "checkpoint_sha256": "b" * 64},
        "generation": {}, "optimizations": {}, "data": {"manifest": str(manifest)},
        "benchmark": {"workload": "single_future", "candidate_count": 1, "warmup_requests": 3,
            "max_warmup_requests": 5, "save_outputs": 0, "input_condition": "uncached"},
        "environment_verification": str(verification), "environment_lock": str(lock)}
    generation, optimizations = validate_run_config(cfg)
    cfg["generation"], cfg["optimizations"] = generation.to_dict(), optimizations.to_dict()
    parent_plan = {"base_config": cfg, "variants": {"upstream": {}, "reuse": {"model": {"backend": "vista"},
        "optimizations": {"conditioning_cache_bytes": 1024}, "benchmark": {"input_condition": "reused_conditioning"}}},
        "timing": {"warmups": 3, "candidate_count": 4}, "manifests": {"evaluation": {"path": str(manifest)},
            "development": {"path": str(manifest)}}}
    put(parent / "plan.json", parent_plan)
    selection = {"variants": ["upstream", "reuse"]}
    for key in ("quality_policy", "development_evaluation"):
        path = parent / f"{key}.json"; put(path, {"scope": "CPU fixture"})
        selection[key], selection[key + "_sha256"] = str(path), file_sha256(path)
    put(parent / "selection.json", selection)
    monkeypatch.setattr(companion.study, "_load_plan", lambda path: (parent, copy.deepcopy(parent_plan)))
    monkeypatch.setattr(companion.study, "_selection", lambda *args: json.loads((parent / "selection.json").read_text()))
    source = digest(companion.pilot._sources())
    calibration = {"configuration": cfg, "identity": companion.pilot._identity(cfg, environment),
        "source_identity": source, "reference_backend": "upstream", "reference_optimizations": cfg["optimizations"],
        "reference_input_condition": "uncached", "reference_compiler_selectors": companion.pilot._compiler_selectors(cfg)}
    policy = tmp_path / "pilot.json"; put(policy, {"scope": "CPU fixture; never a measured policy"})
    monkeypatch.setattr(companion.pilot, "validate_pilot_policy", lambda path: ({}, calibration))
    proof = tmp_path / "numerical.json"
    put(proof, {"passed": True, "atol": 0, "rtol": 0, "implementation_sha256": source, "runtime_identity": companion.pilot._runtime(environment),
        "runs": {"upstream": {"model_identity": {}}, "refactored": {"model_identity": {}}}})
    cfg["numerical_validation"] = str(proof)
    kwargs = {"baseline": "upstream", "candidate": "reuse", "candidate_count": 1,
              "pilot_policy": policy, "numerical_proofs": {"upstream": proof, "reuse": proof}}
    return parent, parent_plan, tmp_path / "companion", kwargs


def test_plan_has_own_workload_and_reuses_selection_without_quality_generation(tmp_path, monkeypatch):
    parent, original, output, kwargs = fixture(tmp_path, monkeypatch)
    before = {p.name: file_sha256(p) for p in parent.iterdir() if p.is_file()}
    plan = companion.plan_companion(parent, output, **kwargs)
    assert plan["workload"] == "single_future" and plan["candidate_count"] == 1
    assert len(plan["schedule"]) == 8
    assert all(sum(r["requests"] for r in plan["schedule"] if r["variant"] == name) == 200 for name in ("upstream", "reuse"))
    assert all(cfg["benchmark"]["save_outputs"] == 0 and cfg["benchmark"]["warmup_requests"] == 3 for cfg in plan["configs"].values())
    assert not plan["quality_generated_here"] and not plan["combined_headline_claim"]
    assert original["timing"]["candidate_count"] == 4
    assert before == {p.name: file_sha256(p) for p in parent.iterdir() if p.is_file()}
    assert companion._load(output)[1] == plan
    with pytest.raises(FileExistsError):
        companion.plan_companion(parent, output, **kwargs)


@pytest.mark.parametrize("fault", ["selection", "quality", "proof", "pilot", "plan", "config"])
def test_stale_or_changed_bindings_reject_before_execution(tmp_path, monkeypatch, fault):
    parent, _, output, kwargs = fixture(tmp_path, monkeypatch)
    plan = companion.plan_companion(parent, output, **kwargs)
    if fault == "selection":
        put(parent / "selection.json", {"changed": True})
    elif fault == "quality":
        put(parent / "quality_policy.json", {"changed": True})
    elif fault == "proof":
        put(kwargs["numerical_proofs"]["upstream"], {"changed": True})
    elif fault == "pilot":
        put(kwargs["pilot_policy"], {"changed": True})
    else:
        plan["configs"]["reuse"]["optimizations"]["conditioning_cache_bytes"] = 2048
        put(output / "plan.json", plan)
        if fault == "config":  # Even a newly hashed plan must reconstruct from the selected variant.
            put(output / "plan-receipt.json", {"plan_sha256": file_sha256(output / "plan.json")})
    with pytest.raises((ValueError, KeyError)):
        companion._load(output)


def test_companion_requires_workload_specific_pilot_and_explicit_proofs(tmp_path, monkeypatch):
    parent, _, output, kwargs = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="workload/generation"):
        companion.plan_companion(parent, output, **{**kwargs, "candidate_count": 4})
    with pytest.raises(ValueError, match="explicit numerical"):
        companion.plan_companion(parent, output, **{**kwargs, "numerical_proofs": {}})


def test_prepare_selected_pilot_config_is_nonexecuting_and_immutable(tmp_path, monkeypatch):
    parent, _, _, kwargs = fixture(tmp_path, monkeypatch)
    target = tmp_path / "pilot-workload.yaml"
    result = companion.prepare_pilot_config(parent, target, baseline="upstream", candidate_count=1,
                                            numerical_proof=kwargs["numerical_proofs"]["upstream"])
    cfg = json.loads(target.read_text())
    assert cfg["benchmark"]["candidate_count"] == 1 and cfg["benchmark"]["requests"] == 50
    assert cfg["benchmark"]["save_outputs"] == 0 and not cfg["benchmark"]["headline"]
    assert not result["gpu_execution_performed"] and not result["quality_generated_here"]
    with pytest.raises(FileExistsError):
        companion.prepare_pilot_config(parent, target, baseline="upstream", candidate_count=1,
                                       numerical_proof=kwargs["numerical_proofs"]["upstream"])


@pytest.mark.parametrize("changed", ["policy", "evaluation", "source"])
def test_reviewed_quality_policy_and_calibration_sources_are_explicitly_bound(tmp_path, changed):
    policy, evaluation, run_path = tmp_path / "policy.json", tmp_path / "evaluation.json", tmp_path / "run"
    put(policy, {"fixture": "reviewed metric limits"})
    put(evaluation, {"fixture": "calibration evaluation"})
    for name in ("config.json", "environment.json", "summary.json"):
        put(run_path / name, {})
    (run_path / "measurements.jsonl").write_text("")
    selection = {"reviewed_quality_policies": {"eager": {
        "path": str(policy), "sha256": file_sha256(policy),
        "evaluation_path": str(evaluation), "evaluation_sha256": file_sha256(evaluation),
        "source_runs": [{"path": str(run_path), "artifact_sha256": companion.artifact_sha256(companion.load_run(run_path))}]}}}
    bindings = companion._quality_bindings(selection)
    assert len(bindings) == 6 and bindings[str(run_path / "summary.json")] == file_sha256(run_path / "summary.json")
    target = {"policy": policy, "evaluation": evaluation, "source": run_path / "summary.json"}[changed]
    put(target, {"changed": True})
    with pytest.raises(ValueError, match="reviewed quality"):
        companion._quality_bindings(selection)


def test_runner_executes_only_eight_timing_blocks_and_preserves_no_quality_claim(tmp_path, monkeypatch):
    parent, _, output, kwargs = fixture(tmp_path, monkeypatch)
    companion.plan_companion(parent, output, **kwargs)
    runs, calls = {}, []
    def execute(directory, name, cfg, records):
        assert cfg["benchmark"]["requests"] == 50 and cfg["benchmark"]["candidate_count"] == 1
        assert cfg["benchmark"]["save_outputs"] == 0
        assert len(records) == 128
        path = directory / name
        runs[str(path)] = SimpleNamespace(path=path, config=cfg)
        calls.append((name, cfg["benchmark"]["workload"]))
        return {"path": str(path)}
    monkeypatch.setattr(companion.study, "_execute_run", execute)
    monkeypatch.setattr(companion.study, "_validate_completed_run", lambda path, *args: runs[path])
    monkeypatch.setattr(companion.pilot, "_valid_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(companion.pilot, "_execution_gates", lambda *args: None)
    def aggregate(paths, path):
        assert len(paths) == 4
        path.mkdir(parents=True)
        runs[str(path)] = SimpleNamespace(path=path)
        return runs[str(path)]
    monkeypatch.setattr(companion.study, "aggregate_timing_blocks", aggregate)
    monkeypatch.setattr(companion, "load_run", lambda path: runs[path])
    monkeypatch.setattr(companion, "artifact_sha256", lambda run: {"fixture": str(run.path)})
    monkeypatch.setattr(companion.pilot, "compare_pilot", lambda *args: {"status": "pass", "scope": "CPU fixture"})
    result = companion.run_companion(output)
    assert len(calls) == 8 and all(workload == "single_future" for _, workload in calls)
    assert result["status"] == "complete" and not result["combined_headline_claim"]
    assert not result["quality_generated_here"] and not list(output.rglob("*.npy"))


def replacement_fixture(parent, parent_plan, kwargs, monkeypatch, *, new_pilot=True):
    directory = parent.parent / "replacement"
    old_proof = json.loads(Path(parent_plan["base_config"]["environment_verification"]).read_text())
    environment = copy.deepcopy(old_proof["environment"])
    environment["gpu"].update(uuid="GPU-NEW-CPU-FIXTURE", pci_bus_id="NEW-PCI-FIXTURE")
    proof, lock, overlay = directory / "verification.json", directory / "lock.json", directory / "runtime.yaml"
    put(proof, {"passed": True, "environment": environment})
    put(lock, companion.pilot._runtime(environment))
    paths = {"environment_verification": str(proof), "environment_lock": str(lock)}
    put(overlay, paths)
    numerical = directory / "numerical.json"
    value = json.loads(Path(kwargs["numerical_proofs"]["upstream"]).read_text())
    value["runtime_identity"] = companion.pilot._runtime(environment)
    put(numerical, value)
    if new_pilot:
        policy, calibration = companion.pilot.validate_pilot_policy(kwargs["pilot_policy"])
        calibration = copy.deepcopy(calibration)
        calibration["configuration"].update(paths)
        calibration["identity"] = companion.pilot._identity(calibration["configuration"], environment)
        monkeypatch.setattr(companion.pilot, "validate_pilot_policy", lambda path: (policy, calibration))
    return overlay, paths, {**kwargs, "runtime_config": overlay, "numerical_proofs": {"upstream": numerical, "reuse": numerical}}


def test_replacement_timing_allocation_preserves_parent_and_binds_full_new_identity(tmp_path, monkeypatch):
    parent, parent_plan, output, kwargs = fixture(tmp_path, monkeypatch)
    before = {str(p): file_sha256(p) for p in parent.iterdir() if p.is_file()}
    overlay, paths, changed = replacement_fixture(parent, parent_plan, kwargs, monkeypatch)
    target = tmp_path / "replacement-pilot.yaml"
    receipt = companion.prepare_pilot_config(parent, target, baseline="upstream", candidate_count=1,
        numerical_proof=changed["numerical_proofs"]["upstream"], runtime_config=overlay)
    cfg = json.loads(target.read_text())
    assert cfg["environment_verification"] == paths["environment_verification"]
    assert "study_runtime_overlay" not in cfg and "quality only" not in cfg["timing_runtime_overlay"]["scope"]
    assert "Never merge timing blocks across GPU UUIDs" in cfg["timing_runtime_overlay"]["scope"]
    plan = companion.plan_companion(parent, output, **changed)
    assert plan["timing_runtime_overlay"] == receipt["timing_runtime_overlay"]
    assert len(plan["timing_runtime_overlay"]["bound_files_sha256"]) == 5
    assert all(c["environment_lock"] == paths["environment_lock"] for c in plan["configs"].values())
    assert companion._load(output)[1] == plan
    assert before == {str(p): file_sha256(p) for p in parent.iterdir() if p.is_file()}


@pytest.mark.parametrize("fault", ["old_pilot", "old_numerical", "old_parent_without_overlay"])
def test_mixed_allocation_pilot_or_numerical_evidence_rejects(tmp_path, monkeypatch, fault):
    parent, parent_plan, output, kwargs = fixture(tmp_path, monkeypatch)
    overlay, _, changed = replacement_fixture(parent, parent_plan, kwargs, monkeypatch, new_pilot=fault != "old_pilot")
    if fault == "old_numerical": changed["numerical_proofs"] = kwargs["numerical_proofs"]
    if fault == "old_parent_without_overlay": changed.pop("runtime_config")
    with pytest.raises(ValueError, match="runtime|identity"):
        companion.plan_companion(parent, output, **changed)
    assert not output.exists()
    with pytest.raises(ValueError, match="runtime|identity"):
        companion.prepare_pilot_config(parent, tmp_path / "bad-pilot.yaml", baseline="upstream", candidate_count=1,
            numerical_proof=kwargs["numerical_proofs"]["upstream"], runtime_config=overlay)


@pytest.mark.parametrize("changed_file", ["original_proof", "original_lock", "new_proof", "new_lock", "overlay"])
def test_overlay_revalidates_original_and_replacement_artifacts(tmp_path, monkeypatch, changed_file):
    parent, parent_plan, output, kwargs = fixture(tmp_path, monkeypatch)
    overlay, paths, changed = replacement_fixture(parent, parent_plan, kwargs, monkeypatch)
    companion.plan_companion(parent, output, **changed)
    selected = {"original_proof": parent_plan["base_config"]["environment_verification"],
        "original_lock": parent_plan["base_config"]["environment_lock"],
        "new_proof": paths["environment_verification"], "new_lock": paths["environment_lock"], "overlay": overlay}[changed_file]
    value = json.loads(Path(selected).read_text()); value["changed_after_plan"] = True; put(Path(selected), value)
    with pytest.raises(ValueError): companion._load(output)


@pytest.mark.parametrize("fault", ["packages", "gpu_model", "extra_config"])
def test_overlay_allows_only_gpu_uuid_and_pci_replacement(tmp_path, monkeypatch, fault):
    parent, parent_plan, output, kwargs = fixture(tmp_path, monkeypatch)
    overlay, paths, changed = replacement_fixture(parent, parent_plan, kwargs, monkeypatch)
    if fault == "extra_config":
        value = json.loads(overlay.read_text()); value["generation"] = {"steps": 1}; put(overlay, value)
    else:
        proof = json.loads(Path(paths["environment_verification"]).read_text())
        if fault == "packages": proof["environment"]["packages"]["torch"] = "changed"
        else: proof["environment"]["gpu"]["name"] = "different model"
        put(Path(paths["environment_verification"]), proof)
        put(Path(paths["environment_lock"]), companion.pilot._runtime(proof["environment"]))
    with pytest.raises(ValueError): companion.plan_companion(parent, output, **changed)


def test_replacement_companion_rejects_changed_helper_and_policy(tmp_path, monkeypatch):
    parent, parent_plan, output, kwargs = fixture(tmp_path, monkeypatch)
    _, _, changed = replacement_fixture(parent, parent_plan, kwargs, monkeypatch)
    plan = companion.plan_companion(parent, output, **changed)
    original = copy.deepcopy(plan)
    plan["helper_sha256"] = "0" * 64
    put(output / "plan.json", plan)
    put(output / "plan-receipt.json", {"plan_sha256": file_sha256(output / "plan.json")})
    with pytest.raises(ValueError, match="helper changed"): companion._load(output)
    put(output / "plan.json", original)
    put(output / "plan-receipt.json", {"plan_sha256": file_sha256(output / "plan.json")})
    put(kwargs["pilot_policy"], {"changed": True})
    with pytest.raises(ValueError, match="pilot or source"): companion._load(output)


def semantic_fixture(tmp_path, monkeypatch):
    from test_numerical_semantics import proof_fixture, semantics
    parent, parent_plan, output, kwargs = fixture(tmp_path, monkeypatch)
    arrays = tmp_path / "arrays"; arrays.mkdir()
    proof_path, proof, sidecar = proof_fixture(arrays)
    # The orchestration fixture uses one exact pilot proof and a separate
    # nonzero candidate proof, so it does not fabricate a measured pilot.
    original = json.loads(Path(kwargs["numerical_proofs"]["upstream"]).read_text())
    proof["runtime_identity"] = original["runtime_identity"]
    for name in ("upstream", "refactored"):
        proof["runs"][name]["model_identity"] = {}
    put(proof_path, proof)
    semantics.create(proof_path, sidecar)
    kwargs["numerical_proofs"]["reuse"] = proof_path
    return parent, output, kwargs, proof_path, sidecar


def test_nonzero_proof_requires_real_sidecar_before_preparation_and_plan(tmp_path, monkeypatch):
    parent, output, kwargs, proof, sidecar = semantic_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="sidecar"):
        companion.prepare_pilot_config(parent, tmp_path / "pilot.yaml", baseline="reuse", candidate_count=1, numerical_proof=proof)
    with pytest.raises(ValueError, match="sidecar"):
        companion.plan_companion(parent, output, **kwargs)
    assert not output.exists()
    receipt = companion.prepare_pilot_config(parent, tmp_path / "pilot.yaml", baseline="reuse", candidate_count=1,
        numerical_proof=proof, semantic_sidecar=sidecar)
    assert receipt["exact_semantics"]["sha256"] == file_sha256(sidecar)
    plan = companion.plan_companion(parent, output, **kwargs, semantic_sidecars={"reuse": sidecar})
    assert plan["exact_semantics"]["reuse"] == receipt["exact_semantics"]
    assert companion._load(output)[1] == plan


@pytest.mark.parametrize("fault", ["sidecar", "array", "missing_sidecar"])
def test_authoritative_load_rechecks_semantics_before_run_aggregate_and_report(tmp_path, monkeypatch, fault):
    from test_numerical_semantics import change_array
    parent, output, kwargs, proof, sidecar = semantic_fixture(tmp_path, monkeypatch)
    plan = companion.plan_companion(parent, output, **kwargs, semantic_sidecars={"reuse": sidecar})
    if fault == "sidecar":
        value = json.loads(sidecar.read_text()); value["checks"] = {}; put(sidecar, value)
    elif fault == "array":
        change_array(json.loads(proof.read_text()), "cached", "normal")
    else:
        plan["exact_semantics"] = {}; put(output / "plan.json", plan)
        put(output / "plan-receipt.json", {"plan_sha256": file_sha256(output / "plan.json")})
    with pytest.raises(ValueError): companion._load(output)
    monkeypatch.setattr(companion.study, "_execute_run", lambda *a: pytest.fail("GPU execution must not start"))
    with pytest.raises(ValueError): companion.run_companion(output)


def test_guarded_pilot_checks_before_and_after_execution_and_freeze(tmp_path, monkeypatch):
    parent, _, _, proof, sidecar = semantic_fixture(tmp_path, monkeypatch)
    config, directory = tmp_path / "pilot.yaml", tmp_path / "guarded-pilot"
    companion.prepare_pilot_config(parent, config, baseline="reuse", candidate_count=1,
        numerical_proof=proof, semantic_sidecar=sidecar)
    def plan(path, output, **kwargs):
        value = {"configuration": json.loads(path.read_text()), "scope": "CPU orchestration fixture"}
        put(Path(output) / "plan.json", value); return value
    monkeypatch.setattr(companion.pilot, "plan_pilot", plan)
    monkeypatch.setattr(companion.pilot, "run_pilot", lambda *a: {"completed_blocks": 4})
    companion.guarded_pilot_plan(config, directory, relative_tolerance_floor=0)
    companion.guarded_pilot_run(directory)
    def freeze(root, output):
        value = {"pilot_directory": str(root)}; put(Path(output), value); return value
    monkeypatch.setattr(companion.pilot, "freeze_pilot", freeze)
    policy = directory / "policy.json"
    companion.guarded_pilot_freeze(directory, policy)
    assert companion._pilot_policy_semantics(policy, {"configuration": json.loads(config.read_text())})
    value = json.loads(sidecar.read_text()); value["passed"] = False; put(sidecar, value)
    monkeypatch.setattr(companion.pilot, "run_pilot", lambda *a: pytest.fail("Pilot cannot run with stale semantics"))
    with pytest.raises(ValueError): companion.guarded_pilot_run(directory)
    with pytest.raises(ValueError): companion.guarded_pilot_freeze(directory, directory / "policy2.json")


def test_sidecar_changed_during_pilot_prevents_completion_receipt(tmp_path, monkeypatch):
    parent, _, _, proof, sidecar = semantic_fixture(tmp_path, monkeypatch)
    config, directory = tmp_path / "pilot.yaml", tmp_path / "guarded-pilot"
    companion.prepare_pilot_config(parent, config, baseline="reuse", candidate_count=1,
        numerical_proof=proof, semantic_sidecar=sidecar)
    def plan(path, output, **kwargs):
        value = {"configuration": json.loads(path.read_text())}; put(Path(output) / "plan.json", value); return value
    monkeypatch.setattr(companion.pilot, "plan_pilot", plan)
    companion.guarded_pilot_plan(config, directory, relative_tolerance_floor=0)
    def run(*args):
        put(sidecar, {"passed": True}); return {"completed_blocks": 4}
    monkeypatch.setattr(companion.pilot, "run_pilot", run)
    with pytest.raises(ValueError): companion.guarded_pilot_run(directory)
    assert not (directory / "companion-run-guard.json").exists()


def test_nonzero_pilot_policy_without_guard_is_rejected(tmp_path, monkeypatch):
    _, _, _, proof, _ = semantic_fixture(tmp_path, monkeypatch)
    policy = tmp_path / "direct-core-policy.json"
    put(policy, {"pilot_directory": str(tmp_path / "unguarded")})
    with pytest.raises(FileNotFoundError):
        companion._pilot_policy_semantics(policy, {"configuration": {"numerical_validation": str(proof)}})
