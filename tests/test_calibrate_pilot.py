"""CPU-only artifact fixtures; fabricated timings are never GPU evidence."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

import drivebench.calibrate_pilot as pilot
from drivebench.config import digest, file_sha256
from drivebench.data.manifests import write_manifest
from drivebench.evaluation.io import load_run


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "_now", lambda: "2026-01-01T00:00:00+00:00")
    records = [{"request_id": f"d{i}", "scene_id": f"ds{i}"} for i in range(32)]
    manifest = tmp_path / "development.jsonl"
    write_manifest(manifest, records, {"partition_role": "development"})
    environment = {"gpu": {"uuid": "GPU-CPU-FIXTURE", "name": "CPU fixture, never a real measurement"},
                   "torch": {"version": "fixture"}, "python": "fixture", "packages": {"torch": "fixture"},
                   "tools": {"ncu": {"stdout": "fixture"}}, "cuda_runtime_environment": {"setting": "fixture"}}
    verify, numerical, lock = (tmp_path / name for name in ("verify.json", "numerical.json", "lock.json"))
    put(verify, {"passed": True, "environment": environment})
    put(lock, pilot._runtime(environment))
    put(numerical, {"passed": True, "implementation_sha256": digest(pilot._sources()),
                    "runtime_identity": pilot._runtime(environment)})
    config = {"model": {"backend": "upstream", "upstream_commit": "a" * 40, "checkpoint_sha256": "b" * 64},
              "generation": {"total_frames": 4, "conditioning_frames": 1, "height": 8, "width": 8, "steps": 50},
              "optimizations": {}, "data": {"manifest": str(manifest)},
              "environment_verification": str(verify), "environment_lock": str(lock), "numerical_validation": str(numerical),
              "benchmark": {"workload": "candidate_set", "candidate_count": 4, "input_condition": "uncached"}}
    config_path = tmp_path / "config.json"
    put(config_path, config)
    directory = tmp_path / "pilot"
    plan = pilot.plan_pilot(config_path, directory, relative_tolerance_floor=.02)
    return directory, plan, environment


def write_run(path, plan, environment, *, block=None, scale=1., at=None, candidate=False):
    cfg = copy.deepcopy(plan["configuration"])
    count = 50 if block is not None else 200
    prefix = "d" if block is not None else "e"
    scene_count = 32 if block is not None else 128
    records = [{"request_id": f"{prefix}{i}", "scene_id": f"{prefix}s{i}"} for i in range(scene_count)]
    if block is None:
        manifest = path.parent / "evaluation.jsonl"
        write_manifest(manifest, records, {"partition_role": "held_out_evaluation"})
        cfg["data"]["manifest"] = str(manifest)
    metadata = json.loads(Path(cfg["data"]["manifest"] + ".meta.json").read_text())
    cfg["data"].update(manifest_metadata=metadata, manifest_sha256=metadata["manifest_sha256"])
    cfg["benchmark"].update(requests=count, headline=block is None)
    cfg["started_at"] = at or (f"2026-01-02T{block:02d}:00:01+00:00" if block is not None else "2026-01-04T00:00:00+00:00")
    if candidate:
        cfg["model"]["backend"] = "vista"
        cfg["optimizations"]["conditioning_cache_bytes"] = 1024
        cfg["benchmark"]["input_condition"] = "reused_conditioning"
    rows = []
    for i in range(count):
        index = i % scene_count
        rows.append({"request_id": f"{prefix}{index}", "scene_id": f"{prefix}s{index}", "repeat": i // scene_count,
                     "request_index": i, "status": "ok", "latency_s": float(scale * (1 + index / 1000)),
                     "candidate_count": 4, "cache_stats": {"denoiser_evaluations_complete": True}, "fallbacks": [],
                     "candidates": [{"candidate_id": f"{prefix}{index}:c{k}", "seed": index * 10 + k,
                                     "action_kind": "recorded" if k == 0 else "alternative",
                                     "trajectory_sha256": digest([index, k]), "frames_path": None} for k in range(4)]})
    numerical = {"passed": True, "report_path": cfg["numerical_validation"],
                 "report_sha256": file_sha256(cfg["numerical_validation"])}
    cold = {"status": "ready", "compilation_and_initialization_excluded": True,
            "warmups": [{"status": "ok", "compiler": {}, "cache_stats": {}} for _ in range(3)]}
    for name, value in (("config.json", cfg), ("environment.json", environment),
                        ("summary.json", {"backend": cfg["model"]["backend"], "headline_eligible": block is None,
                                          "failures": 0, "numerical_validation": numerical}),
                        ("cold_start.json", cold), ("numerical_gate.json", numerical), ("environment_gate.json", {"passed": True})):
        put(path / name, value)
    (path / "measurements.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def fixture_blocks(tmp_path, monkeypatch):
    directory, plan, environment = fixture_plan(tmp_path, monkeypatch)
    for index in range(4):
        path = write_run(directory / "blocks" / f"{index:02d}" / "run", plan, environment, block=index, scale=1 + index * .003)
        receipt = {"path": str(path), "block": index, "exit_code": 0, "source_unchanged": True,
                   "source_identity": plan["source_identity"], "plan_sha256": file_sha256(directory / "plan.json"),
                   "started_at": f"2026-01-02T{index:02d}:00:00+00:00", "finished_at": f"2026-01-02T{index:02d}:01:00+00:00",
                   "artifact_sha256": pilot._run_artifacts(load_run(path))}
        put(path.parent / "completed.json", receipt)
    monkeypatch.setattr(pilot, "_now", lambda: "2026-01-03T00:00:00+00:00")
    return directory, plan, environment


def test_pilot_has_200_development_samples_without_headline_relabel(tmp_path, monkeypatch):
    directory, plan, _ = fixture_blocks(tmp_path, monkeypatch)
    output = tmp_path / "policy.json"
    policy = pilot.freeze_pilot(directory, output)
    assert policy["policy_type"] == pilot.TYPE
    assert not policy["pilot_headline_eligible"] and not policy["quality_preserving"]
    assert len(policy["evidence"]["comparisons"]) == 6
    assert all(row["paired_requests"] == 50 for row in policy["evidence"]["comparisons"])
    assert policy["relative_latency_tolerance"] == .02
    assert 0 < policy["evidence"]["measured_variability_bound"] < .02
    assert policy["evidence"]["relative_median_range"] == pytest.approx(.009)
    assert pilot.validate_pilot_policy(output)[0] == policy
    with pytest.raises(FileExistsError):
        pilot.freeze_pilot(directory, output)
    assert plan["configuration"]["benchmark"]["warmup_requests"] == 3


@pytest.mark.parametrize("fault", ["uuid", "source", "warmup", "oom", "actions", "order", "copied", "overlap", "headline", "policyfloor"])
def test_pilot_rejects_invalid_blocks(tmp_path, monkeypatch, fault):
    directory, plan, _ = fixture_blocks(tmp_path, monkeypatch)
    path = directory / "blocks/01/run"
    receipt_path = path.parent / "completed.json"
    receipt = json.loads(receipt_path.read_text())
    if fault == "source":
        receipt["source_identity"] = "bad"
    elif fault == "overlap":
        receipt["started_at"] = "2026-01-02T00:00:00+00:00"
    elif fault == "policyfloor":
        plan["predeclared_floor"] = .5
        put(directory / "plan.json", plan)  # Existing process receipts reject a revised floor.
    else:
        if fault == "uuid":
            obj = json.loads((path / "environment.json").read_text()); obj["gpu"]["uuid"] = "other"
            put(path / "environment.json", obj)
        elif fault == "warmup":
            obj = json.loads((path / "cold_start.json").read_text()); obj["warmups"] = obj["warmups"][:2]
            put(path / "cold_start.json", obj)
        elif fault == "headline":
            obj = json.loads((path / "summary.json").read_text()); obj["headline_eligible"] = True
            put(path / "summary.json", obj)
        else:
            rows = [json.loads(line) for line in (path / "measurements.jsonl").read_text().splitlines()]
            if fault == "oom": rows[0]["cache_stats"]["oom_attempts"] = ["failed and retried"]
            elif fault == "actions": rows[0]["candidates"][0]["seed"] += 1
            elif fault == "order": rows.reverse()
            elif fault == "copied": rows = [json.loads(line) for line in (directory / "blocks/00/run/measurements.jsonl").read_text().splitlines()]
            (path / "measurements.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        receipt["artifact_sha256"] = pilot._run_artifacts(load_run(path))
    put(receipt_path, receipt)
    with pytest.raises(ValueError):
        pilot.freeze_pilot(directory, tmp_path / "bad.json")


def test_headline_same_machine_scope_and_confirmatory_rerun(tmp_path, monkeypatch):
    directory, plan, environment = fixture_blocks(tmp_path, monkeypatch)
    policy = tmp_path / "policy.json"; pilot.freeze_pilot(directory, policy)
    before = write_run(tmp_path / "baseline", plan, environment)
    after = write_run(tmp_path / "candidate", plan, environment, scale=1.2, candidate=True)
    result = pilot.compare_pilot(policy, before, after)
    assert result["status"] == "rerun_required"
    assert result["paired_requests"] == 200 and not result["quality_preserving"]
    confirm_before = write_run(tmp_path / "confirm-before", plan, environment, scale=1.001)
    confirm_after = write_run(tmp_path / "confirm-after", plan, environment, scale=1.201, candidate=True)
    assert pilot.compare_pilot(policy, before, after, confirm_baseline=confirm_before,
                               confirm_candidate=confirm_after)["status"] == "regression_confirmed"
    with pytest.raises(ValueError, match="independently"):
        pilot.compare_pilot(policy, before, after, confirm_baseline=before, confirm_candidate=after)
    changed = copy.deepcopy(environment); changed["gpu"]["uuid"] = "different-GPU"
    put(after / "environment.json", changed)
    with pytest.raises(ValueError, match="runtime"):
        pilot.compare_pilot(policy, before, after)


@pytest.mark.parametrize("fault", ["beforefreeze", "same_scenes", "changedpolicy", "changedsource", "fewrequests"])
def test_policy_cannot_validate_wrong_headline_scope(tmp_path, monkeypatch, fault):
    directory, plan, environment = fixture_blocks(tmp_path, monkeypatch)
    policy_path = tmp_path / "policy.json"; policy = pilot.freeze_pilot(directory, policy_path)
    before = write_run(tmp_path / "baseline", plan, environment)
    after = write_run(tmp_path / "candidate", plan, environment, scale=1.1, candidate=True)
    if fault == "changedpolicy":
        policy["relative_latency_tolerance"] = .5; put(policy_path, policy)
    elif fault == "changedsource":
        path = directory / "source-snapshot" / next(iter(plan["source_files"]))
        path.write_text("changed")
    elif fault == "same_scenes":
        rows = [json.loads(line) for line in (after / "measurements.jsonl").read_text().splitlines()]
        rows[0]["scene_id"] = "ds0"
        (after / "measurements.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    else:
        cfg = json.loads((after / "config.json").read_text())
        if fault == "beforefreeze": cfg["started_at"] = "2026-01-02T20:00:00+00:00"
        if fault == "fewrequests": cfg["benchmark"]["requests"] = 50
        put(after / "config.json", cfg)
    with pytest.raises(ValueError):
        pilot.compare_pilot(policy_path, before, after)


def test_study_block_aggregate_is_verified_down_to_original_observations(tmp_path, monkeypatch):
    from drivebench.study import aggregate_timing_blocks
    directory, plan, environment = fixture_blocks(tmp_path, monkeypatch)
    policy = tmp_path / "policy.json"; pilot.freeze_pilot(directory, policy)
    aggregates = []
    for variant, scale in (("baseline", 1.), ("candidate", .9)):
        paths = []
        for index in range(4):
            path = write_run(tmp_path / variant / f"block-{index}", plan, environment,
                             scale=scale + index * .001, candidate=variant == "candidate")
            cfg = json.loads((path / "config.json").read_text()); cfg["benchmark"].update(requests=50, headline=False)
            put(path / "config.json", cfg)
            summary = json.loads((path / "summary.json").read_text()); summary["headline_eligible"] = False
            put(path / "summary.json", summary)
            rows = (path / "measurements.jsonl").read_text().splitlines()[:50]
            (path / "measurements.jsonl").write_text("\n".join(rows) + "\n")
            paths.append(path)
        aggregates.append(aggregate_timing_blocks(paths, tmp_path / f"aggregate-{variant}").path)
    result = pilot.compare_pilot(policy, *aggregates)
    assert result["status"] == "pass" and result["paired_requests"] == 200
    assert result["independent_scenes"] == 50
    path = aggregates[1] / "measurements.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["latency_s"] *= .5
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="observations differ"):
        pilot.compare_pilot(policy, *aggregates)


def test_compiler_implementation_is_bound_separately_from_comparison_identity(tmp_path, monkeypatch):
    _, plan, environment = fixture_plan(tmp_path, monkeypatch)
    eager = plan["configuration"]
    compiled = copy.deepcopy(eager)
    compiled["model"].update(compiler_runtime="registered_einops_attention_v1",
                             compiler_feature_boundary="output_blocks.9")
    assert pilot._identity(eager, environment) == pilot._identity(compiled, environment)
    assert plan["reference_compiler_selectors"] == {"compiler_runtime": None, "compiler_feature_boundary": None}
    assert pilot._compiler_selectors(compiled) != plan["reference_compiler_selectors"]
    # A combined proof's candidate settings must never contaminate its eager
    # upstream reference identity.
    proof = {"config": compiled, "runs": {"upstream": {"model_identity": {}},
             "refactored": {"model_identity": {"compiler_runtime": {"name": "registered_einops_attention_v1"}}}}}
    pilot._proof_compiler_identity(eager, proof, {})
    historical = write_run(tmp_path / "historical-eager", plan, environment)
    put(historical / "model_identity.json", {})
    assert "model_identity.json" not in pilot._run_artifacts(load_run(historical))


@pytest.mark.parametrize("fault", [None, "selector", "identity_hash", "missing_identity", "boundary"])
def test_execution_gate_binds_own_compiler_proof_and_saved_identity(tmp_path, monkeypatch, fault):
    _, plan, environment = fixture_plan(tmp_path, monkeypatch)
    path = write_run(tmp_path / "compiled", plan, environment, candidate=True)
    cfg = json.loads((path / "config.json").read_text())
    cfg["model"].update(compiler_runtime="registered_einops_attention_v1",
                         compiler_feature_boundary="output_blocks.9")
    identity = {"compiler_runtime": {"name": "registered_einops_attention_v1", "implementation_sha256": "c" * 64},
                "compiler_feature_boundary": {"module": "output_blocks.9", "source_sha256": "d" * 64}}
    numerical = json.loads(Path(cfg["numerical_validation"]).read_text())
    numerical["runs"] = {"upstream": {"model_identity": {}}, "refactored": {"model_identity": copy.deepcopy(identity)}}
    proof = path / "runtime-proof.json"
    if fault == "selector":
        cfg["model"]["compiler_runtime"] = "different-runtime"
    if fault == "identity_hash":
        identity["compiler_runtime"]["implementation_sha256"] = "e" * 64
    if fault == "boundary":
        numerical["runs"]["refactored"]["model_identity"]["compiler_feature_boundary"]["module"] = "middle_block"
    put(proof, numerical)
    cfg["numerical_validation"] = str(proof)
    put(path / "config.json", cfg)
    put(path / "numerical_gate.json", {"passed": True, "report_path": str(proof), "report_sha256": file_sha256(proof)})
    if fault != "missing_identity":
        put(path / "model_identity.json", identity)
    if fault is None:
        pilot._execution_gates(load_run(path), plan)
        assert "model_identity.json" in pilot._run_artifacts(load_run(path))
    else:
        with pytest.raises(ValueError, match="compiler"):
            pilot._execution_gates(load_run(path), plan)


@pytest.mark.parametrize("selector", pilot.COMPILER_SELECTORS)
def test_headline_reference_cannot_change_compiler_selector(tmp_path, monkeypatch, selector):
    from types import SimpleNamespace
    _, plan, _ = fixture_plan(tmp_path, monkeypatch)
    cfg = copy.deepcopy(plan["configuration"])
    cfg["started_at"] = "2026-01-04T00:00:00+00:00"
    cfg["model"][selector] = "changed"
    run = SimpleNamespace(config=cfg, summary={"headline_eligible": True})
    monkeypatch.setattr(pilot, "_valid_run", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="reference implementation"):
        pilot._headline(run, {"frozen_at": "2026-01-03T00:00:00+00:00"}, plan, reference=True)


@pytest.mark.parametrize("selector", pilot.COMPILER_SELECTORS)
@pytest.mark.parametrize("side", ["confirm-baseline", "confirm-candidate"])
def test_confirmation_cannot_substitute_compiler_implementation(tmp_path, monkeypatch, selector, side):
    from types import SimpleNamespace
    runs = {name: SimpleNamespace(path=Path(name), config={"model": {"backend": "vista"}, "optimizations": {},
            "benchmark": {"input_condition": "uncached"}}) for name in ("baseline", "candidate", "confirm-baseline", "confirm-candidate")}
    runs[side].config["model"][selector] = "changed"
    policy = tmp_path / "policy.json"; put(policy, {})
    monkeypatch.setattr(pilot, "validate_pilot_policy", lambda path: ({}, {}))
    monkeypatch.setattr(pilot, "load_run", lambda path: runs[path])
    monkeypatch.setattr(pilot, "timing_signature", lambda run: str(run.path))
    monkeypatch.setattr(pilot, "_comparison", lambda *args: {"status": "rerun_required"})
    with pytest.raises(ValueError, match="Confirmatory implementation"):
        pilot.compare_pilot(policy, "baseline", "candidate", confirm_baseline="confirm-baseline", confirm_candidate="confirm-candidate")
