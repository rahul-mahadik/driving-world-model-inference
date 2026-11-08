"""Synthetic CPU artifacts only; no GPU execution or real visual votes."""
import copy
import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location("norm_report", Path(__file__).resolve().parents[1] / "scripts/report_norm_timing.py")
report = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(report)
old = report.module("report_full_study")


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def native_fixture(tmp_path):
    root = tmp_path / "bundle"; public = root / "public"; public.mkdir(parents=True)
    cases = []
    for i in range(4):
        key = "case-" + f"{i:024x}"
        frames = []
        for frame in range(1, 25):
            name = f"{key}-frame-{frame:04d}.png"
            (public / name).write_bytes(f"synthetic pixel fixture {i} {frame}".encode()); frames.append(name)
        page = key + "-page-1.html"; (public / page).write_text("synthetic native frame page")
        cases.append({"case_id": key, "future_frames": 24, "panel_width": 1024, "panel_height": 576,
                      "frames": frames, "pages": [page]})
    (public / "index.html").write_text("synthetic anonymous review index")
    (public / "review.csv").write_text(",".join(report.FIELDS) + "\n")
    put(public / "manifest.json", {"receipt_type": "blind_native_frame_export_v1", "cases": cases,
        "frame_sha256": {name: report.file_sha256(public / name) for case in cases for name in case["frames"]}})
    index = tmp_path / "original-review/index.html"; index.parent.mkdir(); index.write_text("anonymous original index")
    blank = index.with_name("review.csv"); blank.write_text((public / "review.csv").read_text())
    key = put(tmp_path / "private_key.json", {"cases": [{"case_id": c["case_id"], "run": "compiled", "reference_run": "upstream",
                "A": "compiled", "B": "upstream"} for c in cases]})
    metric = put(tmp_path / "metric.json", {"review": {"index": str(index), "csv": str(blank), "private_key": str(key), "cases": 4}})
    review_files = {str(p): report.file_sha256(p) for p in (index, blank, key)}
    metric_receipt = put(tmp_path / "metric.receipt.json", {"evaluation_sha256": report.file_sha256(metric), "review_artifact_sha256": review_files})
    private = put(root / "private-receipt.json", {"receipt_type": "blind_native_frame_export_v1", "created_at": "2026-01-02T00:00:00+00:00",
        "binding": {"metric": str(metric), "metric_sha256": report.file_sha256(metric), "metric_receipt": str(metric_receipt),
            "metric_receipt_sha256": report.file_sha256(metric_receipt), "review_artifact_sha256": review_files,
            "source_arrays_sha256": {"retired.npy": "a" * 64},
            "wrapper_sha256": report.file_sha256(Path(report.__file__).with_name("run_quality_bounded.py"))},
        "public_artifact_sha256": report.public_inventory(public)})
    ratings = tmp_path / "votes.csv"
    with ratings.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=report.FIELDS); writer.writeheader()
        for case in cases:
            writer.writerow({**dict.fromkeys(report.FIELDS, ""), "case_id": case["case_id"],
                "preferred_panel_A_B_tie": "tie", "acceptable_A": "yes", "acceptable_B": "yes",
                "reviewer_notes": "Synthetic CPU fixture, no visual assessment"})
    protocol = put(tmp_path / "protocol.json", {"created_at": "2026-01-01T00:00:00+00:00"})
    seal = tmp_path / "sealed.json"
    report.seal_review(public, ratings, protocol, seal,
        original_index_sha256=report.file_sha256(index), started_at="2026-01-03T00:00:00+00:00")
    pin = tmp_path / "pin.json"; report.pin_seal(seal, report.file_sha256(seal), pin)
    item = {"metric": str(metric), "metric_sha256": report.file_sha256(metric), "native_receipt": str(private),
        "native_receipt_sha256": report.file_sha256(private), "seal": str(seal), "seal_sha256": report.file_sha256(seal),
        "pin": str(pin), "pin_sha256": report.file_sha256(pin)}
    registry = put(tmp_path / "registry.json", {"schema_version": 1, "receipt_type": report.REGISTRY, "reviews": [item]})
    return SimpleNamespace(public=public, index=index, key=key, metric=metric, private=private, ratings=ratings,
        protocol=protocol, seal=seal, pin=pin, item=item, registry=registry, cases=cases,
        expected={metric: ("development", report.file_sha256(metric))})


def test_seal_and_external_pin_are_immutable(tmp_path):
    fx = native_fixture(tmp_path)
    assert report.read(fx.seal)["reviewed_frames_per_case"] == {c["case_id"]: 24 for c in fx.cases}
    with pytest.raises(FileExistsError):
        report.pin_seal(fx.seal, report.file_sha256(fx.seal), fx.pin)
    with pytest.raises(ValueError, match="changed"):
        report.pin_seal(fx.seal, "0" * 64, tmp_path / "bad-pin.json")


def test_registry_verifies_all_public_frames_without_parsing_private_keys(tmp_path, monkeypatch):
    fx = native_fixture(tmp_path); evidence = old.Evidence(tmp_path)
    original = evidence.read
    def read(path):
        assert evidence.path(path) != fx.key, "Private key read before seal validation"
        return original(path)
    monkeypatch.setattr(evidence, "read", read)
    ratings, bindings = report.verify_review_registry(evidence, fx.registry, fx.expected)
    assert ratings["reviewer_kind"] == "agent_assisted"
    assert ratings["reviews"][0]["csv"] == str(fx.ratings)
    assert len([p for p in bindings if p.endswith(".png")]) == 96
    assert str(fx.key) not in bindings


@pytest.mark.parametrize("mutation", ["frame", "missing_frame", "ratings", "seal", "pin", "metric", "native_receipt", "protocol"])
def test_changed_review_evidence_rejected(tmp_path, mutation):
    fx = native_fixture(tmp_path)
    path = {"frame": fx.public / fx.cases[0]["frames"][0], "missing_frame": fx.public / fx.cases[0]["frames"][0],
            "ratings": fx.ratings, "seal": fx.seal, "pin": fx.pin, "metric": fx.metric,
            "native_receipt": fx.private, "protocol": fx.protocol}[mutation]
    path.unlink() if mutation == "missing_frame" else path.write_text("changed evidence")
    with pytest.raises((ValueError, OSError, KeyError)):
        report.verify_review_registry(old.Evidence(tmp_path), fx.registry, fx.expected)


def repin(fx):
    fx.item["seal_sha256"] = report.file_sha256(fx.seal)
    pin = report.read(fx.pin); pin["seal"]["sha256"] = fx.item["seal_sha256"]
    put(fx.pin, pin); fx.item["pin_sha256"] = report.file_sha256(fx.pin)
    put(fx.registry, {"schema_version": 1, "receipt_type": report.REGISTRY, "reviews": [fx.item]})


@pytest.mark.parametrize("field,value", [("private_keys_read_before_seal", True), ("all_panels_reviewed", False),
    ("review_started_at", "2026-01-01T12:00:00+00:00"), ("created_at", "2099-01-01T00:00:00+00:00")])
def test_rehashed_invalid_chronology_or_blinding_rejected(tmp_path, field, value):
    fx = native_fixture(tmp_path); seal = report.read(fx.seal); seal[field] = value; put(fx.seal, seal); repin(fx)
    with pytest.raises(ValueError):
        report.verify_review_registry(old.Evidence(tmp_path), fx.registry, fx.expected)


def test_duplicate_or_wrong_comparison_review_rejected(tmp_path):
    fx = native_fixture(tmp_path)
    registry = report.read(fx.registry); registry["reviews"].append(fx.item); put(fx.registry, registry)
    with pytest.raises(ValueError, match="exactly cover"):
        report.verify_review_registry(old.Evidence(tmp_path), fx.registry, fx.expected)


def action_fixture():
    pairs = [{"request_id": "request", "scene_id": "scene", "candidate_id": name, "seed": 7,
        "trajectory_sha256": action, "reference_output_sha256": ref, "candidate_output_sha256": cand,
        "action_kind": kind} for name, action, ref, cand, kind in
        [("recorded", "a" * 64, "b" * 64, "c" * 64, "recorded"), ("changed", "d" * 64, "e" * 64, "f" * 64, "alternative")]]
    return {"pairs": pairs, "action_response": [{"request_id": "request", "scene_id": "scene", "seed": 7,
        "recorded_candidate_id": "recorded", "alternative_candidate_id": "changed", "trajectory_sha256": ["a" * 64, "d" * 64],
        "output_sha256": ["b" * 64, "e" * 64, "c" * 64, "f" * 64],
        "metrics": {k: .01 for k in old.ACTION_METRICS}}]}


def test_action_diagnostic_requires_same_seed_distinct_actions_and_four_hashes():
    entry = action_fixture(); assert len(report.action_integrity(entry, 1)) == 1
    for mutation in ("seed", "action", "output", "duplicate", "nonfinite"):
        broken = copy.deepcopy(entry)
        if mutation == "seed": broken["pairs"][1]["seed"] = 8
        if mutation == "action": broken["pairs"][1]["trajectory_sha256"] = "a" * 64
        if mutation == "output": broken["action_response"][0]["output_sha256"].reverse()
        if mutation == "duplicate": broken["action_response"].append(broken["action_response"][0])
        if mutation == "nonfinite": broken["action_response"][0]["metrics"]["response_error_rmse"] = float("nan")
        with pytest.raises(ValueError): report.action_integrity(broken, 1)


def timing_fixture(tmp_path):
    parent = tmp_path / "study"; put(parent / "plan.json", {}); put(parent / "selection.json", {})
    result = {"type": "compiler_norm_timing_result_v1", "timing_qualified": True, "combined_headline_claim": False,
        "parent_study": str(parent), "parent_plan_sha256": report.file_sha256(parent / "plan.json"),
        "selection_sha256": report.file_sha256(parent / "selection.json"), "baseline": "upstream", "candidate": "compiled",
        "legacy_strict_passed": {"upstream": True, "compiled": False}, "legacy_headline_eligible": {"upstream": False, "compiled": False},
        "blocks": [], "candidate_count": 1, "workload": "single_future", "qualification": {"report_sha256": "a" * 64},
        "source_identity": "b" * 64, "plan_sha256": "c" * 64, "pilot_policy": {"sha256": "d" * 64}, "artifact_sha256": {},
        "comparison": {"status": "pass", "paired_requests": 200, "independent_scenes": 50, "metrics": {
            key: {"relative_latency_change": -.2, "ci95": [-.21, -.19], "tolerance": .02,
                "exceeds_tolerance_and_uncertainty": False, "improvement_exceeds_tolerance_and_uncertainty": True}
            for key in ("p50", "p95")}}}
    for index, variant in enumerate(("upstream", "compiled", "compiled", "upstream", "compiled", "upstream", "upstream", "compiled")):
        block = sum(b["variant"] == variant for b in result["blocks"]); path = tmp_path / "blocks" / str(index)
        put(path / "config.json", {"started_at": "2026-01-04T00:00:00+00:00"})
        put(path / "summary.json", {"headline_eligible": False, "numerical_validation": {"passed": variant == "upstream"}})
        put(path / "environment.json", {"gpu": {"uuid": "GPU-PRIVATE-FIXTURE"}})
        rows = [{"request_id": f"r{i}", "scene_id": f"s{i}", "status": "ok", "latency_s": (10 if variant == "upstream" else 8) + i / 1000,
                 "candidates": [{"candidate_id": "one"}]} for i in range(50)]
        (path / "measurements.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        hashes = {k: report.file_sha256(path / k) for k in old.CORE}
        result["blocks"].append({"index": index, "variant": variant, "variant_block": block, "path": str(path),
            "artifact_sha256": hashes, "receipt_sha256": "e" * 64, "process_sha256": "f" * 64,
            "execution": {"norm_execution_eligible": True}})
        result["artifact_sha256"].update({str(path / k): v for k, v in hashes.items()})
    return parent, result


def test_timing_projection_preserves_raw_flags_and_no_regression_is_not_speedup(tmp_path):
    parent, result = timing_fixture(tmp_path)
    before = {p: report.file_sha256(p) for p in tmp_path.rglob("*") if p.is_file()}
    actual = report.timing_projection(old, old.Evidence(tmp_path), result)
    assert all(r["requests"] == 200 and r["unique_clips"] == 50 for r in actual["runs"])
    assert all(not b["legacy_headline_eligible"] for r in actual["runs"] for b in r["blocks"])
    assert all(b["legacy_numerical_passed"] == (r["variant"] == "upstream") for r in actual["runs"] for b in r["blocks"])
    assert all(b["norm_execution_eligible"] for r in actual["runs"] for b in r["blocks"])
    assert actual["comparison"]["improvement_demonstrated"]
    for value in result["comparison"]["metrics"].values(): value["improvement_exceeds_tolerance_and_uncertainty"] = False
    assert not report.timing_projection(old, old.Evidence(tmp_path), result)["comparison"]["improvement_demonstrated"]
    assert all(report.file_sha256(p) == checksum for p, checksum in before.items())


def patch_build(monkeypatch, result):
    original = report.module
    monkeypatch.setattr(report, "module", lambda name: SimpleNamespace(verify_completed=lambda p: copy.deepcopy(result))
                        if name == "run_norm_timing_companion" else original(name))
    monkeypatch.setattr(report.study_module, "_selection", lambda *args: {})


def test_missing_quality_never_promotes_verified_timing_and_public_excludes_private_data(tmp_path, monkeypatch):
    parent, result = timing_fixture(tmp_path); patch_build(monkeypatch, result)
    monkeypatch.setattr(report, "phase_quality", lambda *a: pytest.fail("No review means no key attribution"))
    data = report.build(parent, {"candidate4": tmp_path / "unused"}, tmp_path / "report", workspace_root=tmp_path)
    assert not data["headline_claims"] and data["companions"][0]["status"] == "timing_verified_quality_missing_or_invalid"
    assert data["companions"][0]["legacy_strict_passed"]["candidate"] is False
    assert not list((tmp_path / "report").glob("attribution-start-*.json"))
    for path in (tmp_path / "report/public").iterdir():
        text = path.read_text(); assert str(tmp_path) not in text and "GPU-PRIVATE-FIXTURE" not in text


def test_tampered_companion_binding_stays_ineligible(tmp_path, monkeypatch):
    parent, result = timing_fixture(tmp_path); result["parent_plan_sha256"] = "0" * 64; patch_build(monkeypatch, result)
    data = report.build(parent, {"single": tmp_path / "unused"}, tmp_path / "report", workspace_root=tmp_path)
    assert not data["headline_claims"] and data["companions"][0]["timing"] is None


def test_all_quality_phases_and_seals_needed_before_combined_claim(tmp_path, monkeypatch):
    parent, result = timing_fixture(tmp_path)
    setup = put(tmp_path / "setup.json", {"fixture": True})
    put(parent / "plan.json", {"quality": {"evaluation_setup_receipt": str(setup)}})
    result["parent_plan_sha256"] = report.file_sha256(parent / "plan.json")
    patch_build(monkeypatch, result)
    registry = put(tmp_path / "registry.json", {"fixture": True})
    monkeypatch.setattr(report, "expected_metrics", lambda *a: {})
    monkeypatch.setattr(report, "verify_review_registry", lambda *a: ({"reviewer_kind": "agent_assisted"}, {str(registry): report.file_sha256(registry)}))
    monkeypatch.setattr(report, "quality_bindings", lambda *a: {})
    inspected = []
    def phase(*args):
        assert list((tmp_path / "report").glob("attribution-start-*.json")), "Private attribution started before its durable receipt"
        inspected.append(args[3]); return {"phase": args[3], "status": "verified", "accepted": args[3] != "evaluation_actions"}
    monkeypatch.setattr(report, "phase_quality", phase)
    data = report.build(parent, {"single": tmp_path / "unused"}, tmp_path / "report", review_registry=registry, workspace_root=tmp_path)
    assert inspected == list(report.PHASES)
    assert not data["headline_claims"] and not data["companions"][0]["combined_headline_claim"]


def quality_fixture(tmp_path, monkeypatch):
    fx = native_fixture(tmp_path)
    phase = "evaluation"; parent = tmp_path / "study"; folder = parent / "quality" / phase
    metadata = {"partition_role": "held_out_evaluation", "manifest_sha256": "e" * 64,
                "clip_count": 128, "scene_ids": [f"scene-{i}" for i in range(128)]}
    setup = put(tmp_path / "setup.json", {"passed": True, "cpu_verification": {"passed": True, "model_state_dict_sha256": "f" * 64}})
    put(parent / "plan.json", {"manifests": {phase: {"sha256": "e" * 64, "metadata": metadata}},
                              "quality": {"evaluation_setup_receipt": str(setup), "lpips_device": "cuda"}})
    policy = put(parent / "policy.json", {"calibrated": True, "frozen": True, "selected_on": "development",
        "policy_type": "reviewed_metric_tolerances", "frozen_at": "2026-01-01T00:00:00+00:00",
        "development_manifest_sha256": "d" * 64, "development_scene_ids": ["development-scene"],
        "metrics": {"rmse": {"maximum": .01, "aggregation": "max"}, "lpips": {"maximum": .003, "aggregation": "max"},
                    "temporal_derivative_mae": {"maximum": 1 / 255, "aggregation": "max"}}})
    pairs = [{"request_id": f"request-{i}", "scene_id": f"scene-{i}", "candidate_id": "recorded", "seed": i,
              "metrics": {key: .001 for key in old.METRICS}} for i in range(128)]
    configs = {}; hashes = {}; paths = {}
    for name in ("upstream", "compiled"):
        path = folder / "aggregate/runs" / name
        cfg = {"model": {"backend": name}, "generation": {"steps": 50}, "optimizations": {},
            "data": {"manifest_sha256": "e" * 64}, "benchmark": {"candidate_count": 1},
            "started_at": "2026-01-03T00:00:00+00:00"}
        put(path / "config.json", cfg); put(path / "summary.json", {"synthetic": False})
        put(path / "environment.json", {}); (path / "measurements.jsonl").write_text("")
        configs[name] = cfg; paths[name] = str(path); hashes[name] = {k: report.file_sha256(path / k) for k in old.CORE}
    entry = {"run": "compiled", "run_path": paths["compiled"], "artifact_sha256": hashes["compiled"],
        "manifest_metadata": metadata, "pairs": pairs, "matched_pairs": 128, "complete_pairs": True,
        "synthetic": False, "errors": [], "aggregates": report.aggregate_pairs(pairs),
        "metric_artifacts": [{"path": str(fx.metric), "sha256": report.file_sha256(fx.metric)}],
        "quality_policy": {"path": str(policy), "sha256": report.file_sha256(policy), "reference": "upstream"}}
    document = {"reference": paths["upstream"], "reference_artifact_sha256": hashes["upstream"],
                "runs": [entry], "perceptual": {"status": "available", "provenance": {"device": "cuda",
                    "setup_receipt_sha256": report.file_sha256(setup), "state_dict_sha256": "f" * 64}}}
    target = put(folder / "aggregate/evaluation-vs-upstream.json", document)
    put(folder / "shards/000/complete.json", {"metric_artifacts": {str(fx.metric): report.file_sha256(fx.metric)},
        "runs": {n: {"path": paths[n], "artifact_sha256": hashes[n]} for n in configs}})
    put(folder / "aggregation-receipt.json", {"artifact_sha256": {str(target): report.file_sha256(target)}})
    # Runtime/source provenance has its own comprehensive tests in the reused
    # old helper; this fixture exercises actual coverage, policy and vote joins.
    monkeypatch.setattr(old, "quality_implementation_matches", lambda *a: True)
    ratings = {"reviewer_kind": "agent_assisted", "reviews": [{"index": str(fx.index), "index_sha256": report.file_sha256(fx.index),
        "csv": str(fx.ratings), "csv_sha256": report.file_sha256(fx.ratings)}]}
    return fx, parent, folder, target, policy, {"baseline": "upstream", "candidate": "compiled", "configs": configs}, ratings


def test_actual_128_pair_metric_policy_and_vote_join(tmp_path, monkeypatch):
    fx, parent, folder, target, policy, value, ratings = quality_fixture(tmp_path, monkeypatch)
    result = report.phase_quality(old, old.Evidence(tmp_path), parent, "evaluation", 128, value, ratings, {"quality_policy": str(policy)})
    assert result["accepted"] and result["clips"] == 128 and result["accepted_blind_cases"] == 4
    assert result["metrics"]["rmse"]["mean"] == pytest.approx(.001)
    with fx.ratings.open(newline="") as stream: rows = list(csv.DictReader(stream))
    rows[0]["acceptable_A"] = "no"
    with fx.ratings.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=report.FIELDS); writer.writeheader(); writer.writerows(rows)
    ratings["reviews"][0]["csv_sha256"] = report.file_sha256(fx.ratings)
    rejected = report.phase_quality(old, old.Evidence(tmp_path), parent, "evaluation", 128, value, ratings, {"quality_policy": str(policy)})
    assert rejected["frozen_metric_limits_met"] and not rejected["accepted"]


@pytest.mark.parametrize("mutation", ["missing_pair", "synthetic", "late_policy", "changed_runtime", "stale_policy"])
def test_quality_join_rejects_incomplete_stale_or_wrong_semantics(tmp_path, monkeypatch, mutation):
    fx, parent, folder, target, policy, value, ratings = quality_fixture(tmp_path, monkeypatch)
    doc = report.read(target)
    if mutation == "missing_pair": doc["runs"][0]["pairs"].pop()
    if mutation == "synthetic": doc["runs"][0]["synthetic"] = True
    if mutation == "changed_runtime": monkeypatch.setattr(old, "quality_implementation_matches", lambda *a: False)
    if mutation in {"late_policy", "stale_policy"}:
        settings = report.read(policy); settings["frozen_at"] = "2026-02-01T00:00:00+00:00"; put(policy, settings)
        if mutation == "late_policy": doc["runs"][0]["quality_policy"]["sha256"] = report.file_sha256(policy)
    put(target, doc); put(folder / "aggregation-receipt.json", {"artifact_sha256": {str(target): report.file_sha256(target)}})
    if mutation == "late_policy":
        actual = report.phase_quality(old, old.Evidence(tmp_path), parent, "evaluation", 128, value, ratings, {"quality_policy": str(policy)})
        assert not actual["accepted"]
    else:
        with pytest.raises(ValueError):
            report.phase_quality(old, old.Evidence(tmp_path), parent, "evaluation", 128, value, ratings, {"quality_policy": str(policy)})


def test_numerical_projection_never_exports_private_identifiers_or_untyped_values():
    private = "/Users/private-account/secret-proof.npz"
    source = {"report_sha256": private, "numerical": {"contract": {"metric": private, "maximum": private},
        "candidates": {private: {"rmse": private, "max_abs_error": .5, "absolute_error_histogram": {"edges": [0, private, 1],
            "counts": [1, private]}, "per_frame": [{"frame": private, "rmse": .001, "p50_p95_p99_abs_error": [private, .1, .2]}]}}},
        "quality_development": {"reference_source_identities": [private]}, "exact_semantics": {"atol": private}}
    public = json.dumps(report.numerical_projection(old, source))
    assert private not in public and "/Users/" not in public and "private-account" not in public


def test_report_refuses_existing_destination(tmp_path):
    (tmp_path / "report").mkdir()
    with pytest.raises(ValueError, match="fresh"):
        report.build(tmp_path / "study", {}, tmp_path / "report", workspace_root=tmp_path)
