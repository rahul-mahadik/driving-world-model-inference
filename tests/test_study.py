"""Tiny CPU fixtures exercise orchestration; none are GPU performance evidence."""
import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import yaml

from drivebench.config import digest, file_sha256
from drivebench.data.manifests import second_seed_subset, write_manifest
from drivebench.evaluate import tolerance_gate
from drivebench.evaluation.metrics import verify_perceptual_setup
from drivebench.freeze import exact_metrics, freeze_exact_tolerances
from drivebench.study import (create_plan, execute_quality, aggregate_quality, timing_schedule,
    aggregate_timing_blocks, compare_timing_blocks, _execute_run, _variant_config)
from test_evaluation import write_run
from test_freeze import fixture as freeze_fixture


def study_fixture(tmp_path):
    frame = tmp_path / "frame.png"
    Image.fromarray(np.full((8, 8, 3), 64, dtype=np.uint8)).save(frame)
    def records(prefix, size):
        return [{"request_id": f"{prefix}-{i}", "scene_id": f"{prefix}-{i}", "dataset": "synthetic",
                 "observed_frames": ["frame.png"], "future_frames": ["frame.png"] * 3,
                 "frame_checksums": {"frame.png": file_sha256(frame)}, "seeds": [i + 1, i + 1001],
                 "trajectory": [0, 2, 0, 4, 0, 6, 0, 8], "categories": ["fixture"],
                 "trajectory_metadata": {"synthetic": True, "units": "meters", "coordinate_frame": "fixture",
                    "axis_order": ["lateral", "longitudinal"], "timestamps_seconds": [.5, 1, 1.5, 2], "provenance": "CPU fixture"}}
                for i in range(size)]
    development, evaluation = records("dev", 32), records("eval", 128)
    manifests = {}
    for name, rows, role in (("development", development, "development"), ("evaluation", evaluation, "held_out_evaluation"),
                             ("evaluation_second_seed", second_seed_subset(evaluation, 16), "held_out_evaluation_second_seed")):
        path = tmp_path / f"{name}.jsonl"
        write_manifest(path, rows, {"partition_role": role})
        manifests[name] = str(path)
    base = {"model": {"backend": "synthetic"}, "data": {"root": str(tmp_path), "manifest": manifests["development"], "verify_checksums": True},
            "generation": {"total_frames": 4, "conditioning_frames": 1, "height": 8, "width": 8, "steps": 2, "dtype": "fp32"},
            "optimizations": {}, "benchmark": {"requests": 32, "warmup_requests": 0, "headline": False, "input_condition": "uncached"}}
    base_path = tmp_path / "base.yaml"; base_path.write_text(yaml.safe_dump(base))
    spec = {"schema_version": 1, "base_config": str(base_path), "reference": "upstream", "optimized_reference": "optimized",
            "manifests": manifests, "variants": {"upstream": {}, "optimized": {}, "steps-1": {"generation": {"steps": 1}}},
            "quality": {"shard_clips": 16, "lpips": False, "lpips_device": "cpu", "warmups": 0,
                        "retain_raw_clips": 1, "review_pairs_per_shard": 1, "minimum_free_bytes": 0}}
    spec_path = tmp_path / "spec.yaml"; spec_path.write_text(yaml.safe_dump(spec, sort_keys=False))
    destination = tmp_path / "study"
    plan = create_plan(spec_path, destination)
    return destination, plan, spec_path


def test_study_plan_checks_disjoint_and_second_seed_provenance(tmp_path):
    destination, plan, spec_path = study_fixture(tmp_path)
    assert plan["manifests"]["development"]["metadata"]["scene_count"] == 32
    assert plan["manifests"]["evaluation_actions"]["metadata"]["clip_count"] == 16
    with pytest.raises(FileExistsError):
        create_plan(spec_path, destination)
    spec = yaml.safe_load(spec_path.read_text())
    second = Path(spec["manifests"]["evaluation_second_seed"])
    rows = [json.loads(line) for line in second.read_text().splitlines()]
    rows[0]["trajectory"][0] = 100
    write_manifest(second, rows, {"partition_role": "held_out_evaluation_second_seed"})
    with pytest.raises(ValueError, match="preserve parent"):
        create_plan(spec_path, tmp_path / "tampered")


def test_quality_resume_pair_commit_retirement_and_aggregation(tmp_path):
    destination, plan, _ = study_fixture(tmp_path)
    with pytest.raises(FileNotFoundError):
        execute_quality(destination, "evaluation", max_shards=1)
    first = execute_quality(destination, "development", max_shards=1)
    assert first["completed_shards"] == 1
    shard = destination / "quality/development/shards/000"
    proof = json.loads((shard / "complete.json").read_text())
    for receipt in proof["runs"].values():
        assert receipt["outputs"] and all(not Path(row["path"]).exists() for row in receipt["outputs"])
    assert execute_quality(destination, "development", max_shards=1) == first
    with pytest.raises(ValueError, match="incomplete"):
        aggregate_quality(destination, "development")
    assert execute_quality(destination, "development")["completed_shards"] == 2
    result = aggregate_quality(destination, "development")
    assert aggregate_quality(destination, "development") == result
    evaluation = json.loads(Path(result["upstream"]).read_text())
    entries = {entry["run"]: entry for entry in evaluation["runs"]}
    assert entries["optimized"]["matched_pairs"] == 32
    assert entries["optimized"]["aggregates"]["max_abs_error"]["max"] == 0
    assert entries["steps-1"]["aggregates"]["rmse"]["mean"] > 0
    assert all(pair["reference_output_sha256"] and pair["candidate_output_sha256"] for pair in entries["optimized"]["pairs"])
    assert not entries["optimized"]["gate"]["quality_preserving"]
    with pytest.raises(ValueError, match="real Vista"):
        freeze_exact_tolerances(result["upstream"], ["optimized"], destination / "forbidden.yaml")
    metric = next(iter(proof["metric_artifacts"]))
    Path(metric).write_text("{}")
    with pytest.raises(ValueError, match="metrics changed"):
        execute_quality(destination, "development", max_shards=1)


def test_adoption_keeps_external_arrays_and_requires_matching_workload(tmp_path):
    destination, plan, _ = study_fixture(tmp_path)
    manifest = plan["manifests"]["development"]["path"]
    rows = [json.loads(line) for line in Path(manifest).read_text().splitlines()]
    cfg = _variant_config(plan, "upstream", manifest, 32, 1)
    from drivebench.benchmark import run_benchmark
    path, _ = run_benchmark(cfg, tmp_path / "external")
    receipt = _execute_run(tmp_path / "adopted", "upstream", cfg, rows, adopted=path)
    assert receipt["adopted"] and all(Path(row["path"]).exists() for row in receipt["outputs"])
    wrong = copy.deepcopy(cfg); wrong["benchmark"]["candidate_count"] = 2
    with pytest.raises(ValueError, match="workload differ"):
        _execute_run(tmp_path / "wrong", "upstream", wrong, rows, adopted=path)


def test_exact_gate_does_not_require_subjective_acceptability_for_identical_pixels():
    metrics = exact_metrics()
    policy = {"policy_type": "exact_output_equivalence", "calibrated": True, "frozen": True,
              "selected_on": "development", "development_manifest_sha256": "d" * 64, "development_scene_ids": ["dev"],
              "frozen_at": "2026-01-01T00:00:00+00:00", "metrics": metrics}
    aggregate = {name: {"mean": 0., "max": 0., "pairs": 1} for name in metrics}
    kwargs = dict(complete_pairs=True, synthetic=False, evaluation_manifest_sha256="e" * 64,
                  evaluation_manifest_metadata={"partition_role": "held_out_evaluation", "manifest_sha256": "e" * 64,
                                                "scene_ids": ["eval"], "clip_count": 1},
                  run_started_at="2026-02-01T00:00:00+00:00", evaluated_scene_ids=["eval"],
                  evaluated_request_ids=["r"], evaluated_pair_count=1, reviewed=False)
    assert tolerance_gate(aggregate, policy, **kwargs)["quality_preserving"]
    aggregate["max_abs_error"]["max"] = 1e-7
    assert not tolerance_gate(aggregate, policy, **kwargs)["quality_preserving"]


def test_timing_schedule_uses_balanced_four_blocks_without_faking_four_200_runs():
    schedule = timing_schedule(["eager", "cached"], 4, 50)
    for name in ("eager", "cached"):
        rows = [row for row in schedule if row["variant"] == name]
        assert sum(row["requests"] for row in rows) == 200
        assert [row["variant_block"] for row in rows] == [0, 1, 2, 3]
    assert [row["variant"] for row in schedule[:2]] == [row["variant"] for row in schedule[2:4]][::-1]
    with pytest.raises(ValueError):
        timing_schedule(["eager", "cached"], 2, 50)


def timing_fixtures(tmp_path, name, factor):
    paths = []
    for block in range(4):
        path = write_run(tmp_path / f"{name}-{block}", backend="vista", saves=0,
                         latencies=[factor * (1 + index / 200 + block / 100) for index in range(50)])
        summary = json.loads((path / "summary.json").read_text())
        summary.update(numerical_validation={"passed": True}, throughput_measurement_wall_s=sum(factor * (1 + index / 200 + block / 100) for index in range(50)))
        (path / "summary.json").write_text(json.dumps(summary))
        (path / "environment_gate.json").write_text('{"passed":true}')
        paths.append(path)
    return paths


def test_timing_block_aggregation_and_real_scene_bootstrap(tmp_path):
    before = aggregate_timing_blocks(timing_fixtures(tmp_path, "before", 1), tmp_path / "before")
    after = aggregate_timing_blocks(timing_fixtures(tmp_path, "after", .8), tmp_path / "after")
    assert before.summary["successful_requests"] == 200
    assert before.summary["headline_eligible"]
    assert len({(r["request_id"], r["repeat"]) for r in before.measurements}) == 200
    comparison = compare_timing_blocks(before.path, after.path)
    assert comparison["metrics"]["p50"]["relative_latency_change"] == pytest.approx(-.2)
    assert comparison["metrics"]["p95"]["ci95"][1] < 0
    assert not comparison["performance_regression_policy_calibrated"]
    assert not comparison["quality_preserving"]


def test_timing_aggregation_rejects_incomplete_and_withholds_bad_numerical_gate(tmp_path):
    paths = timing_fixtures(tmp_path, "before", 1)
    with pytest.raises(ValueError, match="All planned"):
        aggregate_timing_blocks(paths[:3], tmp_path / "missing")
    summary = json.loads((paths[0] / "summary.json").read_text()); summary["numerical_validation"]["passed"] = False
    (paths[0] / "summary.json").write_text(json.dumps(summary))
    run = aggregate_timing_blocks(paths, tmp_path / "ineligible")
    assert not run.summary["headline_eligible"]
    assert "Missing passed scoped numerical validation" in run.summary["eligibility_reasons"]


def test_perceptual_setup_requires_cpu_proof_and_current_asset_hashes(tmp_path):
    package = tmp_path / "package"; package.mkdir(); module = package / "module.py"; module.write_text("fixture")
    weight = tmp_path / "weights"; weight.write_bytes(b"fake test weights")
    record = {"path": str(weight), "sha256": file_sha256(weight), "bytes": weight.stat().st_size}
    receipt = {"passed": True, "status": "cpu_verified", "cpu_verification": {"passed": True},
               "package": {"target": str(package), "files": {"module.py": {"sha256": file_sha256(module), "bytes": module.stat().st_size}}},
               "wheel": record, "alexnet_cached_weight": record, "lpips_learned_weight": record}
    path = tmp_path / "setup.json"; path.write_text(json.dumps(receipt))
    assert verify_perceptual_setup(path)["passed"]
    weight.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed"):
        verify_perceptual_setup(path)


def test_same_seed_action_response_commits_before_retirement(tmp_path):
    destination, plan, _ = study_fixture(tmp_path)
    manifest = plan["manifests"]["development_actions"]["path"]
    rows = [json.loads(line) for line in Path(manifest).read_text().splitlines()]
    assert all(len(set(row["seeds"])) == 1 for row in rows)
    from drivebench.benchmark import run_benchmark
    from drivebench.evaluate import action_response_pairs
    from drivebench.evaluation.io import load_run
    cfg = _variant_config(plan, "upstream", manifest, 16, 2)
    left, _ = run_benchmark(cfg, tmp_path / "left")
    right, _ = run_benchmark(cfg, tmp_path / "right")
    # Inject a known nonzero fixture response; tiny synthetic output can quantize motion to zero.
    for path in (left, right):
        for row in load_run(path).measurements:
            baseline = np.load(path / row["candidates"][0]["frames_path"])
            np.save(path / row["candidates"][1]["frames_path"], np.clip(baseline + .05, 0, 1))
    metrics = action_response_pairs(load_run(left), load_run(right))
    assert len(metrics) == 16
    assert all(row["metrics"]["reference_response_mae"] > 0 for row in metrics)
    assert all(row["metrics"]["response_error_max_abs"] == 0 for row in metrics)
    assert all(len(row["output_sha256"]) == 4 for row in metrics)
    altered = load_run(right)
    altered.measurements[0]["candidates"][1]["seed"] += 1
    with pytest.raises(ValueError, match="identical noise seeds"):
        action_response_pairs(load_run(left), altered)


def test_review_mutation_blocks_resume_even_when_metric_json_is_unchanged(tmp_path):
    destination, _, _ = study_fixture(tmp_path)
    execute_quality(destination, "development", max_shards=1)
    receipt = next((destination / "quality/development/shards/000/metrics").glob("*.receipt.json"))
    data = json.loads(receipt.read_text())
    review = Path(next(iter(data["review_artifact_sha256"])))
    review.write_bytes(review.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="review artifacts changed"):
        execute_quality(destination, "development", max_shards=1)


def test_duplicate_timing_blocks_are_not_independent_runs(tmp_path):
    paths = timing_fixtures(tmp_path, "timing", 1)
    with pytest.raises(ValueError, match="independent executions"):
        aggregate_timing_blocks([paths[0]] * 4, tmp_path / "duplicates")


def test_exact_freeze_requires_all_zero_pairs_and_can_retry_failed_draft(tmp_path):
    evaluation, _, data, _ = freeze_fixture(tmp_path)
    entry = data["runs"][0]
    entry["run"] = "candidate"
    entry["aggregates"] = {name: {"mean": 0., "max": 0., "pairs": 2} for name in exact_metrics()}
    for pair in entry["pairs"]:
        pair["metrics"] = {name: 0. for name in exact_metrics()}
    entry["pairs"][0]["metrics"]["max_abs_error"] = .001
    evaluation.write_text(json.dumps(data))
    output = tmp_path / "zero.yaml"
    with pytest.raises(ValueError, match="exact outputs"):
        freeze_exact_tolerances(evaluation, ["candidate"], output, minimum_scenes=2)
    entry["pairs"][0]["metrics"]["max_abs_error"] = 0.
    evaluation.write_text(json.dumps(data))
    frozen = freeze_exact_tolerances(evaluation, ["candidate"], output, minimum_scenes=2)
    assert frozen["policy_type"] == "exact_output_equivalence"
    with pytest.raises(FileExistsError):
        freeze_exact_tolerances(evaluation, ["candidate"], output, minimum_scenes=2)


def test_quality_overlay_cannot_weaken_timing_identity_or_change_precision(tmp_path):
    from drivebench.study import quality_runtime_overlay, quality_runtime_identity
    from drivebench.benchmarks.environment import reference_identity
    original = {"gpu": {"name": "H100", "uuid": "first", "pci_bus_id": "a"}, "torch": {"version": "fixture"}}
    replacement = copy.deepcopy(original); replacement["gpu"].update(uuid="second", pci_bus_id="b")
    assert quality_runtime_identity(original) == quality_runtime_identity(replacement)
    assert reference_identity(original) != reference_identity(replacement)
    a, b, lock = (tmp_path / name for name in ("a.json", "b.json", "lock.json"))
    a.write_text(json.dumps({"passed": True, "environment": original}))
    b.write_text(json.dumps({"passed": True, "environment": replacement}))
    lock.write_text(json.dumps(reference_identity(replacement)))
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(yaml.safe_dump({"environment_verification": str(b), "environment_lock": str(lock)}))
    plan = {"base_config": {"environment_verification": str(a)}}
    assert quality_runtime_overlay(plan, overlay)["environment_lock"] == str(lock)
    replacement["torch"]["version"] = "changed"
    b.write_text(json.dumps({"passed": True, "environment": replacement}))
    lock.write_text(json.dumps(reference_identity(replacement)))
    with pytest.raises(ValueError, match="all configuration"):
        quality_runtime_overlay(plan, overlay)


def test_two_candidate_companion_quality_cannot_claim_single_future_coverage(tmp_path):
    _, _, spec_path = study_fixture(tmp_path)
    spec = yaml.safe_load(spec_path.read_text())
    spec["quality"]["candidate_count"] = 2
    spec_path.write_text(yaml.safe_dump(spec))
    destination = tmp_path / "companion"
    plan = create_plan(spec_path, destination)
    execute_quality(destination, "development")
    paths = aggregate_quality(destination, "development")
    entries = json.loads(Path(paths["upstream"]).read_text())["runs"]
    assert all(entry["matched_pairs"] == 64 for entry in entries)
    assert all(len(entry["by_action_kind"]) == 2 for entry in entries)
    assert plan["quality"]["candidate_count"] == 2
