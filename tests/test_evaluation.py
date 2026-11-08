from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from drivebench.evaluate import _ground_truth, aggregate_pairs, evaluate_runs, tolerance_gate
from drivebench.evaluation.io import load_run, output_path
from drivebench.evaluation.metrics import PerceptualMetric, as_video, compare_videos, temporal_diagnostics
from drivebench.regression import _bootstrap_ratios, compare_runs
from drivebench.report import build_report


def write_run(path, *, backend="synthetic", latencies=(1.0,), frames=None, action_kind="alternative", saves=None,
              started_at="2035-10-20T00:00:00+00:00"):
    path.mkdir(parents=True)
    config = {"model": {"backend": backend, "upstream_commit": "a" * 40, "checkpoint_sha256": "b" * 64},
              "data": {"manifest_sha256": "c" * 64},
              "generation": {"total_frames": 4, "conditioning_frames": 1, "height": 8, "width": 8, "steps": 50},
              "benchmark": {"workload": "single_future", "candidate_count": 1, "input_condition": "uncached",
                            "requests": len(latencies)},
              "optimizations": {}, "started_at": started_at}
    if saves is not None:
        config["benchmark"]["save_outputs"] = saves
    environment = {"gpu": {"name": "NVIDIA H100 80GB HBM3", "memory_total_mib": "81559", "mig_mode": "Disabled",
                            "uuid": "GPU-test", "driver_version": "test"},
                   "torch": {"version": "test", "cuda": "test"}, "timestamp_utc": started_at,
                   "python": "test-python", "packages": {"torch": "test"},
                   "tools": {"nvcc": {"stdout": "test-cuda"}, "nsys": {"stdout": "test-nsys"},
                             "ncu": {"stdout": "test-ncu"}}}
    video = np.zeros((4, 8, 8, 3), dtype=np.float32) if frames is None else frames
    rows = []
    for index, latency in enumerate(latencies):
        frame_path = f"outputs/r{index}/candidate.npy"
        if saves is None or index < saves:
            (path / frame_path).parent.mkdir(parents=True)
            np.save(path / frame_path, video)
        else:
            frame_path = None
        rows.append({"request_id": f"r{index % 2}", "scene_id": f"s{index % 2}", "repeat": index // 2,
                     "request_index": index, "status": "ok", "latency_s": float(latency), "candidate_count": 1,
                     "batch_size": 1, "stage_timings_s": {"denoising": float(latency) * 0.8},
                     "peak_allocated_bytes": 1024, "peak_reserved_bytes": 2048, "denoiser_evaluations": 50,
                     "candidates": [{"candidate_id": "c0", "seed": 123, "action_kind": action_kind,
                                     "trajectory_sha256": "a" * 64,
                                     "frames_path": frame_path, "conditioning_frames": 1,
                                     "ground_truth_paths": ["/must/not/be/read"]}]})
    # Eligibility is fixture input; these artifacts are not real GPU evidence.
    summary = {"backend": backend, "failures": 0, "headline_eligible": backend in {"vista", "upstream"}}
    for name, obj in (("config.json", config), ("environment.json", environment), ("summary.json", summary)):
        (path / name).write_text(json.dumps(obj))
    (path / "measurements.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def tolerance_file(path):
    path.write_text(yaml.safe_dump({"calibrated": False, "frozen": False, "metrics": {}}))
    return path


def test_numerical_and_freezing_diagnostics():
    moving = np.stack([np.full((8, 8, 3), value, dtype=np.float32) for value in (0, 0.2, 0.4, 0.6)])
    frozen = np.zeros_like(moving)
    same = compare_videos(moving, moving.copy())
    assert same["exact_match"] is True
    assert same["rmse"] == 0
    assert same["psnr_db"] is None
    assert temporal_diagnostics(frozen)["frozen_transition_fraction"] == 1
    comparison = compare_videos(moving, frozen)
    assert comparison["freeze_fraction_increase"] == 1
    assert comparison["temporal_derivative_mae"] > 0.19
    assert not any("adherence" in name for name in comparison)


def test_validation_rejects_nonfinite_and_wrong_layout():
    with pytest.raises(ValueError, match="nonfinite"):
        as_video(np.full((3, 8, 8, 3), np.nan))
    with pytest.raises(ValueError, match="THWC"):
        as_video(np.zeros((3, 3, 8, 8)))
    with pytest.raises(ValueError, match="shape mismatch"):
        compare_videos(np.zeros((3, 8, 8, 3)), np.zeros((2, 8, 8, 3)))
    assert np.max(as_video(np.full((2, 8, 8, 3), 255, dtype=np.uint8))) == 1


def test_perceptual_metric_not_requested_is_explicit():
    metric = PerceptualMetric(False)
    assert metric.status == "not_requested"
    assert metric.compare(np.zeros((3, 8, 8, 3)), np.zeros((3, 8, 8, 3))) is None


def test_recorded_ground_truth_uses_same_center_crop_as_model(tmp_path):
    from PIL import Image
    from drivebench.data.preprocessing import read_preprocess
    image = np.zeros((8, 16, 3), dtype=np.uint8)
    image[:, :4] = 255
    image[:, 12:] = 255
    path = tmp_path / "wide.png"
    Image.fromarray(image).save(path)
    actual = _ground_truth({"action_kind": "recorded", "ground_truth_paths": [str(path)]}, (1, 8, 8, 3))
    expected = ((read_preprocess([path], 8, 8) + 1) / 2).transpose(0, 2, 3, 1)
    np.testing.assert_array_equal(actual, expected)
    assert actual.max() == 0


def test_aggregation_resamples_scenes_not_candidate_repeats():
    pairs = [{"scene_id": "a", "metrics": {"rmse": 1.0}} for _ in range(12)]
    pairs.append({"scene_id": "b", "metrics": {"rmse": 3.0}})
    aggregate = aggregate_pairs(pairs)["rmse"]
    assert aggregate["mean"] == 2
    assert aggregate["units"] == 2
    assert aggregate["pairs"] == 13


def test_gate_requires_frozen_thresholds_complete_pairs_review_and_real_backend():
    aggregates = {metric: {"mean": 0.01, "max": 0.02, "pairs": 1} for metric in ("rmse", "lpips", "freeze_fraction_increase")}
    tolerances = {"calibrated": True, "frozen": True, "selected_on": "development",
                  "development_manifest_sha256": "d" * 64, "frozen_at": "2026-10-01T00:00:00+00:00",
                  "development_scene_ids": ["dev-scene"],
                  "metrics": {metric: {"maximum": 0.05} for metric in aggregates}}
    kwargs = {"complete_pairs": True, "synthetic": False, "evaluation_manifest_sha256": "e" * 64,
              "evaluation_manifest_metadata": {"partition_role": "evaluation", "scene_ids": ["eval-scene"],
                                                "manifest_sha256": "e" * 64, "clip_count": 1},
              "run_started_at": "2026-10-20T00:00:00+00:00", "reviewed": True,
              "evaluated_scene_ids": ["eval-scene"], "evaluated_request_ids": ["eval-request"], "evaluated_pair_count": 1}
    assert tolerance_gate(aggregates, tolerances, **kwargs)["quality_preserving"]
    for field, value in (("complete_pairs", False), ("synthetic", True), ("reviewed", False)):
        assert not tolerance_gate(aggregates, tolerances, **{**kwargs, field: value})["quality_preserving"]
    assert not tolerance_gate(aggregates, {**tolerances, "frozen": False}, **kwargs)["quality_preserving"]
    assert not tolerance_gate(aggregates, tolerances, **{**kwargs, "run_started_at": "2026-09-01T00:00:00+00:00"})["quality_preserving"]
    overlap = {**kwargs["evaluation_manifest_metadata"], "scene_ids": ["dev-scene"]}
    assert not tolerance_gate(aggregates, tolerances, **{**kwargs, "evaluation_manifest_metadata": overlap})["quality_preserving"]
    wrong_role = {**kwargs["evaluation_manifest_metadata"], "partition_role": "smoke"}
    assert not tolerance_gate(aggregates, tolerances, **{**kwargs, "evaluation_manifest_metadata": wrong_role})["quality_preserving"]
    assert not tolerance_gate(aggregates, tolerances, **{**kwargs, "evaluation_manifest_metadata": {}})["quality_preserving"]
    canonical_role = {**kwargs["evaluation_manifest_metadata"], "partition_role": "held_out_evaluation"}
    assert tolerance_gate(aggregates, tolerances, **{**kwargs, "evaluation_manifest_metadata": canonical_role})["quality_preserving"]
    for field, value in (("evaluated_scene_ids", []), ("evaluated_request_ids", []), ("evaluated_pair_count", 2)):
        assert not tolerance_gate(aggregates, tolerances, **{**kwargs, field: value})["quality_preserving"]


def test_end_to_end_evaluation_excludes_conditioning_and_counterfactual_gt(tmp_path):
    runs = tmp_path / "runs"
    baseline = write_run(runs / "baseline", latencies=(1, 1), saves=1)
    video = np.zeros((4, 8, 8, 3), dtype=np.float32)
    video[0] = 1  # A conditioning-frame difference must not affect future-only metrics.
    write_run(runs / "compiled", latencies=(0.8, 0.8), frames=video, saves=1)
    result = evaluate_runs(runs, reference=baseline, tolerances_path=tolerance_file(tmp_path / "t.yaml"))
    run = result["runs"][0]
    assert run["complete_pairs"]
    assert run["matched_pairs"] == 1
    assert run["aggregates"]["rmse"]["mean"] == 0
    assert run["pairs"][0]["recorded_ground_truth"] is None
    assert not run["gate"]["quality_preserving"]
    public = Path(result["review"]["index"]).read_text()
    assert "compiled" not in public and "baseline" not in public
    assert Path(result["review"]["private_key"]).stat().st_mode & 0o777 == 0o600
    from PIL import Image
    private = json.loads(Path(result["review"]["private_key"]).read_text())
    with Image.open(Path(result["review"]["index"]).parent / f"{private['cases'][0]['case_id']}.gif") as animation:
        assert animation.n_frames == 3
        assert animation.info["duration"] == 100
    report = build_report(runs)
    assert "SYNTHETIC FUNCTIONAL TEST DATA PRESENT" in report.read_text()
    assert "not established" in report.read_text()
    assert report.with_suffix(".data.json").exists()


def test_missing_planned_output_does_not_pass_quality_gate(tmp_path):
    runs = tmp_path / "runs"
    baseline = write_run(runs / "baseline")
    candidate = write_run(runs / "compiled")
    (candidate / "outputs/r0/candidate.npy").unlink()
    result = evaluate_runs(runs, reference=baseline, tolerances_path=tolerance_file(tmp_path / "t.yaml"))
    assert not result["runs"][0]["complete_pairs"]
    assert result["runs"][0]["errors"]


def test_matching_truncated_videos_do_not_pass_pair_completeness(tmp_path):
    runs = tmp_path / "runs"
    truncated = np.zeros((3, 8, 8, 3), dtype=np.float32)
    baseline = write_run(runs / "baseline", frames=truncated)
    write_run(runs / "compiled", frames=truncated)
    result = evaluate_runs(runs, reference=baseline, tolerances_path=tolerance_file(tmp_path / "t.yaml"))
    assert not result["runs"][0]["complete_pairs"]
    assert "declared generation workload" in result["runs"][0]["errors"][0]["error"]


def test_output_reader_rejects_escape(tmp_path):
    run = load_run(write_run(tmp_path / "run"))
    with pytest.raises(ValueError, match="escapes"):
        output_path(run, "../outside.npy")


def test_review_results_survive_reevaluation(tmp_path):
    runs = tmp_path / "runs"
    baseline = write_run(runs / "baseline")
    write_run(runs / "compiled")
    tolerances = tolerance_file(tmp_path / "t.yaml")
    result = evaluate_runs(runs, reference=baseline, tolerances_path=tolerances)
    csv_path = Path(result["review"]["csv"])
    with csv_path.open() as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["acceptable_A"], rows[0]["acceptable_B"] = "yes", "yes"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    saved = csv_path.read_text()
    evaluate_runs(runs, reference=baseline, tolerances_path=tolerances, review_results=csv_path)
    assert csv_path.read_text() == saved


def test_changed_outputs_cannot_reuse_completed_review_votes(tmp_path):
    runs = tmp_path / "runs"
    baseline = write_run(runs / "baseline")
    candidate = write_run(runs / "compiled")
    tolerances = tolerance_file(tmp_path / "t.yaml")
    initial = evaluate_runs(runs, reference=baseline, tolerances_path=tolerances)
    csv_path = Path(initial["review"]["csv"])
    rows = list(csv.DictReader(csv_path.read_text().splitlines()))
    rows[0]["acceptable_A"], rows[0]["acceptable_B"] = "yes", "yes"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    saved = csv_path.read_text()
    np.save(candidate / "outputs/r0/candidate.npy", np.full((4, 8, 8, 3), 0.5, dtype=np.float32))
    changed = evaluate_runs(runs, reference=baseline, tolerances_path=tolerances, review_results=csv_path)
    assert changed["review"]["review_results_status"].startswith("rejected:")
    current = list(csv.DictReader(csv_path.read_text().splitlines()))
    assert current[0]["case_id"] != rows[0]["case_id"]
    assert current[0]["acceptable_A"] == ""
    assert saved in [path.read_text() for path in csv_path.parent.glob("review.previous-*.csv")]


def test_evaluation_fingerprints_and_stale_report_guard(tmp_path):
    from drivebench.evaluation.io import artifact_sha256
    runs = tmp_path / "runs"
    baseline = write_run(runs / "baseline")
    candidate = write_run(runs / "compiled")
    result = evaluate_runs(runs, reference=baseline, tolerances_path=tolerance_file(tmp_path / "t.yaml"))
    assert result["reference_artifact_sha256"] == artifact_sha256(load_run(baseline))
    assert result["runs"][0]["artifact_sha256"] == artifact_sha256(load_run(candidate))
    config_path = candidate / "config.json"
    config = json.loads(config_path.read_text())
    config["generation"]["steps"] = 20
    config_path.write_text(json.dumps(config))
    report = build_report(runs).read_text()
    assert "Quality results are stale" in report
    assert "not established" in report


@pytest.mark.parametrize("changed", ["run_failed", "missing_model_identity", "missing_action_identity"])
def test_real_evaluation_rejects_unproven_pairs(tmp_path, changed):
    runs = tmp_path / "runs"
    baseline = write_run(runs / "baseline", backend="vista")
    candidate = write_run(runs / "compiled", backend="vista")
    for run in (baseline, candidate):
        if changed == "run_failed":
            path = run / "summary.json"
            data = json.loads(path.read_text())
            data["status"] = "failed"
            path.write_text(json.dumps(data))
        elif changed == "missing_model_identity":
            path = run / "config.json"
            data = json.loads(path.read_text())
            del data["model"]["checkpoint_sha256"]
            path.write_text(json.dumps(data))
        else:
            path = run / "measurements.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            for row in rows:
                del row["candidates"][0]["trajectory_sha256"]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = evaluate_runs(runs, reference=baseline, tolerances_path=tolerance_file(tmp_path / "t.yaml"))
    assert not result["runs"][0]["complete_pairs"]
    assert result["runs"][0]["errors"]


@pytest.fixture(scope="module")
def performance_policy(tmp_path_factory):
    from drivebench.calibrate_performance import calibrate_performance
    root = tmp_path_factory.mktemp("regression_calibration")
    paths = []
    for index in range(4):
        path = write_run(root / f"baseline-{index}", backend="vista", saves=0,
                         latencies=np.linspace(0.995, 1.005, 200) * (1 + index * 0.001),
                         started_at=f"2000-01-0{index + 1}T00:00:00+00:00")
        summary = json.loads((path / "summary.json").read_text())
        summary["headline_eligible"] = True
        (path / "summary.json").write_text(json.dumps(summary))
        paths.append(path)
    return calibrate_performance(paths, 0.1, root / "performance-v1.yaml")


def test_regression_requires_uncertainty_and_independent_confirmation(tmp_path, performance_policy):
    latency = np.linspace(0.99, 1.01, 200)
    baseline = write_run(tmp_path / "baseline", backend="vista", saves=0, latencies=latency)
    candidate = write_run(tmp_path / "candidate", backend="vista", saves=0, latencies=latency * 1.4)
    result = compare_runs(baseline, candidate, performance_policy)
    assert result["status"] == "rerun_required"
    assert result["metrics"]["p50"]["ci95"][0] > 0.1
    duplicate = compare_runs(baseline, candidate, performance_policy,
                             confirm_baseline_path=baseline, confirm_candidate_path=candidate)
    assert duplicate["status"] == "rerun_required"
    confirmation_baseline = write_run(tmp_path / "baseline2", backend="vista", saves=0, latencies=latency + 0.001)
    confirmation_candidate = write_run(tmp_path / "candidate2", backend="vista", saves=0, latencies=(latency + 0.001) * 1.4)
    confirmed = compare_runs(baseline, candidate, performance_policy,
                             confirm_baseline_path=confirmation_baseline, confirm_candidate_path=confirmation_candidate)
    assert confirmed["status"] == "regression_confirmed"


@pytest.mark.parametrize("field", ["gpu", "manifest", "profiled", "generation_steps", "generation_dtype"])
def test_regression_rejects_incompatible_identity(tmp_path, field, performance_policy):
    baseline = write_run(tmp_path / "baseline", backend="vista", saves=0, latencies=np.ones(200))
    candidate = write_run(tmp_path / "candidate", backend="vista", saves=0, latencies=np.ones(200))
    filename = "environment.json" if field == "gpu" else "config.json"
    path = candidate / filename
    document = json.loads(path.read_text())
    if field == "gpu":
        document["gpu"]["name"] = "NVIDIA H100 PCIe"
    elif field == "manifest":
        document["data"]["manifest_sha256"] = "f" * 64
    elif field == "generation_steps":
        document["generation"]["steps"] = 20
    elif field == "generation_dtype":
        document["generation"]["dtype"] = "bf16"
    else:
        document["profiled"] = True
    path.write_text(json.dumps(document))
    assert compare_runs(baseline, candidate, performance_policy)["status"] == "incompatible"


def test_regression_does_not_flag_below_tolerance(tmp_path, performance_policy):
    baseline = write_run(tmp_path / "baseline", backend="vista", saves=0, latencies=np.ones(200))
    candidate = write_run(tmp_path / "candidate", backend="vista", saves=0, latencies=np.ones(200) * 1.05)
    assert compare_runs(baseline, candidate, performance_policy)["status"] == "pass"


@pytest.mark.parametrize("changed", ["candidate_missing", "action_identity_missing", "missing_request"])
def test_regression_requires_complete_candidate_workload_evidence(tmp_path, changed, performance_policy):
    baseline = write_run(tmp_path / "baseline", backend="vista", saves=0, latencies=np.ones(200))
    candidate = write_run(tmp_path / "candidate", backend="vista", saves=0, latencies=np.ones(200))
    for run in (baseline, candidate):
        if changed == "missing_request":
            path = run / "config.json"
            config = json.loads(path.read_text())
            config["benchmark"]["requests"] = 201
            path.write_text(json.dumps(config))
        else:
            path = run / "measurements.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            for row in rows:
                if changed == "candidate_missing":
                    row["candidates"] = []
                else:
                    del row["candidates"][0]["trajectory_sha256"]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert compare_runs(baseline, candidate, performance_policy)["status"] == "incomplete"


def test_regression_scene_bootstrap_does_not_treat_repeats_as_independent():
    baseline = np.ones(200)
    candidate = np.concatenate([np.ones(100), np.full(100, 1.4)])
    _, bounds = _bootstrap_ratios(baseline, candidate, scene_groups=["a"] * 100 + ["b"] * 100,
                                  quantile=0.5, resamples=2000)
    # Whole-scene resampling retains the two distinctly different performance
    # responses; 100 repeats of each scene must not produce a narrow CI.
    assert bounds[0] == pytest.approx(0)
    assert bounds[1] == pytest.approx(0.4)


@pytest.mark.parametrize("contamination", ["oom_summary", "oom_successful_retry", "recompiled", "run_failed"])
def test_regression_rejects_contaminated_measurements(tmp_path, contamination, performance_policy):
    baseline = write_run(tmp_path / "baseline", backend="vista", saves=0, latencies=np.ones(200))
    candidate = write_run(tmp_path / "candidate", backend="vista", saves=0, latencies=np.ones(200))
    if contamination == "oom_successful_retry":
        path = candidate / "measurements.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert rows[0]["status"] == "ok"
        rows[0]["cache_stats"] = {"oom_attempts": [{"batch_size": 4}], "denoiser_evaluations_complete": False}
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    else:
        path = candidate / "summary.json"
        summary = json.loads(path.read_text())
        if contamination == "oom_summary":
            summary["oom_attempts"] = 1
        elif contamination == "recompiled":
            summary["recompiled_during_measurement"] = True
        else:
            summary["status"] = "failed"
        path.write_text(json.dumps(summary))
    assert compare_runs(baseline, candidate, performance_policy)["status"] == "incomplete"


@pytest.mark.parametrize("changed_field, changed_value", [("steps", 20), ("dtype", "bf16")])
def test_regression_confirmation_requires_same_generation_settings(tmp_path, changed_field, changed_value, performance_policy):
    baseline = write_run(tmp_path / "baseline", backend="vista", saves=0, latencies=np.ones(200))
    candidate = write_run(tmp_path / "candidate", backend="vista", saves=0, latencies=np.ones(200) * 1.4)
    baseline2 = write_run(tmp_path / "baseline2", backend="vista", saves=0, latencies=np.ones(200) * 1.001)
    candidate2 = write_run(tmp_path / "candidate2", backend="vista", saves=0, latencies=np.ones(200) * 1.401)
    path = candidate2 / "config.json"
    config = json.loads(path.read_text())
    config["generation"][changed_field] = changed_value
    path.write_text(json.dumps(config))
    result = compare_runs(baseline, candidate, performance_policy,
                          confirm_baseline_path=baseline2, confirm_candidate_path=candidate2)
    assert result["status"] == "rerun_required"
    assert any("incompatible" in reason for reason in result["reasons"])
