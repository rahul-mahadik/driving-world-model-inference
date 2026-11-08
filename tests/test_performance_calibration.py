"""CPU artifact fixtures exercise provenance; these are not GPU measurements."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import yaml

from drivebench.calibrate_performance import calibrate_performance
from drivebench.regression import compare_runs
from test_evaluation import performance_policy, write_run


def copied_sources(policy, destination):
    paths = []
    for index, entry in enumerate(policy["performance"]["calibration_evidence"]["runs"]):
        target = destination / f"source-{index}"
        shutil.copytree(entry["path"], target)
        paths.append(target)
    return paths


def modify_json(path, update):
    value = json.loads(path.read_text())
    update(value)
    path.write_text(json.dumps(value))


def test_policy_records_actual_drift_and_never_overwrites(performance_policy, tmp_path):
    settings = performance_policy["performance"]
    evidence = settings["calibration_evidence"]
    assert len(evidence["runs"]) == 4 and len(evidence["comparisons"]) == 6
    assert 0 < evidence["required_tolerance"] <= settings["relative_latency_tolerance"]
    assert all(comparison["paired_requests"] == 200 for comparison in evidence["comparisons"])
    assert all(set(entry["artifact_sha256"]) == {"config.json", "environment.json", "measurements.jsonl", "summary.json"}
               for entry in evidence["runs"])
    path = tmp_path / "already-frozen-v1.yaml"
    path.write_text("do not overwrite\n")
    with pytest.raises(FileExistsError, match="cannot be overwritten"):
        calibrate_performance([entry["path"] for entry in evidence["runs"]], 0.1, path)
    assert path.read_text() == "do not overwrite\n"
    assert yaml.safe_load(yaml.safe_dump(performance_policy)) == performance_policy


def test_explicit_tolerance_must_cover_measured_variability(performance_policy, tmp_path):
    paths = [entry["path"] for entry in performance_policy["performance"]["calibration_evidence"]["runs"]]
    with pytest.raises(ValueError, match="does not cover measured variability"):
        calibrate_performance(paths, 0.0, tmp_path / "too-tight.yaml")
    assert not (tmp_path / "too-tight.yaml").exists()


@pytest.mark.parametrize("fault", ["too_few", "duplicate_path", "copied_timings", "synthetic", "profiled", "ineligible",
                                   "oom", "recompile", "failed", "short", "optimizations", "runtime", "one_scene", "seed",
                                   "export_policy", "request_order"])
def test_calibration_rejects_unproven_baseline_sets(performance_policy, tmp_path, fault):
    paths = copied_sources(performance_policy, tmp_path)
    first = paths[0]
    if fault == "too_few":
        paths = paths[:3]
    elif fault == "duplicate_path":
        paths[3] = paths[0]
    elif fault == "copied_timings":
        (paths[1] / "measurements.jsonl").write_bytes((first / "measurements.jsonl").read_bytes())
    elif fault == "synthetic":
        modify_json(first / "summary.json", lambda value: value.update(backend="synthetic"))
    elif fault == "profiled":
        modify_json(first / "config.json", lambda value: value.update(profiled=True))
    elif fault == "ineligible":
        modify_json(first / "summary.json", lambda value: value.update(headline_eligible=False))
    elif fault == "oom":
        modify_json(first / "summary.json", lambda value: value.update(oom_attempts=1))
    elif fault == "recompile":
        modify_json(first / "summary.json", lambda value: value.update(recompiled_during_measurement=True))
    elif fault == "failed":
        modify_json(first / "summary.json", lambda value: value.update(status="failed"))
    elif fault == "short":
        modify_json(first / "config.json", lambda value: value["benchmark"].update(requests=199))
        rows = (first / "measurements.jsonl").read_text().splitlines()[:199]
        (first / "measurements.jsonl").write_text("\n".join(rows) + "\n")
    elif fault == "optimizations":
        modify_json(first / "config.json", lambda value: value["optimizations"].update(compile_denoiser=True))
    elif fault == "runtime":
        modify_json(first / "environment.json", lambda value: value["packages"].update(torch="different-version"))
    elif fault == "export_policy":
        modify_json(first / "config.json", lambda value: value["benchmark"].update(save_outputs=100))
    elif fault == "request_order":
        rows = (first / "measurements.jsonl").read_text().splitlines()
        (first / "measurements.jsonl").write_text("\n".join(reversed(rows)) + "\n")
    else:
        for path in paths if fault == "one_scene" else [first]:
            rows = [json.loads(line) for line in (path / "measurements.jsonl").read_text().splitlines()]
            for row in rows:
                if fault == "one_scene":
                    row["scene_id"] = "one"
                else:
                    row["candidates"][0]["seed"] += 1
            (path / "measurements.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        calibrate_performance(paths, 0.1, tmp_path / "invalid.yaml")
    assert not (tmp_path / "invalid.yaml").exists()


@pytest.mark.parametrize("fault", ["flag_only", "tampered_statistics", "threshold_too_small", "late_freeze",
                                   "identity", "confirmation_disabled", "changed_source", "calibration_reuse"])
def test_regression_rechecks_calibration_provenance(performance_policy, tmp_path, fault):
    baseline = write_run(tmp_path / "baseline", backend="vista", saves=0, latencies=np.ones(200))
    candidate = write_run(tmp_path / "candidate", backend="vista", saves=0, latencies=np.ones(200) * 1.05)
    policy = deepcopy(performance_policy)
    settings = policy["performance"]
    if fault == "flag_only":
        policy = {"performance": {"calibrated": True, "relative_latency_tolerance": 0.1, "minimum_requests": 200}}
    elif fault == "tampered_statistics":
        settings["calibration_evidence"]["comparisons"][0]["metrics"]["p50"]["ci95"] = [0, 0]
    elif fault == "threshold_too_small":
        settings["relative_latency_tolerance"] = 0
    elif fault == "late_freeze":
        settings["frozen_at"] = "2099-01-01T00:00:00+00:00"
    elif fault == "identity":
        settings["calibration_evidence"]["identity"]["optimizations"] = {"changed": True}
    elif fault == "confirmation_disabled":
        settings["require_confirmatory_rerun"] = False
    elif fault == "changed_source":
        paths = copied_sources(policy, tmp_path / "copies")
        for entry, path in zip(settings["calibration_evidence"]["runs"], paths):
            entry["path"] = str(path)
        modify_json(paths[0] / "summary.json", lambda value: value.update(extra_note="changed after freezing"))
    else:
        baseline = Path(settings["calibration_evidence"]["runs"][0]["path"])
        candidate = write_run(tmp_path / "candidate-copy", backend="vista", saves=0,
                              latencies=np.linspace(0.995, 1.005, 200) * 1.01)
    result = compare_runs(baseline, candidate, policy)
    assert result["status"] == "uncalibrated", result
    assert result["reasons"]


def test_regression_requires_eligible_comparison_runs(performance_policy, tmp_path):
    baseline = write_run(tmp_path / "baseline", backend="vista", saves=0, latencies=np.ones(200))
    candidate = write_run(tmp_path / "candidate", backend="vista", saves=0, latencies=np.ones(200) * 1.01)
    modify_json(candidate / "summary.json", lambda value: value.update(headline_eligible=False))
    result = compare_runs(baseline, candidate, performance_policy)
    assert result["status"] == "incomplete"
    assert any("eligible" in reason for reason in result["reasons"])


def test_confirmation_cannot_reuse_timings_with_changed_metadata(performance_policy, tmp_path):
    baseline = write_run(tmp_path / "baseline", backend="vista", saves=0, latencies=np.ones(200))
    candidate = write_run(tmp_path / "candidate", backend="vista", saves=0, latencies=np.ones(200) * 1.4)
    copies = []
    for source in (baseline, candidate):
        target = tmp_path / f"{source.name}-copy"
        shutil.copytree(source, target)
        rows = [json.loads(line) for line in (target / "measurements.jsonl").read_text().splitlines()]
        for row in rows:
            row["note"] = "New labels do not make these independent measurements"
            row["export_s"] = 9.0
        (target / "measurements.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        copies.append(target)
    result = compare_runs(baseline, candidate, performance_policy,
                          confirm_baseline_path=copies[0], confirm_candidate_path=copies[1])
    assert result["status"] == "rerun_required"
    assert any("independent" in reason for reason in result["reasons"])


def test_calibration_timings_cannot_be_reused_with_changed_metadata(performance_policy, tmp_path):
    baseline = write_run(tmp_path / "baseline", backend="vista", saves=0,
                         latencies=np.linspace(0.995, 1.005, 200))
    candidate = write_run(tmp_path / "candidate", backend="vista", saves=0, latencies=np.ones(200))
    result = compare_runs(baseline, candidate, performance_policy)
    assert result["status"] == "uncalibrated"
    assert any("independently collected" in reason for reason in result["reasons"])


@pytest.mark.parametrize("same_path", [True, False])
def test_regression_rejects_primary_self_comparison(performance_policy, tmp_path, same_path):
    baseline = write_run(tmp_path / "baseline", backend="vista", saves=0, latencies=np.ones(200))
    candidate = baseline if same_path else write_run(tmp_path / "candidate", backend="vista", saves=0, latencies=np.ones(200))
    result = compare_runs(baseline, candidate, performance_policy)
    assert result["status"] == "incomplete"
    assert any("independent" in reason for reason in result["reasons"])
