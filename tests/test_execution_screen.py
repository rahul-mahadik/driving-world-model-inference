from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from drivebench.config import digest, file_sha256
from drivebench.data import write_manifest
from drivebench.types import GenerationConfig, OptimizationConfig
from scripts import check_screening_outputs as checker
from scripts import prepare_execution_screen as screen


def test_screen_plan_distinguishes_counts_flags_gates_and_cost(tmp_path):
    records = [{"request_id": f"r{i}", "scene_id": f"scene-{i:02d}", "dataset": "nuscenes"} for i in range(32)]
    manifest = tmp_path / "development.jsonl"
    write_manifest(manifest, records, {"partition_role": "development"})
    base = {"model": {"backend": "vista"}, "data": {"manifest": str(manifest), "root": "/workspace/data/nuscenes"},
            "generation": GenerationConfig().to_dict(), "optimizations": OptimizationConfig().to_dict(),
            "benchmark": {"headline": False}}
    config = tmp_path / "base.yaml"
    config.write_text(yaml.safe_dump(base))
    output = tmp_path / "screen"
    plan = screen.prepare(config, manifest, output)
    assert len(plan["configs"]) == 40 and len(plan["commands"]) == 34
    assert plan["commands"][0]["id"] == "batch2-agreement-strict"
    assert plan["commands"][0]["serial_equivalent_futures"] == 4
    assert plan["commands"][0]["argv"][-2:] == ["--mode", "agreement"]
    assert len(plan["input_output_checks"]) == 7
    assert all(gate["serial_equivalent_futures"] == 14 for gate in plan["full_numerical_gates"].values())
    loaded = {name: yaml.safe_load(Path(value["path"]).read_text()) for name, value in plan["configs"].items()}
    assert loaded["gate-batch2"]["data"]["manifest"] == str(manifest)
    assert loaded["memory-b2"]["benchmark"]["candidate_count"] == 2
    assert loaded["memory-b2"]["optimizations"]["max_batch_size"] == 2
    assert loaded["candidates-16-eager"]["benchmark"]["candidate_count"] == 16
    assert loaded["candidates-16-eager"]["optimizations"]["max_batch_size"] == 1
    assert loaded["input-prefetch-throughput"]["benchmark"]["save_outputs"] == 0
    assert loaded["input-prefetch-correctness"]["benchmark"]["save_outputs"] == 2
    assert loaded["input-prefetch-throughput"]["optimizations"]["pinned_memory"] is False
    assert loaded["input-pinned-async-throughput"]["optimizations"]["prefetch_depth"] == 0
    assert loaded["input-conditioning-fresh-throughput"]["benchmark"]["input_condition"] == "uncached"
    assert loaded["input-repeated-conditioning-throughput"]["benchmark"]["input_condition"] == "reused_conditioning"
    assert all(not value["benchmark"]["headline"] for value in loaded.values())
    assert all(value["generation"] == GenerationConfig().to_dict() for value in loaded.values())
    assert plan["cost_by_priority"]["2"]["serial_reference_hours"] == 1.3
    with pytest.raises(FileExistsError):
        screen.prepare(config, manifest, output)


def write_run(path, *, offset=0, seed=1, missing=False, oom=False):
    path.mkdir()
    config = {"generation": GenerationConfig(total_frames=2, height=8, width=8).to_dict(), "optimizations": OptimizationConfig().to_dict(),
              "data": {"manifest_sha256": "selection", "manifest_metadata": {"partition_role": "development_subset"}}}
    identity = {"checkpoint_sha256": "weights", "inference_config_sha256": "model", "upstream_commit": "source",
                "parameter_grad_policy": "preserve_upstream_loader_flags"}
    for name, value in (("config", config), ("environment", {}), ("model_identity", identity), ("summary", {})):
        (path / f"{name}.json").write_text(json.dumps(value))
    row = {"request_index": 0, "request_id": "r", "scene_id": "s", "status": "ok",
           "cache_stats": {"oom_attempts": [{"batch_size": 2}] if oom else []},
           "candidates": [{"candidate_id": "r:c00", "seed": seed, "trajectory_sha256": digest([0] * 8),
                           "action_kind": "recorded", "frames_path": None if missing else "array.npy"}]}
    (path / "measurements.jsonl").write_text(json.dumps(row) + "\n")
    np.save(path / "array.npy", np.full((2, 8, 8, 3), offset, np.float32))


def test_actual_benchmark_outputs_compare_and_bind_hashes(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    write_run(left)
    write_run(right)
    report = checker.check(left, right)
    assert report["passed"] and not report["headline_eligible"]
    assert report["comparisons"][0]["reference_sha256"] == file_sha256(left / "array.npy")
    np.save(right / "array.npy", np.ones((2, 8, 8, 3), np.float32))
    failed = checker.check(left, right)
    assert failed["passed"] is False
    assert failed["comparisons"][0]["candidates"]["r:c00"]["max_abs_error"] == 1


@pytest.mark.parametrize("kwargs,match", [({"seed": 2}, "seed"), ({"missing": True}, "Missing"),
                                        ({"oom": True}, "OOM")])
def test_input_path_check_rejects_incomplete_or_different_work(tmp_path, kwargs, match):
    left, right = tmp_path / "left", tmp_path / "right"
    write_run(left)
    write_run(right, **kwargs)
    with pytest.raises(ValueError, match=match):
        checker.check(left, right)


def test_input_path_check_records_failure_and_refuses_overwrite(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    write_run(left)
    write_run(right, offset=0.01)
    output = tmp_path / "proof.json"
    args = ["--reference", str(left), "--candidate", str(right), "--output", str(output)]
    assert checker.main(args) == 1
    assert json.loads(output.read_text())["passed"] is False
    with pytest.raises(FileExistsError):
        checker.main(args)
