"""The installed demo must not depend on repository-relative files or CUDA."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

import drivebench_demo as demo


def _cli(directory, output):
    # Source-tree pytest and installed-wheel pytest use the same invocation. The
    # package location, not the repository working directory, resolves imports.
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(demo.__file__).resolve().parent.parent)
    return subprocess.run([sys.executable, "-m", "drivebench_demo", "--output", str(output)],
                          cwd=directory, env=env, capture_output=True, text=True, timeout=120)


def test_demo_completes_from_an_unrelated_directory(tmp_path):
    outside = tmp_path / "outside checkout"
    outside.mkdir()
    output = outside / "new demo"
    result = _cli(outside, output)
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt == json.loads((output / "demo.json").read_text())
    assert receipt["status"] == "completed"
    assert receipt["synthetic"] is True
    assert receipt["gpu_measurement"] is False
    assert receipt["full_model_inference"] is False
    assert receipt["quality_preserving"] is False
    assert [step["step"] for step in receipt["completed_steps"]] == [
        "prepare", "baseline", "candidate", "evaluate", "report"]
    assert len(list((output / "logs").glob("*.log"))) == 5
    assert all(Path(value).is_absolute() for value in receipt["paths"].values())
    assert json.loads((output / "data" / "audit.json").read_text())["synthetic"] is True
    for role in ("baseline", "candidate"):
        summary = json.loads((output / "runs" / role / "summary.json").read_text())
        assert summary["backend"] == "synthetic"
        assert summary["successful_requests"] == 4
        assert summary["failures"] == 0
        assert summary["headline_eligible"] is False
        assert summary["quality_preserving"] is False
        config = yaml.safe_load((output / "configs" / f"{role}.yaml").read_text())
        assert Path(config["data"]["manifest"]).is_absolute()
        assert Path(config["data"]["root"]).is_absolute()
    evaluation = json.loads((output / "runs" / "evaluation.json").read_text())
    assert len(evaluation["runs"]) == 1
    assert evaluation["runs"][0]["matched_pairs"] == 8
    assert evaluation["runs"][0]["gate"]["quality_preserving"] is False
    report = (output / "runs" / "report.html").read_text()
    assert "SYNTHETIC FUNCTIONAL TEST DATA PRESENT" in report
    assert (output / "runs" / "report.data.json").is_file()
    assert not (output / "failed.json").exists()


@pytest.mark.parametrize("populated", [False, True])
def test_demo_refuses_existing_root_without_writing(tmp_path, populated):
    output = tmp_path / "existing"
    output.mkdir()
    if populated:
        (output / "sentinel.txt").write_text("preserve me")
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}
    with pytest.raises(FileExistsError, match="choose a new path"):
        demo.run_demo(output)
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}
    assert before == after


def test_failed_step_preserves_evidence_and_stops(tmp_path, monkeypatch):
    def fail(command, **kwargs):
        kwargs["stdout"].write("intentional subprocess failure\n")
        raise subprocess.CalledProcessError(7, command)
    monkeypatch.setattr(demo.subprocess, "run", fail)
    original_cwd = Path.cwd()
    output = tmp_path / "failed-demo"
    with pytest.raises(RuntimeError, match="failed during prepare"):
        demo.run_demo(output)
    assert Path.cwd() == original_cwd
    failure = json.loads((output / "failed.json").read_text())
    assert failure["step"] == "prepare"
    assert failure["completed_steps"] == []
    assert "intentional subprocess failure" in Path(failure["log"]).read_text()
    assert (output / "plan.json").is_file()
    assert not (output / "demo.json").exists()
