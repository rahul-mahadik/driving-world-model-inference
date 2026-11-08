"""Synthetic orchestration/display preservation tests; no GPU evidence."""
import datetime as dt
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from drivebench import study
from drivebench.config import file_sha256
from drivebench.evaluate import evaluate_runs
from test_evaluation import write_run, tolerance_file
from test_study import study_fixture


@pytest.fixture
def helper():
    path = Path(__file__).resolve().parents[1] / "scripts/run_quality_bounded.py"
    spec = importlib.util.spec_from_file_location("quality_bounded_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def comparison(tmp_path):
    a = np.stack([np.full((8, 8, 3), v, np.float32) for v in (0, .1, .2, .3)])
    b = np.stack([np.full((8, 8, 3), v, np.float32) for v in (0, .4, .5, .6)])
    reference = write_run(tmp_path / "reference", frames=a)
    candidate = write_run(tmp_path / "candidate", frames=b)
    metric = tmp_path / "metrics" / "reference--candidate.json"
    result = evaluate_runs(candidate, reference=reference,
        tolerances_path=tolerance_file(tmp_path / "tolerances.yaml"), output=metric, review_limit=1)
    proof = metric.with_suffix(".receipt.json")
    review = Path(result["review"]["index"]).parent
    proof.write_text(json.dumps({"evaluation_sha256": file_sha256(metric),
        "review_artifact_sha256": {str(p): file_sha256(p) for p in review.rglob("*") if p.is_file()}}))
    registry = {str(p.resolve()): file_sha256(p) for run in (reference, candidate) for p in run.rglob("*.npy")}
    return metric, proof, registry, result


def test_native_export_exact_pixels_blind_and_resume_after_retirement(tmp_path, helper):
    metric, proof, registry, result = comparison(tmp_path)
    original = {p: file_sha256(p) for p in Path(result["review"]["index"]).parent.rglob("*") if p.is_file()}
    bundle = helper.export_comparison(metric, proof, registry, tmp_path / "exports")
    private = json.loads((bundle / "private-receipt.json").read_text())
    assert (bundle / "private-receipt.json").stat().st_mode & 0o077 == 0
    manifest = json.loads((bundle / "public/manifest.json").read_text())
    case = manifest["cases"][0]
    key = json.loads(Path(result["review"]["private_key"]).read_text())["cases"][0]
    expected = [.4, .1] if key["A"] == "candidate" else [.1, .4]
    pixels = np.asarray(Image.open(bundle / "public" / case["frames"][0]))
    assert pixels.shape == (8 + 28, 16, 3)
    assert np.array_equal(pixels[28:, :8], np.full((8, 8, 3), np.rint(expected[0] * 255), np.uint8))
    assert np.array_equal(pixels[28:, 8:], np.full((8, 8, 3), np.rint(expected[1] * 255), np.uint8))
    assert case["future_frames"] == 3
    for p in (bundle / "public").glob("*.html"):
        assert "reference" not in p.read_text() and "candidate" not in p.read_text()
        assert str(tmp_path) not in p.read_text()
    for name in registry:
        Path(name).unlink()
    assert helper.export_comparison(metric, proof, registry, tmp_path / "exports") == bundle
    assert original == {p: file_sha256(p) for p in original}
    assert private["public_bytes"] > 0


@pytest.mark.parametrize("damage", ["array", "metric", "key", "registry"])
def test_changed_input_rejects_before_export(tmp_path, helper, damage):
    metric, proof, registry, result = comparison(tmp_path)
    if damage == "array":
        Path(next(iter(registry))).write_bytes(b"changed")
    elif damage == "metric":
        metric.write_text("{}")
    elif damage == "key":
        Path(result["review"]["private_key"]).write_text("{}")
    else:
        registry[next(iter(registry))] = "0" * 64
    with pytest.raises(ValueError):
        helper.export_comparison(metric, proof, registry, tmp_path / "exports")


@pytest.mark.parametrize("damage", ["missing_png", "changed_png", "public_receipt", "private_mode", "source"])
def test_resumed_export_fails_closed(tmp_path, helper, damage):
    metric, proof, registry, _ = comparison(tmp_path)
    bundle = helper.export_comparison(metric, proof, registry, tmp_path / "exports")
    image = next((bundle / "public").glob("*.png"))
    if damage == "missing_png":
        image.unlink()
    elif damage == "changed_png":
        image.write_bytes(b"changed")
    elif damage == "public_receipt":
        (bundle / "public/manifest.json").write_text("{}")
    elif damage == "private_mode":
        (bundle / "private-receipt.json").chmod(0o644)
    else:
        Path(next(iter(registry))).write_bytes(b"changed")
    with pytest.raises(ValueError):
        helper.export_comparison(metric, proof, registry, tmp_path / "exports")


def test_missing_source_without_export_does_not_authorize_retirement(tmp_path, helper):
    metric, proof, registry, _ = comparison(tmp_path)
    Path(next(iter(registry))).unlink()
    with pytest.raises(ValueError, match="missing or changed"):
        helper.export_comparison(metric, proof, registry, tmp_path / "exports")


def test_disk_reserve_is_checked_before_native_export(tmp_path, helper, monkeypatch):
    metric, proof, registry, _ = comparison(tmp_path)
    monkeypatch.setattr(helper.shutil, "disk_usage", lambda _: helper.SimpleNamespace(free=10))
    with pytest.raises(RuntimeError, match="disk reserve"):
        helper.export_comparison(metric, proof, registry, tmp_path / "exports", minimum_free_bytes=100)
    assert not list((tmp_path / "exports").glob("*/private-receipt.json"))


def test_all_24_frames_in_two_twelve_frame_native_pages(tmp_path, helper):
    metric, proof, registry, result = comparison(tmp_path)
    # Rebuild a real evaluator fixture with the target 1+24 frame contract.
    for path in registry:
        np.save(path, np.linspace(0, 1, 25, dtype=np.float32)[:, None, None, None]
                * np.ones((25, 8, 8, 3), np.float32))
    for name in ("reference", "candidate"):
        path = tmp_path / name / "config.json"
        config = json.loads(path.read_text()); config["generation"]["total_frames"] = 25
        path.write_text(json.dumps(config))
    result = evaluate_runs(tmp_path / "candidate", reference=tmp_path / "reference",
        tolerances_path=tmp_path / "tolerances.yaml", output=metric, review_limit=1)
    review = Path(result["review"]["index"]).parent
    proof.write_text(json.dumps({"evaluation_sha256": file_sha256(metric),
        "review_artifact_sha256": {str(p): file_sha256(p) for p in review.rglob("*") if p.is_file()}}))
    registry = {name: file_sha256(name) for name in registry}
    bundle = helper.export_comparison(metric, proof, registry, tmp_path / "exports")
    case = json.loads((bundle / "public/manifest.json").read_text())["cases"][0]
    assert len(case["frames"]) == 24 and len(case["pages"]) == 2
    assert all((bundle / "public" / page).read_text().count("<figure>") == 12 for page in case["pages"])
    assert "Future frame 24" in (bundle / "public" / case["pages"][1]).read_text()
    assert Image.open(bundle / "public" / case["frames"][-1]).size == (16, 36)


def test_admission_deadline_and_new_run_limit(helper):
    now = dt.datetime(2030, 1, 1, tzinfo=dt.timezone.utc)
    gate = helper.Admission("2030-01-01T00:10:00Z", run_seconds=400, evaluation_seconds=100,
        export_seconds=10, max_new_runs=1, now=lambda: now)
    with pytest.raises(helper.IntentionalPause, match="deadline"):
        gate.new_run()
    gate.run_seconds = 350
    gate.new_run()
    gate.new_runs = 1
    with pytest.raises(helper.IntentionalPause, match="maximum"):
        gate.new_run()


@pytest.mark.parametrize("overrides", [dict(run_seconds=float("nan")), dict(evaluation_seconds=-1),
    dict(export_seconds=0), dict(cleanup_seconds=239), dict(max_new_runs=0), dict(max_new_runs=True)])
def test_invalid_admission_limits(helper, overrides):
    kwargs = dict(run_seconds=400, evaluation_seconds=100, export_seconds=10)
    kwargs.update(overrides)
    with pytest.raises(ValueError):
        helper.Admission("2030-01-01T00:10:00Z", **kwargs)


def test_deadline_requires_utc(helper):
    with pytest.raises(ValueError, match="UTC"):
        helper.parse_deadline("2030-01-01T00:10:00")
    with pytest.raises(ValueError, match="UTC"):
        helper.parse_deadline("2030-01-01T00:10:00+01:00")


def test_synthetic_pause_resume_exports_before_retire_and_restores_hooks(tmp_path, helper):
    destination, plan, _ = study_fixture(tmp_path)
    originals = (study._execute_run, study._retire_outputs, study.subprocess)
    kwargs = dict(deadline="2099-01-01T00:00:00Z", run_seconds=3600,
                  evaluation_seconds=300, export_seconds=60, max_shards=1)
    first = helper.run(destination, "development", tmp_path / "wrapper-1.json", max_new_runs=1, **kwargs)
    assert first["status"] == "intentionally_paused" and first["new_runs_completed"] == 1
    assert first["phase_complete"] is False
    assert not (destination / "quality/development/shards/000/complete.json").exists()
    assert originals == (study._execute_run, study._retire_outputs, study.subprocess)
    second = helper.run(destination, "development", tmp_path / "wrapper-2.json", **kwargs)
    assert second["status"] == "returned" and second["result"]["completed_shards"] == 1
    assert second["new_runs_completed"] == 2 and second["phase_complete"] is False
    assert second["source_unchanged"] and second["plan_unchanged"]
    assert len(second["native_review_indexes"]) == 3
    shard = destination / "quality/development/shards/000"
    completed = json.loads((shard / "complete.json").read_text())
    assert all(not Path(row["path"]).exists() for run in completed["runs"].values() for row in run["outputs"])
    third = helper.run(destination, "development", tmp_path / "wrapper-3.json", **kwargs)
    assert third["new_runs_completed"] == 0 and len(third["native_review_indexes"]) == 3
    image = next((destination / "quality/development/native-review").glob("*/public/*.png"))
    image.unlink()
    with pytest.raises(ValueError, match="file set changed"):
        helper.run(destination, "development", tmp_path / "wrapper-failed.json", **kwargs)
    assert json.loads((tmp_path / "wrapper-failed.json").read_text())["status"] == "failed"
    assert originals == (study._execute_run, study._retire_outputs, study.subprocess)


def test_retirement_failure_never_calls_core_delete(tmp_path, helper, monkeypatch):
    destination, plan, _ = study_fixture(tmp_path)
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic export failure")
    monkeypatch.setattr(helper, "export_comparison", fail)
    with pytest.raises(RuntimeError, match="export failure"):
        helper.run(destination, "development", tmp_path / "failed.json", deadline="2099-01-01T00:00:00Z",
            run_seconds=3600, evaluation_seconds=300, export_seconds=60, max_shards=1)
    receipts = list((destination / "quality/development/shards/000/completed").glob("*.json"))
    assert len(receipts) == 3
    for receipt in receipts:
        assert all(Path(row["path"]).is_file() for row in json.loads(receipt.read_text())["outputs"])


def test_resumed_new_export_checks_deadline_before_any_delete(tmp_path, helper, monkeypatch):
    destination, _, _ = study_fixture(tmp_path)
    original_check = helper.Admission.check
    def pause_export(self, seconds, label):
        if label == "new native review export":
            raise helper.IntentionalPause("Synthetic deadline before resumed export")
        return original_check(self, seconds, label)
    monkeypatch.setattr(helper.Admission, "check", pause_export)
    kwargs = dict(deadline="2099-01-01T00:00:00Z", run_seconds=3600,
                  evaluation_seconds=300, export_seconds=60, max_shards=1)
    first = helper.run(destination, "development", tmp_path / "export-pause1.json", **kwargs)
    assert first["status"] == "intentionally_paused"
    shard = destination / "quality/development/shards/000"
    assert len(list((shard / "metrics").glob("*.receipt.json"))) == 3
    for receipt in (shard / "completed").glob("*.json"):
        assert all(Path(row["path"]).is_file() for row in json.loads(receipt.read_text())["outputs"])
    second = helper.run(destination, "development", tmp_path / "export-pause2.json", **kwargs)
    assert second["status"] == "intentionally_paused" and second["new_runs_completed"] == 0
    assert not list((destination / "quality/development/native-review").glob("*/private-receipt.json"))
    monkeypatch.setattr(helper.Admission, "check", original_check)
    final = helper.run(destination, "development", tmp_path / "export-resume.json", **kwargs)
    assert final["status"] == "returned" and len(final["native_export_receipt_sha256"]) == 3


def fake_ground_truth(helper, monkeypatch, *, fail=False):
    calls = []
    def export_run(run_path, output):
        output = Path(output); path = Path(run_path)
        if fail:
            raise RuntimeError("Synthetic recorded-future export failure")
        if output.exists():
            assert json.loads(output.read_text())["run"] == str(path)
            assert output.with_suffix(".receipt.json").is_file()
            calls.append("verified")
            return
        assert len(list(path.rglob("*.npy"))) == 16, "Ground truth must precede raw retirement"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"run": str(path)}))
        output.with_suffix(".receipt.json").write_text("{}")
        calls.append("new")
    monkeypatch.setattr(helper, "load_ground_truth_helper", lambda: helper.SimpleNamespace(export_run=export_run))
    return calls


def test_optional_gt_export_precedes_upstream_delete_and_revalidates_resume(tmp_path, helper, monkeypatch):
    destination, _, _ = study_fixture(tmp_path)
    calls = fake_ground_truth(helper, monkeypatch)
    kwargs = dict(deadline="2099-01-01T00:00:00Z", run_seconds=3600,
                  evaluation_seconds=300, export_seconds=60, max_shards=1, upstream_ground_truth=True)
    result = helper.run(destination, "development", tmp_path / "gt-first.json", **kwargs)
    assert calls == ["new"] and result["ground_truth_exporter_unchanged"]
    assert len(result["recorded_ground_truth_artifact_sha256"]) == 2
    assert result["ground_truth_exporter_sha256"] == file_sha256(Path(helper.__file__).with_name("export_recorded_ground_truth.py"))
    helper.run(destination, "development", tmp_path / "gt-resume.json", **kwargs)
    assert calls == ["new", "verified"]


def test_gt_failure_preserves_upstream_raw_outputs(tmp_path, helper, monkeypatch):
    destination, _, _ = study_fixture(tmp_path)
    fake_ground_truth(helper, monkeypatch, fail=True)
    with pytest.raises(RuntimeError, match="recorded-future export failure"):
        helper.run(destination, "development", tmp_path / "gt-failed.json", deadline="2099-01-01T00:00:00Z",
            run_seconds=3600, evaluation_seconds=300, export_seconds=60, max_shards=1, upstream_ground_truth=True)
    receipt = json.loads((destination / "quality/development/shards/000/completed/upstream.json").read_text())
    assert all(Path(row["path"]).is_file() for row in receipt["outputs"])


def test_gt_resume_admission_pause_preserves_upstream(tmp_path, helper, monkeypatch):
    destination, _, _ = study_fixture(tmp_path)
    calls = fake_ground_truth(helper, monkeypatch)
    original_check = helper.Admission.check
    def check(self, seconds, label):
        if label == "new recorded-ground-truth export":
            raise helper.IntentionalPause("Synthetic GT deadline")
        return original_check(self, seconds, label)
    monkeypatch.setattr(helper.Admission, "check", check)
    kwargs = dict(deadline="2099-01-01T00:00:00Z", run_seconds=3600,
                  evaluation_seconds=300, export_seconds=60, max_shards=1, upstream_ground_truth=True)
    first = helper.run(destination, "development", tmp_path / "gt-paused.json", **kwargs)
    assert first["status"] == "intentionally_paused" and not calls
    receipt = json.loads((destination / "quality/development/shards/000/completed/upstream.json").read_text())
    assert all(Path(row["path"]).is_file() for row in receipt["outputs"])
    second = helper.run(destination, "development", tmp_path / "gt-paused-again.json", **kwargs)
    assert second["status"] == "intentionally_paused" and second["new_runs_completed"] == 0
