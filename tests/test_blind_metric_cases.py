"""Synthetic-only supplemental review export checks."""
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from drivebench.config import file_sha256
from drivebench.evaluate import evaluate_runs
from test_evaluation import write_run, tolerance_file


@pytest.fixture
def helper():
    path = Path(__file__).resolve().parents[1] / "scripts/export_blind_metric_cases.py"
    spec = importlib.util.spec_from_file_location("blind_metric_cases_fixture", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def fixture(tmp_path):
    frames = np.zeros((4, 8, 8, 3), np.float32)
    reference = write_run(tmp_path / "reference", frames=frames, latencies=(1, 1))
    frames[2, 3, 4] = .8
    candidate = write_run(tmp_path / "candidate", frames=frames, latencies=(1, 1))
    metric = tmp_path / "original-evaluation.json"
    result = evaluate_runs(candidate, reference=reference,
        tolerances_path=tolerance_file(tmp_path / "tolerances.yaml"), output=metric, review_limit=1)
    return metric, result


def test_supplement_native_pixels_and_highlight_do_not_change_original_blind_review(tmp_path, helper, monkeypatch):
    metric, result = fixture(tmp_path)
    original_files = [metric, *Path(result["review"]["index"]).parent.rglob("*")]
    original_files = {p: file_sha256(p) for p in original_files if p.is_file()}
    old_key = Path(result["review"]["private_key"]).resolve()
    original_read = Path.read_text
    def guarded(path, *args, **kwargs):
        assert path.resolve() != old_key, "Original private key must never be opened"
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", guarded)
    value = helper.export(metric, ["r0", "r1"], tmp_path / "supplement", expected_future_frames=3)
    assert value["quality_acceptance_claim"] is False
    assert len(value["opaque_case_highlights"]) == 2
    for item in value["opaque_case_highlights"]:
        assert item["greatest_error_future_frame_1based"] == 2
        assert Image.open(item["native_frame_pair"]).size == (16, 36)
    assert original_files == {p: file_sha256(p) for p in original_files}
    public_text = json.dumps(value)
    assert "reference_path" not in public_text and "candidate_path" not in public_text
    assert "private_key" not in public_text and '"r0"' not in public_text
    receipt = tmp_path / "supplement/private/supplement-receipt.json"
    assert receipt.stat().st_mode & 0o077 == 0
    assert json.loads(receipt.read_text())["original_private_key_read"] is False
    with pytest.raises(FileExistsError):
        helper.export(metric, ["r0", "r1"], tmp_path / "supplement", expected_future_frames=3)


@pytest.mark.parametrize("kind", ["unknown_request", "duplicate", "missing_array", "changed_array", "wrong_frames", "changed_metadata"])
def test_supplement_rejects_incomplete_or_changed_evidence(tmp_path, helper, kind):
    metric, result = fixture(tmp_path)
    requests, count = ["r0", "r1"], 3
    array = tmp_path / "candidate/outputs/r0/candidate.npy"
    if kind == "unknown_request": requests = ["unknown"]
    if kind == "duplicate": requests = ["r0", "r0"]
    if kind == "missing_array": array.unlink()
    if kind == "changed_array": np.save(array, np.zeros((4, 8, 8, 3), np.float32))
    if kind == "wrong_frames": count = 24
    if kind == "changed_metadata": (tmp_path / "candidate/summary.json").write_text("{}")
    with pytest.raises((ValueError, FileNotFoundError)):
        helper.export(metric, requests, tmp_path / "supplement", expected_future_frames=count)
    assert not (tmp_path / "supplement/private/supplement-receipt.json").exists()
