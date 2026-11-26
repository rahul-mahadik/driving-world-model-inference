"""Public-statistics reproduction tests; every timing is a synthetic fixture."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
try:
    import recompute_schedule_results as reproduction
finally:
    sys.path.pop(0)


@pytest.fixture(scope="module")
def fixture_export(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic-public-schedule-study")
    path = ROOT / "tests/test_schedule_study_analysis_v2.py"
    spec = importlib.util.spec_from_file_location("public_reproduction_fixture", path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    fixture = builder.SyntheticStudy(root)
    # A publication can retain complete timing arrays for an unconfirmed
    # submission. Those arrays must not silently promote its failure score.
    fixture.rows[1].update(successful=False, environment_completed=False,
                           stop_reason="unconfirmed_before_deadline",
                           confirmed_latency_ratio=None, failure_adjusted_speed_score=-1.)
    fixture.save()
    builder.analysis.export(fixture.study, root/"export", fixture.manifest,
                            fixture.frozen, fixture.plan, fixture_only=True)
    return root / "export"


@pytest.fixture
def export_copy(fixture_export, tmp_path):
    target = tmp_path / "export"
    shutil.copytree(fixture_export, target)
    return target


def change(export, name, mutate):
    path = export / name
    value = json.loads(path.read_text())
    mutate(value)
    raw = (json.dumps(value) + "\n").encode()
    path.write_bytes(raw)
    # Keep the integrity envelope coherent so semantic checks, rather than a
    # stale file hash, must detect the altered statistic or sample arrays.
    manifest_path = export / "export-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    row = next(row for row in manifest["files"] if row["file"] == name)
    row.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    manifest_path.write_text(json.dumps(manifest) + "\n")


def check(export, **kwargs):
    return reproduction.recompute(export/"summary.json", export/"episodes.json",
                                  allow_test_fixtures=True, **kwargs)


def test_reproduces_all_statistics_and_keeps_unconfirmed_score(fixture_export):
    before = {path.name: path.read_bytes() for path in fixture_export.iterdir()}
    result = check(fixture_export)
    assert result["verified"] and not result["scientific_results"]
    assert result["episodes_verified"] == 54
    assert result["failed_or_unconfirmed_retained"] == 1
    assert result["aggregates"]["policies"]["random_walk"]["mean_primary_score"] == pytest.approx((17*.2-1)/18)
    assert result["aggregates"]["uncertainty"]["bootstrap_replicates"] == 10000
    assert result["aggregates"]["uncertainty"]["bootstrap_seed"] == 20261125
    assert "published success flags" in result["verification_scope"]
    assert before == {path.name: path.read_bytes() for path in fixture_export.iterdir()}


@pytest.mark.parametrize("mutation,match", [
    ("sample_median", "median_ms"),
    ("sample_count", "seven samples"),
    ("latency_ratio", "confirmed_latency_ratio"),
    ("failure_score", "retain score -1"),
])
def test_rejects_array_ratio_and_failure_tampering(export_copy, mutation, match):
    def mutate(document):
        row = document["episodes"][0]
        if mutation == "sample_median":
            row["measurements"]["confirmation"]["raw_samples_ms"] = [123.] * 7
        elif mutation == "sample_count":
            row["measurements"]["confirmation"]["raw_samples_ms"].pop()
        elif mutation == "latency_ratio":
            row["confirmed_latency_ratio"] = .5
        else:
            document["episodes"][1]["primary_score"] = 0.
    change(export_copy, "episodes.json", mutate)
    with pytest.raises(ValueError, match=match):
        check(export_copy)


@pytest.mark.parametrize("field", ["paired_interval", "registered_seed"])
def test_rejects_changed_intervals_or_analysis_settings(export_copy, field):
    def mutate(document):
        if field == "paired_interval":
            document["paired_comparisons"][0]["paired_difference_ci95"][0] += .2
        else:
            document["uncertainty"]["bootstrap_seed"] += 1
    change(export_copy, "summary.json", mutate)
    with pytest.raises(ValueError, match="summary\\."):
        check(export_copy)


def test_requires_explicit_fixture_opt_in_and_v2(export_copy):
    with pytest.raises(ValueError, match="Synthetic data"):
        reproduction.recompute(export_copy/"summary.json", export_copy/"episodes.json")
    change(export_copy, "summary.json", lambda document: document.update(study_id="sequential-schedules-live-v1"))
    with pytest.raises(ValueError, match="only the registered v2"):
        check(export_copy)


def test_receipt_is_new_and_outside_inputs(fixture_export, tmp_path, capsys):
    receipt = tmp_path / "verified.json"
    args = ["--summary", str(fixture_export/"summary.json"), "--episodes", str(fixture_export/"episodes.json"),
            "--receipt", str(receipt), "--allow-test-fixtures"]
    assert reproduction.main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result == json.loads(receipt.read_text()) and not result["scientific_results"]
    with pytest.raises(ValueError, match="new file"):
        reproduction.main(args)
    args[args.index(str(receipt))] = str(fixture_export/"new-receipt.json")
    with pytest.raises(ValueError, match="outside the input"):
        reproduction.main(args)
